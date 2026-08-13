"""
Phase A trainer: pretrain a random-init OlmoeForCausalLM with a chosen router.

Single-file, torchrun-launched, host conventions throughout (AdamW, cosine LR,
softmax-then-topk gating with norm_topk_prob=False, MoE in every layer, no
router noise). The only non-stock component is the gate: hf.patch_olmoe swaps
in a fixed router unless --router linear.

8xH100 (size S example):
  torchrun --standalone --nproc_per_node=8 -m hf.train_olmoe -- \
      --data-dir /data/$USER/hf_data/fineweb_edu_olmoe --out-dir /data/$USER/hf_runs \
      --run-name S_vsa_fpe_s0 --router vsa_fpe --hidden-size 512 --num-layers 8 --num-heads 8 \
      --device-batch-size 8 --lr 6e-4

Mac CPU smoke (synthetic-scale):
  python -m hf.train_olmoe --data-dir <tiny shards> --out-dir /tmp/run --run-name smoke \
      --router vsa_fpe --hidden-size 64 --num-layers 2 --num-heads 4 --vocab-size 256 \
      --seq-len 64 --device-batch-size 2 --total-batch-size 256 --num-iterations 20 \
      --eval-every 10 --eval-tokens 2048

Loss = manual cross-entropy over all T positions (the loader tiles the token
stream exactly) + router_aux_loss_coef * HF-computed load-balancing aux loss.
Metrics stream to <out-dir>/<run-name>/metrics.jsonl for hf/analysis.py.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

from hf.checkpoints import latest_checkpoint, mark_done, prune_checkpoints
from hf.data import distributed_data_loader, tokens_per_byte
from hf.patch_olmoe import load_router_olmoe, patch_olmoe_routers, save_router_olmoe
from hf.routers import ROUTER_TYPES

LN2 = math.log(2.0)

# args that define what the run *is*: a resume that changes any of them is
# describing a different experiment, and load_router_olmoe would silently
# override --router from the checkpoint spec rather than fail. num_iterations is
# checked separately — config.json records the resolved step count, not the raw
# flag, so comparing it against args (-1 by default) would reject every resume.
RESUME_LOCKED_ARGS = ("router", "router_seed", "seed", "hidden_size", "num_layers", "num_heads",
                      "num_experts", "num_experts_per_tok", "norm_topk_prob", "aux_loss_coeff",
                      "vocab_size", "seq_len", "total_batch_size", "lr",
                      "target_param_data_ratio")


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--router", type=str, default="linear", choices=ROUTER_TYPES)
    p.add_argument("--router-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="init/data seed (weights use seed, routers use router-seed)")
    # model (host OlmoeConfig; intermediate = hidden, MoE every layer)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--num-experts-per-tok", type=int, default=2)
    p.add_argument("--norm-topk-prob", action="store_true", help="ablation probe; default off (host convention)")
    p.add_argument("--aux-loss-coeff", type=float, default=0.01)
    p.add_argument("--vocab-size", type=int, default=50304, help="padded; must cover the tokenizer vocab")
    # optimization
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--device-batch-size", type=int, default=8)
    p.add_argument("--total-batch-size", type=int, default=524288, help="tokens per optimizer step")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-frac", type=float, default=0.01)
    p.add_argument("--final-lr-frac", type=float, default=0.1)
    p.add_argument("--num-iterations", type=int, default=-1, help="-1: derive from --target-param-data-ratio")
    p.add_argument("--target-param-data-ratio", type=float, default=20.0, help="tokens per TOTAL param (original convention)")
    # cadence
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-tokens", type=int, default=10_485_760)
    p.add_argument("--save-every", type=int, default=-1, help="-1: only at 25/50/75/100%")
    p.add_argument("--resume", type=str, default="", help="checkpoint dir (…/step_N), or 'auto' for the newest in the run dir")
    p.add_argument("--run", type=str, default="dummy", help="wandb run name; 'dummy' disables wandb")
    p.add_argument("--device", type=str, default="", help="override autodetect (note: VSA routers need torch.fft, incomplete on MPS — use cpu for Mac smokes)")
    return p.parse_args()


def setup_dist(device_override=""):
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            device = f"cuda:{os.environ['LOCAL_RANK']}"
        else:
            device = "cpu"
    else:
        rank, world = 0, 1
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device_override:
        device = device_override
    return rank, world, device


def build_model(args):
    config = OlmoeConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_heads,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        norm_topk_prob=args.norm_topk_prob,
        router_aux_loss_coef=args.aux_loss_coeff,
        max_position_embeddings=args.seq_len,
        tie_word_embeddings=False,
        use_cache=False,
    )
    torch.manual_seed(args.seed)
    model = OlmoeForCausalLM(config)
    patch_olmoe_routers(model, args.router, seed=args.router_seed)
    return model


def check_resume_args(run_dir, args, num_iterations=None):
    """Refuse a resume whose CLI disagrees with the run it is continuing.

    Called twice: once before the model loads (cheap args), then again once
    num_iterations has been resolved, since that sets the LR schedule length and
    silently changing it mid-run would bend the schedule.
    """
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        previous = json.load(f)
    current = dict(vars(args))
    keys = RESUME_LOCKED_ARGS
    if num_iterations is not None:
        current["num_iterations"] = num_iterations
        keys = keys + ("num_iterations",)
    changed = [k for k in keys if k in previous and previous[k] != current[k]]
    if changed:
        detail = ", ".join(f"{k}: {previous[k]} -> {current[k]}" for k in changed)
        raise ValueError(f"resume args disagree with {path} ({detail}); "
                         f"use a new --run-name to start a different run")


def lr_multiplier(step, num_iterations, warmup_frac, final_frac):
    warmup = max(1, int(warmup_frac * num_iterations))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, num_iterations - warmup)
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress))


def forward_loss(model, inputs, targets):
    out = model(input_ids=inputs, output_router_logits=True)
    ce = F.cross_entropy(out.logits.float().view(-1, out.logits.size(-1)), targets.view(-1))
    aux = out.aux_loss if out.aux_loss is not None else torch.zeros((), device=ce.device)
    return ce, aux


@torch.no_grad()
def evaluate_bpb(model, args, device, rank, world, tpb):
    model.eval()
    B, T = args.device_batch_size, args.seq_len
    steps = max(1, args.eval_tokens // (B * T * world))
    loader = distributed_data_loader(args.data_dir, "val", B, T, rank=rank, world_size=world, device=device)
    total = torch.zeros((), device=device)
    for _ in range(steps):
        inputs, targets, _ = next(loader)
        with autocast_ctx(device):
            ce, _ = forward_loss(model, inputs, targets)
        total += ce
    if world > 1:
        dist.all_reduce(total, op=dist.ReduceOp.AVG)
    model.train()
    return (total.item() / steps) / LN2 * tpb


@torch.no_grad()
def expert_telemetry(model, probe_inputs, top_k):
    """Per-layer routing stats on a fixed probe batch: token fraction per expert
    (from top-k dispatch), Gini, normalized entropy, and mean gate mass (sum of
    selected softmax weights — the norm_topk_prob=False health signal)."""
    model.eval()
    out = model(input_ids=probe_inputs, output_router_logits=True)
    stats = []
    for logits in out.router_logits:
        probs = F.softmax(logits.float(), dim=-1)
        weights, selected = torch.topk(probs, top_k, dim=-1)
        counts = torch.bincount(selected.flatten(), minlength=probs.size(-1)).float()
        frac = counts / counts.sum()
        sorted_frac, _ = torch.sort(frac)
        n = len(frac)
        gini = (2 * torch.arange(1, n + 1, device=frac.device) - n - 1).float().dot(sorted_frac) / (n * sorted_frac.sum() + 1e-12)
        entropy = -(frac * (frac + 1e-12).log()).sum() / math.log(n)
        stats.append({
            "frac": [round(f, 5) for f in frac.tolist()],
            "gini": round(gini.item(), 5),
            "entropy": round(entropy.item(), 5),
            "gate_mass": round(weights.sum(-1).mean().item(), 5),
        })
    model.train()
    return stats


def autocast_ctx(device):
    if torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    import contextlib
    return contextlib.nullcontext()


def main():
    args = get_args()
    rank, world, device = setup_dist(args.device)
    is_main = rank == 0
    run_dir = os.path.join(args.out_dir, args.run_name)
    B, T = args.device_batch_size, args.seq_len

    assert args.total_batch_size % (B * T * world) == 0, "total batch must divide evenly"
    grad_accum = args.total_batch_size // (B * T * world)

    # ---- model ----
    resume = (latest_checkpoint(run_dir) or "") if args.resume == "auto" else args.resume
    if resume:
        check_resume_args(run_dir, args)
        model = load_router_olmoe(os.path.join(resume, "model"), dtype=torch.float32, device=device)
        trainer_state = torch.load(os.path.join(resume, "trainer_state.pt"), weights_only=True)
        start_step = trainer_state["step"]
        if is_main:
            print(f"resuming {args.run_name} from step {start_step}")
    else:
        model = build_model(args).to(device)
        trainer_state, start_step = None, 0
    model.train()
    raw_model = model

    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
    active_params = total_params - expert_params + expert_params * args.num_experts_per_tok // args.num_experts

    num_iterations = args.num_iterations
    if num_iterations < 0:
        num_iterations = int(args.target_param_data_ratio * total_params) // args.total_batch_size
    if resume:
        check_resume_args(run_dir, args, num_iterations)

    if world > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()] if torch.device(device).type == "cuda" else None)

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)
    if trainer_state is not None:
        optimizer.load_state_dict(trainer_state["optimizer"])

    # ---- data ----
    loader = distributed_data_loader(args.data_dir, "train", B, T, rank=rank, world_size=world,
                                     device=device, start_step=start_step * grad_accum)
    tpb = tokens_per_byte(args.data_dir, "val")
    probe_inputs, _, _ = next(distributed_data_loader(args.data_dir, "val", min(B, 4), T, device=device))

    # ---- logging ----
    wandb_run = None
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        if not resume:  # keep the original run's record; check_resume_args reads it
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump({**vars(args), "total_params": total_params, "active_params": active_params,
                           "num_iterations": num_iterations, "world_size": world}, f, indent=2)
        print(f"params: {total_params/1e6:.1f}M total / {active_params/1e6:.1f}M active | "
              f"{num_iterations} steps x {args.total_batch_size} tokens | grad_accum {grad_accum}")
        if args.run != "dummy":
            import wandb
            wandb_run = wandb.init(project="nanochat-hf-moe", name=args.run, config=vars(args))
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    def log(record):
        if is_main:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            if wandb_run is not None:
                wandb_run.log({k: v for k, v in record.items() if isinstance(v, (int, float))}, step=record["step"])

    save_marks = {num_iterations * f // 4 for f in (1, 2, 3, 4)}

    def save_checkpoint(step):
        if not is_main:
            return
        ckpt_dir = os.path.join(run_dir, f"step_{step:06d}")
        save_router_olmoe(raw_model, os.path.join(ckpt_dir, "model"))
        torch.save({"step": step, "optimizer": optimizer.state_dict()},
                   os.path.join(ckpt_dir, "trainer_state.pt"))
        prune_checkpoints(run_dir, keep_models=save_marks)
        print(f"saved {ckpt_dir}")

    t0, tokens_seen = time.time(), 0

    for step in range(start_step, num_iterations):
        lr_mult = lr_multiplier(step, num_iterations, args.warmup_frac, args.final_lr_frac)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_mult

        loss_acc, aux_acc = 0.0, 0.0
        for micro in range(grad_accum):
            inputs, targets, _ = next(loader)
            sync = micro == grad_accum - 1
            ctx = model.no_sync() if (world > 1 and not sync) else autocast_ctx("cpu")  # nullcontext
            with ctx:
                with autocast_ctx(device):
                    ce, aux = forward_loss(model, inputs, targets)
                    loss = (ce + args.aux_loss_coeff * aux) / grad_accum
                loss.backward()
            loss_acc += ce.item() / grad_accum
            aux_acc += aux.item() / grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += args.total_batch_size

        if is_main and (step % 10 == 0 or step == num_iterations - 1):
            elapsed = time.time() - t0
            tok_s = tokens_seen / elapsed if elapsed > 0 else 0
            print(f"step {step:6d}/{num_iterations} | loss {loss_acc:.4f} | aux {aux_acc:.4f} | "
                  f"lr {args.lr * lr_mult:.2e} | gnorm {grad_norm:.2f} | {tok_s/1e3:.0f}k tok/s")
        log({"step": step, "train_loss": round(loss_acc, 5), "aux_loss": round(aux_acc, 5),
             "lr": args.lr * lr_mult, "grad_norm": round(float(grad_norm), 4)})

        last = step == num_iterations - 1
        if step % args.eval_every == 0 or last:
            val_bpb = evaluate_bpb(raw_model, args, device, rank, world, tpb)
            util = expert_telemetry(raw_model, probe_inputs, args.num_experts_per_tok) if is_main else None
            if is_main:
                print(f"step {step:6d} | val_bpb {val_bpb:.4f}")
            log({"step": step, "val_bpb": round(val_bpb, 5), "expert_utilization": util})
        if (args.save_every > 0 and step % args.save_every == 0 and step > 0) or (step + 1) in save_marks:
            save_checkpoint(step + 1)
        if world > 1:
            dist.barrier()

    if is_main:
        mark_done(run_dir)
    if wandb_run is not None:
        wandb_run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
