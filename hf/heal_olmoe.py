"""
Phase B1: continued pretraining of OLMoE-1B-7B with an annealed router swap.

Conditions (--condition):
  blend      BlendedRouterGate everywhere: alpha anneals 1 -> 0 linearly over
             --anneal-steps, then training continues on the pure fixed router.
  hard_swap  Same gates but alpha pinned at 0 from step 0 (no-anneal ablation).
  control    Untouched learned gate, same data/budget (the comparison line).

Before training (blend/hard_swap): a calibration pass over --calib-tokens sets
each layer's `scale` so fixed-router logits match the learned gate's logit std.
Optionally apply Hungarian expert permutations from a B0 report
(--b0-report, produced by hf.diagnose_router) and pick --router-seed from its
seed search.

8xH100:
  torchrun --standalone --nproc_per_node=8 -m hf.heal_olmoe -- \
      --data-dir /data/$USER/hf_data/fineweb_edu_olmoe --out-dir /data/$USER/hf_heal \
      --run-name heal_vsa_fpe --condition blend --router vsa_fpe \
      --b0-report b0_report.json --router-seed <best B0 seed>

Memory: default keeps fp32 params + bf16 autocast + gradient checkpointing +
ZeRO-1-sharded Adam states (~63GB/GPU + activations on 8 GPUs for 6.9B params).
If that doesn't fit, --param-dtype bfloat16 drops params/grads/states to bf16
(~35GB) at some optimizer-precision risk. FSDP is the documented fallback.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

from hf.data import distributed_data_loader, tokens_per_byte
from hf.diagnose_router import MoeInputCatcher
from hf.patch_olmoe import patch_olmoe_routers, save_router_olmoe
from hf.routers import BlendedRouterGate, calibrate_scale, permute_expert_ids
from hf.train_olmoe import autocast_ctx, evaluate_bpb, expert_telemetry, forward_loss, lr_multiplier, setup_dist


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--condition", type=str, required=True, choices=["blend", "hard_swap", "control"])
    p.add_argument("--router", type=str, default="vsa_fpe")
    p.add_argument("--router-seed", type=int, default=0, help="use the best seed from the B0 seed search")
    p.add_argument("--b0-report", type=str, default="", help="apply Hungarian perms from this diagnose_router report")
    p.add_argument("--calib-tokens", type=int, default=2_000_000)
    # schedule (defaults = the plan's 5B-token condition)
    p.add_argument("--num-tokens", type=int, default=5_000_000_000)
    p.add_argument("--total-batch-size", type=int, default=2_097_152, help="tokens per step (512 x 4096)")
    p.add_argument("--anneal-steps", type=int, default=750)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--final-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--device-batch-size", type=int, default=2)
    p.add_argument("--param-dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    # cadence
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--telemetry-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--eval-tokens", type=int, default=4_194_304)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--run", type=str, default="dummy")
    return p.parse_args()


def blended_gates(model):
    return [layer.mlp.gate for layer in model.model.layers if isinstance(layer.mlp.gate, BlendedRouterGate)]


def apply_b0_perms(model, report_path, router_type):
    with open(report_path) as f:
        report = json.load(f)
    perms = report["agreement"][router_type]
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp.gate, BlendedRouterGate) and str(i) in perms:
            permute_expert_ids(layer.mlp.gate.router, perms[str(i)]["perm"])


@torch.no_grad()
def calibrate_all_scales(model, args, device, rank):
    """Run --calib-tokens through the frozen model, then std-match each layer's
    fixed-router logits to its learned gate's."""
    gates = {i: layer.mlp.gate for i, layer in enumerate(model.model.layers)
             if isinstance(layer.mlp.gate, BlendedRouterGate)}
    catcher = MoeInputCatcher(model, list(gates.keys()))
    loader = distributed_data_loader(args.data_dir, "val", args.device_batch_size, args.seq_len, device=device)
    hidden = {i: [] for i in gates}
    seen = 0
    while seen < args.calib_tokens:
        inputs, _, _ = next(loader)
        with autocast_ctx(device):
            model(input_ids=inputs)
        for i in gates:
            hidden[i].append(catcher.hidden[i].float())
        seen += inputs.numel()
    catcher.remove()
    scales = {}
    for i, gate in gates.items():
        scales[i] = calibrate_scale(gate, torch.cat(hidden[i]), method="std")
    if rank == 0:
        print("calibrated scales:", {i: round(s, 4) for i, s in scales.items()})
    return scales


@torch.no_grad()
def routing_reference(model, probe_inputs, top_k):
    """Learned-gate top-k on the probe batch, per layer, captured before any
    training — the fixed reference that drift is measured against."""
    catcher = MoeInputCatcher(model, [i for i, _ in enumerate(model.model.layers)])
    model(input_ids=probe_inputs)
    reference = {}
    for i, layer in enumerate(model.model.layers):
        gate = layer.mlp.gate
        learned = gate.learned_gate if isinstance(gate, BlendedRouterGate) else gate
        logits = learned(catcher.hidden[i])
        reference[i] = logits.float().topk(top_k, dim=-1).indices.cpu()
    catcher.remove()
    return reference


@torch.no_grad()
def routing_drift(model, probe_inputs, reference, top_k):
    """Mean fraction of the reference top-k kept by the current (blended) gate."""
    catcher = MoeInputCatcher(model, list(reference.keys()))
    model(input_ids=probe_inputs)
    drift = {}
    for i, ref in reference.items():
        logits = model.model.layers[i].mlp.gate(catcher.hidden[i]).float()
        now = logits.topk(top_k, dim=-1).indices.cpu()
        overlap = (now.unsqueeze(-1) == ref.unsqueeze(-2)).any(-1).float().mean().item()
        drift[str(i)] = round(overlap, 5)
    catcher.remove()
    return drift


def main():
    args = get_args()
    rank, world, device = setup_dist(args.device)
    is_main = rank == 0
    run_dir = os.path.join(args.out_dir, args.run_name)
    B, T = args.device_batch_size, args.seq_len
    assert args.total_batch_size % (B * T * world) == 0
    grad_accum = args.total_batch_size // (B * T * world)
    num_iterations = args.num_tokens // args.total_batch_size

    param_dtype = torch.float32 if args.param_dtype == "float32" else torch.bfloat16
    model = OlmoeForCausalLM.from_pretrained(args.model, dtype=param_dtype, revision=args.revision)
    model.config.use_cache = False
    if args.condition in ("blend", "hard_swap"):
        patch_olmoe_routers(model, args.router, seed=args.router_seed, blend=True)
        if args.b0_report:
            apply_b0_perms(model, args.b0_report, args.router)
    model = model.to(device)
    model.gradient_checkpointing_enable()

    if args.condition in ("blend", "hard_swap"):
        calibrate_all_scales(model, args, device, rank)
        if args.condition == "hard_swap":
            for gate in blended_gates(model):
                gate.set_alpha(0.0)

    top_k = model.config.num_experts_per_tok
    probe_inputs, _, _ = next(distributed_data_loader(args.data_dir, "val", min(B, 2), T, device=device))
    reference = routing_reference(model, probe_inputs, top_k)

    raw_model = model
    model.train()
    if world > 1:
        # learned gates leave the graph once alpha hits 0 -> unused params
        model = DDP(model, device_ids=[torch.cuda.current_device()] if torch.device(device).type == "cuda" else None,
                    find_unused_parameters=args.condition != "control")
        optimizer = ZeroRedundancyOptimizer(raw_model.parameters(), optimizer_class=torch.optim.AdamW,
                                            lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                      weight_decay=args.weight_decay)

    loader = distributed_data_loader(args.data_dir, "train", B, T, rank=rank, world_size=world, device=device)
    tpb = tokens_per_byte(args.data_dir, "val")
    aux_coeff = raw_model.config.router_aux_loss_coef

    wandb_run = None
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({**vars(args), "num_iterations": num_iterations, "world_size": world}, f, indent=2)
        if args.run != "dummy":
            import wandb
            wandb_run = wandb.init(project="nanochat-hf-heal", name=args.run, config=vars(args))
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    def log(record):
        if is_main:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            if wandb_run is not None:
                wandb_run.log({k: v for k, v in record.items() if isinstance(v, (int, float))}, step=record["step"])

    t0 = time.time()
    for step in range(num_iterations):
        if args.condition == "blend":
            alpha = max(0.0, 1.0 - step / max(1, args.anneal_steps))
            for gate in blended_gates(raw_model):
                gate.set_alpha(alpha)
        else:
            alpha = 0.0 if args.condition == "hard_swap" else 1.0

        lr_mult = lr_multiplier(step, num_iterations, args.warmup_steps / num_iterations, args.final_lr_frac)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_mult

        loss_acc = 0.0
        for micro in range(grad_accum):
            inputs, targets, _ = next(loader)
            sync = micro == grad_accum - 1
            ctx = model.no_sync() if (world > 1 and not sync) else autocast_ctx("cpu")  # nullcontext
            with ctx:
                with autocast_ctx(device):
                    ce, aux = forward_loss(model, inputs, targets)
                    loss = (ce + aux_coeff * aux) / grad_accum
                loss.backward()
            loss_acc += ce.item() / grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if is_main and step % 10 == 0:
            tok_s = (step + 1) * args.total_batch_size / (time.time() - t0)
            print(f"step {step:5d}/{num_iterations} | loss {loss_acc:.4f} | alpha {alpha:.3f} | "
                  f"lr {args.lr * lr_mult:.2e} | gnorm {grad_norm:.2f} | {tok_s/1e3:.0f}k tok/s")
        log({"step": step, "train_loss": round(loss_acc, 5), "alpha": alpha,
             "lr": args.lr * lr_mult, "grad_norm": round(float(grad_norm), 4)})

        if step % args.eval_every == 0 or step == num_iterations - 1:
            val_bpb = evaluate_bpb(raw_model, args, device, rank, world, tpb)
            if is_main:
                print(f"step {step:5d} | val_bpb {val_bpb:.4f}")
            log({"step": step, "val_bpb": round(val_bpb, 5)})
        if step % args.telemetry_every == 0 and is_main:
            log({"step": step,
                 "routing_drift": routing_drift(raw_model, probe_inputs, reference, top_k),
                 "expert_utilization": expert_telemetry(raw_model, probe_inputs, top_k)})
        if (step + 1) % args.save_every == 0 or step == num_iterations - 1:
            if world > 1:
                optimizer.consolidate_state_dict(to=0)
            if is_main:
                ckpt = os.path.join(run_dir, f"step_{step + 1:06d}")
                if args.condition == "control":
                    raw_model.save_pretrained(os.path.join(ckpt, "model"))
                else:
                    save_router_olmoe(raw_model, os.path.join(ckpt, "model"))
                torch.save({"step": step + 1, "optimizer": optimizer.state_dict()},
                           os.path.join(ckpt, "trainer_state.pt"))
                print(f"saved {ckpt}")
            if world > 1:
                dist.barrier()

    if wandb_run is not None:
        wandb_run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
