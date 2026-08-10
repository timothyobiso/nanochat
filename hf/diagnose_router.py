"""
Phase B0: zero-training router diagnostics on a pretrained OLMoE.

Modes (--mode, default all):
  agreement  Per-layer top-1 / top-k(Jaccard) agreement between the learned
             gate and each fixed router, after Hungarian matching of expert
             identities on a top-1 confusion matrix built from the first
             --match-tokens, measured on the remaining calibration tokens.
  seeds      Search FPE/VSA base seeds by mean top-1 agreement on a stored
             hidden-state subsample; best seeds feed the B1 healing runs.
  swap       Perplexity under hard router swap: all patched layers at once
             and (--per-layer-swap) one layer at a time.
  distill    Least-squares distillation of each learned gate into VSA
             structure. Uses the linearity scores_e(x) = memory · bind(x, ids_e):
             - 'iso' fit: closed form per FFT frequency, ids fixed from seed
               (d free values vs the gate's E*d = E-fold compression);
             - 'alt' fit: alternating ridge solves for memory and ids on
               captured hidden states (param-matched to the linear gate).

Example (node, 1 GPU):
  python -m hf.diagnose_router --model allenai/OLMoE-1B-7B-0924 \
      --data-dir /data/$USER/hf_data/fineweb_edu_olmoe --device cuda \
      --routers vsa_fpe,vsa_random,direct_fpe,hash --out b0_report.json

Everything runs streaming over batches; nothing model-sized is materialized
beyond one hidden-state subsample for seeds/distill.
"""

import argparse
import json
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

from nanochat.gpt import hrr_bind, hrr_unbind
from hf.data import distributed_data_loader
from hf.patch_olmoe import LAYER_SEED_STRIDE
from hf.routers import FixedRouterGate, build_router, permute_expert_ids


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--mode", type=str, default="all", choices=["all", "agreement", "seeds", "swap", "distill"])
    p.add_argument("--routers", type=str, default="vsa_fpe,vsa_random,direct_fpe,hash")
    p.add_argument("--router-seed", type=int, default=0)
    p.add_argument("--layers", type=str, default="", help="comma list; default all MoE layers")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--calib-tokens", type=int, default=4_000_000)
    p.add_argument("--match-tokens", type=int, default=512_000)
    p.add_argument("--seed-count", type=int, default=512)
    p.add_argument("--seed-tokens", type=int, default=65_536)
    p.add_argument("--swap-eval-tokens", type=int, default=2_000_000)
    p.add_argument("--per-layer-swap", action="store_true")
    p.add_argument("--distill-router", type=str, default="vsa_fpe", choices=["vsa_fpe", "vsa_random"])
    p.add_argument("--distill-tokens", type=int, default=131_072)
    p.add_argument("--distill-ridge", type=float, default=1e-4, help="relative ridge for the alternating fit")
    p.add_argument("--distill-rounds", type=int, default=2)
    p.add_argument("--distill-ppl", action="store_true", help="also eval ppl with fitted routers installed (all layers)")
    p.add_argument("--out", type=str, default="b0_report.json")
    return p.parse_args()


# ---------------- capture machinery ----------------

class MoeInputCatcher:
    """Forward pre-hooks on every MoE block: grabs exactly what the gate sees
    (the post-attention-layernorm hidden states, flattened to (N, d))."""

    def __init__(self, model, layer_indices):
        self.hidden = {}
        self.handles = [
            model.model.layers[i].mlp.register_forward_pre_hook(self._hook(i))
            for i in layer_indices
        ]

    def _hook(self, i):
        def fn(module, args):
            x = args[0]
            self.hidden[i] = x.detach().reshape(-1, x.shape[-1])
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def batches(args, device):
    loader = distributed_data_loader(args.data_dir, args.split, args.batch_size, args.seq_len, device=device)
    while True:
        inputs, targets, _ = next(loader)
        yield inputs, targets


def hungarian_perm(confusion):
    """confusion[a, b] = #tokens with learned top-1 = a and router top-1 = b.
    Returns perm with perm[a] = matched router expert for learned slot a;
    aligned router logits are logits[:, perm]."""
    rows, cols = linear_sum_assignment(-confusion.cpu().numpy())
    perm = torch.empty(confusion.shape[0], dtype=torch.long)
    perm[torch.as_tensor(rows)] = torch.as_tensor(cols)
    return perm


def topk_jaccard(a_logits, b_logits, k):
    a = torch.zeros_like(a_logits, dtype=torch.bool).scatter_(1, a_logits.topk(k, dim=-1).indices, True)
    b = torch.zeros_like(a).scatter_(1, b_logits.topk(k, dim=-1).indices, True)
    inter = (a & b).sum(-1).float()
    union = (a | b).sum(-1).float()
    return (inter / union).sum().item()


# ---------------- agreement ----------------

def run_agreement(model, args, layer_indices, device, report):
    E = model.config.num_experts
    top_k = model.config.num_experts_per_tok
    dim = model.config.hidden_size
    router_types = args.routers.split(",")
    routers = {
        rt: {i: build_router(rt, dim, E, top_k, args.router_seed + LAYER_SEED_STRIDE * i).to(device)
             for i in layer_indices}
        for rt in router_types
    }
    confusion = {rt: {i: torch.zeros(E, E, device=device) for i in layer_indices} for rt in router_types}
    perms = {rt: {} for rt in router_types}
    agree = {rt: {i: {"top1": 0.0, "jaccard": 0.0} for i in layer_indices} for rt in router_types}

    catcher = MoeInputCatcher(model, layer_indices)
    seen, matched = 0, False
    measure_tokens = 0
    with torch.no_grad():
        for inputs, _ in batches(args, device):
            model(input_ids=inputs)
            n = inputs.numel()
            for i in layer_indices:
                h = catcher.hidden[i].float()
                learned = model.model.layers[i].mlp.gate(catcher.hidden[i]).float()
                l_top1 = learned.argmax(-1)
                for rt in router_types:
                    logits = routers[rt][i](h)
                    if not matched:
                        idx = l_top1 * E + logits.argmax(-1)
                        confusion[rt][i] += torch.bincount(idx, minlength=E * E).reshape(E, E).float()
                    else:
                        aligned = logits[:, perms[rt][i]]
                        agree[rt][i]["top1"] += (l_top1 == aligned.argmax(-1)).sum().item()
                        agree[rt][i]["jaccard"] += topk_jaccard(learned, aligned, top_k)
            seen += n
            if not matched and seen >= args.match_tokens:
                for rt in router_types:
                    for i in layer_indices:
                        perms[rt][i] = hungarian_perm(confusion[rt][i]).to(device)
                matched, seen = True, 0
                continue
            if matched:
                measure_tokens = seen
                if seen >= args.calib_tokens - args.match_tokens:
                    break
    catcher.remove()

    report["agreement"] = {
        rt: {
            str(i): {
                "top1": agree[rt][i]["top1"] / measure_tokens,
                f"top{top_k}_jaccard": agree[rt][i]["jaccard"] / measure_tokens,
                "chance_top1": 1.0 / E,
                "perm": perms[rt][i].tolist(),
            }
            for i in layer_indices
        }
        for rt in router_types
    }
    for rt in router_types:
        vals = [report["agreement"][rt][str(i)]["top1"] for i in layer_indices]
        print(f"[agreement] {rt}: top1 mean {sum(vals)/len(vals):.4f} (chance {1/E:.4f}), "
              f"per-layer min/max {min(vals):.4f}/{max(vals):.4f}")
    return report


# ---------------- seed search ----------------

def collect_hidden_sample(model, args, layer_indices, device, num_tokens):
    """One sweep; returns {layer: (hidden fp16 cpu (N,d), learned_logits fp32 cpu)}."""
    catcher = MoeInputCatcher(model, layer_indices)
    stored = {i: [] for i in layer_indices}
    learned = {i: [] for i in layer_indices}
    seen = 0
    with torch.no_grad():
        for inputs, _ in batches(args, device):
            model(input_ids=inputs)
            for i in layer_indices:
                h = catcher.hidden[i]
                stored[i].append(h.half().cpu())
                learned[i].append(model.model.layers[i].mlp.gate(h).float().cpu())
            seen += inputs.numel()
            if seen >= num_tokens:
                break
    catcher.remove()
    return ({i: torch.cat(stored[i])[:num_tokens] for i in layer_indices},
            {i: torch.cat(learned[i])[:num_tokens] for i in layer_indices})


def run_seed_search(model, args, layer_indices, device, report):
    E = model.config.num_experts
    top_k = model.config.num_experts_per_tok
    dim = model.config.hidden_size
    hidden, learned = collect_hidden_sample(model, args, layer_indices, device, args.seed_tokens)
    hidden = {i: h.float().to(device) for i, h in hidden.items()}
    l_top1 = {i: learned[i].to(device).argmax(-1) for i in layer_indices}

    results = []
    for seed in range(args.seed_count):
        scores = []
        for i in layer_indices:
            router = build_router(args.distill_router, dim, E, top_k,
                                  seed + LAYER_SEED_STRIDE * i).to(device)
            logits = router(hidden[i])
            idx = l_top1[i] * E + logits.argmax(-1)
            confusion = torch.bincount(idx, minlength=E * E).reshape(E, E).float()
            perm = hungarian_perm(confusion).to(device)
            scores.append((l_top1[i] == logits[:, perm].argmax(-1)).float().mean().item())
        results.append({"seed": seed, "top1_mean": sum(scores) / len(scores)})
        if seed % 50 == 0:
            print(f"[seeds] {seed}/{args.seed_count} best so far "
                  f"{max(r['top1_mean'] for r in results):.4f}")
    results.sort(key=lambda r: -r["top1_mean"])
    report["seed_search"] = {"router": args.distill_router, "tokens": args.seed_tokens,
                             "top16": results[:16], "chance_top1": 1.0 / E}
    print(f"[seeds] best: {results[0]}")
    return report


# ---------------- swap perplexity ----------------

@contextmanager
def swapped_gates(model, router_type, seed, layer_indices, perms=None, buffers=None):
    """Temporarily replace gates with FixedRouterGates; restores on exit.
    buffers: optional {layer: {buffer_name: tensor}} overrides (distilled fits)."""
    dim, E = model.config.hidden_size, model.config.num_experts
    top_k = model.config.num_experts_per_tok
    originals = {}
    for i in layer_indices:
        block = model.model.layers[i].mlp
        router = build_router(router_type, dim, E, top_k, seed + LAYER_SEED_STRIDE * i)
        if perms is not None and i in perms:
            permute_expert_ids(router, perms[i])
        if buffers is not None and i in buffers:
            for name, value in buffers[i].items():
                getattr(router, name).copy_(value)
        originals[i] = block.gate
        block.gate = FixedRouterGate(router).to(device=next(block.parameters()).device)
    try:
        yield
    finally:
        for i, gate in originals.items():
            model.model.layers[i].mlp.gate = gate


@torch.no_grad()
def eval_ce(model, args, device, num_tokens):
    total, steps = 0.0, 0
    for inputs, targets in batches(args, device):
        logits = model(input_ids=inputs).logits
        total += F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1)).item()
        steps += 1
        if steps * inputs.numel() >= num_tokens:
            break
    return total / steps


def run_swap(model, args, layer_indices, device, report):
    router_types = [rt for rt in args.routers.split(",")]
    perms = None
    if "agreement" in report:  # reuse Hungarian matches when available
        perms = {rt: {i: torch.as_tensor(report["agreement"][rt][str(i)]["perm"])
                      for i in layer_indices} for rt in router_types}
    base_ce = eval_ce(model, args, device, args.swap_eval_tokens)
    out = {"baseline": {"ce": base_ce, "ppl": math.exp(base_ce)}}
    print(f"[swap] baseline ce {base_ce:.4f} ppl {math.exp(base_ce):.2f}")
    for rt in router_types:
        rt_perms = perms[rt] if perms else None
        with swapped_gates(model, rt, args.router_seed, layer_indices, perms=rt_perms):
            ce = eval_ce(model, args, device, args.swap_eval_tokens)
        out[rt] = {"all_layers": {"ce": ce, "ppl": math.exp(ce)}}
        print(f"[swap] {rt} all-layers ce {ce:.4f} ppl {math.exp(ce):.2f}")
        if args.per_layer_swap:
            per_layer = {}
            for i in layer_indices:
                with swapped_gates(model, rt, args.router_seed, [i],
                                   perms={i: rt_perms[i]} if rt_perms else None):
                    ce_i = eval_ce(model, args, device, args.swap_eval_tokens)
                per_layer[str(i)] = {"ce": ce_i, "ppl": math.exp(ce_i)}
                print(f"[swap] {rt} layer {i} ce {ce_i:.4f}")
            out[rt]["per_layer"] = per_layer
    report["swap"] = out
    return report


# ---------------- distillation ----------------

def iso_fit_memory(W, ids):
    """Closed-form isotropic fit: find memory minimizing, per FFT frequency f,
    sum_e |fft(m)_f * conj(fft(ids_e))_f - fft(W_e)_f|^2 — i.e. the fixed
    holographic structure best matching the learned gate for ALL inputs."""
    Wf = torch.fft.fft(W.double())
    If = torch.fft.fft(ids.double())
    m_f = (Wf * If).sum(0) / (If.abs() ** 2).sum(0).clamp(min=1e-12)
    return torch.fft.ifft(m_f).real.float()


def alt_fit(W, ids0, memory0, X, ridge, rounds):
    """Alternating ridge LS on captured hidden states X (N, d):
    targets Y = X @ W.T; features are linear in memory (rows bind(x, ids_e))
    and, given the memory, linear in each ids_e (shared features unbind(m, x))."""
    Y = X @ W.T  # (N, E)
    memory, ids = memory0.clone(), ids0.clone()
    d = X.shape[1]
    eye = torch.eye(d, device=X.device)
    for _ in range(rounds):
        # ids step: shared design matrix R = unbind(m, X)
        R = hrr_unbind(memory.unsqueeze(0), X)
        AtA = R.T @ R
        AtA += ridge * AtA.diagonal().mean() * eye
        ids = torch.linalg.solve(AtA, R.T @ Y).T  # (E, d)
        # memory step: accumulate normal equations over experts
        AtA_m = torch.zeros(d, d, device=X.device)
        Aty_m = torch.zeros(d, device=X.device)
        for e in range(W.shape[0]):
            A_e = hrr_bind(X, ids[e].unsqueeze(0))
            AtA_m += A_e.T @ A_e
            Aty_m += A_e.T @ Y[:, e]
        AtA_m += ridge * AtA_m.diagonal().mean() * eye
        memory = torch.linalg.solve(AtA_m, Aty_m)
    return memory, ids


def distill_metrics(memory, ids, W, X_test, top_k):
    Y = X_test @ W.T
    pred = hrr_unbind(memory.unsqueeze(0), X_test) @ ids.T
    return {
        "logit_mse": F.mse_loss(pred, Y).item(),
        "logit_var": Y.var().item(),
        "top1": (Y.argmax(-1) == pred.argmax(-1)).float().mean().item(),
        f"top{top_k}_jaccard": topk_jaccard(Y, pred, top_k) / len(Y),
    }


def run_distill(model, args, layer_indices, device, report):
    E, dim = model.config.num_experts, model.config.hidden_size
    top_k = model.config.num_experts_per_tok
    hidden, _ = collect_hidden_sample(model, args, layer_indices, device, args.distill_tokens)
    out = {}
    fitted_buffers = {"iso": {}, "alt": {}}
    for i in layer_indices:
        X = hidden[i].float().to(device)
        n_fit = int(0.9 * len(X))
        X_fit, X_test = X[:n_fit], X[n_fit:]
        W = model.model.layers[i].mlp.gate.weight.detach().float().to(device)
        router = build_router(args.distill_router, dim, E, top_k,
                              args.router_seed + LAYER_SEED_STRIDE * i).to(device)
        ids0 = router.expert_ids.clone()

        m_iso = iso_fit_memory(W, ids0)
        m_alt, ids_alt = alt_fit(W, ids0, m_iso, X_fit, args.distill_ridge, args.distill_rounds)

        out[str(i)] = {
            "iso": distill_metrics(m_iso, ids0, W, X_test, top_k),
            "alt": distill_metrics(m_alt, ids_alt, W, X_test, top_k),
            "params": {"gate": W.numel(), "iso": dim, "alt": dim + ids_alt.numel()},
        }
        fitted_buffers["iso"][i] = {"memory": m_iso.cpu()}
        fitted_buffers["alt"][i] = {"memory": m_alt.cpu(), "expert_ids": ids_alt.cpu()}
        print(f"[distill] layer {i}: iso top1 {out[str(i)]['iso']['top1']:.4f} "
              f"alt top1 {out[str(i)]['alt']['top1']:.4f} "
              f"(mse {out[str(i)]['iso']['logit_mse']:.3g}/{out[str(i)]['alt']['logit_mse']:.3g})")

    if args.distill_ppl:
        for variant in ("iso", "alt"):
            with swapped_gates(model, args.distill_router, args.router_seed, layer_indices,
                               buffers=fitted_buffers[variant]):
                ce = eval_ce(model, args, device, args.swap_eval_tokens)
            out[f"{variant}_all_layers_ppl"] = math.exp(ce)
            print(f"[distill] {variant} all-layers ppl {math.exp(ce):.2f}")
    report["distill"] = {"router": args.distill_router, "layers": out}
    return report


def main():
    args = get_args()
    device = args.device
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = OlmoeForCausalLM.from_pretrained(args.model, dtype=dtype, revision=args.revision).to(device).eval()
    layer_indices = ([int(x) for x in args.layers.split(",")] if args.layers
                     else [i for i, layer in enumerate(model.model.layers) if hasattr(layer.mlp, "gate")])
    report = {"model": args.model, "config": vars(args)}
    modes = ["agreement", "seeds", "swap", "distill"] if args.mode == "all" else [args.mode]
    runners = {"agreement": run_agreement, "seeds": run_seed_search, "swap": run_swap, "distill": run_distill}
    for mode in modes:
        report = runners[mode](model, args, layer_indices, device, report)
        with open(args.out, "w") as f:  # checkpoint the report after each mode
            json.dump(report, f, indent=2)
    print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
