"""
Figures for the HF-side experiments, mirroring scripts/paper_analysis.py.

Consumes the metrics.jsonl + config.json each run directory accumulates
(hf/train_olmoe.py, hf/heal_olmoe.py) — no checkpoint loading needed except
for latency, which times freshly-built routers.

  python -m hf.analysis --mode phase_a --runs runA,runB --labels linear,vsa_fpe
  python -m hf.analysis --mode healing --runs heal_ctrl,heal_vsa --labels control,vsa_fpe
  python -m hf.analysis --mode latency --dim 2048 --experts 64 --top-k 8 --device cuda

Outputs (into --output-dir):
  loss_curves.pdf         val bpb vs tokens per run
  expert_utilization.pdf  layer x expert token-fraction heatmaps (latest telemetry)
  router_latency.pdf      isolated gate forward time, B*T=8192, 20 warmup + 200 timed
  healing_curves.pdf      val bpb vs tokens with the alpha anneal overlaid, plus
                          mean routing-drift per run
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def read_run(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    records = []
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        for line in f:
            records.append(json.loads(line))
    return config, records


def series(records, key):
    return ([r["step"] for r in records if key in r and r[key] is not None],
            [r[key] for r in records if key in r and r[key] is not None])


def plot_loss_curves(runs, labels, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (config, records), label in zip(runs, labels):
        steps, bpb = series(records, "val_bpb")
        tokens = [s * config["total_batch_size"] for s in steps]
        ax.plot(np.array(tokens) / 1e9, bpb, label=label, linewidth=1.5)
    ax.set_xlabel("tokens (B)")
    ax.set_ylabel("val bits/byte")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curves.pdf"))
    print("wrote loss_curves.pdf")


def plot_utilization(runs, labels, output_dir):
    panels = []
    for (config, records), label in zip(runs, labels):
        latest = next((r["expert_utilization"] for r in reversed(records)
                       if r.get("expert_utilization")), None)
        if latest:
            panels.append((label, np.array([layer["frac"] for layer in latest]),
                           [layer.get("gate_mass") for layer in latest]))
    if not panels:
        print("no expert_utilization telemetry found; skipping")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.5), squeeze=False)
    uniform = 1.0 / panels[0][1].shape[1]
    for ax, (label, frac, gate_mass) in zip(axes[0], panels):
        im = ax.imshow(frac, cmap="viridis", vmin=0, vmax=max(2.5 * uniform, frac.max()), aspect="auto")
        ax.set_title(f"{label}\n(mean gate mass {np.mean([g for g in gate_mass if g is not None]):.2f})", fontsize=9)
        ax.set_xlabel("expert")
        ax.set_ylabel("MoE layer")
        for (r, c), v in np.ndenumerate(frac):
            if frac.shape[1] <= 16:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 1.5 * uniform else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "expert_utilization.pdf"))
    print("wrote expert_utilization.pdf")


def plot_router_latency(args, output_dir):
    from hf.routers import build_router
    device = args.device
    x = torch.randn(8192, args.dim, device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    linear = torch.nn.Linear(args.dim, args.experts, bias=False).to(device, x.dtype)
    candidates = {"linear": linear}
    for rt in args.routers.split(","):
        if rt != "linear":
            candidates[rt] = build_router(rt, args.dim, args.experts, args.top_k, seed=0).to(device)
    results = {}
    for name, router in candidates.items():
        with torch.no_grad():
            for _ in range(20):
                router(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(200):
                t0 = time.perf_counter()
                router(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
        results[name] = (float(np.mean(times)), float(np.std(times)))
        print(f"[latency] {name}: {results[name][0]:.3f} ± {results[name][1]:.3f} ms")
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(results)
    ax.bar(names, [results[n][0] for n in names], yerr=[results[n][1] for n in names], capsize=3)
    ax.set_ylabel(f"ms per forward (B·T=8192, d={args.dim}, E={args.experts}, {device})")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "router_latency.pdf"))
    print("wrote router_latency.pdf")


def plot_healing_curves(runs, labels, output_dir):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for (config, records), label in zip(runs, labels):
        steps, bpb = series(records, "val_bpb")
        tokens = np.array(steps) * config["total_batch_size"] / 1e9
        ax.plot(tokens, bpb, label=label, linewidth=1.5)
        drift_steps, drifts = series(records, "routing_drift")
        if drifts:
            mean_drift = [np.mean(list(d.values())) for d in drifts]
            ax2.plot(np.array(drift_steps) * config["total_batch_size"] / 1e9, mean_drift, label=label)
    # alpha overlay from the first run that annealed
    for config, records in runs:
        a_steps, alphas = series(records, "alpha")
        if alphas and min(alphas) < 1.0:
            twin = ax.twinx()
            twin.plot(np.array(a_steps) * config["total_batch_size"] / 1e9, alphas,
                      color="gray", linestyle=":", linewidth=1, label="alpha")
            twin.set_ylabel("alpha", color="gray")
            twin.set_ylim(-0.05, 1.05)
            break
    ax.set_xlabel("tokens (B)")
    ax.set_ylabel("val bits/byte")
    ax.set_title("healing")
    ax.legend()
    ax.grid(alpha=0.3)
    ax2.set_xlabel("tokens (B)")
    ax2.set_ylabel("top-k kept vs pre-swap routing")
    ax2.set_title("routing drift (probe batch)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "healing_curves.pdf"))
    print("wrote healing_curves.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", type=str, default="phase_a", choices=["phase_a", "healing", "latency"])
    p.add_argument("--runs", type=str, default="", help="comma list of run dirs (phase_a/healing)")
    p.add_argument("--labels", type=str, default="", help="comma list; defaults to run dir basenames")
    p.add_argument("--output-dir", type=str, default="hf_figures")
    # latency knobs
    p.add_argument("--routers", type=str, default="linear,hash,vsa_random,vsa_fpe,direct_fpe")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "latency":
        plot_router_latency(args, args.output_dir)
        return
    run_dirs = [r for r in args.runs.split(",") if r]
    if not run_dirs:
        raise SystemExit("--runs is required for phase_a/healing modes")
    labels = args.labels.split(",") if args.labels else [os.path.basename(r.rstrip("/")) for r in run_dirs]
    runs = [read_run(r) for r in run_dirs]
    if args.mode == "phase_a":
        plot_loss_curves(runs, labels, args.output_dir)
        plot_utilization(runs, labels, args.output_dir)
    else:
        plot_healing_curves(runs, labels, args.output_dir)


if __name__ == "__main__":
    main()
