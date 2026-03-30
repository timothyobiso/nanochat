"""
Generate all analysis figures for the VSA MoE router paper.

Produces:
  1. Validation loss curves across router types
  2. Expert utilization heatmaps
  3. Router latency comparison
  4. Per-token routing examples

Run as (single GPU is fine for analysis):
  python -m scripts.paper_analysis

Or with DDP for expert utilization on full model:
  torchrun --standalone --nproc_per_node=8 -m scripts.paper_analysis
"""
import os
import re
import json
import glob
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from nanochat.common import get_base_dir, get_dist_info, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model, load_checkpoint, find_last_step
from nanochat.gpt import norm

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default="paper_figures")
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--num-batches", type=int, default=50, help="Batches for utilization analysis")
parser.add_argument("--latency-iters", type=int, default=200, help="Iterations for latency benchmark")
parser.add_argument("--yes", "-y", action="store_true", help="Include all discovered runs without asking")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
device = torch.device(device_type)

# Color palette for up to 10 runs
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

def discover_runs():
    """Scan base_checkpoints for available runs, read their router type from metadata."""
    base_dir = get_base_dir()
    ckpt_root = os.path.join(base_dir, "base_checkpoints")
    if not os.path.exists(ckpt_root):
        print(f"No base_checkpoints directory found at {ckpt_root}")
        return []

    runs = []
    for tag in sorted(os.listdir(ckpt_root)):
        tag_dir = os.path.join(ckpt_root, tag)
        if not os.path.isdir(tag_dir):
            continue
        # Find the latest meta file
        meta_files = sorted(glob.glob(os.path.join(tag_dir, "meta_*.json")))
        if not meta_files:
            continue
        with open(meta_files[-1]) as f:
            meta = json.load(f)
        router_type = meta.get("model_config", {}).get("moe_router_type", "unknown")
        step = meta.get("step", "?")
        n_params = meta.get("model_config", {}).get("n_layer", "?")
        depth = meta.get("user_config", {}).get("depth", "?") if "user_config" in meta else "?"
        val_bpb = meta.get("val_bpb", "?")
        runs.append({
            "tag": tag,
            "router_type": router_type,
            "step": step,
            "depth": depth,
            "val_bpb": val_bpb,
            "num_checkpoints": len(meta_files),
        })
    return runs

def interactive_select(discovered):
    """Ask the user which runs to include."""
    print("\n" + "=" * 70)
    print("Discovered runs in base_checkpoints:")
    print("=" * 70)
    print(f"  {'#':<4} {'Tag':<25} {'Router':<15} {'Depth':<7} {'Steps':<8} {'Val BPB':<10} {'Ckpts'}")
    print(f"  {'-'*79}")
    for i, r in enumerate(discovered):
        bpb_str = f"{r['val_bpb']:.4f}" if isinstance(r['val_bpb'], float) else str(r['val_bpb'])
        print(f"  {i:<4} {r['tag']:<25} {r['router_type']:<15} {r['depth']:<7} {r['step']:<8} {bpb_str:<10} {r['num_checkpoints']}")
    print()

    if args.yes:
        selected_indices = list(range(len(discovered)))
        print("  --yes flag: including all runs")
    else:
        response = input("  Enter run numbers to include (comma-separated, or 'all'): ").strip()
        if response.lower() == "all" or response == "":
            selected_indices = list(range(len(discovered)))
        else:
            selected_indices = [int(x.strip()) for x in response.split(",") if x.strip().isdigit()]

    selected = []
    for idx in selected_indices:
        if 0 <= idx < len(discovered):
            r = discovered[idx]
            # Ask for a display label
            if not args.yes:
                default_label = r["router_type"]
                if r["depth"] != "?":
                    default_label = f"d{r['depth']}_{r['router_type']}"
                label = input(f"  Label for '{r['tag']}' [{default_label}]: ").strip()
                if not label:
                    label = default_label
            else:
                label = r["router_type"] if r["depth"] == "?" else f"d{r['depth']}_{r['router_type']}"
            selected.append({
                "label": label,
                "tag": r["tag"],
                "color": COLORS[len(selected) % len(COLORS)],
                "router_type": r["router_type"],
            })

    print(f"\n  Selected {len(selected)} runs:")
    for s in selected:
        print(f"    {s['label']} ({s['tag']}) [{s['color']}]")
    print()
    return selected

base_dir = get_base_dir()
discovered = discover_runs()
if not discovered:
    print("No runs found. Exiting.")
    exit(0)
RUNS = interactive_select(discovered)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Validation loss curves from checkpoint metadata
# ─────────────────────────────────────────────────────────────────────────────
def extract_loss_curve(model_tag):
    """Read all meta_*.json files from a checkpoint dir and extract (step, val_bpb)."""
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", model_tag)
    if not os.path.exists(checkpoint_dir):
        print0(f"  Checkpoint dir not found: {checkpoint_dir}")
        return [], []
    meta_files = sorted(glob.glob(os.path.join(checkpoint_dir, "meta_*.json")))
    steps, bpbs = [], []
    for f in meta_files:
        with open(f) as fh:
            meta = json.load(fh)
        step = meta.get("step", None)
        val_bpb = meta.get("val_bpb", None)
        if step is not None and val_bpb is not None:
            steps.append(step)
            bpbs.append(val_bpb)
    # Sort by step
    pairs = sorted(zip(steps, bpbs))
    steps = [p[0] for p in pairs]
    bpbs = [p[1] for p in pairs]
    return steps, bpbs

def plot_loss_curves():
    print0("="*60)
    print0("1. Plotting validation loss curves")
    print0("="*60)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for run in RUNS:
        steps, bpbs = extract_loss_curve(run["tag"])
        if steps:
            ax.plot(steps, bpbs, label=run["label"], color=run["color"], linewidth=1.5)
            print0(f"  {run['label']}: {len(steps)} checkpoints, final bpb={bpbs[-1]:.4f}")
        else:
            print0(f"  {run['label']}: no data found")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation BPB")
    ax.set_title("Validation Loss During Pretraining")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(args.output_dir, "loss_curves.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print0(f"  Saved to {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Expert utilization heatmaps
# ─────────────────────────────────────────────────────────────────────────────
def get_expert_counts(model, config, device, num_batches):
    """Run model forward and collect per-layer expert token counts."""
    from nanochat.dataloader import tokenizing_distributed_data_loader
    from contextlib import nullcontext

    moe_layer_indices = [i for i, b in enumerate(model.transformer.h) if b.is_moe]
    num_moe = len(moe_layer_indices)
    total_counts = [torch.zeros(config.num_experts, device=device) for _ in range(num_moe)]

    loader = tokenizing_distributed_data_loader(4, config.sequence_len, "val", device=device)
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    with torch.no_grad(), autocast_ctx:
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            B, T = x.size()
            cos_sin = model.cos[:, :T], model.sin[:, :T]
            h = norm(model.transformer.wte(x))
            x0 = h
            moe_idx = 0
            for i, block in enumerate(model.transformer.h):
                h = model.resid_lambdas[i] * h + model.x0_lambdas[i] * x0
                h_out, _ = block(h, cos_sin, None)
                if block.is_moe:
                    h_normed = norm(h)
                    x_flat = h_normed.view(-1, h_normed.shape[-1])
                    router_logits = block.moe.router(x_flat)
                    _, top_k_idx = torch.topk(router_logits, config.num_experts_per_tok, dim=-1)
                    one_hot = F.one_hot(top_k_idx, config.num_experts).float()
                    total_counts[moe_idx] += one_hot.sum(dim=1).sum(dim=0)
                    moe_idx += 1
                h = h_out

    # Normalize to fractions
    fracs = []
    for c in total_counts:
        f = c / c.sum()
        fracs.append(f.cpu().numpy())
    return moe_layer_indices, fracs

def plot_utilization_heatmaps():
    print0("\n" + "="*60)
    print0("2. Expert utilization heatmaps")
    print0("="*60)

    available_runs = []
    all_fracs = {}
    all_layer_indices = {}

    for run in RUNS:
        try:
            model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=run["tag"])
            model.eval()
            config = model.config
            print0(f"  Analyzing {run['label']}...")
            layer_indices, fracs = get_expert_counts(model, config, device, args.num_batches)
            all_fracs[run["label"]] = fracs
            all_layer_indices[run["label"]] = layer_indices
            available_runs.append(run)
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
        except Exception as e:
            print0(f"  Skipping {run['label']}: {e}")

    if not available_runs:
        print0("  No models available for utilization analysis")
        return

    n = len(available_runs)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 3), squeeze=False)
    perfect = 1.0 / 8  # for 8 experts

    for idx, run in enumerate(available_runs):
        ax = axes[0, idx]
        fracs = all_fracs[run["label"]]
        mat = np.array(fracs)  # (num_moe_layers, num_experts)
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=mat.max())
        ax.set_title(run["label"], fontsize=11)
        ax.set_xlabel("Expert")
        if idx == 0:
            ax.set_ylabel("MoE Layer")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels([str(li) for li in all_layer_indices[run["label"]]])
        # Annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i,j] > mat.max()*0.6 else "black")

    fig.suptitle("Expert Token Fraction by Layer and Router Type", fontsize=13, y=1.02)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Fraction of tokens")
    path = os.path.join(args.output_dir, "expert_utilization.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print0(f"  Saved to {path}")

    # Also print Gini + entropy summary
    print0(f"\n  {'Router':<15} {'Layer':<8} {'Gini':<8} {'Entropy':<10} {'Min/Max':<10}")
    print0(f"  {'-'*51}")
    for run in available_runs:
        fracs = all_fracs[run["label"]]
        for li, f in zip(all_layer_indices[run["label"]], fracs):
            ft = torch.tensor(f)
            sorted_f = torch.sort(ft).values
            n_e = len(sorted_f)
            indices = torch.arange(1, n_e+1, dtype=torch.float)
            gini = (2*(indices*sorted_f).sum()/(n_e*sorted_f.sum())) - (n_e+1)/n_e
            entropy = -(ft * ft.clamp(min=1e-10).log()).sum()
            max_ent = torch.tensor(n_e, dtype=torch.float).log()
            print0(f"  {run['label']:<15} {li:<8} {gini:.4f}   {entropy:.4f}/{max_ent:.4f}  {ft.min():.3f}/{ft.max():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Router latency benchmark
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_router_latency():
    print0("\n" + "="*60)
    print0("3. Router latency benchmark")
    print0("="*60)

    results = {}
    for run in RUNS:
        try:
            model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=run["tag"])
            model.eval()
            config = model.config

            # Find the first MoE layer's router
            router = None
            for block in model.transformer.h:
                if block.is_moe:
                    router = block.moe.router
                    break

            if router is None:
                print0(f"  {run['label']}: no MoE layers found")
                continue

            # Benchmark
            B, T, C = 4, config.sequence_len, config.n_embd
            x_flat = torch.randn(B * T, C, device=device, dtype=torch.bfloat16)

            # Warmup
            for _ in range(20):
                _ = router(x_flat)
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Time it
            times = []
            for _ in range(args.latency_iters):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = router(x_flat)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # ms

            mean_ms = np.mean(times)
            std_ms = np.std(times)
            results[run["label"]] = (mean_ms, std_ms)
            print0(f"  {run['label']:<15} {mean_ms:.3f} ± {std_ms:.3f} ms")

            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
        except Exception as e:
            print0(f"  Skipping {run['label']}: {e}")

    if not results:
        return

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(results.keys())
    means = [results[l][0] for l in labels]
    stds = [results[l][1] for l in labels]
    colors = [r["color"] for r in RUNS if r["label"] in results]
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Router Latency per Forward Call\n(B×T={4*2048} tokens)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{m:.2f}ms", ha="center", va="bottom", fontsize=9)
    path = os.path.join(args.output_dir, "router_latency.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print0(f"  Saved to {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-token routing example
# ─────────────────────────────────────────────────────────────────────────────
def plot_token_routing_examples():
    print0("\n" + "="*60)
    print0("4. Per-token routing examples")
    print0("="*60)

    example_text = "The capital of France is Paris, which is known for the Eiffel Tower."

    available_runs = []
    routing_data = {}

    for run in RUNS:
        try:
            model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=run["tag"])
            model.eval()
            config = model.config

            # Tokenize
            tokens = tokenizer.encode(example_text, add_special_tokens=True)
            token_strs = [tokenizer.decode([t]) for t in tokens]
            x = torch.tensor([tokens], device=device)

            with torch.no_grad():
                B, T = x.size()
                cos_sin = model.cos[:, :T], model.sin[:, :T]
                h = norm(model.transformer.wte(x))
                x0 = h

                layer_assignments = []
                for i, block in enumerate(model.transformer.h):
                    h = model.resid_lambdas[i] * h + model.x0_lambdas[i] * x0
                    h_out, _ = block(h, cos_sin, None)
                    if block.is_moe:
                        h_normed = norm(h)
                        x_flat = h_normed.view(-1, h_normed.shape[-1])
                        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                            router_logits = block.moe.router(x_flat)
                        _, top_k_idx = torch.topk(router_logits, config.num_experts_per_tok, dim=-1)
                        layer_assignments.append(top_k_idx[0].cpu().numpy())  # (T, K)
                    h = h_out

            routing_data[run["label"]] = {
                "tokens": token_strs,
                "assignments": layer_assignments,
            }
            available_runs.append(run)
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
        except Exception as e:
            print0(f"  Skipping {run['label']}: {e}")

    if not available_runs:
        print0("  No models available")
        return

    # Plot: for each router, show a heatmap of (MoE layer × token) colored by primary expert
    n = len(available_runs)
    num_tokens = len(routing_data[available_runs[0]["label"]]["tokens"])
    # Truncate long sequences for readability
    max_tokens = min(num_tokens, 30)

    fig, axes = plt.subplots(n, 1, figsize=(max(12, max_tokens * 0.5), 2 * n + 1), squeeze=False)

    cmap = plt.cm.Set1
    for idx, run in enumerate(available_runs):
        ax = axes[idx, 0]
        data = routing_data[run["label"]]
        tokens = data["tokens"][:max_tokens]
        assignments = data["assignments"]

        # Build matrix of primary expert assignment
        num_layers = len(assignments)
        mat = np.zeros((num_layers, max_tokens))
        for li in range(num_layers):
            for ti in range(max_tokens):
                mat[li, ti] = assignments[li][ti, 0]  # primary expert

        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=7, interpolation="nearest")
        ax.set_title(run["label"], fontsize=11)
        if idx == n - 1:
            ax.set_xticks(range(max_tokens))
            ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=7)
        else:
            ax.set_xticks([])
        ax.set_ylabel("MoE Layer")
        ax.set_yticks(range(num_layers))

    fig.suptitle("Primary Expert Assignment per Token", fontsize=13, y=1.02)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Expert ID", ticks=range(8))
    path = os.path.join(args.output_dir, "token_routing.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print0(f"  Saved to {path}")

    # Print text version
    for run in available_runs:
        data = routing_data[run["label"]]
        print0(f"\n  {run['label']}:")
        for li, asn in enumerate(data["assignments"]):
            experts = [f"{asn[t,0]}" for t in range(min(len(data["tokens"]), 20))]
            print0(f"    Layer {li}: {' '.join(experts)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary table
# ─────────────────────────────────────────────────────────────────────────────
def print_summary_table():
    print0("\n" + "="*60)
    print0("5. Summary table (for paper)")
    print0("="*60)

    print0(f"\n  {'Router':<15} {'Tag':<20} {'Router Params':<15}")
    print0(f"  {'-'*50}")
    for run in RUNS:
        try:
            model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=run["tag"])
            config = model.config
            # Count router params
            router_params = 0
            for block in model.transformer.h:
                if block.is_moe:
                    router_params += sum(p.numel() for p in block.moe.router.parameters())
            total_params = sum(p.numel() for p in model.parameters())
            print0(f"  {run['label']:<15} {run['tag'] or 'default':<20} {router_params:,}")
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
        except Exception as e:
            print0(f"  {run['label']:<15} {'error':<20} {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print0(f"Output directory: {args.output_dir}")
    print0(f"Runs: {[(r['label'], r['tag']) for r in RUNS]}")

    # 1. Loss curves (no model loading needed, just reads JSON)
    plot_loss_curves()

    # 2. Expert utilization (loads each model)
    plot_utilization_heatmaps()

    # 3. Router latency
    benchmark_router_latency()

    # 4. Token routing examples
    plot_token_routing_examples()

    # 5. Summary table
    print_summary_table()

    print0(f"\nAll figures saved to {args.output_dir}/")
