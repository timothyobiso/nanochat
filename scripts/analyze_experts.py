"""
Analyze expert utilization for MoE models.

Run as:
torchrun --standalone --nproc_per_node=8 -m scripts.analyze_experts -- --model-tag d8_fpe_vsa
"""
import argparse
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, print0, compute_cleanup, autodetect_device_type
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.gpt import norm

parser = argparse.ArgumentParser()
parser.add_argument("--model-tag", type=str, default=None)
parser.add_argument("--num-batches", type=int, default=50)
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--device-type", type=str, default="")
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag)
model.eval()

config = model.config
sequence_len = meta["model_config"]["sequence_len"]
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# Find MoE layers
moe_layer_indices = [i for i, b in enumerate(model.transformer.h) if b.is_moe]
num_moe_layers = len(moe_layer_indices)
print0(f"Found {num_moe_layers} MoE layers at indices: {moe_layer_indices}")
print0(f"Router type: {config.moe_router_type}")
print0(f"Experts: {config.num_experts}, top-k: {config.num_experts_per_tok}")

# Collect counts per MoE layer
total_counts = [torch.zeros(config.num_experts, device=device) for _ in range(num_moe_layers)]
total_tokens = 0

loader = tokenizing_distributed_data_loader(args.device_batch_size, sequence_len, "val", device=device)

with torch.no_grad(), autocast_ctx:
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        B, T = x.size()
        T0 = 0
        cos_sin = model.cos[:, T0:T0+T], model.sin[:, T0:T0+T]

        h = model.transformer.wte(x)
        h = norm(h)
        x0 = h

        moe_idx = 0
        for i, block in enumerate(model.transformer.h):
            h = model.resid_lambdas[i] * h + model.x0_lambdas[i] * x0
            h_out, _ = block(h, cos_sin, None)

            if block.is_moe:
                # Capture routing decisions
                h_normed = norm(h)
                C = h_normed.shape[-1]
                x_flat = h_normed.view(-1, C)
                router_logits = block.moe.router(x_flat)
                _, top_k_indices = torch.topk(router_logits, config.num_experts_per_tok, dim=-1)
                one_hot = F.one_hot(top_k_indices, config.num_experts).float()
                counts = one_hot.sum(dim=1).sum(dim=0)
                total_counts[moe_idx] += counts
                moe_idx += 1

            h = h_out
            total_tokens += B * T if block.is_moe else 0

# Only print from rank 0
if ddp_rank == 0:
    print(f"\n{'='*60}")
    print(f"Expert Utilization: {args.model_tag or 'default'}")
    print(f"Router: {config.moe_router_type}")
    print(f"{'='*60}")

    for layer_idx in range(num_moe_layers):
        counts = total_counts[layer_idx]
        fracs = counts / counts.sum()

        # Gini coefficient
        sorted_fracs = torch.sort(fracs).values
        n = len(sorted_fracs)
        indices = torch.arange(1, n + 1, device=device, dtype=torch.float)
        gini = (2 * (indices * sorted_fracs).sum() / (n * sorted_fracs.sum())) - (n + 1) / n

        # Entropy
        entropy = -(fracs * fracs.clamp(min=1e-10).log()).sum()
        max_entropy = torch.tensor(n, dtype=torch.float).log()

        print(f"\nMoE Layer {layer_idx} (block {moe_layer_indices[layer_idx]}):")
        print(f"  Tokens/expert: {counts.long().tolist()}")
        print(f"  Fractions:     {[f'{f:.3f}' for f in fracs.tolist()]}")
        print(f"  Min/Max:       {fracs.min():.4f} / {fracs.max():.4f} (ratio: {fracs.min()/fracs.max():.4f})")
        print(f"  Gini:          {gini:.4f}")
        print(f"  Entropy:       {entropy:.4f} / {max_entropy:.4f} ({entropy/max_entropy*100:.1f}%)")

compute_cleanup()
