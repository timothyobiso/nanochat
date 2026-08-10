"""
Patch transformers OLMoE models to use nanochat's fixed routers.

transformers 4.x contract (verified against the installed 4.57.3 source,
modeling_olmoe.py): `OlmoeSparseMoeBlock.gate` is `nn.Linear(hidden, E,
bias=False)`, the block calls `self.gate(hidden_states)` on flattened
(B*T, d) states, applies softmax-then-topk (`norm_topk_prob` optional), and
returns `(hidden, router_logits)`; `OlmoeForCausalLM` computes the aux loss
from the collected logits when `output_router_logits=True` — independent of
the gate's class. So replacing the `.gate` module instance is the entire seam.

transformers 5.x MIGRATION NOTE (why pyproject pins `<5`): the gate becomes an
`OlmoeTopKRouter` returning a 3-tuple `(router_logits, router_scores,
router_indices)` with softmax/topk inside, and aux-loss collection is done by
output-recorder hooks matched via `isinstance(module, OlmoeTopKRouter)` — a
plain nn.Module adapter would silently disable the aux loss. Porting to 5.x
means subclassing OlmoeTopKRouter, keeping the 3-tuple return, and verifying
OutputRecorder still captures the logits.

Checkpoint round-trip: `save_router_olmoe` stores the router spec under the
`fixed_router` key in config.json (4.x PretrainedConfig round-trips unknown
keys) and the router buffers in the safetensors shards. Patched checkpoints
MUST be loaded with `load_router_olmoe` — bare `from_pretrained` would build
an unpatched skeleton, randomly initialize the (missing) gate.weight, drop the
router buffers, and return a silently broken model.
"""

import glob
import json
import os
import re

import torch

from transformers import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

from hf.routers import BlendedRouterGate, FixedRouterGate, build_router

ROUTER_CONFIG_KEY = "fixed_router"
SPEC_VERSION = 1
# Per-layer seed spacing: layer i uses seed + 1000*i so layers get independent
# buffers while the whole model is reproducible from one base seed.
LAYER_SEED_STRIDE = 1000


def patch_olmoe_routers(model, router_type, seed=0, blend=False, layers=None):
    """Replace the learned gate of each MoE block with a fixed (or blended) router.

    model: OlmoeForCausalLM (transformers 4.x). Modified in place.
    layers: list of layer indices to patch (default: all MoE layers).
    Records the spec on model.config so save_pretrained round-trips it.
    Returns the list of patched layer indices ('linear' patches nothing).
    """
    config = model.config
    if getattr(config, ROUTER_CONFIG_KEY, None) is not None:
        raise ValueError("Model already carries a fixed_router spec; refusing to double-patch.")
    dim = config.hidden_size
    num_experts = config.num_experts
    top_k = config.num_experts_per_tok

    patched = []
    for i, layer in enumerate(model.model.layers):
        if layers is not None and i not in layers:
            continue
        block = layer.mlp
        if not hasattr(block, "gate"):
            continue  # not an MoE block
        if isinstance(block.gate, (FixedRouterGate, BlendedRouterGate)):
            raise ValueError(f"Layer {i} gate is already patched.")
        router = build_router(router_type, dim, num_experts, top_k, seed + LAYER_SEED_STRIDE * i)
        if router is None:  # 'linear': keep the host gate
            continue
        device = block.gate.weight.device
        spec = {"router_type": router_type, "seed": seed, "layer": i}
        if blend:
            gate = BlendedRouterGate(block.gate, router, spec)
        else:
            gate = FixedRouterGate(router, spec)
        gate.to(device=device)  # device only; _FP32BufferModule keeps buffers fp32
        block.gate = gate
        patched.append(i)

    setattr(config, ROUTER_CONFIG_KEY, {
        "version": SPEC_VERSION,
        "router_type": router_type,
        "seed": seed,
        "blend": blend,
        "layers": patched if layers is not None else None,  # None = all
    })
    return patched


def patch_pretrained_olmoe(name_or_path, router_type, seed=0, blend=False,
                           dtype=torch.bfloat16, device="cpu", revision=None):
    """Load an UNPATCHED pretrained OLMoE and patch it. The one-stop entry point
    for B0/B1: standard from_pretrained first, then the gate swap."""
    model = OlmoeForCausalLM.from_pretrained(name_or_path, dtype=dtype, revision=revision)
    patch_olmoe_routers(model, router_type, seed=seed, blend=blend)
    return model.to(device)


def save_router_olmoe(model, path):
    """Save a patched model. Router buffers land in the safetensors shards and
    the spec in config.json; reload with load_router_olmoe only."""
    if getattr(model.config, ROUTER_CONFIG_KEY, None) is None:
        raise ValueError("Model has no fixed_router spec; use save_pretrained directly.")
    model.save_pretrained(path, safe_serialization=True)


_GATE_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.gate\.(weight$|router\.|learned_gate\.|alpha$|scale$)"
)


def load_router_olmoe(path, dtype=torch.bfloat16, device="cpu"):
    """Load a checkpoint saved by save_router_olmoe.

    Two-pass load:
      1. Standard from_pretrained builds the unpatched skeleton and loads every
         non-gate tensor; the gate-scope key delta is captured (not warned away)
         and asserted to be exactly the router seam — anything else is a
         corrupt/incompatible checkpoint and raises.
      2. The skeleton is patched from the stored spec, then all gate-scope
         tensors (router buffers, blended learned weights, alpha/scale) are
         restored from the checkpoint. Loading buffers from disk (rather than
         regenerating from the seed) matters: diagnostics may store *fitted*
         buffers, e.g. least-squares-distilled memories.
    """
    config = OlmoeConfig.from_pretrained(path)
    spec = getattr(config, ROUTER_CONFIG_KEY, None)
    if spec is None:
        raise ValueError(
            f"{path} has no '{ROUTER_CONFIG_KEY}' spec — it is not a patched "
            "checkpoint; load it with OlmoeForCausalLM.from_pretrained instead."
        )

    model, info = OlmoeForCausalLM.from_pretrained(
        path, dtype=dtype, output_loading_info=True
    )
    unexpected = set(info["unexpected_keys"])
    missing = set(info["missing_keys"])
    bad_unexpected = [k for k in unexpected if not _GATE_KEY_RE.match(k)]
    bad_missing = [k for k in missing if not _GATE_KEY_RE.match(k)]
    if bad_unexpected or bad_missing or info["mismatched_keys"]:
        raise RuntimeError(
            "Checkpoint/model key delta extends beyond the router seam:\n"
            f"  unexpected: {bad_unexpected}\n  missing: {bad_missing}\n"
            f"  mismatched: {info['mismatched_keys']}"
        )

    # from_pretrained restored the spec from config.json onto the unpatched
    # skeleton; clear it so the double-patch guard doesn't trip on ourselves.
    setattr(model.config, ROUTER_CONFIG_KEY, None)
    patch_olmoe_routers(
        model,
        spec["router_type"],
        seed=spec["seed"],
        blend=spec["blend"],
        layers=spec["layers"],
    )
    # patch_olmoe_routers re-set the spec; keep the stored one verbatim.
    setattr(model.config, ROUTER_CONFIG_KEY, spec)

    gate_sd = {k: v for k, v in _iter_checkpoint_tensors(path) if _GATE_KEY_RE.match(k)}
    load_missing, load_unexpected = model.load_state_dict(gate_sd, strict=False)
    if load_unexpected:
        raise RuntimeError(f"Gate tensors in checkpoint not present in patched model: {load_unexpected}")
    unrestored = [k for k in load_missing if _GATE_KEY_RE.match(k)]
    if unrestored:
        raise RuntimeError(f"Patched model gate tensors absent from checkpoint: {unrestored}")
    return model.to(device)


def _iter_checkpoint_tensors(path):
    """Yield (key, tensor) from a local safetensors checkpoint (single or sharded)."""
    from safetensors.torch import load_file

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = [os.path.basename(p) for p in glob.glob(os.path.join(path, "*.safetensors"))]
    for shard in shards:
        for key, tensor in load_file(os.path.join(path, shard)).items():
            yield key, tensor
