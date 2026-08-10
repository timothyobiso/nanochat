"""
Router adapters for HuggingFace OLMoE models.

Reuses the nanochat router implementations (nanochat/gpt.py) verbatim — the
routers there are self-contained nn.Modules mapping x: (N, d) -> logits (N, E)
with fixed buffers filled by init_buffers(). This module wraps them so they can
replace the learned `gate` (nn.Linear) inside a transformers-4.x
OlmoeSparseMoeBlock, whose forward calls `self.gate(hidden_states)` on the
flattened (B*T, d) hidden states and expects (B*T, E) logits back.

Two invariants the adapters maintain:
- Router buffers stay float32 even when the surrounding model is cast to bf16
  (the HRR unbind runs FFT in float32 anyway; keeping the buffers fp32 makes
  logits identical across model dtypes and save/load round-trips exact).
- Buffer init is deterministic and independent of the caller's RNG state:
  build_router() forks the CPU RNG and seeds it explicitly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import (
    VSARouter,
    DirectFPERouter,
    CliffordRouter,
    HashRouter,
)

ROUTER_TYPES = (
    "linear",
    "hash",
    "vsa_random",
    "vsa_fpe",
    "direct_fpe",
    "clifford_quat_random",
    "clifford_quat_fpe",
    "clifford_complex_random",
    "clifford_complex_fpe",
)


def build_router(router_type, dim, num_experts, num_experts_per_tok, seed):
    """Construct a fixed router with deterministically-seeded buffers on CPU.

    Returns None for 'linear' (keep the host model's learned gate). Buffers are
    always initialized on CPU with the CPU RNG so the same seed gives the same
    router on any machine; move the module to the target device afterwards.
    """
    if router_type not in ROUTER_TYPES:
        raise ValueError(f"Unknown router_type: {router_type} (choose from {ROUTER_TYPES})")
    if router_type == "linear":
        return None
    if router_type == "hash":
        return HashRouter(num_experts, num_experts_per_tok)
    if router_type in ("vsa_random", "vsa_fpe"):
        router = VSARouter(dim, num_experts, mode=router_type.removeprefix("vsa_"))
    elif router_type == "direct_fpe":
        router = DirectFPERouter(dim, num_experts)
    else:  # clifford_{quat,complex}_{random,fpe}
        _, algebra, key_mode = router_type.split("_")
        algebra = {"quat": "quaternion", "complex": "complex"}[algebra]
        router = CliffordRouter(dim, num_experts, algebra=algebra, key_mode=key_mode)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        router.init_buffers()
    return router


def permute_expert_ids(router, perm):
    """Reorder a router's expert identities in place so that router expert
    perm[e] answers for host expert slot e (perm as produced by Hungarian
    matching against a learned gate). The bundled VSA memory is a set — only
    the row order of the identity/key buffers defines which score lands in
    which expert slot, so no re-binding is needed. No-op for HashRouter."""
    perm = torch.as_tensor(perm, dtype=torch.long)
    if isinstance(router, HashRouter):
        return
    if isinstance(router, (VSARouter, CliffordRouter)):
        router.expert_ids.copy_(router.expert_ids[perm].clone())
        router.expert_keys.copy_(router.expert_keys[perm].clone())
    elif isinstance(router, DirectFPERouter):
        router.expert_keys.copy_(router.expert_keys[perm].clone())
    else:
        raise TypeError(f"Cannot permute router of type {type(router).__name__}")


class _FP32BufferModule(nn.Module):
    """Base that pins every buffer under this module to float32.

    model.to(torch.bfloat16) / .half() route through nn.Module._apply and would
    otherwise cast the router's fixed buffers (and alpha/scale), silently
    perturbing routing. Device moves are honored; dtype downcasts of buffers
    are undone using the pre-cast fp32 tensors, so no precision is ever lost.
    """

    def _apply(self, fn, recurse=True):
        old_buffers = {name: buf for name, buf in self.named_buffers()}
        super()._apply(fn, recurse)
        for name, new_buf in list(self.named_buffers()):
            old = old_buffers.get(name)
            if old is None or old.dtype != torch.float32:
                continue
            if new_buf.dtype != torch.float32:
                mod = self
                *path, leaf = name.split(".")
                for p in path:
                    mod = getattr(mod, p)
                mod._buffers[leaf] = old.to(device=new_buf.device, dtype=torch.float32)
        return self


class FixedRouterGate(_FP32BufferModule):
    """Drop-in replacement for OlmoeSparseMoeBlock.gate holding a fixed router.

    forward: (N, d) -> (N, E) logits, dtype-preserving. No learnable params.
    """

    def __init__(self, router, spec=None):
        super().__init__()
        self.router = router
        self.spec = dict(spec or {})

    @torch.compiler.disable
    def forward(self, hidden_states):
        return self.router(hidden_states)


class BlendedRouterGate(_FP32BufferModule):
    """alpha * learned_logits + (1 - alpha) * scale * router_logits.

    Used for annealed router swaps on pretrained models: alpha starts at 1
    (pure learned gate, bit-identical to the unpatched model), anneals to 0
    (pure fixed router, scaled to match the learned logit magnitude). The
    learned gate stays a trainable nn.Linear; alpha and scale are persistent
    fp32 buffers so checkpoints carry the anneal state.
    """

    def __init__(self, learned_gate, router, spec=None):
        super().__init__()
        self.learned_gate = learned_gate
        self.router = router
        self.spec = dict(spec or {})
        self.register_buffer("alpha", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))

    def set_alpha(self, a):
        self.alpha.fill_(float(a))

    def set_scale(self, s):
        self.scale.fill_(float(s))

    @torch.compiler.disable
    def forward(self, hidden_states):
        alpha = self.alpha.item()
        if alpha >= 1.0:
            # Bit-identical to the unpatched model (and skips the router FFT).
            return self.learned_gate(hidden_states)
        fixed = self.scale * self.router(hidden_states).float()
        if alpha <= 0.0:
            return fixed.to(hidden_states.dtype)
        learned = self.learned_gate(hidden_states).float()
        return (alpha * learned + (1.0 - alpha) * fixed).to(hidden_states.dtype)


@torch.no_grad()
def calibrate_scale(gate, hidden, method="std"):
    """Fit BlendedRouterGate.scale so fixed-router logits match the learned gate.

    hidden: (N, d) sample of real MoE-block inputs for this layer.
    'std'  — ratio of logit standard deviations (primary, closed form).
    'kl'   — 1-param temperature minimizing KL(softmax(learned) || softmax(s*fixed))
             over a log-spaced grid (reported in diagnostics, not used to train).
    Sets gate.scale in place and returns the value.
    """
    learned = gate.learned_gate(hidden).float()
    fixed = gate.router(hidden).float()
    if method == "std":
        s = (learned.std() / fixed.std().clamp(min=1e-12)).item()
    elif method == "kl":
        p = F.softmax(learned, dim=-1)
        grid = torch.logspace(-3, 3, steps=241, device=hidden.device)
        best_s, best_kl = 1.0, float("inf")
        for s_cand in grid.tolist():
            logq = F.log_softmax(s_cand * fixed, dim=-1)
            kl = F.kl_div(logq, p, reduction="batchmean").item()
            if kl < best_kl:
                best_kl, best_s = kl, s_cand
        s = best_s
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    gate.set_scale(s)
    return s
