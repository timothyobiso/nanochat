"""
Tests for the HF-side router adapters (hf/routers.py, hf/patch_olmoe.py).

All CPU, all tiny. These pin down the load-bearing invariants of the port:
- build_router() reproduces nanochat's router init bit-for-bit given the seed.
- A patched OlmoeForCausalLM trains (finite loss, aux loss present, expert
  grads flow) and its gate holds no trainable parameters for fixed routers.
- save_router_olmoe -> load_router_olmoe round-trips logits exactly, and
  bare from_pretrained on a patched checkpoint is refused by load_router_olmoe
  guards (the silent-corruption path documented in hf/patch_olmoe.py).
- BlendedRouterGate: alpha=1 is bit-identical to the unpatched model,
  alpha=0/scale=1 matches FixedRouterGate, calibrate_scale matches stds.
- Router buffers stay fp32 under model-wide bf16 casts.
"""

import torch
import torch.nn as nn
from transformers import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

from nanochat.gpt import VSARouter, DirectFPERouter, CliffordRouter
from hf.routers import (
    ROUTER_TYPES,
    BlendedRouterGate,
    FixedRouterGate,
    build_router,
    calibrate_scale,
    permute_expert_ids,
)
from hf.patch_olmoe import (
    ROUTER_CONFIG_KEY,
    load_router_olmoe,
    patch_olmoe_routers,
    save_router_olmoe,
)

DIM, EXPERTS, TOPK = 64, 8, 2
FIXED_ROUTERS = [t for t in ROUTER_TYPES if t != "linear"]


def tiny_config(**overrides):
    kwargs = dict(
        vocab_size=256,
        hidden_size=DIM,
        intermediate_size=DIM,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=EXPERTS,
        num_experts_per_tok=TOPK,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        router_aux_loss_coef=0.01,
    )
    kwargs.update(overrides)
    return OlmoeConfig(**kwargs)


def tiny_model(seed=0, **config_overrides):
    torch.manual_seed(seed)
    return OlmoeForCausalLM(tiny_config(**config_overrides)).float().eval()


# ---------------- build_router parity with nanochat init ----------------

def test_build_router_matches_nanochat_init():
    """build_router must reproduce the nanochat init path bit-for-bit: same
    router class, same init_buffers(), same CPU RNG seed => identical buffers
    and identical fp32 logits."""
    nanochat_ctors = {
        "vsa_random": lambda: VSARouter(DIM, EXPERTS, mode="random"),
        "vsa_fpe": lambda: VSARouter(DIM, EXPERTS, mode="fpe"),
        "direct_fpe": lambda: DirectFPERouter(DIM, EXPERTS),
        "clifford_quat_fpe": lambda: CliffordRouter(DIM, EXPERTS, algebra="quaternion", key_mode="fpe"),
        "clifford_complex_random": lambda: CliffordRouter(DIM, EXPERTS, algebra="complex", key_mode="random"),
    }
    x = torch.randn(16, DIM)
    for router_type, ctor in nanochat_ctors.items():
        reference = ctor()
        torch.manual_seed(1234)
        reference.init_buffers()

        ours = build_router(router_type, DIM, EXPERTS, TOPK, seed=1234)

        ref_buffers = dict(reference.named_buffers())
        for name, buf in ours.named_buffers():
            assert torch.equal(buf, ref_buffers[name]), (
                f"{router_type}: buffer {name} differs from nanochat init with the same seed"
            )
        assert torch.equal(ours(x), reference(x)), f"{router_type}: logits differ"


def test_build_router_ignores_ambient_rng():
    """Buffer init must not depend on (or perturb) the caller's RNG stream."""
    torch.manual_seed(0)
    a = build_router("vsa_fpe", DIM, EXPERTS, TOPK, seed=7)
    torch.manual_seed(999)
    _ = torch.randn(100)  # scramble ambient RNG
    b = build_router("vsa_fpe", DIM, EXPERTS, TOPK, seed=7)
    assert torch.equal(a.memory, b.memory) and torch.equal(a.expert_ids, b.expert_ids)

    torch.manual_seed(42)
    before = torch.randn(4)
    torch.manual_seed(42)
    build_router("vsa_random", DIM, EXPERTS, TOPK, seed=7)
    after = torch.randn(4)
    assert torch.equal(before, after), "build_router leaked into the ambient RNG stream"


def test_build_router_linear_is_none():
    assert build_router("linear", DIM, EXPERTS, TOPK, seed=0) is None


def test_fixed_gate_bf16_input_close_to_fp32():
    """Routers upcast internally (FFT needs fp32); bf16 activations should give
    logits close to the fp32 ones."""
    gate = FixedRouterGate(build_router("vsa_fpe", DIM, EXPERTS, TOPK, seed=3))
    x = torch.randn(32, DIM)
    ref = gate(x)
    out = gate(x.bfloat16())
    assert out.dtype == torch.bfloat16
    assert torch.allclose(ref, out.float(), atol=0.15, rtol=0.05), (
        "bf16-input logits diverged from fp32 beyond bf16 rounding"
    )


# ---------------- patched tiny OLMoE: forward/backward ----------------

def test_patched_model_trains_all_router_types():
    """Every fixed router type: finite loss, non-zero aux loss, expert grads
    flow, and the gate itself contributes zero trainable parameters."""
    input_ids = torch.randint(0, 256, (2, 16))
    for router_type in FIXED_ROUTERS:
        model = tiny_model(seed=0)
        patched = patch_olmoe_routers(model, router_type, seed=5)
        assert patched == [0, 1], f"{router_type}: expected both layers patched"
        model.train()
        out = model(input_ids, labels=input_ids, output_router_logits=True)
        assert torch.isfinite(out.loss), f"{router_type}: non-finite loss"
        assert out.aux_loss is not None and torch.isfinite(out.aux_loss) and out.aux_loss != 0, (
            f"{router_type}: aux loss missing — the gate swap broke the "
            "output_router_logits collection path"
        )
        out.loss.backward()
        expert_w = model.model.layers[0].mlp.experts[0].gate_proj.weight
        assert expert_w.grad is not None and expert_w.grad.abs().sum() > 0, (
            f"{router_type}: no gradient reached the experts"
        )
        gate = model.model.layers[0].mlp.gate
        assert sum(p.numel() for p in gate.parameters() if p.requires_grad) == 0, (
            f"{router_type}: fixed gate should hold no trainable parameters"
        )


def test_patch_linear_is_noop():
    model = tiny_model(seed=0)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    patched = patch_olmoe_routers(model, "linear", seed=5)
    assert patched == []
    after = model.state_dict()
    assert before.keys() == after.keys()
    for k in before:
        assert torch.equal(before[k], after[k])
    assert getattr(model.config, ROUTER_CONFIG_KEY)["router_type"] == "linear"


def test_double_patch_refused():
    model = tiny_model(seed=0)
    patch_olmoe_routers(model, "vsa_fpe", seed=5)
    try:
        patch_olmoe_routers(model, "vsa_random", seed=6)
        assert False, "double patch should raise"
    except ValueError:
        pass


# ---------------- persistence round-trip ----------------

def test_save_load_round_trip(tmp_path):
    """save_router_olmoe -> load_router_olmoe must reproduce logits exactly
    (fp32 end to end), including the fixed_router config spec."""
    for router_type, blend in [("vsa_fpe", False), ("vsa_random", True)]:
        model = tiny_model(seed=1)
        patch_olmoe_routers(model, router_type, seed=11, blend=blend)
        if blend:
            # exercise non-default anneal state so the round-trip must carry it
            for layer in model.model.layers:
                layer.mlp.gate.set_alpha(0.25)
                layer.mlp.gate.set_scale(3.5)
        input_ids = torch.randint(0, 256, (1, 12))
        ref = model(input_ids).logits

        path = tmp_path / f"ckpt_{router_type}_{blend}"
        save_router_olmoe(model, str(path))
        loaded = load_router_olmoe(str(path), dtype=torch.float32, device="cpu").eval()

        spec = getattr(loaded.config, ROUTER_CONFIG_KEY)
        assert spec["router_type"] == router_type and spec["blend"] == blend
        assert torch.equal(loaded(input_ids).logits, ref), (
            f"{router_type} blend={blend}: round-tripped logits differ"
        )
        if blend:
            gate = loaded.model.layers[0].mlp.gate
            assert gate.alpha.item() == 0.25 and gate.scale.item() == 3.5


def test_load_router_olmoe_refuses_unpatched(tmp_path):
    model = tiny_model(seed=2)
    model.save_pretrained(tmp_path / "plain")
    try:
        load_router_olmoe(str(tmp_path / "plain"), dtype=torch.float32)
        assert False, "loading an unpatched checkpoint should raise"
    except ValueError as e:
        assert ROUTER_CONFIG_KEY in str(e)


def test_load_restores_fitted_buffers(tmp_path):
    """Buffers modified after init (e.g. least-squares-distilled memories) must
    come back from disk, not be regenerated from the seed."""
    model = tiny_model(seed=3)
    patch_olmoe_routers(model, "vsa_fpe", seed=13)
    gate = model.model.layers[0].mlp.gate
    gate.router.memory.copy_(torch.arange(DIM, dtype=torch.float32))
    save_router_olmoe(model, str(tmp_path / "fitted"))
    loaded = load_router_olmoe(str(tmp_path / "fitted"), dtype=torch.float32)
    assert torch.equal(
        loaded.model.layers[0].mlp.gate.router.memory,
        torch.arange(DIM, dtype=torch.float32),
    )


# ---------------- blended gate semantics ----------------

def test_blended_alpha_one_is_bit_identical():
    model = tiny_model(seed=4)
    input_ids = torch.randint(0, 256, (1, 12))
    ref = model(input_ids).logits
    patch_olmoe_routers(model, "vsa_fpe", seed=17, blend=True)  # alpha starts at 1
    assert torch.equal(model(input_ids).logits, ref), (
        "alpha=1 blended model must be bit-identical to the unpatched model"
    )


def test_blended_alpha_zero_matches_fixed():
    x = torch.randn(32, DIM)
    router = build_router("vsa_fpe", DIM, EXPERTS, TOPK, seed=19)
    fixed = FixedRouterGate(router)
    blended = BlendedRouterGate(nn.Linear(DIM, EXPERTS, bias=False), build_router("vsa_fpe", DIM, EXPERTS, TOPK, seed=19))
    blended.set_alpha(0.0)  # scale is 1.0 by default
    assert torch.equal(blended(x), fixed(x))


def test_calibrate_scale_std():
    torch.manual_seed(0)
    learned = nn.Linear(DIM, EXPERTS, bias=False)
    with torch.no_grad():
        learned.weight.mul_(10.0)  # exaggerate the scale mismatch
    gate = BlendedRouterGate(learned, build_router("vsa_random", DIM, EXPERTS, TOPK, seed=23))
    hidden = torch.randn(4096, DIM)
    s = calibrate_scale(gate, hidden, method="std")
    assert s == gate.scale.item()
    scaled_std = (s * gate.router(hidden)).std()
    learned_std = learned(hidden).std()
    assert abs(scaled_std / learned_std - 1.0) < 1e-4


# ---------------- Hungarian expert permutation ----------------

def test_permute_expert_ids_reindexes_logits():
    """permute_expert_ids must make router expert perm[e] answer for slot e, so
    the permuted router's logits equal the original's indexed by perm. Both B0
    (agreement) and B1 (apply_b0_perms) rely on this alignment; if it were
    transposed, healing would start from a worse match than no matching at all.
    """
    x = torch.randn(16, DIM)
    perm = torch.randperm(EXPERTS)
    for router_type in ("vsa_fpe", "vsa_random", "direct_fpe", "clifford_quat_fpe"):
        router = build_router(router_type, DIM, EXPERTS, TOPK, seed=11)
        before = router(x)
        permute_expert_ids(router, perm)
        after = router(x)
        assert torch.allclose(after, before[:, perm], atol=1e-6), router_type


def test_permute_expert_ids_is_a_noop_for_hash():
    """HashRouter has no expert identities to reorder — permuting must not
    raise, since run_phase_b/B0 pass perms uniformly across router types."""
    router = build_router("hash", DIM, EXPERTS, TOPK, seed=0)
    x = torch.randn(16, DIM)
    before = router(x)
    permute_expert_ids(router, torch.randperm(EXPERTS))
    assert torch.equal(router(x), before)


# ---------------- dtype robustness ----------------

def test_router_buffers_survive_bf16_cast():
    """model.to(bfloat16) must leave router buffers fp32 and bit-identical —
    otherwise routing silently shifts when the healing run casts the model."""
    model = tiny_model(seed=5)
    patch_olmoe_routers(model, "vsa_fpe", seed=29)
    memory_before = model.model.layers[0].mlp.gate.router.memory.clone()
    model.to(torch.bfloat16)
    gate = model.model.layers[0].mlp.gate
    assert gate.router.memory.dtype == torch.float32
    assert torch.equal(gate.router.memory, memory_before)
    # and the model still runs in bf16
    out = model(torch.randint(0, 256, (1, 8)))
    assert out.logits.dtype == torch.bfloat16
