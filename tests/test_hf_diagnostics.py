"""Math tests for hf/diagnose_router.py: Hungarian alignment, Jaccard,
the VSA distillation fits, and B0 seed selection."""

import json

import torch

from nanochat.gpt import hrr_unbind
from hf.diagnose_router import (
    alt_fit,
    best_seed_for,
    distill_metrics,
    hungarian_perm,
    iso_fit_memory,
    topk_jaccard,
)
from hf.routers import build_router

DIM, EXPERTS = 64, 8


def write_report(tmp_path, report):
    path = tmp_path / "b0_report.json"
    path.write_text(json.dumps(report))
    return str(path)


def test_best_seed_only_applies_to_the_router_it_was_searched_under(tmp_path):
    """The seed search scores one router's key construction. Carrying its winner
    to a different router would dress an arbitrary seed up as a tuned one, so
    anything but an exact router match falls back to the default."""
    path = write_report(tmp_path, {
        "seed_search": {"router": "vsa_fpe", "top16": [{"seed": 314, "top1_mean": 0.4}]}
    })
    assert best_seed_for(path, "vsa_fpe") == 314
    assert best_seed_for(path, "vsa_random") == 0


def test_best_seed_falls_back_on_missing_or_partial_reports(tmp_path):
    """A report from --mode agreement (or no report at all) must not crash the
    healing launcher — hf/run_phase_b.sh calls this before every condition."""
    assert best_seed_for(str(tmp_path / "nonexistent.json"), "vsa_fpe") == 0
    assert best_seed_for(write_report(tmp_path, {"agreement": {}}), "vsa_fpe") == 0
    assert best_seed_for(write_report(tmp_path, {"seed_search": {"router": "vsa_fpe"}}), "vsa_fpe") == 0


def test_hungarian_perm_recovers_relabeling():
    """If the router is the learned gate with shuffled expert slots, the
    confusion matrix is (near-)diagonal under the shuffle and Hungarian must
    recover it exactly."""
    torch.manual_seed(0)
    true_perm = torch.randperm(EXPERTS)
    learned_top1 = torch.randint(0, EXPERTS, (5000,))
    router_top1 = torch.empty_like(learned_top1)
    inverse = torch.argsort(true_perm)
    router_top1 = inverse[learned_top1]  # router slot b answers for learned expert true_perm[b]... aligned via perm
    confusion = torch.zeros(EXPERTS, EXPERTS)
    for a, b in zip(learned_top1.tolist(), router_top1.tolist()):
        confusion[a, b] += 1
    perm = hungarian_perm(confusion)
    # aligned router logits are logits[:, perm]; top-1 agreement must be perfect
    assert torch.equal(perm[learned_top1], router_top1)


def test_topk_jaccard_bounds():
    logits = torch.randn(100, EXPERTS)
    assert topk_jaccard(logits, logits, 2) == 100.0  # identical => Jaccard 1 per token
    disjoint_a = torch.zeros(1, 4)
    disjoint_b = torch.zeros(1, 4)
    disjoint_a[0, :2] = 1  # top-2 = {0,1}
    disjoint_b[0, 2:] = 1  # top-2 = {2,3}
    assert topk_jaccard(disjoint_a, disjoint_b, 2) == 0.0


def test_iso_fit_exact_for_single_expert():
    """With E=1 the per-frequency system is determined: the holographic
    structure can represent ANY single linear functional exactly."""
    torch.manual_seed(1)
    W = torch.randn(1, DIM)
    ids = torch.nn.functional.normalize(torch.randn(1, DIM), dim=-1)
    memory = iso_fit_memory(W, ids)
    x = torch.randn(64, DIM)
    pred = hrr_unbind(memory.unsqueeze(0), x) @ ids.T
    assert torch.allclose(pred, x @ W.T, atol=1e-3), "E=1 iso fit should be (numerically) exact"


def test_alt_fit_improves_on_iso():
    """The data-weighted alternating fit has E*d + d free values (param-matched
    to the gate) and must beat the d-value iso fit on held-out logit MSE."""
    torch.manual_seed(2)
    W = torch.randn(EXPERTS, DIM) / DIM**0.5
    router = build_router("vsa_fpe", DIM, EXPERTS, 2, seed=0)
    X = torch.randn(4096, DIM)
    X_fit, X_test = X[:3600], X[3600:]

    m_iso = iso_fit_memory(W, router.expert_ids)
    m_alt, ids_alt = alt_fit(W, router.expert_ids, m_iso, X_fit, ridge=1e-4, rounds=2)

    iso = distill_metrics(m_iso, router.expert_ids, W, X_test, 2)
    alt = distill_metrics(m_alt, ids_alt, W, X_test, 2)
    assert alt["logit_mse"] < iso["logit_mse"], (iso, alt)
    assert alt["logit_mse"] < 0.05 * iso["logit_var"], (
        "param-matched alternating fit should capture most of the gate's variance"
    )
    assert alt["top1"] > iso["top1"]
