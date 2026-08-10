# HF-Ecosystem Port of the Router Experiments: Plan & Rationale

*Status snapshot 2026-08-10: all code and CPU-verifiable gates are complete on the
`hf-experiments` branch (see [Milestones](#milestones--status)). Node-side runs
(data prep, Phase A matrix, B0/B1) are pending.*

This document records why this work exists, the reasoning behind every
non-obvious design decision, and the full experimental design. The condensed
run matrices also live in the 2026-08-09 `dev/LOG.md` entry; day-to-day
commands live in `CLAUDE.md`.

---

## 1. Motivation

The nanochat fork demonstrated that parameter-free routers (VSA/HRR holographic,
FPE, Clifford) are competitive with learned linear routers in a small
from-scratch MoE. Two problems block the next step for the paper:

1. **Credibility / reproducibility.** The original result lives entirely inside
   nanochat's bespoke stack: Muon optimizer, topk-then-softmax gating, custom
   training loop, custom tokenizer. Reviewers and users live in the HuggingFace
   stack. Rerunning the core comparison with stock `transformers` models, AdamW,
   and host gating conventions shows the result is a property of the routers,
   not an artifact of nanochat's exotic choices.
2. **Scale without a pretraining budget.** We cannot pretrain a multi-billion-
   parameter MoE. But we *can* swap the fixed routers into a pretrained open MoE
   and "heal" it with a short continued-pretraining run — orders of magnitude
   cheaper than pretraining, and it produces a "works at real scale on a real
   model" claim.

## 2. The core insight that shapes everything

Every non-hash router in this repo is a **fixed linear map of the hidden
state**. HRR unbinding is circular correlation — linear in `x`; quaternion
multiplication is bilinear with one side frozen; `DirectFPERouter` is literally
`x @ keys.T`. The buffers never train, so each router is `scores = x @ W_fixed`
for a structured random `W_fixed`.

That means the nanochat result is really a statement about **representations
adapting to a frozen projection during training** — the router doesn't learn;
the residual stream learns to place tokens so the fixed partition becomes a
good one.

A pretrained MoE is the opposite regime: its experts are co-adapted to their
learned gate, and its representations are (nearly) frozen. Swapping in a random
fixed projection reassigns tokens to experts that never specialized for them,
with no mechanism for recovery at inference time. **Zero-shot drop-in
replacement is therefore expected to be catastrophic, and is measured only as a
motivating baseline (Phase B0).** Any real transfer requires training-time
adaptation: from scratch (Phase A) or via an annealed swap plus healing
(Phase B1).

## 3. Paths considered

| Path | Verdict | Why |
|---|---|---|
| Replicate from scratch in the HF stack | **Phase A** | Cheap, directly answers the "bespoke stack" objection, and shakes out all integration machinery before touching the 7B model. |
| Router-swap + healing on a pretrained open MoE | **Phase B** | The literal version of the scale claim. Made survivable by annealed logit blending, per-layer scale calibration, and Hungarian expert matching (below). |
| Zero-training diagnostics (agreement, swap-ppl, gate distillation) | **Phase B0** | Nearly free, quantifies the co-adaptation gap, and the distillation is a standalone result about how much of a *trained* gate the holographic structure can express. |
| Sparse upcycling of a dense model with a VSA router from step 0 | Deferred (stretch / possible Phase C) | Scientifically the cleanest large-scale test (no co-adaptation to break — experts specialize *under* the fixed router, exactly the regime where the nanochat result held), but the OLMoE path was chosen as the flagship. |
| Scaling-trend argument inside nanochat (extend depth series) | Fallback, not part of this plan | Doesn't address the HF-stack objection. |

## 4. Decisions and rationale

**Code home: an additive `hf/` package inside this repo** (user decision).
Keeps the routers next to their reference implementation — `hf/routers.py`
*imports* `VSARouter`/`DirectFPERouter`/`CliffordRouter`/`HashRouter` from
`nanochat/gpt.py`, never copies them — and keeps the paper's history in one
place. Bit-exact parity between the HF-side init and nanochat's is enforced by
tests, not by convention.

**Compute: one 8xH100 node** (user decision). The local dev machine is an
Apple-Silicon Mac: CPU-only tiny-scale development (MPS lacks the `torch.fft`
coverage the VSA routers need). Everything in the plan is sized to
single-node-days.

**Phase A uses host conventions, not nanochat's** (user decision):
random-init `OlmoeForCausalLM`, AdamW β=(0.9, 0.95), softmax-then-topk with
`norm_topk_prob=False`, MoE in every layer, no router-logit noise. Rationale:
the results then transfer directly to the Phase B OLMoE work (same
architecture, same gating semantics), and the paper's claim becomes "works in
the standard stack" rather than "works under our conventions ported into HF".
Documented deviations from the original runs: nanochat used Muon,
topk-then-softmax (gate weights always summed to 1), MoE every *other* layer,
and Gaussian router-logit noise (std 0.01) during training.

**transformers pinned `>=4.57.3,<5`.** In 4.x — verified against the installed
4.57.3 source — `OlmoeSparseMoeBlock.gate` is `nn.Linear(hidden, E, bias=False)`,
the block calls `self.gate(hidden_states)` on flattened `(B*T, d)` states and
returns `(hidden, router_logits)`, and the aux loss is computed from the
collected logits independent of the gate's class. Replacing the `.gate` module
instance is therefore the *entire* integration seam. In 5.x the contract flips:
the gate becomes an `OlmoeTopKRouter` returning a 3-tuple, and aux-loss
collection uses output-recorder hooks matched via
`isinstance(module, OlmoeTopKRouter)` — a plain adapter would *silently disable
the aux loss*. The `<5` cap turns that silent failure into a loud one.
Migration notes live in `hf/patch_olmoe.py`'s docstring.

**Patched checkpoints load only via `load_router_olmoe`.** Bare
`from_pretrained` on a patched checkpoint builds an unpatched skeleton,
randomly initializes the (missing) `gate.weight`, drops the router buffers, and
returns a silently broken model. The supported loader is two-pass: standard
load of the trunk with the gate-scope key delta captured and asserted to be
exactly the router seam, then patch-from-spec and restore all gate tensors from
disk. Restoring buffers from disk (rather than regenerating from the seed)
matters because diagnostics can produce *fitted* buffers (distilled memories).

**Router buffers are pinned to fp32.** `model.to(bfloat16)` in the healing run
would otherwise cast the fixed buffers and silently perturb routing. The gate
adapters override `_apply` to undo dtype downcasts of buffers (device moves
still apply), losslessly.

**Data: pre-tokenized uint16 shards, not streaming.** One-time tokenization of
FineWeb-Edu (`sample-100BT`) with the OLMoE tokenizer into flat shards. The
loader tiles the concatenated stream with a 1-token overlap and its entire
state is one integer step — resume is exact by construction, and 8-GPU read
throughput is trivially high. Streaming was rejected for training because
approximate resume and network variance are exactly the kind of confound a
controlled comparison doesn't need.

**Phase B checkpoint: `allenai/OLMoE-1B-7B-0924`, base, revision-pinned.**
It is the paper-documented artifact (arXiv 2409.02060) with published benchmark
numbers for the sanity gate, and its pretraining mix (OLMoE-mix-0924) is
public. Instruct variants add SFT/DPO confounds to a continued-pretraining
experiment; the 0125 refresh has weaker documentation of its recipe.

**Healing data: the same FineWeb-Edu shards for ALL conditions, including the
control.** The paper's claim is the swap-vs-control *contrast*; with a shared
corpus, distribution shift from OLMoE's original mix is absorbed by the
control. Distribution-matching with OLMoE-mix-0924 is a nice-to-have robustness
rerun (control + best condition only), not a requirement.

**B1 parallelism: DDP + ZeRO-1-sharded Adam + gradient checkpointing.** Plain
DDP does not fit: 6.9B params with unsharded fp32 Adam moments is ~55 GB before
params and grads. Default configuration keeps fp32 params + bf16 autocast
(~63 GB/GPU + activations on 8 GPUs); `--param-dtype bfloat16` is the documented
~35 GB fallback at some optimizer-precision risk; FSDP is the fallback beyond
that. The pilot re-baselines all of this before the full runs.

**Why blending, calibration, and matching exist (B1 survivability):**
- *Annealed blending* — `alpha·learned + (1−alpha)·scale·fixed`, alpha 1→0 —
  migrates assignments gradually instead of cliff-dropping the model onto a
  router its experts never saw.
- *Per-layer scale calibration* (std-matching against the learned gate) — the
  fixed routers' logit magnitudes differ from the learned gate's, which under
  softmax-then-topk silently changes the effective temperature and the gate
  mass applied to expert outputs.
- *Hungarian expert matching* (from B0) — the fixed router's expert identities
  are arbitrary, so relabeling them to best match the learned gate's partition
  minimizes the initial disagreement the healing run has to repair.

## 5. Phase A — from-scratch replication

Original matrix being replicated in spirit (reconstructed from
`old_speedruns/*.sh` and the committed `paper_figures/` PDFs — it was never
logged): routers {linear, hash, vsa_random, vsa_fpe, direct_fpe} × depths
{4, 8, 10, 12, 16}, 8 experts top-2, aux coeff 0.01, Chinchilla 20:1 on total
params. Reference points: d8 router latency fpe/vsa 0.84 ms vs hash 0.12 ms;
hash utilization exactly uniform; other routers peak 0.19–0.24 worst-layer.

HF-side training: seq 2048, global batch 524,288 tokens (matches original),
AdamW wd 0.1 clip 1.0, cosine to 10% of peak with 1% warmup, bf16 autocast,
vocab 50,304 (OLMoE tokenizer padded), per-expert `intermediate_size = hidden`.

| Size | d | L | Total | Active | Tokens (20:1) | Peak LR | Est. wall-clock |
|------|-----|----|-------|--------|---------------|---------|-----------------|
| S | 512 | 8 | ~110M | ~73M | 2.2B | 6e-4 | 0.5–1h |
| M | 768 | 12 | ~275M | ~148M | 5.5B | 5e-4 | 1–2h |
| L | 1024 | 16 | ~573M | ~271M | 11.5B | 4e-4 | 4–8h |

Runs: 5 routers × (S×2 seeds + M×2 seeds + L×1) = **25 runs ≈ 2.5 node-days**
at a 10–15% MFU assumption for HF's eager per-expert loop (re-baselined at the
pilot: size S × {linear, vsa_fpe} before committing the matrix). Add-ons:
aux-coeff-0 ablation (2×S), `norm_topk_prob=True` probe (S), `clifford_quat_fpe`
at M (stretch). Telemetry every 250 steps: val bits/byte, per-layer expert
token-fractions (Gini/entropy) on a fixed probe batch, and **gate mass** — see
the `norm_topk_prob` risk below.

## 6. Phase B0 — zero-training diagnostics (1 GPU)

Gate first: lm-eval on the *unmodified* checkpoint must reproduce the published
OLMoE-0924 numbers within ~1 point before any swap experiment runs.

Then, streaming over ~4M calibration tokens:
- **Agreement**: per-layer top-1 and top-8-Jaccard agreement between the
  learned gate and each fixed router, after Hungarian matching on a top-1
  confusion matrix built from the first ~512k tokens.
- **Seed search**: ~512 FPE base seeds scored by mean top-1 agreement on a
  stored hidden-state subsample; best seeds feed B1.
- **Hard-swap perplexity**: all layers at once, and one layer at a time
  (a layer-sensitivity ranking).
- **Distillation**: least-squares fit of each learned gate into VSA structure,
  exploiting the linearity `scores_e(x) = memory · bind(x, ids_e)`:
  an isotropic closed-form per-FFT-frequency fit of the memory alone (d free
  values vs the gate's E·d — E-fold compression), and a param-matched
  alternating ridge fit of memory + ids on captured hidden states. Reported:
  logit MSE, agreement, and optionally perplexity with fitted routers
  installed. (Unit tests pin the math: the E=1 fit is exact; the param-matched
  fit captures >95% of gate variance held-out.)

## 7. Phase B1 — healing (8xH100)

Four conditions, identical schedule, **5B tokens each** (~9h/condition at a
~20% MFU assumption; ~1.5–2 node-days total):

| Condition | Gate | Alpha |
|---|---|---|
| blend (vsa_fpe) | BlendedRouterGate | linear 1→0 over first 750 steps (30%) |
| blend (vsa_random) | BlendedRouterGate | same |
| hard_swap (vsa_fpe) | BlendedRouterGate | pinned 0 from step 0 (no-anneal ablation) |
| control | untouched learned gate | — |

Global batch 2M tokens (512 × seq 4096) → 2,500 steps; LR cosine 5e-5 → 5e-6,
100-step warmup; learned gates stay trainable during the anneal (default
continue-training semantics — a deliberate choice, noted). Before training:
per-layer scale calibration on 2M tokens; optionally Hungarian perms + best
seed from the B0 report.

Cadence: val bits/byte every 50 steps; routing-drift (top-k kept vs the
pre-swap gate on a fixed probe batch) and utilization every 100; checkpoints
every 250; lm-eval (MMLU 5-shot, HellaSwag, ARC-e/c, PIQA, WinoGrande, BoolQ)
at 0 / 1B / 2.5B / 5B tokens. **Headline metric: tokens-to-recover-control-
perplexity.** Extend the winning condition to 10B tokens if budget allows.

## 8. Implementation map

```
hf/routers.py          build_router (deterministic seeded init; per-layer seed =
                       base + 1000·layer), FixedRouterGate, BlendedRouterGate,
                       calibrate_scale, permute_expert_ids
hf/patch_olmoe.py      gate instance patching; save/load with strict key-delta
                       verification; the transformers 5.x migration notes
hf/prepare_data.py     FineWeb-Edu -> uint16 shards (OLMoE tokenizer)
hf/data.py             exactly-resumable distributed shard loader
hf/train_olmoe.py      Phase A single-file DDP trainer
hf/diagnose_router.py  B0: agreement / seeds / swap / distill
hf/heal_olmoe.py       B1: calibration + anneal + drift telemetry
hf/eval_lm.py          lm-eval wrapper, patched-checkpoint-aware
hf/analysis.py         figures mirroring scripts/paper_analysis.py
hf/run_phase_a.sh      node driver: setup/data/pilot/matrix/ablations/figures
hf/run_phase_b.sh      node driver: baseline gate/B0/heal (pulls best seed +
                       perms from the B0 report)/milestone evals/figures
tests/test_hf_*.py     24 CPU tests: bit-exact router parity with nanochat init,
                       patched fwd/bwd for every router type, save/load
                       round-trips, blend semantics, loader resume, fit math
```

## 9. Milestones & status

| | Milestone | Gate | Status (2026-08-10) |
|---|---|---|---|
| M0 | Env + ground truth | 4.57.3 gate contract verified in installed source; deps resolve | ✅ done |
| M1 | Routers + patching + tests | pytest green on CPU | ✅ done (14 tests) |
| M2 | Data pipeline | manifest/loader determinism | ✅ code + CPU tests; node shard build pending |
| M3 | Phase A trainer + pilot | CPU smoke; node pilot S×{linear, vsa_fpe}; measured MFU | ✅ code + CPU smoke (bit-exact resume); pilot pending |
| M4 | Phase A matrix + figures | 25 runs; figures render | pending (node) |
| M5 | B0 diagnostics | baseline lm-eval within ~1pt FIRST; then all modes | code done + tiny-model smoke; runs pending |
| M6 | B1 healing | no divergence post-anneal; checkpoints reload | code done + tiny-model smoke incl. reload; runs pending |
| M7 | Analysis + write-up | healing figures; LOG.md results entry | pending |

## 10. Risks being tracked

- **FPE key crowding at E=64.** `make_fpe_keys` spaces fractional powers
  p = i/63, so adjacent keys are highly correlated → noisy top-8 boundaries.
  Mitigations: the B0 seed search is primary; vsa_random is the expected
  stronger healer if vsa_fpe agreement is near chance (the matrix covers it).
- **`norm_topk_prob=False` × fixed routers.** Low-variance fixed-router logits
  → near-uniform softmax → selected gate weights sum well below 1 → the MoE
  branch is systematically down-scaled vs the learned baseline. This
  interaction did not exist under nanochat's topk-then-softmax (weights always
  summed to 1). Tracked via gate-mass telemetry; probed via the
  `norm_topk_prob=True` ablation; handled in B1 by scale calibration.
- **Throughput unknowns.** HF's eager per-expert loop is kernel-launch-bound at
  small scale; all wall-clock numbers assume 10–20% MFU and are re-baselined at
  the M3/M6 pilots. Contingency order: drop L to 15:1 tokens, cut the B1
  hard-swap condition — before cutting seeds.
- **Silent checkpoint corruption.** Only `load_router_olmoe` may load patched
  checkpoints (enforced by tests and by `hf/eval_lm.py`).
- **Aux loss with fixed routers** still shapes hidden states (gradients flow
  into `x` through the softmax term); kept at 0.01 for host fidelity, with the
  coeff-0 ablation quantifying the effect.

## 11. References

- OLMoE: Muennighoff et al., *OLMoE: Open Mixture-of-Experts Language Models*,
  arXiv:2409.02060 (checkpoint `allenai/OLMoE-1B-7B-0924`, data mix
  `allenai/OLMoE-mix-0924`).
- Sparse upcycling (deferred Phase C candidate): Komatsuzaki et al.,
  arXiv:2212.05055.
- FineWeb-Edu: `HuggingFaceFW/fineweb-edu` (`sample-100BT`).
- Hash routing baseline lineage: Roller et al., *Hash Layers for Large Sparse
  Models*, arXiv:2106.04426.
