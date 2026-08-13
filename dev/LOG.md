# Experiment Log

A running summary documenting some experiments and findings. Started ~Jan 7 2026.

---

## 2026-08-13: SLURM runner for the HF experiments; 8×48GB re-baseline

The HF pipeline is now submittable as dependency-chained sbatch jobs, each under
a 24-hour cap: `bash hf/slurm/e2e.sh` (`--check` / `--dry-run` / `--only STAGE`).
Reference: `hf/slurm/README.md`.

**Hardware changed.** Target is `student-gpu-003` / `student-gpu-004`, 8×48GB
each, not the 8×H100 80GB the plan was costed for. Two consequences:

- B1 must run `--param-dtype bfloat16` (~35 GB/GPU). The fp32-param default
  needs ~63 GB/GPU and does not fit. (The 2026-08-09 entry below claimed bf16
  params were already the default — they were not; the code default is fp32,
  correct for 80GB hardware, and the SLURM job passes bf16 explicitly.)
- Wall-clock roughly triples. Phase A ≈ 185 node-hours, B1 ≈ 108, so ~a week
  across two nodes. Phase A's L runs and every B1 condition exceed 24h and are
  split across chained segments.

**Four blocking bugs found while wiring this up**, all fixed:

1. `heal_olmoe.py` had no `--resume` — it wrote `trainer_state.pt` but always
   looped from step 0. Every 24h segment would have restarted a 5B-token run.
   Now mirrors `train_olmoe.py`, and carries the routing-drift reference in the
   checkpoint (recomputing it post-resume would re-baseline against an
   already-drifted learned gate). `tests/test_hf_heal_resume.py` proves the
   continuation is bit-exact by killing a tiny run mid-way and resuming it.
2. `calibrate_all_scales` accumulated hidden states **on device** until
   `--calib-tokens` (2M): ~262 GB per rank at OLMoE scale. It would have OOM'd
   on any GPU. Now a bounded CPU sample; `--calib-tokens` defaults to 65536,
   which is ample for a std ratio.
3. Both drivers used `metrics.jsonl` existence as "already done" — a run killed
   at 5% has one, so a resubmission silently left truncated conditions in the
   results matrix. Completion is now a `DONE` file written only after the last
   step (`hf/checkpoints.py`).
4. Phase A's three ablations derived their run names as `{size}_{router}_s{seed}`
   with no suffix, so all three collided with matrix runs and were skipped by
   the same predicate. The matrix moved to `hf/runs.sh` with explicit names.

**Other changes.** Checkpoint retention (newest kept whole, milestones reduced to
model-only, rest deleted) takes B1 from ~3.3 TB to ~350 GB. B0 and lm-eval now
skip work already present in their output files, so they chain too. The B0 seed
search winner is only adopted for the router it was searched under
(`best_seed_for`) — it was being handed to `vsa_random`, whose keys have nothing
to do with FPE spacing. `DATA_TOKENS` dropped 30B → 15B: Phase A's largest run
needs 11.5B and B1 rereads the same shards, and tokenization is the one stage
that cannot be resumed.

**Kept as-is, deliberately.** Phase A still installs a bare `FixedRouterGate`
with `norm_topk_prob=False`, so fixed-router gate mass sits near `top_k/E`
(~0.25) against ~1 for the learned gate — the MoE branch is ~4× down-scaled in
VSA runs. Rather than deviate from the committed host-fidelity design, the
`norm_topk_prob=True` probe was widened from one run to `linear` + `vsa_fpe` at
size S so the effect is measurable. Remaining known gaps (no step-0 lm-eval,
hash routing depending on batch shape, FPE expert 0 being the binding identity)
are listed in `hf/slurm/README.md`.

---

## 2026-08-09: HF-ecosystem port of the router experiments (plan + matrices)

The original MoE router runs (never logged here — matrix reconstructed from
`old_speedruns/*.sh` and the committed `paper_figures/` PDFs) were:
routers {linear, hash, vsa_random, vsa_fpe, direct_fpe} × depths {4, 8, 10, 12, 16},
8 experts top-2, `moe_layer_freq=2`, aux coeff 0.01, Chinchilla 20:1 on total params,
Muon+AdamW, topk-then-softmax gating, router-logit noise std 0.01 during training.
Reference numbers: d8 router latency fpe/vsa 0.84 ms vs hash 0.12 ms; hash utilization
exactly uniform; other routers peak 0.19–0.24 token fraction in the worst layer.

New work lives in `hf/` (branch `hf-experiments`): rerun the comparison in the
standard HF stack, then swap routers into pretrained OLMoE. Full plan and
design rationale in `docs/HF_PORT_PLAN.md`; summary:

**Phase A — from-scratch replication, host conventions.** Random-init
`OlmoeForCausalLM` (transformers 4.57.3, pinned `<5`: the gate contract flips in 5.x),
MoE every layer, 8 experts top-2, softmax-then-topk, `norm_topk_prob=False`, no router
noise, AdamW β=(0.9,0.95) wd 0.1 clip 1.0, cosine→10% peak, 1% warmup, seq 2048,
global batch 524,288 tokens, FineWeb-Edu tokenized with the OLMoE tokenizer.

| Size | d | L | Total | Active | Tokens (20:1) | Peak LR |
|------|-----|----|-------|--------|---------------|---------|
| S | 512 | 8 | ~110M | ~73M | 2.2B | 6e-4 |
| M | 768 | 12 | ~275M | ~148M | 5.5B | 5e-4 |
| L | 1024 | 16 | ~573M | ~271M | 11.5B | 4e-4 |

Runs: 5 routers {linear, hash, vsa_random, vsa_fpe, direct_fpe} × (S×2 seeds, M×2, L×1)
= 25 + ablations (aux-coeff-0 at S; `norm_topk_prob=True` probe at S; clifford_quat_fpe
at M as stretch). ~2.5 node-days at 10–15% MFU assumption, re-baselined after the pilot.

**Phase B0 — OLMoE-1B-7B-0924 diagnostics (no training).** Baseline lm-eval sanity gate
first (must match published numbers within ~1 pt). Then per-layer routing agreement
(top-1 / top-8 Jaccard) after Hungarian matching on the top-1 confusion matrix; FPE seed
search (~512 seeds); hard-swap perplexity (all layers + one-at-a-time); least-squares
distillation of each learned gate into VSA structure (closed-form memory-only fit = 64×
compression; alternating memory+ids fit = param-matched).

**Phase B1 — healing, 4 conditions × 5B tokens** (vsa_fpe blend+heal, vsa_random
blend+heal, control with untouched gate, vsa_fpe hard-swap): global batch 2M tokens
(512×4096) → 2,500 steps, LR cosine 5e-5→5e-6, alpha 1→0 linear over first 750 steps,
per-layer scale calibration (std-match) before training, DDP+ZeRO-1 with sharded Adam
states / gradient checkpointing. (Param dtype: the code defaults to fp32, ~63 GB/GPU;
the 8×48GB runs pass `--param-dtype bfloat16` — see the 2026-08-13 entry.) Headline metric:
tokens-to-recover-control-ppl. lm-eval (MMLU 5-shot, HellaSwag, ARC-e/c, PIQA,
WinoGrande, BoolQ) at 0/1B/2.5B/5B.

**Known risks tracked:** FPE key crowding at E=64 (adjacent fractional powers highly
correlated — seed search is the mitigation, vsa_random the fallback); fixed routers ×
`norm_topk_prob=False` under-weight the MoE branch (gate-mass telemetry + norm probe);
patched checkpoints must load via `hf.patch_olmoe.load_router_olmoe`, never bare
`from_pretrained`.

---

## 2026-01-11: Per-Layer Residual Scalars (x0 & resid lambdas)

Cherry-picked an idea from modded-nanogpt around learnable per-layer residual connections.

### Changes Made

**1. x0_lambdas (x0 residual connections)**
- Save initial normalized embedding as `x0` after `norm(wte(idx))`
- At each layer, blend x0 back in: `x = resid_lambdas[i] * x + x0_lambdas[i] * x0`
- Zero-initialized, so disabled at start; model learns which layers benefit from the shortcut
- Provides direct path from embedding to deep layers, helps preserve token information

**2. resid_lambdas (residual stream scaling)**
- Per-layer multiplicative scaling of the residual stream
- Initialized to 1.0 (neutral, standard transformer behavior)
- Allows model to learn to amplify/dampen residual at each layer

**3. DistAdamW small parameter handling**
- Added support for parameters with < 1024 elements (like the scalar lambdas)
- Small params use `all_reduce` instead of `reduce_scatter`/`all_gather`
- Fixes crash when param shape isn't divisible by world_size

### Key Finding: Different LR Sensitivity

The two scalar types need very different learning rates:
- **x0_lambdas (additive)**: Can use normal LR (~0.5). Adding a fraction of x0 is forgiving.
- **resid_lambdas (multiplicative)**: Needs ~100x smaller LR (~0.005). Multiplying the residual compounds through layers.

Implementation: `resid_params` gets `scalar_lr * 0.01`, `x0_params` gets full `scalar_lr`.

### Experiment Results

Swept `--scalar_lr` (controlling x0_lambdas) at multiple depths:

| Depth | Baseline (disabled) | Best scalar_lr | Best val_bpb | Δ bpb |
|-------|---------------------|----------------|--------------|-------|
| d8    | 1.0885              | 0.20           | 1.0782       | -0.0103 |
| d12   | 0.9770              | 0.60           | 0.9693       | -0.0077 |
| d16   | 0.9059              | 0.20           | 0.9002       | -0.0057 |
| d20   | 0.8565              | 0.10           | 0.8526       | -0.0039 |

**Observations:**
- Consistent improvement across all model sizes
- Optimal LR varies by depth; default of 0.5 is reasonable, but 0.6 is better for d12
- Adding resid_lambdas (with 0.01x LR) gives small additional improvement over x0 alone

### Meta Device Footgun

Important lesson: `__init__` runs in meta device context, so any tensor values set there are fake. Must initialize actual values in `init_weights()`. Added docstring warning to `__init__`.

### Summary

Added `--scalar_lr` (default 0.5) controlling learnable per-layer scalars. The formula `x = resid_lambdas[i] * x + x0_lambdas[i] * x0` gives the model control over residual scaling and direct shortcuts to the initial embedding. Solid improvement with essentially no compute overhead.

---

## 2026-01-10: Muon Optimizer Upgrades & Cautious Weight Decay

Cherry-picked improvements from NorMuon (modded-nanogpt) into our simpler Muon implementation. Decided against using NorMuon directly due to hard-coded architecture assumptions (expects 32 params split 10 attn + 22 mlp), parameter labeling requirements, and complexity.

### Changes Made

**1. Polar Express Orthogonalization**
- Replaced Newton-Schulz iteration with "Polar Express Sign Method" from [arxiv.org/pdf/2505.16932](https://arxiv.org/pdf/2505.16932)
- Uses 5 different coefficient tuples (one per iteration) instead of fixed coefficients
- Both methods kept in code for easy comparison (`zeropower_via_polar_express` vs `zeropower_via_newtonschulz5`)
- **Result:** No dramatic/noticeable difference in training, but keeping the new Polar Express as default.

**2. Variance Reduction (NorMuon-style)**
- Added low-rank variance estimator similar to Adafactor ([arxiv.org/pdf/2510.05491](https://arxiv.org/pdf/2510.05491))
- Maintains `second_momentum_buffer` with shape `[rows, 1]` or `[1, cols]` (whichever is smaller)
- Normalizes updates based on running per-row/col variance estimate (beta2=0.95)
- Memory overhead: ~1/max(rows, cols) per param, negligible
- **Result:** Led to a very small improvement, kept and enabled by default.

**3. Cautious Weight Decay**
- Only decays weights where `update * weight >= 0` (same sign) from [arxiv.org/abs/2411.16085](https://arxiv.org/abs/2411.16085)
- Standard WD always pulls toward zero; cautious WD skips decay when gradient is pushing weight away from zero
- **Implementation note:** Had to inline the logic rather than use a separate `@torch.compile` function. Passing changing float values (like `weight_decay` during scheduling) as function arguments triggers recompilation. Reading from `group["weight_decay"]` inside the step avoids this.
- **Result:** Solid improvements, especially the cautious version was better than standard wd.
- Now defaults to ON for Muon via the `weight_decay` param. AdamW still has no weight decay and is hardcoded to 0 weight decay, might try to re-tune this later.

**4. Weight decay schedule**
- Added a linear schedule to weight decay that is default on from 1.0 to 0.0 (i.e. start with max weight decay in the beginning of training, them ramp to 0 by the end). Worked better than a static setting in experiments. (modded-nanogpt has the same schedule but it is imlpemented in a more confusing way by multiplying twice by the learning rate, which is already wired up to a decay schedule).

### Weight Decay Scaling Experiments

Swept weight decay values at d8, d12, d16, d20 to find optimal values and scaling law.

**Optimal Values Found:**
| Depth | Width (channels) | Optimal WD |
|-------|------------------|------------|
| d8    | 512              | ~0.40      |
| d12   | 768              | ~0.22      |
| d16   | 1024             | ~0.10      |
| d20   | 1280             | ~0.08      |

**Scaling Law:**
- Fit power law: `WD = k / channels^α` in log-log space
- Found α ≈ 1.97 (approximately 2), meaning WD ∝ 1/width²

**Practical Formula:**
```
WD_target = WD_reference × (d_reference / d_target)²
```
Example: If d12 optimal is 0.22, then d20 optimal ≈ 0.22 × (12/20)² ≈ 0.08

**Reference:** Moonlight paper uses fixed WD=0.1 for their 15B MoE model. Our experiments indicated a scaling law where the optimal WD changed with depth, so we go along with the empirical scaling law.

### Summary

Muon was changed to use Polar Express, added Adafactor-style variance reduction, and cautious weight decay with schedule that ramps linearly to zero. All of these changes follow modded-nanogpt repo, but all of them were also validated piece by piece to yield improvements in nanochat with the exception of the Polar Express change which was in the noise. This is default on and configurable with `--weight_decay`, using simply 0.2 and ∝ 1/width² scaling. The kwarg `--weight_decay` is therefore changing as of this change. It used to configure AdamW via standard weight decay and now it becomes exclusively used in Muon (AdamW is hardcoded to 0.0), and it is scaled based on depth.

---

## 2026-01-08: exp_grad_clip - Gradient Clipping

**Hypothesis:** Gradient clipping may be unnecessary overhead. Tested L2 norm clipping at various thresholds (0.25, 0.5, 1.0, 2.0) and elementwise clipping.

**Results:**
- No benefit at any scale tested (d12, d20)
- All variants within noise (~0.9827 val_bpb)
- Grad norm never exceeds 1.0 naturally, so clipping is always inactive
- Clipping adds ~2% time overhead from the all-reduce

**Bug Found:** Original implementation clipped local gradients before sync. Since this codebase doesn't use DDP (gradient sync is in the optimizers), each rank was clipping based on its own local norm. Fixed on the branch with proper distributed all-reduce.

**Observartion:** modded-nanogpt does not appear to clip either right now.

**Summary:** Deleted all grad-clip code paths. The code naturally produces well-behaved gradients. This improves a bit of MFU because we don't have to calculate and sync grad norms.
