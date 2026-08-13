# Running the HF router experiments on SLURM

End-to-end submission of the Phase A / Phase B pipeline in
[`docs/HF_PORT_PLAN.md`](../../docs/HF_PORT_PLAN.md) as individual SLURM jobs,
each under a 24-hour wall-clock cap.

The pipeline needs far more than 24 hours of compute, so the two long stages are
split across **chained jobs that resume from their last checkpoint**. Nothing
here changes the experiment — it is the same trainers the foreground drivers
(`hf/run_phase_a.sh`, `hf/run_phase_b.sh`) call, driven by a scheduler instead
of by one long-lived shell.

## Quick start

```bash
uv sync --extra gpu --group hf          # once, on a node that can see the GPUs
$EDITOR hf/slurm/config.sh              # partition, account, paths  (see below)

bash hf/slurm/e2e.sh --check            # preflight: config, venv, nodes, disk
bash hf/slurm/e2e.sh --dry-run          # print every sbatch line, submit nothing
bash hf/slurm/e2e.sh                    # submit the whole pipeline
```

Single stages, no dependencies — this is how you re-run one piece:

```bash
bash hf/slurm/e2e.sh --only data
bash hf/slurm/e2e.sh --only train
bash hf/slurm/e2e.sh --only heal
```

Stages are `data train baseline b0 heal evals figures`.

## Before your first real submission

1. **Edit `config.sh`.** Everything marked `# EDIT` is site-specific and cannot
   be guessed: `PARTITION`, `ACCOUNT`, `BIG_NODES`, and whether your site wants
   `--gres=gpu:8` or `--gpus-per-node=8` (`GRES_MULTI` / `GRES_ONE`). Set
   `MODULES` if CUDA needs `module load`.
2. **Run `--check`** and resolve the warnings.
3. **Prove resume works before spending node-days on it.** Submit one Phase A
   task, let it write a checkpoint, cancel it, resubmit, and confirm from
   `metrics.jsonl` that it picked up at the checkpoint step rather than 0:

   ```bash
   sbatch --partition=$PARTITION --array=0 --gres=gpu:8 \
          --nodelist=student-gpu-[003-004] hf/slurm/02_train.sbatch
   # …wait for "saved …/step_000250", then:
   scancel <jobid>
   # resubmit the same line; the log should say "resuming S_linear_s0 from step 250"
   ```

   The whole 24h design rests on this. `tests/test_hf_heal_resume.py` checks the
   same property for Phase B on CPU, but only the cluster tells you whether the
   checkpoint cadence beats your wall-clock limit.
4. **Check the pilot's throughput** against the table below and raise
   `SEGMENTS_A` / `SEGMENTS_B` if the real numbers are slower.
5. **The baseline lm-eval is a hard gate.** `03_baseline.sbatch` evaluates the
   unmodified OLMoE-1B-7B-0924 checkpoint. If it does not reproduce the
   published numbers in `hf/eval_lm.py`'s docstring within ~1 point, nothing in
   Phase B means anything. `e2e.sh` makes `heal` depend on this job completing,
   but it cannot check the numbers for you.

## The jobs

| # | Job | Shape | `--time` | Chained | Depends on |
|---|-----|-------|----------|---------|------------|
| 01 | `01_data.sbatch` | CPU, 16 cores | 24:00 | no | — |
| 02 | `02_train.sbatch` | 1 node × 8 GPU, array `0-29%2` | 23:00 | `SEGMENTS_A` | data |
| 03 | `03_baseline.sbatch` | 1 GPU | 12:00 | no | data |
| 04 | `04_diagnose.sbatch` | 1 GPU | 23:00 | 2 | data |
| 05 | `05_heal.sbatch` | 1 node × 8 GPU, array `0-3` | 23:00 | `SEGMENTS_B` | b0 + baseline |
| 06 | `06_evals.sbatch` | 1 GPU, array `0-3` | 23:00 | no | heal |
| 07 | `07_figures.sbatch` | 1 GPU | 01:00 | no | evals |

`--time=23:00:00` rather than `24:00:00` leaves headroom for model load,
calibration and the resume scan inside the cap.

The array indexes into `PHASE_A_RUNS` / `PHASE_B_RUNS` in
[`hf/runs.sh`](../runs.sh), which is the single definition of the experiment
matrix — the foreground drivers read the same arrays. To see what index maps to
what:

```bash
bash -c 'source hf/runs.sh; printf "%s\n" "${PHASE_A_RUNS[@]}" | cat -n'
```

Phase A is 30 runs: 25 matrix (5 routers × S/M at 2 seeds, L at 1) plus 5
ablations. The `%2` in `0-29%2` caps concurrency at two jobs, one per big node.

## How resume and completion work

Two rules, used identically by every job and by both foreground drivers:

- **`DONE` means finished.** Each trainer writes `<run_dir>/DONE` only after its
  last step. A job whose run has `DONE` exits in seconds.
- **`--resume auto` means continue.** The trainer picks the newest `step_*` dir
  in its run directory and restores model, optimizer, data position, and (for
  Phase B) the routing-drift reference. With no checkpoint it starts fresh, so
  the same command line works for a first run and every continuation.

`e2e.sh` submits each chained stage `SEGMENTS` times with
`--dependency=afterany`. `afterany`, not `afterok`, is deliberate: a wall-clock
kill registers as failure, and that is exactly the case the next segment exists
to continue. Over-submitting is free — surplus segments find `DONE` and exit.

**Do not use `metrics.jsonl` as a completion test.** A run killed at 5% has one.
That was the old skip predicate and it would silently leave truncated runs in
the results matrix.

To force a genuine re-run, delete the run directory (`rm -rf $OUT_DIR/<name>`).
Deleting only `DONE` makes the job resume from the last checkpoint instead.

### Checkpoint retention

Full checkpoints are large and the trainers write many. After each save,
`hf/checkpoints.py:prune_checkpoints` keeps:

- the **newest** checkpoint whole — the only one a resume can use;
- **milestone** checkpoints as `model/` only, with `trainer_state.pt` stripped,
  because they exist for lm-eval, not for resuming;
- nothing else.

Phase B milestones are resolved by `heal_olmoe` from `MILESTONE_FRACS`
(0.2/0.5/1.0 of the run → 1B/2.5B/5B tokens at the default budget), rounded up
to a `--save-every` boundary so the checkpoint actually exists, and recorded as
`milestone_steps` in the run's `config.json`. `06_evals.sbatch` reads them back
from there, so changing `NUM_TOKENS` does not silently produce zero evals.

## Hardware notes (8 × 48GB)

The plan was originally costed for an 8×H100 80GB node. On 48GB cards:

- **Phase B must run `--param-dtype bfloat16`.** fp32 params with ZeRO-1-sharded
  Adam need ~63 GB/GPU for OLMoE's 6.9B params; bf16 params/grads/states are
  ~35 GB, leaving room for activations under gradient checkpointing at
  `--device-batch-size 2`. `config.sh` sets `PARAM_DTYPE=bfloat16`; set it to
  `float32` on 80GB hardware.
- **Phase A fits comfortably** at the existing per-size device batch sizes
  (S/M/L = 16/8/4). Those divide `--total-batch-size 524288` evenly only at
  `GPUS=8`; the trainer asserts if you change the GPU count without retuning.
- One thing to watch at the pilot: `optimizer.consolidate_state_dict(to=0)`
  gathers the whole optimizer state onto rank 0 at every Phase B save.

### Estimates, not measurements

Wall-clock assumes 48GB Ada-class cards at roughly 2.5–3.5× an H100 for this
workload, with no NVLink. **Re-baseline all of it at the pilot** — the plan's
own numbers assume 10–20% MFU and HF's per-expert loop is kernel-launch-bound at
small scale.

| Stage | Plan (H100) | Estimate here | Notes |
|---|---|---|---|
| Phase A, per run S / M / L | 0.5–1h / 1–2h / 4–8h | ~2–3h / ~3–7h / ~10–28h | only L needs a second segment |
| Phase A, all 30 runs | ~2.5 node-days | ~185 node-hours ≈ 4 days on 2 nodes | |
| Phase B, per condition | ~9h | ~23–32h | needs 2 segments |
| Phase B, all 4 | ~1.5–2 node-days | ~108 node-hours ≈ 2.3 days on 2 nodes | |
| Data prep (15B tokens) | — | ~4h, CPU | not resumable |
| B0 diagnostics | "nearly free" | ~10h, 1 GPU | |

Whole pipeline: roughly **a week** across the two nodes.

**Disk** ≈ 350 GB (Phase B, pruned) + ~180 GB (Phase A, pruned) + 30 GB (15B
uint16 tokens) ≈ **560 GB**. Without pruning Phase B alone would be ~3.3 TB.

`DATA_TOKENS` defaults to 15B rather than the plan's 30B: Phase A's largest run
needs 11.5B and Phase B rereads the same shards, so 15B covers the pipeline with
margin while halving the one job that cannot be resumed. `prepare_data` writes
`manifest.json` only after the last shard, so an interrupted run leaves orphan
`.bin` files and must start over — `--check` warns if `DATA_TOKENS` implies more
than ~20h.

## Monitoring

```bash
squeue -u $USER                       # queue and dependency state
tail -f $LOG_DIR/hf-train-*.out       # LOG_DIR from config.sh
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode

# progress across the Phase A matrix
for d in $OUT_DIR/*/; do
  printf '%-24s %s\n' "$(basename "$d")" \
    "$([[ -f $d/DONE ]] && echo DONE || echo "step $(wc -l < "$d/metrics.jsonl" 2>/dev/null || echo 0)")"
done
```

A job stuck in `DependencyNeverSatisfied` means an upstream `afterok` dependency
failed — check the upstream log; `scancel` the orphans and resubmit that stage
with `--only`.

## Known gaps carried into these runs

Documented rather than fixed, so they are accounted for at write-up time.

- **Phase A gate-mass confound.** `FixedRouterGate` has no scale knob and the
  host default is `norm_topk_prob=False`, so a low-variance fixed router leaves
  the selected gate mass near `top_k/E` (~0.25) while a learned gate sits near 1
  — the MoE branch is systematically down-scaled in VSA runs. The matrix keeps
  the committed host-fidelity design; the `*_normtopk` ablations run `linear`
  and `vsa_fpe` with `norm_topk_prob=True` so the size of the effect is
  measurable, and `expert_telemetry` logs `gate_mass` throughout.
- **No step-0 lm-eval.** The plan asks for milestones at 0/1B/2.5B/5B; only the
  latter three are wired up, so the "zero-shot hard swap is catastrophic"
  datapoint comes from B0's swap perplexity rather than from lm-eval.
- **`HashRouter` routing depends on batch shape**, not token identity
  (`nanochat/gpt.py` uses `arange` over the flattened `B*T` buffer), so
  `hf/eval_lm.py`'s `--batch-size auto` makes hash results non-reproducible.
  Pin `--batch-size` if hash numbers ever need to be comparable.
- **FPE expert 0 is structurally special**: `make_fpe_keys` gives it exponent 0,
  i.e. the identity for the binding operation, so experts are not exchangeable
  — worth remembering when reading Hungarian matching results.
- **`safetensors`** is imported directly by `hf/patch_olmoe.py` but is only a
  transitive dependency of `transformers`.

## Related files

| Path | What it is |
|---|---|
| `hf/runs.sh` | the experiment matrix, shared by these jobs and the foreground drivers |
| `hf/checkpoints.py` | `DONE` sentinel, newest-checkpoint lookup, retention policy |
| `hf/run_phase_a.sh`, `hf/run_phase_b.sh` | same pipeline, foreground, single node, no wall-clock cap |
| `docs/HF_PORT_PLAN.md` | why any of this is being run |
| `tests/test_hf_heal_resume.py` | CPU proof that Phase B resume is exact |
