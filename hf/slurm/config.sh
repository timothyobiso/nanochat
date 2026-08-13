#!/bin/bash
# Cluster-specific settings for the HF experiment pipeline.
#
# This is the only file you should need to edit. Every value uses `: "${X:=…}"`,
# so anything already exported in your environment wins — you can override a
# single knob for one submission without editing the file:
#     DATA_TOKENS=2000000000 bash hf/slurm/e2e.sh --only data
#
# Sourced by e2e.sh and, independently, by each *.sbatch job, so the jobs work
# whether they are submitted through e2e.sh or by hand.

# ── EDIT: scheduler ──────────────────────────────────────────────────────────
: "${PARTITION:=gpu}"                       # sinfo -s
: "${ACCOUNT:=}"                            # leave empty if your site doesn't require -A
: "${BIG_NODES:=student-gpu-[003-004]}"     # the 8x48GB nodes; used by the 8-GPU stages
: "${GRES_MULTI:=--gres=gpu:8}"             # some sites want --gpus-per-node=8
: "${GRES_ONE:=--gres=gpu:1}"
: "${MODULES:=}"                            # e.g. "cuda/12.8 gcc/12"; blank to skip

# ── paths ────────────────────────────────────────────────────────────────────
: "${REPO_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${VENV:=$REPO_DIR/.venv}"
: "${DATA_DIR:=/data/$(whoami)/hf_data/fineweb_edu_olmoe}"
: "${OUT_DIR:=/data/$(whoami)/hf_runs}"     # Phase A
: "${HEAL_DIR:=/data/$(whoami)/hf_heal}"    # Phase B
: "${LOG_DIR:=/data/$(whoami)/hf_slurm_logs}"

# ── experiment knobs ─────────────────────────────────────────────────────────
: "${GPUS:=8}"
: "${MODEL:=allenai/OLMoE-1B-7B-0924}"
: "${DATA_TOKENS:=15000000000}"   # Phase A's L run needs 11.5B and B1 rereads the same
                                  # shards, so 15B covers the pipeline with margin. The
                                  # plan's 30B doubles the (unresumable) tokenization job.
: "${NUM_TOKENS:=5000000000}"     # per B1 healing condition
: "${PARAM_DTYPE:=bfloat16}"      # 48GB cards: fp32 params need ~63GB/GPU, bf16 ~35GB
: "${SAVE_EVERY_A:=250}"          # Phase A checkpoint cadence; bounds work lost to a kill
: "${WANDB:=dummy}"               # 'dummy' disables wandb

# Segments per chained stage: each is a separate <=24h job that resumes from the
# previous one's last checkpoint and exits immediately once DONE exists. Raise
# these if the pilot comes in slower than the README's table.
: "${SEGMENTS_A:=2}"
: "${SEGMENTS_B:=2}"

# ─────────────────────────────────────────────────────────────────────────────

activate_env() {
    # shellcheck disable=SC1090
    [[ -n "$MODULES" ]] && module load $MODULES
    cd "$REPO_DIR"
    source "$VENV/bin/activate"
    export OMP_NUM_THREADS=1
    export HF_HUB_ENABLE_HF_TRANSFER=1
    export TOKENIZERS_PARALLELISM=false
}
