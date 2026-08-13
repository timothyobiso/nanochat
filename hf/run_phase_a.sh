#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════════════════════
# Phase A: from-scratch router comparison in the HF stack (8xH100 node)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Runs the plan in docs/HF_PORT_PLAN.md §5. Stages (in order; each skippable):
#   setup   uv sync with the gpu extra + hf group
#   data    tokenize FineWeb-Edu -> uint16 shards (one-time, ~30B tokens)
#   pilot   size S x {linear, vsa_fpe}: measures real MFU BEFORE the matrix —
#           re-check the wall-clock estimates in the plan against its tok/s
#   matrix  5 routers x (S x2 seeds + M x2 seeds + L x1) = 25 runs
#   ablations  5 more (aux-loss-coeff 0, norm_topk_prob probe, clifford stretch)
#   figures loss curves + utilization heatmaps + router latency
#
# The run matrix itself lives in hf/runs.sh, shared with the SLURM jobs in
# hf/slurm/. This driver runs everything in the foreground on one node; for a
# scheduler with a wall-clock cap use hf/slurm/e2e.sh instead. Both treat a
# `DONE` file in the run dir as the completion marker and resume from the newest
# checkpoint otherwise.
#
# Usage:
#   bash hf/run_phase_a.sh --only pilot
#   bash hf/run_phase_a.sh --only matrix --sizes S,M        # trim the matrix
#   bash hf/run_phase_a.sh --only matrix --routers vsa_fpe --sizes L
#   bash hf/run_phase_a.sh                                  # everything
#
# Options:
#   --only STAGE       setup, data, pilot, matrix, ablations, figures
#   --routers LIST     comma list (default: linear,hash,vsa_random,vsa_fpe,direct_fpe)
#   --sizes LIST       comma list of S,M,L (default: S,M,L)
#   --gpus N           (default: 8)
#   --data-dir DIR     (default: /data/$USER/hf_data/fineweb_edu_olmoe)
#   --out-dir DIR      (default: /data/$USER/hf_runs)
#   --data-tokens N    train tokens to tokenize (default: 15000000000 — L needs
#                      11.5B and Phase B rereads the same shards)
#   --wandb NAME       wandb run prefix; 'dummy' disables (default: dummy)

ROUTERS="linear,hash,vsa_random,vsa_fpe,direct_fpe"
SIZES="S,M,L"
GPUS=8
DATA_DIR="/data/$(whoami)/hf_data/fineweb_edu_olmoe"
OUT_DIR="/data/$(whoami)/hf_runs"
DATA_TOKENS=15000000000
ONLY=""
WANDB="dummy"

while [[ $# -gt 0 ]]; do
    case $1 in
        --only) ONLY="$2"; shift 2 ;;
        --routers) ROUTERS="$2"; shift 2 ;;
        --sizes) SIZES="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --data-tokens) DATA_TOKENS="$2"; shift 2 ;;
        --wandb) WANDB="$2"; shift 2 ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

export OMP_NUM_THREADS=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=true

mkdir -p "$OUT_DIR"   # tee below opens the log before Python creates the dir

# size_params and the run manifests live in hf/runs.sh so this driver and the
# SLURM jobs in hf/slurm/ describe the same experiment.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runs.sh"

run_stage() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
in_list() { [[ ",$2," == *",$1,"* ]]; }

train_entry() {  # train_entry "<name> <size> <router> <seed> [extra flags…]"
    local name size router seed extra
    read -r name size router seed extra <<< "$1"
    size_params "$size"
    # DONE, not metrics.jsonl: a run killed partway has metrics but is not done.
    # rm -rf the run dir to start it over; otherwise it resumes.
    if [[ -f "$OUT_DIR/$name/DONE" ]]; then
        echo "=== $name already complete, skipping (rm -rf $OUT_DIR/$name to rerun) ==="
        return 0
    fi
    local wandb_arg="dummy"
    [[ "$WANDB" != "dummy" ]] && wandb_arg="${WANDB}_${name}"
    echo "=== training $name ${extra:+[$extra]} ==="
    # shellcheck disable=SC2086  # $extra is a deliberate flag list
    torchrun --standalone --nproc_per_node="$GPUS" -m hf.train_olmoe -- \
        --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --run-name "$name" \
        --router "$router" --router-seed "$seed" --seed "$seed" \
        --hidden-size "$HIDDEN" --num-layers "$LAYERS" --num-heads "$HEADS" \
        --lr "$LR" --device-batch-size "$DEV_BS" --resume auto --run "$wandb_arg" $extra \
        2>&1 | tee -a "$OUT_DIR/${name}.log"
}

find_entry() {  # echo the manifest entry whose run name is $1
    local entry
    for entry in "${PHASE_A_RUNS[@]}"; do
        [[ "${entry%% *}" == "$1" ]] && { echo "$entry"; return 0; }
    done
    echo "no such run: $1" >&2
    return 1
}

# ── setup ────────────────────────────────────────────────────────────────────
if run_stage setup; then
    uv sync --extra gpu --group hf
    source .venv/bin/activate
else
    source .venv/bin/activate 2>/dev/null || true
fi

# ── data (one-time) ──────────────────────────────────────────────────────────
if run_stage data; then
    if [[ -f "$DATA_DIR/manifest.json" ]]; then
        echo "=== $DATA_DIR/manifest.json exists, skipping tokenization ==="
    else
        python -m hf.prepare_data --data-dir "$DATA_DIR" \
            --num-tokens "$DATA_TOKENS" --val-tokens 50000000
    fi
fi

# ── pilot: measure MFU before committing the matrix ─────────────────────────
if run_stage pilot; then
    for name in "${PHASE_A_PILOT[@]}"; do
        train_entry "$(find_entry "$name")"
    done
    echo "=== PILOT DONE: check tok/s in the logs above against the plan's"
    echo "=== wall-clock table before running the full matrix."
fi

# ── matrix ───────────────────────────────────────────────────────────────────
if run_stage matrix; then
    for entry in "${PHASE_A_MATRIX[@]}"; do
        read -r _ size router _ <<< "$entry"
        if in_list "$size" "$SIZES" && in_list "$router" "$ROUTERS"; then
            train_entry "$entry"
        fi
    done
fi

# ── ablations (plan §5 add-ons) ──────────────────────────────────────────────
if run_stage ablations; then
    for entry in "${PHASE_A_ABLATIONS[@]}"; do
        train_entry "$entry" || true
    done
fi

# ── figures ──────────────────────────────────────────────────────────────────
if run_stage figures; then
    for size in ${SIZES//,/ }; do
        runs=""; labels=""
        for router in ${ROUTERS//,/ }; do
            [[ -f "$OUT_DIR/${size}_${router}_s0/metrics.jsonl" ]] || continue
            runs+="$OUT_DIR/${size}_${router}_s0,"; labels+="$router,"
        done
        [[ -n "$runs" ]] && python -m hf.analysis --mode phase_a \
            --runs "${runs%,}" --labels "${labels%,}" \
            --output-dir "$OUT_DIR/figures_$size"
    done
    size_params S
    python -m hf.analysis --mode latency --dim "$HIDDEN" --experts 8 --top-k 2 \
        --device cuda --output-dir "$OUT_DIR/figures_S"
fi

echo "Phase A driver done."
