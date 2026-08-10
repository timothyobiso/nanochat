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
#   matrix  5 routers x (S x2 seeds + M x2 seeds + L x1) = 25 runs + ablations
#   figures loss curves + utilization heatmaps + router latency
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
#   --data-tokens N    train tokens to tokenize (default: 30000000000)
#   --wandb NAME       wandb run prefix; 'dummy' disables (default: dummy)

ROUTERS="linear,hash,vsa_random,vsa_fpe,direct_fpe"
SIZES="S,M,L"
GPUS=8
DATA_DIR="/data/$(whoami)/hf_data/fineweb_edu_olmoe"
OUT_DIR="/data/$(whoami)/hf_runs"
DATA_TOKENS=30000000000
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

run_stage() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

# Per-size hyperparams from the plan table (docs/HF_PORT_PLAN.md §5).
# device-batch is a starting point — tune at the pilot if OOM/underutilized.
size_params() {
    case $1 in
        S) HIDDEN=512;  LAYERS=8;  HEADS=8;  LR=6e-4; DEV_BS=16 ;;
        M) HIDDEN=768;  LAYERS=12; HEADS=12; LR=5e-4; DEV_BS=8 ;;
        L) HIDDEN=1024; LAYERS=16; HEADS=16; LR=4e-4; DEV_BS=4 ;;
        *) echo "unknown size: $1"; exit 1 ;;
    esac
}

train_one() {  # train_one SIZE ROUTER SEED [extra flags...]
    local size=$1 router=$2 seed=$3; shift 3
    size_params "$size"
    local name="${size}_${router}_s${seed}"
    if [[ -f "$OUT_DIR/$name/metrics.jsonl" ]] && [[ -z "$FORCE" ]]; then
        echo "=== $name already has metrics.jsonl, skipping (rm to rerun) ==="
        return 0
    fi
    local wandb_arg="dummy"
    [[ "$WANDB" != "dummy" ]] && wandb_arg="${WANDB}_${name}"
    echo "=== training $name ==="
    torchrun --standalone --nproc_per_node="$GPUS" -m hf.train_olmoe -- \
        --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" --run-name "$name" \
        --router "$router" --router-seed "$seed" --seed "$seed" \
        --hidden-size "$HIDDEN" --num-layers "$LAYERS" --num-heads "$HEADS" \
        --lr "$LR" --device-batch-size "$DEV_BS" --run "$wandb_arg" "$@" \
        2>&1 | tee -a "$OUT_DIR/${name}.log"
}

seeds_for_size() { case $1 in S|M) echo "0 1" ;; L) echo "0" ;; esac; }

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
    for router in linear vsa_fpe; do
        train_one S "$router" 0
    done
    echo "=== PILOT DONE: check tok/s in the logs above against the plan's"
    echo "=== wall-clock table before running the full matrix."
fi

# ── matrix ───────────────────────────────────────────────────────────────────
if run_stage matrix; then
    for size in ${SIZES//,/ }; do
        for router in ${ROUTERS//,/ }; do
            for seed in $(seeds_for_size "$size"); do
                train_one "$size" "$router" "$seed"
            done
        done
    done
fi

# ── ablations (plan §5 add-ons) ──────────────────────────────────────────────
if run_stage ablations; then
    for router in linear vsa_fpe; do  # aux-loss-coeff 0 at S
        size_params S
        train_one S "$router" 0 --aux-loss-coeff 0.0 || true
    done
    train_one S vsa_fpe 0 --norm-topk-prob || true   # norm_topk_prob probe
    train_one M clifford_quat_fpe 0 || true          # stretch
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
