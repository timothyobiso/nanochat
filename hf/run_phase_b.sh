#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════════════════════
# Phase B: OLMoE-1B-7B router swap — diagnostics + healing (8xH100 node)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Runs the plan in docs/HF_PORT_PLAN.md §6-7. Stages (in order; each skippable):
#   baseline  lm-eval on the UNMODIFIED checkpoint. HARD GATE: results must be
#             within ~1pt of the published OLMoE-0924 numbers (see hf/eval_lm.py
#             docstring) before anything downstream is meaningful.
#   b0        diagnostics: agreement/Hungarian, FPE seed search, hard-swap ppl,
#             gate distillation -> $OUT_DIR/b0_report.json (1 GPU is enough,
#             but running on the full node is fine)
#   heal      4 conditions x 5B tokens: control, blend vsa_fpe, blend
#             vsa_random, hard_swap vsa_fpe. Best FPE seed + Hungarian perms
#             are pulled from the b0 report automatically.
#   evals     lm-eval at the 1B / 2.5B / 5B checkpoints of every condition
#   figures   healing curves + routing drift
#
# Usage:
#   bash hf/run_phase_b.sh --only baseline
#   bash hf/run_phase_b.sh --only b0
#   bash hf/run_phase_b.sh --only heal --conditions blend_vsa_fpe,control
#   bash hf/run_phase_b.sh                       # everything, in order
#
# Options:
#   --only STAGE        baseline, b0, heal, evals, figures
#   --conditions LIST   subset of: control,blend_vsa_fpe,blend_vsa_random,hard_swap_vsa_fpe
#   --model NAME        (default: allenai/OLMoE-1B-7B-0924)
#   --gpus N            (default: 8)
#   --data-dir DIR      shards from hf/run_phase_a.sh --only data
#   --out-dir DIR       (default: /data/$USER/hf_heal)
#   --num-tokens N      per healing condition (default: 5000000000)
#   --wandb NAME        wandb run prefix; 'dummy' disables (default: dummy)

MODEL="allenai/OLMoE-1B-7B-0924"
GPUS=8
DATA_DIR="/data/$(whoami)/hf_data/fineweb_edu_olmoe"
OUT_DIR="/data/$(whoami)/hf_heal"
NUM_TOKENS=5000000000
CONDITIONS="control,blend_vsa_fpe,blend_vsa_random,hard_swap_vsa_fpe"
ONLY=""
WANDB="dummy"

while [[ $# -gt 0 ]]; do
    case $1 in
        --only) ONLY="$2"; shift 2 ;;
        --conditions) CONDITIONS="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --num-tokens) NUM_TOKENS="$2"; shift 2 ;;
        --wandb) WANDB="$2"; shift 2 ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

export OMP_NUM_THREADS=1
export HF_HUB_ENABLE_HF_TRANSFER=1
source .venv/bin/activate 2>/dev/null || true
mkdir -p "$OUT_DIR"

run_stage() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
B0_REPORT="$OUT_DIR/b0_report.json"

# ── baseline gate ────────────────────────────────────────────────────────────
if run_stage baseline; then
    echo "=== baseline lm-eval (published-numbers gate) ==="
    python -m hf.eval_lm --model "$MODEL" --out "$OUT_DIR/baseline_eval.json"
    echo "=== GATE: compare $OUT_DIR/baseline_eval.json against the reference"
    echo "=== numbers in hf/eval_lm.py's docstring (~1pt tolerance) before B0/B1."
fi

# ── B0 diagnostics ───────────────────────────────────────────────────────────
if run_stage b0; then
    python -m hf.diagnose_router --model "$MODEL" --data-dir "$DATA_DIR" \
        --device cuda --per-layer-swap --distill-ppl --out "$B0_REPORT" \
        2>&1 | tee "$OUT_DIR/b0.log"
fi

# ── healing conditions ───────────────────────────────────────────────────────
heal_one() {  # heal_one NAME CONDITION ROUTER [extra flags...]
    local name=$1 condition=$2 router=$3; shift 3
    if [[ -f "$OUT_DIR/$name/metrics.jsonl" ]]; then
        echo "=== $name already has metrics.jsonl, skipping (rm to rerun) ==="
        return 0
    fi
    local wandb_arg="dummy"
    [[ "$WANDB" != "dummy" ]] && wandb_arg="${WANDB}_${name}"
    echo "=== healing: $name ==="
    torchrun --standalone --nproc_per_node="$GPUS" -m hf.heal_olmoe -- \
        --model "$MODEL" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" \
        --run-name "$name" --condition "$condition" --router "$router" \
        --num-tokens "$NUM_TOKENS" --run "$wandb_arg" "$@" \
        2>&1 | tee -a "$OUT_DIR/${name}.log"
}

if run_stage heal; then
    # best FPE base seed from the B0 seed search (0 if the report lacks one)
    BEST_SEED=$(python -c "
import json, sys
try:
    r = json.load(open('$B0_REPORT'))
    print(r['seed_search']['top16'][0]['seed'])
except Exception:
    print(0); sys.exit(0)
")
    echo "=== using router seed $BEST_SEED (from $B0_REPORT) ==="
    PERM_ARGS=()
    [[ -f "$B0_REPORT" ]] && PERM_ARGS=(--b0-report "$B0_REPORT")
    for cond in ${CONDITIONS//,/ }; do
        case $cond in
            control)           heal_one heal_control control linear ;;
            blend_vsa_fpe)     heal_one heal_blend_vsa_fpe blend vsa_fpe \
                                   --router-seed "$BEST_SEED" "${PERM_ARGS[@]}" ;;
            blend_vsa_random)  heal_one heal_blend_vsa_random blend vsa_random \
                                   --router-seed "$BEST_SEED" "${PERM_ARGS[@]}" ;;
            hard_swap_vsa_fpe) heal_one heal_hard_swap_vsa_fpe hard_swap vsa_fpe \
                                   --router-seed "$BEST_SEED" "${PERM_ARGS[@]}" ;;
            *) echo "unknown condition: $cond"; exit 1 ;;
        esac
    done
fi

# ── milestone lm-evals ───────────────────────────────────────────────────────
if run_stage evals; then
    # 5B tokens / 2M per step -> step 2500; milestones at 1B/2.5B/5B
    for step in 000500 001250 002500; do
        for cond in ${CONDITIONS//,/ }; do
            name="heal_${cond}"
            ckpt="$OUT_DIR/$name/step_$step/model"
            [[ -d "$ckpt" ]] || continue
            out="$OUT_DIR/eval_${name}_${step}.json"
            [[ -f "$out" ]] && continue
            python -m hf.eval_lm --model "$ckpt" --out "$out"
        done
    done
fi

# ── figures ──────────────────────────────────────────────────────────────────
if run_stage figures; then
    runs=""; labels=""
    for cond in ${CONDITIONS//,/ }; do
        name="heal_${cond}"
        [[ -f "$OUT_DIR/$name/metrics.jsonl" ]] || continue
        runs+="$OUT_DIR/$name,"; labels+="${cond},"
    done
    [[ -n "$runs" ]] && python -m hf.analysis --mode healing \
        --runs "${runs%,}" --labels "${labels%,}" --output-dir "$OUT_DIR/figures"
    python -m hf.analysis --mode latency --dim 2048 --experts 64 --top-k 8 \
        --device cuda --output-dir "$OUT_DIR/figures"
fi

echo "Phase B driver done."
