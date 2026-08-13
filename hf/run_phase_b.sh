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

# condition list shared with hf/slurm/05_heal.sbatch
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runs.sh"

run_stage() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
B0_REPORT="$OUT_DIR/b0_report.json"

# 48GB cards: fp32 params + ZeRO-1 Adam needs ~63GB/GPU, bf16 ~35GB. Override
# with PARAM_DTYPE=float32 on 80GB hardware.
: "${PARAM_DTYPE:=bfloat16}"

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
heal_entry() {  # heal_entry "<name> <condition> <router>"
    local name condition router
    read -r name condition router <<< "$1"
    # DONE, not metrics.jsonl: a run killed partway has metrics but is not done.
    if [[ -f "$OUT_DIR/$name/DONE" ]]; then
        echo "=== $name already complete, skipping (rm -rf $OUT_DIR/$name to rerun) ==="
        return 0
    fi
    local b0_args=() seed=0
    if [[ -f "$B0_REPORT" ]]; then
        b0_args=(--b0-report "$B0_REPORT")
        seed=$(python -c \
            'import sys; from hf.diagnose_router import best_seed_for; print(best_seed_for(sys.argv[1], sys.argv[2]))' \
            "$B0_REPORT" "$router")
    fi
    local wandb_arg="dummy"
    [[ "$WANDB" != "dummy" ]] && wandb_arg="${WANDB}_${name}"
    echo "=== healing: $name (router-seed $seed) ==="
    torchrun --standalone --nproc_per_node="$GPUS" -m hf.heal_olmoe -- \
        --model "$MODEL" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" \
        --run-name "$name" --condition "$condition" --router "$router" \
        --router-seed "$seed" --num-tokens "$NUM_TOKENS" --param-dtype "$PARAM_DTYPE" \
        --resume auto --run "$wandb_arg" "${b0_args[@]}" \
        2>&1 | tee -a "$OUT_DIR/${name}.log"
}

# entries from PHASE_B_RUNS selected by --conditions (which names them without
# the shared "heal_" prefix)
selected_entries() {
    local entry name cond
    for entry in "${PHASE_B_RUNS[@]}"; do
        name="${entry%% *}"
        for cond in ${CONDITIONS//,/ }; do
            [[ "$name" == "heal_$cond" ]] && { echo "$entry"; break; }
        done
    done
    return 0   # the inner test fails on the last non-match; don't trip set -e
}

# fail loudly on a typo rather than quietly running nothing
for cond in ${CONDITIONS//,/ }; do
    if ! printf '%s\n' "${PHASE_B_RUNS[@]}" | grep -q "^heal_${cond} "; then
        echo "unknown condition: $cond" >&2
        echo "known: $(printf '%s\n' "${PHASE_B_RUNS[@]}" | sed 's/^heal_//;s/ .*//' | tr '\n' ' ')" >&2
        exit 1
    fi
done

if run_stage heal; then
    while read -r entry; do
        [[ -n "$entry" ]] && heal_entry "$entry"
    done <<< "$(selected_entries)"
fi

# ── milestone lm-evals ───────────────────────────────────────────────────────
if run_stage evals; then
    # milestone steps are resolved by heal_olmoe and recorded in config.json, so
    # they stay correct when --num-tokens changes
    while read -r entry; do
        [[ -n "$entry" ]] || continue
        name="${entry%% *}"
        config="$OUT_DIR/$name/config.json"
        [[ -f "$config" ]] || continue
        for step in $(python -c "
import json, sys
print(' '.join(str(s) for s in json.load(open(sys.argv[1]))['milestone_steps']))" "$config"); do
            ckpt=$(printf '%s/%s/step_%06d/model' "$OUT_DIR" "$name" "$step")
            out=$(printf '%s/eval_%s_%06d.json' "$OUT_DIR" "$name" "$step")
            [[ -d "$ckpt" ]] || continue
            python -m hf.eval_lm --model "$ckpt" --out "$out"
        done
    done <<< "$(selected_entries)"
fi

# ── figures ──────────────────────────────────────────────────────────────────
if run_stage figures; then
    runs=""; labels=""
    while read -r entry; do
        [[ -n "$entry" ]] || continue
        name="${entry%% *}"
        [[ -f "$OUT_DIR/$name/metrics.jsonl" ]] || continue
        runs+="$OUT_DIR/$name,"; labels+="${name#heal_},"
    done <<< "$(selected_entries)"
    [[ -n "$runs" ]] && python -m hf.analysis --mode healing \
        --runs "${runs%,}" --labels "${labels%,}" --output-dir "$OUT_DIR/figures"
    python -m hf.analysis --mode latency --dim 2048 --experts 64 --top-k 8 \
        --device cuda --output-dir "$OUT_DIR/figures"
fi

echo "Phase B driver done."
