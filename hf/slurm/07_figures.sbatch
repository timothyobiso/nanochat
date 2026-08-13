#!/bin/bash
# Figures for both phases (hf/analysis.py). 1 GPU — only the router-latency
# benchmark needs it; the curve plots read metrics.jsonl.
#
# Runs whatever has finished: each run dir is included only if it has a
# metrics.jsonl, so this is safe to submit while the matrix is still filling in.
#SBATCH --job-name=hf-figures
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

source "${CONFIG:-$SLURM_SUBMIT_DIR/hf/slurm/config.sh}"
source "$REPO_DIR/hf/runs.sh"
activate_env

# Comma-joined list of the run dirs under $1 that actually have metrics, for
# --runs; the labels are the basenames.
collect() {
    local base="$1"; shift
    local found=()
    for name in "$@"; do
        [[ -f "$base/$name/metrics.jsonl" ]] && found+=("$base/$name")
    done
    (IFS=,; echo "${found[*]}")
}

# Phase A: one figure set per size, comparing routers at seed 0.
for size in S M L; do
    names=()
    for router in "${ROUTERS_DEFAULT[@]}"; do names+=("${size}_${router}_s0"); done
    runs=$(collect "$OUT_DIR" "${names[@]}")
    if [[ -n "$runs" ]]; then
        echo "=== phase A figures: $size ==="
        python -m hf.analysis --mode phase_a --runs "$runs" --output-dir "$OUT_DIR/figures_$size"
    fi
done

# Router latency at Phase A's smallest geometry and at OLMoE's.
size_params S
python -m hf.analysis --mode latency --dim "$HIDDEN" --experts 8 --top-k 2 \
    --device cuda --output-dir "$OUT_DIR/figures_S"

# Phase B: healing curves across conditions.
heal_names=()
for entry in "${PHASE_B_RUNS[@]}"; do heal_names+=("${entry%% *}"); done
runs=$(collect "$HEAL_DIR" "${heal_names[@]}")
if [[ -n "$runs" ]]; then
    echo "=== phase B figures ==="
    python -m hf.analysis --mode healing --runs "$runs" --output-dir "$HEAL_DIR/figures"
    python -m hf.analysis --mode latency --dim 2048 --experts 64 --top-k 8 \
        --device cuda --output-dir "$HEAL_DIR/figures"
fi
