#!/bin/bash
# Phase B0: zero-training router diagnostics on pretrained OLMoE. 1 GPU.
#
# Runs agreement -> seeds -> swap -> distill. diagnose_router rewrites the report
# after each mode and skips modes already present in it, so chaining this script
# resumes at the first unfinished mode. The 6.9B model in bf16 is ~14GB, well
# inside one 48GB card.
#
# Feeds B1: the Hungarian perms (report.agreement) and the best FPE base seed
# (report.seed_search).
#SBATCH --job-name=hf-b0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

source "${CONFIG:-$SLURM_SUBMIT_DIR/hf/slurm/config.sh}"
activate_env

mkdir -p "$HEAL_DIR"
python -m hf.diagnose_router \
    --model "$MODEL" \
    --data-dir "$DATA_DIR" \
    --device cuda \
    --per-layer-swap \
    --distill-ppl \
    --out "$HEAL_DIR/b0_report.json"
