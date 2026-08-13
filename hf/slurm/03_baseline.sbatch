#!/bin/bash
# Phase B hard gate: lm-eval the UNMODIFIED OLMoE-1B-7B checkpoint.
#
# Nothing in Phase B is interpretable until this reproduces the published
# numbers within ~1 point (references in hf/eval_lm.py's docstring: MMLU 5-shot
# ~54.1, HellaSwag ~80.0, ARC-C ~62.1, PIQA ~79.8, WinoGrande ~72.3). Compare by
# hand before letting the healing runs start.
#
# eval_lm writes after each task and skips tasks already present, so a job that
# hits the wall clock loses one task, not all seven.
#SBATCH --job-name=hf-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

source "${CONFIG:-$SLURM_SUBMIT_DIR/hf/slurm/config.sh}"
activate_env

mkdir -p "$HEAL_DIR"
python -m hf.eval_lm --model "$MODEL" --out "$HEAL_DIR/baseline_eval.json"

echo
echo "HARD GATE: compare the above against the published OLMoE-1B-7B-0924"
echo "numbers in hf/eval_lm.py. Do not proceed past ~1pt of drift."
