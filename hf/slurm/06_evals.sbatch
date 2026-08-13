#!/bin/bash
# lm-eval the Phase B milestone checkpoints. One array task per condition (4),
# each looping its milestones in sequence. 1 GPU.
#
# The milestone steps are read from the run's config.json rather than hardcoded,
# so they stay correct if NUM_TOKENS changes. heal_olmoe resolves them from
# MILESTONE_FRACS, rounded up to a --save-every boundary so the checkpoint
# exists; pruning keeps their model/ (it only strips trainer_state.pt).
#SBATCH --job-name=hf-evals
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail

source "${CONFIG:-$SLURM_SUBMIT_DIR/hf/slurm/config.sh}"
source "$REPO_DIR/hf/runs.sh"
activate_env

read -r name _condition _router <<< "${PHASE_B_RUNS[$SLURM_ARRAY_TASK_ID]}"
config="$HEAL_DIR/$name/config.json"
if [[ ! -f "$config" ]]; then
    echo "no $config — has $name run yet?"
    exit 0
fi

mapfile -t steps < <(python -c "
import json, sys
print('\n'.join(str(s) for s in json.load(open(sys.argv[1]))['milestone_steps']))" "$config")

for step in "${steps[@]}"; do
    ckpt=$(printf '%s/%s/step_%06d/model' "$HEAL_DIR" "$name" "$step")
    out=$(printf '%s/eval_%s_%06d.json' "$HEAL_DIR" "$name" "$step")
    if [[ ! -d "$ckpt" ]]; then
        echo "missing $ckpt, skipping"
        continue
    fi
    echo "=== $name @ step $step ==="
    python -m hf.eval_lm --model "$ckpt" --out "$out"
done
