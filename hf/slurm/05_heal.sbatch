#!/bin/bash
# Phase B1: one array task per healing condition (4). 1 node x 8 GPUs.
#
# 5B tokens per condition is ~9h on an H100 node and roughly 23-32h on 48GB
# cards, so this is the stage the 24h cap actually bites. Same contract as
# 02_train: exit if DONE, otherwise `--resume auto`; e2e.sh chains SEGMENTS_B of
# these with --dependency=afterany.
#
# --param-dtype bfloat16 is required here: fp32 params + ZeRO-1 Adam needs
# ~63GB/GPU, bf16 ~35GB.
#SBATCH --job-name=hf-heal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail

source "${CONFIG:-$SLURM_SUBMIT_DIR/hf/slurm/config.sh}"
source "$REPO_DIR/hf/runs.sh"
activate_env

read -r name condition router <<< "${PHASE_B_RUNS[$SLURM_ARRAY_TASK_ID]}"

if [[ -f "$HEAL_DIR/$name/DONE" ]]; then
    echo "$name already complete"
    exit 0
fi

B0_REPORT="$HEAL_DIR/b0_report.json"
B0_ARGS=()
ROUTER_SEED=0
if [[ -f "$B0_REPORT" ]]; then
    B0_ARGS=(--b0-report "$B0_REPORT")
    ROUTER_SEED=$(python -c \
        'import sys; from hf.diagnose_router import best_seed_for; print(best_seed_for(sys.argv[1], sys.argv[2]))' \
        "$B0_REPORT" "$router")
fi

echo "=== $name (condition $condition, router $router, router-seed $ROUTER_SEED) ==="
torchrun --standalone --nproc_per_node="$GPUS" -m hf.heal_olmoe -- \
    --model "$MODEL" \
    --data-dir "$DATA_DIR" \
    --out-dir "$HEAL_DIR" \
    --run-name "$name" \
    --condition "$condition" \
    --router "$router" \
    --router-seed "$ROUTER_SEED" \
    --num-tokens "$NUM_TOKENS" \
    --param-dtype "$PARAM_DTYPE" \
    --resume auto \
    --run "$WANDB" \
    "${B0_ARGS[@]}"
