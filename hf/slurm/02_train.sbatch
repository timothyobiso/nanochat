#!/bin/bash
# Phase A: one array task per run in PHASE_A_RUNS (30 runs). 1 node x 8 GPUs.
#
# Resumable and chained: the task exits immediately if the run is already DONE,
# otherwise `--resume auto` continues from the newest checkpoint. e2e.sh submits
# this script SEGMENTS_A times with --dependency=afterany, so a run too long for
# one 24h slot (size L) finishes across consecutive jobs.
#SBATCH --job-name=hf-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail

source "${CONFIG:-$SLURM_SUBMIT_DIR/hf/slurm/config.sh}"
source "$REPO_DIR/hf/runs.sh"
activate_env

read -r name size router seed extra <<< "${PHASE_A_RUNS[$SLURM_ARRAY_TASK_ID]}"
size_params "$size"

if [[ -f "$OUT_DIR/$name/DONE" ]]; then
    echo "$name already complete"
    exit 0
fi

echo "=== $name (size $size, router $router, seed $seed) ${extra:+[$extra]} ==="
# shellcheck disable=SC2086  # $extra is a deliberate flag list
torchrun --standalone --nproc_per_node="$GPUS" -m hf.train_olmoe -- \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --run-name "$name" \
    --router "$router" \
    --router-seed "$seed" \
    --seed "$seed" \
    --hidden-size "$HIDDEN" \
    --num-layers "$LAYERS" \
    --num-heads "$HEADS" \
    --lr "$LR" \
    --device-batch-size "$DEV_BS" \
    --save-every "$SAVE_EVERY_A" \
    --resume auto \
    --run "$WANDB" \
    $extra
