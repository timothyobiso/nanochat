#!/bin/bash
# Tokenize FineWeb-Edu into uint16 shards with the OLMoE tokenizer. CPU only.
#
# NOT resumable: hf/prepare_data.py writes manifest.json only after the last
# shard, so an interrupted run leaves orphan .bin files and must be redone. That
# is why DATA_TOKENS defaults to 15B rather than the plan's 30B — at the
# observed 1-3M tok/s this lands in 2-4h instead of 3-8h.
#SBATCH --job-name=hf-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

source "${CONFIG:-$SLURM_SUBMIT_DIR/hf/slurm/config.sh}"
activate_env

if [[ -f "$DATA_DIR/manifest.json" ]]; then
    echo "$DATA_DIR/manifest.json exists, nothing to do"
    exit 0
fi

python -m hf.prepare_data \
    --data-dir "$DATA_DIR" \
    --num-tokens "$DATA_TOKENS" \
    --val-tokens 50000000
