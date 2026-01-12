#!/bin/bash

# Training script for VSA-MoE integrated with nanochat
# This demonstrates how to train a small model with VSA-based MoE routing

# Setup environment
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p $NANOCHAT_BASE_DIR

# Wandb setup (optional)
# Set WANDB_RUN environment variable to enable wandb logging
# Example: WANDB_RUN=vsa_moe_test ./train_vsa_moe.sh
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN="dummy"  # "dummy" skips wandb logging
fi

echo "Setting up VSA-MoE training..."
echo "=================================================="
echo "WANDB_RUN: $WANDB_RUN"
echo "Base directory: $NANOCHAT_BASE_DIR"

# Python venv setup (if needed)
if [ ! -d ".venv" ]; then
    echo "Setting up Python environment..."
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv
    uv sync
fi
source .venv/bin/activate

# Setup tokenizer and data
if [ ! -f "tokenizer.pkl" ]; then
    echo "Tokenizer not found. Setting up tokenizer and data..."
    echo "------------------------------------------------"
    
    # Download initial data shards for tokenizer training
    echo "Downloading data shards..."
    uv run python -m nanochat.dataset -n 8
    
    # Train tokenizer (smaller vocab for faster training)
    echo "Training tokenizer..."
    uv run python -m scripts.tok_train --max_chars=500000000 --vocab_size=32768
    
    # Evaluate tokenizer
    echo "Evaluating tokenizer..."
    uv run python -m scripts.tok_eval
    
    # Download more data shards for training (in background)
    echo "Downloading additional data shards for training..."
    uv run python -m nanochat.dataset -n 30 &
    DATASET_DOWNLOAD_PID=$!
else
    echo "Tokenizer found, skipping setup."
fi

# Wait for background downloads if any
if [ ! -z "$DATASET_DOWNLOAD_PID" ]; then
    echo "Waiting for dataset download to complete..."
    wait $DATASET_DOWNLOAD_PID
fi

echo ""
echo "Training a small 124M model with VSA-MoE routing..."
echo "=================================================="

# Small model with MoE (approximately 124M params with experts)
uv run python -m scripts.base_train \
    --run="$WANDB_RUN" \
    --depth=8 \
    --aspect_ratio=64 \
    --head_dim=64 \
    --max_seq_len=1024 \
    --moe_layer_freq=2 \
    --num_experts=8 \
    --num_experts_per_tok=2 \
    --vsa_type="hrr" \
    --load_balance_coef=0.01 \
    --router_z_loss_coef=0.001 \
    --device_batch_size=8 \
    --total_batch_size=32768 \
    --num_iterations=1000 \
    --eval_every=100 \
    --sample_every=200 \
    --model_tag="vsa_moe_8x124M"

# To compare with a linear router, change --vsa_type="linear"
# To compare with dense model, set --moe_layer_freq=0