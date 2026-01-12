#!/bin/bash

# Training script for VSA-MoE integrated with nanochat
# This demonstrates how to train a small model with VSA-based MoE routing

echo "Training a small 124M model with VSA-MoE routing..."
echo "=================================================="

# Small model with MoE (approximately 124M params with experts)
uv run python -m scripts.base_train \
    --run="vsa_moe_test" \
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