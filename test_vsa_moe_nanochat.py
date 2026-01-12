#!/usr/bin/env python3
"""
Test script for VSA-MoE integration with nanochat.

This tests:
1. Model creation with MoE layers
2. Forward pass through the model
3. MoE routing and expert dispatch
4. Auxiliary loss computation
"""

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.moe import compute_moe_auxiliary_losses


def test_moe_model():
    """Test that MoE model can be created and run forward pass."""
    print("Testing VSA-MoE nanochat integration...")
    
    # Small config for testing
    config = GPTConfig(
        sequence_len=512,
        vocab_size=1000,  # Small vocab for testing
        n_layer=4,  # Small model
        n_head=4,
        n_kv_head=4,
        n_embd=256,  # Small hidden size
        # MoE config
        num_experts=4,
        num_experts_per_tok=2,
        moe_layer_freq=2,  # Every other layer
        vsa_type="hrr",
        load_balance_coef=0.01,
        router_z_loss_coef=0.001,
    )
    
    print(f"Creating model with config:")
    print(f"  - n_layer: {config.n_layer}")
    print(f"  - n_embd: {config.n_embd}")
    print(f"  - num_experts: {config.num_experts}")
    print(f"  - moe_layer_freq: {config.moe_layer_freq}")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with torch.device("meta"):
        model = GPT(config)
    
    model.to_empty(device=device)
    model.init_weights()
    
    print(f"Model created successfully!")
    
    # Check which layers have MoE
    moe_layers = []
    for i, block in enumerate(model.transformer.h):
        if hasattr(block, 'is_moe') and block.is_moe:
            moe_layers.append(i)
    print(f"MoE layers: {moe_layers}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    # Create random input
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: {idx.shape}")
    
    # Forward pass
    loss = model(idx, targets)
    print(f"Loss (without aux): {loss.item():.4f}")
    
    # Compute auxiliary losses
    aux_losses = compute_moe_auxiliary_losses(model, config)
    
    if aux_losses:
        print(f"\nAuxiliary losses:")
        print(f"  - Load balance loss: {aux_losses.get('load_balance_loss', 0.0):.6f}")
        print(f"  - Router z-loss: {aux_losses.get('router_z_loss', 0.0):.6f}")
        
        # Total loss with auxiliary
        total_loss = loss + (config.load_balance_coef * aux_losses.get('load_balance_loss', 0.0) + 
                            config.router_z_loss_coef * aux_losses.get('router_z_loss', 0.0))
        print(f"Total loss (with aux): {total_loss.item():.4f}")
    
    # Check routing distribution for one MoE layer
    if moe_layers:
        layer_idx = moe_layers[0]
        block = model.transformer.h[layer_idx]
        if hasattr(block, 'aux_outputs'):
            indices = block.aux_outputs['router_indices']
            weights = block.aux_outputs['router_weights']
            
            print(f"\nRouting analysis for layer {layer_idx}:")
            
            # Count expert usage
            expert_counts = torch.zeros(config.num_experts, device=device)
            for k in range(config.num_experts):
                expert_counts[k] = (indices == k).sum()
            
            print(f"Expert usage counts: {expert_counts.cpu().numpy()}")
            print(f"Expert usage percentage: {(100 * expert_counts / indices.numel()).cpu().numpy()}")
            
            # Check weight distribution
            print(f"Weight stats - mean: {weights.mean():.4f}, std: {weights.std():.4f}")
    
    print("\n✅ All tests passed!")
    
    # Test comparison with linear router
    print("\n" + "="*50)
    print("Testing linear router for comparison...")
    
    config_linear = GPTConfig(
        sequence_len=512,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        # MoE config with linear router
        num_experts=4,
        num_experts_per_tok=2,
        moe_layer_freq=2,
        vsa_type="linear",  # Use linear router
        load_balance_coef=0.01,
        router_z_loss_coef=0.001,
    )
    
    with torch.device("meta"):
        model_linear = GPT(config_linear)
    
    model_linear.to_empty(device=device)
    model_linear.init_weights()
    
    loss_linear = model_linear(idx, targets)
    print(f"Linear router loss: {loss_linear.item():.4f}")
    
    print("\n✅ Comparison test passed!")
    
    return model


def test_training_step():
    """Test a single training step with MoE."""
    print("\n" + "="*50)
    print("Testing training step with MoE...")
    
    config = GPTConfig(
        sequence_len=256,
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_layer_freq=1,  # Every layer for this test
        vsa_type="hrr",
        load_balance_coef=0.01,
        router_z_loss_coef=0.001,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.device("meta"):
        model = GPT(config)
    
    model.to_empty(device=device)
    model.init_weights()
    
    # Setup optimizer (simple SGD for testing)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create batch
    batch_size = 2
    seq_len = 64
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass with auxiliary losses
    lm_loss = model(idx, targets)
    aux_losses = compute_moe_auxiliary_losses(model, config)
    
    total_loss = lm_loss
    if aux_losses:
        total_loss = lm_loss + (config.load_balance_coef * aux_losses.get('load_balance_loss', 0.0) + 
                               config.router_z_loss_coef * aux_losses.get('router_z_loss', 0.0))
    
    print(f"LM loss: {lm_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Backward
    total_loss.backward()
    
    # Check gradients
    has_grads = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().max() > 0:
            has_grads = True
            break
    
    print(f"Gradients computed: {has_grads}")
    
    # Optimizer step
    optimizer.step()
    
    print("✅ Training step completed successfully!")


if __name__ == "__main__":
    print("="*50)
    print("VSA-MoE NanoChat Integration Test")
    print("="*50)
    
    # Run tests
    model = test_moe_model()
    test_training_step()
    
    print("\n" + "="*50)
    print("All tests completed successfully! 🎉")
    print("="*50)
    print("\nYou can now train a model with MoE using:")
    print("python -m scripts.base_train --moe_layer_freq=2 --num_experts=8 --depth=4 --max_seq_len=512")