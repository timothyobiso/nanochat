"""
VSA-MoE integration for nanochat.

This module provides the MoE layer that replaces the MLP block in transformer layers,
using VSA-based routing instead of standard linear routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from nanochat.routers import HRRRouter, VSARouterConfig
from nanochat.moe_block import MoEAuxLosses


class MoELayer(nn.Module):
    """
    Mixture of Experts layer using VSA routing.
    
    Replaces standard MLP with multiple expert MLPs and a VSA router.
    Each expert is a relu^2 MLP matching nanochat's architecture.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Extract config values (will be added to GPTConfig)
        self.num_experts = getattr(config, 'num_experts', 8)
        self.num_experts_per_tok = getattr(config, 'num_experts_per_tok', 2)
        self.hidden_size = config.n_embd
        self.expert_hidden_size = 4 * config.n_embd  # Match MLP hidden size
        
        # VSA Router configuration
        router_config = VSARouterConfig(
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            vsa_dim=self.hidden_size,  # Use same dim as hidden size
            learn_expert_labels=True,
            learn_expert_signatures=False,
            use_input_projection=True,
            router_temperature=1.0,
            normalize_inputs=True,
        )
        
        # Create HRR router
        self.router = HRRRouter(router_config)
        
        # Create expert MLPs (matching nanochat's relu^2 architecture)
        self.experts = nn.ModuleList([
            self._create_expert_mlp() for _ in range(self.num_experts)
        ])
    
    def _create_expert_mlp(self):
        """Create a single expert MLP matching nanochat's architecture."""
        return nn.ModuleDict({
            'c_fc': nn.Linear(self.hidden_size, self.expert_hidden_size, bias=False),
            'c_proj': nn.Linear(self.expert_hidden_size, self.hidden_size, bias=False),
        })
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape [B, T, C]
        
        Returns:
            output: Output tensor of shape [B, T, C]
            aux_outputs: Dict containing routing info for auxiliary losses
        """
        B, T, C = x.shape
        x_flat = x.view(B * T, C)  # Flatten batch and sequence dimensions
        
        # Get routing weights and indices from VSA router
        weights, indices, scores = self.router(x_flat)  # [N, k], [N, k], [N, K]
        
        # Simple expert dispatch (fine for now, can optimize later)
        output = torch.zeros_like(x_flat)
        
        for k in range(self.num_experts):
            # Find tokens routed to expert k
            mask = (indices == k).any(dim=-1)
            if not mask.any():
                continue
                
            # Get tokens for this expert
            expert_input = x_flat[mask]
            
            # Apply expert MLP (relu^2 activation)
            h = self.experts[k]['c_fc'](expert_input)
            h = F.relu(h).square()  # relu^2 activation
            expert_output = self.experts[k]['c_proj'](h)
            
            # Get weights for tokens routed to this expert
            # Sum weights if a token is routed to same expert multiple times
            token_weights = torch.zeros(mask.sum(), device=x.device)
            mask_indices = mask.nonzero(as_tuple=True)[0]
            
            for i, idx in enumerate(mask_indices):
                # Sum weights for all slots where this token goes to expert k
                expert_mask = indices[idx] == k
                token_weights[i] = weights[idx][expert_mask].sum()
            
            # Weighted accumulation
            output[mask] += token_weights.unsqueeze(-1) * expert_output
        
        # Reshape back to original shape
        output = output.view(B, T, C)
        
        # Collect auxiliary outputs for loss computation
        aux_outputs = {
            'router_weights': weights,  # [N, k]
            'router_indices': indices,  # [N, k] 
            'router_scores': scores,    # [N, K] full scores for all experts
        }
        
        return output, aux_outputs


class MoEBlock(nn.Module):
    """
    Transformer block with MoE instead of MLP.
    
    This wraps the attention and MoE layers together, matching nanochat's Block structure.
    """
    
    def __init__(self, config, layer_idx):
        super().__init__()
        # Import CausalSelfAttention from gpt module
        from nanochat.gpt import CausalSelfAttention
        
        self.attn = CausalSelfAttention(config, layer_idx)
        self.moe = MoELayer(config)
        self.layer_idx = layer_idx
    
    def forward(self, x, cos_sin, window_size, kv_cache):
        from nanochat.gpt import norm
        
        # Self-attention
        x = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
        
        # MoE feedforward
        h = norm(x)
        moe_out, aux_outputs = self.moe(h)
        x = x + moe_out
        
        # Store aux outputs as attribute for loss collection
        self.aux_outputs = aux_outputs
        
        return x


def compute_moe_auxiliary_losses(model, config):
    """
    Compute auxiliary losses for all MoE layers in the model.
    
    Args:
        model: GPT model with MoE blocks
        config: Model configuration
    
    Returns:
        Dictionary of auxiliary losses
    """
    aux_losses = {}
    total_load_balance_loss = 0.0
    total_z_loss = 0.0
    
    # Collect losses from all MoE blocks
    num_moe_blocks = 0
    for i, block in enumerate(model.transformer.h):
        if hasattr(block, 'aux_outputs'):
            weights = block.aux_outputs['router_weights']
            indices = block.aux_outputs['router_indices']
            scores = block.aux_outputs['router_scores']
            
            # Load balance loss
            lb_loss = MoEAuxLosses.load_balance_loss(
                weights, indices, config.num_experts
            )
            total_load_balance_loss += lb_loss
            
            # Router z-loss (prevents score explosion)
            z_loss = MoEAuxLosses.router_z_loss(scores)
            total_z_loss += z_loss
            
            num_moe_blocks += 1
    
    # Average losses across MoE blocks
    if num_moe_blocks > 0:
        aux_losses['load_balance_loss'] = total_load_balance_loss / num_moe_blocks
        aux_losses['router_z_loss'] = total_z_loss / num_moe_blocks
    
    return aux_losses