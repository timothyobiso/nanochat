"""
VSA Mixture-of-Experts Block and Model Integration

Provides:
- SwiGLU expert MLP
- VSAMoEBlock combining router and experts
- Auxiliary losses for load balancing
- Integration utilities for OLMo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field

from routers import VSARouterConfig, create_router


@dataclass
class VSAMoEConfig:
    """Full configuration for VSA-MoE model."""
    
    # Base model config (OLMo-style)
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 100352
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    
    # MoE config
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_layer_freq: int = 2  # Apply MoE every N layers (0 = no MoE)
    expert_intermediate_size: Optional[int] = None  # Defaults to intermediate_size
    shared_expert: bool = False  # Add shared expert alongside routed experts
    
    # VSA router config
    vsa_type: str = "hrr"  # "hrr", "fpe", "resonator", "linear"
    vsa_dim: Optional[int] = None
    learn_expert_labels: bool = True
    learn_expert_signatures: bool = False
    use_input_projection: bool = True
    router_temperature: float = 1.0
    
    # Resonator-specific
    resonator_iterations: int = 3
    resonator_sharpness: float = 10.0
    
    # Loss coefficients
    load_balance_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    label_orthogonality_coef: float = 0.0
    
    def __post_init__(self):
        if self.vsa_dim is None:
            self.vsa_dim = self.hidden_size
        if self.expert_intermediate_size is None:
            self.expert_intermediate_size = self.intermediate_size
    
    def get_router_config(self) -> VSARouterConfig:
        """Convert to router-specific config."""
        return VSARouterConfig(
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            vsa_dim=self.vsa_dim,
            learn_expert_labels=self.learn_expert_labels,
            learn_expert_signatures=self.learn_expert_signatures,
            use_input_projection=self.use_input_projection,
            router_temperature=self.router_temperature,
            resonator_iterations=self.resonator_iterations,
            resonator_sharpness=self.resonator_sharpness,
        )


class SwiGLUMLP(nn.Module):
    """
    SwiGLU MLP block (used as expert).
    
    y = (Swish(xW_gate) ⊙ xW_up) W_down
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [*, hidden_size]
        Returns:
            [*, hidden_size]
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VSAMoEBlock(nn.Module):
    """
    VSA Mixture-of-Experts block.
    
    Combines VSA router with multiple expert MLPs.
    """
    
    def __init__(self, config: VSAMoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        
        # Create router
        router_config = config.get_router_config()
        self.router = create_router(router_config, config.vsa_type)
        
        # Create experts
        self.experts = nn.ModuleList([
            SwiGLUMLP(config.hidden_size, config.expert_intermediate_size)
            for _ in range(config.num_experts)
        ])
        
        # Optional shared expert (always active)
        if config.shared_expert:
            self.shared_expert = SwiGLUMLP(
                config.hidden_size, 
                config.expert_intermediate_size
            )
        else:
            self.shared_expert = None
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through MoE block.
        
        Args:
            x: Input tensor of shape [B, T, d]
        
        Returns:
            output: Output tensor of shape [B, T, d]
            aux_outputs: Dict containing routing info for losses
        """
        B, T, d = x.shape
        x_flat = x.view(B * T, d)  # [N, d] where N = B*T
        
        # Route
        weights, indices, scores = self.router(x_flat)  # [N, k], [N, k], [N, K]
        
        # Compute expert outputs
        # Method 1: Loop (simple but slow)
        # output = self._forward_loop(x_flat, weights, indices)
        
        # Method 2: Grouped computation (more efficient)
        output = self._forward_grouped(x_flat, weights, indices)
        
        # Add shared expert if present
        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)
        
        output = output.view(B, T, d)
        
        # Auxiliary outputs for losses
        aux_outputs = {
            'router_weights': weights,  # [N, k]
            'router_indices': indices,  # [N, k]
            'router_scores': scores,    # [N, K]
        }
        
        return output, aux_outputs
    
    def _forward_loop(self, x: Tensor, weights: Tensor, 
                      indices: Tensor) -> Tensor:
        """Simple loop-based forward (for clarity)."""
        N, d = x.shape
        output = torch.zeros_like(x)
        
        for n in range(N):
            for j in range(self.top_k):
                expert_idx = indices[n, j].item()
                w = weights[n, j]
                output[n] += w * self.experts[expert_idx](x[n:n+1]).squeeze(0)
        
        return output
    
    def _forward_grouped(self, x: Tensor, weights: Tensor, 
                         indices: Tensor) -> Tensor:
        """
        Grouped forward pass (more efficient).
        
        Groups tokens by their selected experts and processes in batches.
        """
        N, d = x.shape
        device = x.device
        dtype = x.dtype
        
        output = torch.zeros(N, d, device=device, dtype=dtype)
        
        for k in range(self.num_experts):
            # Find all (token, slot) pairs routed to expert k
            mask = indices == k  # [N, top_k]
            
            if not mask.any():
                continue
            
            # Get token indices and their weights for this expert
            token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]
            
            if len(token_indices) == 0:
                continue
            
            # Extract tokens for this expert
            expert_input = x[token_indices]  # [n_k, d]
            
            # Compute expert output
            expert_output = self.experts[k](expert_input)  # [n_k, d]
            
            # Get weights for this expert (sum across slots if token routes to same expert multiple times)
            expert_weights = (weights * mask.float()).sum(dim=1)[token_indices]  # [n_k]
            
            # Weighted accumulation
            output.index_add_(
                0, 
                token_indices, 
                expert_output * expert_weights.unsqueeze(-1)
            )
        
        return output


class MoEAuxLosses:
    """
    Auxiliary losses for MoE training.
    """
    
    @staticmethod
    def load_balance_loss(weights: Tensor, indices: Tensor, 
                          num_experts: int) -> Tensor:
        """
        Load balancing loss to encourage uniform expert usage.
        
        L = K * Σ_k f_k * p_k
        
        where:
        - f_k = fraction of tokens routed to expert k
        - p_k = average routing probability for expert k
        
        Args:
            weights: Routing weights [N, k]
            indices: Expert indices [N, k]
            num_experts: Total number of experts
        
        Returns:
            Scalar loss
        """
        N = weights.shape[0]
        device = weights.device
        
        # f_k: fraction of tokens assigned to each expert
        # Count how many times each expert is selected
        expert_counts = torch.zeros(num_experts, device=device)
        for k in range(num_experts):
            expert_counts[k] = (indices == k).sum()
        
        f = expert_counts / (N * weights.shape[1])  # Normalize by total assignments
        
        # p_k: average routing probability per expert
        # Sum of weights for each expert divided by N
        p = torch.zeros(num_experts, device=device)
        for k in range(num_experts):
            mask = indices == k
            if mask.any():
                p[k] = weights[mask].sum() / N
        
        # Load balance loss
        loss = num_experts * (f * p).sum()
        
        return loss
    
    @staticmethod
    def router_z_loss(scores: Tensor) -> Tensor:
        """
        Router z-loss to prevent logit explosion.
        
        L = (1/N) * Σ_n log²(Σ_k exp(s_{n,k}))
        
        Args:
            scores: Router logits [N, K]
        
        Returns:
            Scalar loss
        """
        log_z = torch.logsumexp(scores, dim=-1)  # [N]
        return (log_z ** 2).mean()
    
    @staticmethod
    def label_orthogonality_loss(labels: Tensor) -> Tensor:
        """
        Encourage orthogonal expert labels for better capacity.
        
        L = ||L @ L^T - I||_F^2
        
        Args:
            labels: Expert labels [K, d]
        
        Returns:
            Scalar loss
        """
        K = labels.shape[0]
        labels_norm = F.normalize(labels, dim=-1)
        gram = labels_norm @ labels_norm.T  # [K, K]
        target = torch.eye(K, device=labels.device)
        return ((gram - target) ** 2).sum()
    
    @staticmethod
    def compute_all_losses(
        aux_outputs: Dict[str, Tensor],
        router: nn.Module,
        config: VSAMoEConfig
    ) -> Dict[str, Tensor]:
        """
        Compute all auxiliary losses.
        
        Args:
            aux_outputs: Dict from MoE forward pass
            router: Router module (for accessing labels)
            config: Model config
        
        Returns:
            Dict of individual losses
        """
        losses = {}
        
        weights = aux_outputs['router_weights']
        indices = aux_outputs['router_indices']
        scores = aux_outputs['router_scores']
        
        # Load balance loss
        if config.load_balance_coef > 0:
            losses['load_balance'] = (
                config.load_balance_coef * 
                MoEAuxLosses.load_balance_loss(weights, indices, config.num_experts)
            )
        
        # Z-loss
        if config.router_z_loss_coef > 0:
            losses['router_z'] = (
                config.router_z_loss_coef * 
                MoEAuxLosses.router_z_loss(scores)
            )
        
        # Label orthogonality (only for VSA routers with learnable labels)
        if config.label_orthogonality_coef > 0 and hasattr(router, 'labels'):
            losses['label_ortho'] = (
                config.label_orthogonality_coef * 
                MoEAuxLosses.label_orthogonality_loss(router.labels)
            )
        
        return losses


class MoEMetrics:
    """
    Metrics for monitoring MoE routing quality.
    """
    
    @staticmethod
    def expert_utilization(indices: Tensor, num_experts: int) -> Tensor:
        """
        Compute fraction of tokens routed to each expert.
        
        Returns:
            Tensor of shape [K] with utilization per expert
        """
        counts = torch.zeros(num_experts, device=indices.device)
        for k in range(num_experts):
            counts[k] = (indices == k).sum()
        return counts / counts.sum()
    
    @staticmethod
    def utilization_entropy(indices: Tensor, num_experts: int) -> Tensor:
        """
        Entropy of expert utilization (higher = more balanced).
        
        Max entropy = log(K) for uniform distribution.
        """
        util = MoEMetrics.expert_utilization(indices, num_experts)
        util = util + 1e-10  # Avoid log(0)
        entropy = -(util * util.log()).sum()
        return entropy
    
    @staticmethod
    def routing_confidence(weights: Tensor) -> Tensor:
        """
        Average max routing weight (higher = more confident routing).
        """
        return weights.max(dim=-1).values.mean()


# Integration with OLMo-style transformer

class VSAMoETransformerBlock(nn.Module):
    """
    Full transformer block with VSA-MoE.
    
    Replaces dense MLP with MoE block.
    """
    
    def __init__(
        self, 
        config: VSAMoEConfig,
        attention_module: nn.Module,  # Assume this is provided
    ):
        super().__init__()
        self.config = config
        
        # Attention (provided externally)
        self.attention = attention_module
        
        # Norms
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # MoE block
        self.moe = VSAMoEBlock(config)
    
    def forward(
        self, 
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        **kwargs
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass.
        
        Args:
            x: [B, T, d]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
        
        Returns:
            output: [B, T, d]
            aux_outputs: Dict with MoE routing info
        """
        # Attention with residual
        h = x + self.attention(
            self.attn_norm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )
        
        # MoE with residual
        moe_out, aux_outputs = self.moe(self.ffn_norm(h))
        output = h + moe_out
        
        return output, aux_outputs


def convert_dense_to_moe(
    dense_mlp: SwiGLUMLP,
    config: VSAMoEConfig,
    init_from_dense: bool = True
) -> VSAMoEBlock:
    """
    Convert a dense MLP to MoE block.
    
    Optionally initializes all experts from the dense MLP weights.
    
    Args:
        dense_mlp: Original dense MLP
        config: MoE config
        init_from_dense: Whether to copy dense weights to all experts
    
    Returns:
        MoE block
    """
    moe = VSAMoEBlock(config)
    
    if init_from_dense:
        for expert in moe.experts:
            expert.gate_proj.weight.data.copy_(dense_mlp.gate_proj.weight.data)
            expert.up_proj.weight.data.copy_(dense_mlp.up_proj.weight.data)
            expert.down_proj.weight.data.copy_(dense_mlp.down_proj.weight.data)
    
    return moe


def get_moe_layer_indices(num_layers: int, moe_freq: int) -> List[int]:
    """
    Get indices of layers that should be MoE.
    
    Args:
        num_layers: Total number of layers
        moe_freq: Apply MoE every N layers (starting from layer moe_freq)
    
    Returns:
        List of layer indices
    """
    if moe_freq <= 0:
        return []
    return [i for i in range(moe_freq, num_layers + 1, moe_freq)]
