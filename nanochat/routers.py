"""
VSA-based Routers for Mixture-of-Experts

Three routing approaches:
1. HRRRouter: Holographic Reduced Representations (circular convolution)
2. FPERouter: Fractional Power Encoding (position-based)
3. ResonatorRouter: Iterative resonator network cleanup
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
from dataclasses import dataclass

from vsa_ops import VSAOps, FPEOps, ResonatorOps


@dataclass
class VSARouterConfig:
    """Configuration for VSA routers."""
    num_experts: int = 8
    num_experts_per_tok: int = 2
    hidden_size: int = 4096
    vsa_dim: Optional[int] = None  # Defaults to hidden_size
    
    # Learning options
    learn_expert_labels: bool = True
    learn_expert_signatures: bool = False
    use_input_projection: bool = True
    
    # Routing options
    router_temperature: float = 1.0
    normalize_inputs: bool = True
    
    # Resonator-specific
    resonator_iterations: int = 3
    resonator_sharpness: float = 10.0
    
    # FPE-specific
    use_continuous_routing: bool = False
    
    def __post_init__(self):
        if self.vsa_dim is None:
            self.vsa_dim = self.hidden_size


class HRRRouter(nn.Module):
    """
    Holographic Reduced Representations Router.
    
    Uses circular convolution binding and superposition to create
    a constant-memory routing mechanism.
    
    Router memory: R = Σ_k (E_k ⊗ L_k)
    Routing: scores_k = sim(x ⊗^{-1} R, E_k)
    """
    
    def __init__(self, config: VSARouterConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.dim = config.vsa_dim
        self.temperature = config.router_temperature
        
        # Expert labels (learnable keys for each expert)
        if config.learn_expert_labels:
            self.labels = nn.Parameter(
                torch.randn(self.num_experts, self.dim) * 0.02
            )
        else:
            self.register_buffer(
                'labels',
                F.normalize(torch.randn(self.num_experts, self.dim), dim=-1)
            )
        
        # Expert signatures (used for similarity comparison after unbinding)
        if config.learn_expert_signatures:
            self.signatures = nn.Parameter(
                torch.randn(self.num_experts, self.dim) * 0.02
            )
        else:
            self.register_buffer(
                'signatures',
                F.normalize(torch.randn(self.num_experts, self.dim), dim=-1)
            )
        
        # Optional input projection
        self.use_proj = config.use_input_projection
        if self.use_proj:
            self.proj = nn.Linear(config.hidden_size, self.dim, bias=False)
            nn.init.xavier_uniform_(self.proj.weight)
        
        # Cache for router memory (recomputed each forward in training)
        self._cached_router_memory = None
    
    @property
    def labels_normalized(self) -> Tensor:
        """Get normalized labels."""
        return F.normalize(self.labels, dim=-1)
    
    @property
    def signatures_normalized(self) -> Tensor:
        """Get normalized signatures."""
        return F.normalize(self.signatures, dim=-1)
    
    def compute_router_memory(self) -> Tensor:
        """
        Compute superposition router memory.
        
        R = Σ_k (E_k ⊗ L_k)
        
        Returns:
            Router memory tensor of shape [d]
        """
        L = self.labels_normalized  # [K, d]
        E = self.signatures_normalized  # [K, d]
        
        # Bind each signature with its label
        bindings = VSAOps.bind(E, L)  # [K, d]
        
        # Bundle into single superposition
        return bindings.sum(dim=0)  # [d]
    
    def get_router_memory(self) -> Tensor:
        """Get router memory, using cache in eval mode."""
        if self.training or self._cached_router_memory is None:
            self._cached_router_memory = self.compute_router_memory()
        return self._cached_router_memory
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass: compute routing weights and indices.
        
        Args:
            x: Input tensor of shape [N, d] (flattened tokens)
        
        Returns:
            weights: Routing weights of shape [N, top_k]
            indices: Selected expert indices of shape [N, top_k]
            scores: Full score matrix of shape [N, K] (for aux losses)
        """
        N = x.shape[0]
        
        # Project input (optional)
        if self.use_proj:
            x = self.proj(x)
        
        # Normalize input
        if self.config.normalize_inputs:
            x = F.normalize(x, dim=-1)
        
        # Get router memory
        R = self.get_router_memory()  # [d]
        
        # Unbind: S = x ⊗^{-1} R
        # Broadcast R across batch
        S = VSAOps.unbind(x, R.unsqueeze(0).expand(N, -1))  # [N, d]
        
        # Compute similarity with expert signatures
        E = self.signatures_normalized  # [K, d]
        scores = VSAOps.similarity(S, E)  # [N, K]
        
        # Top-k selection
        top_k_scores, indices = scores.topk(self.top_k, dim=-1)  # [N, k], [N, k]
        
        # Softmax over selected experts
        weights = F.softmax(top_k_scores / self.temperature, dim=-1)  # [N, k]
        
        return weights, indices, scores


class FPERouter(nn.Module):
    """
    Fractional Power Encoding Router.
    
    Uses position-based encoding where each expert has a learned
    scalar position. Labels are B^{p_k} for base vector B.
    
    Advantages:
    - Compact parameterization (K scalars vs K×d vectors)
    - Smooth interpolation between experts
    - Can learn semantic similarity via position proximity
    """
    
    def __init__(self, config: VSARouterConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.dim = config.vsa_dim
        self.temperature = config.router_temperature
        
        # Base vector phases (fixed random)
        self.register_buffer(
            'theta',
            FPEOps.create_base_vector(self.dim)
        )
        
        # Expert positions (learnable scalars)
        # Initialize evenly spaced in [-1, 1]
        self.positions = nn.Parameter(
            torch.linspace(-1, 1, self.num_experts)
        )
        
        # Expert signatures (for retrieval comparison)
        if config.learn_expert_signatures:
            self.signatures = nn.Parameter(
                torch.randn(self.num_experts, self.dim) * 0.02
            )
        else:
            self.register_buffer(
                'signatures',
                F.normalize(torch.randn(self.num_experts, self.dim), dim=-1)
            )
        
        # Optional: continuous position decoder
        if config.use_continuous_routing:
            self.pos_decoder = nn.Sequential(
                nn.Linear(config.hidden_size, self.dim // 4),
                nn.ReLU(),
                nn.Linear(self.dim // 4, 1)
            )
        else:
            self.pos_decoder = None
        
        # Input projection
        self.use_proj = config.use_input_projection
        if self.use_proj:
            self.proj = nn.Linear(config.hidden_size, self.dim, bias=False)
    
    def get_labels(self) -> Tensor:
        """
        Compute position-encoded labels.
        
        L_k = B^{p_k}
        
        Returns:
            Labels tensor of shape [K, d]
        """
        return FPEOps.power_encode(self.theta, self.positions)
    
    @property
    def signatures_normalized(self) -> Tensor:
        return F.normalize(self.signatures, dim=-1)
    
    def compute_router_memory(self) -> Tensor:
        """
        Compute router memory with FPE labels.
        
        R = Σ_k (E_k ⊗ B^{p_k})
        """
        L = self.get_labels()  # [K, d]
        E = self.signatures_normalized  # [K, d]
        bindings = VSAOps.bind(E, L)
        return bindings.sum(dim=0)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with FPE routing.
        
        Args:
            x: Input tensor of shape [N, d]
        
        Returns:
            weights: [N, top_k]
            indices: [N, top_k]
            scores: [N, K]
        """
        N = x.shape[0]
        
        if self.use_proj:
            x_proj = self.proj(x)
        else:
            x_proj = x
        
        if self.config.normalize_inputs:
            x_proj = F.normalize(x_proj, dim=-1)
        
        if self.pos_decoder is not None:
            # Continuous routing: predict position and use distance-based scores
            pred_pos = self.pos_decoder(x).squeeze(-1)  # [N]
            dists = (pred_pos.unsqueeze(1) - self.positions.unsqueeze(0)) ** 2  # [N, K]
            scores = -dists  # Negative distance as score
        else:
            # Discrete routing via unbinding (same as HRR)
            R = self.compute_router_memory()
            S = VSAOps.unbind(x_proj, R.unsqueeze(0).expand(N, -1))
            scores = VSAOps.similarity(S, self.signatures_normalized)
        
        top_k_scores, indices = scores.topk(self.top_k, dim=-1)
        weights = F.softmax(top_k_scores / self.temperature, dim=-1)
        
        return weights, indices, scores


class ResonatorRouter(nn.Module):
    """
    Resonator Network Router.
    
    Uses iterative cleanup dynamics to converge to the correct expert(s).
    More robust to noise and can handle distributed retrieval.
    
    Iteration: z^{t+1} = x ⊗^{-1} (R ⊗ cleanup(z^t))
    """
    
    def __init__(self, config: VSARouterConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.dim = config.vsa_dim
        self.iterations = config.resonator_iterations
        self.sharpness = config.resonator_sharpness
        self.temperature = config.router_temperature
        
        # Expert labels
        if config.learn_expert_labels:
            self.labels = nn.Parameter(
                torch.randn(self.num_experts, self.dim) * 0.02
            )
        else:
            self.register_buffer(
                'labels',
                F.normalize(torch.randn(self.num_experts, self.dim), dim=-1)
            )
        
        # Expert signatures (also serve as cleanup codebook)
        if config.learn_expert_signatures:
            self.signatures = nn.Parameter(
                torch.randn(self.num_experts, self.dim) * 0.02
            )
        else:
            self.register_buffer(
                'signatures',
                F.normalize(torch.randn(self.num_experts, self.dim), dim=-1)
            )
        
        # Input projection
        self.use_proj = config.use_input_projection
        if self.use_proj:
            self.proj = nn.Linear(config.hidden_size, self.dim, bias=False)
        
        # Learnable sharpness (optional)
        self.learn_sharpness = False
        if self.learn_sharpness:
            self.sharpness_param = nn.Parameter(torch.tensor(self.sharpness))
    
    @property
    def labels_normalized(self) -> Tensor:
        return F.normalize(self.labels, dim=-1)
    
    @property
    def signatures_normalized(self) -> Tensor:
        return F.normalize(self.signatures, dim=-1)
    
    @property
    def current_sharpness(self) -> float:
        if self.learn_sharpness:
            return self.sharpness_param.item()
        return self.sharpness
    
    def compute_router_memory(self) -> Tensor:
        """Compute router superposition memory."""
        L = self.labels_normalized
        E = self.signatures_normalized
        bindings = VSAOps.bind(E, L)
        return bindings.sum(dim=0)
    
    def cleanup(self, z: Tensor) -> Tensor:
        """
        Soft cleanup via attention over signatures.
        
        Args:
            z: Current estimate of shape [N, d]
        
        Returns:
            Cleaned vector of shape [N, d]
        """
        sharpness = self.current_sharpness
        attn = F.softmax(
            sharpness * VSAOps.similarity(z, self.signatures_normalized),
            dim=-1
        )  # [N, K]
        return attn @ self.signatures_normalized  # [N, d]
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with resonator iterations.
        
        Args:
            x: Input tensor of shape [N, d]
        
        Returns:
            weights: [N, top_k]
            indices: [N, top_k]
            scores: [N, K]
        """
        N = x.shape[0]
        
        if self.use_proj:
            x_proj = self.proj(x)
        else:
            x_proj = x
        
        if self.config.normalize_inputs:
            x_proj = F.normalize(x_proj, dim=-1)
        
        R = self.compute_router_memory()  # [d]
        R_expanded = R.unsqueeze(0).expand(N, -1)  # [N, d]
        
        # Initialize with single unbind
        z = VSAOps.unbind(x_proj, R_expanded)  # [N, d]
        
        # Resonator iterations
        for _ in range(self.iterations):
            z_clean = self.cleanup(z)
            R_bound = VSAOps.bind(R_expanded, z_clean)
            z = VSAOps.unbind(x_proj, R_bound)
        
        # Final scores
        scores = VSAOps.similarity(z, self.signatures_normalized)
        
        top_k_scores, indices = scores.topk(self.top_k, dim=-1)
        weights = F.softmax(top_k_scores / self.temperature, dim=-1)
        
        return weights, indices, scores


class LinearRouter(nn.Module):
    """
    Standard linear router for baseline comparison.
    
    Simple learned projection: scores = x @ W
    """
    
    def __init__(self, config: VSARouterConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.temperature = config.router_temperature
        
        self.linear = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [N, d]
        
        Returns:
            weights: [N, top_k]
            indices: [N, top_k]  
            scores: [N, K]
        """
        scores = self.linear(x)  # [N, K]
        
        top_k_scores, indices = scores.topk(self.top_k, dim=-1)
        weights = F.softmax(top_k_scores / self.temperature, dim=-1)
        
        return weights, indices, scores


def create_router(config: VSARouterConfig, router_type: str = "hrr") -> nn.Module:
    """
    Factory function to create routers.
    
    Args:
        config: Router configuration
        router_type: One of "hrr", "fpe", "resonator", "linear"
    
    Returns:
        Router module
    """
    routers = {
        "hrr": HRRRouter,
        "fpe": FPERouter,
        "resonator": ResonatorRouter,
        "linear": LinearRouter,
    }
    
    if router_type not in routers:
        raise ValueError(f"Unknown router type: {router_type}. Choose from {list(routers.keys())}")
    
    return routers[router_type](config)
