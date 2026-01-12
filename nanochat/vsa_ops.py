"""
Vector Symbolic Architecture (VSA) Operations for MoE Routing

Implements core VSA/HRR operations:
- Binding: circular convolution via FFT
- Unbinding: circular correlation (inverse binding)
- Bundling: superposition with optional normalization
- Similarity: cosine similarity for retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class VSAOps:
    """
    Static methods for VSA operations.
    All operations support batched inputs with shape [..., d].
    """
    
    @staticmethod
    def bind(x: Tensor, y: Tensor) -> Tensor:
        """
        Circular convolution (binding) via FFT.
        
        x ⊗ y = IFFT(FFT(x) ⊙ FFT(y))
        
        Args:
            x: Tensor of shape [..., d]
            y: Tensor of shape [..., d] (broadcastable with x)
        
        Returns:
            Bound tensor of shape [..., d]
        """
        d = x.shape[-1]
        x_fft = torch.fft.rfft(x, dim=-1)
        y_fft = torch.fft.rfft(y, dim=-1)
        return torch.fft.irfft(x_fft * y_fft, n=d, dim=-1)
    
    @staticmethod
    def unbind(x: Tensor, y: Tensor) -> Tensor:
        """
        Circular correlation (unbinding) via FFT.
        
        x ⊗^{-1} y = IFFT(conj(FFT(x)) ⊙ FFT(y))
        
        This retrieves y' ≈ y when x ⊗ y' was stored.
        
        Args:
            x: Tensor of shape [..., d] (the key/query)
            y: Tensor of shape [..., d] (the memory/superposition)
        
        Returns:
            Retrieved tensor of shape [..., d]
        """
        d = x.shape[-1]
        x_fft = torch.fft.rfft(x, dim=-1)
        y_fft = torch.fft.rfft(y, dim=-1)
        return torch.fft.irfft(x_fft.conj() * y_fft, n=d, dim=-1)
    
    @staticmethod
    def bundle(vectors: Tensor, dim: int = 0, normalize: bool = True) -> Tensor:
        """
        Superposition (bundling) of multiple vectors.
        
        z = Σ_i x_i  (optionally normalized)
        
        Args:
            vectors: Tensor of shape [n, ..., d] where n vectors are bundled
            dim: Dimension along which to sum
            normalize: Whether to L2-normalize the result
        
        Returns:
            Bundled tensor with dim reduced
        """
        result = vectors.sum(dim=dim)
        if normalize:
            result = F.normalize(result, dim=-1)
        return result
    
    @staticmethod
    def similarity(x: Tensor, y: Tensor) -> Tensor:
        """
        Cosine similarity between x and y.
        
        Args:
            x: Tensor of shape [N, d]
            y: Tensor of shape [K, d]
        
        Returns:
            Similarity matrix of shape [N, K]
        """
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        return x_norm @ y_norm.T
    
    @staticmethod
    def similarity_single(x: Tensor, y: Tensor) -> Tensor:
        """
        Cosine similarity for single vectors or aligned batches.
        
        Args:
            x: Tensor of shape [..., d]
            y: Tensor of shape [..., d]
        
        Returns:
            Similarity tensor of shape [...]
        """
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(y, dim=-1)
        return (x_norm * y_norm).sum(dim=-1)
    
    @staticmethod
    def inverse(x: Tensor) -> Tensor:
        """
        Compute approximate inverse for unbinding.
        
        For HRR, the inverse is the time-reversal (flip) of the vector,
        which in frequency domain is the complex conjugate.
        
        Args:
            x: Tensor of shape [..., d]
        
        Returns:
            Inverted tensor of shape [..., d]
        """
        # Time reversal: x^{-1}[0] = x[0], x^{-1}[i] = x[d-i] for i > 0
        return torch.flip(x, dims=[-1]).roll(1, dims=-1)
    
    @staticmethod
    def random_hv(shape: tuple, device: torch.device = None, 
                  normalize: bool = True) -> Tensor:
        """
        Generate random hypervector(s).
        
        Args:
            shape: Shape of output tensor
            device: Torch device
            normalize: Whether to L2-normalize
        
        Returns:
            Random tensor of given shape
        """
        hv = torch.randn(shape, device=device)
        if normalize:
            hv = F.normalize(hv, dim=-1)
        return hv


class FPEOps:
    """
    Fractional Power Encoding operations for position-based VSA.
    """
    
    @staticmethod
    def create_base_vector(d: int, device: torch.device = None) -> Tensor:
        """
        Create base vector phases for FPE.
        
        B = exp(i * θ) where θ ~ Uniform(-π, π)
        
        Args:
            d: Dimension of hypervector
            device: Torch device
        
        Returns:
            Phase tensor of shape [d]
        """
        return torch.rand(d, device=device) * 2 * torch.pi - torch.pi
    
    @staticmethod
    def power_encode(theta: Tensor, p: Tensor) -> Tensor:
        """
        Compute B^p = exp(i * p * θ).
        
        Returns real part for compatibility with HRR operations.
        
        Args:
            theta: Base vector phases of shape [d]
            p: Power values of shape [K] or scalar
        
        Returns:
            Encoded vectors of shape [K, d] or [d]
        """
        if p.dim() == 0:
            phases = p * theta
        else:
            phases = p.unsqueeze(-1) * theta.unsqueeze(0)  # [K, d]
        
        # Complex exponential, return real part
        # (Alternatively, could work in full complex domain)
        return torch.cos(phases)
    
    @staticmethod
    def power_encode_complex(theta: Tensor, p: Tensor) -> Tensor:
        """
        Compute B^p in complex domain.
        
        Args:
            theta: Base vector phases of shape [d]
            p: Power values of shape [K] or scalar
        
        Returns:
            Complex tensor of shape [K, d] or [d]
        """
        if p.dim() == 0:
            phases = p * theta
        else:
            phases = p.unsqueeze(-1) * theta.unsqueeze(0)
        
        return torch.exp(1j * phases)
    
    @staticmethod
    def similarity_kernel(p: Tensor, q: Tensor, d: int) -> Tensor:
        """
        Compute expected similarity between B^p and B^q.
        
        For large d, this is approximately a sinc-like function of |p - q|.
        
        Args:
            p: Position values of shape [N]
            q: Position values of shape [K]
            d: Dimension (for normalization)
        
        Returns:
            Similarity matrix of shape [N, K]
        """
        diff = p.unsqueeze(-1) - q.unsqueeze(0)  # [N, K]
        # Approximate kernel (exact for uniform random phases)
        return torch.sinc(diff / torch.pi)


class ResonatorOps:
    """
    Operations for resonator network-based cleanup.
    """
    
    @staticmethod
    def cleanup_step(z: Tensor, codebook: Tensor, 
                     sharpness: float = 10.0) -> Tensor:
        """
        Single cleanup step via soft attention over codebook.
        
        Args:
            z: Current estimate of shape [N, d]
            codebook: Known symbols of shape [K, d]
            sharpness: Temperature for softmax (higher = sharper)
        
        Returns:
            Cleaned up vector of shape [N, d]
        """
        # Compute attention weights
        attn = F.softmax(
            sharpness * VSAOps.similarity(z, codebook),
            dim=-1
        )  # [N, K]
        
        # Weighted sum of codebook vectors
        return attn @ codebook  # [N, d]
    
    @staticmethod
    def resonator_iteration(x: Tensor, R: Tensor, z: Tensor, 
                           codebook: Tensor, sharpness: float = 10.0) -> Tensor:
        """
        Single resonator network iteration.
        
        z^{t+1} = x ⊗^{-1} (R ⊗ cleanup(z^t))
        
        Args:
            x: Query tensor of shape [N, d]
            R: Router memory of shape [d] or [N, d]
            z: Current state of shape [N, d]
            codebook: Symbol codebook of shape [K, d]
            sharpness: Cleanup sharpness
        
        Returns:
            Updated state of shape [N, d]
        """
        z_clean = ResonatorOps.cleanup_step(z, codebook, sharpness)
        R_bound = VSAOps.bind(R, z_clean)
        return VSAOps.unbind(x, R_bound)


# Convenience module wrappers for nn.Module compatibility
class BindModule(nn.Module):
    """Module wrapper for binding operation."""
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return VSAOps.bind(x, y)


class UnbindModule(nn.Module):
    """Module wrapper for unbinding operation."""
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return VSAOps.unbind(x, y)


class BundleModule(nn.Module):
    """Module wrapper for bundling operation."""
    
    def __init__(self, dim: int = 0, normalize: bool = True):
        super().__init__()
        self.dim = dim
        self.normalize = normalize
    
    def forward(self, vectors: Tensor) -> Tensor:
        return VSAOps.bundle(vectors, self.dim, self.normalize)
