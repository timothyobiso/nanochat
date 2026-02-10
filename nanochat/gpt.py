"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Mixture of Experts (MoE) config
    num_experts: int = 8           # total number of expert MLPs per MoE layer
    num_experts_per_tok: int = 2   # top-K experts activated per token
    moe_layer_freq: int = 0        # replace MLP with MoE every N layers (0 = disabled, 1 = every layer, 2 = every other)
    moe_aux_loss_coeff: float = 0.01  # load-balancing auxiliary loss coefficient
    moe_router_type: str = 'linear'   # 'linear', 'vsa_random', 'vsa_fpe'


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


# --- VSA / HRR (Holographic Reduced Representations) ---

def hrr_bind(a, b):
    """Circular convolution (binding). a, b: (..., d) -> (..., d)"""
    return torch.fft.ifft(torch.fft.fft(a, dim=-1) * torch.fft.fft(b, dim=-1), dim=-1).real

def hrr_unbind(s, key):
    """Circular correlation (unbinding). s: (..., d), key: (..., d) -> (..., d)"""
    return torch.fft.ifft(torch.fft.fft(s, dim=-1) * torch.fft.fft(key, dim=-1).conj(), dim=-1).real

def make_fpe_keys(base, num_keys):
    """Fractional Power Encoding. base: (d,) -> (num_keys, d)"""
    base_freq = torch.fft.fft(base)
    keys = []
    for i in range(num_keys):
        p = i / max(num_keys - 1, 1)
        mag = base_freq.abs().pow(p)
        phase = base_freq.angle() * p
        key_freq = mag * torch.exp(1j * phase)
        keys.append(torch.fft.ifft(key_freq).real)
    return torch.stack(keys)


class VSARouter(nn.Module):
    """VSA/HRR router: bundles (key, id) pairs into a single holographic memory.
    Routes by unbinding the token from memory and comparing to expert identifiers.
    No learnable parameters. Supports meta device init (actual data filled in init_buffers)."""

    def __init__(self, dim, num_experts, mode='random'):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.mode = mode
        # Register buffers with correct shapes (may be meta tensors at this point)
        self.register_buffer('memory', torch.zeros(dim))
        self.register_buffer('expert_ids', torch.zeros(num_experts, dim))
        self.register_buffer('expert_keys', torch.zeros(num_experts, dim))

    def init_buffers(self):
        """Compute and fill VSA buffers with actual data. Called from GPT.init_weights()."""
        device = self.memory.device
        if self.mode == 'random':
            keys = torch.randn(self.num_experts, self.dim, device=device)
        elif self.mode == 'fpe':
            keys = make_fpe_keys(torch.randn(self.dim, device=device), self.num_experts)
        else:
            raise ValueError(f"Unknown VSA router mode: {self.mode}")
        keys = F.normalize(keys, dim=-1)
        ids = F.normalize(torch.randn(self.num_experts, self.dim, device=device), dim=-1)
        memory = sum(hrr_bind(keys[i], ids[i]) for i in range(self.num_experts))
        self.memory.copy_(memory)
        self.expert_ids.copy_(ids)
        self.expert_keys.copy_(keys)

    def forward(self, x):
        """x: (N, d) -> router_logits: (N, num_experts)"""
        # FFT requires float32
        retrieved = hrr_unbind(self.memory.float().unsqueeze(0), x.float())
        scores = retrieved @ self.expert_ids.float().T
        return scores.to(x.dtype)


class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Router
        if config.moe_router_type == 'linear':
            self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)
        elif config.moe_router_type in ('vsa_random', 'vsa_fpe'):
            mode = 'random' if config.moe_router_type == 'vsa_random' else 'fpe'
            self.router = VSARouter(config.n_embd, config.num_experts, mode=mode)
        else:
            raise ValueError(f"Unknown moe_router_type: {config.moe_router_type}")
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.num_experts)])

    @torch.compiler.disable
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (B*T, C)
        # Router: compute gating logits and select top-K experts per token
        router_logits = self.router(x_flat)  # (B*T, num_experts)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)  # (B*T, K)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B*T, K)
        # Dispatch tokens to experts and combine outputs
        out = torch.zeros_like(x_flat)
        for k in range(self.num_experts_per_tok):
            expert_indices = top_k_indices[:, k]  # (B*T,)
            weights = top_k_weights[:, k]  # (B*T,)
            for e in range(self.num_experts):
                mask = expert_indices == e  # (B*T,)
                if mask.any():
                    expert_input = x_flat[mask]  # (n_tokens, C)
                    expert_output = self.experts[e](expert_input)  # (n_tokens, C)
                    out[mask] += weights[mask].unsqueeze(-1) * expert_output
        # Load-balancing auxiliary loss (Mixtral-style)
        # f_i = fraction of tokens routed to expert i, p_i = mean router prob for expert i
        router_probs = F.softmax(router_logits, dim=-1)  # (B*T, num_experts)
        # Fraction of tokens dispatched to each expert (from top-k selection)
        one_hot = F.one_hot(top_k_indices, self.num_experts).float()  # (B*T, K, num_experts)
        tokens_per_expert = one_hot.sum(dim=1).mean(dim=0)  # (num_experts,)
        mean_prob_per_expert = router_probs.mean(dim=0)  # (num_experts,)
        aux_loss = self.num_experts * (tokens_per_expert * mean_prob_per_expert).sum()
        return out.view(B, T, C), aux_loss


class Block(nn.Module):
    def __init__(self, config, layer_idx, is_moe=False):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.is_moe = is_moe
        if is_moe:
            self.moe = MoELayer(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        if self.is_moe:
            moe_out, aux_loss = self.moe(norm(x))
            x = x + moe_out
            return x, aux_loss
        else:
            x = x + self.mlp(norm(x))
            return x, 0


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # For DDP, we want vocab_size divisible by world_size. Also, there are potential performance benefits, see:
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} to be divisible by {pad_vocab_size_to}")
        # Determine which layers are MoE (if moe_layer_freq > 0, every Nth layer is MoE)
        def _is_moe_layer(layer_idx):
            if config.moe_layer_freq <= 0:
                return False
            return layer_idx % config.moe_layer_freq == 0
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx, is_moe=_is_moe_layer(layer_idx)) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            if block.is_moe:
                # Router init
                if hasattr(block.moe.router, 'weight'):
                    # Linear router: small normal init so routing starts near-uniform
                    torch.nn.init.normal_(block.moe.router.weight, mean=0.0, std=0.01)
                elif hasattr(block.moe.router, 'init_buffers'):
                    # VSA router: compute and fill holographic memory buffers
                    block.moe.router.init_buffers()
                # Expert MLPs: same init as dense MLP
                for expert in block.moe.experts:
                    torch.nn.init.uniform_(expert.c_fc.weight, -s, s)
                    torch.nn.init.zeros_(expert.c_proj.weight)
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        with torch.no_grad():
            self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
            self.x0_lambdas.fill_(0.0)      # 0.0 => skip connection to input is disabled at init

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast token embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, the term 12 * l * h * q * t accounts for key @ query matmul flops inside attention.
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        For MoE layers, only top-K out of N experts are active per token, so we exclude inactive expert params.
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        # For MoE layers, exclude the inactive expert params from FLOPs count
        moe_inactive_params = 0
        for block in self.transformer.h:
            if block.is_moe:
                expert_params = sum(p.numel() for p in block.moe.experts[0].parameters())
                inactive = (self.config.num_experts - self.config.num_experts_per_tok) * expert_params
                moe_inactive_params += inactive
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding - moe_inactive_params) + 12 * l * h * q * t
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return all of the parameters, same as Chinchilla paper.
        Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
        But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
        My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
        For MoE models, also reports total MoE params and active-per-token MoE params.
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Count MoE-specific params (total and active per token)
        moe_total_params = 0
        moe_active_params = 0
        for block in self.transformer.h:
            if block.is_moe:
                moe_block_params = sum(p.numel() for p in block.moe.parameters())
                moe_total_params += moe_block_params
                router_params = sum(p.numel() for p in block.moe.router.parameters())
                expert_params = sum(p.numel() for p in block.moe.experts[0].parameters())
                moe_active_params += router_params + self.config.num_experts_per_tok * expert_params
        if moe_total_params > 0:
            print0(f"MoE params: {moe_total_params:,} total, {moe_active_params:,} active per token")
        return nparams

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 5 groups (matrix, embedding, lm_head, resid_lambdas, x0_lambdas)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params)
        # Create the AdamW optimizer for the embedding, lm_head, and per-layer scalars
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=resid_params, lr=scalar_lr * 0.01), # these are a lot more sensitive because they accumulate in the residual stream
            dict(params=x0_params, lr=scalar_lr),
        ]
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0) # NOTE: weight decay is hardcoded to 0.0 for AdamW, only used in Muon
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        total_aux_loss = 0
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x, aux_loss = block(x, cos_sin, kv_cache)
            total_aux_loss = total_aux_loss + aux_loss
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            # Add MoE load-balancing auxiliary loss
            if total_aux_loss != 0:
                loss = loss + self.config.moe_aux_loss_coeff * total_aux_loss
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
