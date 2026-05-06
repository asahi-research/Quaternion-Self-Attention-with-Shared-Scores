##########################################################
# Quaternion Self-Attention Implementation
# Based on "Lightweight and Efficient Neural Natural Language Processing with Quaternion Networks"
# PyTorch implementation
##########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module
import numpy as np
from numpy.random import RandomState
from .quaternion_ops import *
from .quaternion_layers import QuaternionLinear
import math
from typing import Optional, Tuple, Union


class QuaternionRMSNorm(nn.Module):
    """RMSNorm that normalizes quaternion components (r,i,j,k) as a group"""
    def __init__(self, embed_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        assert embed_dim % 4 == 0
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(embed_dim // 4))  # Gain per quaternion group

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, D]
        B, L, D = x.shape
        xq = x.view(B, L, -1, 4)  # [B, L, Q, 4]
        rms = (xq.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()  # [B,L,Q,1]
        yq = xq / rms * self.gain.view(1, 1, -1, 1)
        return yq.view(B, L, D)


class QuaternionMultiHeadAttention(Module):
    """
    Quaternion Multi-Head Attention with Shared Score
    - Score = Re(Q × K*^⊤)  (real part of quaternion product)
    - Single softmax shared across all quaternion components of V
    - Optional RMSNorm applied to Q/K for stability
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0, bias: bool = True,
                 init_criterion: str = 'he', weight_init: str = 'quaternion', seed: Optional[int] = None,
                 use_rmsnorm: bool = True, rmsnorm_eps: float = 1e-6) -> None:
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_layer = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4"

        # Projections
        self.q_proj = QuaternionLinear(embed_dim, embed_dim, bias=bias,
                                       init_criterion=init_criterion,
                                       weight_init=weight_init, seed=seed)
        self.k_proj = QuaternionLinear(embed_dim, embed_dim, bias=bias,
                                       init_criterion=init_criterion,
                                       weight_init=weight_init, seed=seed)
        self.v_proj = QuaternionLinear(embed_dim, embed_dim, bias=bias,
                                       init_criterion=init_criterion,
                                       weight_init=weight_init, seed=seed)
        self.out_proj = QuaternionLinear(embed_dim, embed_dim, bias=bias,
                                         init_criterion=init_criterion,
                                         weight_init=weight_init, seed=seed)

        # Optional RMSNorm for quaternion group normalization
        self.use_rmsnorm = use_rmsnorm
        if use_rmsnorm:
            self.q_norm = QuaternionRMSNorm(embed_dim, eps=rmsnorm_eps)
            self.k_norm = QuaternionRMSNorm(embed_dim, eps=rmsnorm_eps)

    @staticmethod
    def _real_scores_q_k_conj(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores using real part of quaternion product q ⊗ k̄^⊤

        Args:
            q, k: [B, H, L, Dh] - Query and Key tensors
        Returns:
            scores: [B, H, L, L] - Attention scores

        Re(Q × K*^⊤) = <(qr,qi,qj,qk), (kr,ki,kj,kk)> = sum of component-wise inner products
        """
        qr, qi, qj, qk = torch.chunk(q, 4, dim=-1)
        kr, ki, kj, kk = torch.chunk(k, 4, dim=-1)
        # Each: [B, H, L, Dh/4]
        scores = (qr @ kr.transpose(-2, -1)
                + qi @ ki.transpose(-2, -1)
                + qj @ kj.transpose(-2, -1)
                + qk @ kk.transpose(-2, -1))
        return scores

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if key is None:   key = query
        if value is None: value = query

        B, L, _ = query.size()

        # Proj
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Optional RMSNorm (more effective in Q/K space)
        if self.use_rmsnorm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # [B, H, L, Dh]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Score: real part of q ⊗ k̄ (= 4-channel inner product)
        scores = self._real_scores_q_k_conj(q, k)  # [B,H,L,L]
        scores = scores / math.sqrt(self.head_dim)

        # Mask (supports various shapes)
        if attn_mask is not None:
            # Expected shapes: [L,L], [B,L,L], [B,1,L,L], [B,H,L,L]
            while attn_mask.dim() < scores.dim():
                attn_mask = attn_mask.unsqueeze(0)
            # Relies on broadcasting
            min_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attn_mask == 0, min_val)

        # Single softmax shared across all components
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_layer(attn)  # [B,H,L,L]

        # Apply same attention weights to all V components
        vr, vi, vj, vk = torch.chunk(v, 4, dim=-1)
        out_r = attn @ vr
        out_i = attn @ vi
        out_j = attn @ vj
        out_k = attn @ vk
        out = torch.cat([out_r, out_i, out_j, out_k], dim=-1)  # [B,H,L,Dh]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        out = self.out_proj(out)

        if need_weights:
            return out, attn  # [B,H,L,L]
        return out


class QuaternionSelfAttention(Module):
    """
    Single-head Quaternion Self-Attention with Shared Score
    Uses real part of q ⊗ k̄ for scoring with optional RMSNorm
    """
    def __init__(self, embed_dim: int, dropout: float = 0.0, bias: bool = True,
                 init_criterion: str = 'he', weight_init: str = 'quaternion', seed: Optional[int] = None,
                 use_rmsnorm: bool = True, rmsnorm_eps: float = 1e-6) -> None:
        super().__init__()
        assert embed_dim % 4 == 0
        self.embed_dim = embed_dim
        self.dropout_layer = nn.Dropout(dropout)

        self.q_proj = QuaternionLinear(embed_dim, embed_dim, bias=bias,
                                       init_criterion=init_criterion,
                                       weight_init=weight_init, seed=seed)
        self.k_proj = QuaternionLinear(embed_dim, embed_dim, bias=bias,
                                       init_criterion=init_criterion,
                                       weight_init=weight_init, seed=seed)
        self.v_proj = QuaternionLinear(embed_dim, embed_dim, bias=bias,
                                       init_criterion=init_criterion,
                                       weight_init=weight_init, seed=seed)
        self.out_proj = QuaternionLinear(embed_dim, embed_dim, bias=bias,
                                         init_criterion=init_criterion,
                                         weight_init=weight_init, seed=seed)

        self.use_rmsnorm = use_rmsnorm
        if use_rmsnorm:
            self.q_norm = QuaternionRMSNorm(embed_dim, eps=rmsnorm_eps)
            self.k_norm = QuaternionRMSNorm(embed_dim, eps=rmsnorm_eps)

    @staticmethod
    def _real_scores_q_k_conj(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        qr, qi, qj, qk = torch.chunk(q, 4, dim=-1)
        kr, ki, kj, kk = torch.chunk(k, 4, dim=-1)
        return (qr @ kr.transpose(-2, -1)
              + qi @ ki.transpose(-2, -1)
              + qj @ kj.transpose(-2, -1)
              + qk @ kk.transpose(-2, -1))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_rmsnorm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        scores = self._real_scores_q_k_conj(q, k) / math.sqrt(self.embed_dim)  # [B,L,L]

        if attn_mask is not None:
            min_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attn_mask == 0, min_val)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_layer(attn)
        # Shared across all components
        vr, vi, vj, vk = torch.chunk(v, 4, dim=-1)
        out = torch.cat([attn @ vr, attn @ vi, attn @ vj, attn @ vk], dim=-1)
        out = self.out_proj(out)

        if need_weights:
            return out, attn
        return out


if __name__ == "__main__":
    # Test the implementation
    batch_size = 2
    seq_len = 256
    embed_dim = 256  # Must be divisible by 4
    num_heads = 4
    
    # Create random input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test single-head attention
    print("Testing QuaternionSelfAttention...")
    attn = QuaternionSelfAttention(embed_dim, dropout=0.1)
    output = attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test multi-head attention
    print("\nTesting QuaternionMultiHeadAttention...")
    mha = QuaternionMultiHeadAttention(embed_dim, num_heads=num_heads, dropout=0.1)
    output = mha(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with need_weights
    output, attn_weights = mha(x, need_weights=True)
    print(f"Attention weights shape: {attn_weights.shape}")
    
    print("\nAll tests passed!")
