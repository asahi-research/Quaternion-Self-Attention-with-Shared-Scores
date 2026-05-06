"""
Quaternion Conformer Module for Speech Enhancement
Separated from generator.py for modularity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Literal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_qnn.quaternion_layers import QuaternionConv, QuaternionLinear, QuaternionBatchNorm2d
from core_qnn.quaternion_attention import QuaternionMultiHeadAttention


class QuaternionConformer(nn.Module):
    """Two-stage Quaternion Conformer (Time -> Freq) for bottleneck"""
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Conformer-Time blocks
        self.time_layers = nn.ModuleList([
            QuaternionConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                axis='time'
            ) for _ in range(n_layers)
        ])
        
        # Conformer-Freq blocks  
        self.freq_layers = nn.ModuleList([
            QuaternionConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                axis='freq'
            ) for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        B, C, T, F = x.shape
        
        # Process with Conformer-Time blocks
        # Reshape for time processing: [B*F, T, C]
        x = x.permute(0, 3, 2, 1).reshape(B*F, T, C)
        for layer in self.time_layers:
            x = layer(x)
        
        # Reshape back and process with Conformer-Freq blocks
        # [B*F, T, C] -> [B, F, T, C] -> [B*T, F, C]
        x = x.reshape(B, F, T, C).permute(0, 2, 1, 3).reshape(B*T, F, C)
        for layer in self.freq_layers:
            x = layer(x)
        
        # Reshape back to original format: [B*T, F, C] -> [B, T, F, C] -> [B, C, T, F]
        x = x.reshape(B, T, F, C).permute(0, 3, 1, 2)
        
        return x


class QuaternionConformerBlock(nn.Module):
    """Single Conformer Block following TS-Conformer architecture"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        conv_kernel_size: int,
        dropout: float,
        axis: Literal['time', 'freq'] = 'time'
    ) -> None:
        super().__init__()
        
        self.axis = axis  # 'time' or 'freq' to indicate processing axis
        
        # First half Feed-forward module
        self.ff1 = QuaternionFeedForward(d_model, d_model * 4, dropout)
        
        # Quaternion Multi-head self-attention
        self.mhsa = QuaternionMultiHeadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=dropout)
        
        # Quaternion Convolution module
        self.conv = QuaternionConvolutionModule(d_model, conv_kernel_size, dropout)
        
        # Second half Feed-forward module
        self.ff2 = QuaternionFeedForward(d_model, d_model * 4, dropout)
        
        # Layer normalization (post-norm style as shown in the image)
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.norm_mhsa = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_ff2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Seq, C] where Seq is either T (time) or F (freq)
        # TS-Conformer architecture: FF1/2 -> MHSA -> Conv -> FF1/2
        # with residual connections and post-layer normalization
        
        # First half feed-forward with residual
        residual = x
        x = self.ff1(x)
        x = self.dropout(x)
        x = residual + 0.5 * x  # Half of FF as per TS-Conformer
        x = self.norm_ff1(x)  # Post-norm
        
        # Multi-head self-attention with residual
        residual = x
        # Don't request attention weights to save memory
        x_att = self.mhsa(x, need_weights=False)  # QuaternionMultiHeadAttention
        x = residual + self.dropout(x_att)
        x = self.norm_mhsa(x)  # Post-norm
        
        # Convolution module with residual
        residual = x
        x = self.conv(x)
        x = residual + x  # Full residual for conv module
        x = self.norm_conv(x)  # Post-norm
        
        # Second half feed-forward with residual
        residual = x
        x = self.ff2(x)
        x = self.dropout(x)
        x = residual + 0.5 * x  # Half of FF as per TS-Conformer
        x = self.norm_ff2(x)  # Post-norm
        
        return x


class QuaternionFeedForward(nn.Module):
    """Feed-forward module for Quaternion Conformer"""
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        
        self.linear1 = QuaternionLinear(d_model, d_ff)
        self.linear2 = QuaternionLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()  # Swish activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class QuaternionConvolutionModule(nn.Module):
    """Quaternion Convolution module for Quaternion Conformer"""
    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        
        # Ensure d_model is divisible by 4 for quaternion operations
        assert d_model % 4 == 0, "d_model must be divisible by 4 for quaternion operations"
        
        # LayerNorm -> PointwiseConv -> GLU -> DepthwiseConv -> LayerNorm -> Swish -> PointwiseConv
        
        # First LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        
        # Pointwise conv (expansion for GLU)
        self.pointwise1 = QuaternionLinear(d_model, d_model * 2)
        
        # Quaternion convolution (depthwise-like but for quaternion)
        # Note: True depthwise is not straightforward with quaternion, so we use regular quaternion conv
        self.qconv = QuaternionConv(
            d_model, d_model, 
            kernel_size, 1,  # kernel_size, stride
            operation='convolution1d',
            padding=kernel_size // 2,
            bias=False
        )
        
        # Second LayerNorm
        self.norm2 = nn.LayerNorm(d_model)
        
        # Swish activation
        self.swish = nn.SiLU()
        
        # Pointwise projection using QuaternionConv
        self.pointwise2 = QuaternionConv(
            d_model, d_model,
            1, 1,  # kernel_size=1, stride=1 for pointwise
            operation='convolution1d',
            padding=0,
            bias=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T*F, C]
        B, TF, C = x.shape
        
        # First LayerNorm
        x = self.norm1(x)
        
        # Pointwise expansion with GLU
        x = self.pointwise1(x)  # [B, T*F, 2C]
        x = F.glu(x, dim=-1)     # [B, T*F, C]
        
        # Quaternion convolution
        x = x.transpose(1, 2)    # [B, C, T*F]
        x = self.qconv(x)  # Quaternion conv
        x = x.transpose(1, 2)    # [B, T*F, C]
        
        # Second LayerNorm
        x = self.norm2(x)
        
        # Swish activation
        x = self.swish(x)
        
        # Pointwise projection with QuaternionConv
        x = x.transpose(1, 2)    # [B, C, T*F]
        x = self.pointwise2(x)   # QuaternionConv with kernel_size=1
        x = x.transpose(1, 2)    # [B, T*F, C]
        
        x = self.dropout(x)
        
        return x





if __name__ == "__main__":
    # Test the modules
    print("Testing Quaternion Conformer modules...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test standard Conformer
    batch_size = 2
    channels = 64  # Must be divisible by 4
    time_frames = 161  # 1 second
    freq_bins = 100  # After downsampling
    
    # Create dummy input
    x = torch.randn(batch_size, channels, time_frames, freq_bins, device=device)
    
    # Test full Conformer
    print("\n1. Testing QuaternionConformer...")
    conformer = QuaternionConformer(
        d_model=channels,
        n_heads=4,
        n_layers=1,
        conv_kernel_size=31,
        dropout=0.1
    ).to(device)
    
    with torch.no_grad():
        out = conformer(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in conformer.parameters()):,}")
    
    
    # Compare memory usage
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test memory for standard Conformer
        with torch.no_grad():
            _ = conformer(x)
        memory_standard = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"\n2. Memory Usage:")
        print(f"   Conformer Peak Memory: {memory_standard:.3f} GB")
    
    print("\nAll tests passed!")
