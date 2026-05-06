"""
Quaternion Generator with Dilated Convolutions for Speech Enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import yaml
from typing import Any, Dict, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_qnn.quaternion_layers import QuaternionConv, QuaternionTransposeConv, QuaternionBatchNorm2d
from core_qnn.quaternion_ops import get_r, get_i, get_j, get_k
from .conformer import QuaternionConformer


class QuaternionDilatedBlock(nn.Module):
    """DenseNet-style Dilated Convolution Block for Quaternion Networks"""
    def __init__(self, in_channels: int, out_channels: int = 16, kernel_size: Tuple[int, int] = (3, 3),
                 dilation: Tuple[int, int] = (1, 1), use_gate: bool = True) -> None:
        super().__init__()
        
        # Fixed growth rate of 16
        self.growth_rate = out_channels
        out_channels = self.growth_rate
        
        # Calculate padding for 'same' convolution with dilation
        padding = (
            (kernel_size[0] - 1) * dilation[0] // 2,
            (kernel_size[1] - 1) * dilation[1] // 2
        )
        
        # Bottleneck layer (1x1 conv) for dimension reduction
        self.bottleneck = QuaternionConv(
            in_channels, out_channels * 4,  # 4*growth_rate intermediate channels
            (1, 1), (1, 1),
            operation='convolution2d',
            padding=0,
            bias=False
        )
        self.bottleneck_norm = QuaternionBatchNorm2d(out_channels * 4)
        self.bottleneck_activation = nn.PReLU()
        
        # Main convolution - always stride (1,1) for DenseNet
        self.conv = QuaternionConv(
            out_channels * 4, out_channels,
            kernel_size, (1, 1),  # Always stride (1,1)
            operation='convolution2d',
            padding=padding,
            dilation=dilation,
            bias=False
        )
        
        # Gated activation (optional)
        self.use_gate = use_gate
        if use_gate:
            self.gate_conv = QuaternionConv(
                out_channels * 4, out_channels,
                kernel_size, (1, 1),  # Always stride (1,1)
                operation='convolution2d',
                padding=padding,
                dilation=dilation,
                bias=False
            )
        
        self.norm = QuaternionBatchNorm2d(out_channels)
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottleneck
        out = self.bottleneck(x)
        out = self.bottleneck_norm(out)
        out = self.bottleneck_activation(out)
        
        if self.use_gate:
            # Gated activation unit (like WaveNet)
            conv_out = self.conv(out)
            gate_out = torch.sigmoid(self.gate_conv(out))
            out = conv_out * gate_out
        else:
            out = self.conv(out)
        
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # DenseNet concatenation
        return torch.cat([x, out], dim=1)


class QuaternionDilatedEncoder(nn.Module):
    """DenseNet-style Encoder with Dilated Quaternion Convolutions"""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        
        # Default values
        default_config = {
            'in_channels': 4,
            'num_blocks': 4,  # Number of dense blocks
            'growth_rate': 16,  # Fixed growth rate
            'dilations': [1, 2, 4, 8],
            'kernel_size': [3, 3],
            'downsample_freq': True,
            'use_gate': True,
            'compression': 0.5  # Transition layer compression factor
        }
        
        # Merge with provided config
        if config is not None:
            default_config.update(config)
        config = default_config
        
        self.growth_rate = config.get("growth_rate", 16)  # Use config value
        self.encoders = nn.ModuleList()
        self.transitions = nn.ModuleList()
        kernel_size = tuple(config['kernel_size'])
        
        # Track accumulated channels due to DenseNet concatenation
        current_channels = config['in_channels']
        
        num_blocks = min(config.get('num_blocks', 4), len(config['dilations']))
        
        for i, dilation in enumerate(config['dilations'][:num_blocks]):
            # Dense block (always outputs 16 channels, but concatenates with input)
            self.encoders.append(
                QuaternionDilatedBlock(
                    in_channels=current_channels,
                    out_channels=self.growth_rate,  # Always 16
                    kernel_size=kernel_size,
                    dilation=(dilation, 1),  # Dilate only time dimension
                    use_gate=config['use_gate']
                ))
            
            # Update channel count (concatenation adds growth_rate channels)
            current_channels += self.growth_rate
            
            # Add transition layer to reduce channels (except for last block)
            if i < num_blocks - 1:
                out_channels = int(current_channels * config.get('compression', 0.5))
                # Ensure output channels is divisible by 4 for quaternion operations
                out_channels = (out_channels // 4) * 4
                if out_channels == 0:
                    out_channels = 4
                self.transitions.append(
                    nn.Sequential(
                        QuaternionBatchNorm2d(current_channels),
                        nn.PReLU(),
                        QuaternionConv(current_channels, out_channels, (1, 1), (1, 1),
                                     operation='convolution2d', bias=False)
                    ))
                current_channels = out_channels
            else:
                # For the last block, ensure final channels is divisible by 4
                if current_channels % 4 != 0:
                    # Add a transition to make it divisible by 4
                    out_channels = (current_channels // 4) * 4
                    if out_channels == 0:
                        out_channels = 4
                    self.transitions.append(
                        nn.Sequential(
                            QuaternionBatchNorm2d(current_channels),
                            nn.PReLU(),
                            QuaternionConv(current_channels, out_channels, (1, 1), (1, 1),
                                         operation='convolution2d', bias=False)
                        ))
                    current_channels = out_channels
                else:
                    self.transitions.append(None)
        
        # Add downsampling layer at the end (after all DenseNet blocks)
        self.downsample_freq = config['downsample_freq']
        if self.downsample_freq:
            # Use stride convolution for downsampling in frequency
            self.downsample = QuaternionConv(
                current_channels, current_channels,
                (3, 3), (1, 2),  # Downsample frequency by 2
                operation='convolution2d',
                padding=(1, 1),
                bias=False
            )
            self.downsample_norm = QuaternionBatchNorm2d(current_channels)
        
        # Output projection to adjust channels if specified
        self.output_channels = config.get('output_channels', None)
        if self.output_channels is not None and self.output_channels != current_channels:
            # Ensure output channels is divisible by 4 for quaternion
            assert self.output_channels % 4 == 0, "output_channels must be divisible by 4"
            self.output_proj = nn.Sequential(
                QuaternionConv(
                    current_channels, self.output_channels,
                    (1, 1), (1, 1),  # 1x1 conv for channel adjustment
                    operation='convolution2d',
                    bias=False
                ),
                QuaternionBatchNorm2d(self.output_channels),
                nn.PReLU()
            )
            self.final_channels = self.output_channels
        else:
            self.output_proj = None
            self.final_channels = current_channels
        
        # Add PReLU to output layer
        self.output_activation = nn.PReLU()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        encoder_outputs = []
        
        # Process DenseNet blocks
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)  # This already concatenates internally
            encoder_outputs.append(x)
            
            # Apply transition layer if exists
            if i < len(self.transitions) and self.transitions[i] is not None:
                x = self.transitions[i](x)
        
        # Apply downsampling at the end
        if self.downsample_freq:
            x = self.downsample(x)
            x = self.downsample_norm(x)
        
        # Apply output projection if specified
        if self.output_proj is not None:
            x = self.output_proj(x)
        
        # Apply output activation
        x = self.output_activation(x)
        
        return x, encoder_outputs


class QuaternionSubPixelConv(nn.Module):
    """Sub-Pixel Convolution for Quaternion Networks (upsampling)"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (1, 3),
        r: int = 2
    ) -> None:
        super().__init__()
        self.r = r
        self.out_channels = out_channels
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.conv = QuaternionConv(
            in_channels, out_channels * r,
            kernel_size, (1, 1),
            operation='convolution2d',
            padding=0,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class QuaternionMaskDecoder(nn.Module):
    """Quaternion Mask Decoder for magnitude mask estimation (like CMGAN MaskDecoder)"""
    def __init__(self, in_channels: int, num_features: int = 201, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()

        # Default config
        default_config = {
            'num_channel': 64,
            'num_dense_layers': 4,
            'dilations': [1, 2, 4, 8],
            'kernel_size': [3, 3],
            'sub_pixel_r': 2
        }
        if config is not None:
            default_config.update(config)
        config = default_config

        num_channel = config['num_channel']
        dilations = config['dilations']
        kernel_size = tuple(config['kernel_size'])
        sub_pixel_r = config['sub_pixel_r']

        # Build dense block dynamically
        layers = []
        for i, dilation in enumerate(dilations[:config['num_dense_layers']]):
            if i == 0:
                layers.append(QuaternionConv(in_channels, num_channel, kernel_size, (1, 1),
                              operation='convolution2d', padding=(dilation, 1), dilation=(dilation, 1), bias=False))
            else:
                layers.append(QuaternionConv(num_channel, num_channel, kernel_size, (1, 1),
                              operation='convolution2d', padding=(dilation, 1), dilation=(dilation, 1), bias=False))
            layers.append(QuaternionBatchNorm2d(num_channel))
            layers.append(nn.PReLU())
        self.dense_block = nn.Sequential(*layers)

        # Sub-pixel upsampling
        self.sub_pixel = QuaternionSubPixelConv(num_channel, num_channel, (1, 3), r=sub_pixel_r)

        # Output projection
        self.conv_1 = nn.Conv2d(num_channel, 1, (1, 2))
        self.norm = nn.InstanceNorm2d(1, affine=True)
        self.prelu = nn.PReLU(1)
        self.final_conv = nn.Conv2d(1, 1, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.prelu_out(x)
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        return x


class QuaternionComplexDecoder(nn.Module):
    """Quaternion Complex Decoder for complex residual estimation (like CMGAN ComplexDecoder)"""
    def __init__(self, in_channels: int, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()

        # Default config
        default_config = {
            'num_channel': 64,
            'num_dense_layers': 4,
            'dilations': [1, 2, 4, 8],
            'kernel_size': [3, 3],
            'sub_pixel_r': 2
        }
        if config is not None:
            default_config.update(config)
        config = default_config

        num_channel = config['num_channel']
        dilations = config['dilations']
        kernel_size = tuple(config['kernel_size'])
        sub_pixel_r = config['sub_pixel_r']

        # Build dense block dynamically
        layers = []
        for i, dilation in enumerate(dilations[:config['num_dense_layers']]):
            if i == 0:
                layers.append(QuaternionConv(in_channels, num_channel, kernel_size, (1, 1),
                              operation='convolution2d', padding=(dilation, 1), dilation=(dilation, 1), bias=False))
            else:
                layers.append(QuaternionConv(num_channel, num_channel, kernel_size, (1, 1),
                              operation='convolution2d', padding=(dilation, 1), dilation=(dilation, 1), bias=False))
            layers.append(QuaternionBatchNorm2d(num_channel))
            layers.append(nn.PReLU())
        self.dense_block = nn.Sequential(*layers)

        # Sub-pixel upsampling
        self.sub_pixel = QuaternionSubPixelConv(num_channel, num_channel, (1, 3), r=sub_pixel_r)

        # Output projection
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        x = x.permute(0, 1, 3, 2)
        return x


class QuaternionDilatedDecoder(nn.Module):
    """DenseNet-style Decoder with Transposed Dilated Quaternion Convolutions"""
    def __init__(self, in_channels: int, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        
        # Default values
        default_config = {
            'num_blocks': 4,  # Number of dense blocks
            'growth_rate': 16,  # Fixed growth rate (same as encoder)
            'dilations': [8, 4, 2, 1],
            'kernel_size': [3, 3],
            'upsample_freq': True,
            'use_gate': True,
            'compression': 0.5  # Transition layer compression
        }
        
        # Merge with provided config
        if config is not None:
            default_config.update(config)
        config = default_config
        
        self.growth_rate = config.get("growth_rate", 16)  # Use config value
        self.decoders = nn.ModuleList()
        self.transitions = nn.ModuleList()
        kernel_size = tuple(config['kernel_size'])
        
        # Track accumulated channels
        current_channels = in_channels
        
        num_blocks = min(config.get('num_blocks', 4), len(config['dilations']))
        
        for i, dilation in enumerate(config['dilations'][:num_blocks]):
            # Use DenseNet block
            self.decoders.append(
                QuaternionDilatedBlock(
                    in_channels=current_channels,
                    out_channels=self.growth_rate,  # Always 16
                    kernel_size=kernel_size,
                    dilation=(dilation, 1),
                    use_gate=config.get('use_gate', True)
                ))
            
            # Update channel count (concatenation adds growth_rate channels)
            current_channels += self.growth_rate
            
            # Add transition layer to reduce channels (except for last block)
            if i < num_blocks - 1:
                out_channels = int(current_channels * config.get('compression', 0.5))
                self.transitions.append(
                    nn.Sequential(
                        QuaternionBatchNorm2d(current_channels),
                        nn.PReLU(),
                        QuaternionConv(current_channels, out_channels, (1, 1), (1, 1),
                                     operation='convolution2d', bias=False)
                    ))
                current_channels = out_channels
            else:
                self.transitions.append(None)
        
        # Add upsampling layer at the end (after DenseNet blocks)
        self.upsample_freq = config['upsample_freq']
        if self.upsample_freq:
            # Use transpose convolution for upsampling
            self.upsample = QuaternionTransposeConv(
                current_channels, current_channels,
                (3, 3), (1, 2),  # Upsample frequency by 2
                operation='convolution2d',
                padding=(1, 1),
                output_padding=(0, 0),  # Adjust to get exact size
                bias=False
            )
            self.upsample_norm = QuaternionBatchNorm2d(current_channels)
        
        self.final_channels = current_channels
    
    def forward(self, x: torch.Tensor, encoder_outputs: Optional[list[torch.Tensor]] = None) -> torch.Tensor:
        # Process decoder layers with DenseNet connections
        for i, decoder in enumerate(self.decoders):
            x = decoder(x)
            
            # Apply transition layer if exists
            if i < len(self.transitions) and self.transitions[i] is not None:
                x = self.transitions[i](x)
        
        # Apply upsampling at the end
        if self.upsample_freq:
            x = self.upsample(x)
            x = self.upsample_norm(x)
        
        return x




class QuaternionGenerator(nn.Module):
    """Main Quaternion Generator with Dilated Convolutions"""
    def __init__(self, config_path: Optional[str] = None) -> None:
        super().__init__()
        
        # Load configuration
        if not config_path:
            raise ValueError("Configuration file path must be provided")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if self.config is None:
            raise ValueError(f"Failed to load configuration from {config_path}")
        if 'model' not in self.config:
            raise KeyError(f"'model' key not found in configuration. Available keys: {list(self.config.keys())}")
        
        model_config = self.config['model']
        
        # Encoder pathway
        self.encoder = QuaternionDilatedEncoder(model_config['encoder'])
        
        # Bottleneck Conformer (optional)
        self.use_bottleneck = model_config['bottleneck']['enabled']
        if self.use_bottleneck:
            bottleneck_config = model_config['bottleneck']
            # Ensure encoder final channels is divisible by 4 for quaternion
            encoder_channels = self.encoder.final_channels
            # Use encoder channels as d_model if it doesn't match config
            d_model = encoder_channels if encoder_channels != bottleneck_config['d_model'] else bottleneck_config['d_model']
            
          
            self.bottleneck = QuaternionConformer(
                d_model=d_model,
                n_heads=bottleneck_config['n_heads'],
                n_layers=bottleneck_config['n_layers'],
                conv_kernel_size=bottleneck_config['conv_kernel_size'],
                dropout=bottleneck_config['dropout']
            )
        
        # Decoder pathway
        encoder_out_channels = self.encoder.final_channels
        self.decoder = QuaternionDilatedDecoder(
            in_channels=encoder_out_channels,
            config=model_config['decoder']
        )
        
        # Output projection
        # Use the actual final channel count from the decoder
        decoder_out_channels = self.decoder.final_channels
        proj_config = model_config['output_proj']
        kernel_size = tuple(proj_config['kernel_size'])
        
        # First use QuaternionConv for feature extraction
        self.output_proj_quaternion = nn.Sequential(
            QuaternionConv(
                decoder_out_channels, 
                proj_config['intermediate_channels'],
                kernel_size, (1, 1),
                operation='convolution2d', 
                padding=(kernel_size[0]//2, kernel_size[1]//2),
                bias=proj_config['use_bias_first']
            ),
            nn.PReLU()
        )
        
        # Then use real-valued Conv2d to output 2 channels (real, imag)
        self.output_proj_complex = nn.Conv2d(
            proj_config['intermediate_channels'],
            2,  # Output 2 channels (real and imaginary parts)
            kernel_size=1,
            stride=1,
            padding=0,
            bias=proj_config['use_bias_second']
        )
        
        # Mask layer for cIRM (complex Ideal Ratio Mask)
        self.mask_layer = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh()  # cIRM: complex Ideal Ratio Mask [-1, 1]
        )
        
        # Skip connection layer (Bottleneck Skip only - Input Skip removed for theoretical consistency)
        # Bottleneck Skip: Encoder output → Decoder input (connection in same quaternion space)
        self.bottleneck_skip_fusion = nn.Sequential(
            QuaternionConv(
                encoder_out_channels * 2, encoder_out_channels,
                (1, 1), (1, 1),
                operation='convolution2d',
                padding=0,
                bias=False
            ),
            QuaternionBatchNorm2d(encoder_out_channels),
            nn.PReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input quaternion features [B, 4, T, F]
                - x[:, 0]: zero (pure quaternion)
                - x[:, 1]: magnitude
                - x[:, 2]: real part
                - x[:, 3]: imaginary part
        
        Returns:
            mask: Complex mask [B, 2, F, T] with magnitude and phase masks
        """
        # Encode with dilated convolutions
        bottleneck_input, _ = self.encoder(x)
        
        # Bottleneck processing (optional)
        if self.use_bottleneck:
            # QuaternionConformer now expects [B, C, T, F]
            bottleneck_output = self.bottleneck(bottleneck_input)
        else:
            bottleneck_output = bottleneck_input
        
        # ========== Bottleneck Skip Connection ==========
        # Concatenate encoder output with bottleneck output for richer features
        decoder_input = torch.cat([bottleneck_output, bottleneck_input], dim=1)  # [B, 2*C, T, F/2]
        decoder_input = self.bottleneck_skip_fusion(decoder_input)  # [B, C, T, F/2]
        
        # Decode with skip connection from encoder
        decoded = self.decoder(decoder_input)
        
        # Final projection (Input Skip removed - direct processing for theoretical consistency)
        # First quaternion processing
        features = self.output_proj_quaternion(decoded)
        # Convert to 2 channels (real, imag)
        output = self.output_proj_complex(features)
        
        # Apply mask layer to get magnitude and phase masks
        mask = self.mask_layer(output)  # [B, 2, T, F]
        
        # Permute to [B, 2, F, T] for consistency with STFT format
        mask = mask.permute(0, 1, 3, 2)  # [B, 2, F, T]
        
        return mask
    
    def get_output_complex(self, enhanced: torch.Tensor) -> torch.Tensor:
        """Convert output to complex spectrogram format
        
        Args:
            enhanced: [B, 2, T, F] tensor with real and imaginary parts
        
        Returns:
            complex_spec: [B, 2, F, T] for compatibility with STFT
        """
        # enhanced is already [B, 2, T, F] with channels [real, imag]
        # Permute to [B, 2, F, T] for compatibility with STFT
        complex_spec = enhanced.permute(0, 1, 3, 2)  # [B, 2, F, T]
        
        return complex_spec
