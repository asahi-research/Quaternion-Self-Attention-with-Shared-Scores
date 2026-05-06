import torch
import torch.utils.data
import torchaudio
import numpy as np
import random
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
from natsort import natsorted
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
import librosa
import librosa.display


PathLike = Union[str, os.PathLike[str]]


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features: int, beta: float = 1) -> None:
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


def power_compress(x: torch.Tensor) -> torch.Tensor:
    """Power compression for magnitude spectrograms (same as CMGAN)"""
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 0)


def power_uncompress(real: torch.Tensor, imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Power uncompression for magnitude spectrograms (same as CMGAN)"""
    # Add small epsilon to prevent numerical issues
    eps = 1e-8
    
    # Clip values to prevent overflow in power operation
    real = torch.clamp(real, min=-10, max=10)
    imag = torch.clamp(imag, min=-10, max=10)
    
    spec = torch.complex(real, imag)
    mag = torch.abs(spec) + eps  # Add epsilon to prevent zero magnitude
    phase = torch.angle(spec)
    
    # Clamp magnitude before power operation to prevent overflow
    mag = torch.clamp(mag, min=eps, max=10.0)
    mag = mag ** (1.0 / 0.3)
    
    real_uncompress = mag * torch.cos(phase)
    imag_uncompress = mag * torch.sin(phase)
    
    # Final safety check
    real_uncompress = torch.nan_to_num(real_uncompress, nan=0.0, posinf=10.0, neginf=-10.0)
    imag_uncompress = torch.nan_to_num(imag_uncompress, nan=0.0, posinf=10.0, neginf=-10.0)
    
    return real_uncompress, imag_uncompress  # Return as separate tensors


def compute_stft_quaternion_features(
    audio: torch.Tensor,
    n_fft: int,
    hop: int,
    window: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute STFT and convert to quaternion features
    
    Args:
        audio: Audio waveform tensor
        n_fft: FFT window size
        hop: Hop length
        window: Window function tensor
    
    Returns:
        features_4ch: 4-channel quaternion features [4, T, F]
            - Channel 0: zero (pure quaternion)
            - Channel 1: magnitude
            - Channel 2: real part
            - Channel 3: imaginary part
        spec_compressed: Compressed spectrogram [2, F, T]
    """
    # Compute STFT
    spec = torch.stft(
        audio,
        n_fft,
        hop,
        window=window,
        onesided=True,
        return_complex=False
    )
    
    # Apply power compression (disabled)
    # spec_compressed = power_compress(spec)  # [2, F, T]
    # spec has shape [F, T, 2] where last dimension is [real, imag]
    spec_compressed = spec.permute(2, 0, 1)  # [2, F, T]
    
    # Extract real and imaginary parts
    real = spec_compressed[0, :, :]  # [F, T]
    imag = spec_compressed[1, :, :]  # [F, T]
    
    # Calculate magnitude for the quaternion input
    mag = torch.sqrt(real**2 + imag**2)  # [F, T]
    
    # Create 4-channel quaternion feature (r=0, i=mag, j=real, k=imag)
    zero_channel = torch.zeros_like(mag)  # [F, T]
    features_4ch = torch.stack([zero_channel, mag, real, imag], dim=0)  # [4, F, T]
    
    # Permute to [4, T, F] to match quaternion input format
    features_4ch = features_4ch.permute(0, 2, 1)  # [4, T, F]
    
    return features_4ch, spec_compressed


def istft(
    real: torch.Tensor,
    imag: torch.Tensor,
    n_fft: int,
    hop: int,
    window: torch.Tensor,
    length: Optional[int] = None
) -> torch.Tensor:
    """Compute ISTFT from QuaternionGenerator output
    
    Args:
        real: Real part from QuaternionGenerator [B, T, F]
        imag: Imaginary part from QuaternionGenerator [B, T, F]
        n_fft: FFT window size
        hop: Hop length
        window: Window function tensor
        length: Optional length of the output signal
    
    Returns:
        audio: Reconstructed audio waveform tensor [B, L]
    """
    # Safety check for NaN in input (silent replacement to avoid spam)
    if torch.isnan(real).any() or torch.isnan(imag).any():
        real = torch.nan_to_num(real, nan=0.0)
        imag = torch.nan_to_num(imag, nan=0.0)
    
    # Clamp input values to prevent extreme values
   
    
    # Permute from [B, T, F] to [B, F, T]
    real = real.permute(0, 2, 1)  # [B, F, T]
    imag = imag.permute(0, 2, 1)  # [B, F, T]
    
    # Stack real and imag to create complex spectrogram format
    spec_compressed = torch.stack([real, imag], dim=1)  # [B, 2, F, T]
    
    # Process batch-wise
    batch_size = real.shape[0]
    audio_list = []
    
    for b in range(batch_size):
        # Extract single sample
        spec_b = spec_compressed[b]  # [2, F, T]
        
        # Apply power uncompression
        # real_uncomp, imag_uncomp = power_uncompress(spec_b[0], spec_b[1])  # returns separate tensors (disabled)
        real_uncomp = spec_b[0]  # [F, T] - use directly without uncompression
        imag_uncomp = spec_b[1]  # [F, T] - use directly without uncompression
        
        # Create complex tensor for ISTFT
        spec_complex = torch.complex(real_uncomp, imag_uncomp)  # [F, T]
        
        # Compute ISTFT for single sample
        audio_b = torch.istft(
            spec_complex,
            n_fft,
            hop,
            window=window,
            onesided=True,
            length=length
        )
        
        # Safety check for output
        if torch.isnan(audio_b).any():
            print("WARNING: NaN detected in istft output!")
            audio_b = torch.nan_to_num(audio_b, nan=0.0)
        
        # Clamp output to prevent extreme values (wider range for better dynamics)
        
        audio_list.append(audio_b)
    
    # Stack back to batch
    audio = torch.stack(audio_list, dim=0)  # [B, L]
    
    return audio




# ================================
# Distributed Training Setup
# ================================
def setup_distributed(rank: int, world_size: int, master_port: int = 12381) -> None:
    """Initialize distributed training

    Args:
        rank: Process rank
        world_size: Total number of processes
        master_port: Master port for distributed training (default: 12381)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """Clean up distributed training"""
    dist.destroy_process_group()


def count_parameters(model: nn.Module) -> int:
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: Mapping[str, Any], filename: PathLike) -> None:
    """Save checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(
    checkpoint_path: PathLike,
    generator: nn.Module,
    discriminators: Mapping[str, nn.Module],
    optimizers: Optional[Mapping[str, torch.optim.Optimizer]] = None
) -> Tuple[int, float]:
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    generator.load_state_dict(checkpoint['generator'])
    
    # Load discriminators (if any)
    for name in discriminators:
        if name in checkpoint:
            discriminators[name].load_state_dict(checkpoint[name])
    
    if optimizers:
        if 'g_optimizer' in checkpoint:
            optimizers['generator'].load_state_dict(checkpoint['g_optimizer'])
    
    # Handle both old (best_val_loss) and new (best_val_pesq) checkpoints
    if 'best_val_pesq' in checkpoint:
        return checkpoint.get('epoch', 0), checkpoint.get('best_val_pesq', 0.0)
    else:
        # For old checkpoints, convert SI-SDR to a proxy score
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        # Convert to higher-is-better metric (negate SI-SDR loss)
        best_val_pesq = -best_val_loss if best_val_loss != float('inf') else 0.0
        return checkpoint.get('epoch', 0), best_val_pesq


def load_inference_checkpoint(
    model: nn.Module,
    checkpoint_path: PathLike,
    device: torch.device,
) -> nn.Module:
    """Load weights into ``model`` from a checkpoint, supporting both training
    checkpoints (``model_state_dict`` / ``generator``) and bare ``state_dict``
    files."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'generator' in checkpoint:
        model.load_state_dict(checkpoint['generator'])
    else:
        model.load_state_dict(checkpoint)

    print("Checkpoint loaded successfully!")
    return model


def resolve_checkpoint_dir(checkpoint_dir: PathLike) -> Tuple[str, str]:
    """Auto-discover ``(config.yaml, checkpoint.pt)`` inside a directory.

    Config priority: ``temp_model_config_rank*.yaml`` →
    ``temp_model_config.yaml`` → ``config.yaml``.
    Checkpoint priority: ``best_model.pt`` → ``latest.pt`` → newest
    ``checkpoint_epoch_*.pt``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    config_candidates = list(sorted(checkpoint_dir.glob('temp_model_config_rank*.yaml')))
    if (checkpoint_dir / 'temp_model_config.yaml').exists():
        config_candidates.append(checkpoint_dir / 'temp_model_config.yaml')
    if (checkpoint_dir / 'config.yaml').exists():
        config_candidates.append(checkpoint_dir / 'config.yaml')

    config_path = next((c for c in config_candidates if c.exists()), None)
    if not config_path:
        print(f"Files in {checkpoint_dir}:")
        for f in checkpoint_dir.iterdir():
            print(f"  - {f.name}")
        raise FileNotFoundError(f"No config file found in {checkpoint_dir}")
    print(f"Found config: {config_path}")

    checkpoint_candidates = [
        checkpoint_dir / 'best_model.pt',
        checkpoint_dir / 'latest.pt',
    ]
    epoch_checkpoints = sorted(
        checkpoint_dir.glob('checkpoint_epoch_*.pt'),
        key=lambda x: int(x.stem.split('_')[-1]),
    )
    if epoch_checkpoints:
        checkpoint_candidates.append(epoch_checkpoints[-1])

    checkpoint_path = next((c for c in checkpoint_candidates if c.exists()), None)
    if not checkpoint_path:
        raise FileNotFoundError(f"No checkpoint file found in {checkpoint_dir}")

    return str(config_path), str(checkpoint_path)


# ================================
# TensorBoard Visualization
# ================================
def log_audio_and_spectrogram_batch(
    writer: Any,
    epoch: int,
    samples: Sequence[Mapping[str, Any]],
    sample_rate: int = 16000
) -> None:
    """Log multiple audio samples with waveforms and spectrograms to TensorBoard"""
    
    # Process each sample
    for idx, sample in enumerate(samples):
        clean = sample['clean'].squeeze().cpu().numpy()
        noisy = sample['noisy'].squeeze().cpu().numpy()
        enhanced = sample['enhanced'].squeeze().cpu().numpy()
        filename = sample.get('filename', f'sample_{idx}')
        
        # Log audio
        writer.add_audio(f'Audio_{idx}/clean', clean, epoch, sample_rate=sample_rate)
        writer.add_audio(f'Audio_{idx}/noisy', noisy, epoch, sample_rate=sample_rate)
        writer.add_audio(f'Audio_{idx}/enhanced', enhanced, epoch, sample_rate=sample_rate)
        
        # Create time axis for full audio length
        duration = len(clean) / sample_rate
        time_axis = np.linspace(0, duration, len(clean))
        
        # Create comprehensive figure with waveforms and spectrograms
        fig = plt.figure(figsize=(20, 12))
        
        # === Waveform plots (top row) ===
        # Noisy waveform
        ax1 = plt.subplot(4, 3, 1)
        ax1.plot(time_axis, noisy, color='gray', alpha=0.7, linewidth=0.5)
        ax1.set_title('Noisy Waveform')
        ax1.set_ylabel('Amplitude')
        ax1.set_xlim([0, duration])
        ax1.grid(True, alpha=0.3)
        
        # Clean waveform
        ax2 = plt.subplot(4, 3, 2)
        ax2.plot(time_axis, clean, color='blue', alpha=0.7, linewidth=0.5)
        ax2.set_title('Clean Waveform')
        ax2.set_xlim([0, duration])
        ax2.grid(True, alpha=0.3)
        
        # Enhanced waveform
        ax3 = plt.subplot(4, 3, 3)
        ax3.plot(time_axis, enhanced, color='green', alpha=0.7, linewidth=0.5)
        ax3.set_title('Enhanced Waveform')
        ax3.set_xlim([0, duration])
        ax3.grid(True, alpha=0.3)
        
        # === Overlay comparison (second row) ===
        # Clean vs Enhanced overlay
        ax4 = plt.subplot(4, 3, 4)
        ax4.plot(time_axis, clean, color='blue', alpha=0.6, linewidth=0.5, label='Clean')
        ax4.plot(time_axis, enhanced, color='green', alpha=0.6, linewidth=0.5, label='Enhanced')
        ax4.set_title('Clean vs Enhanced Overlay')
        ax4.set_ylabel('Amplitude')
        ax4.set_xlim([0, duration])
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # Difference plot (Enhanced - Clean)
        ax5 = plt.subplot(4, 3, 5)
        difference = enhanced - clean
        ax5.plot(time_axis, difference, color='red', alpha=0.7, linewidth=0.5)
        ax5.set_title('Difference (Enhanced - Clean)')
        ax5.set_xlim([0, duration])
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Noisy vs Enhanced overlay
        ax6 = plt.subplot(4, 3, 6)
        ax6.plot(time_axis, noisy, color='gray', alpha=0.6, linewidth=0.5, label='Noisy')
        ax6.plot(time_axis, enhanced, color='green', alpha=0.6, linewidth=0.5, label='Enhanced')
        ax6.set_title('Noisy vs Enhanced Overlay')
        ax6.set_xlim([0, duration])
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
        
        # === Spectrograms (third row) ===
        # Noisy spectrogram
        ax7 = plt.subplot(4, 3, 7)
        D_noisy = librosa.stft(noisy, n_fft=1024, hop_length=256)
        D_noisy_db = librosa.amplitude_to_db(np.abs(D_noisy), ref=np.max)
        img7 = librosa.display.specshow(D_noisy_db, sr=sample_rate, hop_length=256, 
                                        x_axis='time', y_axis='hz', ax=ax7)
        ax7.set_title('Noisy Spectrogram')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Frequency (Hz)')
        fig.colorbar(img7, ax=ax7, format='%+2.0f dB')
        
        # Clean spectrogram
        ax8 = plt.subplot(4, 3, 8)
        D_clean = librosa.stft(clean, n_fft=1024, hop_length=256)
        D_clean_db = librosa.amplitude_to_db(np.abs(D_clean), ref=np.max)
        img8 = librosa.display.specshow(D_clean_db, sr=sample_rate, hop_length=256, 
                                        x_axis='time', y_axis='hz', ax=ax8)
        ax8.set_title('Clean Spectrogram')
        ax8.set_xlabel('Time (s)')
        fig.colorbar(img8, ax=ax8, format='%+2.0f dB')
        
        # Enhanced spectrogram
        ax9 = plt.subplot(4, 3, 9)
        D_enhanced = librosa.stft(enhanced, n_fft=1024, hop_length=256)
        D_enhanced_db = librosa.amplitude_to_db(np.abs(D_enhanced), ref=np.max)
        img9 = librosa.display.specshow(D_enhanced_db, sr=sample_rate, hop_length=256, 
                                        x_axis='time', y_axis='hz', ax=ax9)
        ax9.set_title('Enhanced Spectrogram')
        ax9.set_xlabel('Time (s)')
        fig.colorbar(img9, ax=ax9, format='%+2.0f dB')
        
        # === Spectrogram difference (fourth row) ===
        # Difference spectrogram (Enhanced - Clean)
        ax10 = plt.subplot(4, 3, 10)
        D_diff = D_enhanced_db - D_clean_db
        img10 = ax10.imshow(D_diff, aspect='auto', origin='lower', 
                           extent=[0, duration, 0, sample_rate/2],
                           cmap='RdBu_r', vmin=-20, vmax=20)
        ax10.set_title('Spectrogram Difference (Enhanced - Clean)')
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('Frequency (Hz)')
        fig.colorbar(img10, ax=ax10, format='%+2.0f dB')
        
        # Noise reduction visualization (Noisy - Enhanced)
        ax11 = plt.subplot(4, 3, 11)
        D_noise_reduction = D_noisy_db - D_enhanced_db
        img11 = ax11.imshow(D_noise_reduction, aspect='auto', origin='lower',
                           extent=[0, duration, 0, sample_rate/2],
                           cmap='hot', vmin=0, vmax=30)
        ax11.set_title('Noise Reduction (Noisy - Enhanced)')
        ax11.set_xlabel('Time (s)')
        ax11.set_ylabel('Frequency (Hz)')
        fig.colorbar(img11, ax=ax11, format='%+2.0f dB')
        
        # Statistics
        ax12 = plt.subplot(4, 3, 12)
        ax12.axis('off')
        stats_text = f"""Sample Statistics:
        
            Filename: {filename}
            Duration: {duration:.2f} seconds
            Sample Rate: {sample_rate} Hz

            RMS Levels:
            Noisy:    {np.sqrt(np.mean(noisy**2)):.4f}
            Clean:    {np.sqrt(np.mean(clean**2)):.4f}
            Enhanced: {np.sqrt(np.mean(enhanced**2)):.4f}

            Peak Levels:
            Noisy:    {np.max(np.abs(noisy)):.4f}
            Clean:    {np.max(np.abs(clean)):.4f}
            Enhanced: {np.max(np.abs(enhanced)):.4f}

            SNR Improvement:
            {20*np.log10(np.sqrt(np.mean(clean**2))/np.sqrt(np.mean(difference**2)) + 1e-8):.2f} dB"""
                    
        ax12.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                 verticalalignment='center')
        
        plt.suptitle(f'Sample {idx}: {filename} - Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        writer.add_figure(f'DetailedAnalysis_{idx}/full', fig, epoch)
        plt.close()
    
    # Create a compact grid view of all samples
    if len(samples) > 1:
        fig, axes = plt.subplots(len(samples), 4, figsize=(20, 5*len(samples)))
        if len(samples) == 1:
            axes = axes.reshape(1, -1)
        
        for sample_idx, sample in enumerate(samples):
            clean = sample['clean'].squeeze().cpu().numpy()
            noisy = sample['noisy'].squeeze().cpu().numpy()
            enhanced = sample['enhanced'].squeeze().cpu().numpy()
            filename = sample.get('filename', f'sample_{sample_idx}')
            duration = len(clean) / sample_rate
            time_axis = np.linspace(0, duration, len(clean))
            
            # Waveform overlay
            ax_wave = axes[sample_idx, 0] if len(samples) > 1 else axes[0]
            ax_wave.plot(time_axis, clean, 'b-', alpha=0.6, linewidth=0.5, label='Clean')
            ax_wave.plot(time_axis, enhanced, 'g-', alpha=0.6, linewidth=0.5, label='Enhanced')
            ax_wave.set_xlim([0, duration])
            ax_wave.set_ylabel(f'S{sample_idx}: {filename[:15]}', fontsize=9)
            if sample_idx == 0:
                ax_wave.set_title('Waveform Overlay')
                ax_wave.legend(loc='upper right', fontsize=8)
            if sample_idx == len(samples)-1:
                ax_wave.set_xlabel('Time (s)')
            ax_wave.grid(True, alpha=0.3)
            
            # Spectrograms
            for ax_idx, (audio, title) in enumerate([(noisy, 'Noisy'), 
                                                      (clean, 'Clean'),
                                                      (enhanced, 'Enhanced')]):
                ax = axes[sample_idx, ax_idx+1] if len(samples) > 1 else axes[ax_idx+1]
                D = librosa.stft(audio, n_fft=1024, hop_length=256)
                D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display.specshow(D_db, sr=sample_rate, hop_length=256, 
                                               x_axis='time' if sample_idx == len(samples)-1 else None, 
                                               y_axis=None, ax=ax)
                if sample_idx == 0:
                    ax.set_title(f'{title} Spectrogram')
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        plt.suptitle(f'All Validation Samples - Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        writer.add_figure('CompactGrid/all_samples', fig, epoch)
        plt.close()
