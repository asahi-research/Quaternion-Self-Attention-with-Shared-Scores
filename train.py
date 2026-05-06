"""
Speech Enhancement Training Script
Supports multi-GPU distributed training, TensorBoard logging, and checkpoint management
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.generator import QuaternionGenerator
from models.loss import (
    si_sdr_loss, MRSTFTLoss,
    rms_loss, complex_l1_loss, pesq_loss
)
from pesq import pesq
from utils import (
    setup_distributed, cleanup_distributed,
    count_parameters, save_checkpoint, load_checkpoint,
    log_audio_and_spectrogram_batch
)
from dataloader import VoiceBankDemandDataset, ValidationDataset


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================================
# Training Functions
# ================================
def train_epoch(
    rank: int,
    epoch: int,
    generator: torch.nn.Module,
    train_loader: DataLoader,
    optimizers: Mapping[str, torch.optim.Optimizer],
    loss_fns: Mapping[str, Any],
    config: Mapping[str, Any],
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """Train one epoch"""
    generator.train()
    
    g_opt = optimizers['generator']
    
    # Loss weights
    w = config['loss_weights']
    
    # Progress bar (only on rank 0)
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} Train")
    else:
        pbar = train_loader
    
    # Accumulate losses
    losses_acc = {
        'g_loss': 0,
        'g_multi_stft': 0, 'g_sisdr': 0, 'g_rms': 0,
        'g_complex_l1': 0, 'g_pesq': 0
    }
    
    # Create window once outside the loop
    n_fft = config.get('n_fft', 400)
    hop = config.get('hop_length', 100)
    window_type = config.get('window_type', 'hann')
    
    # Create window based on window_type
    if window_type == 'hann':
        window = torch.hann_window(n_fft).cuda(rank)
    elif window_type == 'hamming':
        window = torch.hamming_window(n_fft).cuda(rank)
    elif window_type == 'blackman':
        window = torch.blackman_window(n_fft).cuda(rank)
    elif window_type == 'bartlett':
        window = torch.bartlett_window(n_fft).cuda(rank)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    
    for batch_idx, batch in enumerate(pbar):
        # VoiceBankDemandDataset returns: [features_4ch, noisy_stft, noisy_audio, clean_audio, scale_factor]
        features_4ch, noisy_stft, noisy_audio, clean_audio, scale_factors = batch
        features_4ch = features_4ch.cuda(rank)  # [B, 4, T, F] - already quaternion features
        noisy_stft = noisy_stft.cuda(rank)  # [B, F, T] - complex STFT
        clean = clean_audio.cuda(rank)  # [B, 1, T]
        noisy = noisy_audio.cuda(rank)  # [B, 1, T]
        scale_factors = scale_factors.cuda(rank)  # [B]
        
        batch_size = clean.shape[0]
        
        # Generator outputs mask
        mask = generator(features_4ch)  # [B, 2, F, T] - magnitude and phase masks
        
        # Apply complex mask to noisy STFT
        # Split mask into magnitude and phase components
        mag_mask = mask[:, 0, :, :]  # [B, F, T]
        phase_mask = mask[:, 1, :, :]  # [B, F, T]
        
        # Apply magnitude mask
        enhanced_mag = torch.abs(noisy_stft) * mag_mask
        
        # Apply phase mask (as rotation)
        noisy_phase = torch.angle(noisy_stft)
        enhanced_phase = noisy_phase + phase_mask * torch.pi  # Convert [-1,1] to [-π, π]
        
        # Reconstruct complex spectrogram
        enhanced_stft = enhanced_mag * torch.exp(1j * enhanced_phase)
        
        # Get clean STFT for complex losses
        clean_stft = torch.stft(clean.squeeze(1), n_fft, hop, window=window,
                               onesided=True, return_complex=True)
        clean_real = clean_stft.real  # [B, F, T]
        clean_imag = clean_stft.imag  # [B, F, T]
        
        # Extract real and imaginary parts from enhanced output
        real_gen_ft = enhanced_stft.real  # [B, F, T]
        imag_gen_ft = enhanced_stft.imag  # [B, F, T]
        
        # Convert back to waveform using torch.istft
        fake = torch.istft(enhanced_stft, n_fft, hop, window=window, 
                          onesided=True, length=clean.shape[-1]).unsqueeze(1)  # [B, 1, T]
        
        # Keep clean normalized for consistent scale comparison (clean is already normalized from dataset)
        # clean is already [B, 1, T] and normalized
        
        # ===== Loss calculations =====
        # Multi-resolution STFT loss
        g_multi_stft = loss_fns['stft'](fake, clean)

        # SI-SDR loss
        g_sisdr = si_sdr_loss(fake, clean)

        # RMS loss for volume matching
        g_rms = rms_loss(fake, clean)

        # Complex L1 loss
        g_complex_l1 = complex_l1_loss(real_gen_ft, imag_gen_ft, clean_real, clean_imag)
        
        # PESQ loss for perceptual quality
        if 'pesq' in loss_fns and w.get('pesq', 0.0) > 0:
            g_pesq = loss_fns['pesq'](clean.squeeze(1), fake.squeeze(1))
        else:
            g_pesq = torch.tensor(0.0).cuda(rank)
        
        # Total generator loss
        g_loss = (w.get('multi_stft', 2.0) * g_multi_stft +
                  w.get('sisdr', 5.0) * g_sisdr +
                  w.get('rms', 1.0) * g_rms +
                  w.get('complex_l1', 1.0) * g_complex_l1 +
                  w.get('pesq', 0.0) * g_pesq)
        
        g_opt.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), config['grad_clip'])
        g_opt.step()
        
        # Accumulate losses
        losses_acc['g_loss'] += g_loss.item()
        losses_acc['g_multi_stft'] += g_multi_stft.item()
        losses_acc['g_sisdr'] += g_sisdr.item()
        losses_acc['g_rms'] += g_rms.item()
        losses_acc['g_complex_l1'] += g_complex_l1.item()
        losses_acc['g_pesq'] += g_pesq.item() if isinstance(g_pesq, torch.Tensor) else 0
        
        # Update progress bar
        if rank == 0:
            si_sdr_db = -g_sisdr.item()
            pbar.set_postfix({
                'G': f"{g_loss.item():.3f}",
                'MSTFT': f"{g_multi_stft.item():.3f}",
                'SI-SDR': f"{si_sdr_db:.2f}dB",
                'CL1': f"{g_complex_l1.item():.3f}",
                'PESQ': f"{g_pesq.item():.3f}" if isinstance(g_pesq, torch.Tensor) and g_pesq.item() > 0 else "N/A"
            })
        
        # Log to TensorBoard (every N steps)
        if rank == 0 and writer and batch_idx % config['log_interval'] == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Total_Loss', g_loss.item(), global_step)
            writer.add_scalar('Loss/Multi_STFT', g_multi_stft.item(), global_step)
            writer.add_scalar('Loss/SI-SDR', g_sisdr.item(), global_step)
            writer.add_scalar('Loss/SI-SDR_dB', -g_sisdr.item(), global_step)
            writer.add_scalar('Loss/RMS', g_rms.item(), global_step)
            writer.add_scalar('Loss/Complex_L1', g_complex_l1.item(), global_step)
            writer.add_scalar('Loss/PESQ', g_pesq.item(), global_step)
            
    
    # Average losses
    n_batches = len(train_loader)
    for key in losses_acc:
        losses_acc[key] /= n_batches
    
    return losses_acc


def validate(
    rank: int,
    epoch: int,
    generator: torch.nn.Module,
    val_loader: DataLoader,
    loss_fns: Mapping[str, Any],
    config: Mapping[str, Any],
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """Validation loop"""
    generator.eval()
    
    val_losses = {'multi_stft': 0, 'sisdr': 0, 'pesq': 0, 'pesq_count': 0}
    
    # Save sample audios and spectrograms
    saved_samples = []
    max_samples = 4  # Save up to 4 samples
    
    # Create window once outside the loop
    n_fft = config.get('n_fft', 400)
    hop = config.get('hop_length', 100)
    window_type = config.get('window_type', 'hann')
    
    # Create window based on window_type
    if window_type == 'hann':
        window = torch.hann_window(n_fft).cuda(rank)
    elif window_type == 'hamming':
        window = torch.hamming_window(n_fft).cuda(rank)
    elif window_type == 'blackman':
        window = torch.blackman_window(n_fft).cuda(rank)
    elif window_type == 'bartlett':
        window = torch.bartlett_window(n_fft).cuda(rank)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch:03d} Val", disable=(rank != 0))):
            # ValidationDataset returns: [features_4ch, noisy_stft, clean_norm, noisy_norm, filename, scale_factor]
            features_4ch, noisy_stft, clean_norm, noisy_norm, filename, scale_factors = batch
            features_4ch = features_4ch.cuda(rank)  # [B, 4, T, F] - already quaternion features
            noisy_stft = noisy_stft.cuda(rank)  # [B, F, T] - complex STFT
            clean = clean_norm.unsqueeze(1).cuda(rank)  # Keep normalized clean [B, 1, T]
            noisy = noisy_norm.unsqueeze(1).cuda(rank)  # Keep normalized noisy [B, 1, T]
            scale_factors = scale_factors.cuda(rank)  # [B] - for reference but not used
            
            batch_size = clean.shape[0]
            
            # Generate mask and apply to noisy STFT
            mask = generator(features_4ch)  # [B, 2, F, T] - magnitude and phase masks
            
            # Apply complex mask to noisy STFT
            mag_mask = mask[:, 0, :, :]  # [B, F, T]
            phase_mask = mask[:, 1, :, :]  # [B, F, T]
            
            # Apply magnitude mask
            enhanced_mag = torch.abs(noisy_stft) * mag_mask
            
            # Apply phase mask (as rotation)
            noisy_phase = torch.angle(noisy_stft)
            enhanced_phase = noisy_phase + phase_mask * torch.pi  # Convert [-1,1] to [-π, π]
            
            # Reconstruct complex spectrogram
            enhanced_stft = enhanced_mag * torch.exp(1j * enhanced_phase)
            
            # Convert back to waveform using torch.istft
            fake = torch.istft(enhanced_stft, n_fft, hop, window=window, 
                              onesided=True, length=clean.shape[-1]).unsqueeze(1)  # [B, 1, T]
            
            # Calculate losses
            multi_stft_loss = loss_fns['stft'](fake, clean)
            sisdr_loss = si_sdr_loss(fake, clean)

            val_losses['multi_stft'] += multi_stft_loss.item()
            val_losses['sisdr'] += sisdr_loss.item()
            
            # Calculate PESQ (only for first sample in batch for efficiency)
            if batch_size > 0:
                try:
                    # Convert to numpy and denormalize for PESQ calculation
                    clean_np = clean[0].squeeze().cpu().numpy()
                    fake_np = fake[0].squeeze().cpu().numpy()
                    
                    # PESQ requires 16kHz audio
                    pesq_score = pesq(16000, clean_np, fake_np, 'wb')  # Wide band
                    val_losses['pesq'] += pesq_score
                    val_losses['pesq_count'] += 1
                except Exception as e:
                    # PESQ calculation can fail for various reasons
                    pass
            
            # Save samples for visualization (up to max_samples)
            if rank == 0 and writer and len(saved_samples) < max_samples:
                for i in range(min(batch_size, max_samples - len(saved_samples))):
                    saved_samples.append({
                        'clean': clean[i],
                        'noisy': noisy[i],
                        'enhanced': fake[i],
                        'filename': filename[i] if isinstance(filename, list) else filename
                    })
                    if len(saved_samples) >= max_samples:
                        break
    
    # Average losses
    n_batches = len(val_loader)
    for key in val_losses:
        if key == 'pesq':
            if val_losses['pesq_count'] > 0:
                val_losses[key] /= val_losses['pesq_count']
            else:
                val_losses[key] = 0  # No valid PESQ scores
        elif key != 'pesq_count':
            val_losses[key] /= n_batches
    
    # Log to TensorBoard
    if rank == 0 and writer:
        writer.add_scalar('Val/multi_stft', val_losses['multi_stft'], epoch)
        writer.add_scalar('Val/sisdr', val_losses['sisdr'], epoch)
        writer.add_scalar('Val/SI-SDR_dB', -val_losses['sisdr'], epoch)
        if val_losses['pesq'] > 0:
            writer.add_scalar('Val/PESQ', val_losses['pesq'], epoch)
        
        # Log audio and spectrograms for saved samples
        if saved_samples:
            log_audio_and_spectrogram_batch(writer, epoch, saved_samples, config['sample_rate'])
    
    return val_losses


# ================================
# Main Training Process
# ================================

def train_worker(rank: int, world_size: int, config: Dict[str, Any]) -> None:
    """Training worker for each GPU"""
    
    # Set global seed for reproducibility
    global_seed = config.get('global_seed', 42)
    set_seed(global_seed + rank)  # Different seed for each rank to avoid identical initialization
    
    # Setup distributed training
    if world_size > 1:
        master_port = config.get('master_port', 12381)
        setup_distributed(rank, world_size, master_port)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer (only on rank 0)
    writer = None
    if rank == 0:
        log_dir = output_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        # Save config as YAML
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Create datasets using dataloder.py implementations
    full_train_dataset = VoiceBankDemandDataset(
        data_dir=None,
        clean_dir=config['train_clean_dir'],
        noisy_dir=config['train_noisy_dir'],
        cut_len=config['segment_length'],
        n_fft=config['n_fft'],
        hop=config['hop_length'],
        return_all=True,
        normalize='peak',  # Use peak normalization like original code
        window_type=config.get('window_type', 'hann')
    )
    
    # Create validation dataset from train files (using full-length audio)
    clean_files = sorted(list(Path(config['train_clean_dir']).glob('*.wav')))
    noisy_files = sorted(list(Path(config['train_noisy_dir']).glob('*.wav')))
    
    val_dataset_full = ValidationDataset(
        clean_files=[str(f) for f in clean_files],
        noisy_files=[str(f) for f in noisy_files],
        n_fft=config['n_fft'],
        hop=config['hop_length'],
        normalize='peak',  # Use peak normalization
        window_type=config.get('window_type', 'hann')
    )
    
    # Split train dataset for validation with fixed seed
    val_split_size = config.get('val_split_size', 100)
    val_seed = config.get('val_seed', 42)
    
    # Set seed for reproducible split
    torch.manual_seed(val_seed)
    np.random.seed(val_seed)
    
    # Create indices for train and validation
    total_size = len(full_train_dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    val_indices = indices[:val_split_size]
    train_indices = indices[val_split_size:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    if rank == 0:
        print(f"Using {len(train_dataset)} samples for training, {len(val_dataset)} for validation (full-length)")
    
    # Create data loaders
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    if 'model' in config:
        # Save model config to temp file (unique per process)
        temp_config = {'model': config['model']}
        temp_config_path = output_dir / f'temp_model_config_rank{rank}.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(temp_config, f)

        generator = QuaternionGenerator(str(temp_config_path)).cuda(rank)
    else:
        raise ValueError("Model configuration not found in config")
    
    # No discriminators
    
    # Print model parameters (rank 0 only)
    if rank == 0:
        g_params = count_parameters(generator)
        
        print("\nModel Parameters:")
        print(f"  Generator: {g_params:,} parameters")
        print("="*60)

    # Wrap with DDP if distributed
    if world_size > 1:
        generator = DDP(generator, device_ids=[rank])
    
    # Create optimizers
    g_params = generator.parameters()
    
    # Only generator optimizer
    optimizers = {
        'generator': torch.optim.AdamW(g_params, lr=config['g_lr'], betas=config['betas'])
    }
    
    # Create loss functions (multi-resolution STFT and PESQ)
    loss_fns = {
        'stft': MRSTFTLoss(
            fft_sizes=config['stft_fft_sizes'],
            hop_sizes=config['stft_hop_sizes'],
            win_lengths=config['stft_win_lengths'],
            sc_weight=config.get('stft_sc_weight', 0.3),
            mag_weight=config.get('stft_mag_weight', 0.7)
        ).cuda(rank)
    }
    
    # Add PESQ loss if configured
    if config['loss_weights'].get('pesq', 0.0) > 0:
        loss_fns['pesq'] = lambda real, gen: pesq_loss(
            real, gen, 
            sample_rate=config['sample_rate'],
            alpha=config.get('pesq_alpha', 0.5)
        )
    
    # Load checkpoint if specified
    start_epoch = 0
    best_val_pesq = 0.0  # PESQ ranges from -0.5 to 4.5, higher is better
    if config.get('resume_checkpoint'):
        start_epoch, best_val_pesq = load_checkpoint(
            config['resume_checkpoint'],
            generator.module if world_size > 1 else generator,
            {},
            optimizers
        )
        print(f"Resumed from epoch {start_epoch} with best val PESQ {best_val_pesq:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, config['num_epochs']):
        if world_size > 1 and train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_losses = train_epoch(
            rank, epoch, generator,
            train_loader, optimizers, loss_fns, config, writer
        )
        
        # Validate
        val_losses = validate(
            rank, epoch, generator, val_loader, loss_fns, config, writer
        )
        
        # Print epoch summary (rank 0 only)
        if rank == 0:
            print(f"\nEpoch {epoch:03d} Summary:")
            print(f"  Train - G: {train_losses['g_loss']:.4f}, "
                  f"MSTFT: {train_losses['g_multi_stft']:.4f}, SI-SDR: {-train_losses['g_sisdr']:.2f} dB")
            pesq_str = f", PESQ: {val_losses['pesq']:.3f}" if val_losses['pesq'] > 0 else ""
            print(f"  Val   - MSTFT: {val_losses['multi_stft']:.4f}, SI-SDR: {-val_losses['sisdr']:.2f} dB{pesq_str}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'generator': generator.module.state_dict() if world_size > 1 else generator.state_dict(),
                'g_optimizer': optimizers['generator'].state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_pesq': best_val_pesq,
                'config': config
            }
            
            # Save periodic checkpoint
            if (epoch + 1) % config['save_interval'] == 0:
                save_checkpoint(
                    checkpoint,
                    output_dir / f'checkpoint_epoch_{epoch+1:04d}.pt'
                )
            
            # Save best model based on PESQ (if available) or SI-SDR
            if val_losses['pesq'] > 0:  # PESQ is available
                current_val_pesq = val_losses['pesq']
                if current_val_pesq > best_val_pesq:  # Higher PESQ is better
                    best_val_pesq = current_val_pesq
                    checkpoint['best_val_pesq'] = best_val_pesq
                    save_checkpoint(
                        checkpoint,
                        output_dir / 'best_model.pt'
                    )
                    print(f"  --> New best model! PESQ: {best_val_pesq:.3f}")
            else:  # Fall back to SI-SDR if PESQ not available
                current_val_sisdr = -val_losses['sisdr']  # Negate for higher-is-better
                if best_val_pesq == 0.0 and current_val_sisdr > -100:  # First epoch or no PESQ
                    best_val_pesq = current_val_sisdr  # Use SI-SDR as proxy
                    checkpoint['best_val_pesq'] = best_val_pesq
                    save_checkpoint(
                        checkpoint,
                        output_dir / 'best_model.pt'
                    )
                    print(f"  --> New best model! SI-SDR: {current_val_sisdr:.2f} dB")
            
            # Save latest checkpoint
            save_checkpoint(checkpoint, output_dir / 'latest.pt')
    
    # Cleanup
    if rank == 0 and writer:
        writer.close()
    
    if world_size > 1:
        cleanup_distributed()


def main() -> None:
    """Main function with simplified argument interface"""
    parser = argparse.ArgumentParser(
        description='Quaternion Self-Attention with Shared Score Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required config file
    parser.add_argument('--config', type=str, required=True,
                       help='YAML configuration file path')
    
    # Common overrides
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size per GPU')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--gpus', type=str, default='0',
                       help='GPU IDs (e.g., "0" or "0,1")')
    parser.add_argument('--workers', type=int, default=None,
                       help='Override number of data workers')
    
    # Paths
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=None,
                       help='Log every N iterations')
    parser.add_argument('--save-interval', type=int, default=None,
                       help='Save checkpoint every N epochs')
    
    # Validation
    parser.add_argument('--val-size', type=int, default=None,
                       help='Number of validation samples from train set')
    parser.add_argument('--val-seed', type=int, default=None,
                       help='Seed for validation split')
    parser.add_argument('--global-seed', type=int, default=1234,
                       help='Global seed for reproducibility')

    
    # Quick modes
    parser.add_argument('--test', action='store_true',
                       help='Test mode: 1 epoch, frequent logging')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode: verbose output')
    
    args = parser.parse_args()
    
    # Load base config from file
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {args.config}')
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            file_config = yaml.safe_load(f)
        else:
            file_config = json.load(f)
    
    # Build flattened config
    config = {
        # Data paths
        'train_clean_dir': file_config['data']['train']['clean_dir'],
        'train_noisy_dir': file_config['data']['train']['noisy_dir'],
        'val_clean_dir': file_config['data']['train']['clean_dir'],  # Use train data for validation split
        'val_noisy_dir': file_config['data']['train']['noisy_dir'],  # Use train data for validation split
        
        # Model
        'model': file_config.get('model', {}),
        
        # Training
        'batch_size': file_config['training']['batch_size'],
        'num_epochs': file_config['training']['num_epochs'],
        'g_lr': file_config['training']['generator_lr'],
        'betas': file_config['training']['betas'],
        'grad_clip': file_config['training']['grad_clip'],
        
        # Loss (simplified to only 3 losses)
        'loss_weights': file_config['loss']['weights'],
        'stft_fft_sizes': file_config['loss']['stft']['fft_sizes'],
        'stft_hop_sizes': file_config['loss']['stft']['hop_sizes'],
        'stft_win_lengths': file_config['loss']['stft']['win_lengths'],
        'stft_sc_weight': file_config['loss']['stft'].get('sc_weight', 0.3),
        'stft_mag_weight': file_config['loss']['stft'].get('mag_weight', 0.7),
        'pesq_alpha': file_config['loss'].get('pesq_alpha', 0.5),
        
        # Audio
        'segment_length': file_config['audio']['segment_length'],
        'sample_rate': file_config['audio']['sample_rate'],
        'n_fft': file_config['audio'].get('n_fft', 400),
        'hop_length': file_config['audio'].get('hop_length', 100),
        'window_type': file_config['audio'].get('window_type', 'hann'),
        
        # System
        'num_workers': file_config['system']['num_workers'],
        'master_port': file_config['system'].get('master_port', 12381),
        'output_dir': file_config['checkpoint']['output_dir'],
        'save_interval': file_config['checkpoint']['save_interval'],
        'resume_checkpoint': file_config['checkpoint'].get('resume', None),
        
        # Logging
        'log_interval': file_config['logging']['log_interval'],
        
        # Validation
        'val_split_size': file_config['validation'].get('val_split_size', 100),
        'val_seed': file_config['validation'].get('val_seed', 42),
        'global_seed': file_config['validation'].get('global_seed', 42),
    }
    
    # Apply command-line overrides
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    if args.workers is not None:
        config['num_workers'] = args.workers
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    if args.resume is not None:
        config['resume_checkpoint'] = args.resume
    if args.save_interval is not None:
        config['save_interval'] = args.save_interval
    if args.log_interval is not None:
        config['log_interval'] = args.log_interval
    if args.val_size is not None:
        config['val_split_size'] = args.val_size
    if args.val_seed is not None:
        config['val_seed'] = args.val_seed
    if args.global_seed is not None:
        config['global_seed'] = args.global_seed
    
    # Test mode overrides
    if args.test:
        config['num_epochs'] = 1
        config['log_interval'] = 10
        config['val_split_size'] = 10
        print("\n" + "="*60)
        print("TEST MODE: Running 1 epoch with frequent logging")
        print("="*60 + "\n")
    
    # Debug mode
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    world_size = len(gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    # Print configuration
    print("="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Config file: {args.config}")
    print(f"Output dir: {config['output_dir']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"GPUs: {gpu_ids}")
    print(f"Validation: {config['val_split_size']} samples (seed={config['val_seed']})")
    if config.get('resume_checkpoint'):
        print(f"Resuming from: {config['resume_checkpoint']}")
    print("="*60)
    
    # Start training
    if world_size > 1:
        mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, config)


if __name__ == '__main__':
    main()
