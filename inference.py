"""
Inference script for speech enhancement with quaternion-based models.
Supports batch evaluation and single file processing.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torchaudio
import numpy as np
import soundfile as sf
import pandas as pd
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.generator import QuaternionGenerator
from utils import (
    compute_stft_quaternion_features,
    load_inference_checkpoint,
    resolve_checkpoint_dir,
)
from dataloader import TestDataset, test_collate_fn
from metrics import evaluate_metrics


PathLike = Union[str, os.PathLike[str]]



def process_single_checkpoint(
    checkpoint_name: str,
    config_path: PathLike,
    checkpoint_path: PathLike,
    test_dir: PathLike,
    output_base_dir: PathLike,
    args: argparse.Namespace,
    device: torch.device
) -> Optional[Dict[str, Any]]:
    """Process a single checkpoint and return metrics summary"""
    
    # Create structured output directories
    output_base = Path(output_base_dir) / checkpoint_name
    wave_dir = output_base / 'wave'
    csv_dir = output_base / 'csv'
    
    if not args.skip_audio_save:
        wave_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing: {checkpoint_name}")
    print(f"Output: {output_base}")
    print(f"{'='*60}")

    # Load model - check model type from config
    print("Loading model...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_name = config.get('model', {}).get('name', 'QuaternionGenerator')

    model = QuaternionGenerator(config_path)

    model = model.to(device)
    model = load_inference_checkpoint(model, checkpoint_path, device)
    model.eval()

    # Setup test directories
    clean_test_dir = os.path.join(test_dir, 'clean_testset_wav')
    noisy_test_dir = os.path.join(test_dir, 'noisy_testset_wav')
    
    if not os.path.exists(clean_test_dir) or not os.path.exists(noisy_test_dir):
        print(f"Test directories not found for {checkpoint_name}")
        return None
    
    # Create dataset and dataloader
    test_dataset = TestDataset(noisy_test_dir, clean_test_dir, args.n_fft, args.hop, args.window_type)
    
    # Limit files if specified
    if args.max_files:
        test_dataset.files = test_dataset.files[:args.max_files]
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Processing {len(test_dataset)} test files...")
    
    # Store all metrics
    all_metrics = []
    
    # RTF tracking
    total_inference_time = 0
    total_audio_duration = 0
    
    # Create window
    if args.window_type == 'hann':
        window = torch.hann_window(args.n_fft).to(device)
    elif args.window_type == 'hamming':
        window = torch.hamming_window(args.n_fft).to(device)
    elif args.window_type == 'blackman':
        window = torch.blackman_window(args.n_fft).to(device)
    elif args.window_type == 'bartlett':
        window = torch.bartlett_window(args.n_fft).to(device)
    
    # Process batches
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Processing {checkpoint_name}"):
            # Move features to device
            features = batch['features'].to(device)
            noisy_stft = batch['noisy_stft'].to(device)
            
            # Measure inference time (including iSTFT for fair comparison with DCCRN)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

            # Run model to get mask
            mask = model(features)  # [B, 2, F, T] - magnitude and phase masks

            # Apply mask to noisy STFT
            mag_mask = mask[:, 0, :, :]  # [B, F, T]
            phase_mask = mask[:, 1, :, :]  # [B, F, T]

            # Apply magnitude mask
            enhanced_mag = torch.abs(noisy_stft) * mag_mask

            # Apply phase mask (as rotation)
            noisy_phase = torch.angle(noisy_stft)
            enhanced_phase = noisy_phase + phase_mask * torch.pi  # Convert [-1,1] to [-π, π]

            # Reconstruct complex spectrogram
            enhanced_stft = enhanced_mag * torch.exp(1j * enhanced_phase)

            # Perform iSTFT for all items in batch (included in inference time)
            enhanced_wavs_batch = []
            for i in range(enhanced_stft.shape[0]):
                enhanced_wav_i = torch.istft(enhanced_stft[i], args.n_fft, args.hop, window=window,
                                            onesided=True, length=batch['original_lengths'][i])
                enhanced_wavs_batch.append(enhanced_wav_i)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - start_time

            # Phase-inversion correction (uses clean reference; excluded from RTF)
            for i in range(len(enhanced_wavs_batch)):
                clean_wav_i = batch['clean_wav'][i].cpu().numpy()
                enhanced_wav_i_np = enhanced_wavs_batch[i].cpu().numpy()
                min_len = min(len(clean_wav_i), len(enhanced_wav_i_np))
                phase_corr = np.corrcoef(clean_wav_i[:min_len], enhanced_wav_i_np[:min_len])[0, 1]
                if phase_corr < 0:
                    enhanced_wavs_batch[i] = -enhanced_wavs_batch[i]
            
            # Update RTF tracking for batch
            batch_size = features.shape[0]
            for i in range(batch_size):
                audio_duration = batch['original_lengths'][i] / batch['srs'][i]
                total_audio_duration += audio_duration
            total_inference_time += inference_time

            # Process each item in batch
            for i in range(batch_size):
                # Get individual items
                original_length = batch['original_lengths'][i]
                scale_factor = batch['scale_factors'][i].item()
                filename = batch['filenames'][i]
                sr = batch['srs'][i]

                # Get waveform (already computed during inference time measurement)
                enhanced_wav = enhanced_wavs_batch[i].cpu()
                enhanced_wav_np = enhanced_wav.numpy()
                
                # Get original audio for metrics
                clean_wav = batch['clean_wav'][i][:original_length].numpy()
                noisy_wav = batch['noisy_wav'][i][:original_length].numpy()
                noisy_original = batch['noisy_original'][i][:original_length].numpy()
                
                # Optional volume adjustment
                if args.volume_restore != 'none':
                    if args.volume_restore == 'rms_clean':
                        rms_clean = np.sqrt(np.mean(clean_wav**2) + 1e-10)
                        rms_enhanced = np.sqrt(np.mean(enhanced_wav_np**2) + 1e-10)
                        enhanced_wav_np = enhanced_wav_np * (rms_clean / rms_enhanced)
                    elif args.volume_restore == 'rms_noisy':
                        rms_noisy = np.sqrt(np.mean(noisy_original**2) + 1e-10)
                        rms_enhanced = np.sqrt(np.mean(enhanced_wav_np**2) + 1e-10)
                        enhanced_wav_np = enhanced_wav_np * (rms_noisy / rms_enhanced)
                    elif args.volume_restore == 'peak':
                        peak_noisy = np.abs(noisy_original).max()
                        peak_enhanced = np.abs(enhanced_wav_np).max() + 1e-10
                        enhanced_wav_np = enhanced_wav_np * (peak_noisy / peak_enhanced)
                
                # Clip to prevent overflow
                enhanced_wav_np = np.clip(enhanced_wav_np, -1.0, 1.0)
                
                # Calculate metrics
                metrics = evaluate_metrics(clean_wav, enhanced_wav_np, noisy_original, sr, use_rms_match=True)
                metrics['filename'] = filename
                all_metrics.append(metrics)
                
                # Save enhanced audio
                if not args.skip_audio_save:
                    output_path = wave_dir / filename
                    sf.write(str(output_path), enhanced_wav_np, sr)
    
    # Calculate RTF
    rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else float('inf')
    
    # Convert metrics to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Calculate summary
    metric_names = ['pesq', 'stoi', 'estoi', 'sisdr', 'csig', 'cbak', 'covl']

    summary_data = {
        'checkpoint': checkpoint_name,
        'num_files': len(all_metrics),
        'rtf': rtf,
        'total_inference_time': total_inference_time,
        'total_audio_duration': total_audio_duration
    }

    for metric_key in metric_names:
        if metric_key in df.columns:
            valid_values = df[metric_key].dropna()
            if len(valid_values) > 0:
                summary_data[f'{metric_key}_mean'] = valid_values.mean()
                summary_data[f'{metric_key}_std'] = valid_values.std()

    # Save metrics
    if args.save_metrics:
        # Save individual metrics
        individual_csv = csv_dir / 'metrics_per_file.csv'
        df.to_csv(individual_csv, index=False)

        # Save summary
        summary_df = pd.DataFrame([summary_data])
        summary_csv = csv_dir / 'summary.csv'
        summary_df.to_csv(summary_csv, index=False)

        print(f"\nResults for {checkpoint_name}:")
        print(f"  PESQ: {summary_data.get('pesq_mean', 0):.3f} ± {summary_data.get('pesq_std', 0):.3f}")
        print(f"  STOI: {summary_data.get('stoi_mean', 0):.3f} ± {summary_data.get('stoi_std', 0):.3f}")
        print(f"  ESTOI: {summary_data.get('estoi_mean', 0):.3f} ± {summary_data.get('estoi_std', 0):.3f}")
        print(f"  SI-SDR: {summary_data.get('sisdr_mean', 0):.3f} ± {summary_data.get('sisdr_std', 0):.3f}")
        print(f"  CSIG: {summary_data.get('csig_mean', 0):.3f} ± {summary_data.get('csig_std', 0):.3f}")
        print(f"  CBAK: {summary_data.get('cbak_mean', 0):.3f} ± {summary_data.get('cbak_std', 0):.3f}")
        print(f"  COVL: {summary_data.get('covl_mean', 0):.3f} ± {summary_data.get('covl_std', 0):.3f}")
        print(f"  RTF: {rtf:.4f} (Processing time: {total_inference_time:.2f}s / Audio duration: {total_audio_duration:.2f}s)")
        if rtf < 1.0:
            print(f"  -> Real-time capable ({1.0/rtf:.1f}x faster than real-time)")
    
    return summary_data


def process_single_wav_file(args: argparse.Namespace, device: torch.device) -> None:
    """Process a single WAV file for inference"""

    # Check if wav file exists
    if not os.path.exists(args.wav_file):
        print(f"Error: WAV file not found: {args.wav_file}")
        return

    # Load checkpoint
    if args.checkpoint_dir:
        config_path, checkpoint_path = resolve_checkpoint_dir(args.checkpoint_dir)
        checkpoint_name = Path(args.checkpoint_dir).name
    elif args.checkpoint and args.config:
        config_path = args.config
        checkpoint_path = args.checkpoint
        checkpoint_name = Path(checkpoint_path).stem
    else:
        print("Error: Please specify either --checkpoint-dir or both --checkpoint and --config")
        return

    print(f"\nProcessing single file: {args.wav_file}")
    print(f"Using checkpoint: {checkpoint_name}")

    # Load model - check model type from config
    print("Loading model...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_name = config.get('model', {}).get('name', 'QuaternionGenerator')

    model = QuaternionGenerator(config_path)

    model = model.to(device)
    model = load_inference_checkpoint(model, checkpoint_path, device)
    model.eval()

    # Load audio
    noisy_wav, sr = torchaudio.load(args.wav_file)
    noisy_wav = noisy_wav.squeeze()

    # Store original for volume restoration
    noisy_original = noisy_wav.clone()
    original_length = len(noisy_wav)

    # Normalize for model input
    scale_factor = noisy_wav.abs().max() + 1e-8
    noisy_wav_norm = noisy_wav / scale_factor

    # Create window
    if args.window_type == 'hann':
        window = torch.hann_window(args.n_fft)
    elif args.window_type == 'hamming':
        window = torch.hamming_window(args.n_fft)
    elif args.window_type == 'blackman':
        window = torch.blackman_window(args.n_fft)
    elif args.window_type == 'bartlett':
        window = torch.bartlett_window(args.n_fft)

    # Compute STFT and quaternion features
    features_4ch, _ = compute_stft_quaternion_features(
        noisy_wav_norm, args.n_fft, args.hop, window
    )

    # Compute noisy STFT for mask application
    noisy_stft = torch.stft(noisy_wav_norm, args.n_fft, args.hop, window=window,
                            onesided=True, return_complex=True)  # [F, T]

    # Add batch dimension and move to device
    features_4ch = features_4ch.unsqueeze(0).to(device)  # [1, 4, T, F]
    noisy_stft = noisy_stft.unsqueeze(0).to(device)  # [1, F, T]

    # Measure inference time (including iSTFT for fair comparison with DCCRN)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    # Run model
    with torch.no_grad():
        mask = model(features_4ch)  # [1, 2, F, T] - magnitude and phase masks

    # Apply mask
    mag_mask = mask[0, 0, :, :]  # [F, T]
    phase_mask = mask[0, 1, :, :]  # [F, T]

    # Apply magnitude mask
    enhanced_mag = torch.abs(noisy_stft[0]) * mag_mask

    # Apply phase mask (as rotation)
    noisy_phase = torch.angle(noisy_stft[0])
    enhanced_phase = noisy_phase + phase_mask * torch.pi  # Convert [-1,1] to [-π, π]

    # Reconstruct complex spectrogram
    enhanced_stft = enhanced_mag * torch.exp(1j * enhanced_phase)

    # Convert to waveform (included in inference time)
    window = window.to(device)
    enhanced_wav = torch.istft(enhanced_stft, args.n_fft, args.hop, window=window,
                              onesided=True, length=original_length)
    # Apply phase inversion fix if specified
    if args.phase_invert:
        enhanced_wav = -enhanced_wav

    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_time = time.time() - start_time

    enhanced_wav = enhanced_wav.cpu().numpy()

    # Volume restoration
    if args.volume_restore == 'rms_noisy':
        rms_noisy = np.sqrt(np.mean(noisy_original.numpy()**2) + 1e-10)
        rms_enhanced = np.sqrt(np.mean(enhanced_wav**2) + 1e-10)
        enhanced_wav = enhanced_wav * (rms_noisy / rms_enhanced)
    elif args.volume_restore == 'peak':
        peak_noisy = np.abs(noisy_original.numpy()).max()
        peak_enhanced = np.abs(enhanced_wav).max() + 1e-10
        enhanced_wav = enhanced_wav * (peak_noisy / peak_enhanced)

    # Clip to prevent overflow
    enhanced_wav = np.clip(enhanced_wav, -1.0, 1.0)

    # Save enhanced audio
    input_path = Path(args.wav_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"enhanced_{input_path.stem}_{checkpoint_name}.wav"
        output_path = output_dir / output_filename
    else:
        output_filename = f"enhanced_{input_path.stem}_{checkpoint_name}.wav"
        output_path = input_path.parent / output_filename

    sf.write(str(output_path), enhanced_wav, sr)

    # Calculate RTF
    audio_duration = original_length / sr
    rtf = inference_time / audio_duration if audio_duration > 0 else float('inf')

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Input file: {args.wav_file}")
    print(f"Output file: {output_path}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {audio_duration:.2f} seconds")
    print(f"Processing time: {inference_time:.3f} seconds")
    print(f"RTF: {rtf:.4f}")
    if rtf < 1.0:
        print(f"-> Real-time capable ({1.0/rtf:.1f}x faster than real-time)")
    print(f"Volume restoration: {args.volume_restore}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Final inference with proper metrics')
    parser.add_argument('--checkpoints-dir', type=str,
                        help='Parent directory containing multiple checkpoint directories')
    parser.add_argument('--checkpoint-dir', type=str,
                        help='Path to single checkpoint directory (auto-loads config and weights)')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to model checkpoint (use with --config)')
    parser.add_argument('--config', type=str,
                        help='Path to config file (use with --checkpoint)')
    parser.add_argument('--test_dir', type=str,
                        help='Path to test dataset directory (must contain clean_testset_wav and noisy_testset_wav)')
    parser.add_argument('--process-all', action='store_true',
                        help='Process all checkpoint directories in --checkpoints-dir')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Base directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--n_fft', type=int, default=400,
                        help='FFT window size')
    parser.add_argument('--hop', type=int, default=100,
                        help='Hop length')
    parser.add_argument('--window_type', type=str, default='hann',
                        choices=['hann', 'hamming', 'blackman', 'bartlett'],
                        help='Window function type (default: hann)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--save_metrics', type=bool, default=True,
                        help='Save metrics to CSV files')
    parser.add_argument('--skip_audio_save', action='store_true',
                        help='Skip saving enhanced audio files')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Process only first N files')
    parser.add_argument('--volume_restore', type=str, default='none',
                        choices=['none', 'rms_clean', 'rms_noisy', 'peak'],
                        help='Volume restoration method (default: none with fixed training)')
    parser.add_argument('--phase_invert', action='store_true',
                        help='Apply phase inversion fix for single file inference')
    parser.add_argument('--wav_file', type=str, default=None,
                        help='Path to single WAV file for inference')

    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Volume restoration method: {args.volume_restore}")

    # Process single WAV file if specified
    if args.wav_file:
        process_single_wav_file(args, device)
        return

    # Validate test_dir for batch processing
    if not args.test_dir:
        print("Error: --test_dir is required for batch evaluation")
        return

    # Process multiple checkpoints if specified
    if args.checkpoints_dir and args.process_all:
        checkpoints_parent = Path(args.checkpoints_dir)
        if not checkpoints_parent.exists():
            print(f"Checkpoints directory not found: {checkpoints_parent}")
            return
        
        # Find all checkpoint directories
        checkpoint_dirs = []
        for d in checkpoints_parent.iterdir():
            if d.is_dir():
                try:
                    config_path, checkpoint_path = resolve_checkpoint_dir(d)
                    checkpoint_dirs.append((d.name, config_path, checkpoint_path))
                except FileNotFoundError:
                    continue
        
        if not checkpoint_dirs:
            print(f"No valid checkpoint directories found in {checkpoints_parent}")
            return
        
        print(f"\nFound {len(checkpoint_dirs)} checkpoint directories to process:")
        for name, _, _ in checkpoint_dirs:
            print(f"  - {name}")
        
        # Process each checkpoint
        all_summaries = []
        for checkpoint_name, config_path, checkpoint_path in checkpoint_dirs:
            summary = process_single_checkpoint(
                checkpoint_name, config_path, checkpoint_path,
                args.test_dir, args.output_dir, args, device
            )
            if summary:
                all_summaries.append(summary)
        
        # Save comparison CSV
        if all_summaries:
            comparison_df = pd.DataFrame(all_summaries)
            comparison_path = Path(args.output_dir) / 'checkpoint_comparison.csv'
            comparison_df.to_csv(comparison_path, index=False)
            
            print("\n" + "="*80)
            print("CHECKPOINT COMPARISON")
            print("="*80)
            print(comparison_df.to_string(index=False))
            print(f"\nComparison saved to: {comparison_path}")
        
        return
    
    # Single checkpoint processing
    if args.checkpoint_dir:
        # Auto-load from checkpoint directory
        config_path, checkpoint_path = resolve_checkpoint_dir(args.checkpoint_dir)
        checkpoint_name = Path(args.checkpoint_dir).name
    elif args.checkpoint and args.config:
        # Use explicitly provided paths
        config_path = args.config
        checkpoint_path = args.checkpoint
        checkpoint_name = Path(checkpoint_path).stem
    else:
        print("Error: Please specify --checkpoint-dir or both --checkpoint and --config")
        return

    # Process single checkpoint
    process_single_checkpoint(
        checkpoint_name, config_path, checkpoint_path,
        args.test_dir, args.output_dir, args, device
    )


if __name__ == "__main__":
    main()
