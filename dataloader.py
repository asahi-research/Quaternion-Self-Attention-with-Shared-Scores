import torch
import torch.utils.data
import torchaudio
import numpy as np
import random
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
from natsort import natsorted
from torch.utils.data.distributed import DistributedSampler
from utils import power_compress, compute_stft_quaternion_features


PathLike = Union[str, os.PathLike[str]]
CutLength = Union[int, bool, None]
NormalizeMode = Union[bool, str]


class VoiceBankDemandDataset(torch.utils.data.Dataset):
    """Dataset for VoiceBank+DEMAND"""

    def __init__(
        self,
        data_dir: Optional[PathLike],
        clean_dir: Optional[PathLike] = None,
        noisy_dir: Optional[PathLike] = None,
        cut_len: CutLength = 16000 * 2,
        n_fft: int = 400,
        hop: int = 100,
        return_all: bool = True,
        normalize: NormalizeMode = True,
        window_type: str = 'hann'
    ) -> None:
        """
        Args:
            data_dir: Base directory (can contain clean/noisy subdirs or be None if clean_dir/noisy_dir provided)
            clean_dir: Explicit path to clean directory (optional)
            noisy_dir: Explicit path to noisy directory (optional)
            cut_len: Length of audio segment (default: 2 seconds at 16kHz). Set to False/None to use original length
            n_fft: FFT window size (default: 400)
            hop: Hop length (default: 100)
            return_all: If True, returns [features, noisy_audio, clean_spec, clean_audio]
            normalize: If True, apply peak normalization
        """
        self.cut_len = cut_len
        self.n_fft = n_fft
        self.hop = hop
        self.return_all = return_all
        self.normalize = normalize
        
        # Determine clean and noisy directories
        if clean_dir and noisy_dir:
            self.clean_dir = clean_dir
            self.noisy_dir = noisy_dir
        else:
            # Try standard directory structure first
            self.clean_dir = os.path.join(data_dir, "clean")
            self.noisy_dir = os.path.join(data_dir, "noisy")
        
        # Check if directories exist, otherwise raise error
        if not os.path.exists(self.clean_dir) or not os.path.exists(self.noisy_dir):
            raise ValueError(f"Clean or noisy directory not found. Clean: {self.clean_dir}, Noisy: {self.noisy_dir}")
        
        # Get sorted file list
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)
        
        # Pre-compute window based on window_type
        self.window_type = window_type
        if window_type == 'hann':
            self.window = torch.hann_window(self.n_fft)
        elif window_type == 'hamming':
            self.window = torch.hamming_window(self.n_fft)
        elif window_type == 'blackman':
            self.window = torch.blackman_window(self.n_fft)
        elif window_type == 'bartlett':
            self.window = torch.bartlett_window(self.n_fft)
        else:
            raise ValueError(f"Unsupported window type: {window_type}")

    def __len__(self) -> int:
        return len(self.clean_wav_name)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        
        length = len(clean_ds)
        assert length == len(noisy_ds)
        
        # Handle audio length
        if self.cut_len is False or self.cut_len is None:
            # Use original length (for test/inference)
            pass
        elif length < self.cut_len:
            # Padding for short audio
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len % length])
            noisy_ds_final.append(noisy_ds[: self.cut_len % length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut segment for training
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start : wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start : wav_start + self.cut_len]
        
        # Store original audio
        clean_audio = clean_ds.clone()
        noisy_audio = noisy_ds.clone()
        
        # Normalization
        scale_factor = torch.tensor(1.0)  # Default scale factor
        if self.normalize:
            # Peak normalization
            scale_factor = noisy_ds.abs().max() + 1e-8
            noisy_ds = noisy_ds / scale_factor
            clean_ds = clean_ds / scale_factor
        
        # Compute STFT and quaternion features for noisy audio
        features_4ch, noisy_spec_compressed = compute_stft_quaternion_features(
            noisy_ds, self.n_fft, self.hop, self.window
        )
        
        # Also compute noisy STFT for mask application
        noisy_stft = torch.stft(noisy_ds, self.n_fft, self.hop, window=self.window,
                                onesided=True, return_complex=True)  # [F, T]
        
        if self.return_all:
            # Return format: [4ch quaternion features, noisy_stft, noisy_audio, clean_audio, scale_factor]
            # Return normalized audio for consistent processing
            return features_4ch, noisy_stft, noisy_ds.unsqueeze(0), clean_ds.unsqueeze(0), scale_factor
        else:
            # Return features, stft, audio, and scale_factor for simpler usage
            return features_4ch, noisy_stft, clean_ds, scale_factor




class ValidationDataset(torch.utils.data.Dataset):
    """Dataset for validation with full-length audio (no segmentation)"""
    
    def __init__(
        self,
        clean_files: Sequence[PathLike],
        noisy_files: Sequence[PathLike],
        n_fft: int = 400,
        hop: int = 100,
        normalize: NormalizeMode = True,
        window_type: str = 'hann'
    ) -> None:
        """
        Args:
            clean_files: List of clean file paths
            noisy_files: List of noisy file paths
            n_fft: FFT window size
            hop: Hop length
            normalize: If True, apply peak normalization
        """
        self.clean_files = clean_files
        self.noisy_files = noisy_files
        self.n_fft = n_fft
        self.hop = hop
        self.normalize = normalize
        self.window_type = window_type
        if window_type == 'hann':
            self.window = torch.hann_window(n_fft)
        elif window_type == 'hamming':
            self.window = torch.hamming_window(n_fft)
        elif window_type == 'blackman':
            self.window = torch.blackman_window(n_fft)
        elif window_type == 'bartlett':
            self.window = torch.bartlett_window(n_fft)
        else:
            raise ValueError(f"Unsupported window type: {window_type}")
    
    def __len__(self) -> int:
        return len(self.clean_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor]:
        # Load full audio without segmentation
        clean_wav, _ = torchaudio.load(self.clean_files[idx])
        noisy_wav, _ = torchaudio.load(self.noisy_files[idx])
        clean_wav = clean_wav.squeeze()
        noisy_wav = noisy_wav.squeeze()
        
        # Store original for output
        clean_orig = clean_wav.clone()
        noisy_orig = noisy_wav.clone()
        
        # Normalization and store scale_factor
        scale_factor = torch.tensor(1.0)  # Default scale factor
        if self.normalize:
            # Peak normalization
            scale_factor = noisy_wav.abs().max() + 1e-8
            noisy_wav = noisy_wav / scale_factor
            clean_wav = clean_wav / scale_factor
        
        # Compute STFT features
        features_4ch, _ = compute_stft_quaternion_features(
            noisy_wav, self.n_fft, self.hop, self.window
        )
        
        # Also compute noisy STFT for mask application
        noisy_stft = torch.stft(noisy_wav, self.n_fft, self.hop, window=self.window,
                                onesided=True, return_complex=True)  # [F, T]
        
        # Get filename for logging
        filename = os.path.basename(self.clean_files[idx])
        
        # Return normalized audio for consistent processing
        return features_4ch, noisy_stft, clean_wav, noisy_wav, filename, scale_factor


def load_voicebank_dataloader(
    ds_dir: PathLike,
    batch_size: int,
    n_cpu: int,
    cut_len: CutLength = 16000 * 2,
    test_cut_len: CutLength = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load VoiceBank+DEMAND dataset

    Args:
        ds_dir: Dataset directory path
        batch_size: Batch size
        n_cpu: Number of CPU workers
        cut_len: Audio segment length for training (default: 2 seconds)
        test_cut_len: Audio segment length for test (default: False to use original length)

    Returns:
        train_dataset, test_dataset
    """
    torchaudio.set_audio_backend("sox_io")  # in linux
    
    train_dir = os.path.join(ds_dir, "train")
    test_dir = os.path.join(ds_dir, "test")

    train_ds = VoiceBankDemandDataset(train_dir, cut_len=cut_len)
    test_ds = VoiceBankDemandDataset(test_dir, cut_len=test_cut_len)

    train_dataset = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_ds),
        drop_last=True,
        num_workers=n_cpu,
    )
    
    test_dataset = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(test_ds),
        drop_last=False,
        num_workers=n_cpu,
    )

    return train_dataset, test_dataset


class TestDataset(torch.utils.data.Dataset):
    """Dataset for batch processing test files during inference.

    Loads paired clean/noisy WAVs from the given directories, normalizes the
    noisy waveform, and returns the precomputed quaternion features and
    complex STFT alongside the raw waveforms for metric computation.
    """

    def __init__(
        self,
        noisy_dir: PathLike,
        clean_dir: PathLike,
        n_fft: int = 400,
        hop: int = 100,
        window_type: str = 'hann',
    ) -> None:
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.n_fft = n_fft
        self.hop = hop
        self.window_type = window_type

        if window_type == 'hann':
            self.window = torch.hann_window(n_fft)
        elif window_type == 'hamming':
            self.window = torch.hamming_window(n_fft)
        elif window_type == 'blackman':
            self.window = torch.blackman_window(n_fft)
        elif window_type == 'bartlett':
            self.window = torch.bartlett_window(n_fft)
        else:
            raise ValueError(f"Unsupported window type: {window_type}")

        self.files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename = self.files[idx]
        noisy_path = os.path.join(self.noisy_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)

        noisy_wav, sr = torchaudio.load(noisy_path)
        clean_wav, _ = torchaudio.load(clean_path)
        noisy_wav = noisy_wav.squeeze()
        clean_wav = clean_wav.squeeze()

        noisy_original = noisy_wav.clone()
        original_length = len(noisy_wav)

        scale_factor = noisy_wav.abs().max() + 1e-8
        noisy_wav_norm = noisy_wav / scale_factor

        features_4ch, _ = compute_stft_quaternion_features(
            noisy_wav_norm, self.n_fft, self.hop, self.window
        )

        noisy_stft = torch.stft(
            noisy_wav_norm, self.n_fft, self.hop, window=self.window,
            onesided=True, return_complex=True,
        )

        return {
            'features': features_4ch,
            'noisy_stft': noisy_stft,
            'noisy_original': noisy_original,
            'noisy_wav': noisy_wav,
            'clean_wav': clean_wav,
            'scale_factor': scale_factor,
            'original_length': original_length,
            'filename': filename,
            'sr': sr,
        }


def test_collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate function for ``TestDataset`` that pads variable-length items
    to the batch maximum along time / waveform dimensions."""
    max_time = max(item['features'].shape[1] for item in batch)
    max_length = max(item['original_length'] for item in batch)

    padded_features = []
    for item in batch:
        features = item['features']  # [4, T, F]
        pad_amount = max_time - features.shape[1]
        if pad_amount > 0:
            features = torch.nn.functional.pad(features, (0, 0, 0, pad_amount), 'constant', 0)
        padded_features.append(features)
    features_batch = torch.stack(padded_features)

    padded_stft = []
    for item in batch:
        stft = item['noisy_stft']  # [F, T]
        pad_time = max_time - stft.shape[1]
        if pad_time > 0:
            stft = torch.nn.functional.pad(stft, (0, pad_time), 'constant', 0)
        padded_stft.append(stft)
    stft_batch = torch.stack(padded_stft)

    padded_noisy, padded_noisy_orig, padded_clean = [], [], []
    for item in batch:
        noisy = item['noisy_wav']
        noisy_orig = item['noisy_original']
        clean = item['clean_wav']
        pad_amount = max_length - len(noisy)
        if pad_amount > 0:
            noisy = torch.nn.functional.pad(noisy, (0, pad_amount), 'constant', 0)
            noisy_orig = torch.nn.functional.pad(noisy_orig, (0, pad_amount), 'constant', 0)
            clean = torch.nn.functional.pad(clean, (0, pad_amount), 'constant', 0)
        padded_noisy.append(noisy)
        padded_noisy_orig.append(noisy_orig)
        padded_clean.append(clean)

    scale_factors = torch.tensor([item['scale_factor'] for item in batch])
    original_lengths = [item['original_length'] for item in batch]
    filenames = [item['filename'] for item in batch]
    srs = [item['sr'] for item in batch]

    return {
        'features': features_batch,
        'noisy_stft': stft_batch,
        'noisy_wav': torch.stack(padded_noisy),
        'noisy_original': torch.stack(padded_noisy_orig),
        'clean_wav': torch.stack(padded_clean),
        'scale_factors': scale_factors,
        'original_lengths': original_lengths,
        'filenames': filenames,
        'srs': srs,
    }
