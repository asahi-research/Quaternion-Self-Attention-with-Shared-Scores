import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Sequence
from torch_pesq import PesqLoss


def si_sdr_loss(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss

    Args:
        estimate, target: [B, 1, T] or [B, T]
    Returns:
        Negative SI-SDR (minimizing this maximizes SI-SDR)
    """
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
        target   = target.squeeze(1)
    # zero-mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target   = target   - target.mean(dim=-1, keepdim=True)
    # projection
    s = (estimate * target).sum(dim=-1, keepdim=True) / (target.pow(2).sum(dim=-1, keepdim=True) + eps) * target
    e = estimate - s
    ratio = (s.pow(2).sum(dim=-1) + eps) / (e.pow(2).sum(dim=-1) + eps)
    si_sdr = 10 * torch.log10(ratio + eps)
    return -si_sdr.mean()


def rms_loss(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Root Mean Square (RMS) loss for preserving energy levels
    Computes log-domain RMS difference for better numerical stability
    estimate, target: [B, 1, T] or [B, T]
    """
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    # Compute RMS for each signal
    r1 = (estimate.pow(2).mean(dim=-1) + eps).sqrt()
    r2 = (target.pow(2).mean(dim=-1) + eps).sqrt()
    # Log-domain difference for better stability
    return (torch.log(r1 + eps) - torch.log(r2 + eps)).abs().mean()


def complex_l1_loss(
    estimate_real: torch.Tensor,
    estimate_imag: torch.Tensor,
    target_real: torch.Tensor,
    target_imag: torch.Tensor
) -> torch.Tensor:
    """
    L1 loss for real and imaginary parts separately

    Args:
        estimate_real, estimate_imag: [B, T, F] or [B, F, T]
        target_real, target_imag: [B, T, F] or [B, F, T]

    Returns:
        Combined L1 loss for real and imaginary parts
    """
    real_loss = F.l1_loss(estimate_real, target_real)
    imag_loss = F.l1_loss(estimate_imag, target_imag)
    return real_loss + imag_loss


def complex_spectrum_loss(
    S_ref: torch.Tensor,
    S_est: torch.Tensor,
    alpha: float = 1.0,   # Magnitude term weight
    beta:  float = 0.2,   # Phase term weight
    use_log_mag: bool = False,
    gamma: float = 0.3,   # Power compression (effective when use_log_mag=False)
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Complex spectrum loss combining magnitude and phase losses

    Args:
        S_ref, S_est: [..., F, T] complex tensors (output of torch.stft(return_complex=True))
    """
    # --- magnitude ---
    mag_r = S_ref.abs().clamp_min(eps)
    mag_e = S_est.abs().clamp_min(eps)

    if use_log_mag:
        # L1 loss on log magnitude
        mag_loss = (mag_r.log() - mag_e.log()).abs().mean()
    else:
        # L1 loss on power-compressed magnitude (gamma~0.3 recommended)
        mag_loss = (mag_r.pow(gamma) - mag_e.pow(gamma)).abs().mean()

    # --- phase: magnitude-weighted cosine distance 1 - cos(Δθ) ---
    # cos(Δθ) = Re(S_ref * conj(S_est)) / (|S_ref||S_est|)
    cos_phase = ((S_ref.real * S_est.real + S_ref.imag * S_est.imag)
                 / (mag_r * mag_e).clamp_min(eps)).clamp(-1 + 1e-7, 1 - 1e-7)
    weight = (mag_r * mag_e).sqrt()  # Reduce contribution from low-energy regions
    phase_loss = ((1.0 - cos_phase) * weight).sum(dim=(-2, -1)) \
                 / weight.sum(dim=(-2, -1)).clamp_min(eps)
    phase_loss = phase_loss.mean()

    return alpha * mag_loss + beta * phase_loss


def spectral_convergence(mag_r: torch.Tensor, mag_e: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Spectral convergence: |||S|-|Ŝ|||_F / |||S|||_F averaged over batch"""
    num = torch.linalg.norm(mag_r - mag_e, ord='fro', dim=(-2, -1))
    den = torch.linalg.norm(mag_r,           ord='fro', dim=(-2, -1)).clamp_min(eps)
    return (num / den).mean()


def multi_resolution_stft_loss(
    real_audio: torch.Tensor,
    gen_audio:  torch.Tensor,
    fft_sizes:   Sequence[int] = (512, 1024, 2048),
    hop_sizes:   Sequence[int] = (128, 256, 512),
    win_lengths: Sequence[int] = (512, 1024, 2048),
    w_magphase: float = 1.0,
    w_sc:       float = 0.5,
    alpha:      float = 1.0,
    beta:       float = 0.2,
    use_log_mag: bool = False,
    gamma: float = 0.3,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Multi-resolution STFT loss for audio quality

    Args:
        real_audio, gen_audio: [B, T] or [T]
    Returns:
        Total loss combining magnitude-phase loss and spectral convergence
    """
    if real_audio.dim() == 1:
        real_audio = real_audio[None, :]
        gen_audio  = gen_audio[None, :]

    device = real_audio.device
    loss_magphase = 0.0
    loss_sc = 0.0

    for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
        win_t = torch.hann_window(win, device=device)

        S_ref = torch.stft(real_audio, n_fft=n_fft, hop_length=hop, win_length=win,
                           window=win_t, center=True, normalized=False,
                           return_complex=True)
        S_est = torch.stft(gen_audio,  n_fft=n_fft, hop_length=hop, win_length=win,
                           window=win_t, center=True, normalized=False,
                           return_complex=True)

        # Complex spectrum loss (magnitude + phase)
        lp = complex_spectrum_loss(
            S_ref, S_est, alpha=alpha, beta=beta,
            use_log_mag=use_log_mag, gamma=gamma, eps=eps
        )
        loss_magphase = loss_magphase + lp

        # Spectral convergence
        mag_r = S_ref.abs().clamp_min(eps)
        mag_e = S_est.abs().clamp_min(eps)
        sc = spectral_convergence(mag_r, mag_e, eps=eps)
        loss_sc = loss_sc + sc

    n = len(fft_sizes)
    loss_magphase = loss_magphase / n
    loss_sc = loss_sc / n

    total = w_magphase * loss_magphase + w_sc * loss_sc
    return total


class MRSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss wrapper for train.py"""

    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Sequence[int] = (128, 256, 512),
        win_lengths: Sequence[int] = (512, 1024, 2048),
        sc_weight: float = 0.3,
        mag_weight: float = 0.7
    ) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """estimate, target: [B, 1, T] or [B, T]"""
        if estimate.dim() == 3:
            estimate = estimate.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        total = multi_resolution_stft_loss(
            target, estimate,
            fft_sizes=self.fft_sizes,
            hop_sizes=self.hop_sizes,
            win_lengths=self.win_lengths,
            w_magphase=self.mag_weight,
            w_sc=self.sc_weight
        )
        return total


def pesq_loss(
    real_audio: torch.Tensor,
    gen_audio: torch.Tensor,
    sample_rate: int = 16000,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    PESQ (Perceptual Evaluation of Speech Quality) loss

    Args:
        real_audio: [B, samples] - target/reference audio waveform
        gen_audio: [B, samples] - generated/estimated audio waveform
        sample_rate: Sample rate of the audio (16000 or 8000 for PESQ)
        alpha: Scaling factor for loss (default 0.5)

    Returns:
        PESQ loss value (higher is worse for training)
    """
    pesq_fn = PesqLoss(
        factor=alpha,
        sample_rate=sample_rate
    ).to(real_audio.device)

    # Ensure waveforms are in the correct range [-1, 1]
    gen_audio = torch.clamp(gen_audio, -1, 1)
    real_audio = torch.clamp(real_audio, -1, 1)

    # Compute PESQ loss (reference, degraded)
    loss = pesq_fn(real_audio, gen_audio)

    if loss.dim() > 0:
        loss = loss.mean()

    return loss
