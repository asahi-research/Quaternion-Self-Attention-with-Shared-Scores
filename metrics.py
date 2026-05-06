"""
Speech Enhancement Performance Metrics

This module provides standard metrics for evaluating speech enhancement quality:
- PESQ (Perceptual Evaluation of Speech Quality)
- Composite measures (CSIG, CBAK, COVL)
- SNR-based metrics

Required packages:
    pip install Cython
    pip install pesq
"""

from scipy.signal import stft, resample
from scipy.linalg import toeplitz
from pesq import pesq as pesq_inner
from pesq import PesqError
from pystoi import stoi as stoi_inner
import numpy as np
from typing import Any, Dict, Optional, Tuple


def extractOverlappedWindows(
    x: np.ndarray,
    nperseg: int,
    noverlap: int,
    window: Optional[np.ndarray] = None
) -> np.ndarray:
    """Extract overlapped windows from signal for frame-based processing"""
    step = nperseg - noverlap
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    if window is not None:
        result = window * result
    return result


def SNRseg(
    clean_speech: np.ndarray,
    processed_speech: np.ndarray,
    fs: int,
    frameLen: float = 0.03,
    overlap: float = 0.75
) -> float:
    """Segmental SNR calculation"""
    eps = np.finfo(np.float64).eps
    winlength = round(frameLen * fs)
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))
    MIN_SNR = -10
    MAX_SNR = 35

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    clean_speech_framed = extractOverlappedWindows(clean_speech, winlength, winlength - skiprate, hannWin)
    processed_speech_framed = extractOverlappedWindows(processed_speech, winlength, winlength - skiprate, hannWin)

    signal_energy = np.power(clean_speech_framed, 2).sum(-1)
    noise_energy = np.power(clean_speech_framed - processed_speech_framed, 2).sum(-1)

    segmental_snr = 10 * np.log10(signal_energy / (noise_energy + eps) + eps)
    segmental_snr[segmental_snr < MIN_SNR] = MIN_SNR
    segmental_snr[segmental_snr > MAX_SNR] = MAX_SNR
    segmental_snr = segmental_snr[:-1]
    return np.mean(segmental_snr)


def lpcoeff(speech_frame: np.ndarray, model_order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Linear prediction coefficients using Levinson-Durbin algorithm"""
    eps = np.finfo(np.float64).eps
    winlength = speech_frame.shape[0]
    R = []
    for k in range(model_order + 1):
        first = speech_frame[:(winlength - k)]
        second = speech_frame[k:winlength]
        R.append(np.sum(first * second))

    a = np.ones((model_order,))
    E = np.zeros((model_order + 1,))
    rcoeff = np.zeros((model_order,))
    E[0] = R[0]
    for i in range(model_order):
        if i == 0:
            sum_term = 0
        else:
            a_past = a[:i]
            sum_term = np.sum(a_past * np.array(R[i:0:-1]))
        rcoeff[i] = (R[i + 1] - sum_term) / max(E[i], eps)
        a[i] = rcoeff[i]
        if i > 0:
            a[:i] = a_past[:i] - rcoeff[i] * a_past[::-1]
        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]

    a = a * -1
    lpparams = np.array([1] + list(a), dtype=np.float32)
    acorr = np.array(R, dtype=np.float32)
    return lpparams, acorr


def llr(
    clean_speech: np.ndarray,
    processed_speech: np.ndarray,
    fs: int,
    frameLen: float = 0.03,
    overlap: float = 0.75
) -> float:
    """Log-likelihood ratio measure"""
    eps = np.finfo(np.float64).eps
    alpha = 0.95
    winlength = round(frameLen * fs)
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))
    P = 10 if fs < 10000 else 16

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    clean_speech_framed = extractOverlappedWindows(clean_speech, winlength, winlength - skiprate, hannWin)
    processed_speech_framed = extractOverlappedWindows(processed_speech, winlength, winlength - skiprate, hannWin)
    numFrames = clean_speech_framed.shape[0]
    numerators = np.zeros((numFrames - 1,))
    denominators = np.zeros((numFrames - 1,))

    for ii in range(numFrames - 1):
        A_clean, R_clean = lpcoeff(clean_speech_framed[ii, :], P)
        A_proc, R_proc = lpcoeff(processed_speech_framed[ii, :], P)
        numerators[ii] = A_proc.dot(toeplitz(R_clean).dot(A_proc.T))
        denominators[ii] = A_clean.dot(toeplitz(R_clean).dot(A_clean.T))

    frac = numerators / denominators
    frac[frac <= 0] = 1000
    distortion = np.log(frac)
    distortion = np.sort(distortion)
    distortion = distortion[:int(round(len(distortion) * alpha))]
    return np.mean(distortion)


def findLocPeaks(slope: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Find local peaks in energy contour"""
    num_crit = len(energy)
    loc_peaks = np.zeros_like(slope)

    for ii in range(len(slope)):
        n = ii
        if slope[ii] > 0:
            while (n < num_crit - 1) and (slope[n] > 0):
                n = n + 1
            loc_peaks[ii] = energy[n - 1]
        else:
            while (n >= 0) and (slope[n] <= 0):
                n = n - 1
            loc_peaks[ii] = energy[n + 1]

    return loc_peaks


def wss(
    clean_speech: np.ndarray,
    processed_speech: np.ndarray,
    fs: int,
    frameLen: float = 0.03,
    overlap: float = 0.75
) -> float:
    """Weighted spectral slope measure"""
    Kmax = 20
    Klocmax = 1
    alpha = 0.95
    if clean_speech.shape != processed_speech.shape:
        raise ValueError('The two signals do not match!')
    eps = np.finfo(np.float64).eps
    clean_speech = clean_speech.astype(np.float64) + eps
    processed_speech = processed_speech.astype(np.float64) + eps
    winlength = round(frameLen * fs)
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))
    max_freq = fs / 2
    num_crit = 25
    n_fft = 2 ** np.ceil(np.log2(2 * winlength))
    n_fftby2 = int(n_fft / 2)

    cent_freq = np.array([50.0, 120.0, 190.0, 260.0, 330.0, 400.0, 470.0, 540.0, 617.372,
                          703.378, 798.717, 904.128, 1020.38, 1148.30, 1288.72, 1442.54,
                          1610.70, 1794.16, 1993.93, 2211.08, 2446.71, 2701.97, 2978.04,
                          3276.17, 3597.63])
    bandwidth = np.array([70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 77.3724, 86.0056,
                          95.3398, 105.411, 116.256, 127.914, 140.423, 153.823, 168.154,
                          183.457, 199.776, 217.153, 235.631, 255.255, 276.072, 298.126,
                          321.465, 346.136])

    bw_min = bandwidth[0]
    min_factor = np.exp(-30.0 / (2.0 * 2.303))

    crit_filter = np.zeros((num_crit, int(n_fftby2)))
    j = np.arange(0, n_fftby2)

    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * n_fftby2
        bw = (bandwidth[i] / max_freq) * n_fftby2
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor)

    num_frames = len(clean_speech) / skiprate - (winlength / skiprate)

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    scale = np.sqrt(1.0 / hannWin.sum() ** 2)

    f, t, Zxx = stft(clean_speech[0:int(num_frames) * skiprate + int(winlength - skiprate)],
                     fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength - skiprate,
                     nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    clean_spec = np.power(np.abs(Zxx) / scale, 2)
    clean_spec = clean_spec[:-1, :]

    f, t, Zxx = stft(processed_speech[0:int(num_frames) * skiprate + int(winlength - skiprate)],
                     fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength - skiprate,
                     nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    proc_spec = np.power(np.abs(Zxx) / scale, 2)
    proc_spec = proc_spec[:-1, :]

    clean_energy = crit_filter.dot(clean_spec)
    log_clean_energy = 10 * np.log10(clean_energy)
    log_clean_energy[log_clean_energy < -100] = -100
    proc_energy = crit_filter.dot(proc_spec)
    log_proc_energy = 10 * np.log10(proc_energy)
    log_proc_energy[log_proc_energy < -100] = -100

    log_clean_energy_slope = np.diff(log_clean_energy, axis=0)
    log_proc_energy_slope = np.diff(log_proc_energy, axis=0)

    dBMax_clean = np.max(log_clean_energy, axis=0)
    dBMax_processed = np.max(log_proc_energy, axis=0)

    numFrames = log_clean_energy_slope.shape[-1]

    clean_loc_peaks = np.zeros_like(log_clean_energy_slope)
    proc_loc_peaks = np.zeros_like(log_proc_energy_slope)
    for ii in range(numFrames):
        clean_loc_peaks[:, ii] = findLocPeaks(log_clean_energy_slope[:, ii], log_clean_energy[:, ii])
        proc_loc_peaks[:, ii] = findLocPeaks(log_proc_energy_slope[:, ii], log_proc_energy[:, ii])

    Wmax_clean = Kmax / (Kmax + dBMax_clean - log_clean_energy[:-1, :])
    Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peaks - log_clean_energy[:-1, :])
    W_clean = Wmax_clean * Wlocmax_clean

    Wmax_proc = Kmax / (Kmax + dBMax_processed - log_proc_energy[:-1])
    Wlocmax_proc = Klocmax / (Klocmax + proc_loc_peaks - log_proc_energy[:-1, :])
    W_proc = Wmax_proc * Wlocmax_proc

    W = (W_clean + W_proc) / 2.0

    distortion = np.sum(W * (log_clean_energy_slope - log_proc_energy_slope) ** 2, axis=0)
    distortion = distortion / np.sum(W, axis=0)
    distortion = np.sort(distortion)
    distortion = distortion[:int(round(len(distortion) * alpha))]
    return np.mean(distortion)


def pesq(clean_speech: np.ndarray, processed_speech: np.ndarray, fs: int) -> float:
    """PESQ (Perceptual Evaluation of Speech Quality) score"""
    try:
        if fs == 8000:
            pesq_mos = pesq_inner(fs, clean_speech, processed_speech, 'nb')
            pesq_mos = 46607 / 14945 - (2000 * np.log(1 / (pesq_mos / 4 - 999 / 4000) - 1)) / 2989
        elif fs == 16000:
            pesq_mos = pesq_inner(fs, clean_speech, processed_speech, 'wb')
        elif fs >= 16000:
            numSamples = round(len(clean_speech) / fs * 16000)
            pesq_mos = pesq_inner(16000, resample(clean_speech, numSamples),
                                  resample(processed_speech, numSamples), 'wb')
        else:
            numSamples = round(len(clean_speech) / fs * 8000)
            pesq_mos = pesq_inner(8000, resample(clean_speech, numSamples),
                                  resample(processed_speech, numSamples), 'nb')
            pesq_mos = 46607 / 14945 - (2000 * np.log(1 / (pesq_mos / 4 - 999 / 4000) - 1)) / 2989
    except PesqError:
        return 0.0

    return pesq_mos


def composite(clean_speech: np.ndarray, processed_speech: np.ndarray, fs: int) -> Tuple[float, float, float, float, float]:
    """
    Compute composite speech quality measures

    Returns:
        segSNR: Segmental SNR
        pesq_mos: PESQ score
        Csig: Signal distortion (1-5)
        Cbak: Background intrusiveness (1-5)
        Covl: Overall quality (1-5)
    """
    wss_dist = wss(clean_speech, processed_speech, fs)
    llr_mean = llr(clean_speech, processed_speech, fs)
    segSNR = SNRseg(clean_speech, processed_speech, fs)
    pesq_mos = pesq(clean_speech, processed_speech, fs)

    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_mos - 0.009 * wss_dist
    Csig = np.clip(Csig, 1, 5)
    Cbak = 1.634 + 0.478 * pesq_mos - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = np.clip(Cbak, 1, 5)
    Covl = 1.594 + 0.805 * pesq_mos - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = np.clip(Covl, 1, 5)

    return segSNR, pesq_mos, Csig, Cbak, Covl


def calculate_sisdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Calculate scale-invariant SDR."""
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-10)
    scaled_reference = alpha * reference

    sisdr_num = np.sum(scaled_reference ** 2)
    sisdr_den = np.sum((scaled_reference - estimate) ** 2)
    return 10 * np.log10(sisdr_num / (sisdr_den + 1e-10))


def evaluate_metrics(
    clean_wav: np.ndarray,
    enhanced_wav: np.ndarray,
    noisy_wav: np.ndarray,
    sr: int = 16000,
    use_rms_match: bool = True,
) -> Dict[str, Any]:
    """Compute PESQ / STOI / ESTOI / SI-SDR / CSIG / CBAK / COVL.

    When ``use_rms_match`` is True, the enhanced signal is rescaled to match
    the clean RMS before PESQ and the composite measures (CSIG/CBAK/COVL),
    matching the convention used in prior speech-enhancement benchmarks.
    """
    metrics: Dict[str, Any] = {}

    min_len = min(len(clean_wav), len(enhanced_wav), len(noisy_wav))
    clean_wav = clean_wav[:min_len]
    enhanced_wav = enhanced_wav[:min_len]
    noisy_wav = noisy_wav[:min_len]

    if use_rms_match:
        rms_clean = np.sqrt(np.mean(clean_wav ** 2) + 1e-10)
        rms_enhanced = np.sqrt(np.mean(enhanced_wav ** 2) + 1e-10)
        enhanced_matched = enhanced_wav * (rms_clean / rms_enhanced)
        enhanced_matched = np.clip(enhanced_matched, -1.0, 1.0)
    else:
        enhanced_matched = enhanced_wav

    try:
        metrics['pesq'] = pesq_inner(sr, clean_wav, enhanced_matched, 'wb')
    except Exception as e:
        print(f"PESQ calculation failed: {e}")
        metrics['pesq'] = None

    try:
        metrics['stoi'] = stoi_inner(clean_wav, enhanced_wav, sr, extended=False)
        metrics['estoi'] = stoi_inner(clean_wav, enhanced_wav, sr, extended=True)
    except Exception as e:
        print(f"STOI calculation failed: {e}")
        metrics['stoi'] = None
        metrics['estoi'] = None

    metrics['sisdr'] = calculate_sisdr(clean_wav, enhanced_wav)

    try:
        _, _, csig, cbak, covl = composite(clean_wav, enhanced_matched, sr)
        metrics['csig'] = csig
        metrics['cbak'] = cbak
        metrics['covl'] = covl
    except Exception as e:
        print(f"Composite metrics calculation failed: {e}")
        metrics['csig'] = None
        metrics['cbak'] = None
        metrics['covl'] = None

    return metrics
