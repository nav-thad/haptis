"""
Produces (40, 61, 1) float32 MFCC tensors from raw audio.
Parameters MUST match ESP-DSP library config on the S3.
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
import sounddevice as sd
from config import (
    SAMPLE_RATE, N_SAMPLES, N_FFT, HOP_LENGTH,
    N_MELS, N_MFCC, INPUT_SHAPE
)


def hz_to_mel(hz: float, fmin: float = 80.0, fmax: float = 7600.0) -> float:
    """Convert Hz to Mel scale."""
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel: float) -> float:
    """Convert Mel to Hz."""
    return 700 * (10**(mel / 2595) - 1)


def mel_filters(n_fft: int, sr: int, n_mels: int, fmin: float, fmax: float):
    """
    Generate Mel filterbank matrix.
    Shape: (n_mels, n_fft//2 + 1)
    """
    # Mel scale points
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Bin frequencies
    bins = np.floor((n_fft + 1) * hz_points / sr)
    
    # Build filters
    filters = np.zeros((n_mels, int(n_fft // 2 + 1)))
    for m in range(n_mels):
        f_left = bins[m]
        f_center = bins[m + 1]
        f_right = bins[m + 2]
        
        for i in range(int(f_left), int(f_right + 1)):
            if i <= f_center:
                filters[m, i] = (i - f_left) / (f_center - f_left)
            else:
                filters[m, i] = (f_right - i) / (f_right - f_center)
    
    return filters


def power_spectrum(audio: np.ndarray, n_fft: int) -> np.ndarray:
    """Compute power spectrum using real FFT."""
    # Hann window
    window = signal.windows.hann(n_fft)
    padded = np.pad(audio, (0, n_fft - len(audio)), mode="constant")
    windowed = padded[:n_fft] * window
    
    # Real FFT
    fft = rfft(windowed)
    return np.abs(fft)**2


def spectrogram(audio: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """STFT power spectrogram."""
    frames = []
    for i in range(0, len(audio) - n_fft + 1, hop_length):
        frame = audio[i:i + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        frames.append(power_spectrum(frame, n_fft))
    
    return np.array(frames).T  # (freq, time)


def power_to_db(S: np.ndarray, ref: float = None) -> np.ndarray:
    """Convert power spectrogram to dB."""
    if ref is None:
        ref = np.max(S)
    if ref == 0:
        ref = 1e-10  # Avoid divide by zero
    log_spec = 10.0 * np.log10(np.maximum(S, 1e-12) / ref)
    return np.clip(log_spec, -80.0, 0.0)  # floor/ceiling



def dct(x: np.ndarray, norm: str = "ortho") -> np.ndarray:
    """Type-II DCT."""
    N = len(x)
    result = np.zeros(N)
    for k in range(N):
        result[k] = np.sum(x * np.cos(np.pi * k * (2 * np.arange(N) + 1) / (2 * N)))
    
    if norm == "ortho":
        result[0] *= np.sqrt(1.0 / N)
        result[1:] *= np.sqrt(2.0 / N)
    
    return result


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """
    Pure numpy/scipy MFCC pipeline → (40, 61, 1).
    
    Steps exactly matching librosa:
    1. STFT → power spectrogram
    2. Mel filterbank → mel spectrogram
    3. Power → log dB
    4. DCT → MFCCs
    5. Per-feature z-score norm
    """
    # 1. Length normalisation
    if len(audio) > N_SAMPLES:
        audio = audio[:N_SAMPLES]
    elif len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))

    # 2. STFT power spectrogram (freq, time)
    stft = spectrogram(audio, N_FFT, HOP_LENGTH)  # (257, ~61)

    # 3. Mel spectrogram
    mel_fb = mel_filters(N_FFT, SAMPLE_RATE, N_MELS, fmin=80.0, fmax=7600.0)
    mel_spec = mel_fb @ stft  # (40, ~61)

    # 4. Power → dB
    mel_db = power_to_db(mel_spec)

    # 5. DCT to MFCCs
    mfcc = np.array([dct(mel_db[m])[:N_MFCC] for m in range(N_MELS)])  # (40, 40)

    # 6. Per-feature z-score normalisation
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True) + 1e-6
    mfcc_norm = (mfcc - mean) / std

    # 7. Pad/truncate time axis to exactly 61
    if mfcc_norm.shape[1] > 61:
        mfcc_norm = mfcc_norm[:, :61]
    elif mfcc_norm.shape[1] < 61:
        mfcc_norm = np.pad(mfcc_norm, ((0, 0), (0, 61 - mfcc_norm.shape[1])))

    return mfcc_norm[..., np.newaxis].astype(np.float32)  # (40, 61, 1)


def record_window(duration: float = 1.0) -> np.ndarray:
    """Live mic capture (unchanged)."""
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def load_file_as_window(path: str) -> np.ndarray:
    """File loading (soundfile + scipy, no librosa)."""
    import soundfile as sf
    from scipy.signal import resample
    
    try:
        audio, sr = sf.read(path, dtype="float32")
    except:
        raise ValueError(f"Could not load {path} (only WAV supported)")
    
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    if sr != SAMPLE_RATE:
        n_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, n_samples)
    
    if len(audio) > N_SAMPLES:
        audio = audio[:N_SAMPLES]
    elif len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    
    return audio
