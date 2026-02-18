import csv
from pathlib import Path

import soundfile as sf
import numpy as np

from config import SAMPLE_RATE, N_SAMPLES

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ── Utility: load, center-crop or pad to 1s @ 16k ─────────
def load_and_fix(path: Path) -> np.ndarray:
    """
    Load audio with soundfile + scipy (no librosa, no numba).
    Center-crop or pad to exactly 1s @ 16kHz.
    """
    
    # Load with soundfile
    audio, sr = sf.read(path, dtype="float32")
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # mono
    
    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        n_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, n_samples)
    
    # Center-crop or pad to exactly N_SAMPLES
    if len(audio) > N_SAMPLES:
        start = (len(audio) - N_SAMPLES) // 2
        audio = audio[start:start + N_SAMPLES]
    elif len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")
    
    return audio.astype(np.float32)


def save_clip(audio: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, SAMPLE_RATE, subtype="PCM_16")

# ── 1. ESC-50 → noise / alarm ────────────────────────────
def build_from_esc50():
    esc_root = BASE_DIR / "external" / "ESC-50"
    audio_dir = esc_root / "audio"
    meta_path = esc_root / "meta" / "esc50.csv"

    alarm_labels = {"siren", "clock_alarm", "church_bells"}
    noise_labels = {
        "dog", "rooster", "rain", "sea_waves", "crackling_fire",
        "crickets", "wind", "crow", "crackling_fire", "airplane",
        "chainsaw", "jackhammer", "keyboard_typing", "engine",
        "helicopter", "vacuum_cleaner", "crowd"
    }

    with open(meta_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["category"]
            fname = row["filename"]
            src = audio_dir / fname
            if not src.is_file():
                continue

            if label in alarm_labels:
                cls = "alarm"
            elif label in noise_labels:
                cls = "noise"
            else:
                continue

            audio = load_and_fix(src)
            out_name = f"esc50_{label}_{fname}"
            save_clip(audio, DATA_DIR / cls / out_name)

# ── 2. UrbanSound8K → noise / alarm ──────────────────────
def build_from_urbansound8k():
    root = BASE_DIR / "external" / "UrbanSound8K"
    meta_path = root / "metadata" / "UrbanSound8K.csv"

    alarm_labels = {"siren", "car_horn"}
    noise_labels = {
        "air_conditioner", "children_playing", "dog_bark", "drilling",
        "engine_idling", "gun_shot", "jackhammer", "street_music"
    }

    with open(meta_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["class"]
            fold = row["fold"]
            fname = row["slice_file_name"]

            src = root / "audio" / f"fold{fold}" / fname
            if not src.is_file():
                continue

            if label in alarm_labels:
                cls = "alarm"
            elif label in noise_labels:
                cls = "noise"
            else:
                continue

            audio = load_and_fix(src)
            out_name = f"urban_{label}_fold{fold}_{fname}"
            save_clip(audio, DATA_DIR / cls / out_name)

# ── 3. Speech Commands → voice (and optional noise) ──────
def build_from_speech_commands():
    root = BASE_DIR / "external" / "speech_commands"
    voice_words = ["yes", "no", "up", "down", "left", "right", "on", "off"]

    for word in voice_words:
        word_dir = root / word
        if not word_dir.is_dir():
            continue
        for wav in word_dir.glob("*.wav"):
            audio = load_and_fix(wav)
            out_name = f"sc_{word}_{wav.name}"
            save_clip(audio, DATA_DIR / "voice" / out_name)

    # Optional: background noise as noise
    noise_dir = root / "_background_noise_"
    if noise_dir.is_dir():
        for wav in noise_dir.glob("*.wav"):
            audio = load_and_fix(wav)
            out_name = f"sc_bg_{wav.name}"
            save_clip(audio, DATA_DIR / "noise" / out_name)

if __name__ == "__main__":
    build_from_esc50()
    # build_from_urbansound8k()
    build_from_speech_commands()
    print("Finished building dataset into data/{noise,voice,alarm}")
