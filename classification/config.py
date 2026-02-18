"""
Central config: all values must match ESP-DSP parameters
on the S3 exactly to avoid train/deploy mismatch.
"""

SAMPLE_RATE     = 16_000   # Hz — must match I2S config on ESP32
DURATION        = 1.0      # seconds per inference window
N_SAMPLES       = int(SAMPLE_RATE * DURATION)  # 16 000

# ── STFT / MFCC ──────────────────────────────────────────
N_FFT           = 512      # ESP-DSP FFT size limit
HOP_LENGTH      = 256      # 50% overlap → ~61 frames/sec
N_MELS          = 40       # Mel filterbank resolution
N_MFCC          = 40       # Use all mel bands as MFCC input
                            # (skip DCT reduction for richer features)

# Derived input shape: (n_mels, time_frames, 1)
# time_frames = 1 + floor((N_SAMPLES - N_FFT) / HOP_LENGTH) = 61
TIME_FRAMES     = 1 + (N_SAMPLES - N_FFT) // HOP_LENGTH   # 61
INPUT_SHAPE     = (N_MELS, TIME_FRAMES, 1)                 # (40, 61, 1)

# ── Classes ───────────────────────────────────────────────
# Index maps directly to Dense output neuron
CLASSES         = ["noise", "voice", "alarm"]   # 3-class setup
NUM_CLASSES     = len(CLASSES)

# ── Gating threshold ──────────────────────────────────────
HAPTIC_THRESHOLD = 0.85    # Min confidence to trigger haptic
HAPTIC_CLASSES   = {"voice", "alarm"}   # Classes that fire the motor

# ── Paths ─────────────────────────────────────────────────
MODEL_SAVE_PATH  = "sound_gate_model.keras"
TFLITE_PATH      = "model.tflite"
