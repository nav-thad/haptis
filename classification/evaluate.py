"""
Laptop-side evaluation of the sound gate:
  - Run inference on audio files or live microphone
  - Apply gating logic to decide haptic trigger
  - Benchmark against hard noise samples
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from config import (
    CLASSES, HAPTIC_THRESHOLD, HAPTIC_CLASSES,
    MODEL_SAVE_PATH, TFLITE_PATH, INPUT_SHAPE
)
from preprocess import extract_mfcc, load_file_as_window, record_window


def apply_gate(probabilities: np.ndarray) -> dict:
    """Core gating logic (unchanged)."""
    pred_idx   = int(np.argmax(probabilities))
    pred_class = CLASSES[pred_idx]
    confidence = float(probabilities[pred_idx])

    trigger = (pred_class in HAPTIC_CLASSES) and (confidence >= HAPTIC_THRESHOLD)

    suppress_reason = None
    if not trigger:
        if pred_class not in HAPTIC_CLASSES:
            suppress_reason = f"class='{pred_class}'"
        else:
            suppress_reason = f"conf {confidence:.2f}<{HAPTIC_THRESHOLD}"

    return {
        "predicted_class": pred_class,
        "confidence":       confidence,
        "trigger_haptic":   trigger,
        "suppress_reason":  suppress_reason,
        "probabilities":    probabilities,
    }


class SoundGateInferencer:
    """TFLite INT8 interpreter (unchanged)."""
    def __init__(self, model_path: str = TFLITE_PATH):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # INT8 dequant params
        out = self.output_details[0]
        self.out_scale = out["quantization_parameters"]["scales"][0]
        self.out_zero_pt = out["quantization_parameters"]["zero_points"][0]

    def preprocess_for_int8(self, mfcc_float32: np.ndarray) -> np.ndarray:
        inp = self.input_details[0]
        scale = inp["quantization_parameters"]["scales"][0]
        zero_pt = inp["quantization_parameters"]["zero_points"][0]
        return (mfcc_float32 / scale + zero_pt).astype(np.int8)

    def infer(self, audio: np.ndarray) -> dict:
        mfcc = extract_mfcc(audio)
        inp = self.preprocess_for_int8(mfcc)[np.newaxis]
        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        self.interpreter.invoke()
        raw_out = self.interpreter.get_tensor(self.output_details[0]["index"])
        probs = self.out_scale * (raw_out.astype(np.float32) - self.out_zero_pt)
        return apply_gate(probs[0])


def run_noise_benchmark(inferencer, noise_dir: str = "data_test/noise",
                        voice_dir: str = "data_test/voice"):
    """Benchmark with progress bars."""
    noise_files = list(Path(noise_dir).glob("*.wav"))
    voice_files = list(Path(voice_dir).glob("*.wav"))
    
    print(f"\n── Gating Benchmark ({len(noise_files)} noise, {len(voice_files)} voice) ──")
    
    results = {"noise_suppressed": 0, "noise_total": 0,
               "voice_triggered": 0,  "voice_total": 0,
               "fails": []}

    # Noise suppression test
    print("Noise suppression:")
    for fpath in tqdm(noise_files, desc="noise"):
        audio = load_file_as_window(str(fpath))
        result = inferencer.infer(audio)
        results["noise_total"] += 1
        if not result["trigger_haptic"]:
            results["noise_suppressed"] += 1
        else:
            results["fails"].append(f"NOISE FALSE POS: {fpath.name} → {result['predicted_class']}@{result['confidence']:.2f}")

    # Voice trigger test
    print("Voice trigger rate:")
    for fpath in tqdm(voice_files, desc="voice"):
        audio = load_file_as_window(str(fpath))
        result = inferencer.infer(audio)
        results["voice_total"] += 1
        if result["trigger_haptic"]:
            results["voice_triggered"] += 1
        else:
            results["fails"].append(f"VOICE FALSE NEG: {fpath.name} → {result['suppress_reason']}")

    # Summary table
    # Debug counters first
    print(f"\nDEBUG COUNTERS:")
    print(f"  noise_total:     {results['noise_total']}")
    print(f"  noise_suppressed:{results['noise_suppressed']}")
    print(f"  voice_total:     {results['voice_total']}")
    print(f"  voice_triggered: {results['voice_triggered']}")
    
    # Safe metrics
    if results["noise_total"] > 0:
        nr = results["noise_suppressed"] / results["noise_total"]
        print(f"Noise suppression: {nr:.1%} ({results['noise_suppressed']}/{results['noise_total']})")
    else:
        print("Noise suppression: N/A (0 noise files)")
    
    if results["voice_total"] > 0:
        tr = results["voice_triggered"] / results["voice_total"]
        print(f"Voice trigger:     {tr:.1%} ({results['voice_triggered']}/{results['voice_total']})")
    else:
        print("Voice trigger:     N/A (0 voice files)")
    
    if results["fails"]:
        print(f"\nFirst 5 failures:")
        for fail in results["fails"][:5]:
            print(f"  {fail}")
    
    return results


def live_demo(inferencer, n_windows: int = 10):
    """Live mic with countdown."""
    print(f"\n[Live Demo: {n_windows} × 1s windows]")
    print("Speak near mic for '🔔 HAPTIC ON', silence for '🔇 suppressed'")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for i in range(n_windows):
            print(f"[{i+1:2d}/{n_windows}] Listening... ", end="", flush=True)
            audio = record_window(1.0)
            result = inferencer.infer(audio)
            
            if result["trigger_haptic"]:
                print(f"🔔 HAPTIC ON  | {result['predicted_class']} ({result['confidence']:.2f})")
            else:
                reason = result.get('suppress_reason', 'low conf')
                print(f"🔇 suppressed | {reason}")
    except KeyboardInterrupt:
        print("\n[Stopped]")


if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("Install tqdm: pip install tqdm")
        exit(1)
    
    engine = SoundGateInferencer(TFLITE_PATH)
    results = run_noise_benchmark(engine)
    
    print("\nLive mic test? (y/n): ", end="")
    if input().lower() == 'y':
        live_demo(engine, n_windows=15)
