# train.py
# ─────────────────────────────────────────────────────────
# Loads data, trains the DS-CNN, and exports to TFLite INT8.
# Replace the data loading section with your actual dataset.
# ─────────────────────────────────────────────────────────

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from pathlib import Path

from config import (
    CLASSES, NUM_CLASSES, INPUT_SHAPE,
    MODEL_SAVE_PATH, TFLITE_PATH
)
from preprocess import extract_mfcc, load_file_as_window
from model import build_model, compile_model, export_tflite


# ── 1. Data Loading ───────────────────────────────────────
# Expected directory structure:
#   data/
#     noise/    *.wav
#     voice/    *.wav
#     alarm/    *.wav

def load_dataset(data_dir: str):
    X, y = [], []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = Path(data_dir) / class_name
        files = list(class_dir.glob("*.wav"))
        print(f"[load] {class_name}: {len(files)} files")
        
        for i, fname in enumerate(files):
            if i % 100 == 0:
                print(f"[load] {class_name}: {i}/{len(files)}")
            try:
                audio = load_file_as_window(fname)
                mfcc  = extract_mfcc(audio)
                X.append(mfcc)
                y.append(label_idx)
            except Exception as e:
                print(f"[skip] {fname}: {e}")
        print(f"[load] {class_name}: processed {len(files)} OK")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"[load] TOTAL: {len(X)} samples, shape {X.shape}")
    return X, y



# ── 2. Representative Dataset Generator (for INT8 export) ─
def make_rep_dataset_gen(X_train, n_samples=200):
    """
    Yields individual input tensors for TFLite calibration.
    Only needs ~100-200 samples to accurately estimate activation ranges.
    """
    indices = np.random.choice(len(X_train), size=min(n_samples, len(X_train)), replace=False)
    def gen():
        for i in indices:
            # TFLite expects a batch dimension: (1, 40, 61, 1)
            yield [X_train[i:i+1]]
    return gen


# ── 3. Main Training Script ───────────────────────────────
if __name__ == "__main__":
    print("[train] Loading dataset...")
    X, y = load_dataset("data/")
    print(f"[train] Loaded {len(X)} samples across {NUM_CLASSES} classes")

    # One-hot encode labels
    y_onehot = keras.utils.to_categorical(y, NUM_CLASSES)

    # Stratified 80/20 split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_onehot, test_size=0.2, stratify=y, random_state=42
    )

    # Build & compile
    model = build_model()
    model = compile_model(model)
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
    )

    # Final validation accuracy
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"[train] Val accuracy: {val_acc:.4f} | Val loss: {val_loss:.4f}")

    # Export
    rep_gen = make_rep_dataset_gen(X_train)
    export_tflite(model, rep_gen)
