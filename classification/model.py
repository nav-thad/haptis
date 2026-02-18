"""
Lightweight DS-CNN for binary sound gating.
Architecture targets < 50 KB weights for ESP32-S3 SRAM.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import INPUT_SHAPE, NUM_CLASSES, CLASSES, MODEL_SAVE_PATH, TFLITE_PATH


# ── Helper: Depthwise Separable Block ────────────────────
def ds_block(x, filters: int, strides=(1, 1)):
    """
    Depthwise Separable Conv block (the core S3-accelerated unit).

    Structure:
      DepthwiseConv2D  →  BN  →  ReLU6
      Conv2D (1×1)     →  BN  →  ReLU6

    ReLU6 is preferred over ReLU for INT8 quantisation because it
    clips activations at 6, preventing outliers that would waste
    the INT8 dynamic range.
    """
    # Depthwise: spatial filtering per channel
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)

    # Pointwise: channel mixing (1×1 conv)
    x = layers.Conv2D(filters, kernel_size=(1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    return x


def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    DS-CNN for sound gating.

    Layer summary (approx. param count):
    ┌─────────────────────────┬──────────────┬───────────┐
    │ Layer                   │ Output Shape │ Params    │
    ├─────────────────────────┼──────────────┼───────────┤
    │ Conv2D (stem)           │ 40×61×32     │ 320       │
    │ DS Block 1              │ 20×31×64     │ 2,368     │
    │ DS Block 2              │ 10×16×128    │ 9,344     │
    │ DS Block 3              │ 5×8×128      │ 17,664    │
    │ GlobalAveragePooling2D  │ 128          │ 0         │
    │ Dropout(0.3)            │ 128          │ 0         │
    │ Dense (softmax)         │ 3            │ 387       │
    └─────────────────────────┴──────────────┴───────────┘
    Total: ~30 KB weights (fits comfortably in 512 KB SRAM)
    """
    inp = keras.Input(shape=input_shape, name="mfcc_input")

    # ── Stem: one standard Conv2D to expand from 1 → 32 channels ──
    x = layers.Conv2D(
        32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu6", name="stem_act")(x)

    # ── DS Blocks (stride=2 halves spatial dims each time) ────────
    x = ds_block(x, filters=64,  strides=(2, 2))   # → 20×31×64
    x = ds_block(x, filters=128, strides=(2, 2))   # → 10×16×128
    x = ds_block(x, filters=128, strides=(2, 2))   # →  5×8×128

    # ── Global Average Pooling → avoids heavy Flatten ─────────────
    # Reduces spatial dims to a single 128-d vector.
    # Critical for TinyML: eliminates the large dense layer otherwise
    # needed after Flatten.
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ── Regularisation ────────────────────────────────────────────
    x = layers.Dropout(0.3, name="dropout")(x)

    # ── Output ────────────────────────────────────────────────────
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="SoundGate_DSCNN")
    return model


def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── TFLite Export with INT8 Quantisation ─────────────────
def export_tflite(model, representative_dataset_gen, output_path=TFLITE_PATH):
    """
    Convert to INT8 TFLite model using a representative dataset.

    The representative_dataset_gen must yield batches of (1, 40, 61, 1)
    float32 tensors from your training set. 100–200 samples is sufficient.

    INT8 quantisation maps float32 activations to int8 using per-tensor
    zero-point and scale factors — the ESP32-S3's AI accelerator operates
    natively in INT8, so this is NOT lossy in practice for CNN workloads.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable default optimisations (weight quantisation)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Full INT8: requires representative data so TFLite can
    # measure activation ranges for each layer
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"[export] Saved {output_path}  ({size_kb:.1f} KB)")
    print(f"[export] Run:  xxd -i {output_path} > model_data.h")
    return tflite_model


if __name__ == "__main__":
    m = build_model()
    m.summary()
