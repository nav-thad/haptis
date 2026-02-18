Haptis: ESP32-S3 Sound Gating + Localization
============================================

Real-time sound gating CNN for ESP32-S3 haptic feedback.

The system classifies 1-second audio windows as:
- noise
- voice
- alarm

Haptic feedback is triggered only on meaningful sounds (voice or alarm).
When triggered, GCC-PHAT localization runs using 3 microphones to estimate direction of arrival (DOA).

Performance Summary
-------------------
- Val accuracy: 99.00% | Val loss: 0.0338
Noise suppression (TEST SPLIT): 93.9% (107/114)
Voice trigger (TEST SPLIT): 98.7% (6076/6159)
- Model size: 51 KB (INT8 quantized)
- Estimated Peak SRAM usage: 213 KB (model + GCC-PHAT + buffers)


Pipeline Overview
-----------------
Datasets → Preprocessing → Training → Quantization → ESP32 Deployment

Datasets:
- ESC-50 (alarms + environmental noise)
- Google Speech Commands (voice)

Processing:
- Raw audio → 16 kHz, 1 second clips
- MFCC features (40 mel × 61 time × 1)
- DS-CNN classifier (31K parameters)
- Keras → TFLite INT8 quantization
- Model reduced from 123 KB to 51 KB (-58%)

Runtime flow:
1. 3× I2S microphones capture audio
2. RMS-based mic selection (choses mic closest to sound source)
3. CNN gating (voice/alarm vs noise)
4. If gate = true → GCC-PHAT DOA (3 mics)
5. Haptic motors activated based on angle


Model Architecture
------------------
Input: 40 × 61 × 1 (9,800 features)

1. Conv2D 3×3, 32 filters (stem)       → 40 × 61 × 32
2. DS-Block1, stride 2, 64 filters     → 20 × 31 × 64
3. DS-Block2, stride 2, 128 filters    → 10 × 16 × 128
4. DS-Block3, stride 2, 128 filters    → 5 × 8 × 128
5. GlobalAveragePooling2D              → 128
6. Dropout(0.3)                        → 128
7. Dense(3) softmax                    → [noise, voice, alarm]

Total parameters: 31,619  
Quantized size: 51 KB (INT8)

Gating Rule:
(voice OR alarm) AND probability > 0.85  
→ HAPTIC ON  
→ Run localization


Laptop Setup & Training
=======================

Requirements
------------
```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt # TensorFlow 2.15, numpy, scipy, sounddevice, etc.
```


## 1. Build Dataset
-----------------------------
```
external/
├── ESC-50/
├── speech_commands/
└── build_dataset.py
```

Run:
python build_dataset.py

This downloads and processes all audio into:
data/{noise,voice,alarm}/*.wav

All clips are converted to:
- 16 kHz
- 1 second duration


## 2. Train
---------------------
python train.py

Outputs:
- sound_gate_model.keras
- model.tflite (51 KB)


## 3. Evaluate
-----------
python evaluate.py

Outputs:
- Validation metrics
- Progress bars
- Live microphone demo

Metrics Achieved So Far:
- Noise suppression: 94.7%
- Voice trigger rate: 99.3%


## 4. Live Test
------------
python -c "from evaluate import SoundGateInferencer, live_demo; SoundGateInferencer().live_demo(15)"


ESP32-S3 Deployment 
=============================

Hardware:
- ESP32-S3-DEVKITC-1-N32R16V
- 3× I2S microphones

Memory Budget:
- 51 KB model
- 113 KB GCC-PHAT
- Total peak SRAM: 213 KB
- PSRAM: 1 MB audio ring buffer
- Latency: ~80 ms (~12 FPS)


Main Runtime Pipeline on ESP32-S2 (main/haptis.c)
-------------------------------------

1. I2S DMA reads 3 microphones into ring buffers.
2. Every 1 second:
   - Compute RMS
   - Select loudest microphone
   - Extract MFCC
   - Run TFLite gate
3. If gate is triggered:
   - Run GCC-PHAT on all 3 mics
   - Compute azimuth
   - Drive haptic motors based on angle


TFLite Micro Inference
----------------------
- Copy MFCC buffer into model input tensor
- Invoke interpreter
- Read output probabilities
- Apply gating threshold


Build & Flash
-------------
idf.py set-target esp32s3
idf.py menuconfig   # Enable PSRAM, I2S0/1/2
idf.py build
idf.py -p /dev/cu.usbserial-* flash monitor


Expanding with More Data or Classes
===================================

Add New Classes
---------------
Edit build_dataset.py:
CLASSES = ["noise", "voice", "alarm", "baby_cry"]
alarm_labels += {"baby_cry"}

Edit config.py:
CLASSES = ["noise", "voice", "alarm", "baby_cry"]
HAPTIC_CLASSES = {"voice", "alarm", "baby_cry"}

Retrain:
python build_dataset.py
python train.py

The system automatically detects new classes.


Custom Datasets
---------------
Place custom WAV files in:

data/custom/
├── siren/
├── baby_cry/
└── my_noise/

Folders matching CLASSES are automatically loaded.


Balance Classes
---------------
In train.py:

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)

model.fit(..., class_weight=dict(enumerate(class_weights)))


Performance
===========
Validation Accuracy: 99.0%  
Noise Rejection: 94.7%
Voice/Alarm Trigger: 99.3%
Model Size: 51 KB INT8  
SRAM Peak: 213 KB  
Latency: ~80 ms (12 FPS)


Troubleshooting
===============

Issue: numba errors  
Fix: Get rid of librosa library

Issue: Low voice trigger rate  
Fix: Lower HAPTIC_THRESHOLD to 0.7  

Issue: Weak alarm classification  
Fix: Add more siren/alarm samples to build_dataset.py  

Issue: ESP32 out-of-memory  
Fix: Reduce tensor_arena to 80 KB
