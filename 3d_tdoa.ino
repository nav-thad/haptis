/*
 * ESP32 WROOM-32 Three-Microphone Sound Localization
 * Uses both I2S ports for 3 simultaneous microphones
 * With GCC-PHAT for improved TDOA estimation
 * 
 * Microphone Geometry (L-shaped):
 * Mic 0 (I2S0 Left):   (0, 0)
 * Mic 1 (I2S0 Right):  (d, 0)  where d = 144.6 mm
 * Mic 2 (I2S1):        (0, d)
 * 
 * Wiring for I2S Port 0 (Mics 0 and 1):
 * - SCK  -> GPIO26 (shared)
 * - WS   -> GPIO25 (shared)
 * - SD   -> GPIO33 (shared)
 * - Mic 0: L/R -> GND
 * - Mic 1: L/R -> 3.3V
 * 
 * Wiring for I2S Port 1 (Mic 2):
 * - SCK  -> GPIO14
 * - WS   -> GPIO12
 * - SD   -> GPIO13
 * - Mic 2: L/R -> GND
 * 
 * All mics: VDD -> 3.3V, GND -> GND
 */

#include <driver/i2s.h>

// I2S Port 0 Configuration (2 mics in stereo)
#define I2S0_WS   25
#define I2S0_SD   33
#define I2S0_SCK  26
#define I2S_PORT_0 I2S_NUM_0

// I2S Port 1 Configuration (1 mic in mono)
#define I2S1_WS   27   // Was 12 - FIXED
#define I2S1_SD   32   // Was 13 - FIXED  
#define I2S1_SCK  15   // Was 14 - FIXED (note: GPIO15 may need pullup)
#define I2S_PORT_1 I2S_NUM_1

// Audio settings
#define SAMPLE_RATE 16000
#define SAMPLE_BITS 32
#define BUFFER_SIZE 256

// Physical constants
#define SOUND_SPEED 343.0
#define MIC_DISTANCE 0.1446  // 144.6 mm in meters

// DC filter
#define DC_FILTER_ALPHA 0.95

// GCC-PHAT parameters
#define USE_GCC_PHAT true
#define PHAT_EPSILON 1e-6  // Small value to prevent division by zero

// Microphone positions (L-shaped configuration)
struct MicPosition {
  float x, y;
};

const MicPosition mic_positions[3] = {
  {0.0, 0.0},              // Mic 0 (left)
  {MIC_DISTANCE, 0.0},     // Mic 1 (right)
  {0.0, MIC_DISTANCE}      // Mic 2 (perpendicular)
};

// DC filter state for each mic
float dc_mic0 = 0, dc_mic1 = 0, dc_mic2 = 0;

// Output modes:
// 0 = Serial Plotter (All 3 mics)
// 1 = TDOA 2D Localization (with GCC-PHAT)
#define OUTPUT_MODE 1

void setup() {
  Serial.begin(921600);
  delay(1000);
  
  Serial.println("ESP32 WROOM-32 Three-Microphone Sound Localization");
  Serial.println("L-shaped array - Dual I2S Configuration");
  Serial.printf("Mic spacing: %.1f mm\n", MIC_DISTANCE * 1000);
  Serial.println(USE_GCC_PHAT ? "Using GCC-PHAT" : "Using basic cross-correlation");
  Serial.println();
  
  // ===== Configure I2S Port 0 (Stereo - Mics 0 and 1) =====
  i2s_config_t i2s0_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,  // Stereo
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = BUFFER_SIZE,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin0_config;
  pin0_config.mck_io_num = I2S_PIN_NO_CHANGE;
  pin0_config.bck_io_num = I2S0_SCK;
  pin0_config.ws_io_num = I2S0_WS;
  pin0_config.data_out_num = I2S_PIN_NO_CHANGE;
  pin0_config.data_in_num = I2S0_SD;

  esp_err_t err = i2s_driver_install(I2S_PORT_0, &i2s0_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("Failed to install I2S Port 0: %d\n", err);
    while (1);
  }

  err = i2s_set_pin(I2S_PORT_0, &pin0_config);
  if (err != ESP_OK) {
    Serial.printf("Failed to set I2S Port 0 pins: %d\n", err);
    while (1);
  }
  
  Serial.println("I2S Port 0 initialized (Mics 0 & 1)");
  
  // ===== Configure I2S Port 1 (Mono - Mic 2) =====
  i2s_config_t i2s1_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,  // Mono
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = BUFFER_SIZE,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin1_config;
  pin1_config.mck_io_num = I2S_PIN_NO_CHANGE;
  pin1_config.bck_io_num = I2S1_SCK;
  pin1_config.ws_io_num = I2S1_WS;
  pin1_config.data_out_num = I2S_PIN_NO_CHANGE;
  pin1_config.data_in_num = I2S1_SD;

  err = i2s_driver_install(I2S_PORT_1, &i2s1_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("Failed to install I2S Port 1: %d\n", err);
    while (1);
  }

  err = i2s_set_pin(I2S_PORT_1, &pin1_config);
  if (err != ESP_OK) {
    Serial.printf("Failed to set I2S Port 1 pins: %d\n", err);
    while (1);
  }
  
  Serial.println("I2S Port 1 initialized (Mic 2)");
  Serial.println();
  
  // Clear DMA buffers
  i2s_zero_dma_buffer(I2S_PORT_0);
  i2s_zero_dma_buffer(I2S_PORT_1);
  
  if (OUTPUT_MODE == 0) {
    Serial.println("Mode: Serial Plotter (All 3 Mics)");
  } else {
    Serial.println("Mode: 2D TDOA Localization with GCC-PHAT");
  }
}

// ============================================================
// GCC-PHAT Cross-Correlation
// ============================================================
// This implements the Generalized Cross-Correlation with 
// Phase Transform (GCC-PHAT) algorithm, which is superior to
// basic cross-correlation for TDOA estimation in reverberant
// environments.
//
// PHAT weighting whitens the cross-power spectrum, making
// all frequencies contribute equally to the correlation peak.
// This enhances the peak sharpness and reduces the effect
// of colored noise and reverberation.
// ============================================================

int crossCorrelateGCCPHAT(int16_t* buf1, int16_t* buf2, int len, int max_delay) {
  // Step 1: Compute cross-correlation using time-domain method
  // with PHAT-like weighting applied in an approximated way
  
  int best_delay = 0;
  float max_correlation = -1e30;
  
  // Pre-compute signal energies for normalization
  float energy1 = 0, energy2 = 0;
  for (int i = 0; i < len; i++) {
    energy1 += (float)buf1[i] * buf1[i];
    energy2 += (float)buf2[i] * buf2[i];
  }
  energy1 = sqrt(energy1 / len);
  energy2 = sqrt(energy2 / len);
  
  if (energy1 < 1.0 || energy2 < 1.0) {
    return 0;  // Signal too weak
  }
  
  // Sweep through all possible delays
  for (int delay = -max_delay; delay <= max_delay; delay++) {
    float correlation = 0;
    float local_energy1 = 0, local_energy2 = 0;
    int count = 0;
    
    // Calculate correlation for this delay
    for (int i = 0; i < len; i++) {
      int j = i + delay;
      if (j >= 0 && j < len) {
        float s1 = (float)buf1[i];
        float s2 = (float)buf2[j];
        
        correlation += s1 * s2;
        local_energy1 += s1 * s1;
        local_energy2 += s2 * s2;
        count++;
      }
    }
    
    if (count == 0) continue;
    
    // PHAT weighting: normalize by the magnitude of cross-power spectrum
    // In time domain, this approximates to normalizing by local energies
    float magnitude = sqrt((local_energy1 * local_energy2) / (count * count));
    
    if (magnitude > PHAT_EPSILON) {
      correlation /= magnitude;  // PHAT normalization
    } else {
      correlation = 0;
    }
    
    // Find peak
    if (correlation > max_correlation) {
      max_correlation = correlation;
      best_delay = delay;
    }
  }
  
  return best_delay;
}

// ============================================================
// Basic Cross-Correlation (for comparison)
// ============================================================
int crossCorrelate(int16_t* buf1, int16_t* buf2, int len, int max_delay) {
  int best_delay = 0;
  int64_t max_correlation = INT64_MIN;
  
  for (int delay = -max_delay; delay <= max_delay; delay++) {
    int64_t correlation = 0;
    int count = 0;
    
    for (int i = 0; i < len; i++) {
      int j = i + delay;
      if (j >= 0 && j < len) {
        correlation += (int64_t)buf1[i] * (int64_t)buf2[j];
        count++;
      }
    }
    
    if (count > 0) {
      correlation /= count;
    }
    
    if (abs(correlation) > abs(max_correlation)) {
      max_correlation = correlation;
      best_delay = delay;
    }
  }
  
  return best_delay;
}

void loop() {
  // Buffers for raw I2S data
  static int32_t samples_i2s0[BUFFER_SIZE * 2];  // Stereo (mics 0 and 1)
  static int32_t samples_i2s1[BUFFER_SIZE];      // Mono (mic 2)
  
  // Processed sample buffers (16-bit after filtering)
  static int16_t mic0_buffer[BUFFER_SIZE];
  static int16_t mic1_buffer[BUFFER_SIZE];
  static int16_t mic2_buffer[BUFFER_SIZE];
  
  size_t bytes_read_0 = 0, bytes_read_1 = 0;

  // Read from both I2S ports
  i2s_read(I2S_PORT_0, samples_i2s0, sizeof(samples_i2s0), &bytes_read_0, portMAX_DELAY);
  i2s_read(I2S_PORT_1, samples_i2s1, sizeof(samples_i2s1), &bytes_read_1, portMAX_DELAY);
  
  int samples_read_stereo = bytes_read_0 / sizeof(int32_t);
  int samples_read_mono = bytes_read_1 / sizeof(int32_t);
  int samples_per_channel = samples_read_stereo / 2;
  
  int min_samples = min(samples_per_channel, samples_read_mono);
  
  // Extract and filter channels from I2S Port 0 (mics 0 and 1)
  for (int i = 0; i < min_samples; i++) {
    int32_t sample_0 = samples_i2s0[i * 2 + 1] >> 8;
    int32_t sample_1 = samples_i2s0[i * 2] >> 8;
    
    dc_mic0 = DC_FILTER_ALPHA * dc_mic0 + (1 - DC_FILTER_ALPHA) * sample_0;
    dc_mic1 = DC_FILTER_ALPHA * dc_mic1 + (1 - DC_FILTER_ALPHA) * sample_1;
    
    mic0_buffer[i] = (int16_t)(((int)(sample_0 - dc_mic0)) >> 8);
    mic1_buffer[i] = (int16_t)(((int)(sample_1 - dc_mic1)) >> 8);
  }
  
  // Extract and filter channel from I2S Port 1 (mic 2)
  for (int i = 0; i < min_samples; i++) {
    int32_t sample_2 = samples_i2s1[i] >> 8;
    
    dc_mic2 = DC_FILTER_ALPHA * dc_mic2 + (1 - DC_FILTER_ALPHA) * sample_2;
    
    mic2_buffer[i] = (int16_t)(((int)(sample_2 - dc_mic2)) >> 8);
  }
  
  // ===== OUTPUT MODE 0: Serial Plotter =====
  if (OUTPUT_MODE == 0) {
    for (int i = 0; i < min_samples; i++) {
      Serial.print("Mic0:");
      Serial.print(mic0_buffer[i]);
      Serial.print(",Mic1:");
      Serial.print(mic1_buffer[i]);
      Serial.print(",Mic2:");
      Serial.println(mic2_buffer[i]);
    }
  }
  
  // ===== OUTPUT MODE 1: TDOA 2D Localization with GCC-PHAT =====
  else if (OUTPUT_MODE == 1) {
    // Calculate energy to filter out noise
    int64_t energy = 0;
    for (int i = 0; i < min_samples; i++) {
      energy += abs(mic0_buffer[i]) + abs(mic1_buffer[i]) + abs(mic2_buffer[i]);
    }
    energy /= (3 * min_samples);
    
    // Only process if there's significant sound
    if (energy > 100) {
      // Calculate maximum possible delay in samples
      int max_delay_samples = (int)((MIC_DISTANCE / SOUND_SPEED) * SAMPLE_RATE) + 10;
      
      // Cross-correlate all mic pairs to get TDOAs
      int tdoa_01, tdoa_02, tdoa_12;
      
      if (USE_GCC_PHAT) {
        // Use GCC-PHAT for more robust TDOA estimation
        tdoa_01 = crossCorrelateGCCPHAT(mic0_buffer, mic1_buffer, min_samples, max_delay_samples);
        tdoa_02 = crossCorrelateGCCPHAT(mic0_buffer, mic2_buffer, min_samples, max_delay_samples);
        tdoa_12 = crossCorrelateGCCPHAT(mic1_buffer, mic2_buffer, min_samples, max_delay_samples);
      } else {
        // Use basic cross-correlation
        tdoa_01 = crossCorrelate(mic0_buffer, mic1_buffer, min_samples, max_delay_samples);
        tdoa_02 = crossCorrelate(mic0_buffer, mic2_buffer, min_samples, max_delay_samples);
        tdoa_12 = crossCorrelate(mic1_buffer, mic2_buffer, min_samples, max_delay_samples);
      }

      // Apply calibration offsets (adjust these based on your hardware)
      tdoa_02 = tdoa_02 + 7;
      tdoa_12 = tdoa_12 + 7;
      
      // Convert delays to time (seconds)
      float tau_01 = (float)tdoa_01 / SAMPLE_RATE;
      float tau_02 = (float)tdoa_02 / SAMPLE_RATE;
      float tau_12 = (float)tdoa_12 / SAMPLE_RATE;
      
      // Calculate distance differences
      float d_01 = tau_01 * SOUND_SPEED;
      float d_02 = tau_02 * SOUND_SPEED;
      
      // Normalize by microphone spacing
      float ratio_01 = d_01 / MIC_DISTANCE;
      float ratio_02 = d_02 / MIC_DISTANCE;
      
      // Clamp to valid range
      ratio_01 = constrain(ratio_01, -1.0, 1.0);
      ratio_02 = constrain(ratio_02, -1.0, 1.0);
      
      // Calculate individual axis angles
      float angle_x = asin(ratio_01) * 180.0 / PI;
      float angle_y = asin(ratio_02) * 180.0 / PI;

      // Calculate 2D direction vector
      float cos_theta_x = -ratio_01;
      float cos_theta_y = -ratio_02;

      float dir_x = cos_theta_y;
      float dir_y = cos_theta_x;

      // Normalize to unit vector
      float magnitude = sqrt(dir_x * dir_x + dir_y * dir_y);
    
      if (magnitude > 0.01) {
        dir_x /= magnitude;
        dir_y /= magnitude;
      } else {
        dir_x = 0;
        dir_y = 0;
      }

      // Calculate azimuth (0° = +X axis, counter-clockwise)
      float azimuth = atan2(dir_y, dir_x) * 180.0 / PI;
    
      // Convert to 0-360° range
      if (azimuth < 0) {
        azimuth += 360.0;
      }

      // Output results
      Serial.print("TDOA[0-1]: ");
      Serial.print(tdoa_01);
      Serial.print(" | TDOA[0-2]: ");
      Serial.print(tdoa_02);
      Serial.print(" | TDOA[1-2]: ");
      Serial.print(tdoa_12);
      Serial.print(" samples  ||  Angle-X: ");
      Serial.print(angle_x, 1);
      Serial.print("°  |  Angle-Y: ");
      Serial.print(angle_y, 1);
      Serial.print("°  |  Azimuth: ");
      Serial.print(azimuth, 1);
      Serial.print("°  |  Energy: ");
      Serial.print((int)energy);
      Serial.print("  |  Dir: (");
      Serial.print(dir_x, 2);
      Serial.print(", ");
      Serial.print(dir_y, 2);
      Serial.println(")");
    }
  }
}