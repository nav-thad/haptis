/*
 * ESP32 WROOM-32 Three-Microphone Sound Localization
 * Uses both I2S ports for 3 simultaneous microphones
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
// #define BUFFER_SIZE 128
#define BUFFER_SIZE 256

// Physical constants
#define SOUND_SPEED 343.0
#define MIC_DISTANCE 0.1446  // 144.6 mm in meters

// DC filter
#define DC_FILTER_ALPHA 0.95

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
// 1 = TDOA 2D Localization
#define OUTPUT_MODE 1

void setup() {
  Serial.begin(921600);
  delay(1000);
  
  Serial.println("ESP32 WROOM-32 Three-Microphone Sound Localization");
  Serial.println("L-shaped array - Dual I2S Configuration");
  Serial.printf("Mic spacing: %.1f mm\n", MIC_DISTANCE * 1000);
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
    Serial.println("Mode: 2D TDOA Localization");
  }
}

// Cross-correlation function
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

// Calculate 2D position from TDOAs using hyperbolic trilateration
// void calculate2DPosition(float tau_01, float tau_02, float tau_12, float* x, float* y) {
//   // Distance differences from time delays
//   float d_01 = tau_01 * SOUND_SPEED;  // Distance diff between mic0 and mic1
//   float d_02 = tau_02 * SOUND_SPEED;  // Distance diff between mic0 and mic2
  
//   // Using simplified geometric solution for L-shaped array
//   // For mic positions: (0,0), (d,0), (0,d)
//   // Hyperbola equations: sqrt(x^2 + y^2) - sqrt((x-d)^2 + y^2) = d_01
//   //                      sqrt(x^2 + y^2) - sqrt(x^2 + (y-d)^2) = d_02
  
//   // Approximate solution using linear approximation for far-field sources
//   float d = MIC_DISTANCE;
  
//   // For sources far from the array (far-field approximation)
//   // The angle from the x-axis (mic 0-1 line)
//   float theta_x = asin(constrain(d_01 / d, -1.0, 1.0));
  
//   // The angle from the y-axis (mic 0-2 line)
//   float theta_y = asin(constrain(d_02 / d, -1.0, 1.0));
  
//   // Convert to Cartesian direction (normalized)
//   float dir_x = cos(theta_y);  // Component along x
//   float dir_y = cos(theta_x);  // Component along y
  
//   // Normalize
//   float mag = sqrt(dir_x * dir_x + dir_y * dir_y);
//   if (mag > 0.001) {
//     *x = dir_x / mag;
//     *y = dir_y / mag;
//   } else {
//     *x = 0;
//     *y = 0;
//   }
// }

// void calculate2DPosition(float tau_01, float tau_02, float tau_12, float* azimuth) {
//   float d = MIC_DISTANCE;

//   // Calculate distance differences
//   float d_01 = tau_01 * SOUND_SPEED;  // Along X-axis (mic 0-1)
//   float d_02 = tau_02 * SOUND_SPEED;  // Along Y-axis (mic 0-2)

//   // Far-field approximation: d_ij ≈ d * sin(theta_ij)
//   // For L-shaped array, we can determine quadrant using sign information

//   // Direction cosines (unnormalized)
//   // Positive tau_01: source closer to mic 1 (positive X direction)
//   // Positive tau_02: source closer to mic 2 (positive Y direction)

//   // The key insight: use the cross-correlation to determine the sign
//   // tau > 0 means signal arrives at second mic first
//   float sin_theta_x = constrain(d_01 / d, -1.0, 1.0);
//   float sin_theta_y = constrain(d_02 / d, -1.0, 1.0);

//   // Calculate complementary angles
//   float cos_theta_x = sqrt(1.0 - sin_theta_x * sin_theta_x);
//   float cos_theta_y = sqrt(1.0 - sin_theta_y * sin_theta_y);

//   // Direction vector components
//   // X component: determined by angle from Y-axis pair
//   // Y component: determined by angle from X-axis pair
//   float dir_x = cos_theta_y;
//   float dir_y = cos_theta_x;

//   // Apply sign based on which microphone received signal first
//   if (tau_01 < 0) dir_x = -dir_x;  // Source in negative X
//   if (tau_02 < 0) dir_y = -dir_y;  // Source in negative Y

//   // Convert to azimuth (0° = +X axis, counterclockwise)
//   azimuth = atan2(dir_y, dir_x) 180.0 / PI;

//   // Normalize to 0-360 range
//   if (azimuth < 0)azimuth += 360.0;
// }

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
  // Note: These are hardware-synchronized since they're on the same MCU clock
  i2s_read(I2S_PORT_0, samples_i2s0, sizeof(samples_i2s0), &bytes_read_0, portMAX_DELAY);
  i2s_read(I2S_PORT_1, samples_i2s1, sizeof(samples_i2s1), &bytes_read_1, portMAX_DELAY);
  
  int samples_read_stereo = bytes_read_0 / sizeof(int32_t);
  int samples_read_mono = bytes_read_1 / sizeof(int32_t);
  int samples_per_channel = samples_read_stereo / 2;
  
  // Ensure we have matching sample counts
  int min_samples = min(samples_per_channel, samples_read_mono);
  
  // Extract and filter channels from I2S Port 0 (mics 0 and 1)
  for (int i = 0; i < min_samples; i++) {
    // Note: Check your hardware to confirm L/R channel ordering
    int32_t sample_0 = samples_i2s0[i * 2 + 1] >> 8;     // Left channel = Mic 0
    int32_t sample_1 = samples_i2s0[i * 2] >> 8;         // Right channel = Mic 1
    
    // Apply DC filters
    dc_mic0 = DC_FILTER_ALPHA * dc_mic0 + (1 - DC_FILTER_ALPHA) * sample_0;
    dc_mic1 = DC_FILTER_ALPHA * dc_mic1 + (1 - DC_FILTER_ALPHA) * sample_1;
    
    // Fixed: Cast to int before bit-shifting
    mic0_buffer[i] = (int16_t)(((int)(sample_0 - dc_mic0)) >> 8);
    mic1_buffer[i] = (int16_t)(((int)(sample_1 - dc_mic1)) >> 8);
  }
  
  // Extract and filter channel from I2S Port 1 (mic 2)
  for (int i = 0; i < min_samples; i++) {
    int32_t sample_2 = samples_i2s1[i] >> 8;
    
    dc_mic2 = DC_FILTER_ALPHA * dc_mic2 + (1 - DC_FILTER_ALPHA) * sample_2;
    
    // Fixed: Cast to int before bit-shifting
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
  
  // ===== OUTPUT MODE 1: TDOA 2D Localization =====
  else if (OUTPUT_MODE == 1) {
    // Calculate energy to filter out noise
    int64_t energy = 0;
    for (int i = 0; i < min_samples; i++) {
      energy += abs(mic0_buffer[i]) + abs(mic1_buffer[i]) + abs(mic2_buffer[i]);
    }
    energy /= (3 * min_samples);
    
    // Only process if there's significant sound
    if (energy > 100) {  // Adjust threshold as needed
      // Calculate maximum possible delay in samples
      int max_delay_samples = (int)((MIC_DISTANCE / SOUND_SPEED) * SAMPLE_RATE) + 10;
      
      // Cross-correlate all mic pairs to get TDOAs
      int tdoa_01 = crossCorrelate(mic0_buffer, mic1_buffer, min_samples, max_delay_samples);
      int tdoa_02 = crossCorrelate(mic0_buffer, mic2_buffer, min_samples, max_delay_samples);
      int tdoa_12 = crossCorrelate(mic1_buffer, mic2_buffer, min_samples, max_delay_samples);

      tdoa_02 = tdoa_02 + 7;
      tdoa_12 = tdoa_12 + 7;
      
      // Convert delays to time (seconds)
      float tau_01 = (float)tdoa_01 / SAMPLE_RATE;
      float tau_02 = (float)tdoa_02 / SAMPLE_RATE;
      float tau_12 = (float)tdoa_12 / SAMPLE_RATE;
      
      // Calculate angles from individual mic pairs
      float d_01 = tau_01 * SOUND_SPEED;
      float d_02 = tau_02 * SOUND_SPEED;
      
      float ratio_01 = d_01 / MIC_DISTANCE;
      float ratio_02 = d_02 / MIC_DISTANCE;
      
      // Clamp to valid range
      ratio_01 = constrain(ratio_01, -1.0, 1.0);
      ratio_02 = constrain(ratio_02, -1.0, 1.0);
      
      float angle_x = asin(ratio_01) * 180.0 / PI;  // Angle from x-axis (mic 0-1)
      float angle_y = asin(ratio_02) * 180.0 / PI;  // Angle from y-axis (mic 0-2)

      float cos_theta_x = -ratio_01;  // Cosine of angle from X-axis
      float cos_theta_y = -ratio_02;  // Cosine of angle from Y-axis

      float dir_x = cos_theta_y;  // X-component of sound direction
      float dir_y = cos_theta_x;  // Y-component of sound direction

      float magnitude = sqrt(dir_x * dir_x + dir_y * dir_y);
    
      if (magnitude > 0.01) {  // Avoid division by zero
        dir_x /= magnitude;
        dir_y /= magnitude;
      } else {
        // If magnitude is too small, we can't determine direction
        dir_x = 0;
        dir_y = 0;
      }

      float azimuth = atan2(dir_y, dir_x) * 180.0 / PI;
    
      // Convert from -180° to +180° range into 0° to 360° range
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