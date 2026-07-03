/*
  ESP32 MPU6050 SCG Firmware

  Purpose:
    Acquires 6-axis MPU6050 inertial data at 100 Hz for SCG-like
    chest vibration measurement and streams CSV over USB Serial.

  Output CSV:
    sample_index,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps

  Research prototype only. Not medical diagnosis software.
*/

/*
  ESP32 / ESP32-S2 + MPU6050 SCG acquisition @ 100 Hz
  Output CSV over USB Serial:
    sample_index,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps

  Wiring:
    MPU6050 VCC -> 3.3V
    MPU6050 GND -> GND
    MPU6050 SDA -> GPIO21
    MPU6050 SCL -> GPIO22

  Notes:
    - Close Arduino Serial Monitor before Python uses COM25.
    - Python assumes SCG sampling = 100 Hz.
    - This sketch prints 8 columns only.
*/

#include <Wire.h>

#define MPU_ADDR 0x68
#define SDA_PIN 21
#define SCL_PIN 22
#define SERIAL_BAUD 115200
#define FS_HZ 100
#define SAMPLE_PERIOD_US (1000000UL / FS_HZ)

const float ACC_SCALE = 16384.0f; // ±2g
const float GYRO_SCALE = 131.0f;  // ±250 dps

long ax_bias = 0, ay_bias = 0, az_bias = 0;
long gx_bias = 0, gy_bias = 0, gz_bias = 0;

uint32_t sample_index = 0;
uint32_t next_sample_us = 0;
uint32_t start_ms = 0;

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission(true);
}

bool readRaw(int16_t &ax, int16_t &ay, int16_t &az,
             int16_t &gx, int16_t &gy, int16_t &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;

  uint8_t n = Wire.requestFrom(MPU_ADDR, (uint8_t)14, (uint8_t)true);
  if (n != 14) return false;

  ax = (int16_t)((Wire.read() << 8) | Wire.read());
  ay = (int16_t)((Wire.read() << 8) | Wire.read());
  az = (int16_t)((Wire.read() << 8) | Wire.read());
  (void)((Wire.read() << 8) | Wire.read());
  gx = (int16_t)((Wire.read() << 8) | Wire.read());
  gy = (int16_t)((Wire.read() << 8) | Wire.read());
  gz = (int16_t)((Wire.read() << 8) | Wire.read());
  return true;
}

void calibrateBias(int n = 300) {
  long sx = 0, sy = 0, sz = 0;
  long sgx = 0, sgy = 0, sgz = 0;
  int valid = 0;

  delay(500);
  for (int i = 0; i < n; i++) {
    int16_t ax, ay, az, gx, gy, gz;
    if (readRaw(ax, ay, az, gx, gy, gz)) {
      sx += ax; sy += ay; sz += az;
      sgx += gx; sgy += gy; sgz += gz;
      valid++;
    }
    delay(5);
  }

  if (valid > 0) {
    ax_bias = sx / valid;
    ay_bias = sy / valid;
    az_bias = sz / valid;
    gx_bias = sgx / valid;
    gy_bias = sgy / valid;
    gz_bias = sgz / valid;
  }
}

void setupMPU6050() {
  writeReg(0x6B, 0x00);
  delay(100);
  writeReg(0x1A, 0x03);
  writeReg(0x1B, 0x00);
  writeReg(0x1C, 0x00);
  writeReg(0x19, 0x09);
  delay(100);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1200);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  setupMPU6050();
  calibrateBias(300);

  start_ms = millis();
  next_sample_us = micros();
  sample_index = 0;

  Serial.println("# ESP32_MPU6050_SCG_100Hz_INDEXED_8COL");
  Serial.println("# sample_index,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
}

void loop() {
  uint32_t now_us = micros();
  if ((int32_t)(now_us - next_sample_us) < 0) return;
  next_sample_us += SAMPLE_PERIOD_US;

  int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
  if (!readRaw(ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw)) return;

  float ax_g = (float)(ax_raw - ax_bias) / ACC_SCALE;
  float ay_g = (float)(ay_raw - ay_bias) / ACC_SCALE;
  float az_g = (float)(az_raw - az_bias) / ACC_SCALE;
  float gx_dps = (float)(gx_raw - gx_bias) / GYRO_SCALE;
  float gy_dps = (float)(gy_raw - gy_bias) / GYRO_SCALE;
  float gz_dps = (float)(gz_raw - gz_bias) / GYRO_SCALE;

  uint32_t t_ms = millis() - start_ms;

  Serial.print(sample_index);
  Serial.print(',');
  Serial.print(t_ms);
  Serial.print(',');
  Serial.print(ax_g, 6);
  Serial.print(',');
  Serial.print(ay_g, 6);
  Serial.print(',');
  Serial.print(az_g, 6);
  Serial.print(',');
  Serial.print(gx_dps, 4);
  Serial.print(',');
  Serial.print(gy_dps, 4);
  Serial.print(',');
  Serial.println(gz_dps, 4);

  sample_index++;
}
