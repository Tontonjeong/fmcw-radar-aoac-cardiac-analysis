# ESP32 MPU6050 SCG Firmware

## Purpose

This firmware acquires 6-axis inertial data from an MPU6050 sensor at 100 Hz for SCG-like chest vibration measurement.

## Wiring

| MPU6050 | ESP32 |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | GPIO21 |
| SCL | GPIO22 |

## Serial Output

CSV format:

```text
sample_index,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
```

## Parameters

| Parameter | Value |
|---|---|
| Baudrate | 115200 |
| Sampling rate | 100 Hz |
| Accelerometer scale | +/-2g |
| Gyroscope scale | +/-250 dps |
| I2C clock | 400 kHz |

## Notes

- The sketch performs initial bias calibration.
- Keep the sensor still during startup calibration.
- Close Arduino Serial Monitor before running the Python acquisition script.
