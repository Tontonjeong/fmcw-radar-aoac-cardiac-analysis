# Firmware

This directory contains ECG and SCG acquisition firmware.

## STM32 ECG

- STM32CubeIDE project
- ADC-based ECG signal acquisition
- TIM1 interrupt based 100 Hz sampling
- USART2 CSV output
- Output:

```text
sample_index,ADCValue,Smooth_ECG
```

## ESP32 MPU6050 SCG

- Arduino sketch
- MPU6050 6-axis IMU
- 100 Hz acquisition
- USB Serial CSV output
- Output:

```text
sample_index,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
```

## Synchronization Note

- ECG R-peak is used as the alignment anchor in the Python analysis.
- STM32 and ESP32 both output `sample_index` to support uniform time-axis reconstruction.
- Actual hardware-level synchronization may require further trigger-based synchronization if stricter timing validation is required.

## Before Running Python

- Confirm serial ports.
- Close Arduino Serial Monitor.
- Confirm STM32 UART output.
- Confirm radar SDK installation.
- Confirm BGT60TR13C device connection.
