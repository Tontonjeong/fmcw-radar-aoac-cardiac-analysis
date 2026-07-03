# Firmware

This folder contains firmware resources and documentation for ECG and SCG acquisition used by the AO/AC timing analysis pipeline.

## STM32 ECG

- STM32CubeIDE project or project archive
- ADC-based ECG signal acquisition
- TIM1 interrupt-based 100 Hz target sampling
- USART2 CSV output
- Output:

```csv
sample_index,ADCValue,Smooth_ECG
```

## ESP32 MPU6050 SCG

- Arduino sketch
- MPU6050 6-axis IMU acquisition
- 100 Hz target acquisition
- USB Serial CSV output
- Output:

```csv
sample_index,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
```

## Synchronization Note

ECG R-peak is used as the alignment anchor in the Python analysis. STM32 and ESP32 both output `sample_index` to support uniform time-axis reconstruction. Actual hardware-level synchronization may require further trigger-based synchronization if stricter timing validation is required.

## Before Running Python

- Confirm serial ports.
- Close Arduino Serial Monitor.
- Confirm STM32 UART output.
- Confirm radar SDK installation.
- Confirm BGT60TR13C device connection.
