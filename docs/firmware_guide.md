# Firmware Guide

## Documentation

- [Algorithm Details](algorithm_details.md)
- [Configuration Reference](configuration_reference.md)
- [Code Reference](code_reference.md)
- [Firmware Guide](firmware_guide.md)
- [Output Reference](output_reference.md)
- [Research Notes](research_notes.md)
- [STM32F411 ECG Firmware Configuration](stm32_f411_ecg_firmware.md)


## STM32F411 ECG Firmware

The STM32 project is stored under `firmware/stm32_ecg/ECG_project/`. It targets an STM32F411RETx / STM32F411RET6-class board in an LQFP64 package and streams ECG-like ADC samples over USART2.

### Wiring and Peripheral Mapping

| Function | Setting |
|---|---|
| ECG analog input | ADC1_IN0 on PA0 |
| UART TX | USART2_TX on PA2 |
| UART RX | USART2_RX on PA3 |
| Baudrate | 115200 |
| Timer | TIM1 update interrupt |
| TIM1 settings | Prescaler 9999, period 99 |
| Target sampling | 100 Hz |
| Clock | 100 MHz SYSCLK/HCLK |

### CSV Output

```csv
sample_index,ADCValue,Smooth_ECG
0,1870,1860
1,1872,1861
```

`Smooth_ECG` is produced using a 5-sample moving average in the embedded firmware. The Python script can use the raw or smoothed column depending on configuration.

### Flashing Notes

Open the project in STM32CubeIDE, confirm the board/clock/ADC/UART settings, build, and flash. Before recording data, confirm that the host PC sees the USART2 virtual COM port and that a serial terminal can observe CSV rows at 115200 baud.

## ESP32 MPU6050 SCG Firmware

The Arduino sketch is stored under `firmware/esp32_mpu6050_scg/`.

### Wiring

| MPU6050 | ESP32 |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | GPIO21 |
| SCL | GPIO22 |

### Parameters

| Item | Value |
|---|---|
| Baudrate | 115200 |
| Sampling target | 100 Hz |
| I2C clock | 400 kHz |
| Accelerometer scale | +/-2 g |
| Gyroscope scale | +/-250 dps |
| Startup bias samples | 300 |

### CSV Output

```csv
sample_index,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
0,0,0.001234,-0.002345,0.003456,0.1234,-0.2345,0.3456
```

Keep the sensor still during startup calibration. Close Arduino Serial Monitor before starting Python acquisition, because only one process can usually hold the COM port.

## Time-Axis Reconstruction

| Stream | Preferred Time Axis |
|---|---|
| STM32 ECG | `sample_index / ECG_FS_HINT_HZ` |
| ESP32 SCG | `t_ms / 1000` or `sample_index / SCG_FS_HINT_HZ` |
| Radar | Radar frame timestamps, interpolated to analysis rate |

## Common Troubleshooting

| Symptom | Likely Cause | Action |
|---|---|---|
| Python cannot open COM port | Port is wrong or already open | Check Device Manager and close serial monitors |
| Arduino Serial Monitor still open | COM port locked by Arduino IDE | Close monitor before Python acquisition |
| No ECG data arriving | STM32 UART not streaming or wrong baudrate | Confirm USART2, PA2/PA3, and 115200 baud |
| SCG lines start with `#` | Firmware header/comment lines | Parser should ignore comment/header lines |
| Sampling rate mismatch | Timer or loop timing differs from target | Verify TIM1 settings and ESP32 loop timing |
| Radar SDK not installed | Missing `ifxradarsdk` | Install Infineon SDK per vendor instructions |
| BGT60TR13C not detected | USB/device/driver issue | Reconnect radar and verify SDK examples |
