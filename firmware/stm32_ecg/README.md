# STM32 ECG Firmware

## Purpose

This firmware samples ECG analog output using STM32 ADC and streams ECG data to Python over UART.

## Project

- STM32CubeIDE project or project archive
- MCU family: STM32F4
- Example MCU: STM32F411RET6

## Peripheral Configuration

| Peripheral | Configuration |
|---|---|
| ADC | ADC1_IN0, PA0 |
| Timer | TIM1 interrupt |
| UART | USART2 |
| UART TX | PA2 |
| UART RX | PA3 |
| Baudrate | 115200 |
| Sampling rate | 100 Hz target |

## Serial Output

CSV format:

```csv
sample_index,ADCValue,Smooth_ECG
```

Example:

```csv
0,1870,1860
1,1872,1861
```

## Signal Processing on MCU

- Raw ADC value acquisition
- Moving average smoothing
- Smoothing window size: 5 samples

## Notes

- Confirm timer clock before assuming exact 100 Hz.
- Confirm ADC reference voltage and analog front-end gain.
- Confirm UART COM port on PC.
- This firmware only streams ECG-like ADC data; medical-grade ECG validation is not implied.
