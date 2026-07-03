# Serial Output Examples

## STM32 ECG

```csv
sample_index,ADCValue,Smooth_ECG
0,1870,1860
1,1872,1861
```

## ESP32 MPU6050 SCG

```csv
sample_index,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
0,0,0.001234,-0.002345,0.003456,0.1234,-0.2345,0.3456
```

## Python Parsing Notes

- ECG time axis can be reconstructed from `sample_index / ECG_FS_HINT_HZ`.
- SCG time axis can be reconstructed from `sample_index / SCG_FS_HINT_HZ` or `t_ms`.
- The Python analysis should treat lines starting with `#` as comments or headers.
