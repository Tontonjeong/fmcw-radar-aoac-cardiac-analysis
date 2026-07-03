# STM32 ECG Project Archive

The STM32 ECG firmware project was provided as `ECG.zip` during repository preparation.

Because the GitHub connector used in this session supports reliable UTF-8 text-file creation but does not provide a direct large binary upload workflow, the STM32 project archive should be uploaded manually if it needs to be distributed in this repository.

Recommended target path:

```text
firmware/stm32_ecg/ECG_project.zip
```

Before uploading, verify that:

- STMicroelectronics HAL/CMSIS license files are preserved.
- Build artifacts such as `Debug/`, `Release/`, `*.elf`, `*.bin`, `*.hex`, and `*.map` are excluded if not needed.
- No local user paths or private metadata are included.
- No biosignal data files are included.

If the project is unpacked instead of uploaded as a ZIP, place it under:

```text
firmware/stm32_ecg/ECG_project/
```
