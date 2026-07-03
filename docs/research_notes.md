# Research Notes

## Documentation

- [Algorithm Details](algorithm_details.md)
- [Configuration Reference](configuration_reference.md)
- [Code Reference](code_reference.md)
- [Firmware Guide](firmware_guide.md)
- [Output Reference](output_reference.md)
- [Research Notes](research_notes.md)
- [STM32F411 ECG Firmware Configuration](stm32_f411_ecg_firmware.md)


## What AO and AC Mean

AO denotes aortic valve opening and AC denotes aortic valve closure. In this repository, those labels refer to candidate timing landmarks inferred from signal morphology and reference comparison, not direct imaging of the valve.

The naming follows cardiac mechanics terminology, but the radar signal is a remote chest micro-motion measurement. For that reason, the repository consistently uses terms such as candidate timing, morphology-based landmark, and reference comparison.

## What PEP, LVET, and QS2 Mean

| Interval | Definition | Interpretation |
|---|---|---|
| PEP | `t_AO - t_Q` | Pre-ejection period estimate from Q-like ECG timing to AO timing |
| LVET | `t_AC - t_AO` | Left ventricular ejection time candidate interval |
| QS2 | `t_AC - t_Q` | Electromechanical systolic interval candidate |

## Why ECG R-Peak Is Used as Anchor

The ECG R-peak is a stable electrical beat marker that supports beat-wise alignment across ECG, SCG, and radar streams. It is not treated as AO/AC ground truth.

## Why SCG Is Used as Reference

SCG contains mechanical vibration morphology associated with cardiac events. SCG fiducials provide a practical reference timing for comparison with radar morphology, while still requiring independent validation for absolute valve timing.

## Why Radar AO/AC Is Morphology-Based

FMCW radar measures non-contact chest micro-motion, not valve leaflet movement. Radar AO/AC points are therefore morphology-based candidate events extracted from beat shape, slope, curvature, notch, and template evidence.

This distinction matters for interpretation. A radar candidate can be temporally close to an SCG fiducial while still being a waveform landmark rather than a direct anatomical event.

## Why Radar AO and Radar AC May Behave Differently

AO-like morphology may appear as an early systolic transition, while AC-like morphology may be a later notch, inflection, or downstroke transition. These events can differ in amplitude, timing variability, and susceptibility to respiration/motion contamination.

## Why Independent Validation Is Necessary

Echocardiography, ICG, or PCG can provide more direct or independently validated timing references. Without such modalities, radar AO/AC analysis should be reported as candidate timing and relative comparison, not absolute valve-event validation.

SCG itself is also a derived mechanical signal. It is valuable as a reference in this repository because its fiducial morphology is closer to cardiac mechanical events than ECG R-peaks, but it does not replace independent validation when absolute valve timing is the scientific claim.

## Reporting Guidance

| Phrase to Use | Phrase to Avoid |
|---|---|
| Radar AO/AC candidate timing | Direct radar valve detection |
| Morphology-based landmark | Imaged valve opening/closure |
| SCG reference comparison | Clinical ground truth |
| Relative timing difference | Absolute diagnostic error |
| Research prototype | Medical device |

## Future Work

- Increase subject count and recording conditions.
- Evaluate arrhythmia and irregular rhythm conditions.
- Compare against echocardiography, ICG, or PCG.
- Train learned radar morphology models with independent labels.
- Extend the method toward vehicle/SDV driver monitoring where non-contact radar may be useful.
