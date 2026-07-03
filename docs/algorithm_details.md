# Algorithm Details

This page explains the acquisition and signal-processing logic implemented in `src/ecg_scg_radar_aoac_analysis.py`. It is a documentation layer only; it does not modify algorithm thresholds, timing windows, SQI equations, detector scores, or output calculations.

## Documentation

- [Algorithm Details](algorithm_details.md)
- [Configuration Reference](configuration_reference.md)
- [Code Reference](code_reference.md)
- [Firmware Guide](firmware_guide.md)
- [Output Reference](output_reference.md)
- [Research Notes](research_notes.md)
- [STM32F411 ECG Firmware Configuration](stm32_f411_ecg_firmware.md)


## ECG Processing

### STM32 CSV Parsing

The ECG stream is expected from the STM32F411 firmware over USART2. The public serial schema is `sample_index,ADCValue,Smooth_ECG`; the parser also includes robust handling for simpler STM32 CSV rows and diagnostic text. `sample_index` is used with `ECG_FS_HINT_HZ` to reconstruct the ECG time axis when explicit timestamps are not present.

### Hampel Filtering

The ECG preprocessing path includes Hampel filtering to suppress impulsive outliers before R-peak detection. This is useful for contact noise, ADC spikes, or transient motion artifacts that can otherwise dominate slope and energy terms.

### ECG Artifact LMS

`preprocess_stm32_ecg` can build a low-frequency artifact reference and pass it through `lms_adaptive_filter_ecg`. The intent is to reduce baseline and motion-related contamination while preserving a QRS-focused waveform for beat anchoring.

### FFT-Domain Motion Suppression

The ECG path includes optional FFT-domain attenuation of configured low-frequency motion bands. This is a conservative cleanup stage around the ECG anchor signal, not an AO/AC timing estimator.

### Display ECG vs QRS ECG Separation

The code separates display-oriented ECG from QRS-band ECG. The display signal is useful for figures and quality review, while the QRS-band output is used by `robust_ecg_rpeak_detector`.

### Robust R-Peak Detector

`robust_ecg_rpeak_detector` is Pan-Tompkins-like but adds morphology scoring. It combines amplitude, rising/falling slope, local energy, prominence, zero-crossing context, and refractory constraints to identify R-peak anchors.

### Short RR Post-Processing

`postprocess_rpeaks_short_rr` removes likely double detections after the primary detector. This keeps the core detector intact while suppressing physiologically implausible short-RR duplicates.

### Q/T Pseudo-Landmark Detection

`detect_ecg_q_t_landmarks` estimates Q/T-like pseudo-landmarks for quality and interval context. These landmarks are not treated as direct AO/AC ground truth and should not be interpreted as valve event validation.

## SCG Processing

### ESP32 MPU6050 CSV Parsing

The ESP32 firmware emits an 8-column stream: `sample_index,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps`. The parser reconstructs time from either `t_ms` or `sample_index / SCG_FS_HINT_HZ`.

### Axis Selection and Vector Magnitude

`preprocess_scg_signal` prepares SCG-like waveforms from the MPU6050 acceleration axes. Depending on configuration, the analysis can use a selected axis or a magnitude-style representation for chest vibration morphology.

### SCG Band-Pass Filtering

The SCG preprocessing path applies cardiac-band filtering to emphasize mechanical vibration content relevant to AO/AC-like fiducials.

### LMS Respiration/Motion Cancellation

The SCG path can apply LMS-based respiration/motion cancellation using a low-frequency reference band. This supports cleaner mechanical event morphology before fiducial extraction.

### SCG AO/AC Reference Extraction

`scg_reference_aoac_pipeline`, `_p21_build_scg_reference_table`, `_p23_dirienzo_fiducials`, and related patch functions create beat-wise SCG reference timing rows. The reference is useful for comparison but is still not equivalent to echocardiography.

### SCG Diagnostic Figures

The script includes SCG diagnostic figure functions that show representative beats, template behavior, fiducial positions, and comparison views with radar morphology.

## Radar Processing

### FMCW Radar Configuration

The radar path uses Infineon `DeviceFmcw` with BGT60TR13C-style FMCW configuration values from `RadarConfig`. Chirp/frame settings, frequency limits, sample counts, and receiver/channel choices are controlled by that dataclass.

### Frame Acquisition

`IfxRadarBackend` and `RadarCollector` acquire radar frames, coordinate thread timing with the ECG/SCG collectors, and store frame-time information for downstream interpolation and beat slicing.

### Range FFT

`range_fft` transforms raw radar samples into a range-domain representation. This enables ROI selection around the chest micro-motion range bin.

### Angle/ROI Selection

`dbf_range_angle` and `beamformed_complex_at` support range-angle or beamformed complex sample selection. These functions are used to isolate the radar return most relevant to the subject's chest motion.

### Phase Extraction

The selected complex radar return is converted into a displacement-like phase trace. Phase unwrapping and displacement conversion provide the basis for non-contact cardiac motion analysis.

### Respiration Cancellation

`radar_respiration_lms_pipeline` suppresses low-frequency respiration and body motion components with adaptive cancellation and post filtering.

### Cardiac Band Extraction

The radar path filters the displacement-like signal into a cardiac motion band so that beat-wise morphology can be compared to ECG and SCG timing anchors.

### Interpolation

`interpolate_signal` resamples radar beat segments to a common rate such as 100 Hz. This simplifies template construction, beat alignment, and detector fusion.

## Beat Alignment

### ECG R-Peak Anchor

ECG R-peaks define beat boundaries for ECG, SCG, and radar. ECG is used as the time anchor only, not as direct AO/AC ground truth.

### Beat Windows

`beat_pre_sec` and `beat_post_sec` define how much signal is retained around each R-peak. AO/AC detectors then operate inside narrower event windows relative to the beat anchor.

### Template Construction

`build_initial_beats`, `make_template_from_beats`, and template refinement logic build representative morphology from accepted beats.

### Cross-Correlation and DTW Alignment

`estimate_beat_lag_xcorr`, `align_beats_to_template`, and `limited_dtw_distance` support beat-level alignment correction. Lag acceptance guards prevent large shifts from forcing poor morphology into the template.

### Rejected Beats

Beats may be rejected when signal quality, lag, morphology, or detector confidence is insufficient. Rejection is necessary because radar and wearable-like SCG signals are vulnerable to motion and respiration contamination.

## AO/AC Candidate Detection

### AO and AC Search Windows

AO and AC candidate windows are stored in `AnalysisConfig`. These windows constrain candidate search to physiologically plausible regions relative to the ECG R-peak anchor.

### Slope Detector

`derivative_detector` scores local derivative behavior. AO tends to emphasize early systolic rising or rapid transition morphology, while AC often emphasizes later falling or transition behavior.

### Notch/Tidal Detector

`notch_tidal_detector` searches for notch-like morphology often associated with AC-style events. This detector is useful when AC is more of an inflection/notch than a clean peak.

### Wavelet Detector

`wavelet_ridge_detector` uses CWT/morlet-style ridge evidence when SciPy wavelet support is available, with fallback behavior otherwise. It is intended to capture localized transient energy.

### Template Detector

`template_detector` compares beat morphology against an ensemble/template context. It provides a morphology-consistency signal rather than a standalone physiological proof.

### Morphology Detector

`morphology_event_detector` combines slope, zero-crossing, local extrema, curvature, and timing prior terms to produce AO/AC candidate scores.

### SCG-Inspired Detector

`scg_inspired_aoac_detector` adapts SCG fiducial principles to radar beat morphology, while preserving the distinction between SCG reference timing and radar candidate timing.

### Zheng Seventh-Power AO Detector

`zheng_seventh_power_ao_detector` applies a seventh-power/envelope style AO emphasis inspired by SCG literature. It is used as an AO candidate source and should not be read as direct valve imaging.

### Paper-Tight Prior Lock

`paper_tight_event_lock` supports constrained refinement/audit views used in paper-style export. Documentation should treat this as an analysis/export path, not as an independent validation source.

### Fusion Strategy

`fuse_candidates` combines detector outputs using median/confidence style aggregation. Fusion is designed to reduce dependence on any single detector when morphology is ambiguous.

## SQI

`compute_beat_sqi` calculates beat quality metrics from amplitude stability, cardiac bandpower ratio, template correlation, slope energy, and respiration/motion contamination proxy terms. `min_sqi_accept` controls accepted vs rejected beat behavior. SQI is essential because non-contact radar and body-mounted SCG can degrade under motion, posture change, or weak return conditions.

## CTI

Cardiac timing intervals are computed from candidate or reference event timings:

$$
PEP = t_{AO} - t_Q
$$

$$
LVET = t_{AC} - t_{AO}
$$

$$
QS2 = t_{AC} - t_Q
$$

## Validation Limitation

SCG reference timing is useful for relative comparison, but it is not the same as echocardiography, ICG, or PCG validation. Radar AO/AC landmarks in this repository are morphology-based candidate timings and should not be described as direct aortic valve opening or closure imaging.
