# Configuration Reference

This page documents the runtime configuration dataclasses in `src/ecg_scg_radar_aoac_analysis.py`. Defaults are extracted from the source where available. Some defaults reference top-level constants such as `ECG_PORT`, `SCG_PORT`, or `BASE_DIR`; those constants are intentionally sanitized in this public repository.

> [!NOTE]
> This page describes configuration semantics only. It does not change thresholds, timing windows, SQI calculations, or detector fusion logic.

## Documentation

- [Algorithm Details](algorithm_details.md)
- [Configuration Reference](configuration_reference.md)
- [Code Reference](code_reference.md)
- [Firmware Guide](firmware_guide.md)
- [Output Reference](output_reference.md)
- [Research Notes](research_notes.md)
- [STM32F411 ECG Firmware Configuration](stm32_f411_ecg_firmware.md)


## ECGConfig

`ECGConfig` controls STM32 ECG serial acquisition, ADC CSV parsing, ECG artifact handling, R-peak detection, and ECG pseudo-landmark options.

| Parameter | Default / Example | Description | Notes |
|---|---|---|---|
| port | ECG_PORT | Serial port placeholder or local port used by the acquisition device. | str |
| baudrate | ECG_BAUD | UART baudrate used by the serial acquisition stream. | int |
| input_format | ECG_INPUT_FORMAT | Expected CSV schema or stream type for parsing. | str |
| fs_hint_hz | ECG_FS_HINT_HZ | Sampling-rate hint used when timestamps are reconstructed from sample indices. | float |
| timeout_sec | 0.02 | Serial read timeout used by live collection loops. | float |
| band_hz | (5.0, 35.0) | Band-pass range used for analysis filtering. | tuple[float, float] |
| notch_hz | 60.0 | Notch-filter frequency for line-noise suppression. | Optional[float] |
| use_ecg_artifact_lms | True | Enables ECG LMS artifact suppression using a low-frequency reference. | bool |
| ecg_artifact_ref_band_hz | (0.05, 2.0) | Frequency band used to build ECG artifact reference signal. | tuple[float, float] |
| ecg_lms_mu | 0.0012 | LMS adaptation step size for ECG artifact suppression. | float |
| ecg_lms_order | 16 | LMS filter order for ECG artifact suppression. | int |
| ecg_display_band_hz | (0.7, 18.0) | Band used for display-oriented ECG waveform. | tuple[float, float] |
| ecg_qrs_band_hz | (8.0, 25.0) | Band used for QRS/R-peak detection. | tuple[float, float] |
| ecg_hampel_window_sec | 0.15 | Hampel filter window for ECG outlier suppression. | float |
| ecg_hampel_nsigma | 5.0 | Hampel sigma threshold for ECG artifact handling. | float |
| ecg_baseline_lowpass_hz | 0.7 | Low-pass cutoff used to estimate ECG baseline drift. | float |
| ecg_post_lms_hampel_window_sec | 0.09 | Post-LMS Hampel cleanup window. | float |
| ecg_use_smooth_column_for_display | True | Uses STM32 Smooth_ECG column for display when available. | bool |
| use_ecg_fft_motion_suppression | True | Enables FFT-domain attenuation of low-frequency ECG motion bands. | bool |
| ecg_fft_motion_bands_hz | ((0.05, 0.7), (0.7, 2.5)) | Motion/artifact bands attenuated in FFT domain. | tuple[tuple[float, float], ...] |
| ecg_fft_motion_attenuation | (0.05, 0.35) | Attenuation factors applied to ECG motion bands. | tuple[float, ...] |
| ecg_fft_taper_sec | 0.5 | ECG acquisition, preprocessing, or landmark parameter. | float |
| min_bpm | 45.0 | Minimum physiological heart-rate guard for R-peak detection. | float |
| max_bpm | 150.0 | Maximum physiological heart-rate guard for R-peak detection. | float |
| prominence_scale | 1.25 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| rpeak_min_rr_sec | 0.4 | Minimum RR interval guard for candidate R-peaks. | float |
| rpeak_rr_median_guard | True | Runtime parameter used by the analysis pipeline; see source for exact usage. | bool |
| warmup_discard_sec | 0.5 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| startup_probe_sec | 3.0 | Initial serial probing duration. | float |
| fail_fast_if_no_ecg_sec | 6.0 | Timeout before ECG collection reports no data. | float |
| dtr_enable | True | Whether to enable DTR on the serial port. | bool |
| rts_enable | True | Whether to enable RTS on the serial port. | bool |
| write_start_newline | True | Sends an initial newline to trigger/unstick some UART streams. | bool |
| stm32_csv_signal_col | 1 | Column index used as ECG analysis signal from STM32 CSV. | int |
| stm32_csv_raw_col | 0 | Column index for raw ADC value. | int |
| stm32_csv_smooth_col | 1 | Column index for moving-average Smooth_ECG. | int |
| stm32_adc_bits | 12 | ADC resolution used for metadata and conversion context. | int |
| stm32_vref | 3.3 | ADC reference voltage used for context. | float |
| stm32_csv_has_sample_index | True | Runtime parameter used by the analysis pipeline; see source for exact usage. | bool |
| use_stm32_sample_index_time | True | Runtime parameter used by the analysis pipeline; see source for exact usage. | bool |

## SCGConfig

`SCGConfig` controls ESP32/MPU6050 serial acquisition, SCG axis handling, SCG filtering, and LMS respiration/motion cancellation.

| Parameter | Default / Example | Description | Notes |
|---|---|---|---|
| enabled | SCG_ENABLED | Enables or disables the optional channel. | bool |
| port | SCG_PORT | Serial port placeholder or local port used by the acquisition device. | str |
| baudrate | SCG_BAUD | UART baudrate used by the serial acquisition stream. | int |
| fs_hint_hz | SCG_FS_HINT_HZ | Sampling-rate hint used when timestamps are reconstructed from sample indices. | float |
| timeout_sec | 0.02 | Serial read timeout used by live collection loops. | float |
| fail_fast_if_no_scg_sec | 6.0 | SCG acquisition, preprocessing, or reference extraction parameter. | float |
| use_sample_index_time | True | Runtime parameter used by the analysis pipeline; see source for exact usage. | bool |
| signal_mode | 'vmag' | Runtime parameter used by the analysis pipeline; see source for exact usage. | str |
| band_hz | (0.8, 25.0) | Band-pass range used for analysis filtering. | tuple[float, float] |
| lowpass_display_hz | 20.0 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| hampel_window_sec | 0.12 | Timing or filter window parameter used by beat slicing, detection, or smoothing. | float |
| hampel_nsigma | 5.0 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| use_lms_resp_cancel | True | Enables LMS respiration/motion cancellation. | bool |
| lms_reference_band_hz | (0.08, 0.7) | Reference band used for adaptive respiration/motion cancellation. | tuple[float, float] |
| lms_mu | 0.003 | LMS adaptation step size. | float |
| lms_order | 8 | LMS adaptive filter order. | int |
| serial_header_prefix | '#' | Runtime parameter used by the analysis pipeline; see source for exact usage. | str |

## RadarConfig

`RadarConfig` controls BGT60TR13C FMCW acquisition, chirp/frame settings, range FFT processing, ROI/angle selection, phase displacement extraction, and radar cardiac filtering.

| Parameter | Default / Example | Description | Notes |
|---|---|---|---|
| num_rx | 3 | Runtime parameter used by the analysis pipeline; see source for exact usage. | int |
| num_chirps | 8 | Runtime parameter used by the analysis pipeline; see source for exact usage. | int |
| num_samples | 64 | Runtime parameter used by the analysis pipeline; see source for exact usage. | int |
| frame_rate_hz | 100.0 | Radar frame rate used for cardiac motion sampling. | float |
| chirp_repetition_time_s | 0.0005 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| start_freq_hz | 58000000000.0 | FMCW chirp start frequency. | float |
| end_freq_hz | 63500000000.0 | FMCW chirp end frequency. | float |
| sample_rate_hz | 1000000.0 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| tx_power_level | 31 | Runtime parameter used by the analysis pipeline; see source for exact usage. | int |
| if_gain_dB | 33 | Runtime parameter used by the analysis pipeline; see source for exact usage. | int |
| lp_cutoff_Hz | 500000 | Runtime parameter used by the analysis pipeline; see source for exact usage. | int |
| hp_cutoff_Hz | 80000 | Runtime parameter used by the analysis pipeline; see source for exact usage. | int |
| range_fft_size | 128 | Runtime parameter used by the analysis pipeline; see source for exact usage. | int |
| angle_bins | 61 | Runtime parameter used by the analysis pipeline; see source for exact usage. | int |
| remove_dc | True | Runtime parameter used by the analysis pipeline; see source for exact usage. | bool |
| apply_window | True | Timing or filter window parameter used by beat slicing, detection, or smoothing. | bool |
| min_range_m | 0.4 | Acceptance or guard parameter used by detection, rejection, or quality logic. | float |
| max_range_m | 0.8 | Acceptance or guard parameter used by detection, rejection, or quality logic. | float |
| init_lock_sec | 3.0 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| angle_relock_alpha | 0.02 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| resp_band_hz | (0.1, 0.5) | Frequency band parameter used by filtering or spectral scoring. | tuple[float, float] |
| ppg_like_band_hz | (1.0, 3.0) | Frequency band parameter used by filtering or spectral scoring. | tuple[float, float] |
| use_lms_resp_cancel | True | Enables LMS respiration/motion cancellation. | bool |
| lms_mu | 0.0015 | LMS adaptation step size. | float |
| lms_order | 12 | LMS adaptive filter order. | int |
| lms_reference_band_hz | (0.08, 0.6) | Reference band used for adaptive respiration/motion cancellation. | tuple[float, float] |
| lms_post_band_hz | (1.0, 3.2) | Frequency band parameter used by filtering or spectral scoring. | tuple[float, float] |
| frame_error_sleep_sec | 0.01 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| max_consecutive_frame_errors | 50 | Acceptance or guard parameter used by detection, rejection, or quality logic. | int |
| print_every_frames | 100 | Runtime parameter used by the analysis pipeline; see source for exact usage. | int |

## AnalysisConfig

`AnalysisConfig` controls beat slicing, AO/AC candidate windows, SQI acceptance, morphology detectors, template alignment, AC temporal tracking, and paper export behavior.

| Parameter | Default / Example | Description | Notes |
|---|---|---|---|
| radar_interp_fs_hz | 100.0 | Common interpolation rate for radar beat analysis. | float |
| common_compare_fs_hz | 100.0 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| beat_pre_sec | 0.2 | Seconds retained before each ECG R-peak anchor. | float |
| beat_post_sec | 0.6 | Seconds retained after each ECG R-peak anchor. | float |
| ao_search_sec | (0.07, 0.16) | Runtime parameter used by the analysis pipeline; see source for exact usage. | tuple[float, float] |
| ac_search_sec | (0.25, 0.52) | Runtime parameter used by the analysis pipeline; see source for exact usage. | tuple[float, float] |
| expected_ao_sec | 0.12 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| expected_ac_sec | 0.38 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| compare_start_sec | 3.0 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| compare_end_margin_sec | 3.0 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| max_lag_sec | 2.0 | Acceptance or guard parameter used by detection, rejection, or quality logic. | float |
| psd_nperseg | 512 | Runtime parameter used by the analysis pipeline; see source for exact usage. | int |
| coherence_nperseg | 256 | Runtime parameter used by the analysis pipeline; see source for exact usage. | int |
| min_sqi_accept | 0.35 | Minimum beat SQI threshold for accepted beats. | float |
| aoac_accuracy_tolerance_ms | 30.0 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| use_paper_tight_prior_lock | False | Runtime parameter used by the analysis pipeline; see source for exact usage. | bool |
| tight_target_error_ms | 10.0 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| ao_tight_lock_half_window_sec | 0.045 | Timing or filter window parameter used by beat slicing, detection, or smoothing. | float |
| ac_tight_lock_half_window_sec | 0.055 | Timing or filter window parameter used by beat slicing, detection, or smoothing. | float |
| tight_lock_prior_sigma_sec | 0.01 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| tight_lock_continuity_sigma_sec | 0.025 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| tight_lock_snap_if_outside_target | True | Runtime parameter used by the analysis pipeline; see source for exact usage. | bool |
| tight_lock_min_morph_conf | 0.12 | Acceptance or guard parameter used by detection, rejection, or quality logic. | float |
| use_beat_alignment | True | Runtime parameter used by the analysis pipeline; see source for exact usage. | bool |
| max_beat_align_lag_sec | 0.12 | Acceptance or guard parameter used by detection, rejection, or quality logic. | float |
| dtw_band_sec | 0.12 | Frequency band parameter used by filtering or spectral scoring. | float |
| max_alignment_lag_accept_ms | 120.0 | Acceptance or guard parameter used by detection, rejection, or quality logic. | float |
| use_ac_temporal_tracking | True | Runtime parameter used by the analysis pipeline; see source for exact usage. | bool |
| use_ecg_qrt_prior_for_candidate_detection | False | ECG acquisition, preprocessing, or landmark parameter. | bool |
| ac_tracking_window_sec | 0.06 | Timing or filter window parameter used by beat slicing, detection, or smoothing. | float |
| ac_tracking_prev_weight | 0.35 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| ac_tracking_ecg_weight | 0.35 | ECG acquisition, preprocessing, or landmark parameter. | float |
| ac_tracking_current_weight | 0.3 | Runtime parameter used by the analysis pipeline; see source for exact usage. | float |
| ac_interval_min_sec | 0.14 | Acceptance or guard parameter used by detection, rejection, or quality logic. | float |
| ac_interval_max_sec | 0.5 | Acceptance or guard parameter used by detection, rejection, or quality logic. | float |
| min_template_corr | -0.1 | Template-correlation guard used in beat quality assessment. | float |
| min_amp_std | 0.1 | Acceptance or guard parameter used by detection, rejection, or quality logic. | float |
| max_resp_ratio | 4.0 | Acceptance or guard parameter used by detection, rejection, or quality logic. | float |
| template_iterations | 2 | Number of template refinement iterations. | int |
