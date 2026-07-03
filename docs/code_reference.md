# Code Reference

This page lists the main classes and top-level functions detected in `src/ecg_scg_radar_aoac_analysis.py`. It is intentionally broad so readers can navigate the large research prototype without first reading the entire source file.

## Documentation Navigation

| Document | Description |
|---|---|
| [Algorithm Details](algorithm_details.md) | End-to-end algorithm narrative |
| [Signal Processing Formulas](signal_processing_formulas.md) | Equations used throughout the pipeline |
| [Detector Methods](detector_methods.md) | AO/AC detector ensemble details |
| [Filtering Methods](filtering_methods.md) | Filters and artifact suppression methods |
| [Radar Processing](radar_processing.md) | FMCW radar processing and micro-motion extraction |
| [ECG Processing](ecg_processing.md) | ECG parsing, preprocessing, R-peaks, and Q/T pseudo-landmarks |
| [SCG Processing](scg_processing.md) | MPU6050 SCG preprocessing and reference fiducials |
| [Beat Alignment and CTI](beat_alignment_and_cti.md) | Beat slicing, alignment, timing metrics, and CTI |
| [SQI and Rejection](sqi_and_rejection.md) | Signal quality metrics and beat rejection |
| [Configuration Reference](configuration_reference.md) | Runtime dataclass defaults |
| [Code Reference](code_reference.md) | Extracted class/function map |
| [Firmware Guide](firmware_guide.md) | STM32 and ESP32 firmware notes |
| [Output Reference](output_reference.md) | Result files and paper export structure |
| [References](references.md) | Literature basis and conceptual adaptation notes |


## Classes

| Name | Type | Role | Related doc |
|---|---|---|---|
| ECGConfig | class | Class that encapsulates acquisition, backend, configuration, or model behavior. | [code_reference.md](code_reference.md) |
| SCGConfig | class | Class that encapsulates acquisition, backend, configuration, or model behavior. | [code_reference.md](code_reference.md) |
| RadarConfig | class | Class that encapsulates acquisition, backend, configuration, or model behavior. | [code_reference.md](code_reference.md) |
| AnalysisConfig | class | Class that encapsulates acquisition, backend, configuration, or model behavior. | [code_reference.md](code_reference.md) |
| ECGCollector | class | Class that encapsulates acquisition, backend, configuration, or model behavior. | [code_reference.md](code_reference.md) |
| SCGCollector | class | Class that encapsulates acquisition, backend, configuration, or model behavior. | [code_reference.md](code_reference.md) |
| IfxRadarBackend | class | Class that encapsulates acquisition, backend, configuration, or model behavior. | [code_reference.md](code_reference.md) |
| RadarCollector | class | Class that encapsulates acquisition, backend, configuration, or model behavior. | [code_reference.md](code_reference.md) |
| NumpyRidgeMultiOutput | class | Class that encapsulates acquisition, backend, configuration, or model behavior. | [code_reference.md](code_reference.md) |

## Utility functions

| Name | Type | Role | Related doc |
|---|---|---|---|
| create_result_dir | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| list_serial_ports | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| robust_scale_01 | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| estimate_fs | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| safe_lowpass | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| safe_notch | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| integrate_trapz | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| shift_by_samples | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| refine_event_highres | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| get_center_freq | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| get_lambda | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| steering_vector | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| interpolate_signal | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| mti_first_order_highpass | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| shift_signal_by_samples_fill_edge | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| compare_signals | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| add_morphology_vs_tight_report | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| countdown_for_second_measurement | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| _finite_float | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| _interp_signal_at | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| get_duration_from_cli_or_required_input | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| main | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| _detect_event_generic | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| _interval_metrics | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| _clean_finite | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| _clean_z | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| _clean_slice | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| _clean_event | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| _clean_landmarks | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| _clean_interval | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| _draw_vline_label | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
| _clean_common_data | function | Helper function used by the processing pipeline. | [code_reference.md](code_reference.md) |
## Filtering and signal-processing helpers

| Name | Type | Role | Related doc |
|---|---|---|---|
| zscore_safe | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
| safe_bandpass | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
| bandpower | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
| spectral_corr | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
| normalized_xcorr | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
| mean_coherence | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
| safe_corr | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
| triangular_smooth_envelope | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
| median_smooth_nan | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
| hampel_filter_1d | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
| fft_band_attenuate_zero_phase | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
| lms_adaptive_cancel | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
| zheng_mti_band_component | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
| _clean_smooth | function | Helper function used by the processing pipeline. | [filtering_methods.md](filtering_methods.md) |
## ECG functions

| Name | Type | Role | Related doc |
|---|---|---|---|
| postprocess_rpeaks_short_rr | function | Helper function used by the processing pipeline. | [ecg_processing.md](ecg_processing.md) |
| detect_ecg_q_t_landmarks | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [ecg_processing.md](ecg_processing.md) |
| build_ecg_adaptive_reference_series | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [ecg_processing.md](ecg_processing.md) |
| lms_adaptive_filter_ecg | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [ecg_processing.md](ecg_processing.md) |
| preprocess_stm32_ecg | function | Preprocesses raw signal data before beat-level analysis. | [ecg_processing.md](ecg_processing.md) |
| parse_stm32_ecg_csv_lines | function | Parses incoming serial or text data into numeric arrays. | [ecg_processing.md](ecg_processing.md) |
| looks_like_stm32_csv_text | function | Helper function used by the processing pipeline. | [ecg_processing.md](ecg_processing.md) |
| load_STM32_txt | function | Helper function used by the processing pipeline. | [ecg_processing.md](ecg_processing.md) |
| local_rr_qt_features | function | Helper function used by the processing pipeline. | [ecg_processing.md](ecg_processing.md) |
| synth_ecg_peak_train | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [ecg_processing.md](ecg_processing.md) |
| snap_marker_times_to_visible_ecg_apex | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [ecg_processing.md](ecg_processing.md) |
| _robust_parse_ecg_csv_line | function | Parses incoming serial or text data into numeric arrays. | [ecg_processing.md](ecg_processing.md) |
| _robust_ecg_arrays_from_serial_bytes | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [ecg_processing.md](ecg_processing.md) |
| robust_ecg_serial_diagnostic_from_error_bytes | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [ecg_processing.md](ecg_processing.md) |
## SCG functions

| Name | Type | Role | Related doc |
|---|---|---|---|
| compute_psd | function | Helper function used by the processing pipeline. | [scg_processing.md](scg_processing.md) |
| parse_esp32_mpu6050_scg_csv_lines | function | Parses incoming serial or text data into numeric arrays. | [scg_processing.md](scg_processing.md) |
| preprocess_scg_signal | function | Preprocesses raw signal data before beat-level analysis. | [scg_processing.md](scg_processing.md) |
| _estimate_scg_landmarks | function | Processes SCG data or reference mechanical landmarks. | [scg_processing.md](scg_processing.md) |
| make_empty_scg_result | function | Processes SCG data or reference mechanical landmarks. | [scg_processing.md](scg_processing.md) |
## Radar backend functions

| Name | Type | Role | Related doc |
|---|---|---|---|
| radar_respiration_lms_pipeline | function | Processes radar frames, displacement, or radar-only analysis paths. | [radar_processing.md](radar_processing.md) |
| get_chirp_duration | function | Helper function used by the processing pipeline. | [radar_processing.md](radar_processing.md) |
| get_chirp_slope | function | Helper function used by the processing pipeline. | [radar_processing.md](radar_processing.md) |
| get_range_axis | function | Helper function used by the processing pipeline. | [radar_processing.md](radar_processing.md) |
| get_angle_axis_deg | function | Helper function used by the processing pipeline. | [radar_processing.md](radar_processing.md) |
| preprocess_frame | function | Preprocesses raw signal data before beat-level analysis. | [radar_processing.md](radar_processing.md) |
| range_fft | function | Helper function used by the processing pipeline. | [radar_processing.md](radar_processing.md) |
| dbf_range_angle | function | Helper function used by the processing pipeline. | [radar_processing.md](radar_processing.md) |
| beamformed_complex_at | function | Helper function used by the processing pipeline. | [radar_processing.md](radar_processing.md) |
| add_radar_raw_multicycle_diagnostic | function | Processes radar frames, displacement, or radar-only analysis paths. | [radar_processing.md](radar_processing.md) |
| compute_radar_morphology_visibility | function | Processes radar frames, displacement, or radar-only analysis paths. | [radar_processing.md](radar_processing.md) |
| _estimate_radar_landmarks | function | Processes radar frames, displacement, or radar-only analysis paths. | [radar_processing.md](radar_processing.md) |
| _clean_get_radar_resp_removed | function | Processes radar frames, displacement, or radar-only analysis paths. | [radar_processing.md](radar_processing.md) |
| _clean_get_radar_light | function | Processes radar frames, displacement, or radar-only analysis paths. | [radar_processing.md](radar_processing.md) |
## Beat alignment functions

| Name | Type | Role | Related doc |
|---|---|---|---|
| get_te_delay_for_beat | function | Helper function used by the processing pipeline. | [beat_alignment_and_cti.md](beat_alignment_and_cti.md) |
| template_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [beat_alignment_and_cti.md](beat_alignment_and_cti.md) |
| compute_beat_sqi | function | Computes beat-level signal quality metrics. | [beat_alignment_and_cti.md](beat_alignment_and_cti.md) |
| limited_dtw_distance | function | Aligns beat morphology or estimates beat-to-template lag. | [beat_alignment_and_cti.md](beat_alignment_and_cti.md) |
| estimate_beat_lag_xcorr | function | Aligns beat morphology or estimates beat-to-template lag. | [beat_alignment_and_cti.md](beat_alignment_and_cti.md) |
| align_beats_to_template | function | Aligns beat morphology or estimates beat-to-template lag. | [beat_alignment_and_cti.md](beat_alignment_and_cti.md) |
| build_initial_beats | function | Helper function used by the processing pipeline. | [beat_alignment_and_cti.md](beat_alignment_and_cti.md) |
| make_template_from_beats | function | Helper function used by the processing pipeline. | [beat_alignment_and_cti.md](beat_alignment_and_cti.md) |
| _slice_radar_beat_by_anchor | function | Processes radar frames, displacement, or radar-only analysis paths. | [beat_alignment_and_cti.md](beat_alignment_and_cti.md) |
| _template_corr_features | function | Helper function used by the processing pipeline. | [beat_alignment_and_cti.md](beat_alignment_and_cti.md) |
| _slice_aligned_beat | function | Aligns beat morphology or estimates beat-to-template lag. | [beat_alignment_and_cti.md](beat_alignment_and_cti.md) |
| _clean_rep_beat | function | Helper function used by the processing pipeline. | [beat_alignment_and_cti.md](beat_alignment_and_cti.md) |
| _clean_anchor_from_beat | function | Helper function used by the processing pipeline. | [beat_alignment_and_cti.md](beat_alignment_and_cti.md) |
## AO/AC pipeline functions

| Name | Type | Role | Related doc |
|---|---|---|---|
| enforce_refractory_by_score | function | Helper function used by the processing pipeline. | [detector_methods.md](detector_methods.md) |
| robust_ecg_rpeak_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| scg_inspired_aoac_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| find_adjacent_minima_distance | function | Helper function used by the processing pipeline. | [detector_methods.md](detector_methods.md) |
| ac_fallback_timing_prior_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| ao_fallback_timing_prior_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| ac_inflection_zero_cross_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| radar_event_score_detector_with_ecg_prior | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| zheng_seventh_power_ao_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| morphology_event_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| scg_reference_aoac_pipeline | function | Processes SCG data or reference mechanical landmarks. | [detector_methods.md](detector_methods.md) |
| curvature_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| local_energy_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| derivative_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| notch_tidal_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| wavelet_ridge_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| fuse_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| ecg_estimated_ao_ac_adaptive | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [detector_methods.md](detector_methods.md) |
| ecg_estimated_ao_ac_from_landmarks | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [detector_methods.md](detector_methods.md) |
| accuracy_within_tolerance_ms | function | Helper function used by the processing pipeline. | [detector_methods.md](detector_methods.md) |
| ac_temporal_tracking_refine | function | Helper function used by the processing pipeline. | [detector_methods.md](detector_methods.md) |
| ao_ac_pipeline | function | Helper function used by the processing pipeline. | [detector_methods.md](detector_methods.md) |
| summarize_aoac_timing | function | Helper function used by the processing pipeline. | [detector_methods.md](detector_methods.md) |
| run_acquisition | function | Helper function used by the processing pipeline. | [detector_methods.md](detector_methods.md) |
| _find_candidate_markers | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| scg_reference_aoac_pipeline | function | Processes SCG data or reference mechanical landmarks. | [detector_methods.md](detector_methods.md) |
| _clean_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. | [detector_methods.md](detector_methods.md) |
| _draw_interval_bracket | function | Helper function used by the processing pipeline. | [detector_methods.md](detector_methods.md) |
| scg_reference_aoac_pipeline | function | Processes SCG data or reference mechanical landmarks. | [detector_methods.md](detector_methods.md) |
| scg_reference_aoac_pipeline | function | Processes SCG data or reference mechanical landmarks. | [detector_methods.md](detector_methods.md) |
## Figure generation functions

| Name | Type | Role | Related doc |
|---|---|---|---|
| add_scg_diagnostic_figures | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| safe_pearson_for_fig | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| extract_aoac_arrays_for_plots | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| add_aoac_timing_extra_figures | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| add_combined_overview_figures | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| add_qt_pseudo_landmark_quality_figure | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| add_compact_paper_figures | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| cleanup_legacy_figures | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| add_time_index_alignment_figure | function | Aligns beat morphology or estimates beat-to-template lag. | [output_reference.md](output_reference.md) |
| add_ecg_vs_radar_aoac_correlation_figure | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| add_single_cycle_aoac_label_figure | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| add_ac_temporal_tracking_figure | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| add_beat_alignment_figure | function | Aligns beat morphology or estimates beat-to-template lag. | [output_reference.md](output_reference.md) |
| export_paper_tables_and_figures | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _fig13_rows_from_current_aoac | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _fig13_rows_from_result_dir | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _fig13_find_previous_result_dir | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _fig13_safe_pearson | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| force_add_fig13_previous_vs_current_correlation | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| add_fig4_stage_and_candidate_figures | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _rep_beat_from_aoac_for_joint_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| add_scg_diagnostic_figures | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _clean_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
## Paper export functions

| Name | Type | Role | Related doc |
|---|---|---|---|
| save_csv | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| scg_paper_style_ao_ac_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| paper_tight_event_lock | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _setup_paper_table_font | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _render_csv_table_to_png | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _export_table_pngs_from_existing_csvs | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_scg_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_two_phase_protocol_summary | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
## Two-phase protocol functions

| Name | Type | Role | Related doc |
|---|---|---|---|
| run_radar_only_acquisition | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| extract_candidate_consistency_features | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| build_candidate_consistency_training_dataset | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _model_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| train_candidate_consistency_models | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| build_radar_only_beats | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| predict_radar_only_aoac | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
## Patch/final figure regeneration functions

| Name | Type | Role | Related doc |
|---|---|---|---|
| _paper_safe_float | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _paper_fmt_mean_sd | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _paper_load_json_if_exists | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _paper_metric_block_from_errors | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _paper_copy_if_exists | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _paper_ascii_cell | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _patch_ecg_collector_methods_for_robust_serial | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _patch_choose_scg_branch | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _patch_pick_representative_r_index | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _patch_build_scg_reference_landmarks | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _patch_make_fig01_compact_signal_overview_with_scg | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _patch_make_fig02_scg_reference | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _patch_corr_metrics | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _patch_scatter | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _patch_make_fig04_with_scg | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _patch_make_table09_interval_summary | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _patch_refresh_paper_export_figures | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p3_arr | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p3_z | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p3_branch_scg | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p3_branch_radar | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p3_smooth | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p3_slice | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p3_candidate_score | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p3_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p3_landmarks | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p3_rep_idx | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p3_qt_rel | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p3_interval | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p3_bracket | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p3_vline | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p3_make_fig04_scatter | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p3_make_scg_reference_fig | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p3_make_stage_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p3_make_fig10 | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p3_regenerate_paper_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| _p4_np | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p4_z | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p4_scg_signal | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p4_radar_signal | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p4_smooth | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p4_slice | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p4_score_event | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p4_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p4_validate_sequence | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p4_template_landmarks | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p4_build_scg_ensemble_template | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p4_refine_beat_landmarks_from_template | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p4_interval | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p4_qt_rel | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p4_vline | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p4_bracket | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p4_reference_beat_index | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p4_make_scg_template_reference_fig | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p4_make_stage_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p4_radar_aoac | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p4_make_fig10 | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p4_make_fig04_scatter | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p4_regenerate_paper_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| _p5_np | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p5_z | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p5_sig_scg | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p5_sig_radar | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p5_smooth | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p5_slice | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p5_event_score | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p5_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p5_validate | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p5_template_landmarks | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p5_build_scg_template | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p5_refine_from_template | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p5_qt_rel | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p5_interval | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p5_vline | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p5_bracket | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p5_make_scg_template_fig | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p5_make_scg_periodicity_fig | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p5_stage_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p5_radar_aoac | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p5_make_fig10 | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p5_make_fig04_scatter | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p5_regenerate_paper_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| _p6_get_aoac_ref_for_beat | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p6_representative_cycle_context | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p6_landmark_line | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p6_candidate_markers | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p6_bracket | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p6_plot_panel | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p6_make_single_cycle_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p6_regenerate_paper_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| _p8_enforce_order | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p8_refined_scg_landmarks | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p8_refined_radar_landmarks | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p8_pick_best_scg_beat | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p8_pick_common_representative | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p8_landmark_draw | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p8_window | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p8_bracket | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p8_style_axis | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p8_plot_common_panel | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p8_make_fig02_morphology | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p8_make_fig02_scg_reference | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p8_make_fig06_fig10 | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p8_update_paper_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| _p9_ecg_fig_signal | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p9_scg_fig_signal | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p9_radar_fig_signal | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p9_local_extrema_peak | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p9_slope_curv_event | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p9_scg_landmarks_spike | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p9_radar_landmarks_clean | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p9_ecg_landmarks_clean | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p9_pick_single_cycle | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p9_draw_landmark | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p9_draw_bracket | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p9_plot | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p9_make_all_clean_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p10_bandpass_or_z | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p10_ecg_wave_for_fig | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p10_scg_wave_for_fig | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p10_radar_wave_for_fig | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p10_slice | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p10_local_peak | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p10_slope_event | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p10_scg_landmarks_visible | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p10_radar_landmarks | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p10_ecg_landmarks | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p10_pick_cycle | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p10_label | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p10_bracket | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p10_panel | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p10_make_clean_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p12_z | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p12_ecg_sig | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p12_scg_sig | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p12_radar_sig | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p12_slice | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p12_peak | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p12_slope | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p12_scg_lm | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p12_radar_lm | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p12_ecg_lm | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p12_pick | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p12_txt | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p12_mark | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p12_bracket | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p12_panel | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p12_make | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p13_ecg_qrs_preserved_signal | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p13_core_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| _p14_raw_ecg_array | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p14_ecg_for_display | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p14_scg_for_display | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p14_radar_for_display | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p14_slice_interp | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p14_recenter_ecg_beat | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p14_detect_ecg_qt | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p14_peak | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p14_slope_candidate | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p14_scg_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p14_radar_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p14_pick_best | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p14_mark | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p14_bracket | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p14_panel | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p14_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p14_core_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| _p15_raw_ecg_array | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p15_ecg_for_display | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p15_scg_for_display | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p15_radar_for_display | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p15_slice | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p15_recenter_ecg_beat | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p15_detect_ecg_qrt | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p15_peak | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p15_slope_candidate | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p15_scg_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p15_radar_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p15_pick_best | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p15_mark | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p15_bracket | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p15_panel | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p15_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p15_core_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| _p17_raw_ecg_array | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p17_ecg_for_display | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p17_scg_for_display | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p17_radar_for_display | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p17_slice | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p17_recenter_ecg_beat | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p17_detect_ecg_qrt | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p17_peak | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p17_slope_candidate | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p17_scg_landmarks | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p17_radar_landmarks | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p17_pick_best | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p17_mark | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p17_bracket | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p17_window_label | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p17_panel | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p17_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p17_core_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| _p18_idx | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p18_peak | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p18_slope | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p18_normalize_scg_polarity | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p18_ordered_bounds | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p18_scg_landmarks_signed | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p17_scg_landmarks | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p17_pick_best | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p17_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p19_make_all_beat_scg_radar_table | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p19_make_distribution_figures | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p19_postprocess_literature_guided_reference | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_z | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_raw_ecg_array | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p20_ecg_for_display | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p20_scg_for_display | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p20_radar_for_display | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p20_slice | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_recenter_ecg_beat | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p20_detect_ecg_qrt | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p20_idx | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_peak | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_constrained_pick_sequence | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_signed_score | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_literature_guided_scg_landmarks | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p20_radar_landmarks | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p20_score_representative | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_pick_best | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_all_beat_table | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_distribution_figures | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p20_mark | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_bracket | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_window_label | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_panel | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p20_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p20_core_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| _p21_z | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_raw_ecg_array | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p21_ecg_for_display | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p21_scg_base | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p21_radar_for_display | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p21_slice | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_recenter_ecg_beat | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p21_detect_ecg_qrt | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. | [output_reference.md](output_reference.md) |
| _p21_idx | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_peak | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_mti_interference_cancellation | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_mode_bank | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_waveform_factor | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_reconstruct_ao_signal | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_ao_from_zheng_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p21_signed_fiducials_from_scz | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_measurement_confidence | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_hikaf_track | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_build_scg_reference_table | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p21_radar_landmarks | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p21_pick_best | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_make_scg_radar_table | function | Processes radar frames, displacement, or radar-only analysis paths. | [output_reference.md](output_reference.md) |
| _p21_distribution_figures | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p21_mark | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_bracket | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_window_label | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_panel | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_make_stage_figure | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p21_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p21_core_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. | [output_reference.md](output_reference.md) |
| _p22_require_vmdpy | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p22_successive_vmd_modes | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p22_reconstruct_ao_signal_successive_vmd | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_reconstruct_ao_signal | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p22_make_svmd_mode_figure | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p21_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p23_mti_highpass | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p23_zheng_preprocess | function | Preprocesses raw signal data before beat-level analysis. | [output_reference.md](output_reference.md) |
| _p23_waveform_factor | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p23_fft_freqs | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p23_single_svmd_extract | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p23_svmd_paper_reimplementation | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p23_reconstruct_ao_signal_from_svmd | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_reconstruct_ao_signal | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_ao_from_zheng_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p23_local_maxima | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p23_local_minima | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p23_refine_time_highres | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p23_dirienzo_fiducials | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p23_candidate_peaks | function | Detects or scores morphology-based AO/AC candidate landmarks. | [output_reference.md](output_reference.md) |
| _p23_kmeans_1d | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p23_hikaf_baseline | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p23_hikaf_runtime_track | function | Helper function used by the processing pipeline. | [output_reference.md](output_reference.md) |
| _p21_build_scg_reference_table | function | Processes SCG data or reference mechanical landmarks. | [output_reference.md](output_reference.md) |
| _p23_make_reference_style_figures | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
| _p21_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. | [output_reference.md](output_reference.md) |
