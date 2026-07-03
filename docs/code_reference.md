# Code Reference

This page lists the main classes and top-level functions detected in `src/ecg_scg_radar_aoac_analysis.py`. It is intentionally broad so readers can navigate the large research prototype without first reading the entire source file.

## Documentation

- [Algorithm Details](algorithm_details.md)
- [Configuration Reference](configuration_reference.md)
- [Code Reference](code_reference.md)
- [Firmware Guide](firmware_guide.md)
- [Output Reference](output_reference.md)
- [Research Notes](research_notes.md)
- [STM32F411 ECG Firmware Configuration](stm32_f411_ecg_firmware.md)


## Classes

| Name | Type | Role |
|---|---|---|
| ECGConfig | class | Class that encapsulates acquisition, backend, configuration, or model behavior. |
| SCGConfig | class | Class that encapsulates acquisition, backend, configuration, or model behavior. |
| RadarConfig | class | Class that encapsulates acquisition, backend, configuration, or model behavior. |
| AnalysisConfig | class | Class that encapsulates acquisition, backend, configuration, or model behavior. |
| ECGCollector | class | Class that encapsulates acquisition, backend, configuration, or model behavior. |
| SCGCollector | class | Class that encapsulates acquisition, backend, configuration, or model behavior. |
| IfxRadarBackend | class | Class that encapsulates acquisition, backend, configuration, or model behavior. |
| RadarCollector | class | Class that encapsulates acquisition, backend, configuration, or model behavior. |
| NumpyRidgeMultiOutput | class | Class that encapsulates acquisition, backend, configuration, or model behavior. |

## Utility functions

| Name | Type | Role |
|---|---|---|
| create_result_dir | function | Helper function used by the processing pipeline. |
| list_serial_ports | function | Helper function used by the processing pipeline. |
| robust_scale_01 | function | Helper function used by the processing pipeline. |
| estimate_fs | function | Helper function used by the processing pipeline. |
| safe_lowpass | function | Helper function used by the processing pipeline. |
| safe_notch | function | Helper function used by the processing pipeline. |
| integrate_trapz | function | Helper function used by the processing pipeline. |
| shift_by_samples | function | Helper function used by the processing pipeline. |
| refine_event_highres | function | Helper function used by the processing pipeline. |
| get_center_freq | function | Helper function used by the processing pipeline. |
| get_lambda | function | Helper function used by the processing pipeline. |
| steering_vector | function | Helper function used by the processing pipeline. |
| interpolate_signal | function | Helper function used by the processing pipeline. |
| mti_first_order_highpass | function | Helper function used by the processing pipeline. |
| shift_signal_by_samples_fill_edge | function | Helper function used by the processing pipeline. |
| compare_signals | function | Helper function used by the processing pipeline. |
| add_morphology_vs_tight_report | function | Helper function used by the processing pipeline. |
| countdown_for_second_measurement | function | Helper function used by the processing pipeline. |
| _finite_float | function | Helper function used by the processing pipeline. |
| _interp_signal_at | function | Helper function used by the processing pipeline. |
| get_duration_from_cli_or_required_input | function | Helper function used by the processing pipeline. |
| main | function | Helper function used by the processing pipeline. |
| _detect_event_generic | function | Helper function used by the processing pipeline. |
| _interval_metrics | function | Helper function used by the processing pipeline. |
| _clean_finite | function | Helper function used by the processing pipeline. |
| _clean_z | function | Helper function used by the processing pipeline. |
| _clean_slice | function | Helper function used by the processing pipeline. |
| _clean_event | function | Helper function used by the processing pipeline. |
| _clean_landmarks | function | Helper function used by the processing pipeline. |
| _clean_interval | function | Helper function used by the processing pipeline. |
| _draw_vline_label | function | Helper function used by the processing pipeline. |
| _clean_common_data | function | Helper function used by the processing pipeline. |
## Filtering and signal-processing helpers

| Name | Type | Role |
|---|---|---|
| zscore_safe | function | Helper function used by the processing pipeline. |
| safe_bandpass | function | Helper function used by the processing pipeline. |
| bandpower | function | Helper function used by the processing pipeline. |
| spectral_corr | function | Helper function used by the processing pipeline. |
| normalized_xcorr | function | Helper function used by the processing pipeline. |
| mean_coherence | function | Helper function used by the processing pipeline. |
| safe_corr | function | Helper function used by the processing pipeline. |
| triangular_smooth_envelope | function | Helper function used by the processing pipeline. |
| median_smooth_nan | function | Helper function used by the processing pipeline. |
| hampel_filter_1d | function | Helper function used by the processing pipeline. |
| fft_band_attenuate_zero_phase | function | Helper function used by the processing pipeline. |
| lms_adaptive_cancel | function | Helper function used by the processing pipeline. |
| zheng_mti_band_component | function | Helper function used by the processing pipeline. |
| _clean_smooth | function | Helper function used by the processing pipeline. |
## ECG functions

| Name | Type | Role |
|---|---|---|
| postprocess_rpeaks_short_rr | function | Helper function used by the processing pipeline. |
| detect_ecg_q_t_landmarks | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| build_ecg_adaptive_reference_series | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| lms_adaptive_filter_ecg | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| preprocess_stm32_ecg | function | Preprocesses raw signal data before beat-level analysis. |
| parse_stm32_ecg_csv_lines | function | Parses incoming serial or text data into numeric arrays. |
| looks_like_stm32_csv_text | function | Helper function used by the processing pipeline. |
| load_STM32_txt | function | Helper function used by the processing pipeline. |
| local_rr_qt_features | function | Helper function used by the processing pipeline. |
| synth_ecg_peak_train | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| snap_marker_times_to_visible_ecg_apex | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _robust_parse_ecg_csv_line | function | Parses incoming serial or text data into numeric arrays. |
| _robust_ecg_arrays_from_serial_bytes | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| robust_ecg_serial_diagnostic_from_error_bytes | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
## SCG functions

| Name | Type | Role |
|---|---|---|
| compute_psd | function | Helper function used by the processing pipeline. |
| parse_esp32_mpu6050_scg_csv_lines | function | Parses incoming serial or text data into numeric arrays. |
| preprocess_scg_signal | function | Preprocesses raw signal data before beat-level analysis. |
| _estimate_scg_landmarks | function | Processes SCG data or reference mechanical landmarks. |
| make_empty_scg_result | function | Processes SCG data or reference mechanical landmarks. |
## Radar backend functions

| Name | Type | Role |
|---|---|---|
| radar_respiration_lms_pipeline | function | Processes radar frames, displacement, or radar-only analysis paths. |
| get_chirp_duration | function | Helper function used by the processing pipeline. |
| get_chirp_slope | function | Helper function used by the processing pipeline. |
| get_range_axis | function | Helper function used by the processing pipeline. |
| get_angle_axis_deg | function | Helper function used by the processing pipeline. |
| preprocess_frame | function | Preprocesses raw signal data before beat-level analysis. |
| range_fft | function | Helper function used by the processing pipeline. |
| dbf_range_angle | function | Helper function used by the processing pipeline. |
| beamformed_complex_at | function | Helper function used by the processing pipeline. |
| add_radar_raw_multicycle_diagnostic | function | Processes radar frames, displacement, or radar-only analysis paths. |
| compute_radar_morphology_visibility | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _estimate_radar_landmarks | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _clean_get_radar_resp_removed | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _clean_get_radar_light | function | Processes radar frames, displacement, or radar-only analysis paths. |
## Beat alignment functions

| Name | Type | Role |
|---|---|---|
| get_te_delay_for_beat | function | Helper function used by the processing pipeline. |
| template_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| compute_beat_sqi | function | Computes beat-level signal quality metrics. |
| limited_dtw_distance | function | Aligns beat morphology or estimates beat-to-template lag. |
| estimate_beat_lag_xcorr | function | Aligns beat morphology or estimates beat-to-template lag. |
| align_beats_to_template | function | Aligns beat morphology or estimates beat-to-template lag. |
| build_initial_beats | function | Helper function used by the processing pipeline. |
| make_template_from_beats | function | Helper function used by the processing pipeline. |
| _slice_radar_beat_by_anchor | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _template_corr_features | function | Helper function used by the processing pipeline. |
| _slice_aligned_beat | function | Aligns beat morphology or estimates beat-to-template lag. |
| _clean_rep_beat | function | Helper function used by the processing pipeline. |
| _clean_anchor_from_beat | function | Helper function used by the processing pipeline. |
## AO/AC pipeline functions

| Name | Type | Role |
|---|---|---|
| enforce_refractory_by_score | function | Helper function used by the processing pipeline. |
| robust_ecg_rpeak_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| scg_inspired_aoac_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| find_adjacent_minima_distance | function | Helper function used by the processing pipeline. |
| ac_fallback_timing_prior_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| ao_fallback_timing_prior_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| ac_inflection_zero_cross_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| radar_event_score_detector_with_ecg_prior | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| zheng_seventh_power_ao_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| morphology_event_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| scg_reference_aoac_pipeline | function | Processes SCG data or reference mechanical landmarks. |
| curvature_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| local_energy_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| derivative_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| notch_tidal_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| wavelet_ridge_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| fuse_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| ecg_estimated_ao_ac_adaptive | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| ecg_estimated_ao_ac_from_landmarks | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| accuracy_within_tolerance_ms | function | Helper function used by the processing pipeline. |
| ac_temporal_tracking_refine | function | Helper function used by the processing pipeline. |
| ao_ac_pipeline | function | Helper function used by the processing pipeline. |
| summarize_aoac_timing | function | Helper function used by the processing pipeline. |
| run_acquisition | function | Helper function used by the processing pipeline. |
| _find_candidate_markers | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| scg_reference_aoac_pipeline | function | Processes SCG data or reference mechanical landmarks. |
| _clean_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _draw_interval_bracket | function | Helper function used by the processing pipeline. |
| scg_reference_aoac_pipeline | function | Processes SCG data or reference mechanical landmarks. |
| scg_reference_aoac_pipeline | function | Processes SCG data or reference mechanical landmarks. |
## Figure generation functions

| Name | Type | Role |
|---|---|---|
| add_scg_diagnostic_figures | function | Creates diagnostic or paper-ready visual outputs. |
| safe_pearson_for_fig | function | Creates diagnostic or paper-ready visual outputs. |
| extract_aoac_arrays_for_plots | function | Creates diagnostic or paper-ready visual outputs. |
| add_aoac_timing_extra_figures | function | Creates diagnostic or paper-ready visual outputs. |
| add_combined_overview_figures | function | Creates diagnostic or paper-ready visual outputs. |
| add_qt_pseudo_landmark_quality_figure | function | Creates diagnostic or paper-ready visual outputs. |
| add_compact_paper_figures | function | Creates diagnostic or paper-ready visual outputs. |
| cleanup_legacy_figures | function | Creates diagnostic or paper-ready visual outputs. |
| add_time_index_alignment_figure | function | Aligns beat morphology or estimates beat-to-template lag. |
| add_ecg_vs_radar_aoac_correlation_figure | function | Creates diagnostic or paper-ready visual outputs. |
| add_single_cycle_aoac_label_figure | function | Creates diagnostic or paper-ready visual outputs. |
| add_ac_temporal_tracking_figure | function | Creates diagnostic or paper-ready visual outputs. |
| add_beat_alignment_figure | function | Aligns beat morphology or estimates beat-to-template lag. |
| export_paper_tables_and_figures | function | Creates diagnostic or paper-ready visual outputs. |
| _fig13_rows_from_current_aoac | function | Creates diagnostic or paper-ready visual outputs. |
| _fig13_rows_from_result_dir | function | Creates diagnostic or paper-ready visual outputs. |
| _fig13_find_previous_result_dir | function | Creates diagnostic or paper-ready visual outputs. |
| _fig13_safe_pearson | function | Creates diagnostic or paper-ready visual outputs. |
| force_add_fig13_previous_vs_current_correlation | function | Creates diagnostic or paper-ready visual outputs. |
| add_fig4_stage_and_candidate_figures | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _rep_beat_from_aoac_for_joint_figs | function | Creates diagnostic or paper-ready visual outputs. |
| add_scg_diagnostic_figures | function | Creates diagnostic or paper-ready visual outputs. |
| _clean_figs | function | Creates diagnostic or paper-ready visual outputs. |
## Paper export functions

| Name | Type | Role |
|---|---|---|
| save_csv | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| scg_paper_style_ao_ac_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| paper_tight_event_lock | function | Helper function used by the processing pipeline. |
| _setup_paper_table_font | function | Helper function used by the processing pipeline. |
| _render_csv_table_to_png | function | Helper function used by the processing pipeline. |
| _export_table_pngs_from_existing_csvs | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_scg_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_two_phase_protocol_summary | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| save_all | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
## Two-phase protocol functions

| Name | Type | Role |
|---|---|---|
| run_radar_only_acquisition | function | Processes radar frames, displacement, or radar-only analysis paths. |
| extract_candidate_consistency_features | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| build_candidate_consistency_training_dataset | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _model_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| train_candidate_consistency_models | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| build_radar_only_beats | function | Processes radar frames, displacement, or radar-only analysis paths. |
| predict_radar_only_aoac | function | Processes radar frames, displacement, or radar-only analysis paths. |
## Patch/final figure regeneration functions

| Name | Type | Role |
|---|---|---|
| _paper_safe_float | function | Helper function used by the processing pipeline. |
| _paper_fmt_mean_sd | function | Helper function used by the processing pipeline. |
| _paper_load_json_if_exists | function | Helper function used by the processing pipeline. |
| _paper_metric_block_from_errors | function | Helper function used by the processing pipeline. |
| _paper_copy_if_exists | function | Helper function used by the processing pipeline. |
| _paper_ascii_cell | function | Helper function used by the processing pipeline. |
| _patch_ecg_collector_methods_for_robust_serial | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _patch_choose_scg_branch | function | Processes SCG data or reference mechanical landmarks. |
| _patch_pick_representative_r_index | function | Helper function used by the processing pipeline. |
| _patch_build_scg_reference_landmarks | function | Processes SCG data or reference mechanical landmarks. |
| _patch_make_fig01_compact_signal_overview_with_scg | function | Creates diagnostic or paper-ready visual outputs. |
| _patch_make_fig02_scg_reference | function | Creates diagnostic or paper-ready visual outputs. |
| _patch_corr_metrics | function | Helper function used by the processing pipeline. |
| _patch_scatter | function | Helper function used by the processing pipeline. |
| _patch_make_fig04_with_scg | function | Creates diagnostic or paper-ready visual outputs. |
| _patch_make_table09_interval_summary | function | Helper function used by the processing pipeline. |
| _patch_refresh_paper_export_figures | function | Creates diagnostic or paper-ready visual outputs. |
| _p3_arr | function | Helper function used by the processing pipeline. |
| _p3_z | function | Helper function used by the processing pipeline. |
| _p3_branch_scg | function | Processes SCG data or reference mechanical landmarks. |
| _p3_branch_radar | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p3_smooth | function | Helper function used by the processing pipeline. |
| _p3_slice | function | Helper function used by the processing pipeline. |
| _p3_candidate_score | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p3_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p3_landmarks | function | Helper function used by the processing pipeline. |
| _p3_rep_idx | function | Helper function used by the processing pipeline. |
| _p3_qt_rel | function | Helper function used by the processing pipeline. |
| _p3_interval | function | Helper function used by the processing pipeline. |
| _p3_bracket | function | Helper function used by the processing pipeline. |
| _p3_vline | function | Helper function used by the processing pipeline. |
| _p3_make_fig04_scatter | function | Creates diagnostic or paper-ready visual outputs. |
| _p3_make_scg_reference_fig | function | Creates diagnostic or paper-ready visual outputs. |
| _p3_make_stage_figs | function | Creates diagnostic or paper-ready visual outputs. |
| _p3_make_fig10 | function | Creates diagnostic or paper-ready visual outputs. |
| _p3_regenerate_paper_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| _p4_np | function | Helper function used by the processing pipeline. |
| _p4_z | function | Helper function used by the processing pipeline. |
| _p4_scg_signal | function | Processes SCG data or reference mechanical landmarks. |
| _p4_radar_signal | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p4_smooth | function | Helper function used by the processing pipeline. |
| _p4_slice | function | Helper function used by the processing pipeline. |
| _p4_score_event | function | Helper function used by the processing pipeline. |
| _p4_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p4_validate_sequence | function | Helper function used by the processing pipeline. |
| _p4_template_landmarks | function | Helper function used by the processing pipeline. |
| _p4_build_scg_ensemble_template | function | Processes SCG data or reference mechanical landmarks. |
| _p4_refine_beat_landmarks_from_template | function | Helper function used by the processing pipeline. |
| _p4_interval | function | Helper function used by the processing pipeline. |
| _p4_qt_rel | function | Helper function used by the processing pipeline. |
| _p4_vline | function | Helper function used by the processing pipeline. |
| _p4_bracket | function | Helper function used by the processing pipeline. |
| _p4_reference_beat_index | function | Helper function used by the processing pipeline. |
| _p4_make_scg_template_reference_fig | function | Creates diagnostic or paper-ready visual outputs. |
| _p4_make_stage_figs | function | Creates diagnostic or paper-ready visual outputs. |
| _p4_radar_aoac | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p4_make_fig10 | function | Creates diagnostic or paper-ready visual outputs. |
| _p4_make_fig04_scatter | function | Creates diagnostic or paper-ready visual outputs. |
| _p4_regenerate_paper_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| _p5_np | function | Helper function used by the processing pipeline. |
| _p5_z | function | Helper function used by the processing pipeline. |
| _p5_sig_scg | function | Processes SCG data or reference mechanical landmarks. |
| _p5_sig_radar | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p5_smooth | function | Helper function used by the processing pipeline. |
| _p5_slice | function | Helper function used by the processing pipeline. |
| _p5_event_score | function | Helper function used by the processing pipeline. |
| _p5_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p5_validate | function | Helper function used by the processing pipeline. |
| _p5_template_landmarks | function | Helper function used by the processing pipeline. |
| _p5_build_scg_template | function | Processes SCG data or reference mechanical landmarks. |
| _p5_refine_from_template | function | Helper function used by the processing pipeline. |
| _p5_qt_rel | function | Helper function used by the processing pipeline. |
| _p5_interval | function | Helper function used by the processing pipeline. |
| _p5_vline | function | Helper function used by the processing pipeline. |
| _p5_bracket | function | Helper function used by the processing pipeline. |
| _p5_make_scg_template_fig | function | Creates diagnostic or paper-ready visual outputs. |
| _p5_make_scg_periodicity_fig | function | Creates diagnostic or paper-ready visual outputs. |
| _p5_stage_figs | function | Creates diagnostic or paper-ready visual outputs. |
| _p5_radar_aoac | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p5_make_fig10 | function | Creates diagnostic or paper-ready visual outputs. |
| _p5_make_fig04_scatter | function | Creates diagnostic or paper-ready visual outputs. |
| _p5_regenerate_paper_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| _p6_get_aoac_ref_for_beat | function | Helper function used by the processing pipeline. |
| _p6_representative_cycle_context | function | Helper function used by the processing pipeline. |
| _p6_landmark_line | function | Helper function used by the processing pipeline. |
| _p6_candidate_markers | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p6_bracket | function | Helper function used by the processing pipeline. |
| _p6_plot_panel | function | Creates diagnostic or paper-ready visual outputs. |
| _p6_make_single_cycle_figs | function | Creates diagnostic or paper-ready visual outputs. |
| _p6_regenerate_paper_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| _p8_enforce_order | function | Helper function used by the processing pipeline. |
| _p8_refined_scg_landmarks | function | Processes SCG data or reference mechanical landmarks. |
| _p8_refined_radar_landmarks | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p8_pick_best_scg_beat | function | Processes SCG data or reference mechanical landmarks. |
| _p8_pick_common_representative | function | Helper function used by the processing pipeline. |
| _p8_landmark_draw | function | Helper function used by the processing pipeline. |
| _p8_window | function | Helper function used by the processing pipeline. |
| _p8_bracket | function | Helper function used by the processing pipeline. |
| _p8_style_axis | function | Helper function used by the processing pipeline. |
| _p8_plot_common_panel | function | Creates diagnostic or paper-ready visual outputs. |
| _p8_make_fig02_morphology | function | Creates diagnostic or paper-ready visual outputs. |
| _p8_make_fig02_scg_reference | function | Creates diagnostic or paper-ready visual outputs. |
| _p8_make_fig06_fig10 | function | Creates diagnostic or paper-ready visual outputs. |
| _p8_update_paper_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| _p9_ecg_fig_signal | function | Creates diagnostic or paper-ready visual outputs. |
| _p9_scg_fig_signal | function | Creates diagnostic or paper-ready visual outputs. |
| _p9_radar_fig_signal | function | Creates diagnostic or paper-ready visual outputs. |
| _p9_local_extrema_peak | function | Helper function used by the processing pipeline. |
| _p9_slope_curv_event | function | Helper function used by the processing pipeline. |
| _p9_scg_landmarks_spike | function | Processes SCG data or reference mechanical landmarks. |
| _p9_radar_landmarks_clean | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p9_ecg_landmarks_clean | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p9_pick_single_cycle | function | Helper function used by the processing pipeline. |
| _p9_draw_landmark | function | Helper function used by the processing pipeline. |
| _p9_draw_bracket | function | Helper function used by the processing pipeline. |
| _p9_plot | function | Creates diagnostic or paper-ready visual outputs. |
| _p9_make_all_clean_figs | function | Creates diagnostic or paper-ready visual outputs. |
| _p10_bandpass_or_z | function | Helper function used by the processing pipeline. |
| _p10_ecg_wave_for_fig | function | Creates diagnostic or paper-ready visual outputs. |
| _p10_scg_wave_for_fig | function | Creates diagnostic or paper-ready visual outputs. |
| _p10_radar_wave_for_fig | function | Creates diagnostic or paper-ready visual outputs. |
| _p10_slice | function | Helper function used by the processing pipeline. |
| _p10_local_peak | function | Helper function used by the processing pipeline. |
| _p10_slope_event | function | Helper function used by the processing pipeline. |
| _p10_scg_landmarks_visible | function | Processes SCG data or reference mechanical landmarks. |
| _p10_radar_landmarks | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p10_ecg_landmarks | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p10_pick_cycle | function | Helper function used by the processing pipeline. |
| _p10_label | function | Helper function used by the processing pipeline. |
| _p10_bracket | function | Helper function used by the processing pipeline. |
| _p10_panel | function | Helper function used by the processing pipeline. |
| _p10_make_clean_figs | function | Creates diagnostic or paper-ready visual outputs. |
| _p12_z | function | Helper function used by the processing pipeline. |
| _p12_ecg_sig | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p12_scg_sig | function | Processes SCG data or reference mechanical landmarks. |
| _p12_radar_sig | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p12_slice | function | Helper function used by the processing pipeline. |
| _p12_peak | function | Helper function used by the processing pipeline. |
| _p12_slope | function | Helper function used by the processing pipeline. |
| _p12_scg_lm | function | Processes SCG data or reference mechanical landmarks. |
| _p12_radar_lm | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p12_ecg_lm | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p12_pick | function | Helper function used by the processing pipeline. |
| _p12_txt | function | Helper function used by the processing pipeline. |
| _p12_mark | function | Helper function used by the processing pipeline. |
| _p12_bracket | function | Helper function used by the processing pipeline. |
| _p12_panel | function | Helper function used by the processing pipeline. |
| _p12_make | function | Helper function used by the processing pipeline. |
| _p13_ecg_qrs_preserved_signal | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p13_core_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| _p14_raw_ecg_array | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p14_ecg_for_display | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p14_scg_for_display | function | Processes SCG data or reference mechanical landmarks. |
| _p14_radar_for_display | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p14_slice_interp | function | Helper function used by the processing pipeline. |
| _p14_recenter_ecg_beat | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p14_detect_ecg_qt | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p14_peak | function | Helper function used by the processing pipeline. |
| _p14_slope_candidate | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p14_scg_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p14_radar_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p14_pick_best | function | Helper function used by the processing pipeline. |
| _p14_mark | function | Helper function used by the processing pipeline. |
| _p14_bracket | function | Helper function used by the processing pipeline. |
| _p14_panel | function | Helper function used by the processing pipeline. |
| _p14_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. |
| _p14_core_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| _p15_raw_ecg_array | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p15_ecg_for_display | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p15_scg_for_display | function | Processes SCG data or reference mechanical landmarks. |
| _p15_radar_for_display | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p15_slice | function | Helper function used by the processing pipeline. |
| _p15_recenter_ecg_beat | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p15_detect_ecg_qrt | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p15_peak | function | Helper function used by the processing pipeline. |
| _p15_slope_candidate | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p15_scg_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p15_radar_candidates | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p15_pick_best | function | Helper function used by the processing pipeline. |
| _p15_mark | function | Helper function used by the processing pipeline. |
| _p15_bracket | function | Helper function used by the processing pipeline. |
| _p15_panel | function | Helper function used by the processing pipeline. |
| _p15_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. |
| _p15_core_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| _p17_raw_ecg_array | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p17_ecg_for_display | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p17_scg_for_display | function | Processes SCG data or reference mechanical landmarks. |
| _p17_radar_for_display | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p17_slice | function | Helper function used by the processing pipeline. |
| _p17_recenter_ecg_beat | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p17_detect_ecg_qrt | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p17_peak | function | Helper function used by the processing pipeline. |
| _p17_slope_candidate | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p17_scg_landmarks | function | Processes SCG data or reference mechanical landmarks. |
| _p17_radar_landmarks | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p17_pick_best | function | Helper function used by the processing pipeline. |
| _p17_mark | function | Helper function used by the processing pipeline. |
| _p17_bracket | function | Helper function used by the processing pipeline. |
| _p17_window_label | function | Helper function used by the processing pipeline. |
| _p17_panel | function | Helper function used by the processing pipeline. |
| _p17_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. |
| _p17_core_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| _p18_idx | function | Helper function used by the processing pipeline. |
| _p18_peak | function | Helper function used by the processing pipeline. |
| _p18_slope | function | Helper function used by the processing pipeline. |
| _p18_normalize_scg_polarity | function | Processes SCG data or reference mechanical landmarks. |
| _p18_ordered_bounds | function | Helper function used by the processing pipeline. |
| _p18_scg_landmarks_signed | function | Processes SCG data or reference mechanical landmarks. |
| _p17_scg_landmarks | function | Processes SCG data or reference mechanical landmarks. |
| _p17_pick_best | function | Helper function used by the processing pipeline. |
| _p17_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. |
| _p19_make_all_beat_scg_radar_table | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p19_make_distribution_figures | function | Creates diagnostic or paper-ready visual outputs. |
| _p19_postprocess_literature_guided_reference | function | Helper function used by the processing pipeline. |
| _p20_z | function | Helper function used by the processing pipeline. |
| _p20_raw_ecg_array | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p20_ecg_for_display | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p20_scg_for_display | function | Processes SCG data or reference mechanical landmarks. |
| _p20_radar_for_display | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p20_slice | function | Helper function used by the processing pipeline. |
| _p20_recenter_ecg_beat | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p20_detect_ecg_qrt | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p20_idx | function | Helper function used by the processing pipeline. |
| _p20_peak | function | Helper function used by the processing pipeline. |
| _p20_constrained_pick_sequence | function | Helper function used by the processing pipeline. |
| _p20_signed_score | function | Helper function used by the processing pipeline. |
| _p20_literature_guided_scg_landmarks | function | Processes SCG data or reference mechanical landmarks. |
| _p20_radar_landmarks | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p20_score_representative | function | Helper function used by the processing pipeline. |
| _p20_pick_best | function | Helper function used by the processing pipeline. |
| _p20_all_beat_table | function | Helper function used by the processing pipeline. |
| _p20_distribution_figures | function | Creates diagnostic or paper-ready visual outputs. |
| _p20_mark | function | Helper function used by the processing pipeline. |
| _p20_bracket | function | Helper function used by the processing pipeline. |
| _p20_window_label | function | Helper function used by the processing pipeline. |
| _p20_panel | function | Helper function used by the processing pipeline. |
| _p20_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. |
| _p20_core_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| _p21_z | function | Helper function used by the processing pipeline. |
| _p21_raw_ecg_array | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p21_ecg_for_display | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p21_scg_base | function | Processes SCG data or reference mechanical landmarks. |
| _p21_radar_for_display | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p21_slice | function | Helper function used by the processing pipeline. |
| _p21_recenter_ecg_beat | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p21_detect_ecg_qrt | function | Processes ECG data, R-peaks, or ECG pseudo-landmarks. |
| _p21_idx | function | Helper function used by the processing pipeline. |
| _p21_peak | function | Helper function used by the processing pipeline. |
| _p21_mti_interference_cancellation | function | Helper function used by the processing pipeline. |
| _p21_mode_bank | function | Helper function used by the processing pipeline. |
| _p21_waveform_factor | function | Helper function used by the processing pipeline. |
| _p21_reconstruct_ao_signal | function | Helper function used by the processing pipeline. |
| _p21_ao_from_zheng_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p21_signed_fiducials_from_scz | function | Helper function used by the processing pipeline. |
| _p21_measurement_confidence | function | Helper function used by the processing pipeline. |
| _p21_hikaf_track | function | Helper function used by the processing pipeline. |
| _p21_build_scg_reference_table | function | Processes SCG data or reference mechanical landmarks. |
| _p21_radar_landmarks | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p21_pick_best | function | Helper function used by the processing pipeline. |
| _p21_make_scg_radar_table | function | Processes radar frames, displacement, or radar-only analysis paths. |
| _p21_distribution_figures | function | Creates diagnostic or paper-ready visual outputs. |
| _p21_mark | function | Helper function used by the processing pipeline. |
| _p21_bracket | function | Helper function used by the processing pipeline. |
| _p21_window_label | function | Helper function used by the processing pipeline. |
| _p21_panel | function | Helper function used by the processing pipeline. |
| _p21_make_stage_figure | function | Creates diagnostic or paper-ready visual outputs. |
| _p21_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. |
| _p21_core_export | function | Writes CSV, JSON, table, figure, or paper export artifacts. |
| _p22_require_vmdpy | function | Helper function used by the processing pipeline. |
| _p22_successive_vmd_modes | function | Helper function used by the processing pipeline. |
| _p22_reconstruct_ao_signal_successive_vmd | function | Helper function used by the processing pipeline. |
| _p21_reconstruct_ao_signal | function | Helper function used by the processing pipeline. |
| _p22_make_svmd_mode_figure | function | Creates diagnostic or paper-ready visual outputs. |
| _p21_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. |
| _p23_mti_highpass | function | Helper function used by the processing pipeline. |
| _p23_zheng_preprocess | function | Preprocesses raw signal data before beat-level analysis. |
| _p23_waveform_factor | function | Helper function used by the processing pipeline. |
| _p23_fft_freqs | function | Helper function used by the processing pipeline. |
| _p23_single_svmd_extract | function | Helper function used by the processing pipeline. |
| _p23_svmd_paper_reimplementation | function | Helper function used by the processing pipeline. |
| _p23_reconstruct_ao_signal_from_svmd | function | Helper function used by the processing pipeline. |
| _p21_reconstruct_ao_signal | function | Helper function used by the processing pipeline. |
| _p21_ao_from_zheng_detector | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p23_local_maxima | function | Helper function used by the processing pipeline. |
| _p23_local_minima | function | Helper function used by the processing pipeline. |
| _p23_refine_time_highres | function | Helper function used by the processing pipeline. |
| _p23_dirienzo_fiducials | function | Helper function used by the processing pipeline. |
| _p23_candidate_peaks | function | Detects or scores morphology-based AO/AC candidate landmarks. |
| _p23_kmeans_1d | function | Helper function used by the processing pipeline. |
| _p23_hikaf_baseline | function | Helper function used by the processing pipeline. |
| _p23_hikaf_runtime_track | function | Helper function used by the processing pipeline. |
| _p21_build_scg_reference_table | function | Processes SCG data or reference mechanical landmarks. |
| _p23_make_reference_style_figures | function | Creates diagnostic or paper-ready visual outputs. |
| _p21_make_final_figs | function | Creates diagnostic or paper-ready visual outputs. |
