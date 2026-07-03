# Output Reference

## Documentation

- [Algorithm Details](algorithm_details.md)
- [Configuration Reference](configuration_reference.md)
- [Code Reference](code_reference.md)
- [Firmware Guide](firmware_guide.md)
- [Output Reference](output_reference.md)
- [Research Notes](research_notes.md)
- [STM32F411 ECG Firmware Configuration](stm32_f411_ecg_firmware.md)


## Result Directory Naming

The script creates run-specific directories under `BASE_DIR`, which is sanitized to `./results` in the public repository. A typical directory name follows this pattern:

```text
results/ex1(YYYY-MM-DD-HH.MM_60s)/
```

## Output Types

| Output Type | Example | Description |
|---|---|---|
| CSV | beat-wise timing table | Beat-level AO/AC candidate timings, SQI flags, and timing differences |
| JSON | summary metrics | Configuration snapshots, validation summaries, and run metadata |
| Figures | paper-ready figures | Signal, timing, morphology, and diagnostic plots |
| Tables | rendered table PNGs | Paper/report table exports from CSV summaries |
| Logs | acquisition logs | Runtime acquisition diagnostics and error notes |

## CSV Result Files

CSV outputs may include beat-wise AO/AC rows, SQI metrics, SCG reference tables, radar candidate tables, CTI interval summaries, and timing-difference summaries. They are intended for downstream statistical analysis and paper/table generation.

Common CSV-style outputs include:

| CSV Category | Typical Contents | Research Use |
|---|---|---|
| Beat-wise timing | Beat index, R-peak anchor, AO/AC candidate times, SQI flags | Inspect per-beat event timing behavior |
| SCG reference | SCG AO/AC fiducial timing relative to ECG anchor | Compare radar candidates with mechanical reference timing |
| Radar candidates | Detector-specific AO/AC candidates and fused timing | Audit which morphology detectors contributed |
| CTI summary | PEP, LVET, QS2-style intervals | Summarize cardiac timing intervals |
| SQI table | Amplitude, bandpower, template correlation, slope energy, contamination proxy | Explain accepted and rejected beats |

## JSON Summary Files

JSON outputs store run configuration, candidate consistency summaries, model validation when enabled, and other structured metadata. They help reproduce an analysis run without committing raw biosignal data.

JSON files are useful for preserving configuration snapshots, summary statistics, model-selection metadata, and run-level warnings. They should be treated as derived research metadata, not as a replacement for full raw data governance.

## `paper_export`

The `paper_export` directory contains compact figures, rendered table PNGs, paper-style CSV summaries, and curated outputs suitable for manual inspection or manuscript preparation.

The export directory is intended to separate manuscript-facing artifacts from exploratory debug output. It may contain compact overview figures, interval tables, rendered CSV tables, and final candidate-timing plots.

## Figure Categories

| Category | Description |
|---|---|
| Signal overview | ECG, SCG, and radar traces around representative beats |
| Beat alignment | Template alignment and lag/DTW diagnostics |
| AO/AC timing | Candidate landmarks, reference comparison, and timing distribution figures |
| SQI diagnostics | Accepted/rejected beat quality visualization |
| Firmware/documentation figures | STM32CubeIDE screenshots and configuration references |

## Raw Biosignal Data Policy

Raw biosignal data can contain sensitive personal information. The repository ignores CSV, JSON, NPY, pickle, MAT, HDF5, image exports, and PDF files by default except selected documentation figures. Public commits should include code and documentation, not private raw recordings.

## Interpreting Timing Difference Metrics

Timing differences should be interpreted as relative differences between SCG reference fiducials and radar morphology-based candidates. They should not be described as absolute aortic valve event errors unless an independent reference modality such as echocardiography, ICG, or PCG is available.

## Reproducibility Checklist

- Record the exact `ECGConfig`, `SCGConfig`, `RadarConfig`, and `AnalysisConfig` values used for the run.
- Keep raw biosignal data private unless consent and anonymization requirements are satisfied.
- Review rejected beats before summarizing timing distributions.
- Confirm that paper-ready figures are derived from accepted and documented processing settings.
- Preserve JSON summaries and configuration snapshots alongside exported tables for internal reproducibility.
