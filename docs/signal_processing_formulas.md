# Signal Processing Formulas

This page gathers the mathematical notation used to describe the implementation in `src/ecg_scg_radar_aoac_analysis.py`. When a formula is a generalized signal-processing description rather than a literal line-by-line expression, it is labeled as an implementation-level approximation.

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


## ECG and Timing

| Formula | Meaning | Variables | Where Used | Limitation |
|---|---|---|---|---|
| $$RR_i = R_{i+1} - R_i$$ | Beat-to-beat interval | `R_i`: ECG R-peak time | R-peak post-processing, beat windows, Q/T prior | RR depends on robust R-peak detection |
| $$QT_i = T_i - Q_i$$ | ECG pseudo QT interval | `Q_i`, `T_i`: pseudo landmarks | Q/T pseudo-landmark context | Not true diagnostic QT if ECG morphology is nonstandard |
| $$\tau = t - R_i$$ | Beat-relative time | `t`: sample time | Beat slicing for ECG/SCG/radar | Requires synchronized/reconstructed time axes |
| $$C_Q = 0.55C_{amp}+0.30C_{time}+0.15C_{edge}$$ | Q pseudo-landmark confidence | amplitude, timing, edge terms | `detect_ecg_q_t_landmarks` | Code-specific pseudo confidence, not clinical confidence |
| $$C_T = 0.35C_{amp}+0.45C_{time}+0.20C_{prom}$$ | T pseudo-landmark confidence | amplitude, timing, prominence terms | `detect_ecg_q_t_landmarks` | Pseudo landmark only |

## Filtering

| Formula | Meaning | Variables | Where Used | Limitation |
|---|---|---|---|---|
| $$x_{bp}[n] = \mathcal{B}_{f_l,f_h}\{x[n]\}$$ | Generic band-pass filtering | `f_l`, `f_h`: passband edges | ECG, SCG, radar cardiac bands | Filter response depends on sampling rate |
| $$x_{notch}[n] = \mathcal{N}_{f_0,Q}\{x[n]\}$$ | Notch filtering | `f_0`: line frequency, `Q`: quality factor | ECG line-noise suppression | Does not fix broadband motion artifacts |
| $$x_i=\begin{cases}\mathrm{median}(W_i), & |x_i-\mathrm{median}(W_i)|>k\cdot1.4826\cdot\mathrm{MAD}(W_i)\\x_i, & \text{otherwise}\end{cases}$$ | Hampel outlier replacement | `W_i`: local window, `k`: sigma threshold | ECG and SCG artifact suppression | May suppress sharp true events if overused |
| $$\hat d[n]=\mathbf{w}^T[n]\mathbf{u}[n]$$ $$e[n]=x[n]-\hat d[n]$$ $$\mathbf{w}[n+1]=\mathbf{w}[n]+\mu e[n]\mathbf{u}[n]$$ | LMS adaptive cancellation | `u[n]`: artifact reference, `μ`: step size | ECG artifact LMS, SCG/radar respiration cancellation | Depends on reference quality and stable adaptation |
| $$X[k]=\mathcal{F}\{x[n]\},\quad \tilde{X}[k]=a_kX[k],\quad \tilde{x}[n]=\mathcal{F}^{-1}\{\tilde{X}[k]\}$$ | FFT-domain attenuation | `a_k`: attenuation mask | ECG motion-band attenuation | Implementation-level approximation |

## Radar

| Formula | Meaning | Variables | Where Used | Limitation |
|---|---|---|---|---|
| $$R=\frac{cf_b}{2S},\quad S=\frac{B}{T_c}$$ | FMCW beat frequency to range relation | `c`: speed of light, `S`: chirp slope | Range FFT interpretation | Simplified FMCW relation |
| $$\phi(t)=\operatorname{unwrap}(\angle z(t))$$ | Complex phase extraction | `z(t)`: selected complex radar return | ROI phase extraction | Sensitive to ROI and phase noise |
| $$\Delta r(t)=\frac{\lambda}{4\pi}\Delta\phi(t)$$ | Phase-to-displacement relation | `λ`: wavelength | Radar micro-motion extraction | Displacement-like, not direct valve motion |
| $$x_{cardiac}(t)=\mathcal{B}_{f_{heart,low},f_{heart,high}}\{\Delta r(t)\}$$ | Cardiac-band radar signal | cardiac band edges | Radar PPG-like extraction | Band choice affects morphology |
| $$s(t)=d_0+d_r\sin(2\pi f_rt)+d_h\sin(2\pi f_ht)$$ | Qiao-style micro-motion model | respiration and heartbeat terms | Radar interpretation | Conceptual model, not fitted directly in every run |
| $$\tau_k(t)=\frac{2s(t)}{c}$$ | Echo delay model | `s(t)`: displacement | Radar interpretation | Simplified propagation model |

## SCG and Zheng-Inspired AO

| Formula | Meaning | Variables | Where Used | Limitation |
|---|---|---|---|---|
| $$s(t)=s_{SCG}(t)+s_I(t)$$ | SCG plus interference model | `s_I`: interference | SCG artifact discussion | Conceptual decomposition |
| $$s_{SCG}(t)=s_{AO}(t)+s_O(t)$$ | AO component plus other SCG components | `s_AO`: AO-like component | Zheng-inspired AO discussion | Not directly observable |
| $$s_R(t)=\beta s_R(t-1)+(1-\beta)s(t),\quad x(t)=s(t)-s_R(t)$$ | First-order MTI cancellation | `β`: smoothing coefficient | `mti_first_order_highpass`, `_p23_mti_highpass` | Parameter-dependent high-pass behavior |
| $$y(t)=x_{\beta_2}(t)-x_{\beta_1}(t)$$ | Two-MTI band-like output | two beta filters | Zheng-style preprocessing concept | Implementation-level approximation |
| $$p_{AO}(t)=|x_{AO}(t)|^7$$ | Seventh-power AO emphasis | AO-like waveform | `zheng_seventh_power_ao_detector` | Candidate enhancement, not validation |

## AO/AC Reference and Errors

| Formula | Meaning | Variables | Where Used | Limitation |
|---|---|---|---|---|
| $$AO_i^{ref}=R_i+\operatorname{clip}(\lambda_{AO}QT_i,\tau_{AO,min},\tau_{AO,max})$$ | ECG Q/R/T pseudo AO reference | `λ_AO`, clip bounds | ECG prior/reference documentation | Pseudo reference only |
| $$AC_i^{ref}=R_i+\operatorname{clip}(\lambda_{AC}QT_i,\tau_{AC,min},\tau_{AC,max})$$ | ECG Q/R/T pseudo AC reference | `λ_AC`, clip bounds | ECG prior/reference documentation | Pseudo reference only |
| $$e_i^{AO}=(AO_i^{radar}-AO_i^{ref})\times1000$$ | AO pseudo timing error in ms | radar and reference timing | Candidate consistency summaries | Not absolute valve error |
| $$e_i^{AC}=(AC_i^{radar}-AC_i^{ref})\times1000$$ | AC pseudo timing error in ms | radar and reference timing | Candidate consistency summaries | Not absolute valve error |
| $$MAE_{AO}=\frac{1}{N}\sum_i|e_i^{AO}|,\quad MAE_{AC}=\frac{1}{N}\sum_i|e_i^{AC}|$$ | Mean absolute pseudo error | `N`: accepted beats | Summary metrics | Depends on reference definition |
| $$A_\epsilon^{AO/AC}=\frac{1}{N}\sum_i\mathbf{1}(|e_i^{AO}|\le\epsilon \land |e_i^{AC}|\le\epsilon)\times100$$ | Within-tolerance accuracy | `ε`: tolerance | `accuracy_within_tolerance_ms` style reporting | Not clinical accuracy |
| $$\Delta t_i^{AO}=(AO_i^{radar}-AO_i^{SCG})\times1000$$ $$\Delta t_i^{AC}=(AC_i^{radar}-AC_i^{SCG})\times1000$$ | SCG-radar relative timing difference | SCG reference and radar candidate | SCG-radar comparison | Relative comparison only |
| $$PEP_i=t_{AO,i}-t_{Q,i},\quad LVET_i=t_{AC,i}-t_{AO,i},\quad QS2_i=t_{AC,i}-t_{Q,i}$$ | CTI intervals | Q, AO, AC timings | CTI summary/export | Candidate/reference dependent |

## SQI

| Formula | Meaning | Variables | Where Used | Limitation |
|---|---|---|---|---|
| $$\rho_i=\frac{\sum_n(b_i[n]-\bar b_i)(T[n]-\bar T)}{\sqrt{\sum_n(b_i[n]-\bar b_i)^2}\sqrt{\sum_n(T[n]-\bar T)^2}}$$ | Template correlation | beat `b_i`, template `T` | `compute_beat_sqi` | Sensitive to template quality |
| $$r_{cardiac}=\frac{P(f\in[f_{c1},f_{c2}])}{P(f\in[f_{all1},f_{all2}])+\epsilon}$$ | Cardiac bandpower ratio | spectral powers | `compute_beat_sqi` | Band-limited proxy |
| $$E_{slope}=\sum_n\left(\frac{dx[n]}{dn}\right)^2$$ | Slope energy | beat derivative | SQI concept | Code uses mean absolute gradient-like score |
| $$r_{resp}=\frac{P(f\in[f_{r1},f_{r2}])}{P(f\in[f_{c1},f_{c2}])+\epsilon}$$ | Respiration contamination proxy | respiration and cardiac powers | `compute_beat_sqi` | Proxy, not direct motion label |
| $$SQI_i=(0.25q_{amp}+0.25q_{card}+0.25q_{temp}+0.25q_{slope})(1-0.35q_{resp})$$ | Code-level SQI fusion | normalized quality terms | `compute_beat_sqi` | Accepted only with additional guards |
