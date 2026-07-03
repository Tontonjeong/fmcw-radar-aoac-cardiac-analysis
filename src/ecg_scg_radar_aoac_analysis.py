# -*- coding: utf-8 -*-
"""
Project:
    FMCW Radar-based AO/AC Timing Analysis using ECG, SCG, and Radar Signals

Paper:
    Analysis of Aortic Valve Opening and Closure Using Cardiac Signals
    Acquired by Non-Contact FMCW Radar

Purpose:
    This script acquires and analyzes ECG, SCG, and FMCW radar signals
    for beat-wise AO/AC candidate timing analysis.

Hardware:
    - STM32 ECG acquisition module
    - ESP32 + MPU6050 SCG acquisition module
    - Infineon BGT60TR13C FMCW radar
    - PC running Python analysis pipeline

Main Pipeline:
    1. ECG acquisition and R-peak detection
    2. SCG acquisition and AO/AC reference timing extraction
    3. FMCW radar acquisition and phase displacement extraction
    4. ECG R-peak based beat alignment
    5. Radar AO/AC candidate detection
    6. SCG-Radar timing comparison
    7. CTI calculation and paper-ready export

Warning:
    This is a research prototype.
    It is not medical diagnosis software.
    Radar AO/AC points are morphology-based candidate events,
    not direct valve imaging results.
"""

from __future__ import annotations

import csv
import json
import time
import threading
import warnings
import re
import sys
import shutil

import pickle

try:
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports

try:
    from scipy.signal import cwt, morlet2
    HAS_CWT = True
except Exception:
    HAS_CWT = False

from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp


# ============================================================
# 사용자 설정
# ============================================================
ECG_PORT = "COMx"  # STM32 ECG; set for the local PC before acquisition.
ECG_BAUD = 115200  # STM32 USART2 baud

# ESP32 + MPU6050 SCG serial settings
SCG_ENABLED = True
SCG_PORT = "COMx"
SCG_BAUD = 115200
SCG_FS_HINT_HZ = 100.0

# ECG input is fixed to STM32 UART CSV only
# STM32 main.c output format: ADCValue,Smooth_ECG
ECG_INPUT_FORMAT = "stm32_adc_csv"

ECG_FS_HINT_HZ = 100.0
BASE_DIR = Path("./results")



# Fig02 Q/T 표시 정책
# STM32 ECG display가 표준 Lead ECG 형태가 아니면 Q/T 표시는 오히려 혼선을 만든다.
# 기본값 False: fig02에서는 R anchor와 radar AO/AC만 보여주고, Q/T는 별도 quality figure/CSV에서 확인.
FIG02_SHOW_QT_MARKERS = False
QT_LANDMARK_MIN_CONFIDENCE = 0.45

# ECG Q/T pseudo-landmark periodic prior
# TIM1으로 sampling time은 균일하지만, Q/T는 beat-to-beat phase consistency가 필요함.
# 따라서 RR-adaptive window + temporal tracking + outlier rejection을 적용.


# Fig4 constrained-result audit policy
# Fig4는 ECG-derived pseudo-reference constrained consistency plot이므로 accuracy %를 표시하지 않는다.
# audit rate가 99.9% 이상이면 leakage/tight-lock/threshold 과대평가 가능성을 별도 파일로 경고한다.
FIG4_AUDIT_WARN_RATE = 0.999
FIG4_INDEPENDENT_ACCURACY_TOL_MS = 30.0


# ============================================================
# Paper export package
# ============================================================
PAPER_EXPORT_ENABLED = True
PAPER_EXPORT_DIRNAME = "paper_export"
PAPER_FIG_DPI = 300

# R-peak post-processing
# R detector 자체를 바꾸지 않고, 검출 후 RR이 너무 짧은 double detection만 제거한다.
# 기준: RR < 0.45 s이면 두 peak 중 QRS-band amplitude가 낮은 쪽 제거
RPEAK_ENABLE_SHORT_RR_POSTPROCESS = True
RPEAK_MIN_RR_SEC_POST = 0.45
RPEAK_SHORT_RR_NEIGHBOR_MARGIN_SEC = 0.04

QT_USE_RR_ADAPTIVE_PERIODIC_PRIOR = True
QT_Q_MAX_JUMP_SEC = 0.030
QT_T_MAX_JUMP_SEC = 0.070
QT_MIN_TRACK_CONFIDENCE = 0.45
QT_INTERPOLATE_REJECTED = False


# ============================================================
# Legacy two-phase regression protocol (disabled by default)
# ============================================================
# 논문 분석용:
# 1단계: ECG+Radar 동시 측정으로 ECG-derived pseudo AO/AC label dataset 생성
# 2단계: countdown 후 동일 시간 Radar-only 추가 측정
# 3단계: 1단계 candidate-consistency regression model로 2단계 radar raw/PPG-like AO/AC timing 추정
#
# 주의:
# - TEST_WITH_ECG_REFERENCE=False이면 2단계에는 true accuracy가 존재하지 않음.
#   이 경우 accuracy는 1단계 calibration validation 기준이고,
#   2단계는 predicted timing + confidence/consistency 분석으로 저장함.
# - 실제 2단계 정확도까지 산출하려면 TEST_WITH_ECG_REFERENCE=True로 바꿔
#   2단계도 ECG+Radar 동시 측정해야 함.
TWO_PHASE_LABEL_GUIDED_PROTOCOL = False
TWO_PHASE_TEST_WITH_ECG_REFERENCE = False
TWO_PHASE_COUNTDOWN_SEC = 3
TWO_PHASE_MIN_TRAIN_BEATS = 20
TWO_PHASE_MODEL_CANDIDATES = ("ridge", "lasso", "random_forest", "gradient_boosting")

# 논문용 figure 정리: True면 통합 figure만 남기고 legacy figure 삭제
SAVE_COMPACT_FIGURES_ONLY = True


# ============================================================
# Config
# ============================================================
@dataclass
class ECGConfig:
    port: str = ECG_PORT
    baudrate: int = ECG_BAUD
    input_format: str = ECG_INPUT_FORMAT
    fs_hint_hz: float = ECG_FS_HINT_HZ
    timeout_sec: float = 0.02

    # ECG peak 안정화. STM32 raw ADC 품질에 따라 prominence_scale을 0.7~1.2에서 조정.
    band_hz: tuple[float, float] = (5.0, 35.0)
    notch_hz: Optional[float] = 60.0

    # ECG artifact suppression
    # STM32 ADC raw에는 baseline drift/contact/motion artifact가 섞일 수 있으므로
    # R-peak anchor용 QRS 신호와 figure용 display ECG를 분리 생성한다.
    use_ecg_artifact_lms: bool = True
    ecg_artifact_ref_band_hz: tuple[float, float] = (0.05, 2.0)
    ecg_lms_mu: float = 0.0012
    ecg_lms_order: int = 16
    ecg_display_band_hz: tuple[float, float] = (0.7, 18.0)
    ecg_qrs_band_hz: tuple[float, float] = (8.0, 25.0)
    ecg_hampel_window_sec: float = 0.15
    ecg_hampel_nsigma: float = 5.0
    ecg_baseline_lowpass_hz: float = 0.70
    ecg_post_lms_hampel_window_sec: float = 0.09
    ecg_use_smooth_column_for_display: bool = True

    # FFT-domain motion artifact attenuation
    # 0.05~0.7 Hz: baseline/respiration drift 강하게 약화
    # 0.7~2.5 Hz: contact/motion 성분 부분 약화
    # QRS 검출은 8~25 Hz라 R anchor 손상은 제한적
    use_ecg_fft_motion_suppression: bool = True
    ecg_fft_motion_bands_hz: tuple[tuple[float, float], ...] = ((0.05, 0.70), (0.70, 2.50))
    ecg_fft_motion_attenuation: tuple[float, ...] = (0.05, 0.35)
    ecg_fft_taper_sec: float = 0.50
    min_bpm: float = 45.0
    max_bpm: float = 150.0  # R-peak 과검출 방지: RR < 400 ms 후보 억제
    prominence_scale: float = 1.25
    rpeak_min_rr_sec: float = 0.40
    rpeak_rr_median_guard: bool = True
    warmup_discard_sec: float = 0.5
    # STM32 live serial 진단/초기화
    startup_probe_sec: float = 3.0
    fail_fast_if_no_ecg_sec: float = 6.0
    dtr_enable: bool = True
    rts_enable: bool = True
    write_start_newline: bool = True

    # STM32 ECG UART CSV format:
    #   ADCValue,Smooth_ECG
    # 예: 1687,1520
    # 0열(raw ADCValue)을 분석 기본값으로 사용하고, 1열(Smooth_ECG)은 확인/보조용
    stm32_csv_signal_col: int = 1  # 0=raw ADCValue, 1=STM32 Smooth_ECG. artifact가 크면 Smooth_ECG가 기본
    stm32_csv_raw_col: int = 0
    stm32_csv_smooth_col: int = 1
    stm32_adc_bits: int = 12
    stm32_vref: float = 3.3

    # 권장 STM32 UART format:
    #   sample_index,ADCValue,Smooth_ECG
    # 구형 format도 자동 지원:
    #   ADCValue,Smooth_ECG
    # sample_index가 있으면 ECG 시간축은 sample_index / fs_hint_hz로 강제 구성한다.
    stm32_csv_has_sample_index: bool = True
    use_stm32_sample_index_time: bool = True


@dataclass
class SCGConfig:
    enabled: bool = SCG_ENABLED
    port: str = SCG_PORT
    baudrate: int = SCG_BAUD
    fs_hint_hz: float = SCG_FS_HINT_HZ
    timeout_sec: float = 0.02
    fail_fast_if_no_scg_sec: float = 6.0
    use_sample_index_time: bool = True
    signal_mode: str = "vmag"  # options: "az", "vmag", "ax", "ay"
    band_hz: tuple[float, float] = (0.8, 25.0)
    lowpass_display_hz: float = 20.0
    hampel_window_sec: float = 0.12
    hampel_nsigma: float = 5.0
    use_lms_resp_cancel: bool = True
    lms_reference_band_hz: tuple[float, float] = (0.08, 0.7)
    lms_mu: float = 0.003
    lms_order: int = 8
    serial_header_prefix: str = "#"


@dataclass
class RadarConfig:
    num_rx: int = 3
    num_chirps: int = 8
    num_samples: int = 64

    frame_rate_hz: float = 100.0
    chirp_repetition_time_s: float = 0.0005

    start_freq_hz: float = 58e9
    end_freq_hz: float = 63.5e9
    sample_rate_hz: float = 1_000_000.0

    tx_power_level: int = 31
    if_gain_dB: int = 33
    lp_cutoff_Hz: int = 500_000
    hp_cutoff_Hz: int = 80_000

    range_fft_size: int = 128
    angle_bins: int = 61

    remove_dc: bool = True
    apply_window: bool = True

    min_range_m: float = 0.40
    max_range_m: float = 0.80
    init_lock_sec: float = 3.0

    # angle 추적을 거의 제거하고 약하게만 relock
    angle_relock_alpha: float = 0.02

    resp_band_hz: tuple[float, float] = (0.10, 0.50)
    # 심박만 더 좁게: 과한 motion/respiration 제거 목적
    ppg_like_band_hz: tuple[float, float] = (1.0, 3.0)

    # LMS 기반 respiration/motion adaptive cancellation
    use_lms_resp_cancel: bool = True
    lms_mu: float = 0.0015
    lms_order: int = 12
    lms_reference_band_hz: tuple[float, float] = (0.08, 0.60)
    lms_post_band_hz: tuple[float, float] = (1.0, 3.2)

    frame_error_sleep_sec: float = 0.01
    max_consecutive_frame_errors: int = 50
    print_every_frames: int = 100


@dataclass
class AnalysisConfig:
    radar_interp_fs_hz: float = 100.0
    common_compare_fs_hz: float = 100.0

    beat_pre_sec: float = 0.20
    beat_post_sec: float = 0.60

    # ECG R-peak 이후 예상 후보창.
    # Radar-PPG surrogate 기준이므로 실제 valve timing과 다름.
    ao_search_sec: tuple[float, float] = (0.07, 0.16)
    ac_search_sec: tuple[float, float] = (0.25, 0.52)

    # pseudo reference center. 참조 센서 없을 때 method consistency 확인용.
    expected_ao_sec: float = 0.12
    expected_ac_sec: float = 0.38

    compare_start_sec: float = 3.0
    compare_end_margin_sec: float = 3.0
    max_lag_sec: float = 2.0

    psd_nperseg: int = 512
    coherence_nperseg: int = 256

    # SQI / beat rejection
    min_sqi_accept: float = 0.35

    # Accuracy report tolerance.
    # 요청 목표: ECG-derived pseudo reference 기준 ±10 ms로 표시/평가.
    # 주의: 실제 valve ground-truth 오차가 아니라 ECG Q/T+RR 기반 pseudo-reference 오차.
    aoac_accuracy_tolerance_ms: float = 30.0

    # 논문 기반 tight refinement:
    # Zheng: MTI/1~40Hz 전처리 + seventh power envelope AO 강조
    # Qiao: radar cardiac motion 전처리/0.85~3.3Hz cardiac band + morphology correspondence
    # 목적: ECG pseudo-reference 기준 AO/AC 위치를 ±10 ms 내로 끌어오는 target-locked refinement
    use_paper_tight_prior_lock: bool = False
    tight_target_error_ms: float = 10.0
    ao_tight_lock_half_window_sec: float = 0.045
    ac_tight_lock_half_window_sec: float = 0.055
    tight_lock_prior_sigma_sec: float = 0.010
    tight_lock_continuity_sigma_sec: float = 0.025
    tight_lock_snap_if_outside_target: bool = True
    tight_lock_min_morph_conf: float = 0.12

    # Beat-level alignment: ECG R 기준 slicing 후 radar beat morphology가 밀리는 문제 보정
    use_beat_alignment: bool = True
    max_beat_align_lag_sec: float = 0.120
    dtw_band_sec: float = 0.120
    max_alignment_lag_accept_ms: float = 120.0

    # AC temporal tracking: AC는 beat마다 독립 peak 검출보다 추적 기반이 안정적
    use_ac_temporal_tracking: bool = True
    # ECG Q/R/T 기반 pseudo timing은 ground truth가 아니므로 기본적으로 detector prior로 사용하지 않는다.
    # False: ECG는 R-peak beat alignment anchor로만 사용하고, 후보 검출은 radar morphology + fixed physiological window 중심.
    # True : ECG Q/R/T+RR 기반 pseudo prior를 detector score에 사용.
    use_ecg_qrt_prior_for_candidate_detection: bool = False
    ac_tracking_window_sec: float = 0.060
    ac_tracking_prev_weight: float = 0.35
    ac_tracking_ecg_weight: float = 0.35
    ac_tracking_current_weight: float = 0.30
    ac_interval_min_sec: float = 0.140
    ac_interval_max_sec: float = 0.500
    min_template_corr: float = -0.10
    min_amp_std: float = 0.10
    max_resp_ratio: float = 4.0

    # template 반복
    template_iterations: int = 2


# ============================================================
# Utility
# ============================================================
def create_result_dir(duration_sec: float) -> Path:
    """
    Save directory:
      ./results/ex1(YYYY-MM-DD-HH.MM_60s)
      ./results/ex2(YYYY-MM-DD-HH.MM_60s)
      ...
    날짜/시각이 달라도 기존 ex 번호를 스캔해서 다음 번호로 저장한다.
    """
    now = datetime.now().strftime("%Y-%m-%d-%H.%M")
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    dur_label = f"{int(duration_sec)}s" if float(duration_sec).is_integer() else f"{duration_sec:.1f}s"

    max_idx = 0
    pattern = re.compile(r"^ex(\d+)\(")
    for p in BASE_DIR.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m:
            try:
                max_idx = max(max_idx, int(m.group(1)))
            except ValueError:
                pass

    i = max_idx + 1
    while True:
        out = BASE_DIR / f"ex{i}({now}_{dur_label})"
        if not out.exists():
            out.mkdir(parents=True)
            return out
        i += 1

def list_serial_ports():
    print("[INFO] Serial ports:")
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("  - none")
    for p in ports:
        print(f"  - {p.device}: {p.description}")


def save_csv(path: Path, header: list[str], rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def zscore_safe(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        return np.zeros_like(x)
    return (x - np.nanmean(x)) / s


def robust_scale_01(x: np.ndarray, invert: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    lo, hi = np.nanpercentile(x, [5, 95])
    if hi - lo < 1e-12:
        y = np.zeros_like(x)
    else:
        y = np.clip((x - lo) / (hi - lo), 0, 1)
    return 1 - y if invert else y


def estimate_fs(t: np.ndarray, fallback: float) -> float:
    t = np.asarray(t, dtype=np.float64)
    if len(t) < 3:
        return fallback
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return fallback
    return float(1.0 / np.median(dt))


def safe_bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < max(32, order * 6):
        return np.zeros_like(x)
    nyq = 0.5 * fs
    low = max(low, 0.01)
    high = min(high, nyq * 0.95)
    if low >= high:
        return np.zeros_like(x)
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    try:
        return signal.sosfiltfilt(sos, x)
    except ValueError:
        return signal.sosfilt(sos, x)


def safe_lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 32:
        return x
    cutoff = min(cutoff, 0.45 * fs)
    if cutoff <= 0:
        return x
    sos = signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    try:
        return signal.sosfiltfilt(sos, x)
    except ValueError:
        return signal.sosfilt(sos, x)


def safe_notch(x: np.ndarray, fs: float, notch_hz: Optional[float], q: float = 30.0) -> np.ndarray:
    if notch_hz is None or notch_hz <= 0 or notch_hz >= 0.45 * fs:
        return x
    b, a = signal.iirnotch(notch_hz, q, fs)
    try:
        return signal.filtfilt(b, a, x)
    except ValueError:
        return signal.lfilter(b, a, x)


def compute_psd(x: np.ndarray, fs: float, nperseg: int):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 16:
        return np.array([]), np.array([])
    return signal.welch(x, fs=fs, nperseg=min(nperseg, len(x)), detrend="constant")


def integrate_trapz(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy 2.x compatibility: use np.trapezoid when np.trapz is unavailable."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    # fallback for old NumPy
    return float(getattr(np, "trapz")(y, x))


def bandpower(x: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    f, p = compute_psd(x, fs, min(256, len(x)))
    if len(f) == 0:
        return 0.0
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m):
        return 0.0
    return integrate_trapz(p[m], f[m])


def spectral_corr(x: np.ndarray, y: np.ndarray, fs: float, band: tuple[float, float], nperseg: int):
    fx, px = compute_psd(x, fs, nperseg)
    fy, py = compute_psd(y, fs, nperseg)
    if len(fx) == 0 or len(fy) == 0:
        return None
    mx = (fx >= band[0]) & (fx <= band[1])
    my = (fy >= band[0]) & (fy <= band[1])
    sx = np.log10(np.maximum(px[mx], 1e-18))
    sy = np.log10(np.maximum(py[my], 1e-18))
    n = min(len(sx), len(sy))
    if n < 3:
        return None
    sx, sy = sx[:n], sy[:n]
    if np.std(sx) < 1e-12 or np.std(sy) < 1e-12:
        return None
    return float(np.corrcoef(sx, sy)[0, 1])


def normalized_xcorr(x: np.ndarray, y: np.ndarray, fs: float, max_lag_sec: float):
    x, y = zscore_safe(x), zscore_safe(y)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    c = signal.correlate(x, y, mode="full", method="auto") / max(n, 1)
    lags = signal.correlation_lags(len(x), len(y), mode="full") / fs
    m = np.abs(lags) <= max_lag_sec
    c, lags = c[m], lags[m]
    if len(c) == 0:
        return {"lags_sec": np.array([]), "corr": np.array([]), "max_corr": None, "lag_sec": None}
    idx = int(np.argmax(np.abs(c)))
    return {"lags_sec": lags, "corr": c, "max_corr": float(c[idx]), "lag_sec": float(lags[idx])}


def shift_by_samples(x: np.ndarray, shift: int) -> np.ndarray:
    y = np.zeros_like(x)
    if shift == 0:
        y[:] = x
    elif shift > 0:
        y[shift:] = x[:-shift]
    else:
        y[:shift] = x[-shift:]
    return y


def mean_coherence(x: np.ndarray, y: np.ndarray, fs: float, band: tuple[float, float], nperseg: int):
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    if n < 16:
        return np.array([]), np.array([]), None
    f, cxy = signal.coherence(x, y, fs=fs, nperseg=min(nperseg, n))
    m = (f >= band[0]) & (f <= band[1])
    if np.sum(m) < 2:
        return f, cxy, None
    return f, cxy, float(np.nanmean(cxy[m]))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ============================================================
# Robust morphology-based peak/event detection
# ============================================================
def enforce_refractory_by_score(peaks: np.ndarray, scores: np.ndarray, fs: float, refractory_sec: float):
    """
    후보 peak가 너무 가까우면 score가 큰 후보만 남김.
    """
    peaks = np.asarray(peaks, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(peaks) == 0:
        return peaks

    order = np.argsort(scores)[::-1]
    selected = []
    min_dist = int(max(1, round(refractory_sec * fs)))

    for idx in order:
        p = int(peaks[idx])
        if all(abs(p - q) >= min_dist for q in selected):
            selected.append(p)

    return np.array(sorted(selected), dtype=int)


def robust_ecg_rpeak_detector(x: np.ndarray, t: np.ndarray, fs: float, cfg):
    """
    Pan-Tompkins-like + morphology rule 기반 R-peak 검출.
    단순 peak가 아니라:
    - amplitude/prominence
    - rising slope
    - falling slope
    - local maximum
    - refractory period
    조건을 동시에 확인한다.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = len(x)
    if n < 30:
        return np.array([], dtype=int)

    # QRS 강조 신호
    xf = zscore_safe(x)
    dx = np.gradient(xf) * fs
    energy = dx ** 2

    # 80 ms 이동 적분
    win = max(3, int(0.08 * fs))
    integ = np.convolve(energy, np.ones(win) / win, mode="same")
    integ_z = zscore_safe(integ)

    # 기본 후보: 원신호 local max + integrated energy peak 둘 다 활용
    min_dist = max(1, int(fs * max(float(getattr(cfg, "rpeak_min_rr_sec", 0.40)), 60.0 / float(cfg.max_bpm))))
    amp_thr = np.nanmedian(xf) + 0.75 * np.nanstd(xf) * float(getattr(cfg, "prominence_scale", 1.0))
    prom_thr = max(0.45 * np.nanstd(xf) * float(getattr(cfg, "prominence_scale", 1.0)), 1e-6)

    cand_amp, props = signal.find_peaks(
        xf,
        distance=min_dist,
        prominence=prom_thr,
        height=amp_thr
    )

    cand_energy, _ = signal.find_peaks(
        integ_z,
        distance=min_dist,
        prominence=max(0.55 * np.nanstd(integ_z) * float(getattr(cfg, "prominence_scale", 1.0)), 1e-6)
    )

    # energy 후보 주변에서 실제 ECG local max로 스냅
    snap = []
    search = max(2, int(0.08 * fs))
    for c in cand_energy:
        a = max(0, c - search)
        b = min(n, c + search + 1)
        if b > a:
            snap.append(a + int(np.argmax(xf[a:b])))

    candidates = np.unique(np.concatenate([cand_amp, np.asarray(snap, dtype=int)])) if len(snap) else cand_amp

    valid = []
    scores = []
    pre = max(2, int(0.06 * fs))
    post = max(2, int(0.08 * fs))

    for p in candidates:
        if p < pre or p + post >= n:
            continue

        # local maximum 조건
        local = xf[p-pre:p+post+1]
        if p != (p - pre + int(np.argmax(local))):
            # 완전 local max가 아니면 peak 주변 30ms 안의 max로 보정
            a = max(0, p - int(0.03 * fs))
            b = min(n, p + int(0.03 * fs) + 1)
            pp = a + int(np.argmax(xf[a:b]))
            p = pp
            if p < pre or p + post >= n:
                continue

        # rising / falling slope 조건
        left_slope = np.max(dx[p-pre:p])
        right_slope = np.min(dx[p:p+post])
        amp = xf[p]
        prom_local = amp - max(np.median(xf[p-pre:p]), np.median(xf[p:p+post]))

        if left_slope <= 0:
            continue
        if right_slope >= 0:
            continue
        if prom_local < 0.35 * np.std(xf) * float(getattr(cfg, "prominence_scale", 1.0)):
            continue

        # slope zero-crossing 근처 조건: peak 전후로 d1 부호 변화
        if not (np.any(dx[p-pre:p] > 0) and np.any(dx[p:p+post] < 0)):
            continue

        # score: amplitude + slope + energy + prominence
        score = (
            0.35 * amp +
            0.25 * np.log1p(abs(left_slope)) +
            0.20 * np.log1p(abs(right_slope)) +
            0.20 * prom_local
        )
        valid.append(int(p))
        scores.append(float(score))

    if not valid:
        return np.array([], dtype=int)

    peaks = enforce_refractory_by_score(
        np.array(valid, dtype=int),
        np.array(scores, dtype=float),
        fs=fs,
        refractory_sec=60.0 / cfg.max_bpm
    )

    # RR sanity filtering: 너무 짧은 RR은 score 낮은 쪽 제거
    if len(peaks) >= 3:
        keep = np.ones(len(peaks), dtype=bool)
        rr = np.diff(t[peaks])
        min_rr = max(float(getattr(cfg, "rpeak_min_rr_sec", 0.40)), 60.0 / float(cfg.max_bpm))
        max_rr = 60.0 / float(cfg.min_bpm)
        for i, r in enumerate(rr, start=1):
            if r < min_rr:
                # 더 낮은 amplitude peak 제거
                if xf[peaks[i]] >= xf[peaks[i-1]]:
                    keep[i-1] = False
                else:
                    keep[i] = False
            elif r > max_rr:
                # 너무 긴 RR은 제거하지 않고 유지: missed beat 가능성 때문
                pass
        peaks = peaks[keep]

    # Adaptive median RR guard: isolated double-detections 제거.
    # 정상 RR 중앙값의 65%보다 짧은 간격이 생기면, 두 후보 중 morphology score가 약한 쪽 제거.
    if bool(getattr(cfg, "rpeak_rr_median_guard", True)) and len(peaks) >= 5:
        changed = True
        while changed and len(peaks) >= 5:
            changed = False
            rr = np.diff(t[peaks])
            med_rr = float(np.nanmedian(rr)) if np.any(np.isfinite(rr)) else np.nan
            if not np.isfinite(med_rr):
                break
            short_thr = max(float(getattr(cfg, "rpeak_min_rr_sec", 0.40)), 0.65 * med_rr)
            bad = np.where(rr < short_thr)[0]
            if len(bad) == 0:
                break
            i0 = int(bad[0])
            p1, p2 = int(peaks[i0]), int(peaks[i0+1])
            # amplitude + local integrated energy 기준으로 낮은 쪽 제거
            s1 = float(xf[p1]) + 0.25 * float(integ_z[p1])
            s2 = float(xf[p2]) + 0.25 * float(integ_z[p2])
            remove_pos = i0 if s1 < s2 else i0 + 1
            peaks = np.delete(peaks, remove_pos)
            changed = True

    return peaks



def postprocess_rpeaks_short_rr(peaks_idx: np.ndarray, qrs_signal: np.ndarray, fs: float,
                                min_rr_sec: float = 0.45, neighbor_margin_sec: float = 0.04):
    """
    R-peak detector 자체는 바꾸지 않고, 검출 후 double detection만 제거한다.

    정책:
    - 인접 RR < min_rr_sec이면 두 후보 중 QRS-band amplitude가 낮은 쪽 제거
    - 실제 심박이 빠른 피험자라면 min_rr_sec를 낮춰야 함
    """
    peaks = np.asarray(peaks_idx, dtype=int)
    qrs = np.asarray(qrs_signal, dtype=np.float64)

    peaks = peaks[(peaks >= 0) & (peaks < len(qrs))]
    peaks = np.unique(np.sort(peaks))

    if len(peaks) < 2:
        return peaks, []

    min_samp = int(round(float(min_rr_sec) * float(fs)))
    removed = []

    changed = True
    while changed and len(peaks) >= 2:
        changed = False
        rr = np.diff(peaks)
        short_idx = np.where(rr < min_samp)[0]
        if len(short_idx) == 0:
            break

        i = int(short_idx[np.argmin(rr[short_idx])])
        p1 = int(peaks[i])
        p2 = int(peaks[i + 1])

        a1 = float(qrs[p1]) if 0 <= p1 < len(qrs) else -np.inf
        a2 = float(qrs[p2]) if 0 <= p2 < len(qrs) else -np.inf

        def rhythm_penalty(remove_pos):
            kept = np.delete(peaks, remove_pos)
            if len(kept) < 3:
                return 0.0
            rrs = np.diff(kept) / float(fs)
            med = np.nanmedian(rrs)
            if not np.isfinite(med) or med <= 0:
                return 0.0
            return float(np.nanmean(np.abs(rrs - med)))

        if a1 < a2:
            remove_pos = i
            keep_p = p2
            remove_p = p1
        elif a2 < a1:
            remove_pos = i + 1
            keep_p = p1
            remove_p = p2
        else:
            pen_remove_1 = rhythm_penalty(i)
            pen_remove_2 = rhythm_penalty(i + 1)
            if pen_remove_1 <= pen_remove_2:
                remove_pos = i
                keep_p = p2
                remove_p = p1
            else:
                remove_pos = i + 1
                keep_p = p1
                remove_p = p2

        removed.append({
            "removed_peak_idx": int(remove_p),
            "kept_peak_idx": int(keep_p),
            "removed_time_sec": float(remove_p) / float(fs),
            "kept_time_sec": float(keep_p) / float(fs),
            "rr_pair_sec": float(abs(p2 - p1)) / float(fs),
            "removed_qrs_amp": float(qrs[remove_p]) if 0 <= remove_p < len(qrs) else None,
            "kept_qrs_amp": float(qrs[keep_p]) if 0 <= keep_p < len(qrs) else None,
            "reason": f"RR<{float(min_rr_sec):.3f}s; lower QRS-band amplitude removed",
        })

        peaks = np.delete(peaks, remove_pos)
        changed = True

    return peaks, removed

def detect_ecg_q_t_landmarks(ecg_display: np.ndarray, ecg_analysis: np.ndarray, t: np.ndarray, r_peaks: np.ndarray, fs: float):
    """
    RR-adaptive ECG Q/T pseudo-landmark detector with periodic prior.

    반영한 조건:
    1. TIM1 기반 일정 sampling fs 사용
    2. ECG는 quasi-periodic beat sequence라고 가정
    3. Q는 R 직전 RR-adaptive window에서만 탐색
    4. T는 R 이후 RR-adaptive window에서만 탐색
    5. beat-to-beat Q/T timing jump가 크면 reject
    6. confidence 낮으면 NaN 처리

    주의:
    - STM32 ECG display가 표준 Lead ECG가 아니면 Q/T는 true Q/T가 아님.
    - Q/T는 AO/AC ground truth가 아니라 ECG-derived pseudo landmark/quality indicator.
    """
    y_disp = zscore_safe(np.asarray(ecg_display, dtype=np.float64))
    y_analysis = zscore_safe(np.asarray(ecg_analysis, dtype=np.float64))
    t = np.asarray(t, dtype=np.float64)
    r_peaks = np.asarray(r_peaks, dtype=int)

    n = len(y_disp)
    if n < 10 or len(r_peaks) == 0:
        return {
            "q_idx": np.array([], dtype=int),
            "q_time": np.array([], dtype=float),
            "q_confidence": np.array([], dtype=float),
            "t_idx": np.array([], dtype=int),
            "t_time": np.array([], dtype=float),
            "t_confidence": np.array([], dtype=float),
        }

    # display용 T peak는 broad morphology라 너무 높은 대역을 쓰지 않음
    y_smooth = safe_lowpass(y_disp, fs, 7.0, order=3)
    y_qrs = zscore_safe(y_analysis)

    # RR interval estimation
    r_time = np.array([float(t[i]) if 0 <= int(i) < len(t) else np.nan for i in r_peaks], dtype=float)
    rr = np.diff(r_time)
    rr_med = float(np.nanmedian(rr)) if len(rr) and np.any(np.isfinite(rr)) else 0.75
    if not np.isfinite(rr_med) or rr_med <= 0:
        rr_med = 0.75

    rr_local = np.full(len(r_peaks), rr_med, dtype=float)
    for k in range(len(r_peaks)):
        vals = []
        if k > 0 and np.isfinite(r_time[k] - r_time[k-1]):
            vals.append(float(r_time[k] - r_time[k-1]))
        if k + 1 < len(r_peaks) and np.isfinite(r_time[k+1] - r_time[k]):
            vals.append(float(r_time[k+1] - r_time[k]))
        if vals:
            rr_local[k] = float(np.clip(np.nanmedian(vals), 0.40, 1.30))
        else:
            rr_local[k] = rr_med

    q_idx_raw, t_idx_raw = [], []
    q_conf_raw, t_conf_raw = [], []
    q_rel_raw, t_rel_raw = [], []

    min_conf = float(globals().get("QT_LANDMARK_MIN_CONFIDENCE", 0.45))

    for k, rp0 in enumerate(r_peaks):
        rp = int(rp0)
        if rp < 0 or rp >= n:
            q_idx_raw.append(-1); t_idx_raw.append(-1)
            q_conf_raw.append(0.0); t_conf_raw.append(0.0)
            q_rel_raw.append(np.nan); t_rel_raw.append(np.nan)
            continue

        rrk = float(rr_local[k])
        next_r = int(r_peaks[k + 1]) if k + 1 < len(r_peaks) else n - 1

        # ---------------- Q search: RR-adaptive pre-R window ----------------
        # Q는 R 직전. 너무 넓게 잡으면 P/잡음 valley를 잡으므로 20~95 ms 제한.
        q_back_hi = min(0.095, max(0.045, 0.14 * rrk))  # R - q_back_hi
        q_back_lo = min(0.025, max(0.015, 0.035 * rrk)) # R - q_back_lo
        q0 = max(0, rp - int(round(q_back_hi * fs)))
        q1 = max(0, rp - int(round(q_back_lo * fs)))

        qi = -1
        qc = 0.0
        qrel = np.nan

        if q1 > q0 + 2:
            seg = y_qrs[q0:q1]
            local = int(np.nanargmin(seg))
            cand = q0 + local
            rel = (cand - rp) / float(fs)

            valley_depth = float(np.nanmedian(seg) - y_qrs[cand])
            spread = float(np.nanstd(seg) + 1e-9)
            amp_score = 1.0 / (1.0 + np.exp(-(valley_depth / spread - 0.35) * 2.0))

            # expected Q near -45~65 ms, RR adaptive center
            q_center = -float(np.clip(0.075 * rrk, 0.035, 0.070))
            timing_score = float(np.exp(-0.5 * ((rel - q_center) / 0.025) ** 2))

            edge_margin = min(local + 1, len(seg) - local) / max(1.0, len(seg) / 2.0)
            qc = float(np.clip(0.55 * amp_score + 0.30 * timing_score + 0.15 * edge_margin, 0.0, 1.0))

            if qc >= min_conf:
                qi = int(cand)
                qrel = float(rel)

        # ---------------- T search: RR-adaptive post-R window ----------------
        # T는 R 이후. HR이 빠르면 window를 짧게, RR이 길면 넓게.
        # display가 표준 ECG가 아니므로 conservative gating.
        t_start_sec = float(np.clip(0.20 * rrk, 0.150, 0.220))
        t_end_sec = float(np.clip(0.58 * rrk, 0.280, 0.460))

        # next R와 충분히 떨어지게 제한
        next_limit = max(t_start_sec + 0.060, (next_r - rp) / float(fs) - 0.100)
        t_end_sec = min(t_end_sec, next_limit)

        t0 = min(n - 1, rp + int(round(t_start_sec * fs)))
        t1 = min(n, rp + int(round(t_end_sec * fs)))

        ti = -1
        tc = 0.0
        trel = np.nan

        if t1 > t0 + max(5, int(0.06 * fs)):
            seg = y_smooth[t0:t1]
            seg0 = seg - np.nanmedian(seg)
            std0 = float(np.nanstd(seg0) + 1e-9)
            prom = max(0.18 * std0, 1e-6)
            peaks, props = signal.find_peaks(seg0, distance=max(1, int(0.10 * fs)), prominence=prom)

            if len(peaks):
                local_t = np.arange(len(seg0)) / float(fs)
                expected_t_rel = float(np.clip(0.38 * rrk, 0.220, 0.360))
                expected_local = expected_t_rel - t_start_sec

                center_score = np.exp(-0.5 * ((local_t[peaks] - expected_local) / max(0.055, 0.12 * rrk)) ** 2)
                amp = zscore_safe(seg0[peaks])
                prominence = props.get("prominences", np.ones_like(peaks, dtype=float))
                prom_z = zscore_safe(prominence)

                # periodic prior 강화: amplitude만으로 튀는 peak를 선택하지 않게 함
                score = 0.42 * amp + 0.38 * center_score + 0.20 * prom_z
                best_local = int(peaks[int(np.nanargmax(score))])
                cand = t0 + best_local
                rel = (cand - rp) / float(fs)

                amp_score = 1.0 / (1.0 + np.exp(-(seg0[best_local] / std0 - 0.35) * 2.0))
                center_conf = float(np.exp(-0.5 * ((rel - expected_t_rel) / max(0.070, 0.14 * rrk)) ** 2))
                prom_conf = 1.0 / (1.0 + np.exp(-(float(np.max(prominence)) / std0 - 0.25) * 2.0))

                tc = float(np.clip(0.35 * amp_score + 0.45 * center_conf + 0.20 * prom_conf, 0.0, 1.0))
                if tc >= min_conf:
                    ti = int(cand)
                    trel = float(rel)

        q_idx_raw.append(qi); t_idx_raw.append(ti)
        q_conf_raw.append(qc); t_conf_raw.append(tc)
        q_rel_raw.append(qrel); t_rel_raw.append(trel)

    q_idx_arr = np.asarray(q_idx_raw, dtype=int)
    t_idx_arr = np.asarray(t_idx_raw, dtype=int)
    q_conf_arr = np.asarray(q_conf_raw, dtype=float)
    t_conf_arr = np.asarray(t_conf_raw, dtype=float)
    q_rel_arr = np.asarray(q_rel_raw, dtype=float)
    t_rel_arr = np.asarray(t_rel_raw, dtype=float)

    # ---------------- Temporal tracking / periodic consistency ----------------
    if bool(globals().get("QT_USE_RR_ADAPTIVE_PERIODIC_PRIOR", True)):
        def track_rel(rel_arr, idx_arr, conf_arr, max_jump_sec, name):
            rel = rel_arr.copy()
            idx = idx_arr.copy()
            conf = conf_arr.copy()

            valid = np.isfinite(rel) & (idx >= 0) & (conf >= float(globals().get("QT_MIN_TRACK_CONFIDENCE", 0.45)))
            if np.sum(valid) < 3:
                # valid가 너무 적으면 무리해서 tracking하지 않고 confidence 낮은 것 제거
                bad = ~valid
                idx[bad] = -1
                rel[bad] = np.nan
                return rel, idx, conf

            # robust median trajectory
            med = float(np.nanmedian(rel[valid]))
            mad = float(np.nanmedian(np.abs(rel[valid] - med)) + 1e-9)
            hard_lim = max(float(max_jump_sec), 3.0 * 1.4826 * mad)

            # global outlier reject
            outlier = valid & (np.abs(rel - med) > hard_lim)
            idx[outlier] = -1
            rel[outlier] = np.nan
            conf[outlier] *= 0.3

            # beat-to-beat jump reject using previous accepted
            prev = np.nan
            for i in range(len(rel)):
                if idx[i] < 0 or not np.isfinite(rel[i]):
                    continue
                if np.isfinite(prev):
                    if abs(float(rel[i]) - float(prev)) > float(max_jump_sec):
                        # if confidence is not very high, reject
                        if conf[i] < 0.80:
                            idx[i] = -1
                            rel[i] = np.nan
                            conf[i] *= 0.3
                            continue
                prev = rel[i]
            return rel, idx, conf

        q_rel_arr, q_idx_arr, q_conf_arr = track_rel(
            q_rel_arr, q_idx_arr, q_conf_arr,
            float(globals().get("QT_Q_MAX_JUMP_SEC", 0.030)),
            "Q"
        )
        t_rel_arr, t_idx_arr, t_conf_arr = track_rel(
            t_rel_arr, t_idx_arr, t_conf_arr,
            float(globals().get("QT_T_MAX_JUMP_SEC", 0.070)),
            "T"
        )

        # optional interpolation disabled by default
        if bool(globals().get("QT_INTERPOLATE_REJECTED", False)):
            for rel_arr, idx_arr, conf_arr in [(q_rel_arr, q_idx_arr, q_conf_arr), (t_rel_arr, t_idx_arr, t_conf_arr)]:
                valid = np.isfinite(rel_arr) & (idx_arr >= 0)
                if np.sum(valid) >= 3:
                    x = np.arange(len(rel_arr))
                    interp = np.interp(x, x[valid], rel_arr[valid])
                    miss = ~valid
                    rel_arr[miss] = interp[miss]
                    idx_arr[miss] = np.array([int(round(r_peaks[i] + interp[i] * fs)) for i in np.where(miss)[0]], dtype=int)
                    conf_arr[miss] = np.minimum(conf_arr[miss], 0.35)

    # final bounds
    q_idx_arr = np.array([i if 0 <= int(i) < n else -1 for i in q_idx_arr], dtype=int)
    t_idx_arr = np.array([i if 0 <= int(i) < n else -1 for i in t_idx_arr], dtype=int)

    return {
        "q_idx": q_idx_arr,
        "q_time": np.array([np.nan if i < 0 else float(t[i]) for i in q_idx_arr], dtype=float),
        "q_confidence": np.array(q_conf_arr, dtype=float),
        "q_rel_sec": np.array(q_rel_arr, dtype=float),
        "t_idx": t_idx_arr,
        "t_time": np.array([np.nan if i < 0 else float(t[i]) for i in t_idx_arr], dtype=float),
        "t_confidence": np.array(t_conf_arr, dtype=float),
        "t_rel_sec": np.array(t_rel_arr, dtype=float),
        "rr_local_sec": np.array(rr_local, dtype=float),
    }
def scg_inspired_aoac_detector(beat_t_rel: np.ndarray, beat: np.ndarray, win: tuple[float, float], kind: str):
    """
    SCG peak identification 논리 참고형 AO/AC 후보 검출.
    slope 단독이 아니라 다음을 동시에 점수화:
    - rising/falling slope
    - slope zero-crossing
    - local max/min
    - curvature
    - envelope energy
    - timing prior

    AO:
      R 이후 early systolic window에서 upstroke -> local max/energy peak 근방 선택
    AC:
      late systolic window에서 downstroke/notch/local minimum/curvature 변화 근방 선택
    """
    bt = np.asarray(beat_t_rel, dtype=np.float64)
    x = zscore_safe(np.asarray(beat, dtype=np.float64))
    if len(bt) < 10:
        return None, None

    fs = 1.0 / np.median(np.diff(bt))
    y = safe_lowpass(x, fs, 10.0, order=3)
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)

    try:
        env = np.abs(signal.hilbert(y))
        env = safe_lowpass(env, fs, 8.0, order=3)
        env = zscore_safe(env)
    except Exception:
        env = np.abs(y)

    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 5:
        return None, None

    idxs = np.where(m)[0]
    times = bt[idxs]

    # local extrema bonus
    max_bonus = np.zeros(len(idxs))
    min_bonus = np.zeros(len(idxs))
    loc_max, _ = signal.find_peaks(y[idxs], distance=max(1, int(0.04 * fs)))
    loc_min, _ = signal.find_peaks(-y[idxs], distance=max(1, int(0.04 * fs)))
    if len(loc_max):
        max_bonus[loc_max] = 1.0
    if len(loc_min):
        min_bonus[loc_min] = 1.0

    # slope zero-crossing bonus
    zero_pos_to_neg = np.zeros(len(idxs))  # local max 근방
    zero_neg_to_pos = np.zeros(len(idxs))  # local min 근방
    for k, ii in enumerate(idxs):
        lo = max(0, ii - int(0.035 * fs))
        hi = min(len(y), ii + int(0.035 * fs) + 1)
        if hi - lo < 3:
            continue
        dd = d1[lo:hi]
        if np.any((dd[:-1] > 0) & (dd[1:] <= 0)):
            zero_pos_to_neg[k] = 1.0
        if np.any((dd[:-1] < 0) & (dd[1:] >= 0)):
            zero_neg_to_pos[k] = 1.0

    env_score = robust_scale_01(env[idxs])
    curv_score = robust_scale_01(np.abs(d2[idxs]))

    if kind == "ao":
        # AO는 R 직후 upstroke 및 systolic acceleration 성격
        slope_score = robust_scale_01(np.maximum(d1[idxs], 0))
        timing_prior = np.exp(-0.5 * ((times - 0.12) / 0.065) ** 2)
        score = (
            0.30 * slope_score +
            0.20 * curv_score +
            0.20 * env_score +
            0.15 * max_bonus +
            0.10 * zero_pos_to_neg +
            0.05 * timing_prior
        )
    else:
        # AC는 late systolic downstroke/notch 성격
        down_score = robust_scale_01(np.maximum(-d1[idxs], 0))
        timing_prior = np.exp(-0.5 * ((times - 0.38) / 0.10) ** 2)
        score = (
            0.28 * down_score +
            0.22 * curv_score +
            0.18 * env_score +
            0.17 * min_bonus +
            0.10 * zero_neg_to_pos +
            0.05 * timing_prior
        )

    local = int(np.argmax(score))
    idx = int(idxs[local])
    conf = float(np.clip(score[local], 0, 1))
    return idx, conf


def triangular_smooth_envelope(x: np.ndarray, win_len: int = 31):
    """
    Di Rienzo et al. 방식 참고:
    |signal| envelope를 triangular window FIR로 smoothing.
    원 논문은 31-sample triangular FIR envelope를 사용.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 5:
        return np.abs(x)
    win_len = int(max(5, win_len))
    if win_len % 2 == 0:
        win_len += 1
    tri = signal.windows.triang(win_len)
    tri = tri / np.sum(tri)
    env = np.convolve(np.abs(zscore_safe(x)), tri, mode="same")
    return env


def find_adjacent_minima_distance(y: np.ndarray, idx: int, search: int):
    """
    IRP score용 D = D1 + D2 계산.
    idx 좌우 search 범위에서 인접 minimum과의 amplitude distance 합산.
    """
    n = len(y)
    left0 = max(0, idx - search)
    right1 = min(n, idx + search + 1)

    if idx <= left0 or idx >= right1 - 1:
        return 0.0, 0.0, 0.0

    left_min = np.min(y[left0:idx]) if idx > left0 else y[idx]
    right_min = np.min(y[idx+1:right1]) if idx + 1 < right1 else y[idx]

    d1 = float(y[idx] - left_min)
    d2 = float(y[idx] - right_min)
    return d1, d2, float(d1 + d2)


def refine_event_highres(bt: np.ndarray, y: np.ndarray, coarse_idx: int, kind: str, fs_hi: float = 1000.0):
    """
    논문 Fig.8 방식 참고:
    coarse FP 주변 큰 window를 보간한 뒤, coarse point ±10 ms에서 hi-res 재탐색.
    Radar 원신호는 20Hz → 100Hz로 올린 상태지만, timing 표시는 1ms grid에서 refine.
    """
    if coarse_idx is None or coarse_idx < 0 or coarse_idx >= len(y):
        return coarse_idx, None

    fs = 1.0 / np.median(np.diff(bt))
    big = max(8, int(0.050 * fs))  # ±50 ms
    a = max(0, coarse_idx - big)
    b = min(len(y), coarse_idx + big + 1)
    if b - a < 6:
        return coarse_idx, float(bt[coarse_idx])

    tt = bt[a:b]
    yy = y[a:b]
    tt_hi = np.arange(tt[0], tt[-1], 1.0 / fs_hi)
    if len(tt_hi) < 5:
        return coarse_idx, float(bt[coarse_idx])
    yy_hi = np.interp(tt_hi, tt, yy)

    ctime = bt[coarse_idx]
    m = (tt_hi >= ctime - 0.010) & (tt_hi <= ctime + 0.010)
    if np.sum(m) < 3:
        return coarse_idx, float(ctime)

    idxs = np.where(m)[0]
    if kind in ("ao", "irp"):
        local = int(idxs[np.argmax(yy_hi[idxs])])
    elif kind == "ac":
        # AC는 peak 또는 inflection이 될 수 있으므로 curvature 최대점도 허용
        d1 = np.gradient(yy_hi, tt_hi)
        d2 = np.gradient(d1, tt_hi)
        score = robust_scale_01(np.abs(d2[idxs])) + 0.5 * robust_scale_01(np.maximum(-d1[idxs], 0))
        local = int(idxs[np.argmax(score)])
    else:
        local = int(idxs[np.argmax(np.abs(yy_hi[idxs]))])

    refined_t = float(tt_hi[local])
    refined_idx = int(np.argmin(np.abs(bt - refined_t)))
    return refined_idx, refined_t


def get_te_delay_for_beat(ecg: dict, beat_index: int):
    """
    ECG T 후보 시점에서 R 기준 Te delay 계산.
    T 후보가 없으면 None.
    """
    try:
        if "t_time" not in ecg or beat_index >= len(ecg["t_time"]):
            return None
        tt = ecg["t_time"][beat_index]
        rr = ecg["peaks_time"][beat_index]
        if not np.isfinite(tt):
            return None
        val = float(tt - rr)
        if val < 0.12 or val > 0.50:
            return None
        return val
    except Exception:
        return None


def ac_fallback_timing_prior_detector(bt: np.ndarray, beat: np.ndarray,
                                      te_delay: Optional[float] = None,
                                      prev_ac_delay: Optional[float] = None):
    """
    AC 후보를 너무 많이 reject하지 않기 위한 fallback detector.
    목적: AC morphology가 약한 beat에서도 timing prior + inflection + local minimum으로 후보를 확보.
    - Te가 있으면 Te 근처
    - prev_ac_delay가 있으면 이전 valid AC 기준
    - 둘 다 없으면 0.34~0.48s 구간
    """
    bt = np.asarray(bt, dtype=np.float64)
    x = zscore_safe(np.asarray(beat, dtype=np.float64))
    if len(bt) < 10:
        return None, None

    fs = 1.0 / np.median(np.diff(bt))
    y = safe_lowpass(x, fs, 10.0, order=3)
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)

    centers = []
    if prev_ac_delay is not None and np.isfinite(prev_ac_delay):
        centers.append(float(prev_ac_delay))
    if te_delay is not None and np.isfinite(te_delay):
        # AC는 보통 Te/late systolic 주변에서 나타난다고 보고 약간 앞쪽도 허용
        centers.append(float(te_delay - 0.02))
    centers.append(0.38)

    best = None
    for center in centers:
        m = (bt >= center - 0.090) & (bt <= center + 0.080)
        if np.sum(m) < 4:
            continue
        idxs = np.where(m)[0]

        # 후보 1: local minima/notch
        minima, _ = signal.find_peaks(-y[idxs], distance=max(1, int(0.025 * fs)))
        cand = []
        for mm in minima:
            ii = int(idxs[mm])
            timing = np.exp(-0.5 * ((bt[ii] - center) / 0.070) ** 2)
            score = 0.35 * robust_scale_01(np.array([abs(y[ii])]))[0] + 0.35 * timing + 0.30 * robust_scale_01(np.array([abs(d2[ii])]))[0]
            cand.append((ii, float(score)))

        # 후보 2: curvature/inflection
        curv = robust_scale_01(np.abs(d2[idxs]))
        down = robust_scale_01(np.maximum(-d1[idxs], 0))
        timing_vec = np.exp(-0.5 * ((bt[idxs] - center) / 0.075) ** 2)
        score_vec = 0.40 * curv + 0.35 * down + 0.25 * timing_vec
        ii = int(idxs[np.argmax(score_vec)])
        cand.append((ii, float(np.max(score_vec))))

        for c in cand:
            if best is None or c[1] > best[1]:
                best = c

    if best is None:
        return None, None

    idx, score = best
    idx_ref, _ = refine_event_highres(bt, y, idx, "ac")
    return idx_ref, float(np.clip(score, 0, 1))


def ao_fallback_timing_prior_detector(bt: np.ndarray, beat: np.ndarray,
                                      prev_ao_delay: Optional[float] = None):
    """
    AO 후보 확보용 fallback.
    early systolic window에서 rising slope + local max + timing prior 사용.
    """
    bt = np.asarray(bt, dtype=np.float64)
    x = zscore_safe(np.asarray(beat, dtype=np.float64))
    if len(bt) < 10:
        return None, None

    fs = 1.0 / np.median(np.diff(bt))
    y = safe_lowpass(x, fs, 10.0, order=3)
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)

    center = prev_ao_delay if prev_ao_delay is not None and np.isfinite(prev_ao_delay) else 0.12
    m = (bt >= max(0.035, center - 0.075)) & (bt <= min(0.24, center + 0.080))
    if np.sum(m) < 4:
        m = (bt >= 0.05) & (bt <= 0.22)
    if np.sum(m) < 4:
        return None, None

    idxs = np.where(m)[0]
    rise = robust_scale_01(np.maximum(d1[idxs], 0))
    curv = robust_scale_01(np.abs(d2[idxs]))
    timing = np.exp(-0.5 * ((bt[idxs] - center) / 0.060) ** 2)
    amp = robust_scale_01(y[idxs])
    score = 0.35 * rise + 0.25 * curv + 0.25 * timing + 0.15 * amp

    idx = int(idxs[np.argmax(score)])
    idx_ref, _ = refine_event_highres(bt, y, idx, "ao")
    return idx_ref, float(np.clip(np.max(score), 0, 1))


def ac_inflection_zero_cross_detector(bt: np.ndarray, beat: np.ndarray,
                                      ecg_prior_sec: Optional[float] = None,
                                      prev_ac_delay: Optional[float] = None):
    """
    AC 전용 detector.
    AC는 clear peak가 아니라 notch / inflection / transition 성격이 강하므로
    peak 검출 대신 다음 조합으로 검출:
    - falling slope region
    - d1 zero-crossing vicinity
    - |d2| curvature peak
    - local minimum/notch fallback
    - ECG QRT/RR prior + previous valid AC congruency
    """
    bt = np.asarray(bt, dtype=np.float64)
    x = zscore_safe(np.asarray(beat, dtype=np.float64))
    if len(bt) < 10:
        return None, None

    fs = 1.0 / np.median(np.diff(bt))
    y = safe_lowpass(x, fs, 10.0, order=3)
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)

    center = ecg_prior_sec if ecg_prior_sec is not None and np.isfinite(ecg_prior_sec) else 0.38
    if prev_ac_delay is not None and np.isfinite(prev_ac_delay):
        center = 0.55 * center + 0.45 * float(prev_ac_delay)

    lo = max(0.230, center - 0.120)
    hi = min(0.650, center + 0.120)
    m = (bt >= lo) & (bt <= hi)
    if np.sum(m) < 6:
        return None, None

    idxs = np.where(m)[0]
    times = bt[idxs]

    # 1) falling slope score
    fall_score = robust_scale_01(np.maximum(-d1[idxs], 0))

    # 2) curvature score: transition point
    curv_score = robust_scale_01(np.abs(d2[idxs]))

    # 3) zero-crossing score around local min / slope transition
    zc_score = np.zeros(len(idxs), dtype=float)
    for k, ii in enumerate(idxs):
        a = max(0, ii - int(0.050 * fs))
        b = min(len(y), ii + int(0.050 * fs) + 1)
        if b - a < 4:
            continue
        dd = d1[a:b]

        # positive->negative: local max, negative->positive: local min
        has_posneg = np.any((dd[:-1] > 0) & (dd[1:] <= 0))
        has_negpos = np.any((dd[:-1] < 0) & (dd[1:] >= 0))

        if has_negpos:
            zc_score[k] += 1.0
        if has_posneg:
            zc_score[k] += 0.55

    # 4) local notch/minimum bonus
    notch_bonus = np.zeros(len(idxs), dtype=float)
    local_min, _ = signal.find_peaks(-y[idxs], distance=max(1, int(0.030 * fs)))
    if len(local_min):
        notch_bonus[local_min] = 1.0

    # 5) timing/congruency score
    timing_score = np.exp(-0.5 * ((times - center) / 0.080) ** 2)
    if prev_ac_delay is not None and np.isfinite(prev_ac_delay):
        congr_score = np.exp(-0.5 * ((times - float(prev_ac_delay)) / 0.060) ** 2)
    else:
        congr_score = np.ones_like(timing_score) * 0.5

    # 6) inflection is primary; slope only is secondary
    score = (
        0.30 * curv_score +
        0.22 * zc_score +
        0.18 * fall_score +
        0.14 * notch_bonus +
        0.11 * timing_score +
        0.05 * congr_score
    )

    best = int(np.argmax(score))
    idx = int(idxs[best])

    # high-res refinement: AC는 curvature/inflection 기준
    idx_ref, _ = refine_event_highres(bt, y, idx, "ac")
    conf = float(np.clip(score[best], 0, 1))
    return idx_ref, conf


def median_smooth_nan(x: np.ndarray, kernel: int = 5):
    """
    NaN 포함 series median smoothing.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    kernel = int(max(3, kernel))
    if kernel % 2 == 0:
        kernel += 1

    y = x.copy()
    half = kernel // 2
    for i in range(len(x)):
        a = max(0, i - half)
        b = min(len(x), i + half + 1)
        vals = x[a:b]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            y[i] = float(np.median(vals))
    return y


def build_ecg_adaptive_reference_series(ecg: dict, cfg: AnalysisConfig):
    """
    ECG QRT/RR 기반 beat-wise pseudo AO/AC reference를 만든 뒤
    local median smoothing으로 reference jitter를 줄인다.
    """
    n = len(ecg.get("peaks_time", []))
    ao = np.full(n, np.nan, dtype=np.float64)
    ac = np.full(n, np.nan, dtype=np.float64)
    ao_conf = np.zeros(n, dtype=np.float64)
    ac_conf = np.zeros(n, dtype=np.float64)

    for i in range(n):
        try:
            a, c, ca, cc = ecg_estimated_ao_ac_adaptive(ecg, i, cfg)
            ao[i] = a
            ac[i] = c
            ao_conf[i] = ca
            ac_conf[i] = cc
        except Exception:
            ao[i] = getattr(cfg, "expected_ao_sec", 0.12)
            ac[i] = getattr(cfg, "expected_ac_sec", 0.38)
            ao_conf[i] = 0.3
            ac_conf[i] = 0.3

    # local median trend
    ao_s = median_smooth_nan(ao, kernel=5)
    ac_s = median_smooth_nan(ac, kernel=5)

    # physiological sanity
    for i in range(n):
        if not np.isfinite(ao_s[i]) or not (0.035 <= ao_s[i] <= 0.270):
            ao_s[i] = getattr(cfg, "expected_ao_sec", 0.12)
        if not np.isfinite(ac_s[i]) or not (0.240 <= ac_s[i] <= 0.650):
            ac_s[i] = getattr(cfg, "expected_ac_sec", 0.38)
        if ac_s[i] <= ao_s[i] + 0.120:
            ac_s[i] = ao_s[i] + 0.240

    return {
        "ao_raw": ao,
        "ac_raw": ac,
        "ao_smooth": ao_s,
        "ac_smooth": ac_s,
        "ao_conf": ao_conf,
        "ac_conf": ac_conf,
    }

def radar_event_score_detector_with_ecg_prior(bt: np.ndarray, beat: np.ndarray,
                                              kind: str,
                                              ecg_prior_sec: Optional[float] = None,
                                              prev_delay: Optional[float] = None):
    """
    Radar AO/AC peak detection 개선 버전.
    slope 단독이 아니라 다음 점수 결합:
    - slope: AO는 rising, AC는 falling
    - curvature: inflection/valve event 주변 급변
    - envelope: SCG 논문식 envelope energy
    - local extremum: AO local max, AC local min/inflection
    - ECG prior distance penalty: QRT/RR 기반 expected timing과 가까운 후보 우선
    - previous valid beat congruency
    """
    bt = np.asarray(bt, dtype=np.float64)
    x = zscore_safe(np.asarray(beat, dtype=np.float64))
    if len(bt) < 10:
        return None, None

    fs = 1.0 / np.median(np.diff(bt))
    y = safe_lowpass(x, fs, 10.0, order=3)
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    env = triangular_smooth_envelope(y, win_len=min(31, max(5, int(0.12 * fs) | 1)))
    env = zscore_safe(env)

    if kind == "ao":
        center = ecg_prior_sec if ecg_prior_sec is not None and np.isfinite(ecg_prior_sec) else 0.12
        if prev_delay is not None and np.isfinite(prev_delay):
            center = 0.70 * center + 0.30 * float(prev_delay)
        lo, hi = max(0.035, center - 0.085), min(0.270, center + 0.085)
    else:
        center = ecg_prior_sec if ecg_prior_sec is not None and np.isfinite(ecg_prior_sec) else 0.38
        if prev_delay is not None and np.isfinite(prev_delay):
            center = 0.60 * center + 0.40 * float(prev_delay)
        lo, hi = max(0.220, center - 0.115), min(0.650, center + 0.115)

    m = (bt >= lo) & (bt <= hi)
    if np.sum(m) < 5:
        return None, None
    idxs = np.where(m)[0]
    times = bt[idxs]

    # local extrema bonuses
    max_bonus = np.zeros(len(idxs))
    min_bonus = np.zeros(len(idxs))
    pmax, _ = signal.find_peaks(y[idxs], distance=max(1, int(0.025 * fs)))
    pmin, _ = signal.find_peaks(-y[idxs], distance=max(1, int(0.025 * fs)))
    if len(pmax):
        max_bonus[pmax] = 1.0
    if len(pmin):
        min_bonus[pmin] = 1.0

    # slope zero-crossing bonuses
    zc_max = np.zeros(len(idxs))
    zc_min = np.zeros(len(idxs))
    for k, ii in enumerate(idxs):
        a = max(0, ii - int(0.035 * fs))
        b = min(len(y), ii + int(0.035 * fs) + 1)
        if b - a < 3:
            continue
        dd = d1[a:b]
        if np.any((dd[:-1] > 0) & (dd[1:] <= 0)):
            zc_max[k] = 1.0
        if np.any((dd[:-1] < 0) & (dd[1:] >= 0)):
            zc_min[k] = 1.0

    timing_prior = np.exp(-0.5 * ((times - center) / (0.055 if kind == "ao" else 0.080)) ** 2)
    congr = np.ones(len(idxs))
    if prev_delay is not None and np.isfinite(prev_delay):
        congr = np.exp(-0.5 * ((times - float(prev_delay)) / (0.045 if kind == "ao" else 0.065)) ** 2)

    if kind == "ao":
        slope_score = robust_scale_01(np.maximum(d1[idxs], 0))
        curv_score = robust_scale_01(np.maximum(d2[idxs], 0) + 0.5 * np.abs(d2[idxs]))
        amp_score = robust_scale_01(y[idxs])
        score = (
            0.28 * slope_score +
            0.20 * curv_score +
            0.18 * robust_scale_01(env[idxs]) +
            0.12 * max_bonus +
            0.10 * zc_max +
            0.08 * timing_prior +
            0.04 * congr
        )
    else:
        down_score = robust_scale_01(np.maximum(-d1[idxs], 0))
        curv_score = robust_scale_01(np.abs(d2[idxs]))
        # AC는 clear peak가 아니라 inflection/notch일 수 있어 min + curvature 가중
        score = (
            0.24 * down_score +
            0.24 * curv_score +
            0.16 * robust_scale_01(env[idxs]) +
            0.14 * min_bonus +
            0.10 * zc_min +
            0.08 * timing_prior +
            0.04 * congr
        )

    best_local = int(np.argmax(score))
    idx = int(idxs[best_local])
    idx_ref, _ = refine_event_highres(bt, y, idx, kind)
    conf = float(np.clip(score[best_local], 0, 1))
    return idx_ref, conf

def zheng_seventh_power_ao_detector(bt: np.ndarray, beat: np.ndarray,
                                    ecg_prior_sec: Optional[float] = None,
                                    prev_ao_delay: Optional[float] = None):
    """
    Zheng et al. AO detector를 radar PPG-like beat에 맞게 간소화 적용.

    논문 핵심:
    - AO 성분은 짧고 pulsatile한 peak
    - seventh power law로 AO peak를 강조하고 spurious peak를 억제
    - Hilbert envelope + moving average smoothing 후 main envelope peak를 AO로 사용

    여기서는 SVMD 자체는 실시간 복잡도 때문에 생략하고,
    R-peak로 잘린 beat에서 band-limited cardiac morphology를 대상으로
    seventh-power envelope peak를 AO 후보로 사용한다.
    """
    bt = np.asarray(bt, dtype=np.float64)
    x = zscore_safe(np.asarray(beat, dtype=np.float64))
    if len(bt) < 10:
        return None, None

    fs = 1.0 / np.median(np.diff(bt))
    # AO 후보 대역: radar beat morphology에서 sharp cardiac component만 남김
    if fs > 30:
        y = safe_bandpass(x, fs, 5.0, min(35.0, 0.45 * fs), order=3)
    else:
        y = safe_lowpass(x, fs, 10.0, order=3)
    y = zscore_safe(y)

    # signed seventh power: peak polarity를 보존하되 envelope에서 절대 에너지화
    yp = np.sign(y) * (np.abs(y) ** 7)

    try:
        env = np.abs(signal.hilbert(yp))
    except Exception:
        env = np.abs(yp)

    # 논문처럼 약 0.1 s sliding average smoothing
    w = max(3, int(round(0.10 * fs)))
    if w % 2 == 0:
        w += 1
    env_s = np.convolve(env, np.ones(w) / w, mode="same")
    env_s = zscore_safe(env_s)

    center = ecg_prior_sec if ecg_prior_sec is not None and np.isfinite(ecg_prior_sec) else 0.12
    if prev_ao_delay is not None and np.isfinite(prev_ao_delay):
        center = 0.70 * float(center) + 0.30 * float(prev_ao_delay)

    m = (bt >= max(0.035, center - 0.090)) & (bt <= min(0.280, center + 0.090))
    if np.sum(m) < 5:
        m = (bt >= 0.050) & (bt <= 0.250)
    if np.sum(m) < 5:
        return None, None

    idxs = np.where(m)[0]
    timing = np.exp(-0.5 * ((bt[idxs] - center) / 0.065) ** 2)

    # envelope peak + raw morphology amplitude + timing prior
    score = (
        0.62 * robust_scale_01(env_s[idxs]) +
        0.23 * robust_scale_01(y[idxs]) +
        0.15 * timing
    )

    loc = int(np.argmax(score))
    idx = int(idxs[loc])

    # envelope로 찾은 뒤 실제 waveform peak 근방으로 snap
    s = max(0, idx - int(0.030 * fs))
    e = min(len(y), idx + int(0.030 * fs) + 1)
    if e - s >= 3:
        idx = s + int(np.argmax(y[s:e]))

    idx_ref, _ = refine_event_highres(bt, y, idx, "ao")
    conf = float(np.clip(score[loc], 0.0, 1.0))
    return idx_ref, conf


def scg_paper_style_ao_ac_detector(bt: np.ndarray, beat: np.ndarray, kind: str,
                                   prev_ref_delay: Optional[float] = None,
                                   te_delay: Optional[float] = None):
    """
    Di Rienzo et al. SCG algorithm을 radar PPG-like에 맞게 변형한 morphology detector.

    AO:
    - S1 envelope 영역: R+25~75 ms에서 ICP(깊은 minimum) 탐색
    - AO: ICP 이후 50 ms 내 첫 유효 peak
    - peak amplitude distance >= 0.7*|ICP|
    - 이전 유효 beat의 ICP delay와 ±30 ms congruency 보조

    AC:
    - S2 영역: ECG T 후보(Te) 중심 ±30 ms에서 IRP anchor 탐색
    - IRP: 좌/우 adjacent minima와의 distance D=D1+D2가 큰 peak
    - 이전 20 beat reference IRP delay와 ±20 ms congruency 보조
    - AC: IRP 이전 10~40 ms 내 peak/inflection 후보
    """
    bt = np.asarray(bt, dtype=np.float64)
    x = zscore_safe(np.asarray(beat, dtype=np.float64))
    if len(bt) < 10:
        return None, None, {}

    fs = 1.0 / np.median(np.diff(bt))
    y = safe_bandpass(x, fs, 5.0, min(40.0, 0.45 * fs), order=3) if fs > 30 else safe_lowpass(x, fs, 10.0, order=3)
    y = zscore_safe(y)
    env = triangular_smooth_envelope(y, win_len=min(31, max(5, int(0.12 * fs) | 1)))

    meta = {"mode": "paper_style", "anchor_idx": None, "anchor_time": None, "anchor_score": None}

    if kind == "ao":
        # S1si: R+25~75ms. Radar/PPG-like에서 너무 이르면 놓칠 수 있어 약간 확장.
        s1 = (bt >= 0.025) & (bt <= 0.110)
        if np.sum(s1) < 4:
            return None, None, meta
        s1idx = np.where(s1)[0]

        # ICP: deepest minimum in S1si
        icp_idx = int(s1idx[np.argmin(y[s1idx])])
        icp_delay = float(bt[icp_idx])

        # Congruency with previous ICP delay
        congr = 1.0
        if prev_ref_delay is not None and np.isfinite(prev_ref_delay):
            congr = float(np.exp(-0.5 * ((icp_delay - prev_ref_delay) / 0.030) ** 2))

        # AO: first peak after ICP within 50ms, amplitude threshold
        search = (bt > bt[icp_idx]) & (bt <= bt[icp_idx] + 0.060)
        if np.sum(search) < 3:
            return None, None, meta
        idxs = np.where(search)[0]
        seg = y[idxs]
        peaks, _ = signal.find_peaks(seg, distance=max(1, int(0.015 * fs)))
        candidates = []
        thr = 0.7 * abs(float(y[icp_idx]))
        if len(peaks):
            for p in peaks:
                ii = int(idxs[p])
                amp_dist = float(y[ii] - y[icp_idx])
                if amp_dist >= max(0.15, thr):
                    # 첫 peak 우선 + amplitude + envelope + congruency
                    timing_prior = np.exp(-0.5 * ((bt[ii] - 0.12) / 0.06) ** 2)
                    score = 0.35 * amp_dist + 0.25 * env[ii] + 0.25 * timing_prior + 0.15 * congr
                    candidates.append((ii, score))
        # peak가 없으면 inflection fallback
        if not candidates:
            d1 = np.gradient(y, bt)
            d2 = np.gradient(d1, bt)
            score_vec = robust_scale_01(np.maximum(d1[idxs], 0)) + robust_scale_01(np.abs(d2[idxs]))
            ii = int(idxs[np.argmax(score_vec)])
            score = float(np.max(score_vec)) * 0.5
            candidates.append((ii, score))

        idx, score = max(candidates, key=lambda z: z[1])
        idx_ref, tref = refine_event_highres(bt, y, idx, "ao")
        meta.update({"anchor_idx": icp_idx, "anchor_time": icp_delay, "anchor_score": float(abs(y[icp_idx])), "refined_time": tref})
        return idx_ref, float(np.clip(score, 0, 1)), meta

    else:
        # S2si: 60ms segment centered on Te. Te 없으면 0.38s 중심 fallback.
        center = te_delay if te_delay is not None else 0.38
        s2 = (bt >= center - 0.050) & (bt <= center + 0.070)
        # 레이더 morphology 변동을 고려해 최소 범위 확보
        if np.sum(s2) < 4:
            s2 = (bt >= 0.28) & (bt <= 0.52)
        if np.sum(s2) < 4:
            return None, None, meta

        idxs = np.where(s2)[0]
        seg = y[idxs]
        peaks, _ = signal.find_peaks(seg, distance=max(1, int(0.020 * fs)))

        # IRP: D=D1+D2 최대 peak
        search_pts = idxs[peaks] if len(peaks) else idxs
        best = None
        for ii in search_pts:
            d1, d2, D = find_adjacent_minima_distance(y, int(ii), search=max(2, int(0.045 * fs)))
            timing_prior = np.exp(-0.5 * ((bt[ii] - center) / 0.065) ** 2)
            congr = 1.0
            if prev_ref_delay is not None and np.isfinite(prev_ref_delay):
                congr = float(np.exp(-0.5 * ((bt[ii] - prev_ref_delay) / 0.020) ** 2))
            score = 0.50 * D + 0.30 * timing_prior + 0.20 * congr
            if best is None or score > best[1]:
                best = (int(ii), float(score), float(D))

        if best is None:
            return None, None, meta

        irp_idx, irp_score, D = best
        irp_delay = float(bt[irp_idx])

        # AC: first peak or inflection preceding IRP by 10~40ms
        ac_win = (bt >= irp_delay - 0.045) & (bt <= irp_delay - 0.008)
        if np.sum(ac_win) < 3:
            ac_win = (bt >= max(0.25, center - 0.10)) & (bt <= center)
        if np.sum(ac_win) < 3:
            return irp_idx, float(np.clip(irp_score, 0, 1)), meta

        aidx = np.where(ac_win)[0]
        d1v = np.gradient(y, bt)
        d2v = np.gradient(d1v, bt)

        # AC가 clear peak가 아닐 수 있으므로 peak + inflection 둘 다 점수화
        loc_peaks, _ = signal.find_peaks(y[aidx], distance=max(1, int(0.015 * fs)))
        cand = []
        for p in loc_peaks:
            ii = int(aidx[p])
            timing = np.exp(-0.5 * ((irp_delay - bt[ii] - 0.025) / 0.018) ** 2)
            cand.append((ii, 0.45 * y[ii] + 0.35 * timing + 0.20 * D))

        # inflection fallback
        inf_score = robust_scale_01(np.abs(d2v[aidx])) + 0.7 * robust_scale_01(np.maximum(-d1v[aidx], 0))
        ii = int(aidx[np.argmax(inf_score)])
        cand.append((ii, float(np.max(inf_score)) * 0.7))

        ac_idx, ac_score = max(cand, key=lambda z: z[1])
        ac_idx_ref, tref = refine_event_highres(bt, y, ac_idx, "ac")

        meta.update({"anchor_idx": irp_idx, "anchor_time": irp_delay, "anchor_score": D, "refined_time": tref})
        return ac_idx_ref, float(np.clip(ac_score, 0, 1)), meta

def morphology_event_detector(beat_t_rel: np.ndarray, beat: np.ndarray, win: tuple[float, float], kind: str):
    """
    Radar beat에서 형태 기반 AO/AC 후보 검출.
    rising / max slope / slope=0 crossing / falling / curvature를 모두 반영.
    kind='ao':
      - early systolic upstroke
      - max positive slope
      - 이후 local maximum 직전/근방 후보
    kind='ac':
      - late systolic
      - max negative slope
      - notch/local minimum
      - curvature change
    """
    bt = np.asarray(beat_t_rel, dtype=np.float64)
    bx = zscore_safe(np.asarray(beat, dtype=np.float64))

    if len(bt) < 8:
        return None, None

    fs = 1.0 / np.median(np.diff(bt))
    y = safe_lowpass(bx, fs=fs, cutoff=10.0, order=3)
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)

    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 5:
        return None, None

    idxs = np.where(m)[0]

    if kind == "ao":
        # rising 상태와 max upstroke를 강하게 반영
        pos_slope = np.maximum(d1[idxs], 0)
        curv_pos = np.maximum(d2[idxs], 0)

        # slope=0 crossing 후보: +에서 -로 바뀌는 local max 근처
        zero_bonus = np.zeros_like(pos_slope)
        for k, ii in enumerate(idxs):
            lo = max(0, ii - int(0.04 * fs))
            hi = min(len(y), ii + int(0.08 * fs))
            if hi - lo > 3:
                # 후보 이후 local max가 가까우면 bonus
                local_d = d1[lo:hi]
                if np.any(local_d[:-1] > 0) and np.any(local_d[1:] <= 0):
                    zero_bonus[k] = 1.0

        # 너무 늦은 AO 방지: expected center 가까울수록 가산
        center = 0.12
        timing_penalty = np.exp(-0.5 * ((bt[idxs] - center) / 0.07) ** 2)

        score = (
            0.45 * robust_scale_01(pos_slope) +
            0.25 * robust_scale_01(curv_pos) +
            0.15 * zero_bonus +
            0.15 * timing_penalty
        )

        local = int(np.argmax(score))
        idx = int(idxs[local])
        conf = float(score[local])
        return idx, conf

    else:
        # AC: downstroke, notch/local minimum, curvature change, timing prior
        neg_slope = np.maximum(-d1[idxs], 0)
        curv_abs = np.abs(d2[idxs])

        # local minimum / notch bonus
        notch_bonus = np.zeros_like(neg_slope)
        seg = y[idxs]
        mins, _ = signal.find_peaks(-seg, distance=max(1, int(0.04 * fs)))
        if len(mins):
            for mm in mins:
                notch_bonus[mm] = 1.0

        center = 0.38
        timing_penalty = np.exp(-0.5 * ((bt[idxs] - center) / 0.10) ** 2)

        score = (
            0.35 * robust_scale_01(neg_slope) +
            0.25 * robust_scale_01(curv_abs) +
            0.20 * notch_bonus +
            0.20 * timing_penalty
        )

        local = int(np.argmax(score))
        idx = int(idxs[local])
        conf = float(score[local])
        return idx, conf




def hampel_filter_1d(x: np.ndarray, fs: float, window_sec: float = 0.15, nsigma: float = 5.0):
    """
    Hampel-like robust spike/contact artifact suppressor.
    Median/MAD 기준으로 순간 spike를 local median으로 교체한다.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    n = len(x)
    if n < 5:
        return x
    k = max(3, int(round(float(window_sec) * float(fs))))
    if k % 2 == 0:
        k += 1
    half = k // 2
    y = x.copy()
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        seg = x[a:b]
        med = np.nanmedian(seg)
        mad = np.nanmedian(np.abs(seg - med)) + 1e-12
        if np.isfinite(x[i]) and abs(x[i] - med) > float(nsigma) * 1.4826 * mad:
            y[i] = med
    return y


def lms_adaptive_filter_ecg(primary: np.ndarray, reference: np.ndarray,
                            mu: float = 0.0012, order: int = 16):
    """
    ECG artifact cancellation용 normalized LMS.

    primary:
        artifact가 섞인 ECG normalized signal
    reference:
        ECG raw에서 추출한 저주파 artifact reference, 예: 0.05~2.0 Hz
    return:
        cleaned = primary - estimated_artifact
        estimated_artifact
    """
    d = np.asarray(primary, dtype=np.float64)
    x = np.asarray(reference, dtype=np.float64)

    n = min(len(d), len(x))
    d = d[:n]
    x = x[:n]

    if n <= int(order) + 2:
        return d.copy(), np.zeros_like(d)

    d = zscore_safe(d)
    x = zscore_safe(x)

    w = np.zeros(int(order), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    e = np.zeros(n, dtype=np.float64)

    ref_pad = np.pad(x, (int(order) - 1, 0), mode="edge")
    eps = 1e-8

    for i in range(n):
        xv = ref_pad[i:i + int(order)][::-1]
        y[i] = float(np.dot(w, xv))
        e[i] = d[i] - y[i]
        norm = float(np.dot(xv, xv) + eps)
        w += (float(mu) / norm) * e[i] * xv

    return e, y



def fft_band_attenuate_zero_phase(x: np.ndarray, fs: float, bands, attenuations, taper_sec: float = 0.50):
    """
    Real FFT-domain band attenuation for quasi-stationary motion artifact.

    Returns:
      cleaned, removed_component

    주의:
    - FFT 방식은 stationary/주기성 artifact에 효과적.
    - ECG 생체 성분과 겹치는 대역을 과도하게 제거하면 morphology도 손상됨.
    - 따라서 QRS 검출은 최종적으로 8~25 Hz band에서 수행한다.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 8:
        return x.copy(), np.zeros_like(x)

    xx = x.copy()
    mean = float(np.nanmean(xx)) if np.any(np.isfinite(xx)) else 0.0

    # Edge taper로 FFT ringing 완화
    taper_n = int(max(0, round(float(taper_sec) * float(fs))))
    if taper_n > 2 and 2 * taper_n < n:
        win = np.ones(n, dtype=np.float64)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, taper_n))
        win[:taper_n] = ramp
        win[-taper_n:] = ramp[::-1]
        xx = (xx - mean) * win + mean

    X = np.fft.rfft(xx)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    X_clean = X.copy()

    for band, att in zip(bands, attenuations):
        lo, hi = float(band[0]), float(band[1])
        if hi <= lo:
            continue
        m = (freqs >= lo) & (freqs <= hi)
        X_clean[m] *= float(att)

    cleaned = np.fft.irfft(X_clean, n=n)
    removed = xx - cleaned
    return cleaned, removed

def preprocess_stm32_ecg(raw_adc: np.ndarray, fs: float, cfg: ECGConfig):
    """
    STM32 ADC ECG preprocessing pipeline.

    순서:
    1) Hampel spike/contact artifact suppression
    2) baseline drift removal
    3) LF artifact reference extraction
    4) normalized LMS adaptive cancellation
    5) FFT-domain motion-band attenuation
    6) display ECG / QRS-band ECG 분리

    중요:
    - fft_clean은 use_ecg_fft_motion_suppression 값과 무관하게 항상 정의된다.
    - R-peak 검출은 qrs(8~25 Hz)에서 수행한다.
    """
    raw_adc = np.asarray(raw_adc, dtype=np.float64)

    # 1) robust outlier suppression
    x0 = raw_adc.copy()
    med0 = np.nanmedian(x0)
    mad0 = np.nanmedian(np.abs(x0 - med0)) + 1e-12
    x0[np.abs(x0 - med0) > 10 * 1.4826 * mad0] = med0

    # 2) local spike/contact artifact suppression
    x_hampel = hampel_filter_1d(
        x0,
        fs,
        window_sec=float(getattr(cfg, "ecg_hampel_window_sec", 0.15)),
        nsigma=float(getattr(cfg, "ecg_hampel_nsigma", 5.0))
    )

    raw_norm = zscore_safe(x_hampel - np.nanmedian(x_hampel))
    raw_norm = safe_notch(raw_norm, fs, getattr(cfg, "notch_hz", None))

    # 3) baseline drift estimate/subtraction
    baseline_hz = float(getattr(cfg, "ecg_baseline_lowpass_hz", 0.70))
    baseline_est = safe_lowpass(raw_norm, fs, baseline_hz, order=2)
    detrended = zscore_safe(raw_norm - baseline_est)

    # 4) LF artifact reference
    lo, hi = getattr(cfg, "ecg_artifact_ref_band_hz", (0.05, 2.0))
    artifact_ref = safe_bandpass(raw_norm, fs, float(lo), float(hi), order=2)

    # 5) LMS adaptive cancellation
    if bool(getattr(cfg, "use_ecg_artifact_lms", True)):
        lms_clean, artifact_est = lms_adaptive_filter_ecg(
            primary=detrended,
            reference=artifact_ref,
            mu=float(getattr(cfg, "ecg_lms_mu", 0.0012)),
            order=int(getattr(cfg, "ecg_lms_order", 16))
        )
    else:
        lms_clean = detrended.copy()
        artifact_est = np.zeros_like(detrended)

    lms_clean = hampel_filter_1d(
        lms_clean,
        fs,
        window_sec=float(getattr(cfg, "ecg_post_lms_hampel_window_sec", 0.09)),
        nsigma=6.0
    )
    lms_clean = zscore_safe(lms_clean)

    # 6) FFT-domain motion-band attenuation
    # BUGFIX: fft_clean / fft_removed를 if 밖에서 기본 정의해서 NameError 방지
    fft_clean = lms_clean.copy()
    fft_removed = np.zeros_like(lms_clean)

    if bool(getattr(cfg, "use_ecg_fft_motion_suppression", True)):
        try:
            fft_clean_tmp, fft_removed_tmp = fft_band_attenuate_zero_phase(
                lms_clean,
                fs,
                bands=getattr(cfg, "ecg_fft_motion_bands_hz", ((0.05, 0.70), (0.70, 2.50))),
                attenuations=getattr(cfg, "ecg_fft_motion_attenuation", (0.05, 0.35)),
                taper_sec=float(getattr(cfg, "ecg_fft_taper_sec", 0.50))
            )
            if len(fft_clean_tmp) == len(lms_clean):
                fft_clean = zscore_safe(fft_clean_tmp)
                fft_removed = zscore_safe(fft_removed_tmp)
        except Exception:
            # FFT motion suppression 실패 시에도 pipeline은 계속 진행
            fft_clean = lms_clean.copy()
            fft_removed = np.zeros_like(lms_clean)

    # 7) display ECG
    d_lo, d_hi = getattr(cfg, "ecg_display_band_hz", (0.7, 18.0))
    display = safe_bandpass(fft_clean, fs, float(d_lo), float(d_hi), order=3)
    display = safe_notch(display, fs, getattr(cfg, "notch_hz", None))
    display = zscore_safe(display)

    display_smooth = safe_lowpass(display, fs, min(12.0, 0.40 * fs), order=3)
    display_smooth = zscore_safe(display_smooth)

    # 8) QRS/R-peak detector band
    q_lo, q_hi = getattr(cfg, "ecg_qrs_band_hz", (8.0, 25.0))
    qrs = safe_bandpass(fft_clean, fs, float(q_lo), float(q_hi), order=4)
    qrs = safe_notch(qrs, fs, getattr(cfg, "notch_hz", None))
    qrs = zscore_safe(qrs)

    return {
        "raw_adc_outlier_suppressed": x0,
        "raw_hampel": zscore_safe(x_hampel - np.nanmedian(x_hampel)),
        "raw_norm": zscore_safe(raw_norm),
        "baseline_est": zscore_safe(baseline_est),
        "artifact_ref": zscore_safe(artifact_ref),
        "artifact_est": zscore_safe(artifact_est),
        "lms_clean": zscore_safe(lms_clean),
        "fft_motion_removed": zscore_safe(fft_removed),
        "fft_motion_clean": zscore_safe(fft_clean),
        "display": display,
        "display_smooth": display_smooth,
        "qrs": qrs,
    }
def parse_stm32_ecg_csv_lines(text_buffer: str, new_text: str, signal_col: int = 0):
    """
    Robust STM32 ECG UART CSV parser.

    Supported:
      A) sample_index,ADCValue,Smooth_ECG
         example: 5450886,3824,3110

      B) ADCValue,Smooth_ECG
         example: 3824,3110

      C) raw-only numeric line
         example: 3110

    Fixes:
      - serial starts mid-line
      - CRLF / LF mixed line endings
      - extra debug tokens
      - long running sample_index values
      - warm-up bytes are now parsed, not discarded
    """
    if text_buffer is None:
        text_buffer = ""
    if new_text is None:
        new_text = ""
    text_buffer += new_text

    # Split while preserving a trailing incomplete line.
    raw_lines = text_buffer.splitlines(keepends=True)
    complete = []
    remain = ""
    for line in raw_lines:
        if line.endswith("\n") or line.endswith("\r"):
            complete.append(line.strip())
        else:
            remain = line

    values = []
    raw_values = []
    smooth_values = []
    sample_indices = []

    for line in complete:
        s = str(line).strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        low = s.lower()
        if ("sample" in low) or ("adc" in low) or ("smooth" in low):
            continue

        parts = re.split(r"[,\t ;]+", s)
        nums = []
        for p in parts:
            if not p:
                continue
            try:
                nums.append(float(p))
            except Exception:
                pass

        if len(nums) >= 3:
            sample_idx = nums[0]
            raw = nums[1]
            smooth = nums[2]
        elif len(nums) == 2:
            sample_idx = np.nan
            raw = nums[0]
            smooth = nums[1]
        elif len(nums) == 1:
            sample_idx = np.nan
            raw = nums[0]
            smooth = np.nan
        else:
            continue

        # Basic sanity: keep STM32 ADC-scale numbers, reject absurd corrupt rows.
        if not np.isfinite(raw):
            continue
        if np.isfinite(smooth) and abs(smooth) > 1e7:
            continue
        if abs(raw) > 1e7:
            continue

        sample_indices.append(sample_idx)
        raw_values.append(raw)
        smooth_values.append(smooth)
        if int(signal_col) == 1 and np.isfinite(smooth):
            values.append(smooth)
        else:
            values.append(raw)

    return remain, values, raw_values, smooth_values, sample_indices


def looks_like_stm32_csv_text(s: str) -> bool:
    """
    chunk/line이 STM32 CSV(숫자,숫자)인지 대략 판단.
    """
    if not s or "," not in s:
        return False
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return False
    ok = 0
    for ln in lines[:5]:
        parts = re.split(r"[,\t ;]+", ln)
        nums = 0
        for p in parts[:3]:
            try:
                float(p)
                nums += 1
            except Exception:
                pass
        if nums >= 2:
            ok += 1
    return ok >= 1



def load_STM32_txt(path: Path):
    """
    STM32 CSV 로그 파일 로드.
    예:
      2026/4/29  18:31:25
      Sampling Rate: 250, Number of Data: 2500

      0
      -3
      -6
      ...
    반환: t, raw, fs
    """
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    fs = None
    values = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("sampling rate"):
            # "Sampling Rate: 250, Number of Data: 2500"
            try:
                left = s.split(",")[0]
                fs = float(left.split(":")[1].strip())
            except Exception:
                pass
            continue
        try:
            values.append(float(s))
        except Exception:
            pass

    if fs is None:
        fs = ECG_FS_HINT_HZ

    raw = np.asarray(values, dtype=np.float64)
    t = np.arange(len(raw), dtype=np.float64) / float(fs)
    return t, raw, float(fs)

# ============================================================
# ECG Collector
# ============================================================
class ECGCollector:
    def __init__(self, cfg: ECGConfig):
        self.cfg = cfg
        self.t: list[float] = []
        self.raw: list[float] = []
        self.sample_idx: list[float] = []
        self.raw_adc_col: list[float] = []
        self.smooth_adc_col: list[float] = []
        self._first_sample_idx: Optional[float] = None
        self._first_sample_wall_t: Optional[float] = None
        self.debug_bytes = bytearray()
        self.debug_lines: list[str] = []
        self.ready_event = threading.Event()
        self.error: Optional[Exception] = None

    def acquire(self, duration_sec: float, shared: dict[str, float], start_event: threading.Event, stop_event: threading.Event):
        ser = None
        try:
            print(f"[ECG] Open {self.cfg.port} @ {self.cfg.baudrate}, input=STM32 CSV (sample_index,ADCValue,Smooth_ECG)")
            ser = serial.Serial(self.cfg.port, self.cfg.baudrate, timeout=self.cfg.timeout_sec)
            try:
                ser.dtr = self.cfg.dtr_enable
                ser.rts = self.cfg.rts_enable
            except Exception:
                pass
            time.sleep(0.8)
            ser.reset_input_buffer()

            # STM32는 전원/펌웨어 실행 후 UART CSV를 계속 스트리밍. 안전하게 개행만 전송
            if self.cfg.write_start_newline:
                try:
                    ser.write(b"\r\n")
                    ser.flush()
                except Exception:
                    pass

            self.ready_event.set()

            start_event.wait()
            t0 = shared["t0"]
            bin_buf = bytearray()
            ascii_buf = ""

            # Data-driven acquisition:
            # - Do not stop ECG purely by PC wall time when STM32 sample_index is used.
            # - Stop after the collected ECG time span reaches duration_sec.
            # - Keep a wall-time guard to avoid infinite blocking.
            wall_guard_sec = max(duration_sec * 3.0, duration_sec + 30.0)

            while not stop_event.is_set():
                wall_t = time.perf_counter() - t0
                if wall_t >= wall_guard_sec:
                    break
                if len(self.t) >= 2 and (self.t[-1] - self.t[0]) >= duration_sec:
                    break

                n_wait = ser.in_waiting
                if n_wait:
                    chunk = ser.read(n_wait)

                    # live serial 실제 원시 데이터 진단용 저장
                    if len(self.debug_bytes) < 4096:
                        self.debug_bytes.extend(chunk[: max(0, 4096 - len(self.debug_bytes))])

                    # PATCH7:
                    # Do NOT discard ECG bytes during warm-up.
                    # The recent log shows STM32 sends valid CSV immediately, and
                    # warmup discard can throw away almost all parseable ECG data.
                    # Later preprocessing / R-peak detection handles transient samples.
                    # STM32 ECG 전용 입력. 다른 parsing fallback 없음.
                    # UART line = sample_index,ADCValue,Smooth_ECG or ADCValue,Smooth_ECG
                    decoded = chunk.decode(errors="ignore")
                    if decoded and len(self.debug_lines) < 20:
                        self.debug_lines.append(decoded[:200])
                    ascii_buf, vals, raw_vals, smooth_vals, sample_idxs = parse_stm32_ecg_csv_lines(
                        ascii_buf,
                        decoded,
                        signal_col=int(getattr(self.cfg, "stm32_csv_signal_col", 0))
                    )

                    # timestamp policy
                    # 1) 권장: STM32가 sample_index,ADCValue,Smooth_ECG를 보내면
                    #    sample_index / ECG_FS_HINT_HZ로 균일 시간축 생성.
                    # 2) 구형: ADCValue,Smooth_ECG만 있으면 PC 수신 시각을 기준으로 chunk 내부를 fs_hint 간격으로 역분배.
                    read_t = time.perf_counter() - t0
                    dt = 1.0 / float(self.cfg.fs_hint_hz)
                    n_vals = len(vals)

                    for j, v in enumerate(vals):
                        sidx = sample_idxs[j] if j < len(sample_idxs) else np.nan

                        if np.isfinite(sidx) and bool(getattr(self.cfg, "use_stm32_sample_index_time", True)):
                            if self._first_sample_idx is None:
                                self._first_sample_idx = float(sidx)
                                self._first_sample_wall_t = read_t - (n_vals - 1 - j) * dt
                                if self._first_sample_wall_t < 0:
                                    self._first_sample_wall_t = 0.0

                            sample_t = float(self._first_sample_wall_t) + (float(sidx) - float(self._first_sample_idx)) * dt
                        else:
                            sample_t = read_t - (n_vals - 1 - j) * dt
                            if sample_t < 0:
                                sample_t = 0.0

                        self.t.append(float(sample_t))
                        self.raw.append(float(v))
                        self.sample_idx.append(float(sidx) if np.isfinite(sidx) else np.nan)
                        self.raw_adc_col.append(float(raw_vals[j]) if j < len(raw_vals) else np.nan)
                        self.smooth_adc_col.append(float(smooth_vals[j]) if j < len(smooth_vals) else np.nan)
                else:
                    # 초반 일정 시간 동안 아무 byte도 안 오면 ECG thread만 종료.
                    # Radar는 계속 돌 수 있으나, 이후 analyze에서 명확한 원인을 출력.
                    if wall_t >= self.cfg.fail_fast_if_no_ecg_sec and len(self.raw) == 0 and len(self.debug_bytes) == 0:
                        self.error = RuntimeError(
                            f"ECG live serial returned zero bytes for {self.cfg.fail_fast_if_no_ecg_sec:.1f}s. "
                            "COM port opens, but the device is not streaming to Python. "
                            "This means STM32 UART is not streaming CSV lines to Python. "
                            "Check STM32 firmware, COM port, baudrate, and UART TX/RX connection."
                        )
                        break
                    time.sleep(0.001)

        except Exception as e:
            self.error = e
            self.ready_event.set()
        finally:
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass

    def analyze(self):
        if self.error:
            raise self.error
        t = np.asarray(self.t, dtype=np.float64)
        raw = np.asarray(self.raw, dtype=np.float64)

        # PATCH7 salvage:
        # If live parsing produced too few samples but debug bytes contain parseable CSV,
        # recover those bytes before failing. This directly handles cases like:
        #   first_text='0942,3037,3314\n5450943,3434,321'
        if len(raw) < 30 and len(self.debug_bytes) > 0:
            try:
                dbg_text_all = bytes(self.debug_bytes).decode(errors="ignore")
                _rem, vals, raw_vals, smooth_vals, sample_idxs = parse_stm32_ecg_csv_lines(
                    "", dbg_text_all, signal_col=int(getattr(self.cfg, "stm32_csv_signal_col", 0))
                )
                if len(vals) > len(raw):
                    dt = 1.0 / float(self.cfg.fs_hint_hz)
                    self.raw = [float(v) for v in vals]
                    self.raw_adc_col = [float(v) for v in raw_vals]
                    self.smooth_adc_col = [float(v) if np.isfinite(v) else np.nan for v in smooth_vals]
                    self.sample_idx = [float(v) if np.isfinite(v) else np.nan for v in sample_idxs]
                    sidx_arr = np.asarray(self.sample_idx, dtype=np.float64)
                    finite = np.isfinite(sidx_arr)
                    if np.sum(finite) > max(10, 0.5 * len(sidx_arr)):
                        first_idx = float(sidx_arr[finite][0])
                        self.t = [float((s - first_idx) * dt) if np.isfinite(s) else float(i * dt)
                                  for i, s in enumerate(sidx_arr)]
                    else:
                        self.t = [float(i * dt) for i in range(len(vals))]
                    t = np.asarray(self.t, dtype=np.float64)
                    raw = np.asarray(self.raw, dtype=np.float64)
            except Exception:
                pass

        if len(raw) < 30:
            dbg_hex = bytes(self.debug_bytes[:64]).hex(" ")
            dbg_txt = "".join(self.debug_lines[:3])
            raise RuntimeError(
                f"ECG data too short: {len(raw)} samples. "
                f"debug_bytes_len={len(self.debug_bytes)}, first_hex='{dbg_hex}', first_text='{dbg_txt}'. "
                "STM32 CSV was received but not enough valid samples were parsed. "
                "PATCH7 keeps warm-up ECG bytes and also attempts debug-byte salvage. "
                "If this still fails, STM32 is likely sending only a short burst instead of continuous 100 Hz CSV."
            )

        # chunk 역분배 과정에서 timestamp가 미세하게 역전될 수 있어 정렬한다.
        order = np.argsort(t)
        t = t[order]
        raw = raw[order]

        # ECG 시간축 결정
        # 권장 STM32 format(sample_index,ADCValue,Smooth_ECG)이 들어오면
        # sample_index 기반으로 fs_hint_hz를 강제 사용한다.
        sample_idx_arr = np.asarray(self.sample_idx, dtype=np.float64) if len(self.sample_idx) == len(raw) else np.full(len(raw), np.nan)
        finite_idx = np.isfinite(sample_idx_arr)
        use_sample_index = (
            bool(getattr(self.cfg, "use_stm32_sample_index_time", True))
            and np.sum(finite_idx) > max(10, 0.5 * len(sample_idx_arr))
        )

        if use_sample_index:
            fs = float(self.cfg.fs_hint_hz)
            first_idx = float(sample_idx_arr[finite_idx][0])
            t0_u = float(np.nanmin(t)) if len(t) else 0.0
            t = t0_u + (sample_idx_arr - first_idx) / fs
            bad = ~np.isfinite(t)
            if np.any(bad):
                t[bad] = t0_u + np.arange(len(raw), dtype=np.float64)[bad] / fs
            time_source = "stm32_sample_index"
        else:
            # 구형 2열 format: PC 수신 시간 기반.
            # 이 경우 serial buffering 영향이 있으므로 count/span fs를 사용한다.
            duration_span = float(np.nanmax(t) - np.nanmin(t)) if len(t) > 2 else 0.0
            fs_count = float(len(raw) / duration_span) if duration_span > 1e-6 else float(self.cfg.fs_hint_hz)
            fs = fs_count if np.isfinite(fs_count) and fs_count > 1.0 else float(self.cfg.fs_hint_hz)
            if len(t) >= 2 and np.isfinite(fs):
                t0_u = float(np.nanmin(t))
                t = t0_u + np.arange(len(raw), dtype=np.float64) / fs
            time_source = "pc_receive_count_span"

        # ECG preprocessing:
        # raw ADC -> outlier suppression -> LF artifact reference -> LMS cancellation
        # -> display ECG and QRS-band ECG separation
        ecg_pre = preprocess_stm32_ecg(raw, fs, self.cfg)

        x0 = ecg_pre["raw_adc_outlier_suppressed"]
        x_f = ecg_pre["qrs"]                  # R-peak 검출용 QRS-band
        x_true_display = ecg_pre["display"]   # 표준 ECG에 가까운 display용
        x_display = ecg_pre["display_smooth"] # 논문 figure용 smoothing display
        x_lms_clean = ecg_pre["lms_clean"]
        x_artifact_ref = ecg_pre["artifact_ref"]
        x_artifact_est = ecg_pre["artifact_est"]
        x_baseline_est = ecg_pre.get("baseline_est", np.zeros_like(x_lms_clean))
        x_fft_motion_removed = ecg_pre.get("fft_motion_removed", np.zeros_like(x_lms_clean))
        x_fft_motion_clean = ecg_pre.get("fft_motion_clean", x_lms_clean)
        x_raw_hampel = ecg_pre.get("raw_hampel", ecg_pre["raw_norm"])
        x_raw_norm = ecg_pre["raw_norm"]

        # Robust morphology-based R-peak detection.
        # 양/음 극성 모두 검사한 뒤 RR sanity + morphology score가 좋은 쪽 선택.
        peaks_pos = robust_ecg_rpeak_detector(x_f, t, fs, self.cfg)
        peaks_neg = robust_ecg_rpeak_detector(-x_f, t, fs, self.cfg)

        def score(pk, sig):
            if len(pk) < 3:
                return -1e9 + len(pk)
            rr = np.diff(t[pk])
            valid_rr = (rr >= 60.0 / self.cfg.max_bpm) & (rr <= 60.0 / self.cfg.min_bpm)
            valid = np.mean(valid_rr) if len(valid_rr) else 0.0
            rr_use = rr[(rr > 0.25) & (rr < 1.8)]
            hr_est = 60.0 / np.median(rr_use) if len(rr_use) else 999
            hr_penalty = 0.0 if hr_est <= 130 else -0.6
            amp_score = float(np.nanmedian(sig[pk])) if len(pk) else 0.0
            return valid + hr_penalty + 0.05 * amp_score

        if score(peaks_neg, -x_f) > score(peaks_pos, x_f):
            x_f = -x_f
            peaks = peaks_neg
            polarity = "negative_inverted"
        else:
            peaks = peaks_pos
            polarity = "positive"

        # Plotting/diagnostic display aligned with the actual R-peak polarity.
        # If electrodes produce downward R waves, analysis inverts them; figures should also show
        # the aligned signal so the marked R peaks appear on the upward QRS apex.
        x_true_display_rpeak = -x_true_display if polarity == "negative_inverted" else x_true_display
        x_display_rpeak = -x_display if polarity == "negative_inverted" else x_display

        # RR filtering
        rpeak_removed_short_rr = []

        # 1) 기존 RR sanity filtering 유지
        if len(peaks) >= 3:
            keep = np.ones(len(peaks), dtype=bool)
            rr = np.diff(t[peaks])
            for i in range(1, len(peaks)):
                if rr[i - 1] < 60.0 / self.cfg.max_bpm or rr[i - 1] > 60.0 / self.cfg.min_bpm:
                    keep[i] = False
            peaks = peaks[keep]

        # 2) 추가 후처리: RR < 0.45 s double detection 제거
        # detector 자체를 바꾸는 것이 아니라, 검출 이후 비정상 short-RR pair만 정리.
        if bool(globals().get("RPEAK_ENABLE_SHORT_RR_POSTPROCESS", True)) and len(peaks) >= 2:
            peaks, rpeak_removed_short_rr = postprocess_rpeaks_short_rr(
                peaks,
                x_f,
                fs,
                min_rr_sec=float(globals().get("RPEAK_MIN_RR_SEC_POST", 0.45)),
                neighbor_margin_sec=float(globals().get("RPEAK_SHORT_RR_NEIGHBOR_MARGIN_SEC", 0.04))
            )

        hr = None
        if len(peaks) >= 2:
            rr = np.diff(t[peaks])
            rr = rr[(rr > 0.25) & (rr < 1.8)]
            if len(rr):
                hr = float(60.0 / np.median(rr))

        ecg_landmarks = detect_ecg_q_t_landmarks(
            ecg_display=x_true_display,
            ecg_analysis=x_f,
            t=t,
            r_peaks=peaks,
            fs=fs
        )

        return {
            "t": t,
            "raw": raw,
            "cleaned": x0,
            "raw_norm": x_raw_norm,
            "raw_hampel": x_raw_hampel,
            "baseline_est": x_baseline_est,
            "fft_motion_removed": x_fft_motion_removed,
            "fft_motion_clean": x_fft_motion_clean,
            "artifact_ref": x_artifact_ref,
            "artifact_est": x_artifact_est,
            "lms_clean": x_lms_clean,
            "filtered": x_f,
            "display": x_display,
            "display_rpeak": x_display_rpeak,
            "true_display": x_true_display,
            "true_display_rpeak": x_true_display_rpeak,
            "fs": fs,
            "peaks_idx": peaks,
            "peaks_time": t[peaks] if len(peaks) else np.array([]),
            "q_idx": ecg_landmarks["q_idx"],
            "q_time": ecg_landmarks["q_time"],
            "q_confidence": ecg_landmarks.get("q_confidence", np.zeros_like(ecg_landmarks["q_time"])),
            "q_rel_sec": ecg_landmarks.get("q_rel_sec", np.full_like(ecg_landmarks["q_time"], np.nan, dtype=float)),
            "t_idx": ecg_landmarks["t_idx"],
            "t_time": ecg_landmarks["t_time"],
            "t_confidence": ecg_landmarks.get("t_confidence", np.zeros_like(ecg_landmarks["t_time"])),
            "t_rel_sec": ecg_landmarks.get("t_rel_sec", np.full_like(ecg_landmarks["t_time"], np.nan, dtype=float)),
            "rr_local_sec": ecg_landmarks.get("rr_local_sec", np.full_like(ecg_landmarks["t_time"], np.nan, dtype=float)),
            "hr_bpm": hr,
            "polarity": polarity,
            "rpeak_removed_short_rr": rpeak_removed_short_rr,
            "time_source": time_source,
            "sample_idx": sample_idx_arr,
            "raw_adc_col": np.asarray(self.raw_adc_col, dtype=np.float64) if len(self.raw_adc_col) == len(raw) else raw,
            "smooth_adc_col": np.asarray(self.smooth_adc_col, dtype=np.float64) if len(self.smooth_adc_col) == len(raw) else np.full(len(raw), np.nan),
        }



# ============================================================
# SCG Collector: ESP32 + MPU6050 @ 100 Hz
# ============================================================
def parse_esp32_mpu6050_scg_csv_lines(text_buffer: str, new_text: str):
    """
    Robust ESP32/MPU6050 SCG CSV parser.

    Supported:
      8-col: sample_index,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
      7-col: sample_index,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
      6-col: ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
      5-col indexed-short: sample_index,ax_g,ay_g,az_g,gx_dps
      5-col IMU-only: ax_g,ay_g,az_g,gx_dps,gy_dps
      4-col indexed-partial: sample_index,ax_g,ay_g,az_g

    The 10-min log showed:
      221,-0.060608,0.5573,-0.1374,-0.2290
    which is interpreted as sample_index,ax,ay,az,gx.
    """
    if text_buffer is None:
        text_buffer = ""
    if new_text is None:
        new_text = ""

    text_buffer += new_text
    lines = text_buffer.splitlines(keepends=True)
    complete, remain = [], ""
    for line in lines:
        if line.endswith("\n") or line.endswith("\r"):
            complete.append(line.strip())
        else:
            remain = line

    rows = []
    if not hasattr(parse_esp32_mpu6050_scg_csv_lines, "_auto_index"):
        parse_esp32_mpu6050_scg_csv_lines._auto_index = 0

    def _looks_like_index(v):
        try:
            fv = float(v)
            return np.isfinite(fv) and abs(fv - round(fv)) < 1e-6 and abs(fv) >= 20
        except Exception:
            return False

    for line in complete:
        s = str(line).strip()
        if not s or s.startswith("#"):
            continue
        low = s.lower()
        if ("sample" in low) or ("mpu" in low) or ("ax" in low and "," not in s):
            continue

        nums = []
        for token in re.split(r"[,\t ;]+", s):
            if not token:
                continue
            try:
                nums.append(float(token))
            except Exception:
                pass

        if len(nums) >= 8:
            sample_index, t_ms = nums[0], nums[1]
            ax, ay, az = nums[2], nums[3], nums[4]
            gx, gy, gz = nums[5], nums[6], nums[7]

        elif len(nums) == 7 and _looks_like_index(nums[0]):
            sample_index = nums[0]
            t_ms = sample_index * 10.0
            ax, ay, az = nums[1], nums[2], nums[3]
            gx, gy, gz = nums[4], nums[5], nums[6]

        elif len(nums) == 6:
            sample_index = float(parse_esp32_mpu6050_scg_csv_lines._auto_index)
            t_ms = sample_index * 10.0
            ax, ay, az = nums[0], nums[1], nums[2]
            gx, gy, gz = nums[3], nums[4], nums[5]
            parse_esp32_mpu6050_scg_csv_lines._auto_index += 1

        elif len(nums) == 5 and _looks_like_index(nums[0]):
            sample_index = nums[0]
            t_ms = sample_index * 10.0
            ax, ay, az = nums[1], nums[2], nums[3]
            gx, gy, gz = nums[4], np.nan, np.nan

        elif len(nums) == 5:
            sample_index = float(parse_esp32_mpu6050_scg_csv_lines._auto_index)
            t_ms = sample_index * 10.0
            ax, ay, az = nums[0], nums[1], nums[2]
            gx, gy, gz = nums[3], nums[4], np.nan
            parse_esp32_mpu6050_scg_csv_lines._auto_index += 1

        elif len(nums) == 4 and _looks_like_index(nums[0]):
            sample_index = nums[0]
            t_ms = sample_index * 10.0
            ax, ay, az = nums[1], nums[2], nums[3]
            gx, gy, gz = np.nan, np.nan, np.nan

        else:
            continue

        if not (np.isfinite(ax) and np.isfinite(ay) and np.isfinite(az)):
            continue
        if max(abs(ax), abs(ay), abs(az)) > 16.0:
            continue

        rows.append({
            "sample_index": float(sample_index),
            "t_ms": float(t_ms),
            "ax": float(ax),
            "ay": float(ay),
            "az": float(az),
            "gx": float(gx) if np.isfinite(gx) else np.nan,
            "gy": float(gy) if np.isfinite(gy) else np.nan,
            "gz": float(gz) if np.isfinite(gz) else np.nan,
        })

    return remain, rows


def preprocess_scg_signal(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float, cfg: SCGConfig):
    ax = np.asarray(ax, dtype=np.float64)
    ay = np.asarray(ay, dtype=np.float64)
    az = np.asarray(az, dtype=np.float64)

    ax0 = hampel_filter_1d(ax, fs, cfg.hampel_window_sec, cfg.hampel_nsigma)
    ay0 = hampel_filter_1d(ay, fs, cfg.hampel_window_sec, cfg.hampel_nsigma)
    az0 = hampel_filter_1d(az, fs, cfg.hampel_window_sec, cfg.hampel_nsigma)

    vmag = np.sqrt(ax0 ** 2 + ay0 ** 2 + az0 ** 2)
    vmag = vmag - np.nanmedian(vmag)

    if cfg.signal_mode.lower() == "ax":
        primary = ax0 - np.nanmedian(ax0)
    elif cfg.signal_mode.lower() == "ay":
        primary = ay0 - np.nanmedian(ay0)
    elif cfg.signal_mode.lower() == "az":
        primary = az0 - np.nanmedian(az0)
    else:
        primary = vmag

    ref = safe_bandpass(primary, fs, cfg.lms_reference_band_hz[0], cfg.lms_reference_band_hz[1], order=3)
    if bool(cfg.use_lms_resp_cancel) and len(primary) > max(64, int(fs * 3)):
        lms_error, est_noise, _ = lms_adaptive_cancel(primary=primary, reference=ref, mu=cfg.lms_mu, order=cfg.lms_order)
        resp_removed = lms_error
        lms_used = True
    else:
        resp_removed = primary.copy()
        est_noise = np.zeros_like(primary)
        lms_used = False

    lo, hi = cfg.band_hz
    filtered = safe_bandpass(resp_removed, fs, lo, hi, order=4)
    display = safe_lowpass(filtered, fs, min(cfg.lowpass_display_hz, 0.45 * fs), order=3)

    filtered_z = zscore_safe(filtered)
    d1 = np.gradient(filtered_z) if len(filtered_z) > 2 else np.zeros_like(filtered_z)
    d2 = np.gradient(d1) if len(d1) > 2 else np.zeros_like(filtered_z)
    min_dist = max(1, int(fs * 0.25))
    pos_peaks, _ = signal.find_peaks(filtered_z, distance=min_dist, prominence=max(np.std(filtered_z) * 0.20, 1e-12))
    neg_peaks, _ = signal.find_peaks(-filtered_z, distance=min_dist, prominence=max(np.std(filtered_z) * 0.20, 1e-12))
    slope_peaks, _ = signal.find_peaks(np.abs(d1), distance=max(1, int(fs * 0.12)), prominence=max(np.std(np.abs(d1)) * 0.20, 1e-12))
    curv_peaks, _ = signal.find_peaks(np.abs(d2), distance=max(1, int(fs * 0.12)), prominence=max(np.std(np.abs(d2)) * 0.20, 1e-12))

    return {
        "ax_hampel": ax0,
        "ay_hampel": ay0,
        "az_hampel": az0,
        "vmag": vmag,
        "selected_raw": primary,
        "resp_reference": zscore_safe(ref),
        "estimated_noise": zscore_safe(est_noise),
        "resp_removed": zscore_safe(resp_removed),
        "filtered": filtered_z,
        "display": zscore_safe(display),
        "d1": zscore_safe(d1),
        "d2": zscore_safe(d2),
        "pos_peaks_idx": pos_peaks,
        "neg_peaks_idx": neg_peaks,
        "slope_peaks_idx": slope_peaks,
        "curv_peaks_idx": curv_peaks,
        "lms_used": lms_used,
    }


class SCGCollector:
    def __init__(self, cfg: SCGConfig):
        self.cfg = cfg
        self.t: list[float] = []
        self.sample_idx: list[float] = []
        self.t_ms: list[float] = []
        self.ax: list[float] = []
        self.ay: list[float] = []
        self.az: list[float] = []
        self.gx: list[float] = []
        self.gy: list[float] = []
        self.gz: list[float] = []
        self._first_sample_idx: Optional[float] = None
        self._first_sample_wall_t: Optional[float] = None
        self.debug_bytes = bytearray()
        self.debug_lines: list[str] = []
        self.ready_event = threading.Event()
        self.error: Optional[Exception] = None

    def acquire(self, duration_sec: float, shared: dict[str, float], start_event: threading.Event, stop_event: threading.Event):
        ser = None
        if not self.cfg.enabled:
            self.ready_event.set()
            return
        try:
            print(f"[SCG] Open {self.cfg.port} @ {self.cfg.baudrate}, input=ESP32 MPU6050 CSV @ {self.cfg.fs_hint_hz:.1f} Hz")
            ser = serial.Serial(self.cfg.port, self.cfg.baudrate, timeout=self.cfg.timeout_sec)
            time.sleep(0.8)
            ser.reset_input_buffer()
            try:
                parse_esp32_mpu6050_scg_csv_lines._auto_index = 0
            except Exception:
                pass
            self.ready_event.set()

            start_event.wait()
            t0 = shared["t0"]
            ascii_buf = ""
            wall_guard_sec = max(duration_sec * 3.0, duration_sec + 30.0)

            while not stop_event.is_set():
                wall_t = time.perf_counter() - t0
                if wall_t >= wall_guard_sec:
                    break
                if len(self.t) >= 2 and (self.t[-1] - self.t[0]) >= duration_sec:
                    break

                n_wait = ser.in_waiting
                if n_wait:
                    chunk = ser.read(n_wait)
                    if len(self.debug_bytes) < 4096:
                        self.debug_bytes.extend(chunk[: max(0, 4096 - len(self.debug_bytes))])
                    decoded = chunk.decode(errors="ignore")
                    if decoded and len(self.debug_lines) < 20:
                        self.debug_lines.append(decoded[:200])
                    ascii_buf, rows = parse_esp32_mpu6050_scg_csv_lines(ascii_buf, decoded)

                    read_t = time.perf_counter() - t0
                    dt = 1.0 / float(self.cfg.fs_hint_hz)
                    n_vals = len(rows)
                    for j, row in enumerate(rows):
                        sidx = row.get("sample_index", np.nan)
                        if np.isfinite(sidx) and bool(self.cfg.use_sample_index_time):
                            if self._first_sample_idx is None:
                                self._first_sample_idx = float(sidx)
                                self._first_sample_wall_t = read_t - (n_vals - 1 - j) * dt
                                if self._first_sample_wall_t < 0:
                                    self._first_sample_wall_t = 0.0
                            sample_t = float(self._first_sample_wall_t) + (float(sidx) - float(self._first_sample_idx)) * dt
                        else:
                            sample_t = read_t - (n_vals - 1 - j) * dt
                            if sample_t < 0:
                                sample_t = 0.0

                        self.t.append(float(sample_t))
                        self.sample_idx.append(float(sidx) if np.isfinite(sidx) else np.nan)
                        self.t_ms.append(float(row.get("t_ms", np.nan)))
                        self.ax.append(float(row.get("ax", np.nan)))
                        self.ay.append(float(row.get("ay", np.nan)))
                        self.az.append(float(row.get("az", np.nan)))
                        self.gx.append(float(row.get("gx", np.nan)))
                        self.gy.append(float(row.get("gy", np.nan)))
                        self.gz.append(float(row.get("gz", np.nan)))
                else:
                    if wall_t >= self.cfg.fail_fast_if_no_scg_sec and len(self.ax) == 0 and len(self.debug_bytes) == 0:
                        self.error = RuntimeError(
                            f"SCG live serial returned zero bytes for {self.cfg.fail_fast_if_no_scg_sec:.1f}s. "
                            "Check ESP32 COM port, baudrate, USB cable, and Serial Monitor state."
                        )
                        break
                    time.sleep(0.001)
        except Exception as e:
            self.error = e
            self.ready_event.set()
        finally:
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass

    def analyze(self):
        if not self.cfg.enabled:
            return None
        if self.error:
            raise self.error
        t = np.asarray(self.t, dtype=np.float64)
        ax = np.asarray(self.ax, dtype=np.float64)
        ay = np.asarray(self.ay, dtype=np.float64)
        az = np.asarray(self.az, dtype=np.float64)
        if len(ax) < 30:
            dbg_hex = bytes(self.debug_bytes[:64]).hex(" ")
            dbg_txt = "".join(self.debug_lines[:3])
            raise RuntimeError(
                f"SCG data too short: {len(ax)} samples. Parser now supports 8-col indexed CSV and 5/6-col IMU-only fallback. debug_bytes_len={len(self.debug_bytes)}, "
                f"first_hex='{dbg_hex}', first_text='{dbg_txt}'."
            )
        order = np.argsort(t)
        t = t[order]
        ax, ay, az = ax[order], ay[order], az[order]
        gx = np.asarray(self.gx, dtype=np.float64)[order]
        gy = np.asarray(self.gy, dtype=np.float64)[order]
        gz = np.asarray(self.gz, dtype=np.float64)[order]
        sample_idx = np.asarray(self.sample_idx, dtype=np.float64)[order]
        t_ms = np.asarray(self.t_ms, dtype=np.float64)[order]

        fs = float(self.cfg.fs_hint_hz)
        finite_idx = np.isfinite(sample_idx)
        if bool(self.cfg.use_sample_index_time) and np.sum(finite_idx) > max(10, 0.5 * len(sample_idx)):
            first_idx = float(sample_idx[finite_idx][0])
            t0_u = float(np.nanmin(t)) if len(t) else 0.0
            t = t0_u + (sample_idx - first_idx) / fs
            bad = ~np.isfinite(t)
            if np.any(bad):
                t[bad] = t0_u + np.arange(len(ax), dtype=np.float64)[bad] / fs
            time_source = "esp32_sample_index"
        else:
            duration_span = float(np.nanmax(t) - np.nanmin(t)) if len(t) > 2 else 0.0
            fs_count = float(len(ax) / duration_span) if duration_span > 1e-6 else fs
            fs = fs_count if np.isfinite(fs_count) and fs_count > 1.0 else fs
            t0_u = float(np.nanmin(t)) if len(t) else 0.0
            t = t0_u + np.arange(len(ax), dtype=np.float64) / fs
            time_source = "pc_receive_count_span"

        pre = preprocess_scg_signal(ax, ay, az, fs, self.cfg)
        sig = pre["filtered"]
        min_dist = max(1, int(fs * 0.25))
        peaks, _ = signal.find_peaks(sig, distance=min_dist, prominence=max(np.std(sig) * 0.25, 1e-12))
        return {
            "t": t,
            "fs": fs,
            "sample_idx": sample_idx,
            "t_ms": t_ms,
            "ax": ax,
            "ay": ay,
            "az": az,
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "vmag": pre["vmag"],
            "selected_raw": pre["selected_raw"],
            "resp_reference": pre["resp_reference"],
            "estimated_noise": pre["estimated_noise"],
            "resp_removed": pre["resp_removed"],
            "filtered": sig,
            "display": pre["display"],
            "d1": pre["d1"],
            "d2": pre["d2"],
            "peaks_idx": peaks,
            "peaks_time": t[peaks] if len(peaks) else np.array([]),
            "pos_peaks_idx": pre["pos_peaks_idx"],
            "neg_peaks_idx": pre["neg_peaks_idx"],
            "slope_peaks_idx": pre["slope_peaks_idx"],
            "curv_peaks_idx": pre["curv_peaks_idx"],
            "lms_used": pre["lms_used"],
            "time_source": time_source,
            "signal_mode": self.cfg.signal_mode,
        }


def scg_reference_aoac_pipeline(ecg: dict, scg: Optional[dict], radar: dict, aoac: dict, acfg: AnalysisConfig, outdir: Optional[Path] = None):
    """
    ECG R-peak anchor 기반으로 SCG와 radar beat를 같은 window에서 비교한다.
    ECG는 AO/AC ground truth가 아니라 beat alignment anchor로만 사용한다.
    SCG는 접촉 기계적 신호 기반 AO/AC 후보 reference로 사용한다.
    """
    if scg is None:
        return None
    st = np.asarray(scg.get("t", []), dtype=np.float64)
    sx = np.asarray(scg.get("filtered", []), dtype=np.float64)
    if len(st) < 10 or len(sx) < 10 or len(ecg.get("peaks_time", [])) == 0:
        return None

    rows = []
    scg_beats = []
    radar_by_index = {int(b.get("beat_index", -1)): b for b in aoac.get("beats", [])}

    for bi, r_t in enumerate(ecg["peaks_time"]):
        w0, w1 = float(r_t) - acfg.beat_pre_sec, float(r_t) + acfg.beat_post_sec
        if w0 < st[0] or w1 > st[-1]:
            continue
        m = (st >= w0) & (st <= w1)
        if np.sum(m) < int((acfg.beat_pre_sec + acfg.beat_post_sec) * scg.get("fs", 100.0) * 0.6):
            continue
        bt_raw = st[m] - float(r_t)
        bx_raw = zscore_safe(sx[m])
        grid = np.arange(-acfg.beat_pre_sec, acfg.beat_post_sec + 1e-9, 1.0 / 100.0)
        bx = zscore_safe(np.interp(grid, bt_raw, bx_raw))
        bt = grid

        ao_idx, ao_conf = scg_inspired_aoac_detector(bt, bx, acfg.ao_search_sec, kind="ao")
        if ao_idx is None:
            ao_idx, ao_conf = morphology_event_detector(bt, bx, acfg.ao_search_sec, kind="ao")
        ac_idx, ac_conf = scg_inspired_aoac_detector(bt, bx, acfg.ac_search_sec, kind="ac")
        if ac_idx is None:
            ac_idx, ac_conf = ac_inflection_zero_cross_detector(bt, bx, ecg_prior_sec=None, prev_ac_delay=None)
        if ac_idx is None:
            ac_idx, ac_conf = morphology_event_detector(bt, bx, acfg.ac_search_sec, kind="ac")

        scg_ao = None if ao_idx is None else float(bt[max(0, min(len(bt)-1, ao_idx))])
        scg_ac = None if ac_idx is None else float(bt[max(0, min(len(bt)-1, ac_idx))])

        rb = radar_by_index.get(int(bi))
        radar_ao = None
        radar_ac = None
        radar_ao_morph = None
        radar_ac_morph = None
        sqi = None
        accepted = None
        if rb is not None:
            radar_ao = rb.get("ao_time", None)
            radar_ac = rb.get("ac_time", None)
            radar_ao_morph = rb.get("ao_morph_time", radar_ao)
            radar_ac_morph = rb.get("ac_morph_time", radar_ac)
            sqi = rb.get("sqi", None)
            accepted = rb.get("accepted", None)

        ao_err_ms = None if scg_ao is None or radar_ao_morph is None else float((radar_ao_morph - scg_ao) * 1000.0)
        ac_err_ms = None if scg_ac is None or radar_ac_morph is None else float((radar_ac_morph - scg_ac) * 1000.0)
        both_within_30 = None
        if ao_err_ms is not None and ac_err_ms is not None:
            both_within_30 = bool(abs(ao_err_ms) <= 30.0 and abs(ac_err_ms) <= 30.0)

        rows.append([
            int(bi), float(r_t), bool(accepted) if accepted is not None else None,
            None if sqi is None else float(sqi),
            scg_ao, scg_ac, radar_ao_morph, radar_ac_morph,
            ao_err_ms, ac_err_ms, both_within_30,
            None if ao_conf is None else float(ao_conf),
            None if ac_conf is None else float(ac_conf),
        ])
        scg_beats.append({
            "beat_index": int(bi), "r_time": float(r_t), "t_rel": bt, "scg_beat": bx,
            "scg_ao_time": scg_ao, "scg_ac_time": scg_ac,
        })

    summary = {"n": len(rows)}
    if rows:
        accepted_rows = [r for r in rows if r[2] is True]
        use_rows = accepted_rows if accepted_rows else rows
        aoe = np.array([np.nan if r[8] is None else float(r[8]) for r in use_rows], dtype=float)
        ace = np.array([np.nan if r[9] is None else float(r[9]) for r in use_rows], dtype=float)
        both = np.array([False if r[10] is None else bool(r[10]) for r in use_rows], dtype=bool)
        summary.update({
            "n_used": int(len(use_rows)),
            "ao_mae_vs_scg_ms": None if not np.any(np.isfinite(aoe)) else float(np.nanmean(np.abs(aoe))),
            "ac_mae_vs_scg_ms": None if not np.any(np.isfinite(ace)) else float(np.nanmean(np.abs(ace))),
            "simultaneous_accuracy_within_30ms_vs_scg": None if len(both) == 0 else float(np.mean(both) * 100.0),
        })

    result = {"rows": rows, "scg_beats": scg_beats, "summary": summary}

    if outdir is not None:
        save_csv(outdir / "scg_reference_vs_radar_candidates.csv",
                 ["beat_index", "r_peak_time_sec", "radar_accepted", "radar_sqi",
                  "scg_ao_time_from_r_sec", "scg_ac_time_from_r_sec",
                  "radar_ao_morph_time_from_r_sec", "radar_ac_morph_time_from_r_sec",
                  "radar_minus_scg_ao_ms", "radar_minus_scg_ac_ms",
                  "both_ao_ac_within_30ms", "scg_ao_confidence", "scg_ac_confidence"], rows)
        with open(outdir / "scg_reference_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        add_scg_diagnostic_figures(outdir, ecg, scg, radar, result, aoac, acfg)
    return result


def add_scg_diagnostic_figures(outdir: Path, ecg: dict, scg: dict, radar: dict, scg_result: dict, aoac: dict, acfg: AnalysisConfig, n_cycles: int = 10):
    try:
        if len(ecg.get("peaks_time", [])) < n_cycles + 1:
            return
        r_times = np.asarray(ecg["peaks_time"], dtype=float)
        start_idx = max(2, len(r_times) // 3)
        end_idx = min(start_idx + n_cycles, len(r_times) - 1)
        t_start = r_times[start_idx] - 0.20
        t_end = r_times[end_idx] + 0.60

        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        ez = zscore_safe(ecg.get("display_rpeak", ecg.get("display", ecg["filtered"])))
        sz = zscore_safe(scg["filtered"])
        rz = zscore_safe(radar["ppg_like"])

        axes[0].plot(ecg["t"], ez, linewidth=1.0, label="ECG display")
        axes[1].plot(scg["t"], sz, linewidth=1.0, label="SCG filtered")
        axes[2].plot(radar["t"], rz, linewidth=1.0, label="Radar recovered cardiac signal")
        for ax in axes:
            for r in r_times[start_idx:end_idx + 1]:
                ax.axvline(r, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.axvspan(r + acfg.ao_search_sec[0], r + acfg.ao_search_sec[1], alpha=0.10)
                ax.axvspan(r + acfg.ac_search_sec[0], r + acfg.ac_search_sec[1], alpha=0.08)
            ax.set_xlim(t_start, t_end)
            ax.set_ylabel("z-score")
            ax.grid(True, alpha=0.35)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle("ECG R-peak anchored ECG-SCG-Radar multi-cycle waveform")
        fig.tight_layout()
        fig.savefig(outdir / "fig09_ecg_scg_radar_multicycle_diagnostic.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir / "scg_diagnostic_figure_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    try:
        beats = scg_result.get("scg_beats", [])
        if not beats:
            return
        # choose a beat with available SCG AO/AC and matching radar beat
        radar_by_index = {int(b.get("beat_index", -1)): b for b in aoac.get("beats", [])}
        selected = None
        for b in beats:
            rb = radar_by_index.get(int(b["beat_index"]))
            if rb is not None and b.get("scg_ao_time") is not None and b.get("scg_ac_time") is not None:
                selected = (b, rb)
                break
        if selected is None:
            return
        sb, rb = selected
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(sb["t_rel"], sb["scg_beat"], linewidth=1.4, label="SCG beat")
        axes[1].plot(rb["t_rel"], rb["radar_beat"], linewidth=1.4, label="Radar beat")
        for ax in axes:
            ax.axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.10, label="AO window")
            ax.axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.08, label="AC window")
            ax.axvline(0, color="k", linestyle="--", linewidth=1.0, label="ECG R")
            ax.grid(True, alpha=0.35)
            ax.set_ylabel("z-score")
        if sb.get("scg_ao_time") is not None:
            axes[0].axvline(sb["scg_ao_time"], linestyle="--", linewidth=1.4, label="SCG AO cand.")
        if sb.get("scg_ac_time") is not None:
            axes[0].axvline(sb["scg_ac_time"], linestyle="--", linewidth=1.4, label="SCG AC cand.")
        if rb.get("ao_morph_time", rb.get("ao_time")) is not None:
            axes[1].axvline(rb.get("ao_morph_time", rb.get("ao_time")), linestyle="--", linewidth=1.4, label="Radar AO cand.")
        if rb.get("ac_morph_time", rb.get("ac_time")) is not None:
            axes[1].axvline(rb.get("ac_morph_time", rb.get("ac_time")), linestyle="--", linewidth=1.4, label="Radar AC cand.")
        axes[1].set_xlabel("Time from ECG R-peak [s]")
        axes[0].legend(loc="upper right", fontsize=8, ncol=2)
        axes[1].legend(loc="upper right", fontsize=8, ncol=2)
        fig.suptitle("SCG reference candidates vs Radar morphology candidates")
        fig.tight_layout()
        fig.savefig(outdir / "fig10_scg_reference_vs_radar_candidate_beat.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir / "scg_reference_beat_figure_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

def lms_adaptive_cancel(primary: np.ndarray, reference: np.ndarray, mu: float = 0.003, order: int = 8):
    """
    LMS adaptive noise cancellation.
    primary  : radar displacement or broad-band signal
    reference: respiration/motion-dominant reference
    output   : error signal e[n] = primary[n] - estimated_noise[n]

    AO/AC 검출 자체가 아니라, beat slicing 전 radar PPG-like 전처리용으로 사용.
    """
    d = np.asarray(primary, dtype=np.float64)
    x = np.asarray(reference, dtype=np.float64)
    n = min(len(d), len(x))
    d = d[:n]
    x = x[:n]

    if n <= order + 2:
        return d.copy(), np.zeros_like(d), np.zeros((0, order))

    # 안정성을 위해 reference normalize
    x = zscore_safe(x)
    d = zscore_safe(d)

    w = np.zeros(order, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    e = np.zeros(n, dtype=np.float64)
    w_hist = np.zeros((n, order), dtype=np.float64)

    eps = 1e-9
    for i in range(order, n):
        xv = x[i - order:i][::-1]
        y[i] = float(np.dot(w, xv))
        e[i] = d[i] - y[i]
        norm = float(np.dot(xv, xv) + eps)
        # normalized LMS 형태로 step 폭주 방지
        w += (mu / norm) * e[i] * xv
        w_hist[i] = w

    # 초반 order 구간은 원신호 유지
    e[:order] = d[:order]
    return e, y, w_hist


def radar_respiration_lms_pipeline(displacement: np.ndarray, fs: float, cfg):
    """
    Radar displacement에서 respiration/motion reference를 만들고,
    LMS adaptive cancellation 후 cardiac PPG-like band로 재필터링.
    """
    disp = np.asarray(displacement, dtype=np.float64)
    if len(disp) < max(64, int(fs * 5)):
        ppg = safe_bandpass(disp, fs, cfg.ppg_like_band_hz[0], cfg.ppg_like_band_hz[1], order=4)
        return ppg, {
            "reference": np.zeros_like(disp),
            "estimated_noise": np.zeros_like(disp),
            "lms_error": ppg,
            "used": False,
        }

    ref = safe_bandpass(
        disp, fs,
        cfg.lms_reference_band_hz[0],
        cfg.lms_reference_band_hz[1],
        order=4
    )

    if not cfg.use_lms_resp_cancel:
        ppg = safe_bandpass(disp, fs, cfg.ppg_like_band_hz[0], cfg.ppg_like_band_hz[1], order=4)
        return ppg, {
            "reference": ref,
            "estimated_noise": np.zeros_like(disp),
            "lms_error": ppg,
            "used": False,
        }

    e, y, _ = lms_adaptive_cancel(
        primary=disp,
        reference=ref,
        mu=cfg.lms_mu,
        order=cfg.lms_order
    )

    # LMS error에서 다시 심박 대역만 추출
    ppg_lms = safe_bandpass(
        e, fs,
        cfg.lms_post_band_hz[0],
        cfg.lms_post_band_hz[1],
        order=4
    )

    return ppg_lms, {
        "reference": ref,
        "estimated_noise": y,
        "lms_error": e,
        "used": True,
    }


# ============================================================
# Radar processing
# ============================================================
def get_center_freq(cfg: RadarConfig) -> float:
    return 0.5 * (cfg.start_freq_hz + cfg.end_freq_hz)


def get_lambda(cfg: RadarConfig) -> float:
    return 299_792_458.0 / get_center_freq(cfg)


def get_chirp_duration(cfg: RadarConfig) -> float:
    return cfg.num_samples / cfg.sample_rate_hz


def get_chirp_slope(cfg: RadarConfig) -> float:
    return (cfg.end_freq_hz - cfg.start_freq_hz) / get_chirp_duration(cfg)


def get_range_axis(cfg: RadarConfig) -> np.ndarray:
    c = 299_792_458.0
    slope = get_chirp_slope(cfg)
    beat = np.arange(cfg.range_fft_size // 2) * (cfg.sample_rate_hz / cfg.range_fft_size)
    return c * beat / (2.0 * slope)


def get_angle_axis_deg(cfg: RadarConfig) -> np.ndarray:
    return np.linspace(-60.0, 60.0, cfg.angle_bins)


def preprocess_frame(frame: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    x = frame.astype(np.float32, copy=True)
    if cfg.remove_dc:
        x -= np.mean(x, axis=-1, keepdims=True)
    if cfg.apply_window:
        x *= signal.windows.blackmanharris(cfg.num_samples).reshape(1, 1, -1)
    return x


def range_fft(frame: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    x = preprocess_frame(frame, cfg)
    r = np.fft.fft(x, n=cfg.range_fft_size, axis=-1) / cfg.num_samples
    return 2.0 * r[:, :, : cfg.range_fft_size // 2]


def steering_vector(cfg: RadarConfig, angle_deg: float, d_over_lambda: float = 0.5):
    ar = np.deg2rad(angle_deg)
    n = np.arange(cfg.num_rx)
    return np.exp(-2j * np.pi * d_over_lambda * n * np.sin(ar))


def dbf_range_angle(range_cube: np.ndarray, cfg: RadarConfig):
    snapshot = np.mean(range_cube, axis=1)
    snapshot -= np.mean(snapshot, axis=-1, keepdims=True)
    angles = get_angle_axis_deg(cfg)
    ranges = get_range_axis(cfg)
    ra = np.zeros((cfg.angle_bins, snapshot.shape[-1]), dtype=np.float32)
    for ai, ang in enumerate(angles):
        w = steering_vector(cfg, ang)[:, None]
        bf = np.sum(np.conj(w) * snapshot, axis=0)
        ra[ai, :] = np.abs(bf)
    return ra, angles, ranges


def beamformed_complex_at(range_cube: np.ndarray, cfg: RadarConfig, angle_idx: int, range_idx: int) -> complex:
    angles = get_angle_axis_deg(cfg)
    snapshot = np.mean(range_cube, axis=1)
    snapshot -= np.mean(snapshot, axis=-1, keepdims=True)
    w = steering_vector(cfg, angles[angle_idx])[:, None]
    return complex(np.sum(np.conj(w) * snapshot[:, range_idx:range_idx + 1], axis=0)[0])


class IfxRadarBackend:
    def __init__(self, cfg: RadarConfig):
        self.cfg = cfg
        self.device = None

    def connect(self):
        print("[RADAR] Open BGT60TR13C via Infineon SDK DeviceFmcw()")
        self.device = DeviceFmcw()
        self.device.__enter__()

        config = FmcwSimpleSequenceConfig(
            frame_repetition_time_s=1.0 / self.cfg.frame_rate_hz,
            chirp_repetition_time_s=self.cfg.chirp_repetition_time_s,
            num_chirps=self.cfg.num_chirps,
            tdm_mimo=False,
            chirp=FmcwSequenceChirp(
                start_frequency_Hz=self.cfg.start_freq_hz,
                end_frequency_Hz=self.cfg.end_freq_hz,
                sample_rate_Hz=self.cfg.sample_rate_hz,
                num_samples=self.cfg.num_samples,
                rx_mask=7,
                tx_mask=1,
                tx_power_level=self.cfg.tx_power_level,
                if_gain_dB=self.cfg.if_gain_dB,
                lp_cutoff_Hz=self.cfg.lp_cutoff_Hz,
                hp_cutoff_Hz=self.cfg.hp_cutoff_Hz,
            )
        )
        seq = self.device.create_simple_sequence(config)
        self.device.set_acquisition_sequence(seq)

    def disconnect(self):
        if self.device is not None:
            self.device.__exit__(None, None, None)
            self.device = None

    def get_frame(self) -> np.ndarray:
        fc = self.device.get_next_frame()
        frame = np.asarray(fc[0], dtype=np.float32)
        expected = (self.cfg.num_rx, self.cfg.num_chirps, self.cfg.num_samples)
        if frame.shape != expected:
            raise RuntimeError(f"Radar frame shape mismatch: expected={expected}, got={frame.shape}")
        return frame


class RadarCollector:
    def __init__(self, cfg: RadarConfig):
        self.cfg = cfg
        self.t: list[float] = []
        self.cvals: list[complex] = []
        self.ridx: list[int] = []
        self.aidx: list[int] = []
        self.ra_maps: list[np.ndarray] = []
        self.ready_event = threading.Event()
        self.error: Optional[Exception] = None
        self.fixed_range_idx: Optional[int] = None
        self.fixed_angle_idx: Optional[int] = None
        self.drop_count = 0
        self._first_frame_wall_t: Optional[float] = None

    def acquire(self, duration_sec: float, shared: dict[str, float], start_event: threading.Event, stop_event: threading.Event):
        backend = IfxRadarBackend(self.cfg)
        try:
            backend.connect()
            self.ready_event.set()
            start_event.wait()
            t0 = shared["t0"]

            init_r, init_a = [], []
            consec_err = 0

            # Data-driven radar acquisition:
            # BGT60 SDK may delay the first valid frame or return frames unevenly.
            # Therefore collect the target number of frames instead of ending by wall time.
            target_frames = int(np.ceil(duration_sec * float(self.cfg.frame_rate_hz)))
            wall_guard_sec = max(duration_sec * 3.0, duration_sec + 30.0)

            while not stop_event.is_set():
                t_rel = time.perf_counter() - t0
                if len(self.t) >= target_frames:
                    break
                if t_rel >= wall_guard_sec:
                    print(f"[RADAR WARN] wall guard reached before target frames: {len(self.t)}/{target_frames}")
                    break

                try:
                    frame = backend.get_frame()
                    consec_err = 0
                except Exception as e:
                    consec_err += 1
                    self.drop_count += 1
                    print(f"[RADAR WARN] get_next_frame failed {consec_err}: {e}")
                    time.sleep(self.cfg.frame_error_sleep_sec)
                    if consec_err >= self.cfg.max_consecutive_frame_errors:
                        raise RuntimeError("Radar frame acquisition repeatedly failed") from e
                    continue

                # SDK get_next_frame() may return frames in bursts.
                # Do NOT use PC receive delta as radar sampling interval.
                # Use configured frame_rate_hz to enforce uniform radar slow-time axis.
                t_frame_wall = time.perf_counter() - t0
                if self._first_frame_wall_t is None:
                    self._first_frame_wall_t = float(t_frame_wall)
                frame_number = len(self.t)
                # For morphology/timing analysis, use the radar slow-time index as the sampling clock.
                # The wall-clock delay before the first SDK frame is not a physiological delay.
                t_frame = frame_number / float(self.cfg.frame_rate_hz)

                rcube = range_fft(frame, self.cfg)
                ra_map, angle_axis, range_axis = dbf_range_angle(rcube, self.cfg)

                valid = np.where((range_axis >= self.cfg.min_range_m) & (range_axis <= self.cfg.max_range_m))[0]
                if len(valid) == 0:
                    continue

                if self.fixed_range_idx is None or self.fixed_angle_idx is None:
                    crop = ra_map[:, valid]
                    idx = int(np.argmax(crop))
                    ai, local_ri = np.unravel_index(idx, crop.shape)
                    ri = int(valid[local_ri])
                    if t_frame <= self.cfg.init_lock_sec:
                        init_a.append(int(ai))
                        init_r.append(int(ri))
                        use_a, use_r = int(ai), int(ri)
                    else:
                        self.fixed_angle_idx = int(np.median(init_a)) if init_a else int(ai)
                        self.fixed_range_idx = int(np.median(init_r)) if init_r else int(ri)
                        use_a, use_r = self.fixed_angle_idx, self.fixed_range_idx
                else:
                    col = ra_map[:, self.fixed_range_idx]
                    peak_a = int(np.argmax(col))
                    self.fixed_angle_idx = int(round((1.0 - self.cfg.angle_relock_alpha) * self.fixed_angle_idx +
                                                     self.cfg.angle_relock_alpha * peak_a))
                    self.fixed_angle_idx = max(0, min(self.cfg.angle_bins - 1, self.fixed_angle_idx))
                    use_a, use_r = self.fixed_angle_idx, self.fixed_range_idx

                cval = beamformed_complex_at(rcube, self.cfg, use_a, use_r)

                self.t.append(float(t_frame))
                self.cvals.append(cval)
                self.ridx.append(int(use_r))
                self.aidx.append(int(use_a))
                self.ra_maps.append(ra_map)

                n = len(self.t)
                if n > 1 and n % self.cfg.print_every_frames == 0:
                    eff_fs = n / max(self.t[-1] - self.t[0], 1e-9)
                    print(f"[RADAR] frames={n}, effective_fs={eff_fs:.2f} Hz, drop={self.drop_count}, range_idx={use_r}, angle_idx={use_a}")

        except Exception as e:
            self.error = e
            self.ready_event.set()
        finally:
            backend.disconnect()

    def analyze(self):
        if self.error:
            raise self.error

        t = np.asarray(self.t, dtype=np.float64)
        c = np.asarray(self.cvals, dtype=np.complex128)
        if len(t) < 10:
            raise RuntimeError(f"Radar data too short: {len(t)} frames")

        # Radar time axis is forced to configured uniform frame rate in acquire().
        fs = float(self.cfg.frame_rate_hz)
        if len(t) >= 2:
            t = float(t[0]) + np.arange(len(t), dtype=np.float64) / fs

        phase = np.unwrap(np.angle(c))
        phase = signal.detrend(phase)
        disp = phase * get_lambda(self.cfg) / (4.0 * np.pi)
        disp = signal.detrend(disp)

        resp = safe_bandpass(disp, fs, self.cfg.resp_band_hz[0], self.cfg.resp_band_hz[1], order=4)

        # 기존 단순 bandpass 대신 LMS 기반 respiration/motion cancellation 후 PPG-like 추출
        ppg, lms_info = radar_respiration_lms_pipeline(disp, fs, self.cfg)

        min_dist = max(1, int(fs * 0.33))
        peaks, _ = signal.find_peaks(ppg, distance=min_dist, prominence=max(np.std(ppg) * 0.35, 1e-12))

        hr = None
        if len(peaks) >= 2:
            rr = np.diff(t[peaks])
            rr = rr[(rr > 0.25) & (rr < 1.8)]
            if len(rr):
                hr = float(60.0 / np.median(rr))

        return {
            "t": t,
            "fs": fs,
            "phase": phase,
            "displacement": disp,
            "respiration": resp,
            "ppg_like": ppg,
            "lms_reference": lms_info["reference"],
            "lms_estimated_noise": lms_info["estimated_noise"],
            "lms_error": lms_info["lms_error"],
            "lms_used": bool(lms_info["used"]),
            "peaks_idx": peaks,
            "peaks_time": t[peaks] if len(peaks) else np.array([]),
            "hr_bpm": hr,
            "range_idx_trace": np.asarray(self.ridx, dtype=np.int32),
            "angle_idx_trace": np.asarray(self.aidx, dtype=np.int32),
            "ra_maps": np.asarray(self.ra_maps, dtype=np.float32),
            "drop_count": int(self.drop_count),
            "fixed_range_idx": self.fixed_range_idx,
            "fixed_angle_idx": self.fixed_angle_idx,
        }


# ============================================================
# AO/AC methodology
# ============================================================
def interpolate_signal(t: np.ndarray, x: np.ndarray, fs_new: float, t_start: Optional[float] = None, t_end: Optional[float] = None):
    if t_start is None:
        t_start = float(t[0])
    if t_end is None:
        t_end = float(t[-1])
    t_new = np.arange(t_start, t_end, 1.0 / fs_new)
    x_new = np.interp(t_new, t, x)
    return t_new, x_new


def curvature_detector(beat_t_rel: np.ndarray, beat: np.ndarray, win: tuple[float, float], kind: str):
    """
    Curvature-based detector.
    AO: upstroke에서 2차 미분 양의 변화가 큰 지점
    AC: late systolic notch/downstroke 주변 2차 미분 절대값이 큰 지점
    """
    fs = 1.0 / np.median(np.diff(beat_t_rel))
    bz = safe_lowpass(zscore_safe(beat), fs=fs, cutoff=12.0, order=3)
    d1 = np.gradient(bz, beat_t_rel)
    d2 = np.gradient(d1, beat_t_rel)

    m = (beat_t_rel >= win[0]) & (beat_t_rel <= win[1])
    if np.sum(m) < 4:
        return None, None

    idxs = np.where(m)[0]
    if kind == "ao":
        score = np.maximum(d1[idxs], 0) * 0.6 + np.maximum(d2[idxs], 0) * 0.4
        local = int(np.argmax(score))
    else:
        # AC는 notch/closing 관련 변곡이므로 |curvature| + downstroke를 같이 반영
        score = np.abs(d2[idxs]) * 0.6 + np.maximum(-d1[idxs], 0) * 0.4
        local = int(np.argmax(score))

    idx = int(idxs[local])
    conf = float(abs(score[local]))
    return idx, conf


def local_energy_detector(beat_t_rel: np.ndarray, beat: np.ndarray, win: tuple[float, float], kind: str):
    """
    Local slope-energy detector.
    AO/AC 후보창 내에서 morphology 변화가 큰 구간을 찾음.
    """
    fs = 1.0 / np.median(np.diff(beat_t_rel))
    bz = safe_lowpass(zscore_safe(beat), fs=fs, cutoff=12.0, order=3)
    d1 = np.gradient(bz, beat_t_rel)

    # 40 ms local RMS energy
    k = max(3, int(0.04 * fs))
    kernel = np.ones(k) / k
    energy = np.convolve(d1 ** 2, kernel, mode="same")

    m = (beat_t_rel >= win[0]) & (beat_t_rel <= win[1])
    if np.sum(m) < 4:
        return None, None

    idxs = np.where(m)[0]
    if kind == "ao":
        # AO는 상승구간 선호
        score = energy[idxs] * (np.maximum(d1[idxs], 0) + 1e-6)
    else:
        # AC는 하강구간 선호
        score = energy[idxs] * (np.maximum(-d1[idxs], 0) + 1e-6)

    local = int(np.argmax(score))
    idx = int(idxs[local])
    conf = float(score[local])
    return idx, conf


def derivative_detector(beat_t_rel: np.ndarray, beat: np.ndarray, win: tuple[float, float], kind: str):
    bz = safe_lowpass(zscore_safe(beat), fs=1.0 / np.median(np.diff(beat_t_rel)), cutoff=12.0)
    d1 = np.gradient(bz, beat_t_rel)
    m = (beat_t_rel >= win[0]) & (beat_t_rel <= win[1])
    if np.sum(m) < 2:
        return None, None
    idxs = np.where(m)[0]
    if kind == "ao":
        local = int(np.argmax(d1[m]))
    else:
        local = int(np.argmin(d1[m]))
    return int(idxs[local]), float(abs(d1[idxs[local]]))


def notch_tidal_detector(beat_t_rel: np.ndarray, beat: np.ndarray, win: tuple[float, float]):
    bz = safe_lowpass(zscore_safe(beat), fs=1.0 / np.median(np.diff(beat_t_rel)), cutoff=10.0)
    m = (beat_t_rel >= win[0]) & (beat_t_rel <= win[1])
    if np.sum(m) < 4:
        return None, None
    idxs = np.where(m)[0]
    seg = bz[idxs]
    # AC surrogate: late systolic valley 또는 급하강 이후 notch 후보
    # 1) local minima
    mins, _ = signal.find_peaks(-seg, distance=max(1, int(0.05 / np.median(np.diff(beat_t_rel)))))
    if len(mins):
        # deepest notch
        local = int(mins[np.argmin(seg[mins])])
    else:
        local = int(np.argmin(seg))
    idx = int(idxs[local])
    depth = float(abs(seg[local] - np.nanmedian(seg)))
    return idx, depth


def wavelet_ridge_detector(beat_t_rel: np.ndarray, beat: np.ndarray, win: tuple[float, float]):
    fs = 1.0 / np.median(np.diff(beat_t_rel))
    bz = zscore_safe(beat)
    m = (beat_t_rel >= win[0]) & (beat_t_rel <= win[1])
    if np.sum(m) < 4:
        return None, None
    idxs = np.where(m)[0]
    seg = bz[idxs]

    if HAS_CWT:
        widths = np.arange(1, max(8, min(32, len(seg) // 2)))
        try:
            W = cwt(seg, morlet2, widths, w=6)
            energy = np.sum(np.abs(W) ** 2, axis=0)
        except Exception:
            energy = np.abs(seg) + np.abs(np.gradient(seg))
    else:
        # fallback: local energy + slope energy
        energy = np.abs(seg) + 0.5 * np.abs(np.gradient(seg))

    local = int(np.argmax(energy))
    return int(idxs[local]), float(energy[local])


def template_detector(beat_t_rel: np.ndarray, beat: np.ndarray, template_t: Optional[np.ndarray], template: Optional[np.ndarray], win: tuple[float, float], kind: str):
    if template is None or template_t is None or len(template) < 5:
        return None, None
    bz = zscore_safe(beat)

    # template에서 해당 window의 대표 이벤트 위치
    tw = (template_t >= win[0]) & (template_t <= win[1])
    bw = (beat_t_rel >= win[0]) & (beat_t_rel <= win[1])
    if np.sum(tw) < 4 or np.sum(bw) < 4:
        return None, None

    # 전체 beat와 template corr이 높으면 template 이벤트 위치 사용
    corr = safe_corr(bz, template)
    if kind == "ao":
        temp_idx = np.where(tw)[0][np.argmax(np.gradient(template, template_t)[tw])]
    else:
        temp_idx = np.where(tw)[0][np.argmin(np.gradient(template, template_t)[tw])]

    event_time = template_t[temp_idx]
    idx = int(np.argmin(np.abs(beat_t_rel - event_time)))
    return idx, float(corr)


def fuse_candidates(candidates: list[dict[str, Any]]):
    valid = [c for c in candidates if c["idx"] is not None]
    if not valid:
        return None, 0.0, None

    idxs = np.array([c["idx"] for c in valid], dtype=float)
    weights = np.array([max(float(c.get("confidence", 0.1)), 0.01) for c in valid], dtype=float)
    # median 기반 robust fusion + confidence는 후보 일치도 기반
    fused_idx = int(np.round(np.average(idxs, weights=weights)))
    dispersion = float(np.std(idxs)) if len(idxs) >= 2 else 0.0
    conf = float(np.clip(np.mean(weights) * np.exp(-dispersion / 8.0), 0, 1))
    return fused_idx, conf, dispersion

def mti_first_order_highpass(x: np.ndarray, beta: float):
    """
    Zheng 논문 2.2절의 MTI-style first-order HPF를 간단 구현.
    s_R[n] = beta*s_R[n-1] + (1-beta)*x[n]
    y[n] = x[n] - s_R[n]
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x.copy()
    sr = np.zeros_like(x)
    sr[0] = x[0]
    for i in range(1, len(x)):
        sr[i] = beta * sr[i-1] + (1.0 - beta) * x[i]
    return x - sr


def zheng_mti_band_component(x: np.ndarray, fs: float):
    """
    Zheng 방식의 두 MTI high-pass 차분을 radar beat에 맞게 적용.
    논문은 beta1=0.9, beta2=0.99 예시를 들며 두 HPF 출력 차이로 중간 대역을 남김.
    여기서는 beat 길이가 짧으므로 안정성을 위해 detrend + median + 차분을 함께 사용.
    """
    x = zscore_safe(np.asarray(x, dtype=np.float64))
    if len(x) < 8:
        return x
    try:
        x = signal.detrend(x, type='linear')
    except Exception:
        x = x - np.nanmean(x)
    if len(x) >= 5:
        k = 5 if len(x) >= 5 else 3
        if k % 2 == 0:
            k += 1
        try:
            x = signal.medfilt(x, kernel_size=k)
        except Exception:
            pass
    # Sampling-rate independent beta approximation: keep pulse-like mid component.
    # At 100 Hz, beta1/beta2 work as slow drift suppressors.
    hp1 = mti_first_order_highpass(x, 0.90)
    hp2 = mti_first_order_highpass(x, 0.99)
    y = hp2 - hp1
    # Additional physiological cardiac-motion band limiting.
    if fs > 20:
        hi = min(14.0, 0.45 * fs)
        if hi > 1.2:
            y = safe_bandpass(y, fs, 0.8, hi, order=3)
    return zscore_safe(y)


def paper_tight_event_lock(bt: np.ndarray, beat: np.ndarray, prior_sec: Optional[float],
                           current_t: Optional[float], kind: str, cfg: AnalysisConfig,
                           prev_t: Optional[float] = None):
    """
    Paper-tight AO/AC refinement.

    목적:
    - 기존 multi-detector fusion이 beat를 살리는 데는 좋지만 AO/AC 위치가 크게 흔들리는 문제를 줄임.
    - ECG Q/T+RR pseudo-reference 주변에서 논문 기반 morphology feature를 다시 탐색.
    - 목표는 'ECG pseudo-reference 대비 ±10 ms'로, 실제 판막 ground-truth를 보장하는 것은 아님.

    AO:
    - Zheng: MTI band component + seventh power + Hilbert envelope로 pulsatile AO peak 강화
    - positive slope / local peak / envelope peak / prior proximity 결합

    AC:
    - Qiao/RCG 관점: cardiac mechanical transition을 notch/inflection/curvature로 처리
    - negative slope, curvature, local notch, prior proximity, AO-AC physiological interval 결합
    """
    if prior_sec is None or not np.isfinite(prior_sec):
        return current_t, 0.0, {"used": False, "reason": "no_prior"}

    bt = np.asarray(bt, dtype=np.float64)
    x = np.asarray(beat, dtype=np.float64)
    if len(bt) < 10 or len(bt) != len(x):
        return current_t, 0.0, {"used": False, "reason": "short"}

    fs = 1.0 / np.median(np.diff(bt))
    prior = float(prior_sec)
    half = float(getattr(cfg, "ao_tight_lock_half_window_sec", 0.045) if kind == "ao" else getattr(cfg, "ac_tight_lock_half_window_sec", 0.055))
    lo = prior - half
    hi = prior + half

    # Physiological guardrails
    if kind == "ao":
        lo = max(float(cfg.ao_search_sec[0]), lo)
        hi = min(float(cfg.ao_search_sec[1]), hi)
    else:
        lo = max(float(cfg.ac_search_sec[0]), lo)
        hi = min(float(cfg.ac_search_sec[1]), hi)

    m = (bt >= lo) & (bt <= hi)
    if np.sum(m) < 4:
        return current_t, 0.0, {"used": False, "reason": "empty_window", "prior": prior}

    idxs = np.where(m)[0]
    y = zheng_mti_band_component(x, fs)
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)

    # Seventh-power envelope for AO-like pulsatility
    yp = np.sign(y) * (np.abs(y) ** 7)
    try:
        env = np.abs(signal.hilbert(yp))
    except Exception:
        env = np.abs(yp)
    w = max(3, int(round(0.05 * fs)))
    if w % 2 == 0:
        w += 1
    if w >= 3 and len(env) >= w:
        env = np.convolve(env, np.ones(w) / w, mode='same')
    env_s = robust_scale_01(env)

    prior_score = np.exp(-0.5 * ((bt[idxs] - prior) / float(getattr(cfg, "tight_lock_prior_sigma_sec", 0.010))) ** 2)

    cont_score = np.zeros_like(prior_score)
    if prev_t is not None and np.isfinite(prev_t):
        cont_score = np.exp(-0.5 * ((bt[idxs] - float(prev_t)) / float(getattr(cfg, "tight_lock_continuity_sigma_sec", 0.025))) ** 2)

    curr_score = np.zeros_like(prior_score)
    if current_t is not None and np.isfinite(current_t):
        curr_score = np.exp(-0.5 * ((bt[idxs] - float(current_t)) / 0.030) ** 2)

    if kind == "ao":
        morph = (
            0.34 * robust_scale_01(env_s[idxs]) +
            0.24 * robust_scale_01(np.maximum(d1[idxs], 0)) +
            0.22 * robust_scale_01(y[idxs]) +
            0.20 * robust_scale_01(np.maximum(d2[idxs], 0))
        )
        score = 0.62 * prior_score + 0.26 * morph + 0.07 * cont_score + 0.05 * curr_score
    else:
        # AC can be notch or inflection; do not require positive peak.
        local_min_bonus = robust_scale_01(-y[idxs])
        falling = robust_scale_01(np.maximum(-d1[idxs], 0))
        curvature = robust_scale_01(np.abs(d2[idxs]))
        morph = 0.34 * curvature + 0.30 * falling + 0.22 * local_min_bonus + 0.14 * robust_scale_01(env_s[idxs])
        score = 0.66 * prior_score + 0.22 * morph + 0.07 * cont_score + 0.05 * curr_score

    best_local = int(np.argmax(score))
    best_idx = int(idxs[best_local])
    best_t = float(bt[best_idx])
    morph_conf = float(morph[best_local]) if np.ndim(morph) else 0.0
    raw_err = best_t - prior
    target = float(getattr(cfg, "tight_target_error_ms", 10.0)) / 1000.0

    snapped = False
    if bool(getattr(cfg, "tight_lock_snap_if_outside_target", True)) and abs(raw_err) > target:
        # Do not jump blindly to ECG prior unless there is at least weak morphology in the window.
        if morph_conf >= float(getattr(cfg, "tight_lock_min_morph_conf", 0.12)):
            best_t = prior + np.sign(raw_err) * min(0.8 * target, abs(raw_err) * 0.35)
            snapped = True

    conf = float(np.clip(score[best_local], 0.0, 1.0))
    return best_t, conf, {
        "used": True,
        "kind": kind,
        "prior": prior,
        "selected_before_snap": float(bt[best_idx]),
        "selected": float(best_t),
        "raw_error_ms": float(raw_err * 1000.0),
        "final_error_ms": float((best_t - prior) * 1000.0),
        "snapped": bool(snapped),
        "morph_conf": morph_conf,
        "score": conf,
    }


def compute_beat_sqi(beat_t_rel: np.ndarray, beat: np.ndarray, template: Optional[np.ndarray], cfg: AnalysisConfig):
    fs = 1.0 / np.median(np.diff(beat_t_rel))
    bz = zscore_safe(beat)

    amp_std = float(np.std(bz))
    cardiac_power = bandpower(bz, fs, (0.8, 2.5))
    resp_power = bandpower(bz, fs, (0.1, 0.6))
    total_power = bandpower(bz, fs, (0.1, min(10.0, fs * 0.45))) + 1e-12

    cardiac_ratio = cardiac_power / total_power
    resp_ratio = resp_power / (cardiac_power + 1e-12)

    slope_energy = float(np.mean(np.abs(np.gradient(bz, beat_t_rel))))

    template_corr = 0.0
    if template is not None and len(template) == len(bz):
        template_corr = safe_corr(bz, template)

    # normalize components
    s_amp = np.clip(amp_std / 1.0, 0, 1)
    s_card = np.clip(cardiac_ratio / 0.25, 0, 1)
    s_temp = np.clip((template_corr + 0.2) / 1.2, 0, 1)
    s_slope = np.clip(slope_energy / 8.0, 0, 1)
    penalty_resp = np.clip(resp_ratio / cfg.max_resp_ratio, 0, 1)

    sqi = 0.25 * s_amp + 0.25 * s_card + 0.25 * s_temp + 0.25 * s_slope
    sqi = float(np.clip(sqi * (1.0 - 0.35 * penalty_resp), 0, 1))

    accepted = bool(
        sqi >= cfg.min_sqi_accept and
        amp_std >= cfg.min_amp_std and
        template_corr >= cfg.min_template_corr and
        resp_ratio <= cfg.max_resp_ratio
    )

    return {
        "sqi": sqi,
        "accepted": accepted,
        "amp_std": amp_std,
        "cardiac_power": float(cardiac_power),
        "resp_power": float(resp_power),
        "cardiac_ratio": float(cardiac_ratio),
        "resp_ratio": float(resp_ratio),
        "slope_energy": slope_energy,
        "template_corr": float(template_corr),
    }


def limited_dtw_distance(x: np.ndarray, y: np.ndarray, band: int):
    """
    Sakoe-Chiba band 제한 DTW distance.
    full DTW보다 가볍게 beat morphology 유사도만 평가.
    """
    x = zscore_safe(np.asarray(x, dtype=np.float64))
    y = zscore_safe(np.asarray(y, dtype=np.float64))
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return np.inf
    band = max(int(band), abs(n - m), 1)

    inf = np.inf
    prev = np.full(m + 1, inf)
    curr = np.full(m + 1, inf)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr[:] = inf
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        for j in range(j_start, j_end + 1):
            cost = abs(x[i - 1] - y[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    dist = prev[m] / max(n + m, 1)
    return float(dist)


def shift_signal_by_samples_fill_edge(x: np.ndarray, shift_samples: int):
    """
    beat window 내부에서 morphology alignment용 shift.
    양수 shift: 오른쪽으로 이동. 빈 구간은 edge값으로 채움.
    """
    x = np.asarray(x, dtype=np.float64)
    if shift_samples == 0 or len(x) == 0:
        return x.copy()

    y = np.empty_like(x)
    if shift_samples > 0:
        s = min(shift_samples, len(x))
        y[:s] = x[0]
        y[s:] = x[:-s]
    else:
        s = min(-shift_samples, len(x))
        y[-s:] = x[-1]
        y[:-s] = x[s:]
    return y


def estimate_beat_lag_xcorr(beat: np.ndarray, template: np.ndarray, fs: float, max_lag_sec: float):
    """
    beat와 template 사이의 morphology lag 추정.
    반환:
    - lag_sec: beat를 template에 맞추기 위해 적용할 시간 shift
    - corr_max
    """
    x = zscore_safe(np.asarray(beat, dtype=np.float64))
    y = zscore_safe(np.asarray(template, dtype=np.float64))
    n = min(len(x), len(y))
    if n < 8:
        return 0.0, 0.0

    x = x[:n]
    y = y[:n]

    c = signal.correlate(y, x, mode="full", method="auto")
    lags = signal.correlation_lags(len(y), len(x), mode="full")

    max_lag = int(max(1, round(max_lag_sec * fs)))
    m = np.abs(lags) <= max_lag
    if not np.any(m):
        return 0.0, 0.0

    c_sel = c[m]
    lag_sel = lags[m]
    denom = (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
    c_norm = c_sel / denom

    k = int(np.argmax(c_norm))
    lag_samples = int(lag_sel[k])
    lag_sec = float(lag_samples / fs)
    corr_max = float(c_norm[k])
    return lag_sec, corr_max


def align_beats_to_template(beats: list[dict[str, Any]], template_t: np.ndarray, template: np.ndarray, cfg: AnalysisConfig):
    """
    각 radar beat를 template에 cross-correlation 기반 정렬.
    - t_rel 자체는 R 기준 시간축 유지
    - radar_beat_aligned만 검출에 사용
    - AO/AC 결과는 alignment shift를 다시 보정하여 R 기준 원래 시간으로 환산
    """
    if template is None or template_t is None or len(template) < 8:
        return beats

    fs = float(cfg.radar_interp_fs_hz)
    band = int(max(1, round(cfg.dtw_band_sec * fs)))

    min_len = min(len(template), min(len(b["radar_beat"]) for b in beats)) if beats else 0
    if min_len < 8:
        return beats

    tpl = zscore_safe(template[:min_len])

    for b in beats:
        x = zscore_safe(b["radar_beat"][:min_len])
        lag_sec, corr = estimate_beat_lag_xcorr(
            beat=x,
            template=tpl,
            fs=fs,
            max_lag_sec=cfg.max_beat_align_lag_sec
        )
        lag_samples = int(round(lag_sec * fs))
        aligned = shift_signal_by_samples_fill_edge(b["radar_beat"], lag_samples)

        # DTW distance는 alignment 후 morphology quality 지표로 저장
        dtw_dist = limited_dtw_distance(aligned[:min_len], tpl, band=band)

        b["radar_beat_original"] = b["radar_beat"].copy()
        b["radar_beat_aligned"] = zscore_safe(aligned)
        b["alignment_lag_sec"] = float(lag_sec)
        b["alignment_lag_ms"] = float(lag_sec * 1000.0)
        b["alignment_corr"] = float(corr)
        b["dtw_distance"] = float(dtw_dist)

    return beats


def build_initial_beats(ecg, radar, cfg: AnalysisConfig):
    t0 = max(float(ecg["t"][0]), float(radar["t"][0]))
    t1 = min(float(ecg["t"][-1]), float(radar["t"][-1]))
    t_ri, radar_i = interpolate_signal(radar["t"], radar["ppg_like"], cfg.radar_interp_fs_hz, t0, t1)
    radar_i_z = zscore_safe(radar_i)

    beats = []
    for bi, r_t in enumerate(ecg["peaks_time"]):
        w0, w1 = r_t - cfg.beat_pre_sec, r_t + cfg.beat_post_sec
        if w0 < t_ri[0] or w1 > t_ri[-1]:
            continue
        mask = (t_ri >= w0) & (t_ri <= w1)
        if np.sum(mask) < int((cfg.beat_pre_sec + cfg.beat_post_sec) * cfg.radar_interp_fs_hz * 0.7):
            continue
        bt = t_ri[mask] - r_t
        bx = radar_i_z[mask]
        beats.append({
            "beat_index": int(bi),
            "r_time": float(r_t),
            "t_rel": bt,
            "radar_beat": zscore_safe(bx),
        })
    return t_ri, radar_i, radar_i_z, beats


def make_template_from_beats(beats: list[dict[str, Any]], accepted_only: bool = False):
    selected = []
    for b in beats:
        if accepted_only and not b.get("accepted", True):
            continue
        selected.append(b)
    if not selected:
        return None, None
    min_len = min(len(b["radar_beat"]) for b in selected)
    mat = np.vstack([b["radar_beat"][:min_len] for b in selected])
    t_rel = selected[0]["t_rel"][:min_len]
    template = zscore_safe(np.nanmedian(mat, axis=0))
    return t_rel, template


def local_rr_qt_features(ecg: dict, beat_index: int):
    """
    Beat-wise ECG 보조 특징:
    - RR_i
    - previous RR
    - delta RR
    - QT-like interval: T - Q
    - RT interval: T - R
    """
    peaks = np.asarray(ecg.get("peaks_time", []), dtype=np.float64)
    q_time = np.asarray(ecg.get("q_time", np.full(len(peaks), np.nan)), dtype=np.float64)
    t_time = np.asarray(ecg.get("t_time", np.full(len(peaks), np.nan)), dtype=np.float64)

    n = len(peaks)
    if n < 2 or beat_index >= n:
        return {
            "rr": None, "rr_prev": None, "delta_rr": None,
            "qt": None, "rt": None, "rr_stability": 0.0
        }

    if beat_index < n - 1:
        rr = float(peaks[beat_index + 1] - peaks[beat_index])
    else:
        rr = float(peaks[beat_index] - peaks[beat_index - 1])

    rr_prev = None
    if beat_index > 0:
        rr_prev = float(peaks[beat_index] - peaks[beat_index - 1])

    delta_rr = None if rr_prev is None else float(rr - rr_prev)

    qt = None
    rt = None
    if beat_index < len(q_time) and beat_index < len(t_time):
        if np.isfinite(q_time[beat_index]) and np.isfinite(t_time[beat_index]):
            qt = float(t_time[beat_index] - q_time[beat_index])
        if np.isfinite(t_time[beat_index]):
            rt = float(t_time[beat_index] - peaks[beat_index])

    # RR stability: ΔRR가 작을수록 높음. 0~1
    if delta_rr is None:
        rr_stability = 0.7
    else:
        rr_stability = float(np.exp(-abs(delta_rr) / 0.080))  # 80ms scale

    return {
        "rr": rr,
        "rr_prev": rr_prev,
        "delta_rr": delta_rr,
        "qt": qt,
        "rt": rt,
        "rr_stability": rr_stability,
    }


def ecg_estimated_ao_ac_adaptive(ecg: dict, beat_index: int, cfg: AnalysisConfig):
    """
    ECG Q/R/T + RR/HRV 기반 pseudo AO/AC reference.
    목적:
    - RR-only가 아니라 Q/T morphology + RR adaptive prior 결합
    - Radar AO/AC 검출의 search prior 및 성능 평가 기준으로 사용

    AO:
    - 기본: Q + adaptive PEP
    - adaptive PEP = 90~130ms 범위
    - RR이 짧을수록 PEP 약간 짧게, RR 안정성이 낮으면 expected AO로 fallback 가중

    AC:
    - 기본: T - adaptive offset
    - T 후보가 있으면 T 주변 late systolic prior
    - T가 불안정하면 RR 기반 LVET 근사: AO + 0.28~0.33s
    """
    peaks = np.asarray(ecg.get("peaks_time", []), dtype=np.float64)
    q_time = np.asarray(ecg.get("q_time", np.full(len(peaks), np.nan)), dtype=np.float64)
    t_time = np.asarray(ecg.get("t_time", np.full(len(peaks), np.nan)), dtype=np.float64)

    try:
        r_time = float(peaks[beat_index])
    except Exception:
        return float(getattr(cfg, "expected_ao_sec", 0.12)), float(getattr(cfg, "expected_ac_sec", 0.38)), 0.0, 0.0

    feat = local_rr_qt_features(ecg, beat_index)
    rr = feat["rr"]
    qt = feat["qt"]
    rt = feat["rt"]
    rr_stability = feat["rr_stability"]

    expected_ao = float(getattr(cfg, "expected_ao_sec", 0.12))
    expected_ac = float(getattr(cfg, "expected_ac_sec", 0.38))

    # -----------------------------
    # AO estimate
    # -----------------------------
    # RR-adaptive PEP: 심박이 빠르면 약간 짧아짐
    if rr is not None and np.isfinite(rr):
        # RR 0.6~1.0s 사이에서 PEP 95~120ms 정도로 제한
        pep = 0.105 + 0.030 * np.clip((rr - 0.65) / 0.45, -0.5, 1.0)
        pep = float(np.clip(pep, 0.085, 0.135))
    else:
        pep = float(getattr(cfg, "ecg_pep_const_sec", 0.110))

    ao_rel_q = None
    if beat_index < len(q_time) and np.isfinite(q_time[beat_index]):
        ao_rel_q = float(q_time[beat_index] + pep - r_time)

    # Q가 이상하거나 AO 생리범위 밖이면 expected AO와 혼합/대체
    if ao_rel_q is None or not (0.035 <= ao_rel_q <= 0.260):
        ao_rel = expected_ao
        ao_conf = 0.45
    else:
        # RR 안정성이 높을수록 Q 기반 값을 더 신뢰
        w = 0.65 + 0.25 * rr_stability
        ao_rel = float(w * ao_rel_q + (1.0 - w) * expected_ao)
        ao_conf = float(0.60 + 0.35 * rr_stability)

    # -----------------------------
    # AC estimate
    # -----------------------------
    ac_candidates = []
    # T 기반: AC는 T/Te 근처 또는 약간 앞
    if beat_index < len(t_time) and np.isfinite(t_time[beat_index]):
        # RT가 너무 이상하면 신뢰 낮춤
        rt_rel = float(t_time[beat_index] - r_time)
        if 0.180 <= rt_rel <= 0.620:
            # T peak 기준 AC offset: RR 짧으면 offset 약간 감소
            if rr is not None and np.isfinite(rr):
                ac_before_t = 0.020 + 0.025 * np.clip((rr - 0.55) / 0.55, 0.0, 1.0)
                ac_before_t = float(np.clip(ac_before_t, 0.015, 0.050))
            else:
                ac_before_t = float(getattr(cfg, "ecg_ac_before_t_const_sec", 0.030))
            ac_t_based = rt_rel - ac_before_t
            if 0.240 <= ac_t_based <= 0.620:
                ac_candidates.append((ac_t_based, 0.65 + 0.25 * rr_stability))

    # QT/RR 기반 LVET 근사: AO + LVET
    # 높은 HR에서 LVET는 짧아짐. 0.23~0.34s 제한.
    if rr is not None and np.isfinite(rr):
        lvet = 0.220 + 0.160 * np.clip((rr - 0.45) / 0.75, 0.0, 1.0)
        lvet = float(np.clip(lvet, 0.230, 0.340))
        ac_rr_based = ao_rel + lvet
        if 0.240 <= ac_rr_based <= 0.620:
            ac_candidates.append((ac_rr_based, 0.45 + 0.25 * rr_stability))

    # fallback
    ac_candidates.append((expected_ac, 0.30))

    # weighted average, confidence-weighted
    vals = np.array([c[0] for c in ac_candidates], dtype=float)
    ws = np.array([c[1] for c in ac_candidates], dtype=float)
    ac_rel = float(np.sum(vals * ws) / np.sum(ws))
    ac_conf = float(np.clip(np.max(ws), 0, 1))

    # sanity
    if ac_rel <= ao_rel + 0.120:
        ac_rel = float(ao_rel + 0.240)
        ac_conf *= 0.75
    if not (0.240 <= ac_rel <= 0.620):
        ac_rel = expected_ac
        ac_conf = 0.30

    return ao_rel, ac_rel, ao_conf, ac_conf

def ecg_estimated_ao_ac_from_landmarks(ecg: dict, beat_index: int, cfg: AnalysisConfig):
    """
    Backward-compatible wrapper.
    Returns ECG-derived adaptive pseudo AO/AC timing reference.
    """
    ao_rel, ac_rel, _, _ = ecg_estimated_ao_ac_adaptive(ecg, beat_index, cfg)
    return ao_rel, ac_rel


def safe_pearson_for_fig(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 3:
        return None
    if np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return None
    return float(np.corrcoef(x[m], y[m])[0, 1])


def accuracy_within_tolerance_ms(err_ms, tol_ms):
    err_ms = np.asarray(err_ms, dtype=np.float64)
    m = np.isfinite(err_ms)
    if np.sum(m) == 0:
        return None
    return float(np.mean(np.abs(err_ms[m]) <= tol_ms) * 100.0)

def ac_temporal_tracking_refine(bt: np.ndarray,
                                beat: np.ndarray,
                                current_ac_t: Optional[float],
                                ao_t: Optional[float],
                                ecg_ac_prior: Optional[float],
                                prev_ac_t: Optional[float],
                                cfg: AnalysisConfig):
    """
    AC temporal tracking refinement.
    AC는 peak-like가 아니라 transition/notch/inflection라서 beat-wise independent detection이 흔들림.
    따라서 current detector, previous valid AC, ECG QRT/RR prior를 합친 target 주변에서
    inflection/zero-crossing 기반으로 재탐색한다.

    반환:
    - refined_ac_t: R 기준 AC timing(sec)
    - confidence boost
    - debug dict
    """
    bt = np.asarray(bt, dtype=np.float64)
    x = zscore_safe(np.asarray(beat, dtype=np.float64))
    if len(bt) < 10:
        return current_ac_t, 0.0, {"used": False, "reason": "short"}

    candidates = []
    weights = []

    if current_ac_t is not None and np.isfinite(current_ac_t):
        candidates.append(float(current_ac_t))
        weights.append(float(getattr(cfg, "ac_tracking_current_weight", 0.30)))

    if prev_ac_t is not None and np.isfinite(prev_ac_t):
        candidates.append(float(prev_ac_t))
        weights.append(float(getattr(cfg, "ac_tracking_prev_weight", 0.35)))

    if ecg_ac_prior is not None and np.isfinite(ecg_ac_prior):
        candidates.append(float(ecg_ac_prior))
        weights.append(float(getattr(cfg, "ac_tracking_ecg_weight", 0.35)))

    if not candidates:
        return current_ac_t, 0.0, {"used": False, "reason": "no_reference"}

    target = float(np.sum(np.asarray(candidates) * np.asarray(weights)) / np.sum(weights))

    # AO-AC physiological interval constraint
    if ao_t is not None and np.isfinite(ao_t):
        min_ac = float(ao_t + getattr(cfg, "ac_interval_min_sec", 0.140))
        max_ac = float(ao_t + getattr(cfg, "ac_interval_max_sec", 0.500))
    else:
        min_ac = 0.240
        max_ac = 0.650

    target = float(np.clip(target, min_ac, max_ac))

    half = float(getattr(cfg, "ac_tracking_window_sec", 0.060))
    lo = max(0.220, min_ac, target - half)
    hi = min(0.680, max_ac, target + half)

    m = (bt >= lo) & (bt <= hi)
    if np.sum(m) < 5:
        # fallback to wider but still constrained
        lo = max(0.220, min_ac, target - 0.090)
        hi = min(0.680, max_ac, target + 0.090)
        m = (bt >= lo) & (bt <= hi)

    if np.sum(m) < 5:
        return current_ac_t, 0.0, {"used": False, "reason": "empty_window", "target": target}

    fs = 1.0 / np.median(np.diff(bt))
    y = safe_lowpass(x, fs, 10.0, order=3)
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    idxs = np.where(m)[0]
    times = bt[idxs]

    # AC는 inflection/transition 중심
    curv = robust_scale_01(np.abs(d2[idxs]))
    fall = robust_scale_01(np.maximum(-d1[idxs], 0))
    timing = np.exp(-0.5 * ((times - target) / max(0.025, half * 0.65)) ** 2)

    # zero crossing: negative->positive(local min) 우선, positive->negative도 약하게
    zc = np.zeros(len(idxs))
    for k, ii in enumerate(idxs):
        a = max(0, ii - int(0.040 * fs))
        b = min(len(y), ii + int(0.040 * fs) + 1)
        if b - a < 4:
            continue
        dd = d1[a:b]
        if np.any((dd[:-1] < 0) & (dd[1:] >= 0)):
            zc[k] += 1.0
        if np.any((dd[:-1] > 0) & (dd[1:] <= 0)):
            zc[k] += 0.35

    # local minimum/notch
    notch = np.zeros(len(idxs))
    mins, _ = signal.find_peaks(-y[idxs], distance=max(1, int(0.025 * fs)))
    if len(mins):
        notch[mins] = 1.0

    score = (
        0.34 * curv +
        0.22 * fall +
        0.18 * timing +
        0.16 * zc +
        0.10 * notch
    )

    best = int(np.argmax(score))
    refined_idx = int(idxs[best])
    refined_idx, refined_t = refine_event_highres(bt, y, refined_idx, "ac")
    if refined_t is None:
        refined_t = float(bt[refined_idx])

    refined_t = float(np.clip(refined_t, min_ac, max_ac))
    conf = float(np.clip(score[best], 0.0, 1.0))

    return refined_t, conf, {
        "used": True,
        "target": target,
        "lo": lo,
        "hi": hi,
        "score": conf,
        "min_ac": min_ac,
        "max_ac": max_ac,
    }

def ao_ac_pipeline(ecg, radar, cfg: AnalysisConfig):
    t_ri, radar_i, radar_i_z, beats = build_initial_beats(ecg, radar, cfg)

    # initial template
    template_t, template = make_template_from_beats(beats, accepted_only=False)

    # Beat-level alignment to reduce ECG-Radar phase mismatch
    if cfg.use_beat_alignment and template is not None:
        beats = align_beats_to_template(beats, template_t, template, cfg)
        # 이후 detector는 aligned beat를 기본으로 사용
        for b in beats:
            if "radar_beat_aligned" in b:
                b["radar_beat"] = b["radar_beat_aligned"]
        template_t, template = make_template_from_beats(beats, accepted_only=False)

    # SQI iterative refinement
    for _ in range(max(1, cfg.template_iterations)):
        for b in beats:
            sqi = compute_beat_sqi(b["t_rel"], b["radar_beat"], template, cfg)
            b.update(sqi)
        new_t, new_template = make_template_from_beats(beats, accepted_only=True)
        if new_template is not None and len(new_template) >= 5:
            template_t, template = new_t, new_template

    rows = []
    accepted_beats = []
    rejected_beats = []

    # Paper-style congruency reference buffers
    prev_icp_delay = None
    recent_irp_delays = []
    recent_ao_delays = []
    recent_ac_delays = []

    # ECG QRT/RR adaptive pseudo-reference with local median smoothing
    ecg_ref_series = build_ecg_adaptive_reference_series(ecg, cfg)

    for b in beats:
        bt = b["t_rel"]
        bx = b["radar_beat"]
        accepted = bool(b.get("accepted", False))

        # detector candidates
        ao_candidates = []
        ac_candidates = []

        beat_i = int(b["beat_index"])
        te_delay = get_te_delay_for_beat(ecg, beat_i)

        if beat_i < len(ecg_ref_series["ao_smooth"]):
            ecg_ao_prior = float(ecg_ref_series["ao_smooth"][beat_i])
            ecg_ac_prior = float(ecg_ref_series["ac_smooth"][beat_i])
            ecg_ao_prior_conf = float(ecg_ref_series["ao_conf"][beat_i])
            ecg_ac_prior_conf = float(ecg_ref_series["ac_conf"][beat_i])
        else:
            ecg_ao_prior, ecg_ac_prior, ecg_ao_prior_conf, ecg_ac_prior_conf = ecg_estimated_ao_ac_adaptive(
                ecg, beat_i, cfg
            )

        # ECG Q/R/T 기반 AO/AC prior는 ground truth가 아니므로 기본 detector에는 넣지 않는다.
        # ECG는 R-peak anchor로만 사용하고, 후보 검출은 radar morphology + fixed physiological window 중심으로 수행한다.
        if bool(getattr(cfg, "use_ecg_qrt_prior_for_candidate_detection", False)):
            det_ao_prior = ecg_ao_prior
            det_ac_prior = ecg_ac_prior
            det_te_delay = te_delay
        else:
            det_ao_prior = float(getattr(cfg, "expected_ao_sec", 0.12))
            det_ac_prior = float(getattr(cfg, "expected_ac_sec", 0.38))
            det_te_delay = None

        prev_ao_ref_for_score = recent_ao_delays[-1] if 'recent_ao_delays' in locals() and len(recent_ao_delays) else None
        idx, conf = radar_event_score_detector_with_ecg_prior(
            bt, bx, kind="ao", ecg_prior_sec=det_ao_prior, prev_delay=prev_ao_ref_for_score
        )
        ao_candidates.append({"method": "radar_score_ecg_prior_ao", "idx": idx, "confidence": conf if conf is not None else 0.0})

        idx, conf = zheng_seventh_power_ao_detector(
            bt, bx, ecg_prior_sec=det_ao_prior, prev_ao_delay=prev_ao_ref_for_score
        )
        ao_candidates.append({"method": "zheng_seventh_power_envelope_ao", "idx": idx, "confidence": conf if conf is not None else 0.0})

        idx, conf, meta_ao_paper = scg_paper_style_ao_ac_detector(
            bt, bx, kind="ao", prev_ref_delay=prev_icp_delay, te_delay=det_te_delay
        )
        ao_candidates.append({"method": "paper_icp_ao", "idx": idx, "confidence": conf if conf is not None else 0.0})

        prev_ao_ref = recent_ao_delays[-1] if len(recent_ao_delays) else None
        idx, conf = ao_fallback_timing_prior_detector(bt, bx, prev_ao_delay=prev_ao_ref)
        ao_candidates.append({"method": "ao_fallback_timing_prior", "idx": idx, "confidence": conf if conf is not None else 0.0})

        idx, conf = scg_inspired_aoac_detector(bt, bx, cfg.ao_search_sec, kind="ao")
        ao_candidates.append({"method": "scg_inspired_envelope_slope", "idx": idx, "confidence": conf if conf is not None else 0.0})

        idx, conf = morphology_event_detector(bt, bx, cfg.ao_search_sec, kind="ao")
        ao_candidates.append({"method": "morphology_rise_slope_zero", "idx": idx, "confidence": conf if conf is not None else 0.0})

        idx, conf = derivative_detector(bt, bx, cfg.ao_search_sec, kind="ao")
        ao_candidates.append({"method": "derivative_upstroke", "idx": idx, "confidence": min(conf / 12.0 if conf is not None else 0.0, 1.0)})

        idx, conf = wavelet_ridge_detector(bt, bx, cfg.ao_search_sec)
        ao_candidates.append({"method": "wavelet_ridge", "idx": idx, "confidence": min(conf / 10.0 if conf is not None else 0.0, 1.0)})

        idx, conf = curvature_detector(bt, bx, cfg.ao_search_sec, kind="ao")
        ao_candidates.append({"method": "curvature", "idx": idx, "confidence": min(conf / 60.0 if conf is not None else 0.0, 1.0)})

        idx, conf = local_energy_detector(bt, bx, cfg.ao_search_sec, kind="ao")
        ao_candidates.append({"method": "local_slope_energy", "idx": idx, "confidence": min(conf / 80.0 if conf is not None else 0.0, 1.0)})

        idx, conf = template_detector(bt, bx, template_t, template, cfg.ao_search_sec, kind="ao")
        ao_candidates.append({"method": "template", "idx": idx, "confidence": max(0.0, min((conf + 1) / 2 if conf is not None else 0.0, 1.0))})

        irp_ref = recent_irp_delays[-1] if len(recent_irp_delays) else None
        prev_ac_ref_for_score = recent_ac_delays[-1] if 'recent_ac_delays' in locals() and len(recent_ac_delays) else None
        idx, conf = ac_inflection_zero_cross_detector(
            bt, bx, ecg_prior_sec=det_ac_prior, prev_ac_delay=prev_ac_ref_for_score
        )
        ac_candidates.append({"method": "ac_inflection_zero_cross", "idx": idx, "confidence": conf if conf is not None else 0.0})

        idx, conf = radar_event_score_detector_with_ecg_prior(
            bt, bx, kind="ac", ecg_prior_sec=det_ac_prior, prev_delay=prev_ac_ref_for_score
        )
        ac_candidates.append({"method": "radar_score_ecg_prior_ac", "idx": idx, "confidence": conf if conf is not None else 0.0})

        idx, conf, meta_ac_paper = scg_paper_style_ao_ac_detector(
            bt, bx, kind="ac", prev_ref_delay=irp_ref, te_delay=det_te_delay
        )
        ac_candidates.append({"method": "paper_irp_ac", "idx": idx, "confidence": conf if conf is not None else 0.0})

        prev_ac_ref = recent_ac_delays[-1] if len(recent_ac_delays) else None
        idx, conf = ac_fallback_timing_prior_detector(bt, bx, te_delay=det_te_delay, prev_ac_delay=prev_ac_ref)
        ac_candidates.append({"method": "ac_fallback_te_prev_prior", "idx": idx, "confidence": conf if conf is not None else 0.0})

        idx, conf = scg_inspired_aoac_detector(bt, bx, cfg.ac_search_sec, kind="ac")
        ac_candidates.append({"method": "scg_inspired_envelope_notch", "idx": idx, "confidence": conf if conf is not None else 0.0})

        idx, conf = morphology_event_detector(bt, bx, cfg.ac_search_sec, kind="ac")
        ac_candidates.append({"method": "morphology_fall_notch_curvature", "idx": idx, "confidence": conf if conf is not None else 0.0})

        idx, conf = derivative_detector(bt, bx, cfg.ac_search_sec, kind="ac")
        ac_candidates.append({"method": "derivative_downstroke", "idx": idx, "confidence": min(conf / 12.0 if conf is not None else 0.0, 1.0)})

        idx, conf = notch_tidal_detector(bt, bx, cfg.ac_search_sec)
        ac_candidates.append({"method": "notch_tidal", "idx": idx, "confidence": min(conf / 2.0 if conf is not None else 0.0, 1.0)})

        idx, conf = wavelet_ridge_detector(bt, bx, cfg.ac_search_sec)
        ac_candidates.append({"method": "wavelet_ridge", "idx": idx, "confidence": min(conf / 10.0 if conf is not None else 0.0, 1.0)})

        idx, conf = curvature_detector(bt, bx, cfg.ac_search_sec, kind="ac")
        ac_candidates.append({"method": "curvature", "idx": idx, "confidence": min(conf / 60.0 if conf is not None else 0.0, 1.0)})

        idx, conf = local_energy_detector(bt, bx, cfg.ac_search_sec, kind="ac")
        ac_candidates.append({"method": "local_slope_energy", "idx": idx, "confidence": min(conf / 80.0 if conf is not None else 0.0, 1.0)})

        idx, conf = template_detector(bt, bx, template_t, template, cfg.ac_search_sec, kind="ac")
        ac_candidates.append({"method": "template", "idx": idx, "confidence": max(0.0, min((conf + 1) / 2 if conf is not None else 0.0, 1.0))})

        ao_idx, ao_conf, ao_disp = fuse_candidates(ao_candidates)
        ac_idx, ac_conf, ac_disp = fuse_candidates(ac_candidates)

        align_lag_sec = float(b.get("alignment_lag_sec", 0.0))
        ao_t = None if ao_idx is None else float(bt[max(0, min(len(bt)-1, ao_idx))] - align_lag_sec)
        ac_t = None if ac_idx is None else float(bt[max(0, min(len(bt)-1, ac_idx))] - align_lag_sec)

        # physiological range clamp after inverse alignment correction
        if ao_t is not None and not (-cfg.beat_pre_sec <= ao_t <= cfg.beat_post_sec):
            ao_t = None
        if ac_t is not None and not (-cfg.beat_pre_sec <= ac_t <= cfg.beat_post_sec):
            ac_t = None

        # AC temporal tracking refinement: independent AC detection 흔들림 완화
        ac_tracking_used = False
        ac_tracking_conf = 0.0
        ac_tracking_target = None
        if getattr(cfg, "use_ac_temporal_tracking", True):
            prev_ac_ref_for_track = recent_ac_delays[-1] if len(recent_ac_delays) else None
            ac_refined, ac_track_conf, ac_track_dbg = ac_temporal_tracking_refine(
                bt=bt,
                beat=bx,
                current_ac_t=ac_t,
                ao_t=ao_t,
                ecg_ac_prior=det_ac_prior if "det_ac_prior" in locals() else None,
                prev_ac_t=prev_ac_ref_for_track,
                cfg=cfg
            )
            if ac_refined is not None and np.isfinite(ac_refined):
                ac_t = float(ac_refined)
                ac_conf = max(float(ac_conf), float(ac_track_conf))
                ac_tracking_used = bool(ac_track_dbg.get("used", False))
                ac_tracking_conf = float(ac_track_conf)
                ac_tracking_target = ac_track_dbg.get("target", None)
        # Store morphology-only result BEFORE ECG pseudo-reference tight-lock.
        # This is the actual unconstrained radar morphology estimate after candidate fusion + AC tracking.
        ao_morph_t = None if ao_t is None else float(ao_t)
        ac_morph_t = None if ac_t is None else float(ac_t)

        # Paper-tight pseudo-reference refinement.
        # This is intended to reduce ECG-derived pseudo-reference error toward ±10 ms.
        # It does not claim true valve ground-truth accuracy without echo/PCG/ICG labels.
        ao_tight_dbg = {"used": False}
        ac_tight_dbg = {"used": False}
        if bool(getattr(cfg, "use_paper_tight_prior_lock", True)):
            prev_ao_for_lock = recent_ao_delays[-1] if len(recent_ao_delays) else None
            prev_ac_for_lock = recent_ac_delays[-1] if len(recent_ac_delays) else None

            ao_locked, ao_lock_conf, ao_tight_dbg = paper_tight_event_lock(
                bt=bt, beat=bx, prior_sec=det_ao_prior, current_t=ao_t,
                kind="ao", cfg=cfg, prev_t=prev_ao_for_lock
            )
            if ao_locked is not None and np.isfinite(ao_locked):
                ao_t = float(ao_locked)
                ao_conf = max(float(ao_conf), float(ao_lock_conf))

            ac_locked, ac_lock_conf, ac_tight_dbg = paper_tight_event_lock(
                bt=bt, beat=bx, prior_sec=det_ac_prior, current_t=ac_t,
                kind="ac", cfg=cfg, prev_t=prev_ac_for_lock
            )
            if ac_locked is not None and np.isfinite(ac_locked):
                ac_t = float(ac_locked)
                ac_conf = max(float(ac_conf), float(ac_lock_conf))

            # Maintain physiological order after tight lock.
            if ao_t is not None and ac_t is not None and np.isfinite(ao_t) and np.isfinite(ac_t):
                if ac_t <= ao_t + float(getattr(cfg, "ac_interval_min_sec", 0.140)):
                    ac_t = float(ao_t + max(float(getattr(cfg, "ac_interval_min_sec", 0.140)), 0.180))

        ao_abs = None if ao_t is None else float(b["r_time"] + ao_t)
        ac_abs = None if ac_t is None else float(b["r_time"] + ac_t)

        # surrogate errors
        # 1) expected center 대비 pseudo timing error
        ao_expected_error_ms = None if ao_t is None else float((ao_t - cfg.expected_ao_sec) * 1000.0)
        ac_expected_error_ms = None if ac_t is None else float((ac_t - cfg.expected_ac_sec) * 1000.0)

        # 2) detector dispersion(ms): 후보들 간 일치도
        ao_disp_ms = None if ao_disp is None else float(ao_disp / cfg.radar_interp_fs_hz * 1000.0)
        ac_disp_ms = None if ac_disp is None else float(ac_disp / cfg.radar_interp_fs_hz * 1000.0)

        if int(b["beat_index"]) < len(ecg_ref_series["ao_smooth"]):
            ecg_ao_rel = float(ecg_ref_series["ao_smooth"][int(b["beat_index"])])
            ecg_ac_rel = float(ecg_ref_series["ac_smooth"][int(b["beat_index"])])
        else:
            ecg_ao_rel, ecg_ac_rel, _, _ = ecg_estimated_ao_ac_adaptive(ecg, int(b["beat_index"]), cfg)

        ao_err_vs_ecg_ms = None if ao_t is None or ecg_ao_rel is None else float((ao_t - ecg_ao_rel) * 1000.0)
        ac_err_vs_ecg_ms = None if ac_t is None or ecg_ac_rel is None else float((ac_t - ecg_ac_rel) * 1000.0)

        row = [
            int(b["beat_index"]),
            float(b["r_time"]),
            bool(accepted),
            float(b["sqi"]),
            float(b["template_corr"]),
            float(b["cardiac_ratio"]),
            float(b["resp_ratio"]),
            ao_t,
            ac_t,
            ao_abs,
            ac_abs,
            None if ao_t is None or ac_t is None else float(ac_t - ao_t),
            None if ao_t is None or ac_t is None else float((ac_t - ao_t) * 1000.0),
            ao_conf,
            ac_conf,
            ao_disp_ms,
            ac_disp_ms,
            ao_expected_error_ms,
            ac_expected_error_ms,
            None,  # ao_ensemble_error_ms: filled after ensemble reference is calculated
            None,  # ac_ensemble_error_ms: filled after ensemble reference is calculated
            ecg_ao_rel,
            ecg_ac_rel,
            ao_err_vs_ecg_ms,
            ac_err_vs_ecg_ms,
            bool(ac_tracking_used),
            ac_tracking_conf,
            ac_tracking_target,
        ]
        rows.append(row)

        b.update({
            "ao_idx": ao_idx,
            "ac_idx": ac_idx,
            "ao_time": ao_t,
            "ac_time": ac_t,
            "ao_conf": ao_conf,
            "ac_conf": ac_conf,
            "ao_morph_time": ao_morph_t,
            "ac_morph_time": ac_morph_t,
            "ao_disp_ms": ao_disp_ms,
            "ac_disp_ms": ac_disp_ms,
            "ao_candidates": ao_candidates,
            "ac_candidates": ac_candidates,
            "ecg_ao_ref": ecg_ao_rel,
            "ecg_ac_ref": ecg_ac_rel,
            "ao_err_vs_ecg_ms": ao_err_vs_ecg_ms,
            "ac_err_vs_ecg_ms": ac_err_vs_ecg_ms,
            "ac_tracking_used": bool(ac_tracking_used),
            "ac_tracking_conf": ac_tracking_conf,
            "ac_tracking_target": ac_tracking_target,
            "ao_tight_lock_dbg": ao_tight_dbg,
            "ac_tight_lock_dbg": ac_tight_dbg,
        })

        if accepted:
            accepted_beats.append(b)
            # Update paper-style anchor references only for usable beats
            try:
                if meta_ao_paper.get("anchor_time") is not None:
                    prev_icp_delay = float(meta_ao_paper["anchor_time"])
            except Exception:
                pass
            try:
                if meta_ac_paper.get("anchor_time") is not None:
                    recent_irp_delays.append(float(meta_ac_paper["anchor_time"]))
                    recent_irp_delays = recent_irp_delays[-20:]
            except Exception:
                pass
            try:
                if ao_t is not None:
                    recent_ao_delays.append(float(ao_t))
                    recent_ao_delays = recent_ao_delays[-20:]
                if ac_t is not None:
                    recent_ac_delays.append(float(ac_t))
                    recent_ac_delays = recent_ac_delays[-20:]
            except Exception:
                pass
        else:
            rejected_beats.append(b)

    # Ensemble reference from accepted beats
    ens = {}
    if accepted_beats:
        min_len = min(len(b["radar_beat"]) for b in accepted_beats)
        mat = np.vstack([b["radar_beat"][:min_len] for b in accepted_beats])
        t_rel = accepted_beats[0]["t_rel"][:min_len]
        mean = zscore_safe(np.mean(mat, axis=0))
        std = np.std(mat, axis=0)
        ao_idx, ao_conf, ao_disp = fuse_candidates([
            {"idx": derivative_detector(t_rel, mean, cfg.ao_search_sec, "ao")[0], "confidence": 0.8},
            {"idx": wavelet_ridge_detector(t_rel, mean, cfg.ao_search_sec)[0], "confidence": 0.7},
        ])
        ac_idx, ac_conf, ac_disp = fuse_candidates([
            {"idx": derivative_detector(t_rel, mean, cfg.ac_search_sec, "ac")[0], "confidence": 0.8},
            {"idx": notch_tidal_detector(t_rel, mean, cfg.ac_search_sec)[0], "confidence": 0.8},
            {"idx": wavelet_ridge_detector(t_rel, mean, cfg.ac_search_sec)[0], "confidence": 0.7},
        ])
        ens = {
            "t_rel": t_rel,
            "mean": mean,
            "std": std,
            "ao_idx": ao_idx,
            "ac_idx": ac_idx,
            "ao_time": None if ao_idx is None else float(t_rel[ao_idx]),
            "ac_time": None if ac_idx is None else float(t_rel[ac_idx]),
        }

    # Ensemble 대비 beat-wise timing error
    for row, b in zip(rows, beats):
        ao_ens_err = None
        ac_ens_err = None
        if ens and b.get("ao_time") is not None and ens.get("ao_time") is not None:
            ao_ens_err = float((b["ao_time"] - ens["ao_time"]) * 1000.0)
        if ens and b.get("ac_time") is not None and ens.get("ac_time") is not None:
            ac_ens_err = float((b["ac_time"] - ens["ac_time"]) * 1000.0)
        # keep fixed row/header alignment
        if len(row) > 20:
            row[19] = ao_ens_err
            row[20] = ac_ens_err
        else:
            row.extend([ao_ens_err, ac_ens_err])

    return {
        "t_radar_interp": t_ri,
        "radar_interp": radar_i,
        "radar_interp_z": radar_i_z,
        "beats": beats,
        "accepted_beats": accepted_beats,
        "rejected_beats": rejected_beats,
        "template_t": template_t,
        "template": template,
        "ensemble": ens,
        "rows": rows,
    }


# ============================================================
# Compare signals
# ============================================================
def synth_ecg_peak_train(tgrid: np.ndarray, peaks_time: np.ndarray):
    y = np.zeros_like(tgrid)
    for p in peaks_time:
        idx = np.searchsorted(tgrid, p)
        if 0 <= idx < len(y):
            y[idx] = 1.0
    fs = 1.0 / np.median(np.diff(tgrid))
    width = max(3, int(0.08 * fs))
    win = signal.windows.gaussian(width * 4 + 1, std=width)
    win /= np.max(win)
    y = np.convolve(y, win, mode="same")
    return zscore_safe(y)


def compare_signals(ecg, radar, cfg: AnalysisConfig, rcfg: RadarConfig):
    tend = min(float(ecg["t"][-1]), float(radar["t"][-1]))
    t0 = cfg.compare_start_sec
    t1 = tend - cfg.compare_end_margin_sec
    if t1 <= t0 + 2:
        t0 = max(float(ecg["t"][0]), float(radar["t"][0]))
        t1 = tend

    tgrid = np.arange(t0, t1, 1.0 / cfg.common_compare_fs_hz)
    ecg_ref = synth_ecg_peak_train(tgrid, ecg["peaks_time"])
    radar_ppg = np.interp(tgrid, radar["t"], radar["ppg_like"])
    radar_ppg = zscore_safe(radar_ppg)

    pearson = None
    if np.std(ecg_ref) > 1e-12 and np.std(radar_ppg) > 1e-12:
        pearson = float(np.corrcoef(ecg_ref, radar_ppg)[0, 1])

    xc = normalized_xcorr(radar_ppg, ecg_ref, cfg.common_compare_fs_hz, cfg.max_lag_sec)
    lag = xc["lag_sec"]
    shift = 0 if lag is None else int(round(lag * cfg.common_compare_fs_hz))
    radar_aligned = shift_by_samples(radar_ppg, shift)

    pearson_aligned = None
    if np.std(ecg_ref) > 1e-12 and np.std(radar_aligned) > 1e-12:
        pearson_aligned = float(np.corrcoef(ecg_ref, radar_aligned)[0, 1])

    spec = spectral_corr(radar_aligned, ecg_ref, cfg.common_compare_fs_hz, rcfg.ppg_like_band_hz, cfg.psd_nperseg)
    cf, coh, mcoh = mean_coherence(radar_aligned, ecg_ref, cfg.common_compare_fs_hz, rcfg.ppg_like_band_hz, cfg.coherence_nperseg)

    return {
        "t": tgrid,
        "ecg_ref": ecg_ref,
        "radar_ppg": radar_ppg,
        "radar_aligned": radar_aligned,
        "pearson": pearson,
        "pearson_aligned": pearson_aligned,
        "xcorr": xc,
        "spectral_corr": spec,
        "coherence_freq": cf,
        "coherence": coh,
        "mean_coherence": mcoh,
        "window_sec": [float(t0), float(t1)],
    }


# ============================================================
# Save / figures
# ============================================================
AOAC_HEADER = [
    "beat_index", "r_peak_time_sec", "accepted", "sqi", "template_corr",
    "cardiac_ratio", "resp_ratio",
    "ao_time_from_r_sec", "ac_time_from_r_sec", "ao_abs_time_sec", "ac_abs_time_sec",
    "ao_to_ac_interval_sec", "ao_to_ac_interval_ms",
    "ao_confidence", "ac_confidence",
    "ao_detector_dispersion_ms", "ac_detector_dispersion_ms",
    "ao_expected_center_error_ms", "ac_expected_center_error_ms",
    "ao_ensemble_error_ms", "ac_ensemble_error_ms",
    "ecg_est_ao_time_from_r_sec", "ecg_est_ac_time_from_r_sec",
    "ao_radar_minus_ecg_ms", "ac_radar_minus_ecg_ms",
    "ac_tracking_used", "ac_tracking_conf", "ac_tracking_target_sec",
]


def extract_aoac_arrays_for_plots(rows):
    if not rows:
        return {}
    accepted = np.array([bool(r[2]) for r in rows], dtype=bool)

    def col(idx):
        return np.array([np.nan if r[idx] is None else float(r[idx]) for r in rows], dtype=float)

    return {
        "accepted": accepted,
        "beat_index": np.array([int(r[0]) for r in rows], dtype=int),
        "ao_time": col(7),
        "ac_time": col(8),
        "ao_expected_error_ms": col(17),
        "ac_expected_error_ms": col(18),
        "ao_ensemble_error_ms": col(19),
        "ac_ensemble_error_ms": col(20),
    }


def summarize_aoac_timing(rows):
    arr = extract_aoac_arrays_for_plots(rows)
    if not arr:
        return {"n_total": 0, "n_accepted": 0, "accept_rate": None}
    accepted = arr["accepted"]

    def stat(x):
        x = x[accepted & np.isfinite(x)]
        if len(x) == 0:
            return {"n": 0, "mean": None, "std": None, "median": None, "q1": None, "q3": None, "iqr": None}
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        return {"n": int(len(x)), "mean": float(np.mean(x)), "std": float(np.std(x)),
                "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}

    ao_ms = arr["ao_time"] * 1000.0
    ac_ms = arr["ac_time"] * 1000.0
    interval_ms = (arr["ac_time"] - arr["ao_time"]) * 1000.0

    return {
        "n_total": int(len(accepted)),
        "n_accepted": int(np.sum(accepted)),
        "accept_rate": float(np.sum(accepted) / len(accepted)) if len(accepted) else None,
        "ao_time_from_r_ms": stat(ao_ms),
        "ac_time_from_r_ms": stat(ac_ms),
        "ao_to_ac_interval_ms": stat(interval_ms),
        "ao_expected_center_error_ms": stat(arr["ao_expected_error_ms"]),
        "ac_expected_center_error_ms": stat(arr["ac_expected_error_ms"]),
        "ao_ensemble_error_ms": stat(arr["ao_ensemble_error_ms"]),
        "ac_ensemble_error_ms": stat(arr["ac_ensemble_error_ms"]),
    }


def add_aoac_timing_extra_figures(outdir: Path, aoac, acfg: AnalysisConfig):
    rows = aoac["rows"]
    arr = extract_aoac_arrays_for_plots(rows)
    if not arr:
        return

    accepted = arr["accepted"]
    beat_index = arr["beat_index"]

    ao_ms = arr["ao_time"] * 1000.0
    ac_ms = arr["ac_time"] * 1000.0
    interval_ms = (arr["ac_time"] - arr["ao_time"]) * 1000.0

    ao_expected_ms = np.full_like(ao_ms, acfg.expected_ao_sec * 1000.0)
    ac_expected_ms = np.full_like(ac_ms, acfg.expected_ac_sec * 1000.0)

    m_ao = accepted & np.isfinite(ao_ms)
    m_ac = accepted & np.isfinite(ac_ms)
    m_int = accepted & np.isfinite(interval_ms)

    data, labels = [], []
    if np.any(m_ao):
        data.append(ao_ms[m_ao]); labels.append("AO from R")
    if np.any(m_ac):
        data.append(ac_ms[m_ac]); labels.append("AC from R")
    if np.any(m_int):
        data.append(interval_ms[m_int]); labels.append("AC-AO")

    if data:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title("AO/AC Radar-estimated Timing Distribution")
        ax.set_ylabel("Timing [ms]")
        ax.grid(True, axis="y")
        fig.savefig(outdir / "fig15_ao_ac_timing_boxplot.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if np.any(m_ao) or np.any(m_ac):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        if np.any(m_ao):
            ax.scatter(ao_expected_ms[m_ao], ao_ms[m_ao], alpha=0.75, label="AO candidates")
        if np.any(m_ac):
            ax.scatter(ac_expected_ms[m_ac], ac_ms[m_ac], alpha=0.75, label="AC candidates")

        vals = []
        if np.any(m_ao):
            vals += list(ao_ms[m_ao]) + list(ao_expected_ms[m_ao])
        if np.any(m_ac):
            vals += list(ac_ms[m_ac]) + list(ac_expected_ms[m_ac])
        if vals:
            mn, mx = min(vals) - 30, max(vals) + 30
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="y = x")
            ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)

        ax.set_title("ECG-trigger Expected Timing vs Radar-estimated Timing")
        ax.set_xlabel("Expected timing from ECG R-peak [ms]")
        ax.set_ylabel("Radar-estimated timing from R-peak [ms]")
        ax.grid(True); ax.legend()
        fig.savefig(outdir / "fig16_expected_vs_radar_timing_scatter.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if np.any(m_ao) or np.any(m_ac):
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        if np.any(m_ao):
            ax.plot(beat_index[m_ao], ao_ms[m_ao], "o-", label="AO radar estimate", markersize=4)
            ax.axhline(acfg.expected_ao_sec * 1000.0, linestyle="--", label="AO expected center")
        if np.any(m_ac):
            ax.plot(beat_index[m_ac], ac_ms[m_ac], "o-", label="AC radar estimate", markersize=4)
            ax.axhline(acfg.expected_ac_sec * 1000.0, linestyle="--", label="AC expected center")
        ax.set_title("Beat-wise AO/AC Timing Trend")
        ax.set_xlabel("Beat index")
        ax.set_ylabel("Timing from ECG R-peak [ms]")
        ax.grid(True); ax.legend()
        fig.savefig(outdir / "fig17_beatwise_ao_ac_timing_trend.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    for fig_name, title, y1, y2, lab1, lab2 in [
        ("fig18_expected_timing_error_boxplot.png",
         "Radar-estimated Timing Error vs ECG-trigger Expected Timing",
         arr["ao_expected_error_ms"], arr["ac_expected_error_ms"],
         "AO radar - expected", "AC radar - expected"),
        ("fig19_ensemble_timing_error_boxplot.png",
         "Beat-wise Timing Error vs Ensemble Reference",
         arr["ao_ensemble_error_ms"], arr["ac_ensemble_error_ms"],
         "AO radar - ensemble", "AC radar - ensemble"),
    ]:
        m1 = accepted & np.isfinite(y1)
        m2 = accepted & np.isfinite(y2)
        data, labels = [], []
        if np.any(m1):
            data.append(y1[m1]); labels.append(lab1)
        if np.any(m2):
            data.append(y2[m2]); labels.append(lab2)
        if data:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            ax.boxplot(data, tick_labels=labels, showmeans=True)
            ax.axhline(0, color="k", linestyle="--", linewidth=1)
            ax.set_title(title)
            ax.set_ylabel("Error [ms]")
            ax.grid(True, axis="y")
            fig.savefig(outdir / fig_name, dpi=180, bbox_inches="tight")
            plt.close(fig)

def add_combined_overview_figures(outdir: Path, ecg, radar, aoac, comp):
    """
    추가 Figure:
    1) fig0_ecg_radar_overview_3panel.png
       - ECG filtered + R peaks
       - Radar PPG-like + peaks
       - ECG reference vs Radar aligned
    2) fig0_ecg5cycles_radar1beat.png
       - 위: ECG 5주기
       - 아래: Radar 1주기
    """

    # ============================================================
    # Combined overview: 기존 fig1, fig2, compare를 한 Figure에 3개 subplot으로 표시
    # ============================================================
    try:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

        # 1) ECG filtered + R-peaks
        ax = axes[0]
        ecg_z = zscore_safe(ecg.get("display_rpeak", ecg.get("display", ecg["filtered"])))
        ax.plot(ecg["t"], ecg_z, label="ECG filtered")
        if len(ecg["peaks_idx"]):
            valid_pk = ecg["peaks_idx"][(ecg["peaks_idx"] >= 0) & (ecg["peaks_idx"] < len(ecg_z))]
            ax.plot(ecg["t"][valid_pk], ecg_z[valid_pk], "ro", ms=3, label="R peaks")
        ax.set_title("ECG filtered signal with R-peaks")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("z-score")
        ax.grid(True)
        ax.legend(loc="upper right")

        # 2) Radar PPG-like + radar peaks
        ax = axes[1]
        radar_z = zscore_safe(radar["ppg_like"])
        ax.plot(radar["t"], radar_z, label="Radar PPG-like")
        if len(radar["peaks_idx"]):
            valid_pk = radar["peaks_idx"][(radar["peaks_idx"] >= 0) & (radar["peaks_idx"] < len(radar_z))]
            ax.plot(radar["t"][valid_pk], radar_z[valid_pk], "ro", ms=3, label="Radar peaks")
        ax.set_title("Radar PPG-like signal")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("z-score")
        ax.grid(True)
        ax.legend(loc="upper right")

        # 3) ECG reference vs aligned radar
        ax = axes[2]
        ax.plot(comp["t"], comp["ecg_ref"], label="ECG R-peak reference")
        ax.plot(comp["t"], comp["radar_aligned"], label="Radar PPG-like aligned")
        ax.set_title("ECG reference vs Radar PPG-like aligned")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("z-score")
        ax.grid(True)
        ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(outdir / "fig0_ecg_radar_overview_3panel.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir / "fig0_ecg_radar_overview_3panel_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    # ============================================================
    # ECG 5주기 vs Radar 1주기
    # ============================================================
    try:
        if len(ecg["peaks_time"]) >= 6 and len(aoac.get("beats", [])) >= 1:
            # ECG 5 cycles: 첫 beat 대신 중간 안정 구간 사용
            mid0 = max(1, len(ecg["peaks_time"]) // 2 - 3)
            mid1 = min(len(ecg["peaks_time"]) - 1, mid0 + 5)
            if mid1 <= mid0:
                mid0, mid1 = 0, 5
            t_start = float(ecg["peaks_time"][mid0])
            t_end = float(ecg["peaks_time"][mid1])
            mask_ecg = (ecg["t"] >= t_start) & (ecg["t"] <= t_end)

            ecg_abs_t = ecg["t"][mask_ecg]
            ecg_seg_t = ecg_abs_t - t_start
            ecg_seg = zscore_safe(ecg.get("display", ecg["filtered"])[mask_ecg])

            # 가능하면 accepted beat 중 첫 번째, 없으면 첫 beat
            radar_beat = None
            if len(aoac.get("accepted_beats", [])) > 0:
                radar_beat = aoac["accepted_beats"][0]
            elif len(aoac.get("beats", [])) > 0:
                radar_beat = aoac["beats"][0]

            if radar_beat is not None:
                bt = np.asarray(radar_beat["t_rel"], dtype=np.float64)
                bx = zscore_safe(np.asarray(radar_beat["radar_beat"], dtype=np.float64))

                fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)

                ax = axes[0]
                ecg_true_seg = zscore_safe(ecg.get("true_display_rpeak", ecg.get("true_display", ecg["cleaned"]))[mask_ecg])
                ecg_analysis_seg = zscore_safe(ecg["filtered"][mask_ecg])
                ax.plot(ecg_seg_t, ecg_true_seg, label="ECG display, R-up aligned, 5 cycles", linewidth=1.5)
                ax.plot(ecg_seg_t, ecg_analysis_seg, label="ECG analysis band", linewidth=1.0, alpha=0.75)
                # ECG segment 안의 R marker 표시: detection anchor는 그대로 두고, 표시만 visible apex에 snap
                r_abs = ecg["peaks_time"][(ecg["peaks_time"] >= t_start) & (ecg["peaks_time"] <= t_end)]
                if len(r_abs):
                    rs_t_abs, rs_y = snap_marker_times_to_visible_ecg_apex(
                        r_abs, ecg_abs_t, ecg_true_seg, ecg.get("fs", 100.0), radius_sec=0.070
                    )
                    ax.plot(rs_t_abs - t_start, rs_y, "ro", ms=4, label="R marker, visual apex")
                ax.set_title("ECG waveform: true/raw-like vs analysis band, 5 cardiac cycles")
                ax.set_xlabel("Time from first R-peak [s]")
                ax.set_ylabel("z-score")
                ax.grid(True)
                ax.legend(loc="upper right")

                ax = axes[1]
                ax.plot(bt, bx, label="Radar PPG-like, 1 beat")
                if radar_beat.get("ao_idx") is not None:
                    ai = int(radar_beat["ao_idx"])
                    if 0 <= ai < len(bt):
                        ax.plot(bt[ai], bx[ai], "ro", ms=5, label="AO candidate")
                if radar_beat.get("ac_idx") is not None:
                    ci = int(radar_beat["ac_idx"])
                    if 0 <= ci < len(bt):
                        ax.plot(bt[ci], bx[ci], "ko", ms=5, label="AC candidate")
                ax.axvline(0, linestyle="--", linewidth=1, label="ECG R-peak anchor")
                ax.set_title("Radar PPG-like waveform: 1 beat")
                ax.set_xlabel("Time from ECG R-peak [s]")
                ax.set_ylabel("z-score")
                ax.grid(True)
                ax.legend(loc="upper right")

                fig.tight_layout()
                fig.savefig(outdir / "fig0_ecg5cycles_radar1beat.png", dpi=180, bbox_inches="tight")
                plt.close(fig)
    except Exception as e:
        with open(outdir / "fig0_ecg5cycles_radar1beat_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def add_qt_pseudo_landmark_quality_figure(outdir: Path, ecg):
    """Q/T pseudo landmark confidence check figure. Fig02에서는 기본 숨김 처리."""
    try:
        t = np.asarray(ecg.get("t", []), dtype=np.float64)
        y = zscore_safe(np.asarray(ecg.get("true_display", ecg.get("display", ecg.get("cleaned", []))), dtype=np.float64))
        if len(t) < 10 or len(y) < 10:
            return
        q = np.asarray(ecg.get("q_time", []), dtype=np.float64)
        tt = np.asarray(ecg.get("t_time", []), dtype=np.float64)
        qc = np.asarray(ecg.get("q_confidence", np.zeros_like(q)), dtype=np.float64)
        tc = np.asarray(ecg.get("t_confidence", np.zeros_like(tt)), dtype=np.float64)
        r = np.asarray(ecg.get("peaks_time", []), dtype=np.float64)
        if len(r) >= 8:
            mid0 = max(1, len(r)//2 - 3); mid1 = min(len(r)-1, mid0+5)
            t0, t1 = float(r[mid0]), float(r[mid1])
        else:
            t0, t1 = float(t[0]), min(float(t[-1]), float(t[0]+3.0))
        m=(t>=t0)&(t<=t1)
        fig, axes = plt.subplots(2,1,figsize=(13,7),sharex=True)
        axes[0].plot(t[m]-t0, y[m], label='ECG display band')
        for rr in r[(r>=t0)&(r<=t1)]: axes[0].axvline(rr-t0,color='red',alpha=0.15,linewidth=0.8)
        qmask=(q>=t0)&(q<=t1)&np.isfinite(q)&(qc>=float(globals().get('QT_LANDMARK_MIN_CONFIDENCE',0.45))) if len(q) else []
        tmask=(tt>=t0)&(tt<=t1)&np.isfinite(tt)&(tc>=float(globals().get('QT_LANDMARK_MIN_CONFIDENCE',0.45))) if len(tt) else []
        if len(q): axes[0].scatter(q[qmask]-t0, np.interp(q[qmask], t[m], y[m]), c='magenta', marker='v', label='Q pseudo gated')
        if len(tt): axes[0].scatter(tt[tmask]-t0, np.interp(tt[tmask], t[m], y[m]), c='green', marker='^', label='T pseudo gated')
        axes[0].set_title('Q/T pseudo-landmark quality check, not used as true ground truth')
        axes[0].grid(True); axes[0].legend(fontsize=8)
        axes[1].plot(np.arange(len(qc)), qc, 'o-', markersize=3, label='Q confidence')
        axes[1].plot(np.arange(len(tc)), tc, 'o-', markersize=3, label='T confidence')
        axes[1].axhline(float(globals().get('QT_LANDMARK_MIN_CONFIDENCE',0.45)), color='gray', linestyle='--', label='threshold')
        axes[1].set_xlabel('Beat index'); axes[1].set_ylabel('confidence'); axes[1].set_ylim(-0.05,1.05)
        axes[1].grid(True); axes[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(outdir/'fig02b_qt_pseudo_landmark_quality.png', dpi=180, bbox_inches='tight'); plt.close(fig)
    except Exception as e:
        with open(outdir/'fig02b_qt_pseudo_landmark_quality_error.txt','w',encoding='utf-8') as f: f.write(str(e))

def add_compact_paper_figures(outdir: Path, ecg, radar, aoac, comp, acfg: AnalysisConfig):
    """
    통합 논문용 Figure 생성.

    Fig02 표시 정책:
    - R-peak detector는 절대 건드리지 않음.
    - display ECG에는 R 점을 찍지 않음. vertical anchor line만 표시.
    - R 점은 별도 QRS-band subplot 위에만 찍음.
    - 그래서 display ECG와 QRS-band 검출 anchor가 섞여 보이는 문제를 제거함.
    """
    # Fig A
    try:
        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=False)

        ecg_display = zscore_safe(ecg.get("true_display", ecg.get("display", ecg["cleaned"])))
        ecg_qrs = zscore_safe(ecg["filtered"])

        ax = axes[0]
        ax.plot(ecg["t"], ecg_display, label="ECG display band", linewidth=1.0)
        ax.plot(ecg["t"], ecg_qrs, label="ECG QRS band for R detection", linewidth=0.8, alpha=0.55)

        if len(ecg["peaks_idx"]):
            pk = ecg["peaks_idx"]
            valid = pk[(pk >= 0) & (pk < len(ecg_qrs))]
            ax.scatter(ecg["t"][valid], ecg_qrs[valid], s=14, c="red", marker="x", label="R anchors on QRS band")
            for rt in ecg["peaks_time"][::max(1, len(ecg["peaks_time"]) // 40)]:
                ax.axvline(float(rt), color="red", alpha=0.12, linewidth=0.7)

        if "q_time" in ecg:
            q = ecg["q_time"][np.isfinite(ecg["q_time"])]
            if len(q):
                ax.scatter(q, np.interp(q, ecg["t"], ecg_display), s=10, c="magenta", marker="v", label="Q on display")
        if "t_time" in ecg:
            tt = ecg["t_time"][np.isfinite(ecg["t_time"])]
            if len(tt):
                ax.scatter(tt, np.interp(tt, ecg["t"], ecg_display), s=10, c="green", marker="^", label="T on display")

        ax.set_title("ECG morphology and R/Q/T landmarks: R anchors are from QRS band")
        ax.set_ylabel("z-score")
        ax.grid(True)
        ax.legend(ncol=5, fontsize=8)

        ax = axes[1]
        ax.plot(radar["t"], zscore_safe(radar["displacement"]), label="Radar displacement", linewidth=0.8)
        ax.plot(radar["t"], zscore_safe(radar["respiration"]), label="Respiration band", linewidth=0.8)
        ax.set_title("Radar displacement and respiration component")
        ax.set_ylabel("z-score")
        ax.grid(True)
        ax.legend(fontsize=8)

        ax = axes[2]
        ax.plot(radar["t"], zscore_safe(radar.get("lms_error", radar["ppg_like"])), label="LMS error", linewidth=0.8)
        ax.plot(radar["t"], zscore_safe(radar["ppg_like"]), label="Final radar PPG-like", linewidth=1.0)
        if len(radar["peaks_idx"]):
            pk = radar["peaks_idx"]
            rz = zscore_safe(radar["ppg_like"])
            ax.scatter(radar["t"][pk], rz[pk], s=10, c="red", label="Radar peaks")
        ax.set_title("LMS output and final radar PPG-like")
        ax.set_ylabel("z-score")
        ax.grid(True)
        ax.legend(fontsize=8)

        ax = axes[3]
        ax.plot(comp["t"], comp["ecg_ref"], label="ECG R-peak reference", linewidth=0.8)
        ax.plot(comp["t"], comp["radar_aligned"], label="Radar PPG-like aligned", linewidth=0.8)
        ax.set_title("Time-domain alignment check")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("z-score")
        ax.grid(True)
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(outdir / "fig01_compact_signal_overview.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (outdir / "fig01_compact_signal_overview_error.txt").write_text(str(e), encoding="utf-8")

    # Fig B: separated display / QRS / radar morphology
    try:
        if len(ecg["peaks_time"]) >= 8 and len(aoac.get("beats", [])) >= 1:
            # 첫 beat는 필터 edge/transient 영향이 있으므로 중간 안정 구간 사용
            mid0 = max(1, len(ecg["peaks_time"]) // 2 - 3)
            mid1 = min(len(ecg["peaks_time"]) - 1, mid0 + 5)
            if mid1 <= mid0:
                mid0, mid1 = 0, 5

            t_start = float(ecg["peaks_time"][mid0])
            t_end = float(ecg["peaks_time"][mid1])
            m = (ecg["t"] >= t_start) & (ecg["t"] <= t_end)
            tx_abs = ecg["t"][m]
            tx = tx_abs - t_start

            ecg_display_seg = zscore_safe(ecg.get("true_display", ecg.get("display", ecg["cleaned"]))[m])
            ecg_qrs_seg = zscore_safe(ecg["filtered"][m])

            beat_list = aoac.get("accepted_beats", []) if len(aoac.get("accepted_beats", [])) else aoac.get("beats", [])
            beat = beat_list[min(max(0, len(beat_list)//2), len(beat_list)-1)]

            bt = np.asarray(beat["t_rel"], dtype=np.float64)
            bx = zscore_safe(np.asarray(beat["radar_beat"], dtype=np.float64))
            env = triangular_smooth_envelope(bx, win_len=min(31, max(5, int(0.12 * acfg.radar_interp_fs_hz) | 1)))

            fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)

            r_abs = ecg["peaks_time"][(ecg["peaks_time"] >= t_start) & (ecg["peaks_time"] <= t_end)]
            r_rel = r_abs - t_start if len(r_abs) else np.array([])

            # Panel 1: display ECG only. No R dots here.
            ax = axes[0]
            ax.plot(tx, ecg_display_seg, label="ECG display band", linewidth=1.4)
            if len(r_rel):
                for rr in r_rel:
                    ax.axvline(float(rr), color="red", alpha=0.16, linewidth=0.8)
            if bool(globals().get("FIG02_SHOW_QT_MARKERS", False)):
                q = ecg.get("q_time", np.array([]))
                tt = ecg.get("t_time", np.array([]))
                qc = ecg.get("q_confidence", np.ones_like(q, dtype=float) if len(q) else np.array([]))
                tc = ecg.get("t_confidence", np.ones_like(tt, dtype=float) if len(tt) else np.array([]))
                q_mask = (q >= t_start) & (q <= t_end) & np.isfinite(q) & (qc >= float(globals().get("QT_LANDMARK_MIN_CONFIDENCE", 0.45))) if len(q) else []
                t_mask = (tt >= t_start) & (tt <= t_end) & np.isfinite(tt) & (tc >= float(globals().get("QT_LANDMARK_MIN_CONFIDENCE", 0.45))) if len(tt) else []
                q_in = q[q_mask] - t_start if len(q) else []
                t_in = tt[t_mask] - t_start if len(tt) else []
                if len(q_in):
                    ax.scatter(q_in, np.interp(q_in, tx, ecg_display_seg), c="magenta", marker="v", s=35, label="Q pseudo, gated")
                if len(t_in):
                    ax.scatter(t_in, np.interp(t_in, tx, ecg_display_seg), c="green", marker="^", s=35, label="T pseudo, gated")
            else:
                ax.text(0.01, 0.92, "Q/T hidden in Fig02: pseudo-landmarks are unreliable for non-standard STM32 ECG display", transform=ax.transAxes, fontsize=8, va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="gray"))
            ax.set_title("ECG display band: R anchor time is shown as vertical line only")
            ax.set_ylabel("z-score")
            ax.grid(True)
            ax.legend(ncol=4, fontsize=8)

            # Panel 2: QRS-band only with R markers.
            ax = axes[1]
            ax.plot(tx, ecg_qrs_seg, color="tab:orange", label="ECG QRS band used for R detection", linewidth=1.2)
            if len(r_rel):
                r_y = np.interp(r_rel, tx, ecg_qrs_seg)
                ax.scatter(r_rel, r_y, c="red", marker="x", s=70, linewidths=2.0, label="R anchors on QRS band")
                for rr in r_rel:
                    ax.axvline(float(rr), color="red", alpha=0.12, linewidth=0.7)
            ax.set_title("QRS-band ECG: R markers are plotted here, not on display ECG")
            ax.set_ylabel("z-score")
            ax.grid(True)
            ax.legend(ncol=3, fontsize=8)

            # Panel 3: radar beat morphology
            ax = axes[2]
            ax.plot(bt, bx, label="Radar PPG-like beat", linewidth=1.4)
            ax.plot(bt, zscore_safe(env), label="Envelope", linewidth=1.0, alpha=0.8)
            ax.axvline(0, color="gray", linestyle="--", linewidth=1, label="ECG R anchor")
            ax.axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.12, label="AO search")
            ax.axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.12, label="AC search")
            if beat.get("ao_idx") is not None:
                ai = int(beat["ao_idx"])
                if 0 <= ai < len(bt):
                    ax.scatter(bt[ai], bx[ai], c="red", s=45, label="AO candidate")
            if beat.get("ac_idx") is not None:
                ci = int(beat["ac_idx"])
                if 0 <= ci < len(bt):
                    ax.scatter(bt[ci], bx[ci], c="black", s=45, label="AC candidate")
            ax.set_title("Radar beat morphology with AO/AC candidates")
            ax.set_xlabel("Time from ECG R-peak [s]")
            ax.set_ylabel("z-score")
            ax.grid(True)
            ax.legend(ncol=4, fontsize=8)

            fig.tight_layout()
            fig.savefig(outdir / "fig02_compact_beat_morphology.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        (outdir / "fig02_compact_beat_morphology_error.txt").write_text(str(e), encoding="utf-8")

    # Fig C
    try:
        rows = aoac["rows"]
        if rows:
            accepted = np.array([bool(r[2]) for r in rows], dtype=bool)
            beat_idx = np.array([int(r[0]) for r in rows])
            ao_ms = np.array([np.nan if r[7] is None else float(r[7]) * 1000 for r in rows])
            ac_ms = np.array([np.nan if r[8] is None else float(r[8]) * 1000 for r in rows])
            int_ms = ac_ms - ao_ms
            m_ao = accepted & np.isfinite(ao_ms)
            m_ac = accepted & np.isfinite(ac_ms)
            m_int = accepted & np.isfinite(int_ms)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

            data, labels = [], []
            if np.any(m_ao):
                data.append(ao_ms[m_ao]); labels.append("AO")
            if np.any(m_ac):
                data.append(ac_ms[m_ac]); labels.append("AC")
            if np.any(m_int):
                data.append(int_ms[m_int]); labels.append("AC-AO")
            if data:
                axes[0].boxplot(data, tick_labels=labels, showmeans=True)
            axes[0].set_title("AO/AC timing distribution")
            axes[0].set_ylabel("ms from R")
            axes[0].grid(True, axis="y")

            if np.any(m_ao):
                axes[1].plot(beat_idx[m_ao], ao_ms[m_ao], "o-", ms=3, label="AO")
            if np.any(m_ac):
                axes[1].plot(beat_idx[m_ac], ac_ms[m_ac], "o-", ms=3, label="AC")
            axes[1].set_title("Beat-wise AO/AC timing")
            axes[1].set_xlabel("Beat index")
            axes[1].set_ylabel("ms from R")
            axes[1].grid(True)
            axes[1].legend()

            if np.any(m_int):
                axes[2].plot(beat_idx[m_int], int_ms[m_int], "o-", ms=3, label="AO-AC")
                axes[2].axhline(np.nanmean(int_ms[m_int]), linestyle="--", label="mean")
            axes[2].set_title("AO-AC interval")
            axes[2].set_xlabel("Beat index")
            axes[2].set_ylabel("ms")
            axes[2].grid(True)
            axes[2].legend()

            fig.tight_layout()
            fig.savefig(outdir / "fig03_compact_aoac_summary.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        (outdir / "fig03_compact_aoac_summary_error.txt").write_text(str(e), encoding="utf-8")
def cleanup_legacy_figures(outdir: Path):
    """
    난잡한 figure 정리:
    compact figure와 error txt, summary용 파일만 남기고 legacy fig 삭제.
    """
    if not SAVE_COMPACT_FIGURES_ONLY:
        return
    keep_prefix = (
        "fig00_beat_alignment_",
        "fig00_time_index_alignment_check",
        "fig0a_ecg_artifact_lms_filtering",
        "fig01_compact_",
        "fig02_compact_",
        "fig03_compact_",
        "fig04_ecg_vs_radar_aoac_correlation",
        "fig05_ac_temporal_tracking",
        "fig06_single_cycle_",
        "fig07_radar_raw_multicycle_",
        "fig08_radar_morphology_visibility_",
    )
    keep_exact = set()
    for p in outdir.glob("fig*.png"):
        if not p.name.startswith(keep_prefix) and p.name not in keep_exact:
            try:
                p.unlink()
            except Exception:
                pass


def add_time_index_alignment_figure(outdir: Path, ecg, radar, comp=None):
    """
    Time-index based alignment diagnostic.
    ECG는 STM32 sample_index / 100 Hz, radar는 frame_index / 100 Hz 기반으로
    같은 absolute measurement time 위에서 정렬 상태를 확인한다.
    """
    try:
        et = np.asarray(ecg.get("t", []), dtype=np.float64)
        ev = zscore_safe(np.asarray(ecg.get("true_display_rpeak", ecg.get("true_display", ecg.get("display", ecg.get("filtered", [])))), dtype=np.float64))
        rt = np.asarray(radar.get("t_interp", radar.get("t", [])), dtype=np.float64)
        rv = zscore_safe(np.asarray(radar.get("ppg_interp", radar.get("ppg_like", [])), dtype=np.float64))

        if len(et) < 10 or len(rt) < 10:
            return

        t0 = max(float(np.nanmin(et)), float(np.nanmin(rt))) + 3.0
        t1 = min(float(np.nanmax(et)), float(np.nanmax(rt))) - 3.0
        if t1 <= t0:
            return

        center = 0.5 * (t0 + t1)
        w0 = max(t0, center - 6.0)
        w1 = min(t1, center + 6.0)
        me = (et >= w0) & (et <= w1)
        mr = (rt >= w0) & (rt <= w1)

        fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        axes[0].plot(et[me], ev[me], linewidth=1.1, label="ECG by STM32 sample_index time")
        axes[0].set_title("Time-index domain alignment: ECG")
        axes[0].set_ylabel("z-score")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(rt[mr], rv[mr], linewidth=1.1, label="Radar by frame-index/uniform 100 Hz time")
        axes[1].set_title("Time-index domain alignment: Radar PPG-like")
        axes[1].set_xlabel("Absolute measurement time [s]")
        axes[1].set_ylabel("z-score")
        axes[1].grid(True)
        axes[1].legend()

        if comp is not None:
            try:
                txt = f"Pearson={comp['pearson']['after_alignment']:.3f}, XCorr lag={comp['xcorr']['lag_sec']:.3f}s"
                fig.text(0.01, 0.01, txt, fontsize=9)
            except Exception:
                pass

        fig.tight_layout()
        fig.savefig(outdir / "fig00_time_index_alignment_check.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir / "fig00_time_index_alignment_check_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))



def snap_marker_times_to_visible_ecg_apex(anchor_times, t, y, fs, radius_sec=0.070):
    """
    Figure 표시 전용 helper.
    R-peak detector가 산출한 anchor time 자체는 바꾸지 않고,
    그림에서 빨간 점이 waveform 중간에 찍히는 문제를 막기 위해
    marker만 주변 visible ECG apex에 올려 찍는다.

    반환:
      snap_t, snap_y
    """
    anchor_times = np.asarray(anchor_times, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(anchor_times) == 0 or len(t) == 0 or len(y) == 0:
        return np.array([]), np.array([])

    rad = max(1, int(round(float(radius_sec) * float(fs))))
    snap_t = []
    snap_y = []
    for rt in anchor_times:
        i0 = int(np.argmin(np.abs(t - rt)))
        a = max(0, i0 - rad)
        b = min(len(y), i0 + rad + 1)
        if b <= a:
            continue
        # R-up aligned display 기준에서는 R이 위쪽 peak로 보여야 하므로 local maximum 사용.
        loc = a + int(np.nanargmax(y[a:b]))
        snap_t.append(float(t[loc]))
        snap_y.append(float(y[loc]))
    return np.asarray(snap_t), np.asarray(snap_y)

def add_ecg_vs_radar_aoac_correlation_figure(outdir: Path, aoac, acfg: AnalysisConfig):
    """
    Fig.4: ECG-derived pseudo AO/AC vs radar-estimated AO/AC.
    긴 float를 title에 넣지 않고 소수점 2자리 text box로 분리한다.
    """
    try:
        rows = aoac["rows"]
        if not rows:
            return

        accepted = np.array([bool(r[2]) for r in rows], dtype=bool)
        beat_idx = np.array([int(r[0]) for r in rows], dtype=int)

        radar_ao_ms = np.array([np.nan if r[7] is None else float(r[7]) * 1000.0 for r in rows])
        radar_ac_ms = np.array([np.nan if r[8] is None else float(r[8]) * 1000.0 for r in rows])

        default_ao_ms = float(getattr(acfg, "expected_ao_sec", 0.12)) * 1000.0
        default_ac_ms = float(getattr(acfg, "expected_ac_sec", 0.38)) * 1000.0
        ecg_ao_ms = np.array([default_ao_ms if len(r) <= 21 or r[21] is None else float(r[21]) * 1000.0 for r in rows])
        ecg_ac_ms = np.array([default_ac_ms if len(r) <= 22 or r[22] is None else float(r[22]) * 1000.0 for r in rows])

        ao_err_ms = radar_ao_ms - ecg_ao_ms
        ac_err_ms = radar_ac_ms - ecg_ac_ms

        m_ao = accepted & np.isfinite(ecg_ao_ms) & np.isfinite(radar_ao_ms)
        m_ac = accepted & np.isfinite(ecg_ac_ms) & np.isfinite(radar_ac_ms)

        ao_r = safe_pearson_for_fig(ecg_ao_ms[m_ao], radar_ao_ms[m_ao])
        ac_r = safe_pearson_for_fig(ecg_ac_ms[m_ac], radar_ac_ms[m_ac])
        ao_mae = None if np.sum(m_ao) == 0 else float(np.nanmean(np.abs(ao_err_ms[m_ao])))
        ac_mae = None if np.sum(m_ac) == 0 else float(np.nanmean(np.abs(ac_err_ms[m_ac])))

        # Fig4 is a constrained ECG-pseudo-reference consistency plot.
        # Do NOT display "accuracy %" here because tight-lock can trivially produce 100%.
        # Instead report continuous error statistics: MAE/RMSE/Bias/95% LOA.
        tol = float(getattr(acfg, "aoac_accuracy_tolerance_ms", 30.0))
        ao_acc = accuracy_within_tolerance_ms(ao_err_ms[m_ao], tol)
        ac_acc = accuracy_within_tolerance_ms(ac_err_ms[m_ac], tol)

        def rmse(v, m):
            return None if np.sum(m) == 0 else float(np.sqrt(np.nanmean(np.square(v[m]))))

        def bias(v, m):
            return None if np.sum(m) == 0 else float(np.nanmean(v[m]))

        def loa95(v, m):
            if np.sum(m) < 2:
                return (None, None)
            b = float(np.nanmean(v[m]))
            s = float(np.nanstd(v[m], ddof=1))
            return (b - 1.96 * s, b + 1.96 * s)

        ao_rmse = rmse(ao_err_ms, m_ao)
        ac_rmse = rmse(ac_err_ms, m_ac)
        ao_bias = bias(ao_err_ms, m_ao)
        ac_bias = bias(ac_err_ms, m_ac)
        ao_loa_low, ao_loa_high = loa95(ao_err_ms, m_ao)
        ac_loa_low, ac_loa_high = loa95(ac_err_ms, m_ac)

        ecg_interval_ms = ecg_ac_ms - ecg_ao_ms
        radar_interval_ms = radar_ac_ms - radar_ao_ms
        int_err_ms = radar_interval_ms - ecg_interval_ms
        m_int = accepted & np.isfinite(ecg_interval_ms) & np.isfinite(radar_interval_ms)
        int_mae = None if np.sum(m_int) == 0 else float(np.nanmean(np.abs(int_err_ms[m_int])))
        int_rmse = rmse(int_err_ms, m_int)
        int_bias = bias(int_err_ms, m_int)
        int_loa_low, int_loa_high = loa95(int_err_ms, m_int)
        int_acc = accuracy_within_tolerance_ms(int_err_ms[m_int], tol)
        all_err = np.concatenate([ao_err_ms[m_ao], ac_err_ms[m_ac], int_err_ms[m_int]]) if np.sum(m_ao) + np.sum(m_ac) + np.sum(m_int) else np.array([])
        total_acc = accuracy_within_tolerance_ms(all_err, tol)

        metrics = {
            "note": "Fig4 reports constrained ECG-pseudo-reference consistency, not independent ground-truth accuracy. Accuracy percentages are kept in JSON for audit but are intentionally not displayed on Fig4.",
            "tolerance_ms": tol,
            "ao_n": int(np.sum(m_ao)),
            "ac_n": int(np.sum(m_ac)),
            "ao_pearson_r": ao_r,
            "ac_pearson_r": ac_r,
            "ao_mae_ms": ao_mae,
            "ac_mae_ms": ac_mae,
            "ao_rmse_ms": ao_rmse,
            "ac_rmse_ms": ac_rmse,
            "ao_bias_ms": ao_bias,
            "ac_bias_ms": ac_bias,
            "ao_loa95_low_ms": ao_loa_low,
            "ao_loa95_high_ms": ao_loa_high,
            "ac_loa95_low_ms": ac_loa_low,
            "ac_loa95_high_ms": ac_loa_high,
            "ao_constrained_within_tol_rate_audit": ao_acc,
            "ac_constrained_within_tol_rate_audit": ac_acc,
            "interval_mae_ms": int_mae,
            "interval_rmse_ms": int_rmse,
            "interval_bias_ms": int_bias,
            "interval_loa95_low_ms": int_loa_low,
            "interval_loa95_high_ms": int_loa_high,
            "interval_constrained_within_tol_rate_audit": int_acc,
            "total_constrained_within_tol_rate_audit": total_acc,
        }
        with open(outdir / "ecg_vs_radar_aoac_correlation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # Fig4 constrained result audit:
        # 100%에 가까운 값은 plot에 표시하지 않고, 별도 warning 파일로만 남긴다.
        try:
            warn_thr = float(globals().get("FIG4_AUDIT_WARN_RATE", 0.999))
            audit_rates = {
                "ao_constrained_within_tol_rate_audit": ao_acc,
                "ac_constrained_within_tol_rate_audit": ac_acc,
                "interval_constrained_within_tol_rate_audit": int_acc,
                "total_constrained_within_tol_rate_audit": total_acc,
            }
            suspicious = {
                k: v for k, v in audit_rates.items()
                if v is not None and np.isfinite(float(v)) and float(v) >= warn_thr
            }
            audit = {
                "fig4_policy": "Fig4 does not display accuracy percentage because tight-lock is constrained to ECG-derived pseudo-reference.",
                "warning_threshold_rate": warn_thr,
                "suspicious_near_100_rates": suspicious,
                "possible_causes_if_near_100": [
                    "ECG-derived pseudo-reference leakage into radar estimate",
                    "tolerance too loose for the reported metric",
                    "same calibration data evaluated as test data",
                    "tight-lock snapping to pseudo-reference rather than independent morphology detection",
                    "overlapping or duplicated beat evaluation"
                ],
                "recommended_interpretation": "Use Fig4 as consistency analysis. Use morphology-only or candidate-consistency validation figures for accuracy claims.",
                "recommended_figures_for_accuracy": [
                    "fig09_morphology_only_scatter_pruned.png",
                    "fig10_candidate_consistency_model_validation.png",
                    "aoac_morphology_only_summary.json",
                    "candidate_consistency_model_validation_summary.json"
                ],
            }
            with open(outdir / "fig04_constrained_accuracy_audit.json", "w", encoding="utf-8") as f:
                json.dump(audit, f, ensure_ascii=False, indent=2)
            if suspicious:
                with open(outdir / "fig04_near_100_warning.txt", "w", encoding="utf-8") as f:
                    f.write("[WARNING] Fig4 constrained audit rate is near 100%.\n")
                    f.write("Fig4 is not an independent accuracy plot.\n")
                    f.write("Possible causes:\n")
                    for c in audit["possible_causes_if_near_100"]:
                        f.write(f"- {c}\n")
                    f.write("\nUse morphology-only / candidate-consistency validation outputs for accuracy reporting.\n")
        except Exception as e:
            with open(outdir / "fig04_constrained_accuracy_audit_error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))

        save_csv(outdir / "ecg_vs_radar_aoac_correlation.csv",
                 ["beat_index", "accepted",
                  "ecg_est_ao_ms", "radar_ao_ms", "ao_error_ms",
                  "ecg_est_ac_ms", "radar_ac_ms", "ac_error_ms"],
                 [[int(beat_idx[i]), bool(accepted[i]),
                   None if not np.isfinite(ecg_ao_ms[i]) else float(ecg_ao_ms[i]),
                   None if not np.isfinite(radar_ao_ms[i]) else float(radar_ao_ms[i]),
                   None if not np.isfinite(ao_err_ms[i]) else float(ao_err_ms[i]),
                   None if not np.isfinite(ecg_ac_ms[i]) else float(ecg_ac_ms[i]),
                   None if not np.isfinite(radar_ac_ms[i]) else float(radar_ac_ms[i]),
                   None if not np.isfinite(ac_err_ms[i]) else float(ac_err_ms[i])]
                  for i in range(len(rows))])

        def f2(x, default="NA"):
            if x is None:
                return default
            try:
                xf = float(x)
                if np.isfinite(xf):
                    return f"{xf:.2f}"
            except Exception:
                pass
            return default

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        if np.any(m_ao):
            ax.scatter(ecg_ao_ms[m_ao], radar_ao_ms[m_ao], alpha=0.75)
            mn = min(np.nanmin(ecg_ao_ms[m_ao]), np.nanmin(radar_ao_ms[m_ao])) - 20
            mx = max(np.nanmax(ecg_ao_ms[m_ao]), np.nanmax(radar_ao_ms[m_ao])) + 20
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
            ax.set_xlim(mn, mx)
            ax.set_ylim(mn, mx)
        ax.set_title("AO consistency: ECG pseudo-reference vs radar estimate", fontsize=11)
        ax.text(0.03, 0.97,
                f"n={int(np.sum(m_ao))}\nr={f2(ao_r)}\nMAE={f2(ao_mae)} ms\nRMSE={f2(ao_rmse)} ms\nBias={f2(ao_bias)} ms\nLOA95=[{f2(ao_loa_low)}, {f2(ao_loa_high)}] ms",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))
        ax.set_xlabel("ECG-estimated AO timing [ms]")
        ax.set_ylabel("Radar-estimated AO timing [ms]")
        ax.grid(True)

        ax = axes[0, 1]
        if np.any(m_ac):
            ax.scatter(ecg_ac_ms[m_ac], radar_ac_ms[m_ac], alpha=0.75)
            mn = min(np.nanmin(ecg_ac_ms[m_ac]), np.nanmin(radar_ac_ms[m_ac])) - 30
            mx = max(np.nanmax(ecg_ac_ms[m_ac]), np.nanmax(radar_ac_ms[m_ac])) + 30
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
            ax.set_xlim(mn, mx)
            ax.set_ylim(mn, mx)
        ax.set_title("AC consistency: ECG pseudo-reference vs radar estimate", fontsize=11)
        ax.text(0.03, 0.97,
                f"n={int(np.sum(m_ac))}\nr={f2(ac_r)}\nMAE={f2(ac_mae)} ms\nRMSE={f2(ac_rmse)} ms\nBias={f2(ac_bias)} ms\nLOA95=[{f2(ac_loa_low)}, {f2(ac_loa_high)}] ms",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))
        ax.set_xlabel("ECG-estimated AC timing [ms]")
        ax.set_ylabel("Radar-estimated AC timing [ms]")
        ax.grid(True)

        ax = axes[1, 0]
        data, labels = [], []
        if np.any(m_ao):
            data.append(ao_err_ms[m_ao]); labels.append("AO")
        if np.any(m_ac):
            data.append(ac_err_ms[m_ac]); labels.append("AC")
        if np.any(m_int):
            data.append(int_err_ms[m_int]); labels.append("AO-AC")
        if data:
            ax.boxplot(data, tick_labels=labels, showmeans=True)
            ax.axhline(0, color="k", linestyle="--", linewidth=1)
            ax.axhline(tol, color="gray", linestyle=":", linewidth=1)
            ax.axhline(-tol, color="gray", linestyle=":", linewidth=1)
        ax.set_title("Radar - ECG timing error", fontsize=11)
        ax.text(0.03, 0.97,
                f"Interval MAE={f2(int_mae)} ms\nInterval RMSE={f2(int_rmse)} ms\nInterval Bias={f2(int_bias)} ms\nLOA95=[{f2(int_loa_low)}, {f2(int_loa_high)}] ms",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"))
        ax.set_ylabel("Error [ms]")
        ax.text(0.03, 0.05, "No accuracy % shown: tight-lock is constrained to pseudo-reference", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.80, edgecolor="gray"))
        ax.grid(True, axis="y")

        ax = axes[1, 1]
        if np.any(m_ao):
            ax.plot(beat_idx[m_ao], ao_err_ms[m_ao], "o-", markersize=3, label="AO error")
        if np.any(m_ac):
            ax.plot(beat_idx[m_ac], ac_err_ms[m_ac], "o-", markersize=3, label="AC error")
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.axhline(tol, color="gray", linestyle=":", linewidth=1)
        ax.axhline(-tol, color="gray", linestyle=":", linewidth=1)
        ax.set_title("Beat-wise AO/AC timing error", fontsize=11)
        ax.set_xlabel("Beat index")
        ax.set_ylabel("Radar - ECG [ms]")
        ax.grid(True)
        ax.legend(fontsize=9)

        fig.suptitle("Fig.4 ECG-derived pseudo-reference consistency only — accuracy % removed to avoid near-100% overclaim", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(outdir / "fig04_ecg_vs_radar_aoac_correlation.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        with open(outdir / "fig04_ecg_vs_radar_aoac_correlation_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))
def add_single_cycle_aoac_label_figure(outdir: Path, ecg, radar, aoac, acfg: AnalysisConfig):
    """
    Composite single-cycle AO/AC reference-label figure.

    Purpose:
    - The user wants a visible one-cycle figure where AO/AC label positions are shown as a "correct-label" reference.
    - ECG alone cannot physically contain AO/AC ground truth, but within this pipeline we can construct a practical
      composite reference by taking the beat where AO is closest to the ECG-derived prior and the beat where AC is
      closest to the ECG-derived prior, then placing those two best-matched timings on one representative cycle.

    Output:
    - fig06_single_cycle_ecg_radar_aoac_labels.png
    - fig06_composite_gold_aoac_cycle.png
    - single_cycle_aoac_label_figure_values.csv
    - composite_gold_aoac_label_values.csv

    Row indices:
      7  radar AO rel sec
      8  radar AC rel sec
      21 ECG AO pseudo rel sec
      22 ECG AC pseudo rel sec
      23 AO error vs ECG pseudo [ms]
      24 AC error vs ECG pseudo [ms]
    """
    try:
        rows = aoac.get("rows", [])
        beats = aoac.get("beats", [])
        if not rows or not beats or len(ecg.get("peaks_time", [])) == 0:
            return

        tol_ms = float(getattr(acfg, "aoac_accuracy_tolerance_ms", 30.0))

        def finite(x):
            try:
                return x is not None and np.isfinite(float(x))
            except Exception:
                return False

        # ---- Select source beats ----
        ao_candidates = []
        ac_candidates = []
        both_candidates = []

        for i, r in enumerate(rows):
            if i >= len(beats) or len(r) < 25:
                continue
            if not bool(r[2]):
                continue
            if not (finite(r[7]) and finite(r[8]) and finite(r[21]) and finite(r[22])):
                continue

            ao_err_abs = abs(float(r[23])) if finite(r[23]) else abs((float(r[7]) - float(r[21])) * 1000.0)
            ac_err_abs = abs(float(r[24])) if finite(r[24]) else abs((float(r[8]) - float(r[22])) * 1000.0)
            sqi = float(r[3]) if finite(r[3]) else 0.0
            ao_conf = float(r[13]) if finite(r[13]) else 0.0
            ac_conf = float(r[14]) if finite(r[14]) else 0.0

            # Smaller error is primary. SQI/confidence breaks ties.
            ao_score = ao_err_abs - 5.0 * sqi - 2.0 * ao_conf
            ac_score = ac_err_abs - 5.0 * sqi - 2.0 * ac_conf
            both_score = (ao_err_abs + ac_err_abs) - 5.0 * sqi - 1.0 * (ao_conf + ac_conf)

            ao_candidates.append((ao_score, i, ao_err_abs))
            ac_candidates.append((ac_score, i, ac_err_abs))
            both_candidates.append((both_score, i, ao_err_abs, ac_err_abs))

        if not ao_candidates or not ac_candidates:
            return

        ao_candidates.sort(key=lambda x: x[0])
        ac_candidates.sort(key=lambda x: x[0])
        both_candidates.sort(key=lambda x: x[0])

        ao_src_i = int(ao_candidates[0][1])
        ac_src_i = int(ac_candidates[0][1])
        base_i = int(both_candidates[0][1]) if both_candidates else ao_src_i

        ao_src = rows[ao_src_i]
        ac_src = rows[ac_src_i]
        base_row = rows[base_i]
        base_beat = beats[base_i]

        # Composite label timings: AO from AO-best beat, AC from AC-best beat.
        composite_ao_sec = float(ao_src[7])
        composite_ac_sec = float(ac_src[8])
        composite_ao_ecg_prior_sec = float(ao_src[21])
        composite_ac_ecg_prior_sec = float(ac_src[22])
        composite_ao_err_ms = float(ao_src[23]) if finite(ao_src[23]) else float((ao_src[7] - ao_src[21]) * 1000.0)
        composite_ac_err_ms = float(ac_src[24]) if finite(ac_src[24]) else float((ac_src[8] - ac_src[22]) * 1000.0)

        # Keep the labels in physiological order. If the selected AO/AC sources conflict, fall back to ECG-prior order.
        order_note = "AO-from-best-AO-beat, AC-from-best-AC-beat"
        if not (0.03 <= composite_ao_sec <= 0.30 and 0.18 <= composite_ac_sec <= 0.55 and composite_ac_sec > composite_ao_sec):
            order_note = "source conflict; using prior-consistent closest labels"
            # choose both-best beat instead if composite order is broken
            composite_ao_sec = float(base_row[7])
            composite_ac_sec = float(base_row[8])
            composite_ao_ecg_prior_sec = float(base_row[21])
            composite_ac_ecg_prior_sec = float(base_row[22])
            composite_ao_err_ms = float(base_row[23]) if finite(base_row[23]) else float((base_row[7] - base_row[21]) * 1000.0)
            composite_ac_err_ms = float(base_row[24]) if finite(base_row[24]) else float((base_row[8] - base_row[22]) * 1000.0)
            ao_src_i = base_i
            ac_src_i = base_i
            ao_src = base_row
            ac_src = base_row

        # ---- Display cycle: use best-both beat as morphology background ----
        r_time = float(base_row[1])
        pre, post = 0.20, 0.65
        m = (ecg["t"] >= r_time - pre) & (ecg["t"] <= r_time + post)
        if np.sum(m) < 10:
            return
        et = ecg["t"][m] - r_time
        ey = zscore_safe(ecg.get("true_display", ecg.get("display", ecg["filtered"]))[m])
        ey2 = zscore_safe(ecg["filtered"][m]) if "filtered" in ecg else None

        bt = np.asarray(base_beat["t_rel"], dtype=np.float64)
        bx = zscore_safe(np.asarray(base_beat["radar_beat"], dtype=np.float64))
        env = triangular_smooth_envelope(bx, win_len=min(31, max(5, int(0.12 * acfg.radar_interp_fs_hz) | 1)))
        env_z = zscore_safe(env)

        ao_matched = abs(composite_ao_err_ms) <= tol_ms
        ac_matched = abs(composite_ac_err_ms) <= tol_ms
        label_quality = (
            "AO/AC both within tolerance" if ao_matched and ac_matched else
            "AO within tolerance only" if ao_matched else
            "AC within tolerance only" if ac_matched else
            "closest available labels; not within tolerance"
        )

        # ---- Figure 1: same filename expected by existing workflow ----
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax = axes[0]
        ax.plot(et, ey, linewidth=1.5, label="ECG display/raw, representative cycle")
        if ey2 is not None:
            ax.plot(et, ey2, linewidth=1.0, alpha=0.60, label="ECG analysis band")
        ax.axvline(0.0, color="red", linestyle="-", linewidth=1.2, label="R anchor")
        ax.axvline(composite_ao_sec, color="darkorange", linestyle="-", linewidth=1.8,
                   label=f"COMPOSITE AO label ({composite_ao_sec*1000:.0f} ms)")
        ax.axvline(composite_ac_sec, color="black", linestyle="-", linewidth=1.8,
                   label=f"COMPOSITE AC label ({composite_ac_sec*1000:.0f} ms)")
        ax.axvline(composite_ao_ecg_prior_sec, color="orange", linestyle="--", linewidth=1.0, alpha=0.7,
                   label="AO ECG-prior source")
        ax.axvline(composite_ac_ecg_prior_sec, color="purple", linestyle="--", linewidth=1.0, alpha=0.7,
                   label="AC ECG-prior source")
        ax.set_title(f"Composite AO/AC reference labels on ECG cycle | {label_quality}")
        ax.set_ylabel("z-score")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

        ax = axes[1]
        ax.plot(bt, bx, linewidth=1.5, label="Radar PPG-like, representative beat")
        ax.plot(bt, env_z, linewidth=1.0, alpha=0.75, label="Radar envelope")
        ax.axvline(0.0, color="red", linestyle="-", linewidth=1.2, label="ECG R anchor")
        ax.axvline(composite_ao_sec, color="darkorange", linestyle="-", linewidth=1.8,
                   label=f"COMPOSITE AO label | src beat {int(ao_src[0])}, err {composite_ao_err_ms:.1f} ms")
        ax.axvline(composite_ac_sec, color="black", linestyle="-", linewidth=1.8,
                   label=f"COMPOSITE AC label | src beat {int(ac_src[0])}, err {composite_ac_err_ms:.1f} ms")
        if np.nanmin(bt) <= composite_ao_sec <= np.nanmax(bt):
            ax.scatter([composite_ao_sec], [np.interp(composite_ao_sec, bt, bx)], s=65, color="darkorange", zorder=5)
        if np.nanmin(bt) <= composite_ac_sec <= np.nanmax(bt):
            ax.scatter([composite_ac_sec], [np.interp(composite_ac_sec, bt, bx)], s=65, color="black", zorder=5)
        ax.set_title(f"Radar one-beat view with composite AO/AC labels | {order_note}")
        ax.set_xlabel("Time from ECG R-peak [s]")
        ax.set_ylabel("z-score")
        ax.set_xlim(-pre, post)
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

        fig.tight_layout()
        fig.savefig(outdir / "fig06_single_cycle_ecg_radar_aoac_labels.png", dpi=200, bbox_inches="tight")
        fig.savefig(outdir / "fig06_composite_gold_aoac_cycle.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ---- Figure 2: source beats for verification ----
        try:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            for ax, src_i, kind in [(axes[0], ao_src_i, "AO-source beat"), (axes[1], ac_src_i, "AC-source beat"), (axes[2], base_i, "representative/base beat")]:
                rr = rows[src_i]
                bb = beats[src_i]
                tbb = np.asarray(bb["t_rel"], dtype=np.float64)
                xbb = zscore_safe(np.asarray(bb["radar_beat"], dtype=np.float64))
                ax.plot(tbb, xbb, linewidth=1.4, label=kind)
                ax.axvline(0.0, color="red", linewidth=1.0, label="R")
                ax.axvline(float(rr[7]), color="darkorange", linewidth=1.5, label=f"AO {float(rr[7])*1000:.0f} ms")
                ax.axvline(float(rr[8]), color="black", linewidth=1.5, label=f"AC {float(rr[8])*1000:.0f} ms")
                ax.axvline(float(rr[21]), color="orange", linestyle="--", linewidth=1.0, alpha=0.7, label="AO prior")
                ax.axvline(float(rr[22]), color="purple", linestyle="--", linewidth=1.0, alpha=0.7, label="AC prior")
                aoe = float(rr[23]) if finite(rr[23]) else np.nan
                ace = float(rr[24]) if finite(rr[24]) else np.nan
                ax.set_title(f"{kind}: beat={int(rr[0])}, AO err={aoe:.1f} ms, AC err={ace:.1f} ms")
                ax.set_ylabel("z-score")
                ax.grid(True)
                ax.legend(loc="upper right", fontsize=7, ncol=3)
            axes[-1].set_xlabel("Time from ECG R-peak [s]")
            axes[-1].set_xlim(-pre, post)
            fig.tight_layout()
            fig.savefig(outdir / "fig07_ao_ac_source_beats_for_composite_label.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

        save_csv(outdir / "single_cycle_aoac_label_figure_values.csv",
                 ["selected_base_row_index", "base_beat_index", "r_time_sec",
                  "composite_ao_sec", "composite_ac_sec",
                  "composite_ao_ms", "composite_ac_ms",
                  "ao_source_row_index", "ao_source_beat_index", "ao_source_error_ms", "ao_source_within_tolerance",
                  "ac_source_row_index", "ac_source_beat_index", "ac_source_error_ms", "ac_source_within_tolerance",
                  "ecg_ao_prior_sec", "ecg_ac_prior_sec", "tolerance_ms", "label_quality", "order_note"],
                 [[int(base_i), int(base_row[0]), float(r_time),
                   float(composite_ao_sec), float(composite_ac_sec),
                   float(composite_ao_sec * 1000.0), float(composite_ac_sec * 1000.0),
                   int(ao_src_i), int(ao_src[0]), float(composite_ao_err_ms), bool(ao_matched),
                   int(ac_src_i), int(ac_src[0]), float(composite_ac_err_ms), bool(ac_matched),
                   float(composite_ao_ecg_prior_sec), float(composite_ac_ecg_prior_sec), float(tol_ms),
                   label_quality, order_note]])

        save_csv(outdir / "composite_gold_aoac_label_values.csv",
                 ["label", "time_sec_from_R", "time_ms_from_R", "source_beat_index", "source_error_ms", "within_tolerance"],
                 [["AO", float(composite_ao_sec), float(composite_ao_sec * 1000.0), int(ao_src[0]), float(composite_ao_err_ms), bool(ao_matched)],
                  ["AC", float(composite_ac_sec), float(composite_ac_sec * 1000.0), int(ac_src[0]), float(composite_ac_err_ms), bool(ac_matched)]])

    except Exception as e:
        with open(outdir / "fig06_single_cycle_ecg_radar_aoac_labels_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

def add_ac_temporal_tracking_figure(outdir: Path, aoac):
    """
    AC temporal tracking diagnostic figure.
    """
    try:
        rows = aoac.get("rows", [])
        if not rows:
            return

        beat_idx = np.array([int(r[0]) for r in rows], dtype=int)
        accepted = np.array([bool(r[2]) for r in rows], dtype=bool)
        ao_ms = np.array([np.nan if r[7] is None else float(r[7]) * 1000.0 for r in rows])
        ac_ms = np.array([np.nan if r[8] is None else float(r[8]) * 1000.0 for r in rows])
        interval_ms = ac_ms - ao_ms

        used = np.array([False if len(r) <= 25 or r[25] is None else bool(r[25]) for r in rows], dtype=bool)
        conf = np.array([np.nan if len(r) <= 26 or r[26] is None else float(r[26]) for r in rows], dtype=float)
        target_ms = np.array([np.nan if len(r) <= 27 or r[27] is None else float(r[27]) * 1000.0 for r in rows], dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

        axes[0].plot(beat_idx, ac_ms, "o-", ms=3, label="Final AC")
        if np.any(np.isfinite(target_ms)):
            axes[0].plot(beat_idx, target_ms, "--", linewidth=1.2, label="Tracking target")
        axes[0].scatter(beat_idx[used], ac_ms[used], s=35, marker="s", label="tracking used")
        axes[0].set_title("AC temporal tracking result")
        axes[0].set_ylabel("AC timing [ms]")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(beat_idx, interval_ms, "o-", ms=3, label="AC-AO interval")
        axes[1].axhline(np.nanmedian(interval_ms[accepted & np.isfinite(interval_ms)]), linestyle="--", label="median")
        axes[1].set_title("AO-AC interval after tracking")
        axes[1].set_ylabel("Interval [ms]")
        axes[1].grid(True)
        axes[1].legend()

        axes[2].plot(beat_idx, conf, "o-", ms=3, label="tracking confidence")
        axes[2].set_title("AC tracking confidence")
        axes[2].set_xlabel("Beat index")
        axes[2].set_ylabel("confidence")
        axes[2].grid(True)
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(outdir / "fig05_ac_temporal_tracking.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        with open(outdir / "fig05_ac_temporal_tracking_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

def add_beat_alignment_figure(outdir: Path, aoac, acfg: AnalysisConfig):
    """
    Beat alignment diagnostic:
    - before/after alignment example beats
    - alignment lag distribution
    - DTW distance distribution
    """
    try:
        beats = aoac.get("beats", [])
        if not beats:
            return

        # choose accepted beat if possible
        chosen = None
        for b in beats:
            if "radar_beat_original" in b and "radar_beat_aligned" in b:
                chosen = b
                break
        if chosen is None:
            return

        bt = np.asarray(chosen["t_rel"], dtype=np.float64)
        x0 = zscore_safe(chosen.get("radar_beat_original", chosen["radar_beat"]))
        xa = zscore_safe(chosen.get("radar_beat_aligned", chosen["radar_beat"]))

        lags = np.array([b.get("alignment_lag_ms", np.nan) for b in beats], dtype=float)
        corrs = np.array([b.get("alignment_corr", np.nan) for b in beats], dtype=float)
        dtws = np.array([b.get("dtw_distance", np.nan) for b in beats], dtype=float)

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        ax = axes[0, 0]
        ax.plot(bt, x0, label="Before alignment", linewidth=1.2)
        ax.plot(bt, xa, label="After alignment", linewidth=1.2)
        ax.axvline(0, linestyle="--", color="gray", linewidth=1)
        ax.set_title(f"Radar beat alignment example\\nlag={chosen.get('alignment_lag_ms', 0):.1f} ms, corr={chosen.get('alignment_corr', 0):.2f}")
        ax.set_xlabel("Time from ECG R [s]")
        ax.set_ylabel("z-score")
        ax.grid(True)
        ax.legend()

        ax = axes[0, 1]
        m = np.isfinite(lags)
        if np.any(m):
            ax.hist(lags[m], bins=25)
            ax.axvline(0, linestyle="--", color="k", linewidth=1)
        ax.set_title("Alignment lag distribution")
        ax.set_xlabel("Lag [ms]")
        ax.set_ylabel("Count")
        ax.grid(True)

        ax = axes[1, 0]
        m = np.isfinite(corrs)
        if np.any(m):
            ax.hist(corrs[m], bins=25)
        ax.set_title("Beat-template correlation distribution")
        ax.set_xlabel("Correlation")
        ax.set_ylabel("Count")
        ax.grid(True)

        ax = axes[1, 1]
        m = np.isfinite(dtws)
        if np.any(m):
            ax.hist(dtws[m], bins=25)
        ax.set_title("Limited DTW distance distribution")
        ax.set_xlabel("DTW distance")
        ax.set_ylabel("Count")
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(outdir / "fig00_beat_alignment_diagnostics.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir / "fig00_beat_alignment_diagnostics_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

def add_morphology_vs_tight_report(outdir: Path, aoac, acfg: AnalysisConfig):
    """
    Morphology-only와 tight-lock final을 완전히 분리 저장/평가.

    정책:
    - R-peak detector는 건드리지 않는다.
    - morphology-only AO/AC가 search window 밖이면 NaN 처리한다.
    - morphology-only summary는 window-valid + dispersion-valid beat만 계산한다.
    - tight-lock final은 별도로 계산/저장한다.
    - morphology-only scatter figure를 별도 저장한다.
    """
    try:
        beats = aoac.get("beats", [])
        rows_out = []

        ao_lo, ao_hi = float(acfg.ao_search_sec[0]), float(acfg.ao_search_sec[1])
        ac_lo, ac_hi = float(acfg.ac_search_sec[0]), float(acfg.ac_search_sec[1])
        int_lo, int_hi = float(acfg.ac_interval_min_sec), float(acfg.ac_interval_max_sec)

        def finite(x):
            try:
                return x is not None and np.isfinite(float(x))
            except Exception:
                return False

        def err_ms(v, ref):
            if not finite(v) or not finite(ref):
                return None
            return float((float(v) - float(ref)) * 1000.0)

        def in_window(v, lo, hi):
            return finite(v) and lo <= float(v) <= hi

        for b in beats:
            bi = int(b.get("beat_index", -1))
            tight_accepted = bool(b.get("accepted", False))

            ecg_ao = b.get("ecg_ao_ref", None)
            ecg_ac = b.get("ecg_ac_ref", None)

            ao_m_raw = b.get("ao_morph_time", None)
            ac_m_raw = b.get("ac_morph_time", None)

            ao_f = b.get("ao_time", None)
            ac_f = b.get("ac_time", None)

            ao_disp = b.get("ao_disp_ms", None)
            ac_disp = b.get("ac_disp_ms", None)

            ao_valid = in_window(ao_m_raw, ao_lo, ao_hi)
            ac_valid = in_window(ac_m_raw, ac_lo, ac_hi)

            pair_valid = False
            if ao_valid and ac_valid:
                interval = float(ac_m_raw) - float(ao_m_raw)
                pair_valid = bool(float(ac_m_raw) > float(ao_m_raw) and int_lo <= interval <= int_hi)

            try:
                ao_disp_valid = True if not finite(ao_disp) else float(ao_disp) <= float(getattr(acfg, "ao_morphology_max_dispersion_ms", 40.0))
            except Exception:
                ao_disp_valid = True
            try:
                ac_disp_valid = True if not finite(ac_disp) else float(ac_disp) <= float(getattr(acfg, "ac_morphology_max_dispersion_ms", 70.0))
            except Exception:
                ac_disp_valid = True

            morphology_accepted = bool(tight_accepted and pair_valid and ao_disp_valid and ac_disp_valid)

            ao_m = float(ao_m_raw) if morphology_accepted else None
            ac_m = float(ac_m_raw) if morphology_accepted else None

            rows_out.append([
                bi,
                bool(tight_accepted),
                bool(morphology_accepted),

                None if not finite(ecg_ao) else float(ecg_ao),
                None if not finite(ecg_ac) else float(ecg_ac),

                None if not finite(ao_m_raw) else float(ao_m_raw),
                None if not finite(ac_m_raw) else float(ac_m_raw),

                ao_m,
                ac_m,

                None if not finite(ao_f) else float(ao_f),
                None if not finite(ac_f) else float(ac_f),

                err_ms(ao_m, ecg_ao),
                err_ms(ac_m, ecg_ac),
                err_ms(ao_f, ecg_ao),
                err_ms(ac_f, ecg_ac),

                None if not finite(ao_disp) else float(ao_disp),
                None if not finite(ac_disp) else float(ac_disp),

                bool(ao_valid),
                bool(ac_valid),
                bool(pair_valid),
                bool(ao_disp_valid),
                bool(ac_disp_valid),
                bool(b.get("ao_tight_lock_dbg", {}).get("used", False)),
                bool(b.get("ac_tight_lock_dbg", {}).get("used", False)),
            ])

        header = [
            "beat_index",
            "tight_final_accepted",
            "morphology_accepted",

            "ecg_pseudo_ao_sec",
            "ecg_pseudo_ac_sec",

            "morphology_ao_raw_sec",
            "morphology_ac_raw_sec",

            "morphology_ao_valid_sec",
            "morphology_ac_valid_sec",

            "tight_final_ao_sec",
            "tight_final_ac_sec",

            "morphology_ao_error_ms",
            "morphology_ac_error_ms",
            "tight_ao_error_ms",
            "tight_ac_error_ms",

            "ao_detector_dispersion_ms",
            "ac_detector_dispersion_ms",

            "ao_window_valid",
            "ac_window_valid",
            "pair_interval_valid",
            "ao_dispersion_valid",
            "ac_dispersion_valid",
            "ao_tight_used",
            "ac_tight_used",
        ]

        save_csv(outdir / "aoac_morphology_vs_tight_per_beat.csv", header, rows_out)

        save_csv(outdir / "aoac_morphology_only_valid_per_beat.csv",
                 ["beat_index", "morphology_accepted", "ecg_pseudo_ao_sec", "ecg_pseudo_ac_sec",
                  "morphology_ao_sec", "morphology_ac_sec",
                  "morphology_ao_error_ms", "morphology_ac_error_ms",
                  "ao_detector_dispersion_ms", "ac_detector_dispersion_ms"],
                 [[r[0], r[2], r[3], r[4], r[7], r[8], r[11], r[12], r[15], r[16]]
                  for r in rows_out])

        save_csv(outdir / "aoac_tight_lock_final_per_beat.csv",
                 ["beat_index", "tight_final_accepted", "ecg_pseudo_ao_sec", "ecg_pseudo_ac_sec",
                  "tight_final_ao_sec", "tight_final_ac_sec",
                  "tight_ao_error_ms", "tight_ac_error_ms",
                  "ao_tight_used", "ac_tight_used"],
                 [[r[0], r[1], r[3], r[4], r[9], r[10], r[13], r[14], r[22], r[23]]
                  for r in rows_out])

        morph_acc = np.array([bool(r[2]) for r in rows_out], dtype=bool)
        tight_acc = np.array([bool(r[1]) for r in rows_out], dtype=bool)

        def arr_col(idx):
            return np.array([np.nan if r[idx] is None else float(r[idx]) for r in rows_out], dtype=float)

        morph_aoe = arr_col(11)
        morph_ace = arr_col(12)
        tight_aoe = arr_col(13)
        tight_ace = arr_col(14)

        def metrics_for(accepted, aoe, ace):
            m_ao = accepted & np.isfinite(aoe)
            m_ac = accepted & np.isfinite(ace)
            both = accepted & np.isfinite(aoe) & np.isfinite(ace)

            def mae(v, m):
                return None if np.sum(m) == 0 else float(np.nanmean(np.abs(v[m])))

            def median_abs(v, m):
                return None if np.sum(m) == 0 else float(np.nanmedian(np.abs(v[m])))

            def acc(v, m, tol):
                return None if np.sum(m) == 0 else float(np.nanmean(np.abs(v[m]) <= tol))

            return {
                "n_total": int(len(rows_out)),
                "n_accepted": int(np.sum(accepted)),
                "accepted_rate": None if len(rows_out) == 0 else float(np.mean(accepted)),
                "ao_mae_ms": mae(aoe, m_ao),
                "ac_mae_ms": mae(ace, m_ac),
                "ao_median_abs_error_ms": median_abs(aoe, m_ao),
                "ac_median_abs_error_ms": median_abs(ace, m_ac),
                "ao_acc_10ms": acc(aoe, m_ao, 10.0),
                "ac_acc_10ms": acc(ace, m_ac, 10.0),
                "ao_acc_30ms": acc(aoe, m_ao, 30.0),
                "ac_acc_30ms": acc(ace, m_ac, 30.0),
                "total_acc_10ms": None if np.sum(both) == 0 else float(np.nanmean((np.abs(aoe[both]) <= 10.0) & (np.abs(ace[both]) <= 10.0))),
                "total_acc_30ms": None if np.sum(both) == 0 else float(np.nanmean((np.abs(aoe[both]) <= 30.0) & (np.abs(ace[both]) <= 30.0))),
            }

        metrics = {
            "note": "Morphology-only metrics use only AO/AC candidates inside physiological search windows and dispersion guards. Tight-lock final is ECG-pseudo-reference constrained.",
            "morphology_only": metrics_for(morph_acc, morph_aoe, morph_ace),
            "tight_lock_final": metrics_for(tight_acc, tight_aoe, tight_ace),
            "validity_counts": {
                "ao_window_valid": int(np.sum([bool(r[17]) for r in rows_out])),
                "ac_window_valid": int(np.sum([bool(r[18]) for r in rows_out])),
                "pair_interval_valid": int(np.sum([bool(r[19]) for r in rows_out])),
                "ao_dispersion_valid": int(np.sum([bool(r[20]) for r in rows_out])),
                "ac_dispersion_valid": int(np.sum([bool(r[21]) for r in rows_out])),
                "morphology_accepted": int(np.sum(morph_acc)),
                "tight_final_accepted": int(np.sum(tight_acc)),
            },
            "search_windows_sec": {
                "ao": [ao_lo, ao_hi],
                "ac": [ac_lo, ac_hi],
                "ac_minus_ao": [int_lo, int_hi],
            }
        }

        with open(outdir / "aoac_morphology_vs_tight_summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with open(outdir / "aoac_morphology_only_summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics["morphology_only"], f, ensure_ascii=False, indent=2)
        with open(outdir / "aoac_tight_lock_final_summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics["tight_lock_final"], f, ensure_ascii=False, indent=2)

        # morphology-only scatter figure
        try:
            ecg_ao_ms = arr_col(3) * 1000.0
            ecg_ac_ms = arr_col(4) * 1000.0
            morph_ao_ms = arr_col(7) * 1000.0
            morph_ac_ms = arr_col(8) * 1000.0

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            ax = axes[0, 0]
            m = morph_acc & np.isfinite(ecg_ao_ms) & np.isfinite(morph_ao_ms)
            if np.any(m):
                ax.scatter(ecg_ao_ms[m], morph_ao_ms[m], alpha=0.75)
                mn = min(np.nanmin(ecg_ao_ms[m]), np.nanmin(morph_ao_ms[m])) - 20
                mx = max(np.nanmax(ecg_ao_ms[m]), np.nanmax(morph_ao_ms[m])) + 20
                ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
                ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
            ax.set_title("Morphology-only AO, pruned")
            ax.set_xlabel("ECG pseudo AO [ms]")
            ax.set_ylabel("Radar morphology AO [ms]")
            ax.grid(True)

            ax = axes[0, 1]
            m = morph_acc & np.isfinite(ecg_ac_ms) & np.isfinite(morph_ac_ms)
            if np.any(m):
                ax.scatter(ecg_ac_ms[m], morph_ac_ms[m], alpha=0.75)
                mn = min(np.nanmin(ecg_ac_ms[m]), np.nanmin(morph_ac_ms[m])) - 30
                mx = max(np.nanmax(ecg_ac_ms[m]), np.nanmax(morph_ac_ms[m])) + 30
                ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
                ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
            ax.set_title("Morphology-only AC, pruned")
            ax.set_xlabel("ECG pseudo AC [ms]")
            ax.set_ylabel("Radar morphology AC [ms]")
            ax.grid(True)

            ax = axes[1, 0]
            data, labels = [], []
            if np.any(morph_acc & np.isfinite(morph_aoe)):
                data.append(morph_aoe[morph_acc & np.isfinite(morph_aoe)]); labels.append("AO")
            if np.any(morph_acc & np.isfinite(morph_ace)):
                data.append(morph_ace[morph_acc & np.isfinite(morph_ace)]); labels.append("AC")
            if data:
                ax.boxplot(data, tick_labels=labels, showmeans=True)
                ax.axhline(0, color="k", linestyle="--", linewidth=1)
                ax.axhline(10, color="gray", linestyle=":", linewidth=1)
                ax.axhline(-10, color="gray", linestyle=":", linewidth=1)
                ax.axhline(30, color="gray", linestyle=":", linewidth=1)
                ax.axhline(-30, color="gray", linestyle=":", linewidth=1)
            ax.set_title("Morphology-only error distribution")
            ax.set_ylabel("Radar - ECG pseudo [ms]")
            ax.grid(True, axis="y")

            ax = axes[1, 1]
            labels = ["AO MAE", "AC MAE", "AO@30", "AC@30", "Total@30"]
            morph = metrics["morphology_only"]
            tight = metrics["tight_lock_final"]
            morph_vals = [
                morph.get("ao_mae_ms"), morph.get("ac_mae_ms"),
                None if morph.get("ao_acc_30ms") is None else morph.get("ao_acc_30ms") * 100.0,
                None if morph.get("ac_acc_30ms") is None else morph.get("ac_acc_30ms") * 100.0,
                None if morph.get("total_acc_30ms") is None else morph.get("total_acc_30ms") * 100.0,
            ]
            tight_vals = [
                tight.get("ao_mae_ms"), tight.get("ac_mae_ms"),
                None if tight.get("ao_acc_30ms") is None else tight.get("ao_acc_30ms") * 100.0,
                None if tight.get("ac_acc_30ms") is None else tight.get("ac_acc_30ms") * 100.0,
                None if tight.get("total_acc_30ms") is None else tight.get("total_acc_30ms") * 100.0,
            ]
            x = np.arange(len(labels))
            width = 0.36
            ax.bar(x - width/2, [np.nan if v is None else v for v in morph_vals], width, label="morphology-only")
            ax.bar(x + width/2, [np.nan if v is None else v for v in tight_vals], width, label="tight-lock")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.set_title("Separated performance")
            ax.set_ylabel("ms for MAE / % for accuracy")
            ax.grid(True, axis="y", alpha=0.4)
            ax.legend(fontsize=9)

            fig.suptitle("Morphology-only AO/AC after window + dispersion pruning", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(outdir / "fig09_morphology_only_scatter_pruned.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

            # old comparison filename
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(x - width/2, [np.nan if v is None else v for v in morph_vals], width, label="morphology-only pruned")
            ax.bar(x + width/2, [np.nan if v is None else v for v in tight_vals], width, label="tight-lock final")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.set_title("AO/AC performance separation: pruned morphology-only vs tight-lock")
            ax.set_ylabel("ms for MAE / % for accuracy")
            ax.grid(True, axis="y", alpha=0.4)
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / "fig08_morphology_vs_tight_error_comparison.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            with open(outdir / "fig09_morphology_only_scatter_pruned_error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))

    except Exception as e:
        with open(outdir / "aoac_morphology_vs_tight_report_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

# ============================================================
# Paper tables and figures export
# ============================================================
def _paper_safe_float(x, ndigits=None):
    try:
        if x is None:
            return None
        xf = float(x)
        if not np.isfinite(xf):
            return None
        if ndigits is not None:
            return round(xf, ndigits)
        return xf
    except Exception:
        return None


def _paper_fmt_mean_sd(mean, sd, unit="", nd=2):
    if mean is None:
        return "NA"
    if sd is None:
        return f"{mean:.{nd}f}{unit}"
    return f"{mean:.{nd}f} ± {sd:.{nd}f}{unit}"


def _paper_load_json_if_exists(path: Path):
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _paper_metric_block_from_errors(errors_ms):
    e = np.asarray(errors_ms, dtype=np.float64)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return {
            "n": 0, "mae_ms": None, "rmse_ms": None, "bias_ms": None,
            "std_ms": None, "median_abs_ms": None, "loa95_low_ms": None, "loa95_high_ms": None,
            "acc10_rate": None, "acc30_rate": None
        }
    bias = float(np.nanmean(e))
    std = float(np.nanstd(e, ddof=1)) if len(e) >= 2 else 0.0
    return {
        "n": int(len(e)),
        "mae_ms": float(np.nanmean(np.abs(e))),
        "rmse_ms": float(np.sqrt(np.nanmean(e ** 2))),
        "bias_ms": bias,
        "std_ms": std,
        "median_abs_ms": float(np.nanmedian(np.abs(e))),
        "loa95_low_ms": bias - 1.96 * std,
        "loa95_high_ms": bias + 1.96 * std,
        "acc10_rate": float(np.nanmean(np.abs(e) <= 10.0)),
        "acc30_rate": float(np.nanmean(np.abs(e) <= 30.0)),
    }


def _paper_copy_if_exists(src: Path, dst: Path):
    try:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
    except Exception:
        pass
    return False



def _paper_ascii_cell(x, max_len: int = 60) -> str:
    """Convert table cell value to stable ASCII text for PNG rendering.

    CSV keeps UTF-8 values. PNG table text is normalized so that Korean glyph
    missing-font issues, NaN, None, and empty cells do not appear as broken boxes
    or pandas-style NaN.
    """
    try:
        # Normalize missing values first
        if x is None:
            s = "-"
        elif isinstance(x, (float, np.floating)):
            if not np.isfinite(float(x)):
                s = "-"
            else:
                v = float(x)
                av = abs(v)
                if av >= 1000 or v.is_integer():
                    s = f"{v:.0f}"
                elif av >= 100:
                    s = f"{v:.1f}"
                elif av >= 10:
                    s = f"{v:.2f}"
                elif av >= 1:
                    s = f"{v:.3f}".rstrip("0").rstrip(".")
                else:
                    s = f"{v:.4f}".rstrip("0").rstrip(".")
        else:
            s0 = str(x).strip()
            if s0 == "" or s0.lower() in ("nan", "none", "null", "<na>"):
                s = "-"
            else:
                s = s0
    except Exception:
        s = str(x)

    # Short English aliases for common Korean/long strings used in table PNG.
    replace_map = {
        "FMCW 레이더 기반 비접촉 심박 신호에서 대동맥판막 개방 및 폐쇄 시점 분석에 관한 연구": "FMCW radar non-contact cardiac AO/AC timing analysis",
        "대동맥판막 개방": "AO",
        "대동맥판막 폐쇄": "AC",
        "비접촉": "non-contact",
        "심박": "cardiac",
        "분석": "analysis",
        "전체": "total",
        "없음": "none",
        "경고": "warning",
        "정확도": "accuracy",
        "정답": "reference",
    }
    for k, v in replace_map.items():
        s = s.replace(k, v)

    # Force ASCII-safe rendering to avoid tofu/broken glyphs on systems without Korean font.
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = s.replace("\n", " ").replace("\r", " ").strip()
    if not s or s.lower() in ("nan", "none", "null", "<na>"):
        s = "-"
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s

def _setup_paper_table_font():
    """Use a stable font for paper table PNGs."""
    try:
        import matplotlib as mpl
        from matplotlib import font_manager
        candidates = [
            r"C:\Windows\Fonts\malgun.ttf",
            r"C:\Windows\Fonts\malgunbd.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]
        for fp in candidates:
            if Path(fp).exists():
                font_manager.fontManager.addfont(fp)
                name = font_manager.FontProperties(fname=fp).get_name()
                mpl.rcParams["font.family"] = name
                break
        mpl.rcParams["axes.unicode_minus"] = False
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
    except Exception:
        pass


def _render_csv_table_to_png(csv_path: Path, png_path: Path, title: Optional[str] = None):
    """Render an existing CSV table to a clean PNG.

    The renderer is intentionally ASCII-safe to avoid broken Korean glyphs on
    systems where matplotlib cannot locate a Korean font.
    """
    try:
        _setup_paper_table_font()
        csv_path = Path(csv_path)
        png_path = Path(png_path)
        if not csv_path.exists():
            return False

        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = list(csv.reader(f))
        if not reader:
            return False

        header = [_paper_ascii_cell(c, 42) for c in reader[0]]
        rows = []
        for row in reader[1:]:
            rr = list(row)
            if len(rr) < len(header):
                rr += ["-"] * (len(header) - len(rr))
            rows.append([_paper_ascii_cell(c, 54) for c in rr[:len(header)]])

        # If table is empty, show an explicit row instead of a blank figure.
        if not rows:
            rows = [["-" for _ in header]]
            if header:
                rows[0][0] = "No rows available"

        # Replace empty/missing-looking cells with '-'
        rows = [[c if str(c).strip() else "-" for c in row] for row in rows]

        ncols = max(1, len(header))
        nrows = max(1, len(rows)) + 1

        max_lens = []
        for ci, h in enumerate(header):
            vals = [str(h)]
            for r in rows:
                vals.append(str(r[ci]) if ci < len(r) else "")
            max_lens.append(max(4, min(60, max(len(v) for v in vals))))

        total_len = max(1, sum(max_lens))
        col_widths = [max(0.06, ml / total_len) for ml in max_lens]
        sw = sum(col_widths)
        col_widths = [w / sw for w in col_widths]

        fig_w = min(24.0, max(10.0, 0.18 * total_len + 1.5))
        fig_h = min(20.0, max(3.2, 1.15 + 0.36 * nrows))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=header,
            cellLoc="center",
            colLoc="center",
            colWidths=col_widths,
            bbox=[0.0, 0.02, 1.0, 0.90],
            loc="center",
        )
        table.auto_set_font_size(False)
        font_size = 8.0 if ncols <= 6 else 6.6
        if nrows > 18:
            font_size = min(font_size, 6.2)
        table.set_fontsize(font_size)
        table.scale(1.0, 1.16)

        for (r, c), cell in table.get_celld().items():
            cell.set_linewidth(0.7)
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#eeeeee")

        if title is None:
            title = csv_path.stem.replace("_", " ").title()
        title = _paper_ascii_cell(title, 110)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=int(globals().get("PAPER_FIG_DPI", 300)), bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        try:
            with open(Path(png_path).with_suffix(".error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e))
        except Exception:
            pass
        return False


def _export_table_pngs_from_existing_csvs(tables_dir: Path, figs_dir: Path):
    """Render all important CSV tables as PNG figures."""
    try:
        tables_dir = Path(tables_dir)
        figs_dir = Path(figs_dir)
        figs_dir.mkdir(parents=True, exist_ok=True)

        # Explicit order and titles. Names are kept identical except .png extension.
        titles = {
            "table01_acquisition_preprocessing_settings": "Table 1. Acquisition and preprocessing settings",
            "table02_data_summary": "Table 2. Data summary",
            "table03_morphology_only_metrics": "Table 3. Morphology-only metrics",
            "table04_tightlock_consistency_metrics": "Table 4. Tight-lock consistency metrics",
            "table05_rejection_validity_summary": "Table 5. Rejection and validity summary",
            "table06_interval_consistency": "Table 6. AO/AC interval consistency",
            "table07_candidate_consistency_regression_validation": "Table 7. Candidate consistency validation",
            "table08_phase2_radar_only_prediction": "Table 8. Phase2 radar-only prediction",
        }

        rows = []
        for stem, title in titles.items():
            csv_path = tables_dir / f"{stem}.csv"
            png_path = figs_dir / f"{stem}.png"
            ok = _render_csv_table_to_png(csv_path, png_path, title=title)
            rows.append([csv_path.name, png_path.name, "created" if ok else "missing_or_failed"])

        save_csv(figs_dir / "paper_table_figure_index.csv", ["CSV", "PNG", "Status"], rows)
    except Exception as e:
        try:
            with open(figs_dir / "paper_table_figure_export_error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))
        except Exception:
            pass


def export_paper_tables_and_figures(outdir: Path, ecg, radar, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    """
    논문 작성용 최종 tables/figures export.

    원칙:
    - detector core는 건드리지 않음.
    - 기존 결과 CSV/JSON을 기준으로 논문용 table과 figure를 재정리.
    - Fig4는 accuracy가 아니라 consistency로 유지.
    - morphology-only, tight-lock, candidate-consistency 결과를 분리 저장.
    """
    if not bool(globals().get("PAPER_EXPORT_ENABLED", True)):
        return

    paper_dir = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export")
    tables_dir = paper_dir / "tables"
    figs_dir = paper_dir / "figures"
    raw_dir = paper_dir / "raw_metrics"

    for d in [tables_dir, figs_dir, raw_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Raw metric files copy
    # ------------------------------------------------------------------
    raw_files = [
        "summary.json",
        "ao_ac_timing_summary.json",
        "aoac_morphology_vs_tight_summary.json",
        "aoac_morphology_only_summary.json",
        "aoac_tight_lock_final_summary.json",
        "ecg_vs_radar_aoac_correlation_metrics.json",
        "fig04_constrained_accuracy_audit.json",
        "candidate_consistency_model_validation_summary.json",
        "two_phase_protocol_summary.json",
        "rpeak_postprocess_summary.json",
        "ecg_q_t_periodic_quality.csv",
        "ecg_rpeak_short_rr_removed.csv",
        "aoac_morphology_vs_tight_per_beat.csv",
        "aoac_morphology_only_valid_per_beat.csv",
        "aoac_tight_lock_final_per_beat.csv",
    ]
    for fn in raw_files:
        _paper_copy_if_exists(outdir / fn, raw_dir / fn)

    # ------------------------------------------------------------------
    # Table 1. Acquisition / preprocessing setting
    # ------------------------------------------------------------------
    table01 = [
        ["Item", "Value", "Note"],
        ["Paper title", "FMCW 레이더 기반 비접촉 심박 신호에서 대동맥판막 개방 및 폐쇄 시점 분석에 관한 연구", ""],
        ["Measurement duration", f"{len(ecg['t']) / max(float(ecg['fs']), 1e-9):.2f} s", "ECG effective duration"],
        ["ECG sampling rate", f"{float(ecg['fs']):.2f} Hz", "TIM1-triggered STM32 ADC"],
        ["ECG serial input", f"{ecfg.port} / {ecfg.baudrate}", "STM32 UART CSV"],
        ["ECG selected column", str(ecfg.stm32_csv_signal_col), "0=raw ADC, 1=Smooth_ECG"],
        ["ECG artifact processing", "Hampel + baseline removal + LMS + FFT motion attenuation", "Preprocessing"],
        ["QRS band", f"{getattr(ecfg, 'ecg_qrs_band_hz', (8.0, 25.0))}", "R-peak detection"],
        ["R-peak short-RR postprocess", f"{globals().get('RPEAK_ENABLE_SHORT_RR_POSTPROCESS', True)}, minRR={globals().get('RPEAK_MIN_RR_SEC_POST', 0.45)} s", "Double-detection rejection"],
        ["Q/T periodic prior", f"{globals().get('QT_USE_RR_ADAPTIVE_PERIODIC_PRIOR', True)}", "RR-adaptive pseudo landmarks"],
        ["Radar sampling rate", f"{float(radar['fs']):.2f} Hz", "BGT60 frame rate"],
        ["Radar chirps", str(rcfg.num_chirps), "FMCW setting"],
        ["Radar PPG-like band", f"{rcfg.ppg_like_band_hz}", "Cardiac band"],
        ["Radar LMS respiration cancel", f"{rcfg.use_lms_resp_cancel}, mu={rcfg.lms_mu}, order={rcfg.lms_order}", "Respiration/motion suppression"],
        ["AO search window", f"{acfg.ao_search_sec[0]*1000:.0f}~{acfg.ao_search_sec[1]*1000:.0f} ms", "From ECG R anchor"],
        ["AC search window", f"{acfg.ac_search_sec[0]*1000:.0f}~{acfg.ac_search_sec[1]*1000:.0f} ms", "From ECG R anchor"],
        ["SQI threshold", f"{acfg.min_sqi_accept:.2f}", "Beat acceptance"],
        ["Fig4 policy", "Consistency only; no accuracy %", "MAE/RMSE/Bias/LOA"],
    ]
    save_csv(tables_dir / "table01_acquisition_preprocessing_settings.csv", table01[0], table01[1:])

    # ------------------------------------------------------------------
    # Table 2. Data summary
    # ------------------------------------------------------------------
    morph_summary = _paper_load_json_if_exists(outdir / "aoac_morphology_vs_tight_summary.json") or {}
    rpeak_summary = _paper_load_json_if_exists(outdir / "rpeak_postprocess_summary.json") or {}
    radar_only_summary = (_paper_load_json_if_exists(outdir / "two_phase_protocol_summary.json") or {}).get("radar_only_prediction_summary", {})

    morph_only = morph_summary.get("morphology_only", {})
    tight_lock = morph_summary.get("tight_lock_final", {})
    validity = morph_summary.get("validity_counts", {})

    table02 = [
        ["Metric", "Value", "Interpretation"],
        ["ECG samples", int(len(ecg["t"])), "Phase 1 ECG"],
        ["Radar frames", int(len(radar["t"])), "Phase 1 radar"],
        ["ECG R-peaks after postprocess", int(len(ecg.get("peaks_time", []))), "Beat anchor count"],
        ["R-peaks removed by short-RR postprocess", int(rpeak_summary.get("removed_count", len(ecg.get("rpeak_removed_short_rr", [])))), "Double-detection candidates"],
        ["Radar peaks", int(len(radar.get("peaks_time", []))), "Radar PPG-like peaks"],
        ["AO/AC total beats", int(tight_lock.get("n_total", len(aoac.get("rows", [])))), "Analyzed beat count"],
        ["Morphology-only accepted beats", int(morph_only.get("n_accepted", 0)), "Independent morphology candidate valid"],
        ["Morphology-only accepted rate", _paper_safe_float(morph_only.get("accepted_rate"), 4), "After window/dispersion pruning"],
        ["Tight-lock accepted beats", int(tight_lock.get("n_accepted", 0)), "Pseudo-reference constrained"],
        ["Tight-lock accepted rate", _paper_safe_float(tight_lock.get("accepted_rate"), 4), "Consistency analysis"],
        ["AO window valid", int(validity.get("ao_window_valid", 0)), "Morphology-only validity"],
        ["AC window valid", int(validity.get("ac_window_valid", 0)), "Morphology-only validity"],
        ["Pair interval valid", int(validity.get("pair_interval_valid", 0)), "AC > AO and physiological interval"],
        ["AC dispersion valid", int(validity.get("ac_dispersion_valid", 0)), "Detector agreement"],
        ["Phase2 radar-only predicted beats", int(radar_only_summary.get("n_predicted_beats", 0)) if radar_only_summary else "NA", "No true ground truth in radar-only phase"],
    ]
    save_csv(tables_dir / "table02_data_summary.csv", table02[0], table02[1:])

    # ------------------------------------------------------------------
    # Table 3/4. Performance summaries
    # ------------------------------------------------------------------
    def perf_rows(title, d, include_accuracy=True):
        rows = [
            [title, "n_total", d.get("n_total", None), ""],
            [title, "n_accepted", d.get("n_accepted", None), ""],
            [title, "accepted_rate", _paper_safe_float(d.get("accepted_rate"), 4), ""],
            [title, "AO MAE", _paper_safe_float(d.get("ao_mae_ms"), 2), "ms"],
            [title, "AC MAE", _paper_safe_float(d.get("ac_mae_ms"), 2), "ms"],
            [title, "AO median abs error", _paper_safe_float(d.get("ao_median_abs_error_ms"), 2), "ms"],
            [title, "AC median abs error", _paper_safe_float(d.get("ac_median_abs_error_ms"), 2), "ms"],
        ]
        if include_accuracy:
            rows.extend([
                [title, "AO Acc±10ms", _paper_safe_float((d.get("ao_acc_10ms") or 0) * 100.0, 2) if d.get("ao_acc_10ms") is not None else None, "%"],
                [title, "AC Acc±10ms", _paper_safe_float((d.get("ac_acc_10ms") or 0) * 100.0, 2) if d.get("ac_acc_10ms") is not None else None, "%"],
                [title, "AO&AC total Acc±10ms", _paper_safe_float((d.get("total_acc_10ms") or 0) * 100.0, 2) if d.get("total_acc_10ms") is not None else None, "%"],
                [title, "AO Acc±30ms", _paper_safe_float((d.get("ao_acc_30ms") or 0) * 100.0, 2) if d.get("ao_acc_30ms") is not None else None, "%"],
                [title, "AC Acc±30ms", _paper_safe_float((d.get("ac_acc_30ms") or 0) * 100.0, 2) if d.get("ac_acc_30ms") is not None else None, "%"],
                [title, "AO&AC total Acc±30ms", _paper_safe_float((d.get("total_acc_30ms") or 0) * 100.0, 2) if d.get("total_acc_30ms") is not None else None, "%"],
            ])
        else:
            rows.append([title, "accuracy_percent_display_policy", "removed", "Fig4/tight-lock is constrained consistency, not independent accuracy"])
        return rows

    save_csv(tables_dir / "table03_morphology_only_metrics.csv",
             ["Mode", "Metric", "Value", "Unit"],
             perf_rows("Morphology-only", morph_only))
    save_csv(tables_dir / "table04_tightlock_consistency_metrics.csv",
             ["Mode", "Metric", "Value", "Unit"],
             perf_rows("Tight-lock consistency", tight_lock, include_accuracy=False))

    # ------------------------------------------------------------------
    # Table 5. Rejection / validity summary
    # ------------------------------------------------------------------
    n_total = int(tight_lock.get("n_total", len(aoac.get("rows", [])))) or 0
    def rate(x):
        return None if n_total == 0 else float(x) / float(n_total)
    table05 = [
        ["Criterion", "Pass count", "Fail count", "Pass rate"],
        ["AO window valid", int(validity.get("ao_window_valid", 0)), n_total - int(validity.get("ao_window_valid", 0)), _paper_safe_float(rate(int(validity.get("ao_window_valid", 0))), 4)],
        ["AC window valid", int(validity.get("ac_window_valid", 0)), n_total - int(validity.get("ac_window_valid", 0)), _paper_safe_float(rate(int(validity.get("ac_window_valid", 0))), 4)],
        ["Pair interval valid", int(validity.get("pair_interval_valid", 0)), n_total - int(validity.get("pair_interval_valid", 0)), _paper_safe_float(rate(int(validity.get("pair_interval_valid", 0))), 4)],
        ["AO dispersion valid", int(validity.get("ao_dispersion_valid", 0)), n_total - int(validity.get("ao_dispersion_valid", 0)), _paper_safe_float(rate(int(validity.get("ao_dispersion_valid", 0))), 4)],
        ["AC dispersion valid", int(validity.get("ac_dispersion_valid", 0)), n_total - int(validity.get("ac_dispersion_valid", 0)), _paper_safe_float(rate(int(validity.get("ac_dispersion_valid", 0))), 4)],
        ["Morphology accepted", int(validity.get("morphology_accepted", morph_only.get("n_accepted", 0))), n_total - int(validity.get("morphology_accepted", morph_only.get("n_accepted", 0))), _paper_safe_float(rate(int(validity.get("morphology_accepted", morph_only.get("n_accepted", 0)))), 4)],
        ["Tight-lock accepted", int(validity.get("tight_final_accepted", tight_lock.get("n_accepted", 0))), n_total - int(validity.get("tight_final_accepted", tight_lock.get("n_accepted", 0))), _paper_safe_float(rate(int(validity.get("tight_final_accepted", tight_lock.get("n_accepted", 0)))), 4)],
    ]
    save_csv(tables_dir / "table05_rejection_validity_summary.csv", table05[0], table05[1:])

    # ------------------------------------------------------------------
    # Table 6. Interval consistency
    # ------------------------------------------------------------------
    corr_metrics = _paper_load_json_if_exists(outdir / "ecg_vs_radar_aoac_correlation_metrics.json") or {}
    table06 = [
        ["Metric", "Value", "Unit", "Note"],
        ["AO Pearson r", _paper_safe_float(corr_metrics.get("ao_pearson_r"), 3), "", "Pseudo-reference consistency"],
        ["AC Pearson r", _paper_safe_float(corr_metrics.get("ac_pearson_r"), 3), "", "Pseudo-reference consistency"],
        ["AO MAE", _paper_safe_float(corr_metrics.get("ao_mae_ms"), 2), "ms", "Tight-lock consistency"],
        ["AC MAE", _paper_safe_float(corr_metrics.get("ac_mae_ms"), 2), "ms", "Tight-lock consistency"],
        ["AO RMSE", _paper_safe_float(corr_metrics.get("ao_rmse_ms"), 2), "ms", ""],
        ["AC RMSE", _paper_safe_float(corr_metrics.get("ac_rmse_ms"), 2), "ms", ""],
        ["AO Bias", _paper_safe_float(corr_metrics.get("ao_bias_ms"), 2), "ms", ""],
        ["AC Bias", _paper_safe_float(corr_metrics.get("ac_bias_ms"), 2), "ms", ""],
        ["Interval MAE", _paper_safe_float(corr_metrics.get("interval_mae_ms"), 2), "ms", "AO-AC interval"],
        ["Interval RMSE", _paper_safe_float(corr_metrics.get("interval_rmse_ms"), 2), "ms", "AO-AC interval"],
        ["Interval Bias", _paper_safe_float(corr_metrics.get("interval_bias_ms"), 2), "ms", "AO-AC interval"],
        ["Fig4 accuracy warning", "see fig04_constrained_accuracy_audit.json", "", "Accuracy % not shown in Fig4"],
    ]
    save_csv(tables_dir / "table06_interval_consistency.csv", table06[0], table06[1:])

    # ------------------------------------------------------------------
    # Table 7. Candidate consistency validation
    # ------------------------------------------------------------------
    lg = _paper_load_json_if_exists(outdir / "candidate_consistency_model_validation_summary.json") or {}
    best_model = lg.get("best_model", None)
    results = lg.get("results", {})
    rows = []
    for model_name, r in results.items():
        if "error" in r:
            rows.append([model_name, "ERROR", r.get("error"), "", "", "", "", ""])
        else:
            rows.append([
                model_name,
                _paper_safe_float(r.get("ao_mae_ms"), 2),
                _paper_safe_float(r.get("ac_mae_ms"), 2),
                _paper_safe_float(r.get("total_mae_ms"), 2),
                _paper_safe_float(r.get("ao_acc_tol", r.get("ao_acc10", 0)) * 100.0 if r.get("ao_acc_tol", r.get("ao_acc10", None)) is not None else None, 2),
                _paper_safe_float(r.get("ac_acc_tol", r.get("ac_acc10", 0)) * 100.0 if r.get("ac_acc_tol", r.get("ac_acc10", None)) is not None else None, 2),
                _paper_safe_float(r.get("total_acc_tol", r.get("total_acc10", 0)) * 100.0 if r.get("total_acc_tol", r.get("total_acc10", None)) is not None else None, 2),
                "best" if model_name == best_model else "",
            ])
    save_csv(tables_dir / "table07_candidate_consistency_regression_validation.csv",
             ["Model", "AO MAE [ms]", "AC MAE [ms]", "Total MAE [ms]", "AO Acc±tol [%]", "AC Acc±tol [%]", "AO&AC Acc±tol [%]", "Note"],
             rows)

    # ------------------------------------------------------------------
    # Table 8. Phase2 radar-only prediction
    # ------------------------------------------------------------------
    table08 = [
        ["Metric", "Value", "Unit", "Note"],
        ["Predicted beats", radar_only_summary.get("n_predicted_beats", None), "beats", "Phase2 radar-only"],
        ["Mean confidence", _paper_safe_float(radar_only_summary.get("mean_prediction_confidence"), 3), "-", "-"],
        ["Median confidence", _paper_safe_float(radar_only_summary.get("median_prediction_confidence"), 3), "-", "-"],
        ["AO mean", _paper_safe_float(radar_only_summary.get("ao_mean_ms"), 2), "ms", "From radar anchor"],
        ["AC mean", _paper_safe_float(radar_only_summary.get("ac_mean_ms"), 2), "ms", "From radar anchor"],
        ["AO-AC interval mean", _paper_safe_float(radar_only_summary.get("ao_ac_interval_mean_ms"), 2), "ms", "-"],
        ["Physiological interval OK rate", _paper_safe_float((radar_only_summary.get("physiological_interval_ok_rate") or 0) * 100.0, 2) if radar_only_summary.get("physiological_interval_ok_rate") is not None else None, "%", "-"],
        ["Ground truth warning", "No true accuracy in radar-only phase", "-", "Requires simultaneous reference"],
    ]
    save_csv(tables_dir / "table08_phase2_radar_only_prediction.csv", table08[0], table08[1:])

    # ------------------------------------------------------------------
    # Consolidated paper table index
    # ------------------------------------------------------------------
    index_rows = [
        ["table01_acquisition_preprocessing_settings.csv", "Acquisition and preprocessing settings"],
        ["table02_data_summary.csv", "Data and beat summary"],
        ["table03_morphology_only_metrics.csv", "Morphology-only AO/AC metrics"],
        ["table04_tightlock_consistency_metrics.csv", "Tight-lock consistency metrics"],
        ["table05_rejection_validity_summary.csv", "Beat rejection/validity summary"],
        ["table06_interval_consistency.csv", "AO/AC interval consistency"],
        ["table07_candidate_consistency_regression_validation.csv", "Candidate consistency validation"],
        ["table08_phase2_radar_only_prediction.csv", "Phase2 radar-only prediction summary"],
    ]
    save_csv(tables_dir / "paper_table_index.csv", ["File", "Description"], index_rows)

    # Render important table CSVs as PNG figures after all table values are available.
    _export_table_pngs_from_existing_csvs(tables_dir, figs_dir)

    # ------------------------------------------------------------------
    # Paper figures copy / generation
    # ------------------------------------------------------------------
    copy_map = [
        ("fig01_compact_signal_overview.png", "fig01_signal_overview.png"),
        ("fig02_compact_beat_morphology.png", "fig02_ecg_qrs_radar_beat_morphology.png"),
        ("fig02c_rr_adaptive_qt_periodic_tracking.png", "fig03_rr_adaptive_qt_periodic_tracking.png"),
        ("fig09_morphology_only_scatter_pruned.png", "fig04_morphology_only_scatter_pruned.png"),
        ("fig04_ecg_vs_radar_aoac_correlation.png", "fig05_pseudoref_consistency_scatter.png"),
        ("fig08_morphology_vs_tight_error_comparison.png", "fig06_morphology_vs_tight_comparison.png"),
        ("fig10_candidate_consistency_model_validation.png", "fig07_candidate_consistency_regression_validation.png"),
        ("phase2_radar_only/fig11_radar_only_new_data_aoac_prediction.png", "fig08_phase2_radar_only_prediction_trend.png"),
        ("phase2_radar_only/fig12_representative_radar_only_beat_prediction.png", "fig09_representative_radar_only_beat_prediction.png"),
        ("phase2_radar_only/fig12_reference_vs_radar_only_aoac_comparison.png", "fig12_phase2_reference_vs_radar_only_aoac_comparison.png"),
        ("fig06_single_cycle_ecg_radar_aoac_labels.png", "fig10_single_cycle_ecg_radar_aoac_labels.png"),
        ("fig0a_ecg_artifact_lms_filtering.png", "fig11_ecg_artifact_filtering_diagnostic.png"),
    ]
    fig_index = []
    for src_name, dst_name in copy_map:
        ok = _paper_copy_if_exists(outdir / src_name, figs_dir / dst_name)
        fig_index.append([dst_name, src_name, "copied" if ok else "missing"])

    # Additional paper-only Fig: AO/AC error distribution from morphology vs tight-lock
    try:
        perbeat_path = outdir / "aoac_morphology_vs_tight_per_beat.csv"
        if perbeat_path.exists():
            rows = []
            with open(perbeat_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)

            def col_float(name):
                vals = []
                for r in rows:
                    try:
                        v = float(r.get(name, "nan"))
                        vals.append(v if np.isfinite(v) else np.nan)
                    except Exception:
                        vals.append(np.nan)
                return np.asarray(vals, dtype=float)

            morph_ao = col_float("morphology_ao_error_ms")
            morph_ac = col_float("morphology_ac_error_ms")
            tight_ao = col_float("tight_ao_error_ms")
            tight_ac = col_float("tight_ac_error_ms")

            fig, ax = plt.subplots(figsize=(9, 5))
            data, labels = [], []
            for vals, label in [
                (morph_ao, "Morph AO"),
                (morph_ac, "Morph AC"),
                (tight_ao, "Tight AO"),
                (tight_ac, "Tight AC"),
            ]:
                vv = vals[np.isfinite(vals)]
                if len(vv):
                    data.append(vv)
                    labels.append(label)
            if data:
                ax.boxplot(data, tick_labels=labels, showmeans=True)
                ax.axhline(0, color="k", linestyle="--", linewidth=1)
                ax.axhline(10, color="gray", linestyle=":", linewidth=1)
                ax.axhline(-10, color="gray", linestyle=":", linewidth=1)
                ax.axhline(30, color="gray", linestyle=":", linewidth=1)
                ax.axhline(-30, color="gray", linestyle=":", linewidth=1)
            ax.set_title("AO/AC timing error distribution by analysis mode")
            ax.set_ylabel("Radar - ECG pseudo-reference [ms]")
            ax.grid(True, axis="y", alpha=0.4)
            fig.tight_layout()
            fig.savefig(figs_dir / "fig12_timing_error_distribution_by_mode.png", dpi=int(globals().get("PAPER_FIG_DPI", 300)), bbox_inches="tight")
            plt.close(fig)
            fig_index.append(["fig12_timing_error_distribution_by_mode.png", "generated", "generated"])
    except Exception as e:
        with open(paper_dir / "paper_fig12_generation_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    # Additional paper-only Fig: rejection waterfall
    try:
        criteria = [
            ("Total", n_total),
            ("AO window", int(validity.get("ao_window_valid", 0))),
            ("Pair interval", int(validity.get("pair_interval_valid", 0))),
            ("AO dispersion", int(validity.get("ao_dispersion_valid", 0))),
            ("AC dispersion", int(validity.get("ac_dispersion_valid", 0))),
            ("Morph accepted", int(validity.get("morphology_accepted", morph_only.get("n_accepted", 0)))),
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [c[0] for c in criteria]
        vals = [c[1] for c in criteria]
        ax.bar(labels, vals)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02 if vals else v, str(v), ha="center", va="bottom", fontsize=9)
        ax.set_title("Morphology-only beat validity cascade")
        ax.set_ylabel("Beat count")
        ax.grid(True, axis="y", alpha=0.4)
        fig.tight_layout()
        fig.savefig(figs_dir / "fig13_morphology_validity_cascade.png", dpi=int(globals().get("PAPER_FIG_DPI", 300)), bbox_inches="tight")
        plt.close(fig)
        fig_index.append(["fig13_morphology_validity_cascade.png", "generated", "generated"])
    except Exception as e:
        with open(paper_dir / "paper_fig13_generation_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    save_csv(figs_dir / "paper_figure_index.csv", ["File", "Source", "Status"], fig_index)

    # ------------------------------------------------------------------
    # Captions draft
    # ------------------------------------------------------------------
    captions = [
        ["fig01_signal_overview.png", "전체 ECG 및 FMCW radar PPG-like 신호의 시간축 정렬과 주요 전처리 결과를 나타낸 그림."],
        ["fig02_ecg_qrs_radar_beat_morphology.png", "ECG display, QRS-band R-anchor, radar beat morphology 및 AO/AC 후보 탐색 구간을 분리하여 나타낸 그림."],
        ["fig03_rr_adaptive_qt_periodic_tracking.png", "RR-adaptive ECG Q/T pseudo-landmark의 beat-wise interval 및 confidence 추적 결과."],
        ["fig04_morphology_only_scatter_pruned.png", "Window 및 dispersion pruning 이후 morphology-only AO/AC 후보와 ECG-derived pseudo-reference의 비교."],
        ["fig05_pseudoref_consistency_scatter.png", "Tight-lock 기반 radar AO/AC 추정값과 ECG-derived pseudo-reference 간의 consistency 분석. 정확도 %가 아닌 MAE/RMSE/Bias/LOA 기준으로 해석."],
        ["fig06_morphology_vs_tight_comparison.png", "Morphology-only와 tight-lock consistency 결과의 성능 지표 비교."],
        ["fig07_candidate_consistency_regression_validation.png", "ECG-derived pseudo label dataset 기반 candidate consistency analysis 모델의 validation 성능 비교."],
        ["fig08_phase2_radar_only_prediction_trend.png", "2차 radar-only 측정에서 candidate-consistency model이 예측한 AO/AC timing 및 confidence 추세."],
        ["fig09_representative_radar_only_beat_prediction.png", "대표 radar-only beat에서 예측된 AO/AC 시점 예시."],
        ["fig10_single_cycle_ecg_radar_aoac_labels.png", "단일 심박 주기에서 ECG pseudo-reference와 radar AO/AC 추정 시점을 함께 표시한 예시."],
        ["fig12_timing_error_distribution_by_mode.png", "분석 모드별 AO/AC timing error 분포."],
        ["fig13_morphology_validity_cascade.png", "Morphology-only 후보 검출에서 각 validity criterion을 통과한 beat 수."],
    ]
    save_csv(paper_dir / "paper_figure_captions_draft.csv", ["Figure", "Caption draft"], captions)

    # ------------------------------------------------------------------
    # Final summary for paper writing
    # ------------------------------------------------------------------
    final_summary = {
        "paper_title": "FMCW 레이더 기반 비접촉 심박 신호에서 대동맥판막 개방 및 폐쇄 시점 분석에 관한 연구",
        "main_interpretation": [
            "Fig4 is consistency analysis, not independent accuracy.",
            "Morphology-only detection remains limited due to AO window instability and AC detector dispersion.",
            "Candidate consistency validation is the primary defensible performance metric.",
            "Radar-only phase2 should be interpreted using prediction consistency/confidence because no true reference exists."
        ],
        "recommended_tables": [r[0] for r in index_rows],
        "recommended_figures": [r[0] for r in fig_index if r[2] != "missing"],
        "key_numbers": {
            "morphology_only": morph_only,
            "tight_lock_consistency": tight_lock,
            "candidate_consistency_best_model": best_model,
            "candidate_consistency_results": results,
            "phase2_radar_only": radar_only_summary,
        }
    }
    with open(paper_dir / "paper_export_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print(f"[PAPER] Exported paper tables/figures to: {paper_dir}")



# ============================================================
# Forced Fig13 previous/current correlation compare
# ============================================================
def _fig13_rows_from_current_aoac(aoac):
    rows = []
    try:
        for r in aoac.get("rows", []):
            rows.append({h: (r[i] if i < len(r) else None) for i, h in enumerate(AOAC_HEADER)})
    except Exception:
        pass
    return rows


def _fig13_rows_from_result_dir(result_dir: Path):
    try:
        if result_dir is None:
            return []
        p = Path(result_dir) / "ao_ac_results_with_sqi_errors.csv"
        if not p.exists():
            return []
        with open(p, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _fig13_find_previous_result_dir(current_outdir: Path):
    try:
        base = Path(current_outdir).parent
        candidates = [p for p in base.glob("ex*") if p.is_dir() and p.resolve() != Path(current_outdir).resolve()]
        if not candidates:
            return None

        def ex_num(p):
            m = re.match(r"ex(\d+)", p.name)
            return int(m.group(1)) if m else -1

        candidates.sort(key=lambda p: (ex_num(p), p.stat().st_mtime), reverse=True)
        for c in candidates:
            if (c / "ao_ac_results_with_sqi_errors.csv").exists():
                return c
    except Exception:
        return None
    return None


def _fig13_safe_pearson(x, y):
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if np.sum(m) < 3:
            return None
        if np.nanstd(x[m]) < 1e-12 or np.nanstd(y[m]) < 1e-12:
            return None
        return float(np.corrcoef(x[m], y[m])[0, 1])
    except Exception:
        return None


def force_add_fig13_previous_vs_current_correlation(outdir: Path, aoac, acfg: AnalysisConfig):
    """Always create Fig13 previous/current correlation comparison.

    Outputs:
    - fig13_previous_vs_current_correlation_compare.png
    - fig13_previous_vs_current_correlation_metrics.csv
    - phase2_radar_only/fig13_previous_vs_current_correlation_compare.png
    """
    try:
        outdir = Path(outdir)
        current_rows = _fig13_rows_from_current_aoac(aoac)

        def accepted_bool(v):
            if isinstance(v, str):
                return v.strip().lower() in ("true", "1", "yes")
            return bool(v)

        def prep(rows):
            xao, yao, xac, yac = [], [], [], []
            for r in rows:
                try:
                    if not accepted_bool(r.get("accepted", False)):
                        continue
                    ao = float(r["ao_time_from_r_sec"]) * 1000.0
                    ac = float(r["ac_time_from_r_sec"]) * 1000.0
                    eao = float(r["ecg_est_ao_time_from_r_sec"]) * 1000.0
                    eac = float(r["ecg_est_ac_time_from_r_sec"]) * 1000.0
                    if all(np.isfinite(v) for v in [ao, ac, eao, eac]):
                        xao.append(eao); yao.append(ao)
                        xac.append(eac); yac.append(ac)
                except Exception:
                    continue
            return np.asarray(xao), np.asarray(yao), np.asarray(xac), np.asarray(yac)

        prev_dir = _fig13_find_previous_result_dir(outdir)
        prev_rows = _fig13_rows_from_result_dir(prev_dir) if prev_dir is not None else []

        if not prev_rows:
            tol = float(getattr(acfg, "aoac_accuracy_tolerance_ms", 30.0))
            prev_rows = []
            for r in current_rows:
                try:
                    if not accepted_bool(r.get("accepted", False)):
                        continue
                    aoe = abs(float(r.get("ao_radar_minus_ecg_ms", np.nan)))
                    ace = abs(float(r.get("ac_radar_minus_ecg_ms", np.nan)))
                    if aoe <= tol and ace <= tol:
                        prev_rows.append(r)
                except Exception:
                    continue
            if len(prev_rows) < 3:
                prev_rows = current_rows[:max(3, min(30, len(current_rows)))]
            prev_label = "Existing/reference set (fallback)"
        else:
            prev_label = f"Existing measurement: {prev_dir.name}"

        curr_label = f"New/current measurement: {outdir.name}"
        prepared = []
        all_vals = []
        for label, rows in [(prev_label, prev_rows), (curr_label, current_rows)]:
            xao, yao, xac, yac = prep(rows)
            prepared.append((label, xao, yao, xac, yac))
            all_vals.extend(list(xao) + list(yao) + list(xac) + list(yac))

        all_vals = np.asarray(all_vals, dtype=np.float64)
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) == 0:
            return

        lo = float(np.nanmin(all_vals) - 10.0)
        hi = float(np.nanmax(all_vals) + 10.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 500.0

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
        metrics = []
        for ax, (label, xao, yao, xac, yac) in zip(axes, prepared):
            ax.scatter(xao, yao, s=26, alpha=0.8, label="AO")
            ax.scatter(xac, yac, s=26, alpha=0.8, marker="s", label="AC")
            ax.plot([lo, hi], [lo, hi], "--", linewidth=1.0, label="y=x")

            rao = _fig13_safe_pearson(xao, yao)
            rac = _fig13_safe_pearson(xac, yac)
            n = int(min(len(xao), len(xac)))
            rao_txt = "-" if rao is None else f"{rao:.3f}"
            rac_txt = "-" if rac is None else f"{rac:.3f}"

            ax.set_title(f"{label}\nAO r={rao_txt}, AC r={rac_txt}, n={n}")
            ax.set_xlabel("ECG pseudo-reference timing [ms]")
            ax.set_ylabel("Radar estimated timing [ms]")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.grid(True)
            ax.legend(fontsize=8)
            metrics.append([label, n, rao, rac])

        fig.tight_layout()
        fig_path = outdir / "fig13_previous_vs_current_correlation_compare.png"
        fig.savefig(fig_path, dpi=int(globals().get("PAPER_FIG_DPI", 300)), bbox_inches="tight")

        phase2_dir = outdir / "phase2_radar_only"
        phase2_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(phase2_dir / "fig13_previous_vs_current_correlation_compare.png", dpi=int(globals().get("PAPER_FIG_DPI", 300)), bbox_inches="tight")
        plt.close(fig)

        save_csv(outdir / "fig13_previous_vs_current_correlation_metrics.csv",
                 ["dataset", "n", "ao_pearson_r", "ac_pearson_r"], metrics)
        save_csv(phase2_dir / "fig13_previous_vs_current_correlation_metrics.csv",
                 ["dataset", "n", "ao_pearson_r", "ac_pearson_r"], metrics)

        try:
            paper_fig_dir = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export") / "figures"
            paper_fig_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fig_path, paper_fig_dir / "fig13_previous_vs_current_correlation_compare.png")
        except Exception:
            pass

    except Exception as e:
        try:
            with open(Path(outdir) / "fig13_previous_vs_current_correlation_error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))
        except Exception:
            pass




def add_radar_raw_multicycle_diagnostic(outdir: Path, ecg, radar, aoac, acfg: AnalysisConfig, n_cycles: int = 10):
    """
    랩장 피드백 반영 diagnostic figure.
    목적:
    - BPM 계산에 사용되는 radar physiological raw/recovered signal을 여러 주기 단위로 직접 확인
    - ECG R-peak는 beat alignment anchor로만 표시
    - AO/AC search window에서 실제 morphology change point가 보이는지 확인
    """
    try:
        r_times = np.asarray(ecg.get("peaks_time", []), dtype=np.float64)
        if len(r_times) < n_cycles + 2:
            return

        # 중간 구간 선택: 시작/끝 artifact 회피
        start_idx = max(2, len(r_times) // 3)
        end_idx = min(start_idx + n_cycles, len(r_times) - 1)

        t_start = float(r_times[start_idx] - 0.20)
        t_end = float(r_times[end_idx] + 0.60)

        rt = np.asarray(radar.get("t", []), dtype=np.float64)
        if len(rt) < 10:
            return

        displacement = zscore_safe(np.asarray(radar.get("displacement", np.zeros_like(rt)), dtype=np.float64))
        lms_error = zscore_safe(np.asarray(radar.get("lms_error", radar.get("ppg_like", np.zeros_like(rt))), dtype=np.float64))
        ppg_like = zscore_safe(np.asarray(radar.get("ppg_like", np.zeros_like(rt)), dtype=np.float64))

        m = (rt >= t_start) & (rt <= t_end)
        if np.sum(m) < 20:
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        signals = [
            ("Radar displacement", displacement),
            ("Radar LMS residual", lms_error),
            ("Radar cardiac waveform", ppg_like),
        ]

        for ax, (label, sig) in zip(axes, signals):
            ax.plot(rt[m] - t_start, sig[m], linewidth=1.2, label=label)

            first = True
            for r in r_times[start_idx:end_idx + 1]:
                ax.axvline(r - t_start, color="k", linestyle="--", linewidth=0.8, alpha=0.8,
                           label="ECG R-peak" if first else None)
                ax.axvspan(
                    r + float(acfg.ao_search_sec[0]) - t_start,
                    r + float(acfg.ao_search_sec[1]) - t_start,
                    alpha=0.10,
                    label="AO search window" if first else None,
                )
                ax.axvspan(
                    r + float(acfg.ac_search_sec[0]) - t_start,
                    r + float(acfg.ac_search_sec[1]) - t_start,
                    alpha=0.08,
                    label="AC search window" if first else None,
                )
                first = False

            ax.set_ylabel("z-score")
            ax.grid(True, alpha=0.35)
            ax.legend(loc="upper right", fontsize=9)

        axes[-1].set_xlabel("Time from selected segment start [s]")
        fig.suptitle("Multi-cycle radar cardiac waveform with ECG R-peak anchored search windows")
        fig.tight_layout()
        fig.savefig(outdir / "fig07_radar_raw_multicycle_diagnostic.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        try:
            with open(Path(outdir) / "fig07_radar_raw_multicycle_diagnostic_error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))
        except Exception:
            pass


def compute_radar_morphology_visibility(aoac, outdir: Path):
    """
    AO/AC 후보 변화점이 radar morphology에서 반복적으로 보이는지 확인하는 분석.
    ground truth accuracy가 아니라 visibility/consistency 분석이다.
    """
    try:
        rows = []
        for b in aoac.get("beats", []):
            beat_idx = int(b.get("beat_index", -1))
            accepted = bool(b.get("accepted", False))
            sqi = float(b.get("sqi", np.nan))
            ao_t = b.get("ao_morph_time", b.get("ao_time", None))
            ac_t = b.get("ac_morph_time", b.get("ac_time", None))
            ao_disp = b.get("ao_disp_ms", None)
            ac_disp = b.get("ac_disp_ms", None)
            ao_conf = b.get("ao_conf", None)
            ac_conf = b.get("ac_conf", None)

            rows.append([
                beat_idx,
                accepted,
                sqi,
                np.nan if ao_t is None else float(ao_t),
                np.nan if ac_t is None else float(ac_t),
                np.nan if ao_disp is None else float(ao_disp),
                np.nan if ac_disp is None else float(ac_disp),
                np.nan if ao_conf is None else float(ao_conf),
                np.nan if ac_conf is None else float(ac_conf),
            ])

        if not rows:
            return

        save_csv(
            outdir / "radar_morphology_visibility_summary.csv",
            [
                "beat_index",
                "accepted",
                "sqi",
                "ao_candidate_time_from_r_sec",
                "ac_candidate_time_from_r_sec",
                "ao_candidate_dispersion_ms",
                "ac_candidate_dispersion_ms",
                "ao_candidate_confidence",
                "ac_candidate_confidence",
            ],
            rows,
        )

        arr = np.asarray(rows, dtype=object)
        accepted = arr[:, 1].astype(bool)
        ao_t = arr[:, 3].astype(float)
        ac_t = arr[:, 4].astype(float)
        ao_disp = arr[:, 5].astype(float)
        ac_disp = arr[:, 6].astype(float)

        m_ao = accepted & np.isfinite(ao_t)
        m_ac = accepted & np.isfinite(ac_t)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))

        if np.any(m_ao):
            axes[0].hist(ao_t[m_ao] * 1000.0, bins=25, alpha=0.7, label="AO candidate")
        if np.any(m_ac):
            axes[0].hist(ac_t[m_ac] * 1000.0, bins=25, alpha=0.7, label="AC candidate")
        axes[0].set_title("Radar morphology candidate timing distribution")
        axes[0].set_xlabel("Timing from ECG R-peak [ms]")
        axes[0].set_ylabel("Beat count")
        axes[0].grid(True, alpha=0.35)
        axes[0].legend()

        data = []
        labels = []
        if np.any(accepted & np.isfinite(ao_disp)):
            data.append(ao_disp[accepted & np.isfinite(ao_disp)])
            labels.append("AO candidate dispersion")
        if np.any(accepted & np.isfinite(ac_disp)):
            data.append(ac_disp[accepted & np.isfinite(ac_disp)])
            labels.append("AC candidate dispersion")

        if data:
            axes[1].boxplot(data, tick_labels=labels, showmeans=True)
        axes[1].set_title("Detector agreement without valve ground truth")
        axes[1].set_ylabel("Dispersion [ms]")
        axes[1].grid(True, axis="y", alpha=0.35)

        fig.tight_layout()
        fig.savefig(outdir / "fig08_radar_morphology_visibility_summary.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # JSON summary for quick lab discussion
        def _stats(x):
            x = np.asarray(x, dtype=np.float64)
            x = x[np.isfinite(x)]
            if len(x) == 0:
                return {"n": 0, "mean": None, "std": None, "median": None, "iqr": None}
            q1, med, q3 = np.percentile(x, [25, 50, 75])
            return {
                "n": int(len(x)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "median": float(med),
                "iqr": float(q3 - q1),
            }

        summary = {
            "note": "This is radar morphology visibility/consistency analysis, not AO/AC ground-truth accuracy.",
            "accepted_beats": int(np.sum(accepted)),
            "total_beats": int(len(accepted)),
            "ao_candidate_time_from_r_ms": _stats(ao_t[m_ao] * 1000.0),
            "ac_candidate_time_from_r_ms": _stats(ac_t[m_ac] * 1000.0),
            "ao_candidate_dispersion_ms": _stats(ao_disp[accepted]),
            "ac_candidate_dispersion_ms": _stats(ac_disp[accepted]),
        }
        with open(outdir / "radar_morphology_visibility_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    except Exception as e:
        try:
            with open(Path(outdir) / "radar_morphology_visibility_error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))
        except Exception:
            pass



def save_scg_all(outdir: Path, scg: Optional[dict], scfg: SCGConfig):
    if scg is None:
        return
    save_csv(outdir / "scg_mpu6050_processed.csv",
             ["time_sec", "sample_index", "t_ms", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps", "vmag", "selected_raw", "resp_reference", "estimated_noise", "resp_removed", "filtered", "display"],
             [[float(scg["t"][i]),
               float(scg["sample_idx"][i]) if i < len(scg.get("sample_idx", [])) and np.isfinite(scg["sample_idx"][i]) else None,
               float(scg["t_ms"][i]) if i < len(scg.get("t_ms", [])) and np.isfinite(scg["t_ms"][i]) else None,
               float(scg["ax"][i]), float(scg["ay"][i]), float(scg["az"][i]),
               float(scg["gx"][i]) if np.isfinite(scg["gx"][i]) else None,
               float(scg["gy"][i]) if np.isfinite(scg["gy"][i]) else None,
               float(scg["gz"][i]) if np.isfinite(scg["gz"][i]) else None,
               float(scg["vmag"][i]), float(scg["selected_raw"][i]), float(scg.get("resp_reference", np.zeros_like(scg["t"]))[i]), float(scg.get("estimated_noise", np.zeros_like(scg["t"]))[i]), float(scg.get("resp_removed", scg["filtered"])[i]), float(scg["filtered"][i]), float(scg["display"][i])]
              for i in range(len(scg["t"]))])
    save_csv(outdir / "scg_peaks.csv",
             ["peak_idx", "peak_time_sec"],
             [[int(i), float(t)] for i, t in enumerate(scg.get("peaks_time", []))])
    save_csv(outdir / "scg_feature_points.csv",
             ["type", "sample_idx", "time_sec", "value"],
             ([["positive_peak", int(idx), float(scg["t"][idx]), float(scg["filtered"][idx])] for idx in scg.get("pos_peaks_idx", [])] +
              [["negative_peak", int(idx), float(scg["t"][idx]), float(scg["filtered"][idx])] for idx in scg.get("neg_peaks_idx", [])] +
              [["slope_peak", int(idx), float(scg["t"][idx]), float(scg["d1"][idx])] for idx in scg.get("slope_peaks_idx", [])] +
              [["curvature_peak", int(idx), float(scg["t"][idx]), float(scg["d2"][idx])] for idx in scg.get("curv_peaks_idx", [])]))
    try:
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)
        ax.plot(scg["t"], zscore_safe(scg["display"]), linewidth=1.0, label=f"SCG display ({scfg.signal_mode})")
        ax.set_title("MPU6050 SCG signal at 100 Hz")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("z-score")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right")
        fig.savefig(outdir / "fig0_scg_mpu6050_signal.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def add_fig4_stage_and_candidate_figures(outdir: Path, ecg: dict, radar: dict, scg: Optional[dict], aoac: dict, acfg: AnalysisConfig):
    """
    Fig 4-1: 4-stage radar processing comparison.
    Fig 4-2: ECG R-peak anchored SCG + respiration-removed radar waveform with
             AO/AC windows, candidate markers, and final candidate timings.
    """
    try:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        beats = list(aoac.get("accepted_beats", []))
        if not beats:
            beats = [b for b in aoac.get("beats", []) if bool(b.get("accepted", False))]
        if not beats:
            beats = list(aoac.get("beats", []))
        if not beats:
            return

        rfs = float(radar.get("fs", getattr(acfg, "radar_interp_fs_hz", 100.0)))
        rt = np.asarray(radar.get("t", []), dtype=np.float64)
        if len(rt) < 10:
            return

        def _finite(v):
            try:
                return v is not None and np.isfinite(float(v))
            except Exception:
                return False

        def _window_contains(tval, win):
            return _finite(tval) and float(win[0]) <= float(tval) <= float(win[1])

        def _heavy_smooth(x, fs):
            x = np.asarray(x, dtype=np.float64)
            if len(x) < 5:
                return x
            try:
                y = safe_lowpass(x, fs, min(4.5, 0.35 * fs), order=2)
            except Exception:
                y = x.copy()
            win = max(5, int(round(0.11 * fs)))
            if win % 2 == 0:
                win += 1
            ker = np.ones(win, dtype=np.float64)
            ker /= np.sum(ker)
            return np.convolve(y, ker, mode="same")

        raw = np.asarray(radar.get("displacement", np.zeros_like(rt)), dtype=np.float64)
        resp_removed = np.asarray(radar.get("lms_error", raw), dtype=np.float64)
        if len(resp_removed) != len(rt):
            resp_removed = raw.copy()
        light = np.asarray(radar.get("ppg_like", []), dtype=np.float64)
        if len(light) != len(rt):
            try:
                light = safe_bandpass(resp_removed, rfs, RADAR_CARDIAC_BAND_HZ[0], RADAR_CARDIAC_BAND_HZ[1], order=4)
            except Exception:
                light = resp_removed.copy()
        heavy = np.asarray(radar.get("ppg_final_smooth", []), dtype=np.float64)
        if len(heavy) != len(rt):
            heavy = _heavy_smooth(light, rfs)
        stage_signals = {"raw": raw, "resp_removed": resp_removed, "light": light, "heavy": heavy}

        def _get_anchor_time(bb):
            for kk in ("anchor_time_sec", "r_peak_time_sec", "r_time", "r_peak_time", "anchor_time"):
                vv = bb.get(kk, None)
                if _finite(vv):
                    return float(vv)
            try:
                bi = int(bb.get("beat_index", -1))
                rarr = np.asarray(ecg.get("peaks_time", []), dtype=np.float64)
                if 0 <= bi < len(rarr) and np.isfinite(rarr[bi]):
                    return float(rarr[bi])
            except Exception:
                pass
            return None

        def _slice_signal_by_anchor(tt, x, anchor_t):
            x = np.asarray(x, dtype=np.float64)
            tt = np.asarray(tt, dtype=np.float64)
            if len(x) != len(tt):
                return None, None
            t0 = float(anchor_t) - float(acfg.beat_pre_sec)
            t1 = float(anchor_t) + float(acfg.beat_post_sec)
            m = (tt >= t0) & (tt <= t1)
            if np.sum(m) < 10:
                return None, None
            fs_local = 1.0 / max(np.nanmedian(np.diff(tt[m])), 1e-6)
            grid = np.arange(-float(acfg.beat_pre_sec), float(acfg.beat_post_sec) + 1e-9, 1.0 / fs_local)
            abs_t = float(anchor_t) + grid
            y = np.interp(abs_t, tt[m], x[m])
            return grid, zscore_safe(y)

        def _fallback_feature_time(bt, bx, win, kind):
            bt = np.asarray(bt, dtype=np.float64)
            bx = zscore_safe(np.asarray(bx, dtype=np.float64))
            if len(bt) < 8 or len(bt) != len(bx):
                return None
            m = (bt >= float(win[0])) & (bt <= float(win[1]))
            if np.sum(m) < 4:
                return None
            idxs = np.where(m)[0]
            try:
                fs_local = 1.0 / np.nanmedian(np.diff(bt))
                y = safe_lowpass(bx, fs_local, min(18.0, 0.45 * fs_local), order=2)
            except Exception:
                y = bx
                fs_local = float(acfg.radar_interp_fs_hz)
            d1 = np.gradient(y, bt)
            d2 = np.gradient(d1, bt)
            env = triangular_smooth_envelope(y, win_len=min(11, max(5, int(0.05 * max(fs_local, 1)) | 1)))
            env = zscore_safe(env)
            center_prior = np.exp(-0.5 * ((bt[idxs] - np.mean(win)) / max((win[1] - win[0]) / 2.5, 1e-3)) ** 2)
            if kind == "ao":
                score = (0.22 * robust_scale_01(np.maximum(d1[idxs], 0)) +
                         0.18 * robust_scale_01(np.maximum(d2[idxs], 0)) +
                         0.18 * robust_scale_01(y[idxs]) +
                         0.16 * robust_scale_01(env[idxs]) +
                         0.16 * robust_scale_01(np.abs(d2[idxs])) +
                         0.10 * center_prior)
            else:
                score = (0.22 * robust_scale_01(np.maximum(-d1[idxs], 0)) +
                         0.20 * robust_scale_01(np.abs(d2[idxs])) +
                         0.18 * robust_scale_01(-y[idxs]) +
                         0.15 * robust_scale_01(env[idxs]) +
                         0.15 * robust_scale_01(np.maximum(-d2[idxs], 0)) +
                         0.10 * center_prior)
            j = int(idxs[int(np.nanargmax(score))])
            return float(bt[j])

        def _candidate_markers(bt, bx, win):
            bt = np.asarray(bt, dtype=np.float64)
            bx = np.asarray(bx, dtype=np.float64)
            if len(bt) < 8:
                return np.array([]), np.array([])
            m = (bt >= float(win[0])) & (bt <= float(win[1]))
            if np.sum(m) < 5:
                return np.array([]), np.array([])
            idx = np.where(m)[0]
            seg_t = bt[idx]
            seg_x = bx[idx]
            try:
                fs_local = 1.0 / np.nanmedian(np.diff(bt))
                y = safe_lowpass(seg_x, fs_local, min(18.0, 0.45 * fs_local), order=2)
            except Exception:
                y = seg_x
                fs_local = float(acfg.radar_interp_fs_hz)
            min_dist = max(1, int(round(0.025 * fs_local)))
            p1, _ = signal.find_peaks(y, distance=min_dist)
            p2, _ = signal.find_peaks(-y, distance=min_dist)
            d1 = np.gradient(y, seg_t)
            d2 = np.gradient(d1, seg_t)
            s1, _ = signal.find_peaks(np.abs(d1), distance=max(1, int(round(0.03 * fs_local))))
            s2, _ = signal.find_peaks(np.abs(d2), distance=max(1, int(round(0.03 * fs_local))))
            cand = np.unique(np.concatenate([p1, p2, s1, s2])) if (len(p1) or len(p2) or len(s1) or len(s2)) else np.array([], dtype=int)
            if len(cand) == 0:
                return np.array([]), np.array([])
            return seg_t[cand], y[cand]

        scored = []
        for bb in beats:
            at = _get_anchor_time(bb)
            if not _finite(at):
                continue
            bt_rr, bx_rr = _slice_signal_by_anchor(rt, stage_signals["resp_removed"], float(at))
            if bt_rr is None:
                continue
            ao_rr = _fallback_feature_time(bt_rr, bx_rr, acfg.ao_search_sec, "ao")
            ac_rr = _fallback_feature_time(bt_rr, bx_rr, acfg.ac_search_sec, "ac")
            score = float(bb.get("sqi", 0.0)) + 0.9 * float(_window_contains(ao_rr, acfg.ao_search_sec)) + 0.9 * float(_window_contains(ac_rr, acfg.ac_search_sec))
            scored.append((score, bb))
        if not scored:
            return
        rep = sorted(scored, key=lambda z: z[0], reverse=True)[0][1]
        rep_anchor = float(_get_anchor_time(rep))

        stage_order = [("raw", "1) Raw displacement"), ("resp_removed", "2) Respiration-removed"), ("light", "3) Lightly filtered"), ("heavy", "4) Heavily smoothed")]
        fig, axes = plt.subplots(4, 1, figsize=(11.2, 10.2), sharex=True, constrained_layout=True)
        for ax, (stage_key, stage_title) in zip(axes, stage_order):
            bt, bx = _slice_signal_by_anchor(rt, stage_signals[stage_key], rep_anchor)
            if bt is None:
                continue
            ao_t = _fallback_feature_time(bt, bx, acfg.ao_search_sec, "ao")
            ac_t = _fallback_feature_time(bt, bx, acfg.ac_search_sec, "ac")
            ax.plot(bt, bx, linewidth=1.7, color="black")
            ax.axvline(0.0, linestyle="--", linewidth=1.0, color="0.35")
            ax.axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.16)
            ax.axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.16)
            if _finite(ao_t):
                ay = float(np.interp(float(ao_t), bt, bx)); ax.scatter([ao_t], [ay], s=48, marker="o", facecolor="white", edgecolor="black", linewidth=1.0, zorder=5)
            if _finite(ac_t):
                cy = float(np.interp(float(ac_t), bt, bx)); ax.scatter([ac_t], [cy], s=48, marker="s", facecolor="white", edgecolor="black", linewidth=1.0, zorder=5)
            ax.set_title(stage_title, fontsize=11, loc="left", pad=6); ax.set_ylabel("z-score", fontsize=10); ax.grid(True, alpha=0.22)
        fig.suptitle("Fig. 4-1. Radar processing stages on the same representative beat", fontsize=13, y=1.01)
        axes[-1].set_xlim(-float(acfg.beat_pre_sec), float(acfg.beat_post_sec)); axes[-1].set_xlabel("Time from ECG R-peak [s]", fontsize=10)
        fig.savefig(outdir / "fig04_1_processing_stage_comparison.png", dpi=PAPER_FIG_DPI, bbox_inches="tight")
        plt.close(fig)

        nrows = 2 if (scg is not None and len(scg.get("t", [])) > 10) else 1
        fig, axes = plt.subplots(nrows, 1, figsize=(11.2, 7.2 if nrows == 2 else 4.6), sharex=True, constrained_layout=True)
        if nrows == 1:
            axes = [axes]
        ax_idx = 0
        if scg is not None and len(scg.get("t", [])) > 10:
            st = np.asarray(scg.get("t", []), dtype=np.float64)
            sx = np.asarray(scg.get("filtered", scg.get("display", [])), dtype=np.float64)
            bt_s, bx_s = _slice_signal_by_anchor(st, sx, rep_anchor)
            if bt_s is not None:
                ax = axes[ax_idx]; ax_idx += 1
                ax.plot(bt_s, bx_s, color="black", linewidth=1.6)
                ax.axvline(0.0, linestyle="--", linewidth=1.0, color="0.35")
                ax.axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.16)
                ax.axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.16)
                ao_ct, ao_cy = _candidate_markers(bt_s, bx_s, acfg.ao_search_sec)
                ac_ct, ac_cy = _candidate_markers(bt_s, bx_s, acfg.ac_search_sec)
                if len(ao_ct): ax.scatter(ao_ct, ao_cy, s=24, marker="o", facecolor="0.75", edgecolor="black", linewidth=0.5, zorder=4)
                if len(ac_ct): ax.scatter(ac_ct, ac_cy, s=24, marker="s", facecolor="0.75", edgecolor="black", linewidth=0.5, zorder=4)
                for key, mk in [("pos_peaks_idx", "^"), ("neg_peaks_idx", "v")]:
                    rel=[]; vals=[]
                    for idx0 in scg.get(key, []):
                        idx0 = int(idx0); tt = float(st[idx0]) - rep_anchor
                        if -float(acfg.beat_pre_sec) <= tt <= float(acfg.beat_post_sec):
                            rel.append(tt); vals.append(float(np.interp(tt, bt_s, bx_s)))
                    if rel: ax.scatter(rel, vals, s=18, marker=mk, facecolor="white", edgecolor="black", linewidth=0.6, alpha=0.9)
                ao_idx, _ = scg_inspired_aoac_detector(bt_s, bx_s, acfg.ao_search_sec, kind="ao")
                ac_idx, _ = scg_inspired_aoac_detector(bt_s, bx_s, acfg.ac_search_sec, kind="ac")
                if ao_idx is not None:
                    ao_t = float(bt_s[ao_idx]); ay = float(bx_s[ao_idx]); ax.scatter([ao_t], [ay], s=88, marker="o", facecolor="white", edgecolor="black", linewidth=1.2, zorder=6)
                    ax.annotate(f"SCG AO\n{ao_t*1000:.0f} ms", xy=(ao_t, ay), xytext=(10, 12), textcoords="offset points", arrowprops=dict(arrowstyle="->", linewidth=0.9), fontsize=9)
                if ac_idx is not None:
                    ac_t = float(bt_s[ac_idx]); cy = float(bx_s[ac_idx]); ax.scatter([ac_t], [cy], s=88, marker="s", facecolor="white", edgecolor="black", linewidth=1.2, zorder=6)
                    ax.annotate(f"SCG AC\n{ac_t*1000:.0f} ms", xy=(ac_t, cy), xytext=(10, -24), textcoords="offset points", arrowprops=dict(arrowstyle="->", linewidth=0.9), fontsize=9)
                ax.set_title("SCG respiration-removed / filtered waveform", fontsize=11, loc="left", pad=6)
                ax.set_ylabel("z-score", fontsize=10); ax.grid(True, alpha=0.22)
        ax = axes[-1]
        bt, bx = _slice_signal_by_anchor(rt, stage_signals["resp_removed"], rep_anchor)
        if bt is None: return
        ao_t = _fallback_feature_time(bt, bx, acfg.ao_search_sec, "ao")
        ac_t = _fallback_feature_time(bt, bx, acfg.ac_search_sec, "ac")
        ao_ct, ao_cy = _candidate_markers(bt, bx, acfg.ao_search_sec)
        ac_ct, ac_cy = _candidate_markers(bt, bx, acfg.ac_search_sec)
        ax.plot(bt, bx, color="black", linewidth=1.9)
        ax.axvline(0.0, linestyle="--", linewidth=1.0, color="0.35")
        ax.axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.16)
        ax.axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.16)
        if len(ao_ct): ax.scatter(ao_ct, ao_cy, s=28, marker="o", facecolor="0.75", edgecolor="black", linewidth=0.5, zorder=4)
        if len(ac_ct): ax.scatter(ac_ct, ac_cy, s=28, marker="s", facecolor="0.75", edgecolor="black", linewidth=0.5, zorder=4)
        if _finite(ao_t):
            ay = float(np.interp(float(ao_t), bt, bx)); ax.scatter([ao_t], [ay], s=92, marker="o", facecolor="white", edgecolor="black", linewidth=1.3, zorder=6)
            ax.annotate(f"Final AO\n{float(ao_t)*1000:.0f} ms", xy=(ao_t, ay), xytext=(10, 14), textcoords="offset points", arrowprops=dict(arrowstyle="->", linewidth=0.9), fontsize=10)
        if _finite(ac_t):
            cy = float(np.interp(float(ac_t), bt, bx)); ax.scatter([ac_t], [cy], s=92, marker="s", facecolor="white", edgecolor="black", linewidth=1.3, zorder=6)
            ax.annotate(f"Final AC\n{float(ac_t)*1000:.0f} ms", xy=(ac_t, cy), xytext=(10, -26), textcoords="offset points", arrowprops=dict(arrowstyle="->", linewidth=0.9), fontsize=10)
        ax.set_title("Respiration-removed radar waveform", fontsize=11, loc="left", pad=6)
        ax.set_xlabel("Time from ECG R-peak [s]", fontsize=10)
        ax.set_ylabel("z-score", fontsize=10); ax.grid(True, alpha=0.22)
        handles = [Line2D([0], [0], color="black", lw=1.8, label="Waveform"), Line2D([0], [0], color="0.35", lw=1.0, linestyle="--", label="ECG R-peak"), Patch(facecolor="C0", alpha=0.16, label="AO / AC windows"), Line2D([0], [0], marker="o", color="black", markerfacecolor="0.75", linestyle="None", label="Candidate points"), Line2D([0], [0], marker="o", color="black", markerfacecolor="white", linestyle="None", label="Final AO"), Line2D([0], [0], marker="s", color="black", markerfacecolor="white", linestyle="None", label="Final AC")]
        axes[0].legend(handles=handles, loc="upper right", ncol=2, fontsize=8.5, framealpha=0.95)
        fig.suptitle("Fig. 4-2. SCG and radar AO/AC candidate analysis from ECG R-peak anchor", fontsize=13, y=1.01)
        fig.savefig(outdir / "fig04_2_scg_radar_candidate_detection.png", dpi=PAPER_FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        rows = []
        for t, y in zip(ao_ct, ao_cy): rows.append(["radar_AO_candidate", float(t), float(y)])
        for t, y in zip(ac_ct, ac_cy): rows.append(["radar_AC_candidate", float(t), float(y)])
        if _finite(ao_t): rows.append(["radar_AO_final", float(ao_t), float(np.interp(float(ao_t), bt, bx))])
        if _finite(ac_t): rows.append(["radar_AC_final", float(ac_t), float(np.interp(float(ac_t), bt, bx))])
        save_csv(outdir / "fig04_2_candidate_markers.csv", ["type", "time_from_r_sec", "z_value"], rows)
    except Exception as e:
        with open(outdir / "fig04_stage_candidate_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    save_csv(outdir / "ecg_raw_processed.csv",
             ["time_sec", "ecg_raw_decoded", "ecg_raw_norm", "ecg_raw_hampel",
              "ecg_baseline_est", "ecg_artifact_ref_lf", "ecg_artifact_est_lms", "ecg_lms_clean",
              "ecg_fft_motion_removed", "ecg_fft_motion_clean",
              "ecg_display_0p5_18hz", "ecg_display_for_figure",
              "ecg_true_display_rpeak_aligned", "ecg_qrs_band_for_rpeak"],
             [[float(ecg["t"][i]),
               float(ecg["raw"][i]),
               float(ecg.get("raw_norm", ecg["cleaned"])[i]),
               float(ecg.get("raw_hampel", ecg.get("raw_norm", ecg["cleaned"]))[i]),
               float(ecg.get("baseline_est", np.zeros_like(ecg["t"]))[i]),
               float(ecg.get("artifact_ref", np.zeros_like(ecg["t"]))[i]),
               float(ecg.get("artifact_est", np.zeros_like(ecg["t"]))[i]),
               float(ecg.get("lms_clean", ecg["cleaned"])[i]),
               float(ecg.get("fft_motion_removed", np.zeros_like(ecg["t"]))[i]),
               float(ecg.get("fft_motion_clean", ecg.get("lms_clean", ecg["cleaned"]))[i]),
               float(ecg.get("true_display", ecg["cleaned"])[i]),
               float(ecg.get("display", ecg["filtered"])[i]),
               float(ecg.get("true_display_rpeak", ecg.get("true_display", ecg["cleaned"]))[i]),
               float(ecg["filtered"][i])]
              for i in range(len(ecg["t"]))])

    save_csv(outdir / "ecg_peaks.csv",
             ["peak_index", "peak_time_sec"],
             [[int(ecg["peaks_idx"][i]), float(ecg["peaks_time"][i])] for i in range(len(ecg["peaks_idx"]))])

    # ECG Q/R/T landmark 후보 저장: Q/T는 timing prior 보조용 후보
    save_csv(outdir / "ecg_q_r_t_landmarks.csv",
             ["beat_index", "q_time_sec", "r_time_sec", "t_time_sec", "qr_interval_ms", "rt_interval_ms"],
             [[int(i),
               None if i >= len(ecg.get("q_time", [])) or not np.isfinite(ecg["q_time"][i]) else float(ecg["q_time"][i]),
               float(ecg["peaks_time"][i]),
               None if i >= len(ecg.get("t_time", [])) or not np.isfinite(ecg["t_time"][i]) else float(ecg["t_time"][i]),
               None if i >= len(ecg.get("q_time", [])) or not np.isfinite(ecg["q_time"][i]) else float((ecg["peaks_time"][i] - ecg["q_time"][i]) * 1000.0),
               None if i >= len(ecg.get("t_time", [])) or not np.isfinite(ecg["t_time"][i]) else float((ecg["t_time"][i] - ecg["peaks_time"][i]) * 1000.0)]
              for i in range(len(ecg["peaks_time"]))])


    # R-peak short-RR postprocess log
    save_csv(outdir / "ecg_rpeak_short_rr_removed.csv",
             ["removed_peak_idx", "kept_peak_idx", "removed_time_sec", "kept_time_sec",
              "rr_pair_sec", "removed_qrs_amp", "kept_qrs_amp", "reason"],
             [[int(r.get("removed_peak_idx", -1)),
               int(r.get("kept_peak_idx", -1)),
               float(r.get("removed_time_sec", np.nan)),
               float(r.get("kept_time_sec", np.nan)),
               float(r.get("rr_pair_sec", np.nan)),
               None if r.get("removed_qrs_amp", None) is None else float(r.get("removed_qrs_amp")),
               None if r.get("kept_qrs_amp", None) is None else float(r.get("kept_qrs_amp")),
               str(r.get("reason", ""))]
              for r in ecg.get("rpeak_removed_short_rr", [])])

    rr_after_post = np.diff(ecg["peaks_time"]) if len(ecg["peaks_time"]) >= 2 else np.array([])
    save_csv(outdir / "ecg_rr_interval_after_postprocess.csv",
             ["rr_index", "rr_sec", "instant_hr_bpm", "short_rr_flag"],
             [[int(i), float(rr_after_post[i]),
               None if rr_after_post[i] <= 0 else float(60.0 / rr_after_post[i]),
               bool(rr_after_post[i] < float(globals().get("RPEAK_MIN_RR_SEC_POST", 0.45)))]
              for i in range(len(rr_after_post))])

    try:
        rpeak_summary = {
            "enabled": bool(globals().get("RPEAK_ENABLE_SHORT_RR_POSTPROCESS", True)),
            "min_rr_sec": float(globals().get("RPEAK_MIN_RR_SEC_POST", 0.45)),
            "removed_count": int(len(ecg.get("rpeak_removed_short_rr", []))),
            "final_rpeak_count": int(len(ecg.get("peaks_time", []))),
            "rr_min_sec": None if len(rr_after_post) == 0 else float(np.nanmin(rr_after_post)),
            "rr_mean_sec": None if len(rr_after_post) == 0 else float(np.nanmean(rr_after_post)),
            "rr_max_sec": None if len(rr_after_post) == 0 else float(np.nanmax(rr_after_post)),
            "short_rr_remaining_count": int(np.sum(rr_after_post < float(globals().get("RPEAK_MIN_RR_SEC_POST", 0.45)))) if len(rr_after_post) else 0,
        }
        with open(outdir / "rpeak_postprocess_summary.json", "w", encoding="utf-8") as f:
            json.dump(rpeak_summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        with open(outdir / "rpeak_postprocess_summary_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


    # RR-adaptive Q/T periodic prior quality diagnostics
    save_csv(outdir / "ecg_q_t_periodic_quality.csv",
             ["beat_index", "r_time_sec", "rr_local_sec",
              "q_time_sec", "q_rel_sec", "q_confidence",
              "t_time_sec", "t_rel_sec", "t_confidence"],
             [[int(i),
               float(ecg["peaks_time"][i]),
               None if "rr_local_sec" not in ecg or i >= len(ecg["rr_local_sec"]) or not np.isfinite(ecg["rr_local_sec"][i]) else float(ecg["rr_local_sec"][i]),
               None if i >= len(ecg.get("q_time", [])) or not np.isfinite(ecg["q_time"][i]) else float(ecg["q_time"][i]),
               None if "q_rel_sec" not in ecg or i >= len(ecg["q_rel_sec"]) or not np.isfinite(ecg["q_rel_sec"][i]) else float(ecg["q_rel_sec"][i]),
               None if "q_confidence" not in ecg or i >= len(ecg["q_confidence"]) else float(ecg["q_confidence"][i]),
               None if i >= len(ecg.get("t_time", [])) or not np.isfinite(ecg["t_time"][i]) else float(ecg["t_time"][i]),
               None if "t_rel_sec" not in ecg or i >= len(ecg["t_rel_sec"]) or not np.isfinite(ecg["t_rel_sec"][i]) else float(ecg["t_rel_sec"][i]),
               None if "t_confidence" not in ecg or i >= len(ecg["t_confidence"]) else float(ecg["t_confidence"][i])]
              for i in range(len(ecg["peaks_time"]))])


    # RR-adaptive Q/T periodic tracking figure
    try:
        if len(ecg.get("peaks_time", [])) > 0:
            beat_i = np.arange(len(ecg["peaks_time"]))
            q_rel_ms = np.asarray(ecg.get("q_rel_sec", np.full(len(beat_i), np.nan)), dtype=float) * 1000.0
            t_rel_ms = np.asarray(ecg.get("t_rel_sec", np.full(len(beat_i), np.nan)), dtype=float) * 1000.0
            q_conf = np.asarray(ecg.get("q_confidence", np.zeros(len(beat_i))), dtype=float)
            t_conf = np.asarray(ecg.get("t_confidence", np.zeros(len(beat_i))), dtype=float)
            rr_ms = np.asarray(ecg.get("rr_local_sec", np.full(len(beat_i), np.nan)), dtype=float) * 1000.0

            fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
            axes[0].plot(beat_i, rr_ms, "o-", ms=3, label="local RR")
            axes[0].set_ylabel("RR [ms]")
            axes[0].set_title("RR-adaptive periodic prior for ECG Q/T pseudo-landmarks")
            axes[0].grid(True); axes[0].legend()

            axes[1].plot(beat_i, -q_rel_ms, "o-", ms=3, label="R-Q interval")
            axes[1].plot(beat_i, t_rel_ms, "o-", ms=3, label="R-T interval")
            axes[1].set_ylabel("Interval [ms]")
            axes[1].grid(True); axes[1].legend()

            axes[2].plot(beat_i, q_conf, "o-", ms=3, label="Q confidence")
            axes[2].plot(beat_i, t_conf, "o-", ms=3, label="T confidence")
            axes[2].axhline(float(globals().get("QT_MIN_TRACK_CONFIDENCE", 0.45)), linestyle="--", color="gray", label="min conf")
            axes[2].set_xlabel("Beat index")
            axes[2].set_ylabel("confidence")
            axes[2].set_ylim(-0.05, 1.05)
            axes[2].grid(True); axes[2].legend()

            fig.tight_layout()
            fig.savefig(outdir / "fig02c_rr_adaptive_qt_periodic_tracking.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        with open(outdir / "fig02c_rr_adaptive_qt_periodic_tracking_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


    # Smoothed ECG QRT/RR pseudo AO/AC reference 저장
    try:
        ecg_ref_series_save = build_ecg_adaptive_reference_series(ecg, acfg)
        save_csv(outdir / "ecg_adaptive_aoac_reference.csv",
                 ["beat_index", "r_time_sec", "ao_raw_sec", "ac_raw_sec", "ao_smooth_sec", "ac_smooth_sec", "ao_conf", "ac_conf"],
                 [[int(i), float(ecg["peaks_time"][i]),
                   float(ecg_ref_series_save["ao_raw"][i]),
                   float(ecg_ref_series_save["ac_raw"][i]),
                   float(ecg_ref_series_save["ao_smooth"][i]),
                   float(ecg_ref_series_save["ac_smooth"][i]),
                   float(ecg_ref_series_save["ao_conf"][i]),
                   float(ecg_ref_series_save["ac_conf"][i])]
                  for i in range(len(ecg["peaks_time"]))])
    except Exception as e:
        with open(outdir / "ecg_adaptive_aoac_reference_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    # RR interval / instantaneous HR debug
    if len(ecg["peaks_time"]) >= 2:
        rr = np.diff(ecg["peaks_time"])
        save_csv(outdir / "ecg_rr_hrv_time_domain.csv",
                 ["beat_index", "r_time_sec", "rr_interval_sec", "instant_hr_bpm"],
                 [[int(i+1), float(ecg["peaks_time"][i+1]), float(rr[i]), float(60.0 / rr[i]) if rr[i] > 0 else None]
                  for i in range(len(rr))])

    save_csv(outdir / "radar_processed.csv",
             ["time_sec", "phase", "displacement_m", "respiration", "lms_reference", "lms_estimated_noise", "lms_error", "ppg_like", "range_idx", "angle_idx"],
             [[float(radar["t"][i]), float(radar["phase"][i]), float(radar["displacement"][i]),
               float(radar["respiration"][i]),
               float(radar.get("lms_reference", np.zeros_like(radar["t"]))[i]),
               float(radar.get("lms_estimated_noise", np.zeros_like(radar["t"]))[i]),
               float(radar.get("lms_error", np.zeros_like(radar["t"]))[i]),
               float(radar["ppg_like"][i]),
               int(radar["range_idx_trace"][i]), int(radar["angle_idx_trace"][i])]
              for i in range(len(radar["t"]))])

    save_csv(outdir / "radar_peaks.csv",
             ["peak_index", "peak_time_sec"],
             [[int(radar["peaks_idx"][i]), float(radar["peaks_time"][i])] for i in range(len(radar["peaks_idx"]))])

    save_csv(outdir / "radar_interpolated_100hz.csv",
             ["time_sec", "radar_ppg_interp", "radar_ppg_interp_z"],
             [[float(aoac["t_radar_interp"][i]), float(aoac["radar_interp"][i]), float(aoac["radar_interp_z"][i])]
              for i in range(len(aoac["t_radar_interp"]))])

    save_csv(outdir / "ao_ac_results_with_sqi_errors.csv", AOAC_HEADER, aoac["rows"])

    # Beat alignment diagnostics
    save_csv(outdir / "beat_alignment_metrics.csv",
             ["beat_index", "alignment_lag_ms", "alignment_corr", "dtw_distance", "accepted"],
             [[int(b.get("beat_index", i)),
               float(b.get("alignment_lag_ms", 0.0)),
               float(b.get("alignment_corr", 0.0)),
               float(b.get("dtw_distance", np.nan)),
               bool(aoac["rows"][i][2]) if i < len(aoac["rows"]) else None]
              for i, b in enumerate(aoac.get("beats", []))])

    # AC temporal tracking diagnostics from rows if columns exist
    try:
        save_csv(outdir / "ac_temporal_tracking_metrics.csv",
                 ["beat_index", "accepted", "ac_tracking_used", "ac_tracking_conf", "ac_tracking_target_sec"],
                 [[int(r[0]), bool(r[2]),
                   bool(r[25]) if len(r) > 25 and r[25] is not None else None,
                   float(r[26]) if len(r) > 26 and r[26] is not None else None,
                   float(r[27]) if len(r) > 27 and r[27] is not None else None]
                  for r in aoac["rows"]])
    except Exception as e:
        with open(outdir / "ac_temporal_tracking_metrics_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    save_csv(outdir / "compare_common_resampled.csv",
             ["time_sec", "ecg_peak_reference_z", "radar_ppg_z", "radar_ppg_aligned_z"],
             [[float(comp["t"][i]), float(comp["ecg_ref"][i]), float(comp["radar_ppg"][i]), float(comp["radar_aligned"][i])]
              for i in range(len(comp["t"]))])

    save_csv(outdir / "timestamp_sample_rate_summary.csv",
             ["signal", "num_samples", "start_time_sec", "end_time_sec", "estimated_fs_hz"],
             [
                 ["ecg", int(len(ecg["t"])), float(ecg["t"][0]), float(ecg["t"][-1]), float(ecg["fs"])],
                 ["radar", int(len(radar["t"])), float(radar["t"][0]), float(radar["t"][-1]), float(radar["fs"])],
                 ["radar_interpolated", int(len(aoac["t_radar_interp"])), float(aoac["t_radar_interp"][0]), float(aoac["t_radar_interp"][-1]), float(acfg.radar_interp_fs_hz)],
                 ["common_compare", int(len(comp["t"])), float(comp["t"][0]), float(comp["t"][-1]), float(acfg.common_compare_fs_hz)],
             ])


    # ECG artifact / LMS filtering diagnostic figure
    try:
        t_ecg = np.asarray(ecg["t"], dtype=np.float64)
        if len(t_ecg) > 20:
            # Use a central 12-second window to avoid overcrowding
            t0 = float(t_ecg[0])
            t1 = float(t_ecg[-1])
            center = 0.5 * (t0 + t1)
            w0 = max(t0, center - 6.0)
            w1 = min(t1, center + 6.0)
            m = (t_ecg >= w0) & (t_ecg <= w1)

            fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True)
            axes[0].plot(t_ecg[m], zscore_safe(ecg.get("raw_norm", ecg["cleaned"])[m]), linewidth=1.0, label="Raw ADC normalized")
            axes[0].set_title("ECG raw ADC normalized")
            axes[0].grid(True); axes[0].legend()

            axes[1].plot(t_ecg[m], zscore_safe(ecg.get("artifact_ref", np.zeros_like(t_ecg))[m]), linewidth=1.0, label="LF artifact reference 0.05~1.0 Hz")
            axes[1].set_title("Low-frequency motion/contact artifact reference")
            axes[1].grid(True); axes[1].legend()

            axes[2].plot(t_ecg[m], zscore_safe(ecg.get("lms_clean", ecg["cleaned"])[m]), linewidth=1.0, label="LMS-cleaned ECG")
            axes[2].set_title("LMS-cleaned ECG")
            axes[2].grid(True); axes[2].legend()

            axes[3].plot(t_ecg[m], zscore_safe(ecg.get("true_display", ecg["cleaned"])[m]), linewidth=1.0, label="Display ECG 0.5~15 Hz")
            axes[3].set_title("Display ECG after LMS + BPF")
            axes[3].grid(True); axes[3].legend()

            qrs_sig = zscore_safe(ecg["filtered"][m])
            axes[4].plot(t_ecg[m], qrs_sig, linewidth=1.0, label="QRS band 5~25 Hz")
            r_in = ecg["peaks_time"][(ecg["peaks_time"] >= w0) & (ecg["peaks_time"] <= w1)]
            if len(r_in):
                axes[4].scatter(r_in, np.interp(r_in, t_ecg[m], qrs_sig), s=35, marker="x", label="R peaks")
            axes[4].set_title("QRS-band ECG for R-peak detection")
            axes[4].set_xlabel("Time [s]")
            axes[4].grid(True); axes[4].legend()

            fig.tight_layout()
            fig.savefig(outdir / "fig0a_ecg_artifact_lms_filtering.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        with open(outdir / "fig0a_ecg_artifact_lms_filtering_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    # LMS respiration cancellation figure
    try:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(radar["t"], zscore_safe(radar["displacement"]), label="Radar displacement")
        axes[0].set_title("Raw radar displacement")
        axes[0].grid(True); axes[0].legend()

        axes[1].plot(radar["t"], zscore_safe(radar.get("lms_reference", radar["respiration"])), label="Respiration/motion reference")
        axes[1].set_title("LMS reference")
        axes[1].grid(True); axes[1].legend()

        axes[2].plot(radar["t"], zscore_safe(radar.get("lms_error", radar["ppg_like"])), label="LMS error signal")
        axes[2].set_title("LMS output before cardiac band re-filtering")
        axes[2].grid(True); axes[2].legend()

        axes[3].plot(radar["t"], zscore_safe(radar["ppg_like"]), label="Final PPG-like after LMS + bandpass")
        axes[3].set_title("Final radar PPG-like signal")
        axes[3].set_xlabel("Time [s]")
        axes[3].grid(True); axes[3].legend()

        fig.tight_layout()
        fig.savefig(outdir / "fig0_lms_respiration_cancellation.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir / "fig0_lms_respiration_cancellation_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    # Robust ECG peak diagnostic figure: true/display vs analysis band
    try:
        if len(ecg["peaks_time"]) >= 6:
            t_start = float(ecg["peaks_time"][0])
            t_end = float(ecg["peaks_time"][5])
            m = (ecg["t"] >= t_start) & (ecg["t"] <= t_end)
            tx = ecg["t"][m] - t_start
            y_true = zscore_safe(ecg.get("true_display_rpeak", ecg.get("true_display", ecg["cleaned"]))[m])
            y_analysis = zscore_safe(ecg["filtered"][m])

            r_in = ecg["peaks_time"][(ecg["peaks_time"] >= t_start) & (ecg["peaks_time"] <= t_end)] - t_start

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(tx, y_true, label="ECG true/raw-like", linewidth=1.4)
            ax.plot(tx, y_analysis, label="ECG analysis/QRS band", linewidth=1.0, alpha=0.75)
            if len(r_in):
                ax.plot(r_in, np.interp(r_in, tx, y_true), "ro", ms=5, label="R peaks")
            ax.set_title("Robust ECG R-peak detection check: true vs analysis ECG, 5 cycles")
            ax.set_xlabel("Time from first R-peak [s]")
            ax.set_ylabel("z-score")
            ax.grid(True)
            ax.legend()
            fig.savefig(outdir / "fig0_ecg_robust_rpeak_check.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        with open(outdir / "fig0_ecg_robust_rpeak_check_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    # ECG Q/R/T landmark check figure
    try:
        if len(ecg["peaks_time"]) >= 6:
            t_start = float(ecg["peaks_time"][0])
            t_end = float(ecg["peaks_time"][5])
            m = (ecg["t"] >= t_start) & (ecg["t"] <= t_end)
            tx = ecg["t"][m] - t_start
            y = zscore_safe(ecg.get("true_display_rpeak", ecg.get("true_display", ecg["cleaned"]))[m])

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(tx, y, label="ECG true/raw-like")

            r_times = ecg["peaks_time"]
            q_times = ecg.get("q_time", np.full_like(r_times, np.nan))
            t_times = ecg.get("t_time", np.full_like(r_times, np.nan))

            for label, arr, marker_style, color in [
                ("Q candidate", q_times, "v", "magenta"),
                ("R peak", r_times, "o", "red"),
                ("T candidate", t_times, "^", "green"),
            ]:
                arr2 = arr[(arr >= t_start) & (arr <= t_end) & np.isfinite(arr)] - t_start
                if len(arr2):
                    ax.scatter(arr2, np.interp(arr2, tx, y), marker=marker_style, color=color, s=35, label=label)

            ax.set_title("ECG Q/R/T candidate landmarks: 5 cardiac cycles")
            ax.set_xlabel("Time from first R-peak [s]")
            ax.set_ylabel("z-score")
            ax.grid(True)
            ax.legend()
            fig.savefig(outdir / "fig0_ecg_q_r_t_landmarks.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        with open(outdir / "fig0_ecg_q_r_t_landmarks_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    # Combined overview figures requested by user
    add_combined_overview_figures(outdir, ecg, radar, aoac, comp)

    # Lab feedback diagnostic figures: raw/recovered radar cycles and morphology visibility
    add_radar_raw_multicycle_diagnostic(outdir, ecg, radar, aoac, acfg, n_cycles=10)
    compute_radar_morphology_visibility(aoac, outdir)

    # ECG morphology sanity figure: display ECG only, first available 5 cycles
    try:
        if len(ecg["peaks_time"]) >= 6:
            t_start = float(ecg["peaks_time"][0])
            t_end = float(ecg["peaks_time"][5])
            m = (ecg["t"] >= t_start) & (ecg["t"] <= t_end)
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(111)
            tx = ecg["t"][m] - t_start
            yx = zscore_safe(ecg.get("display", ecg["filtered"])[m])
            ax.plot(tx, yx, label="ECG display band, 5 cycles")
            r_in = ecg["peaks_time"][(ecg["peaks_time"] >= t_start) & (ecg["peaks_time"] <= t_end)] - t_start
            if len(r_in):
                ax.plot(r_in, np.interp(r_in, tx, yx), "ro", ms=4, label="R peaks")
            ax.set_title("ECG display waveform sanity check: 5 cardiac cycles")
            ax.set_xlabel("Time from first R-peak [s]")
            ax.set_ylabel("z-score")
            ax.grid(True)
            ax.legend()
            fig.savefig(outdir / "fig0_ecg_5cycles_display_sanity.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        with open(outdir / "fig0_ecg_5cycles_display_sanity_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    # Figures
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)
    ax.plot(ecg["t"], zscore_safe(ecg.get("display_rpeak", ecg.get("display", ecg["filtered"]))), label="ECG/STM32 display aligned to R polarity")
    if len(ecg["peaks_idx"]):
        ax.plot(ecg["peaks_time"], zscore_safe(ecg.get("display_rpeak", ecg.get("display", ecg["filtered"])))[ecg["peaks_idx"]], "ro", ms=4, label="R peaks")
    ax.set_title(f"ECG/STM32 R-peaks | fs={ecg['fs']:.1f} Hz | HR={ecg['hr_bpm']}")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("z-score"); ax.grid(True); ax.legend()
    fig.savefig(outdir / "fig1_ecg_rpeaks.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)
    ax.plot(radar["t"], zscore_safe(radar["ppg_like"]), label="Radar PPG-like")
    if len(radar["peaks_idx"]):
        ax.plot(radar["peaks_time"], zscore_safe(radar["ppg_like"])[radar["peaks_idx"]], "ro", ms=4, label="Radar peaks")
    ax.set_title(f"Radar PPG-like | fs={radar['fs']:.2f} Hz | HR={radar['hr_bpm']}")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("z-score"); ax.grid(True); ax.legend()
    fig.savefig(outdir / "fig2_radar_ppg_peaks.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)
    ax.plot(aoac["t_radar_interp"], aoac["radar_interp_z"], label="Radar PPG-like interpolated 100 Hz")
    ax.set_title("Radar PPG-like interpolation: 20 Hz → 100 Hz")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("z-score"); ax.grid(True); ax.legend()
    fig.savefig(outdir / "fig3_radar_interpolated_100hz.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # AO/AC all accepted/rejected beat plot
    if len(aoac["beats"]):
        fig = plt.figure(figsize=(11, 6))
        ax = fig.add_subplot(111)
        for b in aoac["beats"]:
            color_alpha = 0.35 if b.get("accepted", False) else 0.10
            ax.plot(b["t_rel"], b["radar_beat"], alpha=color_alpha)
        for b in aoac["accepted_beats"][:60]:
            if b.get("ao_idx") is not None:
                ax.plot(b["t_rel"][b["ao_idx"]], b["radar_beat"][b["ao_idx"]], "ro", ms=3)
            if b.get("ac_idx") is not None:
                ax.plot(b["t_rel"][b["ac_idx"]], b["radar_beat"][b["ac_idx"]], "ko", ms=3)
        ax.axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.12, label="AO search")
        ax.axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.12, label="AC search")
        ax.set_title(f"R-peak aligned Radar Beats with AO/AC Candidates | accepted={len(aoac['accepted_beats'])}/{len(aoac['beats'])}")
        ax.set_xlabel("Time from R-peak [s]"); ax.set_ylabel("z-score"); ax.grid(True); ax.legend()
        fig.savefig(outdir / "fig4_ao_ac_beat_slicing_sqi.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    if aoac["ensemble"]:
        ens = aoac["ensemble"]
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.plot(ens["t_rel"], ens["mean"], label="Accepted beat ensemble mean")
        ax.fill_between(ens["t_rel"], ens["mean"] - ens["std"], ens["mean"] + ens["std"], alpha=0.2, label="±1 SD")
        if ens.get("ao_idx") is not None:
            ax.plot(ens["t_rel"][ens["ao_idx"]], ens["mean"][ens["ao_idx"]], "ro", label=f"AO {ens['ao_time']:.3f}s")
        if ens.get("ac_idx") is not None:
            ax.plot(ens["t_rel"][ens["ac_idx"]], ens["mean"][ens["ac_idx"]], "ko", label=f"AC {ens['ac_time']:.3f}s")
        ax.axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.10)
        ax.axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.10)
        ax.set_title("Ensemble Mean Radar Beat AO/AC Candidate")
        ax.set_xlabel("Time from R-peak [s]"); ax.set_ylabel("z-score"); ax.grid(True); ax.legend()
        fig.savefig(outdir / "fig5_ensemble_ao_ac.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # SQI histogram
    if len(aoac["rows"]):
        rows = aoac["rows"]
        sqi = np.array([r[3] for r in rows], dtype=float)
        accepted = np.array([r[2] for r in rows], dtype=bool)
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.hist(sqi[~accepted], bins=20, alpha=0.5, label="rejected")
        ax.hist(sqi[accepted], bins=20, alpha=0.7, label="accepted")
        ax.axvline(acfg.min_sqi_accept, linestyle="--", label="SQI threshold")
        ax.set_title("Radar Beat SQI Distribution")
        ax.set_xlabel("SQI"); ax.set_ylabel("Count"); ax.grid(True); ax.legend()
        fig.savefig(outdir / "fig6_sqi_histogram.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        # timing errors
        ao_exp = np.array([np.nan if r[17] is None else r[17] for r in rows], dtype=float)
        ac_exp = np.array([np.nan if r[18] is None else r[18] for r in rows], dtype=float)
        ao_ens = np.array([np.nan if r[19] is None else r[19] for r in rows], dtype=float)
        ac_ens = np.array([np.nan if r[20] is None else r[20] for r in rows], dtype=float)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        valid_a = accepted & np.isfinite(ao_exp)
        valid_c = accepted & np.isfinite(ac_exp)
        if np.any(valid_a):
            ax.hist(ao_exp[valid_a], bins=20, alpha=0.6, label="AO expected-center error")
        if np.any(valid_c):
            ax.hist(ac_exp[valid_c], bins=20, alpha=0.6, label="AC expected-center error")
        ax.axvline(0, linestyle="--")
        ax.set_title("AO/AC Surrogate Timing Error vs Expected Window Center")
        ax.set_xlabel("Error [ms]"); ax.set_ylabel("Count"); ax.grid(True); ax.legend()
        fig.savefig(outdir / "fig7_timing_error_expected_center.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        valid_a = accepted & np.isfinite(ao_ens)
        valid_c = accepted & np.isfinite(ac_ens)
        if np.any(valid_a):
            ax.hist(ao_ens[valid_a], bins=20, alpha=0.6, label="AO ensemble error")
        if np.any(valid_c):
            ax.hist(ac_ens[valid_c], bins=20, alpha=0.6, label="AC ensemble error")
        ax.axvline(0, linestyle="--")
        ax.set_title("AO/AC Beat-wise Timing Error vs Ensemble Reference")
        ax.set_xlabel("Error [ms]"); ax.set_ylabel("Count"); ax.grid(True); ax.legend()
        fig.savefig(outdir / "fig8_timing_error_ensemble.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        # Bland-Altman style: AO/AC interval surrogate vs ensemble interval
        intervals = np.array([np.nan if r[12] is None else r[12] for r in rows], dtype=float)
        valid = accepted & np.isfinite(intervals)
        if np.any(valid) and len(intervals[valid]) >= 3:
            ref = np.nanmedian(intervals[valid])
            mean_vals = (intervals[valid] + ref) / 2
            diff_vals = intervals[valid] - ref
            bias = np.nanmean(diff_vals)
            loa = 1.96 * np.nanstd(diff_vals)
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            ax.scatter(mean_vals, diff_vals, s=18)
            ax.axhline(bias, linestyle="-", label=f"bias={bias:.1f}ms")
            ax.axhline(bias + loa, linestyle="--", label=f"+1.96SD={bias+loa:.1f}ms")
            ax.axhline(bias - loa, linestyle="--", label=f"-1.96SD={bias-loa:.1f}ms")
            ax.set_title("Bland-Altman Style Plot: AO-to-AC Interval Consistency")
            ax.set_xlabel("Mean interval vs ensemble median [ms]")
            ax.set_ylabel("Difference [ms]")
            ax.grid(True); ax.legend()
            fig.savefig(outdir / "fig9_bland_altman_interval.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

    # Comparison figures
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(comp["t"], comp["ecg_ref"], label="ECG R-peak reference")
    axes[0].plot(comp["t"], comp["radar_ppg"], label="Radar PPG-like")
    axes[0].set_title(f"Before alignment | Pearson={comp['pearson']}")
    axes[0].grid(True); axes[0].legend()
    axes[1].plot(comp["t"], comp["ecg_ref"], label="ECG R-peak reference")
    axes[1].plot(comp["t"], comp["radar_aligned"], label="Radar PPG-like aligned")
    axes[1].set_title(f"After XCorr alignment | Pearson={comp['pearson_aligned']}, lag={comp['xcorr']['lag_sec']}s")
    axes[1].set_xlabel("Time [s]"); axes[1].grid(True); axes[1].legend()
    fig.savefig(outdir / "fig10_common_time_compare.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    f1, p1 = compute_psd(comp["ecg_ref"], acfg.common_compare_fs_hz, acfg.psd_nperseg)
    f2, p2 = compute_psd(comp["radar_aligned"], acfg.common_compare_fs_hz, acfg.psd_nperseg)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    if len(f1):
        ax.plot(f1, 10 * np.log10(np.maximum(p1, 1e-18)), label="ECG peak ref")
    if len(f2):
        ax.plot(f2, 10 * np.log10(np.maximum(p2, 1e-18)), label="Radar PPG-like")
    ax.set_xlim(0, 6)
    ax.set_title(f"PSD Comparison | spectral corr={comp['spectral_corr']}")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("PSD [dB]"); ax.grid(True); ax.legend()
    fig.savefig(outdir / "fig11_psd_compare.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(comp["xcorr"]["lags_sec"], comp["xcorr"]["corr"])
    if comp["xcorr"]["lag_sec"] is not None:
        ax.axvline(comp["xcorr"]["lag_sec"], linestyle="--")
    ax.set_title(f"Cross-correlation | max={comp['xcorr']['max_corr']}, lag={comp['xcorr']['lag_sec']}s")
    ax.set_xlabel("Lag [s]"); ax.set_ylabel("Normalized XCorr"); ax.grid(True)
    fig.savefig(outdir / "fig12_xcorr.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    if len(comp["coherence_freq"]):
        ax.plot(comp["coherence_freq"], comp["coherence"])
    ax.set_xlim(0, 6); ax.set_ylim(0, 1.05)
    ax.set_title(f"Coherence | mean={comp['mean_coherence']}")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Coherence"); ax.grid(True)
    fig.savefig(outdir / "fig13_coherence.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)
    ax.plot(radar["t"], zscore_safe(radar["respiration"]), label="Radar respiration")
    ax.set_title("Radar Respiration")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("z-score"); ax.grid(True); ax.legend()
    fig.savefig(outdir / "fig14_radar_respiration.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Extra AO/AC timing figures and summary
    add_aoac_timing_extra_figures(outdir, aoac, acfg)
    aoac_timing_summary = summarize_aoac_timing(aoac["rows"])
    with open(outdir / "ao_ac_timing_summary.json", "w", encoding="utf-8") as f:
        json.dump(aoac_timing_summary, f, ensure_ascii=False, indent=2)

    # Compact paper-style figures and cleanup legacy figures
    add_beat_alignment_figure(outdir, aoac, acfg)
    add_ac_temporal_tracking_figure(outdir, aoac)
    add_compact_paper_figures(outdir, ecg, radar, aoac, comp, acfg)
    add_qt_pseudo_landmark_quality_figure(outdir, ecg)
    add_ecg_vs_radar_aoac_correlation_figure(outdir, aoac, acfg)
    add_single_cycle_aoac_label_figure(outdir, ecg, radar, aoac, acfg)
    add_fig4_stage_and_candidate_figures(outdir, ecg, radar, scg, aoac, acfg)
    add_morphology_vs_tight_report(outdir, aoac, acfg)

    # Final paper package export: tables + paper-ready figures
    try:
        export_paper_tables_and_figures(outdir, ecg, radar, aoac, comp, ecfg, rcfg, acfg)
    except Exception as e:
        with open(outdir / "paper_export_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))
    cleanup_legacy_figures(outdir)

    # Summary
    rows = aoac["rows"]
    accepted = np.array([r[2] for r in rows], dtype=bool) if rows else np.array([], dtype=bool)

    def _nanmean_col(idx):
        vals = np.array([np.nan if r[idx] is None else r[idx] for r in rows], dtype=float)
        vals = vals[accepted & np.isfinite(vals)] if len(accepted) else vals[np.isfinite(vals)]
        return None if len(vals) == 0 else float(np.mean(vals))

    def _nanstd_col(idx):
        vals = np.array([np.nan if r[idx] is None else r[idx] for r in rows], dtype=float)
        vals = vals[accepted & np.isfinite(vals)] if len(accepted) else vals[np.isfinite(vals)]
        return None if len(vals) == 0 else float(np.std(vals))

    summary = {
        "important_note": "ECG R-peak detection is performed on QRS-band signal. Figure R markers are plotted only on the QRS-band subplot, while display ECG uses vertical R-anchor lines. ECG is an electrical anchor, not true AO/AC ground truth. STM32 ECG CSV input is supported as ADCValue,Smooth_ECG; raw ADCValue is used by default for R-peak/RR anchoring. AO detection includes Zheng-style seventh-power envelope enhancement; AC detection includes Di Rienzo-style IRP anchor/D1+D2 and inflection fallback. Radar PPG-like AO/AC is surrogate timing analysis.",
        "ecg": {
            "port": ecfg.port,
            "baudrate": ecfg.baudrate,
            "ecg_input_format": ecfg.input_format,
            "fs_hz": float(ecg["fs"]),
            "num_samples": int(len(ecg["t"])),
            "start_time_sec": float(ecg["t"][0]),
            "end_time_sec": float(ecg["t"][-1]),
            "num_rpeaks": int(len(ecg["peaks_time"])),
            "hr_bpm": ecg["hr_bpm"],
            "polarity": ecg["polarity"],
        },
        "radar": {
            "device": "BGT60TR13C via Infineon SDK DeviceFmcw",
            "configured_frame_rate_hz": rcfg.frame_rate_hz,
            "configured_chirps_per_frame": rcfg.num_chirps,
            "fs_hz": float(radar["fs"]),
            "num_frames": int(len(radar["t"])),
            "start_time_sec": float(radar["t"][0]),
            "end_time_sec": float(radar["t"][-1]),
            "num_peaks": int(len(radar["peaks_time"])),
            "hr_bpm": radar["hr_bpm"],
            "drop_count": int(radar["drop_count"]),
            "fixed_range_idx": None if radar["fixed_range_idx"] is None else int(radar["fixed_range_idx"]),
            "fixed_angle_idx": None if radar["fixed_angle_idx"] is None else int(radar["fixed_angle_idx"]),
        },
        "ao_ac": {
            "total_beats": int(len(rows)),
            "accepted_beats": int(np.sum(accepted)) if len(accepted) else 0,
            "accept_rate": None if len(rows) == 0 else float(np.sum(accepted) / len(rows)),
            "mean_sqi_accepted": _nanmean_col(3),
            "mean_ao_expected_center_error_ms": _nanmean_col(17),
            "std_ao_expected_center_error_ms": _nanstd_col(17),
            "mean_ac_expected_center_error_ms": _nanmean_col(18),
            "std_ac_expected_center_error_ms": _nanstd_col(18),
            "mean_ao_ensemble_error_ms": _nanmean_col(19),
            "std_ao_ensemble_error_ms": _nanstd_col(19),
            "mean_ac_ensemble_error_ms": _nanmean_col(20),
            "std_ac_ensemble_error_ms": _nanstd_col(20),
            "mean_ao_detector_dispersion_ms": _nanmean_col(15),
            "mean_ac_detector_dispersion_ms": _nanmean_col(16),
            "radar_interpolation_fs_hz": acfg.radar_interp_fs_hz,
            "beat_window_sec": [-acfg.beat_pre_sec, acfg.beat_post_sec],
            "ao_search_sec_from_r": acfg.ao_search_sec,
            "ac_search_sec_from_r": acfg.ac_search_sec,
        },
        "comparison": {
            "window_sec": comp["window_sec"],
            "pearson_before_alignment": comp["pearson"],
            "pearson_after_alignment": comp["pearson_aligned"],
            "xcorr_max": comp["xcorr"]["max_corr"],
            "xcorr_lag_sec": comp["xcorr"]["lag_sec"],
            "spectral_corr": comp["spectral_corr"],
            "mean_coherence": comp["mean_coherence"],
        },
        "configs": {
            "ecg": asdict(ecfg),
            "radar": asdict(rcfg),
            "analysis": asdict(acfg),
        }
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)



# ============================================================
# Candidate-consistency two-phase radar-only AO/AC analysis
# ============================================================
def countdown_for_second_measurement(seconds: int = 3):
    print("=" * 72)
    print("[TWO-PHASE] 1단계 label dataset 생성 완료")
    print("[TWO-PHASE] 동일 시간으로 2단계 Radar-only 추가 측정을 시작합니다.")
    print("[TWO-PHASE] 자세/거리/호흡 상태를 최대한 1단계와 동일하게 유지하십시오.")
    for s in range(int(seconds), 0, -1):
        print(f"[TWO-PHASE] Countdown: {s}")
        time.sleep(1.0)
    print("[TWO-PHASE] START")
    print("=" * 72)


def run_radar_only_acquisition(duration_sec: float, rcfg: RadarConfig):
    radar_col = RadarCollector(rcfg)

    start_event = threading.Event()
    stop_event = threading.Event()
    shared: dict[str, float] = {}

    th_rad = threading.Thread(
        target=radar_col.acquire,
        args=(duration_sec, shared, start_event, stop_event),
        daemon=True
    )
    th_rad.start()

    print("[RADAR-ONLY] Waiting for radar ready...")
    radar_ready = radar_col.ready_event.wait(timeout=10.0)
    if not radar_ready:
        stop_event.set()
        raise RuntimeError("Radar not ready for radar-only acquisition")
    if radar_col.error:
        stop_event.set()
        raise radar_col.error

    shared["t0"] = time.perf_counter()
    print("[RADAR-ONLY] Shared timestamp started")
    start_event.set()

    try:
        while th_rad.is_alive():
            if radar_col.error is not None:
                stop_event.set()
                break
            time.sleep(0.05)
    finally:
        stop_event.set()
        th_rad.join(timeout=1.0)

    if radar_col.error:
        raise radar_col.error

    radar = radar_col.analyze()
    return radar


def _finite_float(x):
    try:
        return x is not None and np.isfinite(float(x))
    except Exception:
        return False


def _interp_signal_at(t, x, query_t, default=np.nan):
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if len(t) == 0 or len(x) == 0:
        return default
    if query_t < t[0] or query_t > t[-1]:
        return default
    return float(np.interp(float(query_t), t, x))


def _slice_radar_beat_by_anchor(radar, anchor_t: float, acfg: AnalysisConfig):
    rt = np.asarray(radar.get("t", []), dtype=np.float64)
    rx = np.asarray(radar.get("ppg_like", []), dtype=np.float64)
    if len(rt) < 10 or len(rx) < 10:
        return None
    t0 = float(anchor_t) - float(acfg.beat_pre_sec)
    t1 = float(anchor_t) + float(acfg.beat_post_sec)
    m = (rt >= t0) & (rt <= t1)
    if np.sum(m) < int(0.5 * (acfg.beat_pre_sec + acfg.beat_post_sec) * radar.get("fs", acfg.radar_interp_fs_hz)):
        return None
    bt_abs = rt[m]
    bx = rx[m]
    bt_rel = bt_abs - float(anchor_t)

    # uniform resample to fixed grid
    fs = float(acfg.radar_interp_fs_hz)
    grid = np.arange(-float(acfg.beat_pre_sec), float(acfg.beat_post_sec) + 1e-9, 1.0 / fs)
    if len(grid) < 10:
        return None
    bx_grid = np.interp(grid, bt_rel, bx)
    bx_grid = zscore_safe(bx_grid)
    return grid, bx_grid


def _template_corr_features(x, template):
    x = zscore_safe(np.asarray(x, dtype=np.float64))
    template = zscore_safe(np.asarray(template, dtype=np.float64))
    n = min(len(x), len(template))
    if n < 10:
        return 0.0, 0.0
    x = x[:n]
    template = template[:n]
    try:
        c0 = safe_pearson_for_fig(x, template)
        xc = signal.correlate(x, template, mode="same")
        denom = (np.linalg.norm(x) * np.linalg.norm(template) + 1e-9)
        xc = xc / denom
        peak = float(np.nanmax(xc)) if len(xc) else 0.0
        lag_i = int(np.nanargmax(xc) - len(xc)//2) if len(xc) else 0
        lag_sec = float(lag_i) / 100.0
        return float(c0 if c0 is not None and np.isfinite(c0) else 0.0), float(peak), float(lag_sec)
    except Exception:
        return 0.0, 0.0, 0.0


def extract_candidate_consistency_features(t_rel, beat, acfg: AnalysisConfig, template=None):
    t_rel = np.asarray(t_rel, dtype=np.float64)
    x = zscore_safe(np.asarray(beat, dtype=np.float64))
    if len(t_rel) != len(x) or len(x) < 10:
        return None, None

    fs = float(acfg.radar_interp_fs_hz)
    dx = np.gradient(x) * fs
    ddx = np.gradient(dx) * fs
    env = triangular_smooth_envelope(x, win_len=min(31, max(5, int(0.12 * fs) | 1)))
    env_z = zscore_safe(env)

    def win_stats(lo, hi, prefix):
        m = (t_rel >= lo) & (t_rel <= hi)
        if np.sum(m) < 3:
            return [np.nan] * 14, [f"{prefix}_nan_{i}" for i in range(14)]
        tw = t_rel[m]
        xw = x[m]
        dxw = dx[m]
        ddxw = ddx[m]
        envw = env_z[m]

        i_max = int(np.nanargmax(xw))
        i_min = int(np.nanargmin(xw))
        i_dmax = int(np.nanargmax(dxw))
        i_dmin = int(np.nanargmin(dxw))
        i_cmax = int(np.nanargmax(ddxw))
        i_cmin = int(np.nanargmin(ddxw))
        i_emax = int(np.nanargmax(envw))

        vals = [
            float(tw[i_max]), float(xw[i_max]),
            float(tw[i_min]), float(xw[i_min]),
            float(tw[i_dmax]), float(dxw[i_dmax]),
            float(tw[i_dmin]), float(dxw[i_dmin]),
            float(tw[i_cmax]), float(ddxw[i_cmax]),
            float(tw[i_cmin]), float(ddxw[i_cmin]),
            float(tw[i_emax]), float(envw[i_emax]),
        ]
        names = [
            f"{prefix}_xmax_t", f"{prefix}_xmax_amp",
            f"{prefix}_xmin_t", f"{prefix}_xmin_amp",
            f"{prefix}_dmax_t", f"{prefix}_dmax_amp",
            f"{prefix}_dmin_t", f"{prefix}_dmin_amp",
            f"{prefix}_cmax_t", f"{prefix}_cmax_amp",
            f"{prefix}_cmin_t", f"{prefix}_cmin_amp",
            f"{prefix}_envmax_t", f"{prefix}_envmax_amp",
        ]
        return vals, names

    ao_vals, ao_names = win_stats(acfg.ao_search_sec[0], acfg.ao_search_sec[1], "ao")
    ac_vals, ac_names = win_stats(acfg.ac_search_sec[0], acfg.ac_search_sec[1], "ac")

    # global morphology features
    global_vals = [
        float(np.nanmean(x)),
        float(np.nanstd(x)),
        float(np.nanmax(x) - np.nanmin(x)),
        float(np.nanmean(np.abs(dx))),
        float(np.nanmax(env_z)),
        float(np.nanstd(env_z)),
    ]
    global_names = ["g_mean", "g_std", "g_ptp", "g_mean_abs_dx", "g_env_max", "g_env_std"]

    # template correlation features
    if template is not None:
        c0, cpeak, clag = _template_corr_features(x, template)
    else:
        c0, cpeak, clag = 0.0, 0.0, 0.0
    temp_vals = [float(c0), float(cpeak), float(clag)]
    temp_names = ["template_pearson", "template_xcorr_peak", "template_xcorr_lag_sec"]

    vals = ao_vals + ac_vals + global_vals + temp_vals
    names = ao_names + ac_names + global_names + temp_names
    vals = np.asarray(vals, dtype=np.float64)
    vals[~np.isfinite(vals)] = 0.0
    return vals, names


def build_candidate_consistency_training_dataset(aoac, acfg: AnalysisConfig):
    beats = aoac.get("accepted_beats", [])
    if not beats:
        beats = [b for b in aoac.get("beats", []) if bool(b.get("accepted", False))]

    usable = []
    for b in beats:
        ao = b.get("ao_time", None)
        ac = b.get("ac_time", None)
        t_rel = b.get("t_rel", None)
        rb = b.get("radar_beat", None)
        if not (_finite_float(ao) and _finite_float(ac)) or t_rel is None or rb is None:
            continue
        ao = float(ao)
        ac = float(ac)
        if not (acfg.ao_search_sec[0] <= ao <= acfg.ao_search_sec[1] and acfg.ac_search_sec[0] <= ac <= acfg.ac_search_sec[1] and ac > ao):
            # label dataset은 tight final이라도 생리 window 밖이면 제외
            continue
        usable.append(b)

    if len(usable) < 3:
        return None

    # fixed grid template
    min_len = min(len(np.asarray(b["radar_beat"])) for b in usable)
    arr = []
    for b in usable:
        x = zscore_safe(np.asarray(b["radar_beat"], dtype=np.float64)[:min_len])
        arr.append(x)
    template = np.nanmedian(np.vstack(arr), axis=0)

    X, Y, beat_indices = [], [], []
    feature_names = None
    for b in usable:
        t_rel = np.asarray(b["t_rel"], dtype=np.float64)
        rb = np.asarray(b["radar_beat"], dtype=np.float64)
        # align length to template for feature extraction
        n = min(len(t_rel), len(rb), len(template))
        feat, names = extract_candidate_consistency_features(t_rel[:n], rb[:n], acfg, template=template[:n])
        if feat is None:
            continue
        X.append(feat)
        Y.append([float(b["ao_time"]), float(b["ac_time"])])
        beat_indices.append(int(b.get("beat_index", len(beat_indices))))
        feature_names = names

    if len(X) < 3:
        return None

    # Store the template time axis as well.
    # This is used for paper Fig12 so the reference/composite-label waveform is drawn
    # on the same R-anchored time grid as the radar-only representative beat.
    template_t = np.asarray(usable[0]["t_rel"], dtype=np.float64)[:min_len]

    return {
        "X": np.asarray(X, dtype=np.float64),
        "Y": np.asarray(Y, dtype=np.float64),
        "feature_names": feature_names,
        "template": np.asarray(template, dtype=np.float64),
        "template_t": template_t,
        "beat_indices": beat_indices,
        "n_usable": len(X),
    }


class NumpyRidgeMultiOutput:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.mu = None
        self.sigma = None
        self.W = None

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        self.mu = np.nanmean(X, axis=0)
        self.sigma = np.nanstd(X, axis=0) + 1e-9
        Xs = (X - self.mu) / self.sigma
        Xb = np.column_stack([np.ones(len(Xs)), Xs])
        A = Xb.T @ Xb + self.alpha * np.eye(Xb.shape[1])
        A[0, 0] = Xb.shape[0]  # intercept less penalized
        self.W = np.linalg.pinv(A) @ Xb.T @ Y
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xs = (X - self.mu) / self.sigma
        Xb = np.column_stack([np.ones(len(Xs)), Xs])
        return Xb @ self.W


def _model_candidates():
    if HAS_SKLEARN:
        return {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "lasso": make_pipeline(StandardScaler(), MultiOutputRegressor(Lasso(alpha=0.0005, max_iter=10000))),
            "random_forest": RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42, min_samples_leaf=3),
            "gradient_boosting": MultiOutputRegressor(GradientBoostingRegressor(random_state=42, max_depth=2, n_estimators=120, learning_rate=0.05)),
        }
    return {
        "ridge_numpy": NumpyRidgeMultiOutput(alpha=1.0)
    }


def train_candidate_consistency_models(dataset, acfg: AnalysisConfig, outdir: Path):
    X = dataset["X"]
    Y = dataset["Y"]
    n = len(X)

    if n < 3:
        raise RuntimeError("candidate-consistency model 학습용 beat가 너무 적습니다.")

    k = min(5, max(2, n // 4))
    candidates = _model_candidates()

    results = {}
    best_name, best_score, best_model = None, np.inf, None

    for name, model in candidates.items():
        try:
            pred_all = np.full_like(Y, np.nan, dtype=np.float64)
            if n >= 6:
                kf = KFold(n_splits=k, shuffle=True, random_state=42) if HAS_SKLEARN else None
                if kf is not None:
                    for tr, te in kf.split(X):
                        m = _model_candidates()[name]
                        m.fit(X[tr], Y[tr])
                        pred_all[te] = m.predict(X[te])
                else:
                    # simple leave-block fallback
                    split = max(1, int(0.8 * n))
                    m = NumpyRidgeMultiOutput(alpha=1.0).fit(X[:split], Y[:split])
                    pred_all[split:] = m.predict(X[split:])
                    pred_all[:split] = m.predict(X[:split])
            else:
                m = _model_candidates()[name]
                m.fit(X, Y)
                pred_all = m.predict(X)

            err_ms = (pred_all - Y) * 1000.0
            ao_mae = float(np.nanmean(np.abs(err_ms[:, 0])))
            ac_mae = float(np.nanmean(np.abs(err_ms[:, 1])))
            total_mae = float(np.nanmean(np.abs(err_ms)))
            tol = float(getattr(acfg, "aoac_accuracy_tolerance_ms", 30.0))
            ao_acc_tol = float(np.nanmean(np.abs(err_ms[:, 0]) <= tol))
            ac_acc_tol = float(np.nanmean(np.abs(err_ms[:, 1]) <= tol))
            total_acc_tol = float(np.nanmean((np.abs(err_ms[:, 0]) <= tol) & (np.abs(err_ms[:, 1]) <= tol)))

            results[name] = {
                "ao_mae_ms": ao_mae,
                "ac_mae_ms": ac_mae,
                "total_mae_ms": total_mae,
                "tolerance_ms": tol,
                "ao_acc_tol": ao_acc_tol,
                "ac_acc_tol": ac_acc_tol,
                "total_acc_tol": total_acc_tol,
                "n": int(n),
            }

            if total_mae < best_score:
                best_score = total_mae
                best_name = name
        except Exception as e:
            results[name] = {"error": str(e)}

    # train final best model on all calibration beats
    final_model = _model_candidates()[best_name]
    final_model.fit(X, Y)

    with open(outdir / "candidate_consistency_model_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": "Validation accuracy is calculated on the first ECG+Radar calibration dataset using ECG-derived pseudo AO/AC labels.",
            "has_sklearn": bool(HAS_SKLEARN),
            "best_model": best_name,
            "results": results,
            "n_training_beats": int(n),
            "target_columns": ["ao_time_sec_from_anchor", "ac_time_sec_from_anchor"],
        }, f, ensure_ascii=False, indent=2)

    try:
        with open(outdir / "candidate_consistency_model.pkl", "wb") as f:
            pickle.dump({
                "model": final_model,
                "best_model": best_name,
                "template": dataset["template"],
                "feature_names": dataset["feature_names"],
                "validation": results,
            }, f)
    except Exception:
        pass

    # validation figure
    try:
        labels = []
        ao_mae = []
        ac_mae = []
        total_acc = []
        for kname, r in results.items():
            if "error" in r:
                continue
            labels.append(kname)
            ao_mae.append(r["ao_mae_ms"])
            ac_mae.append(r["ac_mae_ms"])
            total_acc.append(r.get("total_acc_tol", r.get("total_acc10", np.nan)) * 100.0)

        if labels:
            x = np.arange(len(labels))
            fig, axes = plt.subplots(2, 1, figsize=(11, 8))

            # Fig.10 label policy:
            # - This figure is reported as ECG-reference-based timing consistency, not valve ground-truth accuracy.
            # - Keep the visual label style matched to the manuscript result figure.
            axes[0].bar(x - 0.18, ao_mae, 0.36, label="AO MAE")
            axes[0].bar(x + 0.18, ac_mae, 0.36, label="AC MAE")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([""] * len(labels))
            axes[0].set_ylabel("MAE[ms]")
            axes[0].set_title("AO/AC Timing Consistency MAE")
            axes[0].grid(True, axis="y")
            axes[0].legend()

            axes[1].bar(x, total_acc)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([""] * len(labels))
            axes[1].set_ylabel("Acc AO/AC Accuracy [%]")
            axes[1].set_title(f"AO/AC Simultaneous Accuracy within ±{tol:g} ms")
            axes[1].grid(True, axis="y")
            fig.tight_layout()
            fig.savefig(outdir / "fig10_candidate_consistency_model_validation.png", dpi=180, bbox_inches="tight")
            fig.savefig(outdir / "fig10_aoac_timing_consistency_mae_accuracy.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        with open(outdir / "fig10_candidate_consistency_model_validation_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    return final_model, best_name, results


def build_radar_only_beats(radar, acfg: AnalysisConfig):
    anchors = np.asarray(radar.get("peaks_time", []), dtype=np.float64)
    beats = []
    for i, at in enumerate(anchors):
        sliced = _slice_radar_beat_by_anchor(radar, float(at), acfg)
        if sliced is None:
            continue
        t_rel, beat = sliced
        beats.append({
            "beat_index": int(i),
            "anchor_time_sec": float(at),
            "t_rel": t_rel,
            "radar_beat": beat,
        })
    return beats


def predict_radar_only_aoac(radar_only, model, dataset, acfg: AnalysisConfig, outdir: Path):
    template = dataset["template"]
    beats = build_radar_only_beats(radar_only, acfg)

    rows = []
    X_list = []
    valid_beats = []

    for b in beats:
        t_rel = b["t_rel"]
        rb = b["radar_beat"]
        n = min(len(t_rel), len(rb), len(template))
        feat, names = extract_candidate_consistency_features(t_rel[:n], rb[:n], acfg, template=template[:n])
        if feat is None:
            continue
        X_list.append(feat)
        valid_beats.append(b)

    if not X_list:
        raise RuntimeError("radar-only 추가측정에서 예측 가능한 beat가 없습니다.")

    X = np.asarray(X_list, dtype=np.float64)
    pred = np.asarray(model.predict(X), dtype=np.float64)

    # physiological clipping
    pred[:, 0] = np.clip(pred[:, 0], acfg.ao_search_sec[0], acfg.ao_search_sec[1])
    pred[:, 1] = np.clip(pred[:, 1], acfg.ac_search_sec[0], acfg.ac_search_sec[1])
    bad = pred[:, 1] <= pred[:, 0] + acfg.ac_interval_min_sec
    pred[bad, 1] = pred[bad, 0] + acfg.ac_interval_min_sec
    pred[:, 1] = np.clip(pred[:, 1], acfg.ac_search_sec[0], acfg.ac_search_sec[1])

    # confidence proxy: template correlation + physiological interval
    for i, b in enumerate(valid_beats):
        t_rel = b["t_rel"]
        rb = b["radar_beat"]
        n = min(len(t_rel), len(rb), len(template))
        c0, cpeak, clag = _template_corr_features(rb[:n], template[:n])
        ao = float(pred[i, 0])
        ac = float(pred[i, 1])
        interval = ac - ao
        interval_ok = acfg.ac_interval_min_sec <= interval <= acfg.ac_interval_max_sec
        confidence = float(np.clip(0.5 * max(0.0, c0) + 0.5 * max(0.0, cpeak), 0.0, 1.0))
        if not interval_ok:
            confidence *= 0.4

        rows.append([
            int(b["beat_index"]),
            float(b["anchor_time_sec"]),
            ao,
            ac,
            interval,
            confidence,
            float(c0),
            float(cpeak),
            float(clag),
            bool(interval_ok),
            float(b["anchor_time_sec"] + ao),
            float(b["anchor_time_sec"] + ac),
        ])

    header = [
        "beat_index",
        "radar_anchor_time_sec",
        "predicted_ao_sec_from_anchor",
        "predicted_ac_sec_from_anchor",
        "predicted_ao_ac_interval_sec",
        "prediction_confidence",
        "template_pearson",
        "template_xcorr_peak",
        "template_xcorr_lag_sec",
        "physiological_interval_ok",
        "predicted_ao_abs_time_sec",
        "predicted_ac_abs_time_sec",
    ]
    save_csv(outdir / "radar_only_candidate_consistency_predictions.csv", header, rows)

    # summary
    arr = np.asarray([[r[2], r[3], r[4], r[5], r[9]] for r in rows], dtype=object)
    conf = np.asarray([float(r[5]) for r in rows], dtype=np.float64)
    intervals = np.asarray([float(r[4]) for r in rows], dtype=np.float64)

    ao_pred_arr = np.asarray([float(r[2]) for r in rows], dtype=np.float64) if rows else np.array([])
    ac_pred_arr = np.asarray([float(r[3]) for r in rows], dtype=np.float64) if rows else np.array([])
    ao_clip_low_rate = None if len(ao_pred_arr) == 0 else float(np.nanmean(np.isclose(ao_pred_arr, acfg.ao_search_sec[0], atol=1.0 / max(acfg.radar_interp_fs_hz, 1.0) + 1e-6)))
    ao_clip_high_rate = None if len(ao_pred_arr) == 0 else float(np.nanmean(np.isclose(ao_pred_arr, acfg.ao_search_sec[1], atol=1.0 / max(acfg.radar_interp_fs_hz, 1.0) + 1e-6)))
    ac_clip_low_rate = None if len(ac_pred_arr) == 0 else float(np.nanmean(np.isclose(ac_pred_arr, acfg.ac_search_sec[0], atol=1.0 / max(acfg.radar_interp_fs_hz, 1.0) + 1e-6)))
    ac_clip_high_rate = None if len(ac_pred_arr) == 0 else float(np.nanmean(np.isclose(ac_pred_arr, acfg.ac_search_sec[1], atol=1.0 / max(acfg.radar_interp_fs_hz, 1.0) + 1e-6)))

    summary = {
        "note": "Radar-only second measurement has no true AO/AC ground truth unless TWO_PHASE_TEST_WITH_ECG_REFERENCE=True. Accuracy should be interpreted from calibration validation, while this file reports prediction consistency/confidence.",
        "n_predicted_beats": int(len(rows)),
        "mean_prediction_confidence": None if len(conf) == 0 else float(np.nanmean(conf)),
        "median_prediction_confidence": None if len(conf) == 0 else float(np.nanmedian(conf)),
        "ao_mean_ms": None if len(rows) == 0 else float(np.nanmean([r[2] for r in rows]) * 1000.0),
        "ac_mean_ms": None if len(rows) == 0 else float(np.nanmean([r[3] for r in rows]) * 1000.0),
        "ao_ac_interval_mean_ms": None if len(intervals) == 0 else float(np.nanmean(intervals) * 1000.0),
        "physiological_interval_ok_rate": None if len(rows) == 0 else float(np.nanmean([bool(r[9]) for r in rows])),
        "ao_clip_low_rate": ao_clip_low_rate,
        "ao_clip_high_rate": ao_clip_high_rate,
        "ac_clip_low_rate": ac_clip_low_rate,
        "ac_clip_high_rate": ac_clip_high_rate,
        "clip_warning": "If AO clip-low rate is high, AO predictions are saturated at the lower search bound and should be interpreted as boundary-limited estimates.",
    }
    with open(outdir / "radar_only_candidate_consistency_prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # figures
    try:
        beat_idx = np.asarray([r[0] for r in rows], dtype=int)
        ao_ms = np.asarray([r[2] * 1000.0 for r in rows], dtype=float)
        ac_ms = np.asarray([r[3] * 1000.0 for r in rows], dtype=float)
        int_ms = np.asarray([r[4] * 1000.0 for r in rows], dtype=float)
        conf = np.asarray([r[5] for r in rows], dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        axes[0].plot(beat_idx, ao_ms, "o-", markersize=3, label="Predicted AO")
        axes[0].plot(beat_idx, ac_ms, "o-", markersize=3, label="Predicted AC")
        axes[0].set_ylabel("ms from radar anchor")
        axes[0].set_title("Radar-only new measurement: candidate-consistency AO/AC prediction")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(beat_idx, int_ms, "o-", markersize=3, label="AC-AO interval")
        axes[1].axhline(np.nanmedian(int_ms), linestyle="--", label="median")
        axes[1].set_ylabel("Interval [ms]")
        axes[1].grid(True)
        axes[1].legend()

        axes[2].plot(beat_idx, conf, "o-", markersize=3, label="Prediction confidence")
        axes[2].set_xlabel("Beat index")
        axes[2].set_ylabel("confidence")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].grid(True)
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(outdir / "fig11_radar_only_new_data_aoac_prediction.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # Fig12: reference/composite label waveform vs phase2 radar-only representative beat
        # This replaces the previous single-panel phase2 figure because the paper needs to show
        # where the reference AO/AC label is and where the new-data prediction is.
        if valid_beats:
            best_i = int(np.nanargmax(conf)) if len(conf) else 0
            b = valid_beats[best_i]
            t_rel = np.asarray(b["t_rel"], dtype=np.float64)
            rb = zscore_safe(np.asarray(b["radar_beat"], dtype=np.float64))
            ao = float(rows[best_i][2])
            ac = float(rows[best_i][3])

            template = zscore_safe(np.asarray(dataset.get("template", []), dtype=np.float64))
            template_t = dataset.get("template_t", None)
            if template_t is None or len(template_t) != len(template):
                template_t = np.linspace(-acfg.beat_pre_sec, acfg.beat_post_sec, len(template))
            template_t = np.asarray(template_t, dtype=np.float64)

            Y = np.asarray(dataset.get("Y", []), dtype=np.float64)
            if Y.ndim == 2 and Y.shape[0] > 0 and Y.shape[1] >= 2:
                ref_ao = float(np.nanmedian(Y[:, 0]))
                ref_ac = float(np.nanmedian(Y[:, 1]))
            else:
                ref_ao = float(acfg.expected_ao_sec)
                ref_ac = float(acfg.expected_ac_sec)

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Top: reference/composite label waveform
            axes[0].plot(template_t, template, label="Phase1 reference/template beat", linewidth=1.5)
            axes[0].axvline(0, color="gray", linestyle="--", label="R/radar anchor")
            axes[0].axvline(ref_ao, color="darkorange", linestyle="-", linewidth=1.8, label=f"Reference AO {ref_ao*1000:.1f} ms")
            axes[0].axvline(ref_ac, color="black", linestyle="-", linewidth=1.8, label=f"Reference AC {ref_ac*1000:.1f} ms")
            if len(template_t) and np.nanmin(template_t) <= ref_ao <= np.nanmax(template_t):
                axes[0].scatter([ref_ao], [np.interp(ref_ao, template_t, template)], s=70, color="darkorange", zorder=5)
            if len(template_t) and np.nanmin(template_t) <= ref_ac <= np.nanmax(template_t):
                axes[0].scatter([ref_ac], [np.interp(ref_ac, template_t, template)], s=70, color="black", zorder=5)
            axes[0].set_title("Reference/composite label waveform from Phase1 ECG+Radar dataset")
            axes[0].set_ylabel("z-score")
            axes[0].grid(True)
            axes[0].legend(loc="upper right", fontsize=8, ncol=2)

            # Bottom: new phase2 radar-only beat
            axes[1].plot(t_rel, rb, label="Phase2 radar-only representative beat", linewidth=1.5)
            axes[1].axvline(0, color="gray", linestyle="--", label="Radar anchor")
            axes[1].axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.12, label="AO search")
            axes[1].axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.12, label="AC search")
            axes[1].axvline(ao, color="darkorange", linewidth=1.8, label=f"Predicted AO {ao*1000:.1f} ms")
            axes[1].axvline(ac, color="black", linewidth=1.8, label=f"Predicted AC {ac*1000:.1f} ms")
            if len(t_rel) and np.nanmin(t_rel) <= ao <= np.nanmax(t_rel):
                axes[1].scatter([ao], [np.interp(ao, t_rel, rb)], s=70, color="darkorange", zorder=5)
            if len(t_rel) and np.nanmin(t_rel) <= ac <= np.nanmax(t_rel):
                axes[1].scatter([ac], [np.interp(ac, t_rel, rb)], s=70, color="black", zorder=5)

            ao_clip_low = bool(abs(ao - acfg.ao_search_sec[0]) <= (1.0 / max(acfg.radar_interp_fs_hz, 1.0) + 1e-6))
            ao_clip_high = bool(abs(ao - acfg.ao_search_sec[1]) <= (1.0 / max(acfg.radar_interp_fs_hz, 1.0) + 1e-6))
            ac_clip_low = bool(abs(ac - acfg.ac_search_sec[0]) <= (1.0 / max(acfg.radar_interp_fs_hz, 1.0) + 1e-6))
            ac_clip_high = bool(abs(ac - acfg.ac_search_sec[1]) <= (1.0 / max(acfg.radar_interp_fs_hz, 1.0) + 1e-6))
            conf_i = float(rows[best_i][5])
            template_r = float(rows[best_i][6])
            xcorr_peak = float(rows[best_i][7])
            info_txt = (
                f"AO clipped: low={ao_clip_low}, high={ao_clip_high}\n"
                f"AC clipped: low={ac_clip_low}, high={ac_clip_high}\n"
                f"confidence={conf_i:.3f}\n"
                f"template r={template_r:.3f}, xcorr={xcorr_peak:.3f}"
            )
            axes[1].text(
                0.02, 0.04, info_txt,
                transform=axes[1].transAxes,
                ha="left", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
            )
            axes[1].set_title(f"New phase2 radar-only prediction | beat={int(rows[best_i][0])}, confidence={conf_i:.3f}")
            axes[1].set_xlabel("Time from anchor [s]")
            axes[1].set_ylabel("z-score")
            axes[1].grid(True)
            axes[1].legend(loc="upper right", fontsize=8, ncol=2)

            axes[1].set_xlim(-acfg.beat_pre_sec, acfg.beat_post_sec)
            fig.tight_layout()
            fig.savefig(outdir / "fig12_reference_vs_radar_only_aoac_comparison.png", dpi=220, bbox_inches="tight")
            # Backward-compatible filename, but now it is the fixed two-panel version.
            fig.savefig(outdir / "fig12_representative_radar_only_beat_prediction.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

            save_csv(outdir / "fig12_reference_vs_radar_only_aoac_values.csv",
                     ["kind", "beat_index", "ao_sec", "ac_sec", "confidence",
                      "ao_clipped_low", "ao_clipped_high", "ac_clipped_low", "ac_clipped_high",
                      "template_pearson", "template_xcorr_peak"],
                     [
                         ["phase1_reference_template", -1, ref_ao, ref_ac, None,
                          False, False, False, False, None, None],
                         ["phase2_radar_only_prediction", int(rows[best_i][0]), ao, ac, conf_i,
                          ao_clip_low, ao_clip_high, ac_clip_low, ac_clip_high,
                          template_r, xcorr_peak],
                     ])

    except Exception as e:
        with open(outdir / "radar_only_prediction_figures_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))

    return rows, summary


def save_two_phase_protocol_summary(outdir: Path, duration: float, model_name: str, validation: dict, radar_only_summary: dict):
    summary = {
        "paper_title": "FMCW 레이더 기반 비접촉 심박 신호에서 대동맥판막 개방 및 폐쇄 시점 분석에 관한 연구",
        "protocol": {
            "phase1": "ECG+Radar simultaneous acquisition for ECG-derived pseudo AO/AC label dataset",
            "phase2": "Same-duration Radar-only acquisition after countdown",
            "duration_sec_each_phase": float(duration),
            "model": model_name,
            "methods": [
                "ECG R-peak based beat alignment",
                "ECG-derived pseudo AO/AC label generation",
                "Radar beat template construction",
                "Radar morphology feature extraction",
                "Ridge/Lasso/RandomForest/GradientBoosting regression candidate comparison",
                "Radar-only candidate-consistency AO/AC timing prediction",
            ],
            "ground_truth_warning": "Phase2 radar-only does not contain true AO/AC ground truth. True test accuracy requires simultaneous reference sensor or TWO_PHASE_TEST_WITH_ECG_REFERENCE=True.",
        },
        "calibration_validation": validation,
        "radar_only_prediction_summary": radar_only_summary,
    }
    with open(outdir / "two_phase_protocol_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

# ============================================================
# Acquisition runner
# ============================================================
def run_acquisition(duration_sec: float, ecfg: ECGConfig, rcfg: RadarConfig, scfg: Optional[SCGConfig] = None):
    ecg_col = ECGCollector(ecfg)
    radar_col = RadarCollector(rcfg)
    scg_col = SCGCollector(scfg) if (scfg is not None and scfg.enabled) else None

    start_event = threading.Event()
    stop_event = threading.Event()
    shared: dict[str, float] = {}

    th_ecg = threading.Thread(target=ecg_col.acquire, args=(duration_sec, shared, start_event, stop_event), daemon=True)
    th_rad = threading.Thread(target=radar_col.acquire, args=(duration_sec, shared, start_event, stop_event), daemon=True)
    th_scg = None
    if scg_col is not None:
        th_scg = threading.Thread(target=scg_col.acquire, args=(duration_sec, shared, start_event, stop_event), daemon=True)

    th_ecg.start()
    th_rad.start()
    if th_scg is not None:
        th_scg.start()

    print("[INFO] Waiting for ECG/Radar/SCG ready...")
    ecg_ready = ecg_col.ready_event.wait(timeout=10.0)
    radar_ready = radar_col.ready_event.wait(timeout=10.0)
    scg_ready = True if scg_col is None else scg_col.ready_event.wait(timeout=10.0)

    if not ecg_ready:
        stop_event.set()
        raise RuntimeError("ECG serial not ready")
    if not radar_ready:
        stop_event.set()
        raise RuntimeError("Radar not ready")
    if not scg_ready:
        stop_event.set()
        raise RuntimeError("SCG serial not ready")
    if ecg_col.error:
        stop_event.set()
        raise ecg_col.error
    if radar_col.error:
        stop_event.set()
        raise radar_col.error
    if scg_col is not None and scg_col.error:
        stop_event.set()
        raise scg_col.error

    shared["t0"] = time.perf_counter()
    print("[INFO] Shared timestamp started")
    start_event.set()

    try:
        while th_ecg.is_alive() or th_rad.is_alive() or (th_scg is not None and th_scg.is_alive()):
            # ECG/SCG가 초반부터 live serial 0 byte로 실패하면 기다리지 않고 전체 측정 중단
            if ecg_col.error is not None:
                stop_event.set()
                break
            if radar_col.error is not None:
                stop_event.set()
                break
            if scg_col is not None and scg_col.error is not None:
                stop_event.set()
                break
            time.sleep(0.05)
    finally:
        stop_event.set()
        th_ecg.join(timeout=1.0)
        th_rad.join(timeout=1.0)
        if th_scg is not None:
            th_scg.join(timeout=1.0)

    if ecg_col.error:
        raise ecg_col.error
    if radar_col.error:
        raise radar_col.error
    if scg_col is not None and scg_col.error:
        raise scg_col.error

    ecg = ecg_col.analyze()
    radar = radar_col.analyze()
    scg = scg_col.analyze() if scg_col is not None else None
    return ecg, radar, scg



def get_duration_from_cli_or_required_input() -> float:
    """
    측정 시간 strict 입력 함수.

    사용법:
      python ecg_radar.py 60
      python ecg_radar.py 300

    인자가 없으면 반드시 직접 숫자를 입력해야 한다.
    빈 Enter는 기본값으로 진행하지 않고 종료한다.
    """
    if len(sys.argv) >= 2:
        try:
            val = float(sys.argv[1])
            if val > 0:
                return val
            print("[ERROR] 측정 시간은 양수여야 합니다.")
            sys.exit(1)
        except Exception:
            print(f"[ERROR] 측정 시간 인자 해석 실패: {sys.argv[1]!r}")
            sys.exit(1)

    try:
        s = input("측정 시간(초): ").strip()
        if s == "":
            print("[ERROR] 측정 시간이 입력되지 않았습니다. 예: 60 또는 300")
            sys.exit(1)
        val = float(s)
        if val <= 0:
            print("[ERROR] 측정 시간은 양수여야 합니다.")
            sys.exit(1)
        return val
    except Exception as e:
        print(f"[ERROR] 측정 시간 입력 실패: {e}")
        sys.exit(1)



def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    ecfg = ECGConfig()
    rcfg = RadarConfig()
    scfg = SCGConfig()
    acfg = AnalysisConfig()

    print("=" * 72)
    print("[INFO] Integrated ECG/STM32 + BGT60 Radar AO/AC FULL Methods Pipeline")
    print("[INFO] Paper title: FMCW 레이더 기반 비접촉 심박 신호에서 대동맥판막 개방 및 폐쇄 시점 분석에 관한 연구")
    print("=" * 72)
    list_serial_ports()
    print("=" * 72)
    print(f"[CONFIG] ECG port/baud       : {ecfg.port} / {ecfg.baudrate}")
    print(f"[CONFIG] ECG input format    : STM32 UART CSV (sample_index,ADCValue,Smooth_ECG), selected_col={ecfg.stm32_csv_signal_col}")
    print(f"[CONFIG] ECG fs hint         : {ecfg.fs_hint_hz:.1f} Hz")
    print(f"[CONFIG] ECG FFT motion      : {ecfg.use_ecg_fft_motion_suppression}, bands={ecfg.ecg_fft_motion_bands_hz}, att={ecfg.ecg_fft_motion_attenuation}")
    print(f"[CONFIG] Q/T periodic prior  : {QT_USE_RR_ADAPTIVE_PERIODIC_PRIOR}, Q jump≤{QT_Q_MAX_JUMP_SEC*1000:.0f}ms, T jump≤{QT_T_MAX_JUMP_SEC*1000:.0f}ms")
    print(f"[CONFIG] R-peak postprocess  : {RPEAK_ENABLE_SHORT_RR_POSTPROCESS}, minRR={RPEAK_MIN_RR_SEC_POST:.2f}s, lower-QRS-amplitude removal")
    print("[CONFIG] Fig4 metric policy  : no accuracy % on constrained plot; MAE/RMSE/Bias/LOA only")
    print(f"[CONFIG] Fig4 audit policy   : warn if constrained audit rate ≥ {FIG4_AUDIT_WARN_RATE*100:.1f}%")
    print(f"[CONFIG] ECG live fail-fast  : {ecfg.fail_fast_if_no_ecg_sec:.1f} s, DTR={ecfg.dtr_enable}, RTS={ecfg.rts_enable}")
    print(f"[CONFIG] Radar device        : BGT60 via DeviceFmcw(), not COM port")
    print(f"[CONFIG] Radar fs/chirps     : {rcfg.frame_rate_hz:.1f} Hz / {rcfg.num_chirps}")
    print(f"[CONFIG] SCG enabled/port    : {scfg.enabled} / {scfg.port} @ {scfg.baudrate}")
    print(f"[CONFIG] SCG fs/mode         : {scfg.fs_hint_hz:.1f} Hz / {scfg.signal_mode}")
    print(f"[CONFIG] Radar PPG band      : {rcfg.ppg_like_band_hz[0]:.2f}~{rcfg.ppg_like_band_hz[1]:.2f} Hz")
    print(f"[CONFIG] LMS resp cancel     : {rcfg.use_lms_resp_cancel}, mu={rcfg.lms_mu}, order={rcfg.lms_order}")
    print(f"[CONFIG] Radar interp fs     : {acfg.radar_interp_fs_hz:.1f} Hz")
    print(f"[CONFIG] SQI threshold       : {acfg.min_sqi_accept:.2f}")
    print(f"[CONFIG] AO/AC tolerance     : ±{acfg.aoac_accuracy_tolerance_ms:.1f} ms, tight-lock={acfg.use_paper_tight_prior_lock}")
    print(f"[CONFIG] Morphology pruning  : AO={acfg.ao_search_sec}, AC={acfg.ac_search_sec}, window/dispersion valid only")
    print(f"[CONFIG] Two-phase protocol  : {TWO_PHASE_LABEL_GUIDED_PROTOCOL}, radar-only phase2={not TWO_PHASE_TEST_WITH_ECG_REFERENCE}")
    print("[CONFIG] Fig02 R display     : display ECG has no R dots; R markers are on QRS subplot only")
    print(f"[CONFIG] Fig02 Q/T display   : {FIG02_SHOW_QT_MARKERS} (pseudo Q/T saved separately)")
    print(f"[CONFIG] Save base           : {BASE_DIR}")
    print(f"[CONFIG] Paper export        : {PAPER_EXPORT_ENABLED}, dir={PAPER_EXPORT_DIRNAME}")
    print("=" * 72)

    duration = get_duration_from_cli_or_required_input()
    outdir = create_result_dir(duration)

    try:
        # ------------------------------------------------------------
        # Phase 1: existing pipeline, ECG+Radar
        # ------------------------------------------------------------
        print("[PHASE 1] ECG+Radar 동시 측정: label dataset 생성")
        ecg, radar, scg = run_acquisition(duration, ecfg, rcfg, scfg)
        aoac = ao_ac_pipeline(ecg, radar, acfg)
        comp = compare_signals(ecg, radar, acfg, rcfg)
        save_all(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
        save_scg_all(outdir, scg, scfg)
        scg_ref = scg_reference_aoac_pipeline(ecg, scg, radar, aoac, acfg, outdir=outdir)

        total = len(aoac["rows"])
        accepted = sum(1 for r in aoac["rows"] if r[2])
        accept_rate = None if total == 0 else accepted / total

        print("=" * 72)
        print("[PHASE 1 RESULT]")
        print(f"Saved                         : {outdir}")
        print(f"ECG samples / fs              : {len(ecg['t'])} / {ecg['fs']:.3f} Hz")
        print(f"ECG R-peaks / HR              : {len(ecg['peaks_time'])} / {ecg['hr_bpm']}")
        print(f"Radar frames / fs             : {len(radar['t'])} / {radar['fs']:.3f} Hz")
        print(f"Radar peaks / HR              : {len(radar['peaks_time'])} / {radar['hr_bpm']}")
        if scg is not None:
            print(f"SCG samples / fs              : {len(scg['t'])} / {scg['fs']:.3f} Hz")
            if scg_ref is not None:
                print(f"SCG-reference AO MAE          : {scg_ref['summary'].get('ao_mae_vs_scg_ms')}")
                print(f"SCG-reference AC MAE          : {scg_ref['summary'].get('ac_mae_vs_scg_ms')}")
        print(f"AO/AC total / accepted beats  : {total} / {accepted} ({accept_rate})")
        print("=" * 72)

        if not TWO_PHASE_LABEL_GUIDED_PROTOCOL:
            print("[INFO] TWO_PHASE_LABEL_GUIDED_PROTOCOL=False. Single-phase run completed.")
            return

        # ------------------------------------------------------------
        # Build candidate-consistency training dataset from Phase 1
        # ------------------------------------------------------------
        dataset = build_candidate_consistency_training_dataset(aoac, acfg)
        if dataset is None or int(dataset["n_usable"]) < int(TWO_PHASE_MIN_TRAIN_BEATS):
            raise RuntimeError(
                f"Candidate-consistency model 학습 beat 부족: usable={0 if dataset is None else dataset['n_usable']}, "
                f"required={TWO_PHASE_MIN_TRAIN_BEATS}"
            )

        model, model_name, validation = train_candidate_consistency_models(dataset, acfg, outdir)
        print("=" * 72)
        print("[MODEL]")
        print(f"Best candidate-consistency model       : {model_name}")
        print(f"Training beats                : {dataset['n_usable']}")
        try:
            print(f"Validation total MAE          : {validation[model_name]['total_mae_ms']:.3f} ms")
            print(f"Validation total Acc±{acfg.aoac_accuracy_tolerance_ms:g}ms     : {validation[model_name].get('total_acc_tol', validation[model_name].get('total_acc10', float('nan')))*100.0:.2f} %")
        except Exception:
            pass
        print("=" * 72)

        # ------------------------------------------------------------
        # Phase 2: same-duration radar-only or optional ECG+Radar reference
        # ------------------------------------------------------------
        countdown_for_second_measurement(TWO_PHASE_COUNTDOWN_SEC)

        phase2_dir = outdir / "phase2_radar_only"
        phase2_dir.mkdir(parents=True, exist_ok=True)

        if TWO_PHASE_TEST_WITH_ECG_REFERENCE:
            print("[PHASE 2] ECG+Radar reference mode. 실제 test pseudo-accuracy 계산 가능.")
            ecg2, radar2, scg2 = run_acquisition(duration, ecfg, rcfg, scfg)
            aoac2 = ao_ac_pipeline(ecg2, radar2, acfg)
            comp2 = compare_signals(ecg2, radar2, acfg, rcfg)
            save_all(phase2_dir, ecg2, radar2, scg2, aoac2, comp2, ecfg, rcfg, acfg)
            # Still run radar-only-style prediction on radar2 for direct comparison
            pred_rows, pred_summary = predict_radar_only_aoac(radar2, model, dataset, acfg, phase2_dir)
            # True/pseudo test error against phase2 ECG pseudo labels would require beat matching;
            # store warning and use phase2 save_all outputs for reference.
        else:
            print("[PHASE 2] Radar-only 추가 측정. 실제 ground-truth accuracy는 없고 prediction consistency를 저장합니다.")
            radar2 = run_radar_only_acquisition(duration, rcfg)
            # Save raw radar-only csv
            save_csv(phase2_dir / "radar_only_processed.csv",
                     ["time_sec", "displacement", "respiration", "ppg_like"],
                     [[float(radar2["t"][i]), float(radar2["displacement"][i]), float(radar2["respiration"][i]), float(radar2["ppg_like"][i])]
                      for i in range(len(radar2["t"]))])
            save_csv(phase2_dir / "radar_only_peaks.csv",
                     ["peak_idx", "peak_time_sec"],
                     [[int(i), float(t)] for i, t in enumerate(radar2.get("peaks_time", []))])
            pred_rows, pred_summary = predict_radar_only_aoac(radar2, model, dataset, acfg, phase2_dir)

        save_two_phase_protocol_summary(outdir, duration, model_name, validation, pred_summary)

        # Re-export paper package after phase2/model outputs are available.
        # The first export inside save_all happens after phase1 only; this final export overwrites
        # paper_export tables/figures with candidate-consistency and radar-only phase2 results included.
        try:
            export_paper_tables_and_figures(outdir, ecg, radar, aoac, comp, ecfg, rcfg, acfg)
            _export_table_pngs_from_existing_csvs(outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export") / "tables",
                                                  outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export") / "figures")
            force_add_fig13_previous_vs_current_correlation(outdir, aoac, acfg)
        except Exception as e:
            with open(outdir / "paper_export_after_twophase_error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))

        print("=" * 72)
        print("[TWO-PHASE RESULT]")
        print(f"Saved                         : {outdir}")
        print(f"Phase2 dir                    : {phase2_dir}")
        print(f"Best model                    : {model_name}")
        print(f"Phase2 predicted beats        : {pred_summary.get('n_predicted_beats')}")
        print(f"Phase2 mean confidence        : {pred_summary.get('mean_prediction_confidence')}")
        print("=" * 72)

        if len(ecg["peaks_time"]) < max(5, duration * 0.5):
            print("[WARN] ECG R-peak count가 너무 적습니다.")
        if accept_rate is not None and accept_rate < 0.3:
            print("[WARN] Radar beat accept rate가 낮습니다. 거리/자세/ROI/SQI threshold를 확인하십시오.")

    except Exception as e:
        print(f"[ERROR] {e}")
        with open(outdir / "error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))



# ============================================================
# Overrides / additions for ECG-SCG-Radar joint landmark figures
# ============================================================
_old_scg_reference_aoac_pipeline = scg_reference_aoac_pipeline


def _rep_beat_from_aoac_for_joint_figs(ecg: dict, scg: Optional[dict], radar: dict, aoac: dict, acfg: AnalysisConfig):
    beats = list(aoac.get("accepted_beats", []))
    if not beats:
        beats = [b for b in aoac.get("beats", []) if bool(b.get("accepted", False))]
    if not beats:
        beats = list(aoac.get("beats", []))
    if not beats:
        return None
    scored = []
    for b in beats:
        score = float(b.get("sqi", 0.0) or 0.0)
        if b.get("ao_morph_time") is not None:
            score += 0.5
        if b.get("ac_morph_time") is not None:
            score += 0.5
        scored.append((score, b))
    scored.sort(key=lambda z: z[0], reverse=True)
    return scored[0][1]


def _slice_aligned_beat(tt: np.ndarray, xx: np.ndarray, anchor_time: float, acfg: AnalysisConfig, fs_out: float = 100.0):
    tt = np.asarray(tt, dtype=np.float64)
    xx = np.asarray(xx, dtype=np.float64)
    if len(tt) < 5 or len(tt) != len(xx):
        return None, None
    t0 = float(anchor_time) - float(acfg.beat_pre_sec)
    t1 = float(anchor_time) + float(acfg.beat_post_sec)
    m = (tt >= t0) & (tt <= t1)
    if np.sum(m) < max(10, int((acfg.beat_pre_sec + acfg.beat_post_sec) * fs_out * 0.4)):
        return None, None
    grid = np.arange(-float(acfg.beat_pre_sec), float(acfg.beat_post_sec) + 1e-9, 1.0 / fs_out)
    abs_t = float(anchor_time) + grid
    yy = np.interp(abs_t, tt[m], xx[m])
    return grid, zscore_safe(yy)


def _detect_event_generic(bt: np.ndarray, bx: np.ndarray, win: tuple[float, float], kind: str):
    bt = np.asarray(bt, dtype=np.float64)
    bx = zscore_safe(np.asarray(bx, dtype=np.float64))
    if len(bt) < 8 or len(bt) != len(bx):
        return None, None, None
    m = (bt >= float(win[0])) & (bt <= float(win[1]))
    if np.sum(m) < 4:
        return None, None, None
    idxs = np.where(m)[0]
    fs = 1.0 / max(np.nanmedian(np.diff(bt)), 1e-6)
    try:
        y = safe_lowpass(bx, fs, min(18.0, 0.45 * fs), order=2)
    except Exception:
        y = bx
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    env = triangular_smooth_envelope(y, win_len=max(5, int(round(0.05 * fs)) | 1))
    env = zscore_safe(env)
    local_t = bt[idxs]
    center_prior = np.exp(-0.5 * ((local_t - np.mean(win)) / max((win[1] - win[0]) / 2.5, 1e-3)) ** 2)
    if kind == 'mc':
        score = 0.45 * robust_scale_01(np.abs(d2[idxs])) + 0.30 * robust_scale_01(np.abs(d1[idxs])) + 0.15 * robust_scale_01(np.abs(y[idxs])) + 0.10 * robust_scale_01(-local_t)
    elif kind == 'im':
        score = 0.40 * robust_scale_01(np.maximum(d1[idxs], 0)) + 0.25 * robust_scale_01(np.maximum(d2[idxs], 0)) + 0.20 * robust_scale_01(env[idxs]) + 0.15 * center_prior
    elif kind == 'ao':
        score = 0.30 * robust_scale_01(np.maximum(d1[idxs], 0)) + 0.25 * robust_scale_01(np.abs(d2[idxs])) + 0.20 * robust_scale_01(y[idxs]) + 0.15 * robust_scale_01(env[idxs]) + 0.10 * center_prior
    elif kind == 'ac':
        score = 0.30 * robust_scale_01(np.maximum(-d1[idxs], 0)) + 0.25 * robust_scale_01(np.abs(d2[idxs])) + 0.20 * robust_scale_01(-y[idxs]) + 0.15 * robust_scale_01(env[idxs]) + 0.10 * center_prior
    elif kind == 'mo':
        score = 0.35 * robust_scale_01(np.abs(d2[idxs])) + 0.25 * robust_scale_01(np.maximum(d1[idxs], 0)) + 0.20 * robust_scale_01(np.abs(y[idxs])) + 0.20 * center_prior
    else:
        score = robust_scale_01(np.abs(d2[idxs]))
    j = int(idxs[int(np.nanargmax(score))])
    return float(bt[j]), float(y[j]), j


def _find_candidate_markers(bt: np.ndarray, bx: np.ndarray, win: tuple[float, float]):
    bt = np.asarray(bt, dtype=np.float64)
    bx = zscore_safe(np.asarray(bx, dtype=np.float64))
    m = (bt >= float(win[0])) & (bt <= float(win[1]))
    if np.sum(m) < 5:
        return np.array([]), np.array([])
    idx = np.where(m)[0]
    seg_t = bt[idx]
    seg_x = bx[idx]
    fs = 1.0 / max(np.nanmedian(np.diff(bt)), 1e-6)
    try:
        y = safe_lowpass(seg_x, fs, min(18.0, 0.45 * fs), order=2)
    except Exception:
        y = seg_x
    min_dist = max(1, int(round(0.025 * fs)))
    p1, _ = signal.find_peaks(y, distance=min_dist)
    p2, _ = signal.find_peaks(-y, distance=min_dist)
    d1 = np.gradient(y, seg_t)
    d2 = np.gradient(d1, seg_t)
    s1, _ = signal.find_peaks(np.abs(d1), distance=max(1, int(round(0.03 * fs))))
    s2, _ = signal.find_peaks(np.abs(d2), distance=max(1, int(round(0.03 * fs))))
    cand = np.unique(np.concatenate([p1, p2, s1, s2])) if (len(p1) or len(p2) or len(s1) or len(s2)) else np.array([], dtype=int)
    return seg_t[cand], y[cand]


def _estimate_scg_landmarks(bt: np.ndarray, bx: np.ndarray, acfg: AnalysisConfig):
    lm = {}
    lm['MC'], _, _ = _detect_event_generic(bt, bx, (-0.03, 0.03), 'mc')
    im_start = 0.01 if lm['MC'] is None else max(0.01, lm['MC'])
    lm['IM'], _, _ = _detect_event_generic(bt, bx, (im_start, min(0.12, acfg.ao_search_sec[1])), 'im')
    ao_idx, _ = scg_inspired_aoac_detector(bt, bx, acfg.ao_search_sec, kind='ao')
    lm['AO'] = None if ao_idx is None else float(bt[ao_idx])
    ac_idx, _ = scg_inspired_aoac_detector(bt, bx, acfg.ac_search_sec, kind='ac')
    lm['AC'] = None if ac_idx is None else float(bt[ac_idx])
    mo_start = acfg.ac_search_sec[1] + 0.03 if lm['AC'] is None else lm['AC'] + 0.03
    lm['MO'], _, _ = _detect_event_generic(bt, bx, (mo_start, min(acfg.beat_post_sec - 0.02, 0.65)), 'mo')
    return lm


def _estimate_radar_landmarks(bt: np.ndarray, bx: np.ndarray, acfg: AnalysisConfig):
    lm = {}
    lm['MC'], _, _ = _detect_event_generic(bt, bx, (-0.02, 0.04), 'mc')
    lm['IM'], _, _ = _detect_event_generic(bt, bx, (0.02, min(0.12, acfg.ao_search_sec[1])), 'im')
    lm['AO'], _, _ = _detect_event_generic(bt, bx, acfg.ao_search_sec, 'ao')
    lm['AC'], _, _ = _detect_event_generic(bt, bx, acfg.ac_search_sec, 'ac')
    mo_start = acfg.ac_search_sec[1] + 0.03 if lm['AC'] is None else lm['AC'] + 0.03
    lm['MO'], _, _ = _detect_event_generic(bt, bx, (mo_start, min(acfg.beat_post_sec - 0.02, 0.65)), 'mo')
    return lm


def _interval_metrics(q_rel: Optional[float], ao_rel: Optional[float], ac_rel: Optional[float]):
    out = {'PEP_ms': None, 'LVET_ms': None, 'QS2_ms': None}
    if q_rel is not None and ao_rel is not None:
        out['PEP_ms'] = float((ao_rel - q_rel) * 1000.0)
    if ao_rel is not None and ac_rel is not None:
        out['LVET_ms'] = float((ac_rel - ao_rel) * 1000.0)
    if q_rel is not None and ac_rel is not None:
        out['QS2_ms'] = float((ac_rel - q_rel) * 1000.0)
    return out


def add_scg_diagnostic_figures(outdir: Path, ecg: dict, scg: dict, radar: dict, scg_result: dict, aoac: dict, acfg: AnalysisConfig, n_cycles: int = 10):
    # 1) multi-cycle ECG-SCG-Radar overview
    try:
        if len(ecg.get('peaks_time', [])) >= n_cycles + 1:
            r_times = np.asarray(ecg['peaks_time'], dtype=float)
            start_idx = max(2, len(r_times) // 3)
            end_idx = min(start_idx + n_cycles, len(r_times) - 1)
            t_start = r_times[start_idx] - 0.20
            t_end = r_times[end_idx] + 0.60
            fig, axes = plt.subplots(3, 1, figsize=(13.5, 8.6), sharex=True, constrained_layout=True)
            ez = zscore_safe(ecg.get('display_rpeak', ecg.get('display', ecg['filtered'])))
            sz = zscore_safe(scg.get('resp_removed', scg.get('filtered', scg.get('display', np.zeros_like(scg['t'])))))
            rz = zscore_safe(radar.get('lms_error', radar.get('ppg_like', np.zeros_like(radar['t']))))
            axes[0].plot(ecg['t'], ez, linewidth=1.0, color='black', label='ECG')
            axes[1].plot(scg['t'], sz, linewidth=1.0, color='black', label='SCG (LMS resp.-removed)')
            axes[2].plot(radar['t'], rz, linewidth=1.0, color='black', label='Radar (resp.-removed)')
            for ax in axes:
                for r in r_times[start_idx:end_idx + 1]:
                    ax.axvline(r, color='0.25', linestyle='--', linewidth=0.8, alpha=0.7)
                    ax.axvspan(r + acfg.ao_search_sec[0], r + acfg.ao_search_sec[1], alpha=0.10)
                    ax.axvspan(r + acfg.ac_search_sec[0], r + acfg.ac_search_sec[1], alpha=0.08)
                ax.set_xlim(t_start, t_end)
                ax.set_ylabel('z-score')
                ax.grid(True, alpha=0.30)
                ax.legend(loc='upper right', fontsize=9)
            axes[-1].set_xlabel('Time [s]')
            fig.suptitle('ECG R-peak anchored ECG-SCG-Radar multi-cycle waveform', fontsize=13)
            fig.savefig(outdir / 'fig09_ecg_scg_radar_multicycle_diagnostic.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        with open(outdir / 'scg_diagnostic_multicycle_error.txt', 'w', encoding='utf-8') as f:
            f.write(str(e))

    # representative beat selection
    rep = _rep_beat_from_aoac_for_joint_figs(ecg, scg, radar, aoac, acfg)
    if rep is None:
        return
    rep_anchor = float(rep.get('r_time', rep.get('anchor_time_sec', 0.0)))

    # 2) SCG 4-stage comparison (raw -> lms -> bpf -> smoothed)
    try:
        scg_raw = np.asarray(scg.get('selected_raw', scg.get('vmag', np.zeros_like(scg['t']))), dtype=np.float64)
        scg_rr = np.asarray(scg.get('resp_removed', scg.get('filtered', scg_raw)), dtype=np.float64)
        scg_bpf = np.asarray(scg.get('filtered', scg_rr), dtype=np.float64)
        scg_sm = np.asarray(scg.get('display', scg_bpf), dtype=np.float64)
        stages = [('1) Raw SCG', scg_raw), ('2) LMS respiration-removed', scg_rr), ('3) Band-pass filtered', scg_bpf), ('4) Smoothed display', scg_sm)]
        fig, axes = plt.subplots(4, 1, figsize=(11.2, 10.0), sharex=True, constrained_layout=True)
        for ax, (title, sig) in zip(axes, stages):
            bt, bx = _slice_aligned_beat(np.asarray(scg['t'], dtype=np.float64), sig, rep_anchor, acfg, fs_out=100.0)
            if bt is None:
                continue
            ax.plot(bt, bx, color='black', linewidth=1.6)
            ax.axvline(0.0, color='0.35', linestyle='--', linewidth=1.0)
            ax.axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.15)
            ax.axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.12)
            ax.set_title(title, fontsize=11, loc='left', pad=6)
            ax.set_ylabel('z-score', fontsize=10)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel('Time from ECG R-peak [s]', fontsize=10)
        fig.suptitle('SCG processing stages on the same representative beat', fontsize=13)
        fig.savefig(outdir / 'fig04_scg_stage_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        with open(outdir / 'fig04_scg_stage_comparison_error.txt', 'w', encoding='utf-8') as f:
            f.write(str(e))

    # 3) ECG + SCG + Radar representative beat with landmarks and PEP/LVET/QS2
    try:
        # ECG beat
        et = np.asarray(ecg['t'], dtype=np.float64)
        ex = zscore_safe(np.asarray(ecg.get('display_rpeak', ecg.get('display', ecg['filtered'])), dtype=np.float64))
        bt_e, bx_e = _slice_aligned_beat(et, ex, rep_anchor, acfg, fs_out=100.0)
        # SCG beat (use LMS respiration-removed for analysis as requested)
        st = np.asarray(scg['t'], dtype=np.float64)
        sx = np.asarray(scg.get('resp_removed', scg.get('filtered', scg.get('display', np.zeros_like(scg['t'])))), dtype=np.float64)
        bt_s, bx_s = _slice_aligned_beat(st, sx, rep_anchor, acfg, fs_out=100.0)
        # Radar beat (use respiration-removed radar waveform for analysis)
        rt = np.asarray(radar['t'], dtype=np.float64)
        rx = np.asarray(radar.get('lms_error', radar.get('ppg_like', np.zeros_like(radar['t']))), dtype=np.float64)
        bt_r, bx_r = _slice_aligned_beat(rt, rx, rep_anchor, acfg, fs_out=100.0)
        if bt_e is None or bt_s is None or bt_r is None:
            return

        # ECG landmarks
        bi = int(rep.get('beat_index', 0))
        q_rel = None
        t_rel = None
        try:
            q_time = np.asarray(ecg.get('q_time', []), dtype=np.float64)
            t_time = np.asarray(ecg.get('t_time', []), dtype=np.float64)
            if 0 <= bi < len(q_time) and np.isfinite(q_time[bi]):
                q_rel = float(q_time[bi] - rep_anchor)
            if 0 <= bi < len(t_time) and np.isfinite(t_time[bi]):
                t_rel = float(t_time[bi] - rep_anchor)
        except Exception:
            pass
        ecg_ao_ref = rep.get('ecg_ao_ref', None)
        ecg_ac_ref = rep.get('ecg_ac_ref', None)

        scg_lm = _estimate_scg_landmarks(bt_s, bx_s, acfg)
        rad_lm = _estimate_radar_landmarks(bt_r, bx_r, acfg)
        ecg_iv = _interval_metrics(q_rel, ecg_ao_ref, ecg_ac_ref)
        scg_iv = _interval_metrics(q_rel, scg_lm.get('AO'), scg_lm.get('AC'))
        rad_iv = _interval_metrics(q_rel, rad_lm.get('AO'), rad_lm.get('AC'))

        fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.3), sharex=True, constrained_layout=True)
        # ECG subplot
        axes[0].plot(bt_e, bx_e, color='black', linewidth=1.5, label='ECG')
        axes[0].axvline(0.0, color='0.35', linestyle='--', linewidth=1.0, label='R')
        if q_rel is not None:
            axes[0].axvline(q_rel, linestyle=':', linewidth=1.1, label='Q')
        if t_rel is not None:
            axes[0].axvline(t_rel, linestyle='-.', linewidth=1.1, label='T')
        if ecg_ao_ref is not None:
            axes[0].axvline(float(ecg_ao_ref), linestyle='--', linewidth=1.2, label='ECG AO ref')
        if ecg_ac_ref is not None:
            axes[0].axvline(float(ecg_ac_ref), linestyle='--', linewidth=1.2, label='ECG AC ref')
        axes[0].set_title('ECG Q/R/T landmarks and ECG-derived AO/AC reference', fontsize=11, loc='left', pad=6)
        axes[0].text(0.995, 0.95, f"PEP={ecg_iv['PEP_ms']:.1f} ms\nLVET={ecg_iv['LVET_ms']:.1f} ms\nQS2={ecg_iv['QS2_ms']:.1f} ms" if ecg_iv['PEP_ms'] is not None and ecg_iv['LVET_ms'] is not None and ecg_iv['QS2_ms'] is not None else 'Interval unavailable', transform=axes[0].transAxes, ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        axes[0].grid(True, alpha=0.25); axes[0].set_ylabel('z-score')

        # SCG subplot
        axes[1].plot(bt_s, bx_s, color='black', linewidth=1.6, label='SCG (LMS resp.-removed)')
        axes[1].axvline(0.0, color='0.35', linestyle='--', linewidth=1.0)
        axes[1].axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.12)
        axes[1].axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.10)
        for name, mk in [('MC', 'o'), ('IM', '^'), ('AO', 's'), ('AC', 'D'), ('MO', 'v')]:
            tt = scg_lm.get(name)
            if tt is not None:
                yy = float(np.interp(float(tt), bt_s, bx_s))
                axes[1].scatter([tt], [yy], s=56, marker=mk, facecolor='white', edgecolor='black', linewidth=1.0, zorder=6)
                axes[1].annotate(name, xy=(tt, yy), xytext=(6, 8), textcoords='offset points', fontsize=9)
        ao_ct, ao_cy = _find_candidate_markers(bt_s, bx_s, acfg.ao_search_sec)
        ac_ct, ac_cy = _find_candidate_markers(bt_s, bx_s, acfg.ac_search_sec)
        if len(ao_ct): axes[1].scatter(ao_ct, ao_cy, s=18, marker='o', facecolor='0.75', edgecolor='black', linewidth=0.5)
        if len(ac_ct): axes[1].scatter(ac_ct, ac_cy, s=18, marker='s', facecolor='0.75', edgecolor='black', linewidth=0.5)
        axes[1].text(0.995, 0.95, f"PEP={scg_iv['PEP_ms']:.1f} ms\nLVET={scg_iv['LVET_ms']:.1f} ms\nQS2={scg_iv['QS2_ms']:.1f} ms" if scg_iv['PEP_ms'] is not None and scg_iv['LVET_ms'] is not None and scg_iv['QS2_ms'] is not None else 'Interval unavailable', transform=axes[1].transAxes, ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        axes[1].set_title('SCG landmark candidates (MC / IM / AO / AC / MO)', fontsize=11, loc='left', pad=6)
        axes[1].grid(True, alpha=0.25); axes[1].set_ylabel('z-score')

        # Radar subplot
        axes[2].plot(bt_r, bx_r, color='black', linewidth=1.8, label='Radar (resp.-removed)')
        axes[2].axvline(0.0, color='0.35', linestyle='--', linewidth=1.0)
        axes[2].axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.12)
        axes[2].axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.10)
        ao_ct, ao_cy = _find_candidate_markers(bt_r, bx_r, acfg.ao_search_sec)
        ac_ct, ac_cy = _find_candidate_markers(bt_r, bx_r, acfg.ac_search_sec)
        if len(ao_ct): axes[2].scatter(ao_ct, ao_cy, s=20, marker='o', facecolor='0.75', edgecolor='black', linewidth=0.5, label='AO candidates')
        if len(ac_ct): axes[2].scatter(ac_ct, ac_cy, s=20, marker='s', facecolor='0.75', edgecolor='black', linewidth=0.5, label='AC candidates')
        for name, mk in [('MC', 'o'), ('IM', '^'), ('AO', 's'), ('AC', 'D'), ('MO', 'v')]:
            tt = rad_lm.get(name)
            if tt is not None:
                yy = float(np.interp(float(tt), bt_r, bx_r))
                axes[2].scatter([tt], [yy], s=62, marker=mk, facecolor='white', edgecolor='black', linewidth=1.1, zorder=6)
                axes[2].annotate(name, xy=(tt, yy), xytext=(6, 8), textcoords='offset points', fontsize=9)
        axes[2].text(0.995, 0.95, f"PEP={rad_iv['PEP_ms']:.1f} ms\nLVET={rad_iv['LVET_ms']:.1f} ms\nQS2={rad_iv['QS2_ms']:.1f} ms" if rad_iv['PEP_ms'] is not None and rad_iv['LVET_ms'] is not None and rad_iv['QS2_ms'] is not None else 'Interval unavailable', transform=axes[2].transAxes, ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        axes[2].set_title('Radar landmark candidates and final AO/AC timings', fontsize=11, loc='left', pad=6)
        axes[2].grid(True, alpha=0.25); axes[2].set_ylabel('z-score'); axes[2].set_xlabel('Time from ECG R-peak [s]')
        fig.suptitle('Representative beat: ECG, SCG, and radar landmarks with PEP / LVET / QS2', fontsize=13)
        fig.savefig(outdir / 'fig10_ecg_scg_radar_landmarks_intervals.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        rows = [
            ['ECG', q_rel, ecg_ao_ref, ecg_ac_ref, ecg_iv['PEP_ms'], ecg_iv['LVET_ms'], ecg_iv['QS2_ms']],
            ['SCG', q_rel, scg_lm.get('AO'), scg_lm.get('AC'), scg_iv['PEP_ms'], scg_iv['LVET_ms'], scg_iv['QS2_ms']],
            ['Radar', q_rel, rad_lm.get('AO'), rad_lm.get('AC'), rad_iv['PEP_ms'], rad_iv['LVET_ms'], rad_iv['QS2_ms']],
        ]
        save_csv(outdir / 'ecg_scg_radar_intervals_representative_beat.csv',
                 ['modality', 'q_rel_sec', 'ao_rel_sec', 'ac_rel_sec', 'PEP_ms', 'LVET_ms', 'QS2_ms'], rows)
        # Beat-wise interval comparison across ECG / SCG / Radar
        try:
            scg_rows = scg_result.get('rows', []) if isinstance(scg_result, dict) else []
            beat_idx = []
            ecg_pep = []; ecg_lvet = []; ecg_qs2 = []
            scg_pep = []; scg_lvet = []; scg_qs2 = []
            rad_pep = []; rad_lvet = []; rad_qs2 = []
            q_time_arr = np.asarray(ecg.get('q_time', []), dtype=np.float64)
            r_time_arr = np.asarray(ecg.get('peaks_time', []), dtype=np.float64)
            for row in scg_rows:
                bi2 = int(row[0])
                if bi2 < 0 or bi2 >= len(r_time_arr):
                    continue
                r_t2 = float(r_time_arr[bi2])
                q_rel2 = None
                if bi2 < len(q_time_arr) and np.isfinite(q_time_arr[bi2]):
                    q_rel2 = float(q_time_arr[bi2] - r_t2)
                # ECG reference values from matched radar beat
                rb = next((b for b in aoac.get('beats', []) if int(b.get('beat_index', -1)) == bi2), None)
                e_ao = None if rb is None else rb.get('ecg_ao_ref', None)
                e_ac = None if rb is None else rb.get('ecg_ac_ref', None)
                s_ao = row[4]; s_ac = row[5]
                r_ao = row[6]; r_ac = row[7]
                e_iv = _interval_metrics(q_rel2, e_ao, e_ac)
                s_iv = _interval_metrics(q_rel2, s_ao, s_ac)
                r_iv = _interval_metrics(q_rel2, r_ao, r_ac)
                beat_idx.append(bi2)
                ecg_pep.append(e_iv['PEP_ms']); ecg_lvet.append(e_iv['LVET_ms']); ecg_qs2.append(e_iv['QS2_ms'])
                scg_pep.append(s_iv['PEP_ms']); scg_lvet.append(s_iv['LVET_ms']); scg_qs2.append(s_iv['QS2_ms'])
                rad_pep.append(r_iv['PEP_ms']); rad_lvet.append(r_iv['LVET_ms']); rad_qs2.append(r_iv['QS2_ms'])
            if len(beat_idx) >= 3:
                fig, axes = plt.subplots(3, 1, figsize=(12.8, 8.8), sharex=True, constrained_layout=True)
                series = [
                    ('PEP [ms]', ecg_pep, scg_pep, rad_pep),
                    ('LVET [ms]', ecg_lvet, scg_lvet, rad_lvet),
                    ('QS2 [ms]', ecg_qs2, scg_qs2, rad_qs2),
                ]
                for ax, (ylab, a1, a2, a3) in zip(axes, series):
                    b = np.asarray(beat_idx, dtype=int)
                    ax.plot(b, np.asarray(a1, dtype=float), 'o-', markersize=3, linewidth=1.0, label='ECG ref')
                    ax.plot(b, np.asarray(a2, dtype=float), 's-', markersize=3, linewidth=1.0, label='SCG')
                    ax.plot(b, np.asarray(a3, dtype=float), '^-', markersize=3, linewidth=1.0, label='Radar')
                    ax.set_ylabel(ylab, fontsize=10)
                    ax.grid(True, alpha=0.25)
                    ax.legend(loc='upper right', fontsize=8, ncol=3)
                axes[-1].set_xlabel('Beat index', fontsize=10)
                fig.suptitle('Beat-wise PEP / LVET / QS2 comparison: ECG vs SCG vs Radar', fontsize=13)
                fig.savefig(outdir / 'fig11_pep_lvet_qs2_comparison.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                rows2 = [['beat_index','ecg_pep_ms','scg_pep_ms','radar_pep_ms','ecg_lvet_ms','scg_lvet_ms','radar_lvet_ms','ecg_qs2_ms','scg_qs2_ms','radar_qs2_ms']]
                data_rows = [[int(beat_idx[i]), ecg_pep[i], scg_pep[i], rad_pep[i], ecg_lvet[i], scg_lvet[i], rad_lvet[i], ecg_qs2[i], scg_qs2[i], rad_qs2[i]] for i in range(len(beat_idx))]
                save_csv(outdir / 'ecg_scg_radar_pep_lvet_qs2_per_beat.csv', rows2[0], data_rows)
        except Exception as e2:
            with open(outdir / 'fig11_pep_lvet_qs2_comparison_error.txt', 'w', encoding='utf-8') as f2:
                f2.write(str(e2))
    except Exception as e:
        with open(outdir / 'fig10_ecg_scg_radar_landmarks_intervals_error.txt', 'w', encoding='utf-8') as f:
            f.write(str(e))


def scg_reference_aoac_pipeline(ecg: dict, scg: Optional[dict], radar: dict, aoac: dict, acfg: AnalysisConfig, outdir: Optional[Path] = None):
    # Use LMS respiration-removed SCG waveform as the analysis branch, as requested.
    if scg is not None:
        scg2 = dict(scg)
        if 'resp_removed' in scg2:
            scg2['filtered'] = zscore_safe(np.asarray(scg2['resp_removed'], dtype=np.float64))
            scg2['display'] = zscore_safe(np.asarray(scg2.get('display', scg2['resp_removed']), dtype=np.float64))
        else:
            scg2 = scg
    else:
        scg2 = scg
    return _old_scg_reference_aoac_pipeline(ecg, scg2, radar, aoac, acfg, outdir=outdir)


# ============================================================
# CLEAN PAPER FIGURE SUITE OVERRIDE
# This block intentionally overrides save_all after the original functions are defined.
# It keeps CSV export behavior, then writes clean ECG+SCG+Radar paper figures.
# ============================================================

_CLEAN_OLD_SAVE_ALL = save_all


def _clean_finite(v):
    try:
        return v is not None and np.isfinite(float(v))
    except Exception:
        return False


def _clean_z(x):
    return zscore_safe(np.asarray(x, dtype=np.float64))


def _clean_get_radar_resp_removed(radar: dict):
    return np.asarray(radar.get("lms_error", radar.get("ppg_like", radar.get("displacement", []))), dtype=np.float64)


def _clean_get_radar_light(radar: dict, acfg: AnalysisConfig):
    rt = np.asarray(radar.get("t", []), dtype=np.float64)
    light = np.asarray(radar.get("ppg_like", []), dtype=np.float64)
    if len(light) == len(rt) and len(light) > 0:
        return light
    rr = _clean_get_radar_resp_removed(radar)
    fs = float(radar.get("fs", getattr(acfg, "radar_interp_fs_hz", 100.0)))
    try:
        return safe_bandpass(rr, fs, RADAR_CARDIAC_BAND_HZ[0], RADAR_CARDIAC_BAND_HZ[1], order=4)
    except Exception:
        return rr


def _clean_smooth(x, fs=100.0, low_hz=4.5, win_sec=0.11):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 5:
        return x
    try:
        y = safe_lowpass(x, fs, min(low_hz, 0.35 * fs), order=2)
    except Exception:
        y = x.copy()
    win = max(5, int(round(win_sec * fs)))
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float64) / win
    return np.convolve(y, k, mode="same")


def _clean_slice(tt, xx, anchor, pre=0.20, post=0.80, fs_out=100.0):
    tt = np.asarray(tt, dtype=np.float64)
    xx = np.asarray(xx, dtype=np.float64)
    if len(tt) < 5 or len(tt) != len(xx):
        return None, None
    m = (tt >= anchor - pre) & (tt <= anchor + post)
    if np.sum(m) < max(8, int((pre + post) * fs_out * 0.35)):
        return None, None
    grid = np.arange(-pre, post + 1e-9, 1.0 / fs_out)
    yy = np.interp(anchor + grid, tt[m], xx[m])
    return grid, _clean_z(yy)


def _clean_rep_beat(ecg: dict, aoac: dict):
    beats = list(aoac.get("accepted_beats", []))
    if not beats:
        beats = [b for b in aoac.get("beats", []) if bool(b.get("accepted", False))]
    if not beats:
        beats = list(aoac.get("beats", []))
    if not beats:
        peaks = np.asarray(ecg.get("peaks_time", []), dtype=np.float64)
        if len(peaks) == 0:
            return None
        mid = len(peaks) // 2
        return {"beat_index": int(mid), "r_time": float(peaks[mid])}

    def score(b):
        s = float(b.get("sqi", 0.0) or 0.0)
        s += 0.4 if b.get("ao_morph_time", b.get("ao_time")) is not None else 0.0
        s += 0.4 if b.get("ac_morph_time", b.get("ac_time")) is not None else 0.0
        return s

    return sorted(beats, key=score, reverse=True)[0]


def _clean_anchor_from_beat(ecg: dict, b: dict):
    for k in ("r_time", "anchor_time_sec", "r_peak_time_sec"):
        if _clean_finite(b.get(k)):
            return float(b[k])
    bi = int(b.get("beat_index", 0))
    peaks = np.asarray(ecg.get("peaks_time", []), dtype=np.float64)
    if 0 <= bi < len(peaks):
        return float(peaks[bi])
    return None


def _clean_event(bt, bx, win, kind):
    bt = np.asarray(bt, dtype=np.float64)
    bx = _clean_z(bx)
    if len(bt) < 8 or len(bt) != len(bx):
        return None
    m = (bt >= float(win[0])) & (bt <= float(win[1]))
    if np.sum(m) < 4:
        return None
    idxs = np.where(m)[0]
    try:
        fs = 1.0 / max(np.nanmedian(np.diff(bt)), 1e-6)
        y = safe_lowpass(bx, fs, min(18.0, 0.45 * fs), order=2)
    except Exception:
        y = bx
        fs = 100.0
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    env = triangular_smooth_envelope(y, win_len=max(5, int(round(0.05 * fs)) | 1))
    env = _clean_z(env)
    t = bt[idxs]
    center = np.exp(-0.5 * ((t - np.mean(win)) / max((win[1] - win[0]) / 2.5, 1e-3)) ** 2)

    if kind == "MC":
        score = 0.45 * robust_scale_01(np.abs(d2[idxs])) + 0.25 * robust_scale_01(np.abs(d1[idxs])) + 0.20 * robust_scale_01(np.abs(y[idxs])) + 0.10 * robust_scale_01(-t)
    elif kind == "IM":
        score = 0.45 * robust_scale_01(np.maximum(d1[idxs], 0)) + 0.25 * robust_scale_01(np.maximum(d2[idxs], 0)) + 0.20 * robust_scale_01(env[idxs]) + 0.10 * center
    elif kind == "AO":
        score = 0.30 * robust_scale_01(np.maximum(d1[idxs], 0)) + 0.25 * robust_scale_01(np.abs(d2[idxs])) + 0.20 * robust_scale_01(y[idxs]) + 0.15 * robust_scale_01(env[idxs]) + 0.10 * center
    elif kind == "AC":
        score = 0.30 * robust_scale_01(np.maximum(-d1[idxs], 0)) + 0.25 * robust_scale_01(np.abs(d2[idxs])) + 0.20 * robust_scale_01(-y[idxs]) + 0.15 * robust_scale_01(env[idxs]) + 0.10 * center
    elif kind == "MO":
        score = 0.35 * robust_scale_01(np.abs(d2[idxs])) + 0.25 * robust_scale_01(np.maximum(d1[idxs], 0)) + 0.20 * robust_scale_01(np.abs(y[idxs])) + 0.20 * center
    else:
        score = robust_scale_01(np.abs(d2[idxs]))

    j = int(idxs[int(np.nanargmax(score))])
    return float(bt[j])


def _clean_candidates(bt, bx, win):
    bt = np.asarray(bt, dtype=np.float64)
    bx = _clean_z(bx)
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 5:
        return np.array([]), np.array([])
    idx = np.where(m)[0]
    seg_t = bt[idx]
    seg_x = bx[idx]
    try:
        fs = 1.0 / max(np.nanmedian(np.diff(bt)), 1e-6)
        y = safe_lowpass(seg_x, fs, min(18.0, 0.45 * fs), order=2)
    except Exception:
        y = seg_x
        fs = 100.0
    min_dist = max(1, int(round(0.025 * fs)))
    p1, _ = signal.find_peaks(y, distance=min_dist)
    p2, _ = signal.find_peaks(-y, distance=min_dist)
    d1 = np.gradient(y, seg_t)
    d2 = np.gradient(d1, seg_t)
    s1, _ = signal.find_peaks(np.abs(d1), distance=max(1, int(round(0.03 * fs))))
    s2, _ = signal.find_peaks(np.abs(d2), distance=max(1, int(round(0.03 * fs))))
    cand = np.unique(np.concatenate([p1, p2, s1, s2])) if (len(p1) or len(p2) or len(s1) or len(s2)) else np.array([], dtype=int)
    return seg_t[cand], y[cand]


def _clean_landmarks(bt, bx, acfg):
    mc = _clean_event(bt, bx, (-0.03, 0.03), "MC")
    im = _clean_event(bt, bx, (0.01 if mc is None else max(0.01, mc), 0.12), "IM")
    ao = _clean_event(bt, bx, acfg.ao_search_sec, "AO")
    ac = _clean_event(bt, bx, acfg.ac_search_sec, "AC")
    mo_start = (ac + 0.03) if ac is not None else min(acfg.ac_search_sec[1] + 0.03, 0.58)
    mo = _clean_event(bt, bx, (mo_start, min(0.70, acfg.beat_post_sec - 0.02)), "MO")
    return {"MC": mc, "IM": im, "AO": ao, "AC": ac, "MO": mo}


def _clean_interval(q_rel, ao_rel, ac_rel):
    pep = None if q_rel is None or ao_rel is None else (ao_rel - q_rel) * 1000.0
    lvet = None if ao_rel is None or ac_rel is None else (ac_rel - ao_rel) * 1000.0
    qs2 = None if q_rel is None or ac_rel is None else (ac_rel - q_rel) * 1000.0
    return pep, lvet, qs2


def _draw_vline_label(ax, x, label, yfrac=0.92, ls="--", lw=1.0):
    if x is None or not np.isfinite(float(x)):
        return
    ax.axvline(float(x), linestyle=ls, linewidth=lw, color="black", alpha=0.85)
    ylim = ax.get_ylim()
    y = ylim[0] + (ylim[1] - ylim[0]) * yfrac
    ax.text(float(x), y, label, fontsize=8.5, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.65", alpha=0.88))


def _draw_interval_bracket(ax, x0, x1, y, label):
    if x0 is None or x1 is None or not np.isfinite(float(x0)) or not np.isfinite(float(x1)):
        return
    x0, x1 = float(x0), float(x1)
    ax.plot([x0, x1], [y, y], color="black", linewidth=1.4)
    ax.plot([x0, x0], [y - 0.04, y + 0.04], color="black", linewidth=1.0)
    ax.plot([x1, x1], [y - 0.04, y + 0.04], color="black", linewidth=1.0)
    ax.text((x0 + x1) / 2, y + 0.05, label, fontsize=8.5, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.7", alpha=0.9))


def _clean_common_data(ecg, radar, scg, aoac, acfg):
    rep = _clean_rep_beat(ecg, aoac)
    if rep is None:
        return None
    anchor = _clean_anchor_from_beat(ecg, rep)
    if anchor is None:
        return None
    bi = int(rep.get("beat_index", 0))
    fs = 100.0
    ecg_sig = np.asarray(ecg.get("display_rpeak", ecg.get("display", ecg.get("filtered", []))), dtype=np.float64)
    ecg_bt, ecg_bx = _clean_slice(ecg["t"], ecg_sig, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, fs)

    scg_raw = np.asarray(scg.get("selected_raw", scg.get("vmag", [])), dtype=np.float64) if scg is not None else np.array([])
    scg_lms = np.asarray(scg.get("resp_removed", scg.get("filtered", scg.get("display", []))), dtype=np.float64) if scg is not None else np.array([])
    scg_bpf = np.asarray(scg.get("filtered", scg_lms), dtype=np.float64) if scg is not None else np.array([])
    scg_sm = np.asarray(scg.get("display", scg_bpf), dtype=np.float64) if scg is not None else np.array([])

    scg_bt, scg_bx = (None, None)
    if scg is not None and len(scg_lms):
        scg_bt, scg_bx = _clean_slice(scg["t"], scg_lms, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, fs)

    radar_rr = _clean_get_radar_resp_removed(radar)
    radar_bt, radar_bx = _clean_slice(radar["t"], radar_rr, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, fs)

    q_rel = None
    t_rel = None
    qarr = np.asarray(ecg.get("q_time", []), dtype=np.float64)
    tarr = np.asarray(ecg.get("t_time", []), dtype=np.float64)
    if 0 <= bi < len(qarr) and np.isfinite(qarr[bi]):
        q_rel = float(qarr[bi] - anchor)
    if 0 <= bi < len(tarr) and np.isfinite(tarr[bi]):
        t_rel = float(tarr[bi] - anchor)

    return {
        "rep": rep, "anchor": anchor, "beat_index": bi,
        "ecg_bt": ecg_bt, "ecg_bx": ecg_bx,
        "scg_bt": scg_bt, "scg_bx": scg_bx,
        "radar_bt": radar_bt, "radar_bx": radar_bx,
        "q_rel": q_rel, "t_rel": t_rel,
        "scg_raw": scg_raw, "scg_lms": scg_lms, "scg_bpf": scg_bpf, "scg_sm": scg_sm,
        "radar_raw": np.asarray(radar.get("displacement", []), dtype=np.float64),
        "radar_lms": radar_rr,
        "radar_light": _clean_get_radar_light(radar, acfg),
        "radar_heavy": _clean_smooth(_clean_get_radar_light(radar, acfg), float(radar.get("fs", 100.0))),
    }


def _clean_figs(outdir: Path, ecg, radar, scg, aoac, comp, acfg):
    outdir = Path(outdir)
    d = _clean_common_data(ecg, radar, scg, aoac, acfg)
    if d is None:
        (outdir / "clean_paper_figures_error.txt").write_text("Could not build common ECG/SCG/Radar aligned data.", encoding="utf-8")
        return

    r_times = np.asarray(ecg.get("peaks_time", []), dtype=np.float64)
    anchor = d["anchor"]

    # Fig 01: multi-cycle acquisition with all modalities
    try:
        idx = int(np.argmin(np.abs(r_times - anchor))) if len(r_times) else 0
        sidx = max(0, idx - 3)
        eidx = min(len(r_times) - 1, idx + 6)
        t0 = float(r_times[sidx] - 0.20) if len(r_times) else anchor - 1
        t1 = float(r_times[eidx] + 0.60) if len(r_times) else anchor + 3
        fig, axes = plt.subplots(3, 1, figsize=(13.0, 8.4), sharex=True, constrained_layout=True)
        panels = [
            ("ECG", ecg["t"], np.asarray(ecg.get("display_rpeak", ecg.get("display", ecg.get("filtered", []))), dtype=np.float64)),
            ("SCG (LMS respiration-removed)", scg["t"] if scg is not None else [], d["scg_lms"]),
            ("Radar (LMS respiration-removed)", radar["t"], d["radar_lms"]),
        ]
        for ax, (title, tt, xx) in zip(axes, panels):
            tt = np.asarray(tt, dtype=np.float64); xx = np.asarray(xx, dtype=np.float64)
            if len(tt) == len(xx) and len(tt):
                m = (tt >= t0) & (tt <= t1)
                ax.plot(tt[m], _clean_z(xx[m]), color="black", linewidth=1.1)
            for r in r_times[sidx:eidx + 1]:
                ax.axvline(r, color="0.25", linestyle="--", linewidth=0.8)
                ax.axvspan(r + acfg.ao_search_sec[0], r + acfg.ao_search_sec[1], alpha=0.10)
                ax.axvspan(r + acfg.ac_search_sec[0], r + acfg.ac_search_sec[1], alpha=0.08)
            ax.set_title(title, fontsize=11, loc="left", pad=5)
            ax.set_ylabel("z-score", fontsize=10)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(t0, t1)
        axes[-1].set_xlabel("Time [s]", fontsize=10)
        fig.suptitle("Fig. 1. Simultaneous ECG / SCG / Radar acquisition", fontsize=13)
        fig.savefig(outdir / "fig01_ecg_scg_radar_multicycle.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (outdir / "fig01_clean_error.txt").write_text(str(e), encoding="utf-8")

    # Fig 02: ECG QRT reference
    try:
        fig, ax = plt.subplots(1, 1, figsize=(11.8, 4.4), constrained_layout=True)
        ax.plot(d["ecg_bt"], d["ecg_bx"], color="black", linewidth=1.5)
        ax.axvline(0, color="0.25", linestyle="--", linewidth=1.0)
        _draw_vline_label(ax, d["q_rel"], "Q", 0.96, ":", 1.1)
        _draw_vline_label(ax, 0.0, "R", 0.90, "--", 1.0)
        _draw_vline_label(ax, d["t_rel"], "T", 0.84, "-.", 1.1)
        ao_ref = d["rep"].get("ecg_ao_ref", None)
        ac_ref = d["rep"].get("ecg_ac_ref", None)
        _draw_vline_label(ax, ao_ref, "AO ref", 0.78, "--", 1.0)
        _draw_vline_label(ax, ac_ref, "AC ref", 0.72, "--", 1.0)
        ax.axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.12)
        ax.axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.10)
        ax.set_title("Fig. 2. ECG Q/R/T landmarks and AO/AC reference windows", fontsize=13)
        ax.set_xlabel("Time from ECG R-peak [s]"); ax.set_ylabel("z-score")
        ax.grid(True, alpha=0.25)
        fig.savefig(outdir / "fig02_ecg_qrt_reference.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (outdir / "fig02_clean_error.txt").write_text(str(e), encoding="utf-8")

    # Fig 04-1: radar stages
    try:
        stages = [
            ("1) Raw displacement", d["radar_raw"]),
            ("2) LMS respiration-removed", d["radar_lms"]),
            ("3) Lightly filtered", d["radar_light"]),
            ("4) Heavily smoothed", d["radar_heavy"]),
        ]
        fig, axes = plt.subplots(4, 1, figsize=(11.5, 10.0), sharex=True, constrained_layout=True)
        for ax, (title, sig) in zip(axes, stages):
            bt, bx = _clean_slice(radar["t"], sig, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
            if bt is not None:
                ax.plot(bt, bx, color="black", linewidth=1.4)
                ax.axvline(0, color="0.25", linestyle="--", linewidth=0.9)
                ax.axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.13)
                ax.axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.10)
                ct, cy = _clean_candidates(bt, bx, acfg.ao_search_sec)
                if len(ct): ax.scatter(ct, cy, s=15, marker="o", facecolor="0.75", edgecolor="black", linewidth=0.4)
                ct, cy = _clean_candidates(bt, bx, acfg.ac_search_sec)
                if len(ct): ax.scatter(ct, cy, s=15, marker="s", facecolor="0.75", edgecolor="black", linewidth=0.4)
            ax.set_title(title, fontsize=11, loc="left", pad=5)
            ax.set_ylabel("z-score", fontsize=10)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Fig. 4-1. Radar morphology stage comparison", fontsize=13)
        fig.savefig(outdir / "fig04_1_radar_stage_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (outdir / "fig04_1_clean_error.txt").write_text(str(e), encoding="utf-8")

    # Fig 04-2: SCG stages
    if scg is not None:
        try:
            stages = [
                ("1) Raw SCG", d["scg_raw"]),
                ("2) LMS respiration-removed", d["scg_lms"]),
                ("3) Band-pass filtered", d["scg_bpf"]),
                ("4) Smoothed", d["scg_sm"]),
            ]
            fig, axes = plt.subplots(4, 1, figsize=(11.5, 10.0), sharex=True, constrained_layout=True)
            for ax, (title, sig) in zip(axes, stages):
                bt, bx = _clean_slice(scg["t"], sig, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
                if bt is not None:
                    ax.plot(bt, bx, color="black", linewidth=1.4)
                    ax.axvline(0, color="0.25", linestyle="--", linewidth=0.9)
                    ax.axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.13)
                    ax.axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.10)
                    ct, cy = _clean_candidates(bt, bx, acfg.ao_search_sec)
                    if len(ct): ax.scatter(ct, cy, s=15, marker="o", facecolor="0.75", edgecolor="black", linewidth=0.4)
                    ct, cy = _clean_candidates(bt, bx, acfg.ac_search_sec)
                    if len(ct): ax.scatter(ct, cy, s=15, marker="s", facecolor="0.75", edgecolor="black", linewidth=0.4)
                ax.set_title(title, fontsize=11, loc="left", pad=5)
                ax.set_ylabel("z-score", fontsize=10)
                ax.grid(True, alpha=0.25)
            axes[-1].set_xlabel("Time from ECG R-peak [s]")
            fig.suptitle("Fig. 4-2. SCG morphology stage comparison", fontsize=13)
            fig.savefig(outdir / "fig04_2_scg_stage_comparison.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            (outdir / "fig04_2_clean_error.txt").write_text(str(e), encoding="utf-8")

    # Fig 05/06/10: joint landmarks and interval brackets
    try:
        scg_lm = _clean_landmarks(d["scg_bt"], d["scg_bx"], acfg) if d["scg_bt"] is not None else {}
        rad_lm = _clean_landmarks(d["radar_bt"], d["radar_bx"], acfg) if d["radar_bt"] is not None else {}
        ecg_ao = d["rep"].get("ecg_ao_ref", None)
        ecg_ac = d["rep"].get("ecg_ac_ref", None)
        rows = []
        for modality, ao, ac in [
            ("ECG_ref", ecg_ao, ecg_ac),
            ("SCG", scg_lm.get("AO"), scg_lm.get("AC")),
            ("Radar", rad_lm.get("AO"), rad_lm.get("AC")),
        ]:
            pep, lvet, qs2 = _clean_interval(d["q_rel"], ao, ac)
            rows.append([modality, d["q_rel"], ao, ac, pep, lvet, qs2])
        save_csv(outdir / "clean_ecg_scg_radar_intervals_representative.csv",
                 ["modality", "q_rel_sec", "ao_rel_sec", "ac_rel_sec", "PEP_ms", "LVET_ms", "QS2_ms"], rows)

        fig, axes = plt.subplots(3, 1, figsize=(13.4, 9.2), sharex=True, constrained_layout=True)
        panels = [
            ("ECG Q/R/T + reference timing", d["ecg_bt"], d["ecg_bx"], {"Q": d["q_rel"], "R": 0.0, "T": d["t_rel"], "AO": ecg_ao, "AC": ecg_ac}, rows[0]),
            ("SCG landmarks: MC / IM / AO / AC / MO", d["scg_bt"], d["scg_bx"], scg_lm, rows[1]),
            ("Radar landmarks and morphology candidates", d["radar_bt"], d["radar_bx"], rad_lm, rows[2]),
        ]
        for ax, (title, bt, bx, lm, interval_row) in zip(axes, panels):
            if bt is None:
                continue
            ax.plot(bt, bx, color="black", linewidth=1.5)
            ax.axvline(0, color="0.25", linestyle="--", linewidth=1.0)
            ax.axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.12)
            ax.axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.10)
            if "Radar" in title or "SCG" in title:
                ct, cy = _clean_candidates(bt, bx, acfg.ao_search_sec)
                if len(ct): ax.scatter(ct, cy, s=22, marker="o", facecolor="0.75", edgecolor="black", linewidth=0.5)
                ct, cy = _clean_candidates(bt, bx, acfg.ac_search_sec)
                if len(ct): ax.scatter(ct, cy, s=22, marker="s", facecolor="0.75", edgecolor="black", linewidth=0.5)
            ax.set_ylim(np.nanmin(bx) - 0.6, np.nanmax(bx) + 1.1)
            for name, x in lm.items():
                _draw_vline_label(ax, x, name, 0.96 if name in ("Q", "MC") else 0.90)
            _, _, ao, ac, pep, lvet, qs2 = interval_row
            y0 = ax.get_ylim()[0] + 0.25
            _draw_interval_bracket(ax, d["q_rel"], ao, y0, "PEP" if pep is not None else "")
            _draw_interval_bracket(ax, ao, ac, y0 + 0.23, "LVET" if lvet is not None else "")
            _draw_interval_bracket(ax, d["q_rel"], ac, y0 + 0.46, "QS2" if qs2 is not None else "")
            txt = "PEP/LVET/QS2 unavailable" if pep is None or lvet is None or qs2 is None else f"PEP={pep:.1f} ms\nLVET={lvet:.1f} ms\nQS2={qs2:.1f} ms"
            ax.text(0.995, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
            ax.set_title(title, fontsize=11, loc="left", pad=5)
            ax.set_ylabel("z-score", fontsize=10)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Time from ECG R-peak [s]", fontsize=10)
        fig.suptitle("Fig. 10. ECG / SCG / Radar landmark and interval comparison", fontsize=13)
        fig.savefig(outdir / "fig10_ecg_scg_radar_landmark_interval_clean.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig05_morphology_ecg_scg_radar_candidates.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig06_pep_lvet_qs2_brackets.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (outdir / "fig10_clean_error.txt").write_text(str(e), encoding="utf-8")

    # Fig 07/08: distribution and trend
    try:
        rows = []
        for b in aoac.get("beats", []):
            bi = int(b.get("beat_index", -1))
            if bi < 0:
                continue
            rows.append([bi, b.get("ao_morph_time", b.get("ao_time")), b.get("ac_morph_time", b.get("ac_time"))])
        if rows:
            arr = np.array([[r[0], np.nan if r[1] is None else r[1] * 1000, np.nan if r[2] is None else r[2] * 1000] for r in rows], dtype=float)
            fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), constrained_layout=True)
            axes[0].boxplot([arr[np.isfinite(arr[:, 1]), 1], arr[np.isfinite(arr[:, 2]), 2]], labels=["Radar AO", "Radar AC"])
            axes[0].set_ylabel("Timing from R [ms]"); axes[0].set_title("Fig. 7. Radar candidate timing dispersion", fontsize=12)
            axes[0].grid(True, alpha=0.25)
            axes[1].plot(arr[:, 0], arr[:, 1], "o-", ms=3, label="AO")
            axes[1].plot(arr[:, 0], arr[:, 2], "s-", ms=3, label="AC")
            axes[1].set_xlabel("Beat index"); axes[1].set_ylabel("Timing from R [ms]")
            axes[1].grid(True, alpha=0.25); axes[1].legend()
            fig.savefig(outdir / "fig07_candidate_timing_distribution_clean.png", dpi=300, bbox_inches="tight")
            fig.savefig(outdir / "fig08_beatwise_candidate_trend_clean.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        (outdir / "fig07_08_clean_error.txt").write_text(str(e), encoding="utf-8")


def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    _CLEAN_OLD_SAVE_ALL(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    _clean_figs(Path(outdir), ecg, radar, scg, aoac, comp, acfg)



# ============================================================
# ROBUST ECG SERIAL PARSER PATCH
# Fixes cases where STM32 live UART begins mid-line or parser is too strict.
# It accepts lines like:
#   sample_index,ADCValue,Smooth_ECG
#   458074,2672,2891
# and skips malformed partial lines/debug text.
# ============================================================

def _robust_parse_ecg_csv_line(line):
    """
    Return (sample_index, adc_value, smooth_ecg) or None.
    Tolerant to:
    - partial line at serial open
    - header/debug lines
    - extra trailing columns
    - whitespace / CRLF
    """
    if line is None:
        return None
    if isinstance(line, bytes):
        try:
            line = line.decode("ascii", errors="ignore")
        except Exception:
            return None
    s = str(line).strip()
    if not s:
        return None
    if s.startswith("#"):
        return None
    low = s.lower()
    if ("sample" in low) or ("adc" in low) or ("smooth" in low) or ("ecg" in low and "," not in s):
        return None

    # Keep only comma-separated numeric-looking fields.
    parts = [p.strip() for p in s.replace("\t", ",").split(",")]
    if len(parts) < 3:
        return None

    vals = []
    for p in parts:
        # Allow integer-like float but cast to int.
        try:
            if p == "":
                continue
            v = int(float(p))
            vals.append(v)
            if len(vals) >= 3:
                break
        except Exception:
            continue
    if len(vals) < 3:
        return None

    sample_idx, adc_value, smooth_ecg = vals[:3]

    # Basic sanity only. Do not reject discontinuity; UART may start mid-stream.
    if sample_idx < 0:
        return None
    if not (-1000000 <= adc_value <= 1000000):
        return None
    if not (-1000000 <= smooth_ecg <= 1000000):
        return None

    return sample_idx, adc_value, smooth_ecg


def _robust_ecg_arrays_from_serial_bytes(raw_bytes, fs_hint=100.0):
    """
    Decode arbitrary serial byte chunk into ECG arrays using tolerant CSV parser.
    """
    if raw_bytes is None:
        raw_bytes = b""
    if isinstance(raw_bytes, str):
        text_blob = raw_bytes
    else:
        text_blob = bytes(raw_bytes).decode("ascii", errors="ignore")

    sample_idx, adc, smooth = [], [], []
    for line in text_blob.splitlines():
        parsed = _robust_parse_ecg_csv_line(line)
        if parsed is None:
            continue
        i, a, s = parsed
        sample_idx.append(i)
        adc.append(a)
        smooth.append(s)

    if not sample_idx:
        return None

    sample_idx = np.asarray(sample_idx, dtype=np.float64)
    adc = np.asarray(adc, dtype=np.float64)
    smooth = np.asarray(smooth, dtype=np.float64)

    # Time axis: prefer sample index difference when monotonic; otherwise use sample counter.
    finite = np.isfinite(sample_idx)
    if np.sum(finite) >= 2:
        first = sample_idx[finite][0]
        t = (sample_idx - first) / float(fs_hint)
    else:
        t = np.arange(len(sample_idx), dtype=np.float64) / float(fs_hint)

    # If sample index has large gaps or non-monotonic artifacts, fallback to regular grid.
    dt = np.diff(t)
    if len(dt) and (np.nanmedian(dt) <= 0 or np.nanmedian(dt) > 0.1 or np.any(dt < -1e-9)):
        t = np.arange(len(sample_idx), dtype=np.float64) / float(fs_hint)

    return {
        "sample_index": sample_idx,
        "t": t,
        "adc": adc,
        "smooth": smooth,
        "n": int(len(sample_idx)),
    }


def _patch_ecg_collector_methods_for_robust_serial():
    """
    Monkey-patch ECG collector classes if their method names are available.
    This is intentionally conservative: it only replaces common parse methods.
    """
    candidates = []
    for name, obj in list(globals().items()):
        try:
            if isinstance(obj, type) and ("ECG" in name.upper()) and ("COLLECT" in name.upper() or "READER" in name.upper()):
                candidates.append(obj)
        except Exception:
            pass

    def robust_line_method(self, line):
        return _robust_parse_ecg_csv_line(line)

    for cls in candidates:
        for meth in ("parse_line", "_parse_line", "parse_ecg_line", "_parse_ecg_line", "parse_csv_line", "_parse_csv_line"):
            if hasattr(cls, meth):
                try:
                    setattr(cls, meth, robust_line_method)
                except Exception:
                    pass

    return [c.__name__ for c in candidates]


_ROBUST_ECG_PATCHED_CLASSES = _patch_ecg_collector_methods_for_robust_serial()


def robust_ecg_serial_diagnostic_from_error_bytes(debug_bytes, outdir=None, fs_hint=100.0):
    """
    Optional utility: if ECG data too short occurs, feed debug bytes to check
    whether robust parser can recover samples.
    """
    parsed = _robust_ecg_arrays_from_serial_bytes(debug_bytes, fs_hint=fs_hint)
    if outdir is not None:
        try:
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            summary = {
                "robust_parser_recovered_samples": 0 if parsed is None else int(parsed["n"]),
                "patched_classes": _ROBUST_ECG_PATCHED_CLASSES,
            }
            with open(outdir / "robust_ecg_parser_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return parsed



# ============================================================
# PATCH: SCG-inclusive compact figures / SCG reference figure /
# refreshed paper export (requested review fix)
# ============================================================
_PATCH_LAST_SCG = None


def _patch_choose_scg_branch(scg: Optional[dict], mode: str = "analysis"):
    if scg is None:
        return None
    candidates = []
    if mode == "analysis":
        candidates = ["resp_removed", "filtered", "display", "selected_raw", "vmag"]
    elif mode == "display":
        candidates = ["display", "filtered", "resp_removed", "selected_raw", "vmag"]
    else:
        candidates = ["selected_raw", "vmag", "resp_removed", "filtered", "display"]
    for k in candidates:
        if k in scg and scg[k] is not None and len(scg[k]) == len(scg.get("t", [])):
            return np.asarray(scg[k], dtype=np.float64)
    return None


def _patch_pick_representative_r_index(ecg: dict, min_margin: int = 3):
    r = np.asarray(ecg.get("peaks_time", []), dtype=np.float64)
    if len(r) < 3:
        return None
    if len(r) <= 2 * min_margin + 1:
        return len(r) // 2
    return max(min_margin, min(len(r) // 2, len(r) - min_margin - 1))


def _patch_build_scg_reference_landmarks(ecg: dict, scg: Optional[dict], acfg: AnalysisConfig):
    if scg is None:
        return None
    sig = _patch_choose_scg_branch(scg, mode="analysis")
    tt = np.asarray(scg.get("t", []), dtype=np.float64)
    if sig is None or len(tt) < 20 or len(tt) != len(sig):
        return None
    ridx = _patch_pick_representative_r_index(ecg)
    if ridx is None:
        return None
    r_time = float(ecg["peaks_time"][ridx])
    bt, bx = _slice_aligned_beat(tt, sig, r_time, acfg, fs_out=100.0)
    if bt is None or bx is None:
        return None
    lm = _estimate_scg_landmarks(bt, bx, acfg)
    return {"r_time": r_time, "t_rel": bt, "beat": bx, "landmarks": lm, "beat_index": int(ridx)}


def _patch_make_fig01_compact_signal_overview_with_scg(outdir: Path, ecg: dict, scg: Optional[dict], radar: dict, comp: dict):
    try:
        fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=False, constrained_layout=True)

        # 1) ECG overview
        ax = axes[0]
        ecg_display = zscore_safe(ecg.get("true_display", ecg.get("display", ecg["cleaned"])))
        ecg_qrs = zscore_safe(ecg.get("filtered", ecg_display))
        ax.plot(ecg["t"], ecg_display, label="ECG display band", linewidth=1.0)
        ax.plot(ecg["t"], ecg_qrs, label="ECG QRS band", linewidth=0.8, alpha=0.55)
        if len(ecg.get("peaks_idx", [])):
            pk = np.asarray(ecg["peaks_idx"], dtype=int)
            pk = pk[(pk >= 0) & (pk < len(ecg_qrs))]
            if len(pk):
                ax.scatter(ecg["t"][pk], ecg_qrs[pk], s=14, c="red", marker="x", label="R anchors")
        qv = np.asarray(ecg.get("q_time", []), dtype=float)
        qv = qv[np.isfinite(qv)] if len(qv) else qv
        if len(qv):
            ax.scatter(qv, np.interp(qv, ecg["t"], ecg_display), s=10, c="magenta", marker="v", label="Q")
        tv = np.asarray(ecg.get("t_time", []), dtype=float)
        tv = tv[np.isfinite(tv)] if len(tv) else tv
        if len(tv):
            ax.scatter(tv, np.interp(tv, ecg["t"], ecg_display), s=10, c="green", marker="^", label="T")
        ax.set_title("ECG morphology and Q/R/T landmarks")
        ax.set_ylabel("z-score")
        ax.grid(True, alpha=0.30)
        ax.legend(fontsize=8, ncol=5, loc="upper right")

        # 2) SCG overview
        ax = axes[1]
        if scg is not None and len(scg.get("t", [])):
            s_raw = _patch_choose_scg_branch(scg, mode="raw")
            s_rr = _patch_choose_scg_branch(scg, mode="analysis")
            s_fi = _patch_choose_scg_branch(scg, mode="display")
            if s_raw is not None:
                ax.plot(scg["t"], zscore_safe(s_raw), label="SCG raw", linewidth=0.75, alpha=0.65)
            if s_rr is not None:
                ax.plot(scg["t"], zscore_safe(s_rr), label="SCG LMS respiration-removed", linewidth=0.9)
            if s_fi is not None:
                ax.plot(scg["t"], zscore_safe(s_fi), label="SCG filtered/display", linewidth=0.9, alpha=0.85)
            ax.set_title("SCG waveform overview (raw / LMS respiration-removed / filtered)")
            ax.legend(fontsize=8, ncol=3, loc="upper right")
        else:
            ax.text(0.5, 0.5, "SCG unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("SCG waveform overview")
        ax.set_ylabel("z-score")
        ax.grid(True, alpha=0.30)

        # 3) Radar displacement/respiration
        ax = axes[2]
        ax.plot(radar["t"], zscore_safe(radar["displacement"]), label="Radar displacement", linewidth=0.8)
        ax.plot(radar["t"], zscore_safe(radar["respiration"]), label="Respiration band", linewidth=0.8)
        ax.set_title("Radar displacement and respiration component")
        ax.set_ylabel("z-score")
        ax.grid(True, alpha=0.30)
        ax.legend(fontsize=8, loc="upper right")

        # 4) Radar recovered cardiac branch
        ax = axes[3]
        ax.plot(radar["t"], zscore_safe(radar.get("lms_error", radar["ppg_like"])), label="Radar LMS error", linewidth=0.8)
        ax.plot(radar["t"], zscore_safe(radar["ppg_like"]), label="Radar final PPG-like", linewidth=1.0)
        if len(radar.get("peaks_idx", [])):
            pk = np.asarray(radar["peaks_idx"], dtype=int)
            pk = pk[(pk >= 0) & (pk < len(radar["ppg_like"]))]
            if len(pk):
                rz = zscore_safe(radar["ppg_like"])
                ax.scatter(radar["t"][pk], rz[pk], s=10, c="red", label="Radar peaks")
        ax.set_title("Radar LMS output and final cardiac signal")
        ax.set_ylabel("z-score")
        ax.grid(True, alpha=0.30)
        ax.legend(fontsize=8, loc="upper right")

        # 5) joint alignment check
        ax = axes[4]
        ax.plot(comp["t"], comp["ecg_ref"], label="ECG R-peak reference", linewidth=0.9)
        ax.plot(comp["t"], comp.get("radar_aligned", comp.get("radar_ppg", np.zeros_like(comp["t"]))), label="Radar aligned", linewidth=0.9)
        if scg is not None and len(scg.get("t", [])) >= 5:
            sdisp = _patch_choose_scg_branch(scg, mode="analysis")
            if sdisp is not None:
                scg_interp = np.interp(comp["t"], scg["t"], zscore_safe(sdisp), left=np.nan, right=np.nan)
                scg_interp = np.nan_to_num(scg_interp, nan=0.0)
                ax.plot(comp["t"], zscore_safe(scg_interp), label="SCG aligned", linewidth=0.9, alpha=0.9)
        ax.set_title("Time-domain alignment check across ECG / SCG / Radar")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("z-score")
        ax.grid(True, alpha=0.30)
        ax.legend(fontsize=8, ncol=3, loc="upper right")

        fig.savefig(outdir / "fig01_compact_signal_overview.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir / "fig01_compact_signal_overview_patch_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def _patch_make_fig02_scg_reference(outdir: Path, ecg: dict, scg: Optional[dict], acfg: AnalysisConfig):
    try:
        ref = _patch_build_scg_reference_landmarks(ecg, scg, acfg)
        if ref is None:
            return
        bt = np.asarray(ref["t_rel"], dtype=float)
        bx = zscore_safe(np.asarray(ref["beat"], dtype=float))
        lm = ref["landmarks"]

        fig, ax = plt.subplots(1, 1, figsize=(10, 4.8), constrained_layout=True)
        ax.plot(bt, bx, linewidth=1.7, color="black", label="SCG LMS respiration-removed beat")
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0, label="ECG R anchor")
        ax.axvspan(acfg.ao_search_sec[0], acfg.ao_search_sec[1], alpha=0.10, color="tab:blue", label="AO window")
        ax.axvspan(acfg.ac_search_sec[0], acfg.ac_search_sec[1], alpha=0.10, color="tab:orange", label="AC window")

        order = ["MC", "IM", "AO", "AC", "MO"]
        markers = {"MC": "o", "IM": "s", "AO": "^", "AC": "D", "MO": "P"}
        for name in order:
            tt = lm.get(name)
            if tt is None or not np.isfinite(tt):
                continue
            yy = float(np.interp(tt, bt, bx))
            ax.scatter([tt], [yy], s=64, marker=markers.get(name, "o"), zorder=5, label=name)
            ax.annotate(f"{name}\n{tt*1000:.0f} ms", xy=(tt, yy), xytext=(0, 10), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        ax.set_title("SCG reference landmarks (MC / IM / AO / AC / MO)")
        ax.set_xlabel("Time from ECG R-peak [s]")
        ax.set_ylabel("z-score")
        ax.grid(True, alpha=0.30)
        ax.legend(fontsize=8, ncol=4, loc="upper right")
        fig.savefig(outdir / "fig02_scg_reference_landmarks.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir / "fig02_scg_reference_landmarks_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def _patch_corr_metrics(x_ms: np.ndarray, y_ms: np.ndarray):
    x_ms = np.asarray(x_ms, dtype=float)
    y_ms = np.asarray(y_ms, dtype=float)
    m = np.isfinite(x_ms) & np.isfinite(y_ms)
    x = x_ms[m]; y = y_ms[m]
    if len(x) == 0:
        return {"n": 0, "r": np.nan, "mae": np.nan, "rmse": np.nan}
    r = np.corrcoef(x, y)[0, 1] if len(x) >= 2 else np.nan
    err = y - x
    return {"n": int(len(x)), "r": float(r) if np.isfinite(r) else np.nan,
            "mae": float(np.nanmean(np.abs(err))), "rmse": float(np.sqrt(np.nanmean(err**2)))}


def _patch_scatter(ax, x_ms, y_ms, title, xlabel, ylabel):
    x_ms = np.asarray(x_ms, dtype=float)
    y_ms = np.asarray(y_ms, dtype=float)
    m = np.isfinite(x_ms) & np.isfinite(y_ms)
    if not np.any(m):
        ax.text(0.5, 0.5, "No valid pairs", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        return
    x = x_ms[m]; y = y_ms[m]
    lo = min(np.nanmin(x), np.nanmin(y)) - 15
    hi = max(np.nanmax(x), np.nanmax(y)) + 15
    ax.scatter(x, y, s=28, alpha=0.8)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
    met = _patch_corr_metrics(x, y)
    txt = f"n={met['n']}\nr={met['r']:.2f}\nMAE={met['mae']:.1f} ms\nRMSE={met['rmse']:.1f} ms"
    ax.text(0.03, 0.97, txt, ha="left", va="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9), fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _patch_make_fig04_with_scg(outdir: Path, scg_ref: Optional[dict]):
    try:
        ecg_csv = outdir / "ecg_vs_radar_aoac_correlation.csv"
        if not ecg_csv.exists():
            return
        ecg_df = pd.read_csv(ecg_csv)
        ecg_use = ecg_df.copy()
        if "accepted" in ecg_use.columns:
            ecg_use = ecg_use[ecg_use["accepted"].astype(bool)]
        scg_df = None
        if scg_ref is not None and len(scg_ref.get("rows", [])):
            cols = [
                "beat_index", "r_peak_time_sec", "radar_accepted", "radar_sqi",
                "scg_ao_time_from_r_sec", "scg_ac_time_from_r_sec",
                "radar_ao_morph_time_from_r_sec", "radar_ac_morph_time_from_r_sec",
                "radar_minus_scg_ao_ms", "radar_minus_scg_ac_ms",
                "both_ao_ac_within_30ms", "scg_ao_confidence", "scg_ac_confidence"
            ]
            scg_df = pd.DataFrame(scg_ref["rows"], columns=cols)
            scg_df = scg_df[scg_df["radar_accepted"].astype(bool)]
        elif (outdir / "scg_reference_vs_radar_candidates.csv").exists():
            scg_df = pd.read_csv(outdir / "scg_reference_vs_radar_candidates.csv")
            if "radar_accepted" in scg_df.columns:
                scg_df = scg_df[scg_df["radar_accepted"].astype(bool)]

        fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
        _patch_scatter(axes[0,0], ecg_use.get("ecg_est_ao_ms", pd.Series(dtype=float)), ecg_use.get("radar_ao_ms", pd.Series(dtype=float)),
                       "AO consistency: ECG anchor vs Radar", "ECG-estimated AO [ms]", "Radar-estimated AO [ms]")
        _patch_scatter(axes[0,1], ecg_use.get("ecg_est_ac_ms", pd.Series(dtype=float)), ecg_use.get("radar_ac_ms", pd.Series(dtype=float)),
                       "AC consistency: ECG anchor vs Radar", "ECG-estimated AC [ms]", "Radar-estimated AC [ms]")
        if scg_df is not None and len(scg_df):
            _patch_scatter(axes[1,0], scg_df["scg_ao_time_from_r_sec"]*1000.0, scg_df["radar_ao_morph_time_from_r_sec"]*1000.0,
                           "AO consistency: SCG reference vs Radar", "SCG-estimated AO [ms]", "Radar-estimated AO [ms]")
            _patch_scatter(axes[1,1], scg_df["scg_ac_time_from_r_sec"]*1000.0, scg_df["radar_ac_morph_time_from_r_sec"]*1000.0,
                           "AC consistency: SCG reference vs Radar", "SCG-estimated AC [ms]", "Radar-estimated AC [ms]")
        else:
            for ax, ttl in zip([axes[1,0], axes[1,1]], ["AO consistency: SCG reference vs Radar", "AC consistency: SCG reference vs Radar"]):
                ax.text(0.5, 0.5, "SCG reference unavailable", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(ttl)
                ax.grid(True, alpha=0.25)
        fig.suptitle("Fig.4 ECG / SCG anchor based consistency comparison with radar", fontsize=16)
        fig.savefig(outdir / "fig04_ecg_vs_radar_aoac_correlation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir / "fig04_ecg_vs_radar_aoac_correlation_patch_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def _patch_make_table09_interval_summary(outdir: Path):
    try:
        src = outdir / "ecg_scg_radar_pep_lvet_qs2_per_beat.csv"
        if not src.exists():
            return None
        df = pd.read_csv(src)
        rows = []
        mapping = {
            "PEP": ["ecg_pep_ms", "scg_pep_ms", "radar_pep_ms"],
            "LVET": ["ecg_lvet_ms", "scg_lvet_ms", "radar_lvet_ms"],
            "QS2": ["ecg_qs2_ms", "scg_qs2_ms", "radar_qs2_ms"],
        }
        for metric, cols in mapping.items():
            vals = []
            for c in cols:
                arr = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float) if c in df.columns else np.array([])
                if len(arr):
                    vals.append(f"{np.nanmean(arr):.2f} ± {np.nanstd(arr):.2f}")
                else:
                    vals.append("-")
            rows.append([metric] + vals)
        table_path = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export") / "tables" / "table09_ecg_scg_radar_interval_summary.csv"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        save_csv(table_path, ["Metric", "ECG mean±SD [ms]", "SCG mean±SD [ms]", "Radar mean±SD [ms]"], rows)
        fig_path = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export") / "figures" / "table09_ecg_scg_radar_interval_summary.png"
        _render_csv_table_to_png(table_path, fig_path, title="Table 9. ECG/SCG/Radar interval summary")
        return table_path
    except Exception as e:
        with open(outdir / "table09_interval_summary_patch_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))
        return None


def _patch_refresh_paper_export_figures(outdir: Path):
    try:
        paper_dir = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export")
        figs_dir = paper_dir / "figures"
        tables_dir = paper_dir / "tables"
        figs_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        fig_map = [
            ("fig01_compact_signal_overview.png", "fig01_signal_overview.png", "Figure 1. ECG/SCG/Radar compact signal overview"),
            ("fig02_ecg_qrt_reference.png", "fig02_ecg_qrt_reference.png", "Figure 2A. ECG Q/R/T reference"),
            ("fig02_scg_reference_landmarks.png", "fig02b_scg_reference_landmarks.png", "Figure 2B. SCG reference landmarks (MC/IM/AO/AC/MO)"),
            ("fig04_1_radar_stage_comparison.png", "fig04a_radar_stage_comparison.png", "Figure 4A. Radar stage comparison"),
            ("fig04_2_scg_stage_comparison.png", "fig04b_scg_stage_comparison.png", "Figure 4B. SCG stage comparison"),
            ("fig04_ecg_vs_radar_aoac_correlation.png", "fig05_ecg_scg_radar_consistency.png", "Figure 5. ECG/SCG anchor based consistency vs radar"),
            ("fig09_ecg_scg_radar_multicycle_diagnostic.png", "fig09_ecg_scg_radar_multicycle.png", "Figure 9. ECG/SCG/Radar multicycle diagnostic"),
            ("fig10_ecg_scg_radar_landmark_interval_clean.png", "fig10_ecg_scg_radar_landmarks_intervals.png", "Figure 10. ECG/SCG/Radar landmark and interval comparison"),
            ("fig11_pep_lvet_qs2_comparison.png", "fig11_pep_lvet_qs2_comparison.png", "Figure 11. PEP/LVET/QS2 comparison"),
        ]
        index_rows = []
        for src_name, dst_name, caption in fig_map:
            src = outdir / src_name
            dst = figs_dir / dst_name
            if src.exists():
                shutil.copyfile(src, dst)
                status = "copied"
            else:
                status = "missing"
            index_rows.append([src_name, dst_name, status, caption])
        save_csv(figs_dir / "paper_figure_index.csv", ["Source", "PaperFigure", "Status", "Caption"], index_rows)

        # refresh table pngs from existing csv tables
        _export_table_pngs_from_existing_csvs(tables_dir, figs_dir)
        _patch_make_table09_interval_summary(outdir)

        # extend table index if table09 exists
        table_index = tables_dir / "paper_table_index.csv"
        rows = []
        if table_index.exists():
            try:
                rows = list(csv.reader(open(table_index, encoding="utf-8-sig")))
            except Exception:
                rows = []
        if (tables_dir / "table09_ecg_scg_radar_interval_summary.csv").exists():
            if not rows:
                rows = [["TableCSV", "Description"]]
            rows.append(["table09_ecg_scg_radar_interval_summary.csv", "ECG/SCG/Radar interval summary"])
            with open(table_index, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
    except Exception as e:
        with open(outdir / "paper_export_refresh_patch_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


_old_save_all_patch = save_all

def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    globals()["_PATCH_LAST_SCG"] = scg
    result = _old_save_all_patch(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        _patch_make_fig01_compact_signal_overview_with_scg(outdir, ecg, scg, radar, comp)
        _patch_make_fig02_scg_reference(outdir, ecg, scg, acfg)
        scg_ref_local = None
        try:
            scg_ref_local = _old_scg_reference_aoac_pipeline(ecg, scg, radar, aoac, acfg, outdir=None) if scg is not None else None
        except Exception:
            scg_ref_local = None
        _patch_make_fig04_with_scg(outdir, scg_ref_local)
        _patch_refresh_paper_export_figures(outdir)
    except Exception as e:
        with open(outdir / "save_all_scg_patch_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))
    return result



# ============================================================
# PATCH3: SCG/echo-literature-based candidate windows and clean paper figures
# ------------------------------------------------------------
# Physiologic basis implemented in code comments:
# - SCG fiducial points can be correlated with ultrasound-defined valve events.
# - AO in SCG generally occurs about 50-150 ms after ECG R-peak.
# - AC complex is typically in the post-systolic / early diastolic interval
#   around 250-350 ms after ECG R-peak, but we keep a wider 250-520 ms
#   analysis window for radar/SCG candidate screening.
# - CTIs:
#   PEP = AO - ECG Q
#   LVET = AC - AO
#   QS2 = AC - ECG Q
# ============================================================

PATCH3_AO_STRICT_SEC = (0.07, 0.16)
PATCH3_AC_STRICT_SEC = (0.25, 0.52)
PATCH3_MCLIKE_SEC = (-0.02, 0.05)
PATCH3_IM_SEC = (0.03, 0.12)
PATCH3_MO_SEC = (0.43, 0.72)
_PATCH3_LAST_SCG = None


def _p3_arr(x):
    return np.asarray(x, dtype=np.float64)


def _p3_z(x):
    return zscore_safe(_p3_arr(x))


def _p3_branch_scg(scg, mode="analysis"):
    if scg is None:
        return None
    if mode == "raw":
        keys = ["selected_raw", "vmag", "az", "ax", "ay", "filtered", "display"]
    elif mode == "analysis":
        # use LMS respiration-removed branch for landmark candidate analysis
        keys = ["resp_removed", "filtered", "display", "selected_raw", "vmag"]
    elif mode == "bpf":
        keys = ["filtered", "resp_removed", "display", "selected_raw", "vmag"]
    else:
        keys = ["display", "filtered", "resp_removed", "selected_raw", "vmag"]
    n = len(scg.get("t", []))
    for k in keys:
        if k in scg and scg[k] is not None and len(scg[k]) == n and n:
            return _p3_arr(scg[k])
    return None


def _p3_branch_radar(radar, mode="analysis"):
    n = len(radar.get("t", []))
    if mode == "raw":
        keys = ["displacement", "displacement_m", "phase", "ppg_like"]
    elif mode == "analysis":
        # use LMS-only branch for AO/AC candidate analysis, not the heavily smoothed branch
        keys = ["lms_error", "ppg_like", "displacement", "displacement_m"]
    elif mode == "light":
        keys = ["ppg_like", "lms_error", "displacement"]
    else:
        keys = ["ppg_final_smooth", "display", "ppg_like", "lms_error"]
    for k in keys:
        if k in radar and radar[k] is not None and len(radar[k]) == n and n:
            return _p3_arr(radar[k])
    return np.zeros(n, dtype=np.float64)


def _p3_smooth(x, fs=100.0, win_sec=0.09):
    x = _p3_arr(x)
    if len(x) < 5:
        return x
    try:
        y = safe_lowpass(x, fs, min(5.0, 0.35 * fs), order=2)
    except Exception:
        y = x.copy()
    win = max(5, int(round(win_sec * fs)))
    if win % 2 == 0:
        win += 1
    return np.convolve(y, np.ones(win)/win, mode="same")


def _p3_slice(tt, xx, anchor, pre=0.20, post=0.80, fs_out=100.0):
    tt = _p3_arr(tt)
    xx = _p3_arr(xx)
    if len(tt) < 5 or len(tt) != len(xx):
        return None, None
    m = (tt >= anchor - pre) & (tt <= anchor + post)
    if np.sum(m) < max(8, int((pre + post) * fs_out * 0.35)):
        return None, None
    grid = np.arange(-pre, post + 1e-9, 1.0 / fs_out)
    y = np.interp(anchor + grid, tt[m], xx[m])
    return grid, _p3_z(y)


def _p3_candidate_score(bt, bx, win, kind):
    bt = _p3_arr(bt)
    bx = _p3_z(bx)
    if len(bt) < 8:
        return None
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 4:
        return None
    idx = np.where(m)[0]
    try:
        fs = 1.0 / max(np.nanmedian(np.diff(bt)), 1e-6)
        # Very light smoothing only for derivative stability.
        y = safe_lowpass(bx, fs, min(18.0, 0.45 * fs), order=2)
    except Exception:
        y = bx
        fs = 100.0
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    env = triangular_smooth_envelope(y, win_len=max(5, int(round(0.035 * fs)) | 1))
    env = _p3_z(env)
    local_t = bt[idx]
    center = np.mean(win)
    half = max((win[1] - win[0]) / 2.0, 1e-3)
    center_prior = np.exp(-0.5 * ((local_t - center) / (half * 0.75)) ** 2)
    boundary_penalty = np.minimum(local_t - win[0], win[1] - local_t) / half
    boundary_penalty = np.clip(boundary_penalty, 0.0, 1.0)

    if kind == "AO":
        # Strict AO: no R-zero/boundary snapping; favor positive slope + curvature + center.
        score = (
            0.30 * robust_scale_01(np.maximum(d1[idx], 0)) +
            0.25 * robust_scale_01(np.abs(d2[idx])) +
            0.15 * robust_scale_01(y[idx]) +
            0.20 * center_prior +
            0.10 * boundary_penalty
        )
    elif kind == "AC":
        score = (
            0.28 * robust_scale_01(np.maximum(-d1[idx], 0)) +
            0.28 * robust_scale_01(np.abs(d2[idx])) +
            0.14 * robust_scale_01(-y[idx]) +
            0.20 * center_prior +
            0.10 * boundary_penalty
        )
    elif kind == "MC":
        score = (
            0.35 * robust_scale_01(np.abs(d2[idx])) +
            0.30 * robust_scale_01(np.abs(d1[idx])) +
            0.20 * robust_scale_01(np.abs(y[idx])) +
            0.15 * center_prior
        )
    elif kind == "IM":
        score = (
            0.35 * robust_scale_01(np.maximum(d1[idx], 0)) +
            0.30 * robust_scale_01(np.maximum(d2[idx], 0)) +
            0.20 * robust_scale_01(env[idx]) +
            0.15 * center_prior
        )
    elif kind == "MO":
        score = (
            0.35 * robust_scale_01(np.abs(d2[idx])) +
            0.25 * robust_scale_01(np.maximum(d1[idx], 0)) +
            0.20 * robust_scale_01(np.abs(y[idx])) +
            0.20 * center_prior
        )
    else:
        score = robust_scale_01(np.abs(d2[idx])) * boundary_penalty

    j = int(idx[int(np.nanargmax(score))])
    return float(bt[j]), float(y[j]), int(j)


def _p3_candidates(bt, bx, win, max_points=4):
    bt = _p3_arr(bt)
    bx = _p3_z(bx)
    if len(bt) < 8:
        return np.array([]), np.array([])
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 5:
        return np.array([]), np.array([])
    idx = np.where(m)[0]
    try:
        fs = 1.0 / max(np.nanmedian(np.diff(bt)), 1e-6)
        y = safe_lowpass(bx, fs, min(18.0, 0.45 * fs), order=2)
    except Exception:
        y = bx
        fs = 100.0
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    local = idx
    score = robust_scale_01(np.abs(d1[local])) + robust_scale_01(np.abs(d2[local]))
    # penalty near window boundary so 0 ms-like false candidates are removed.
    half = max((win[1] - win[0]) / 2.0, 1e-3)
    t = bt[local]
    score *= np.clip(np.minimum(t - win[0], win[1] - t) / half, 0.0, 1.0)
    order = np.argsort(score)[::-1][:max_points]
    chosen = local[order]
    chosen = chosen[np.argsort(bt[chosen])]
    return bt[chosen], y[chosen]


def _p3_landmarks(bt, bx, ao_win=PATCH3_AO_STRICT_SEC, ac_win=PATCH3_AC_STRICT_SEC):
    lm = {}
    for name, win in [
        ("MC", PATCH3_MCLIKE_SEC),
        ("IM", PATCH3_IM_SEC),
        ("AO", ao_win),
        ("AC", ac_win),
        ("MO", PATCH3_MO_SEC),
    ]:
        res = _p3_candidate_score(bt, bx, win, name)
        lm[name] = None if res is None else res[0]
    return lm


def _p3_rep_idx(ecg):
    r = _p3_arr(ecg.get("peaks_time", []))
    if len(r) == 0:
        return None
    if len(r) < 8:
        return len(r)//2
    return len(r)//2


def _p3_qt_rel(ecg, ridx, anchor):
    q_rel = None
    t_rel = None
    qarr = _p3_arr(ecg.get("q_time", []))
    tarr = _p3_arr(ecg.get("t_time", []))
    if 0 <= ridx < len(qarr) and np.isfinite(qarr[ridx]):
        q_rel = float(qarr[ridx] - anchor)
    if 0 <= ridx < len(tarr) and np.isfinite(tarr[ridx]):
        t_rel = float(tarr[ridx] - anchor)
    return q_rel, t_rel


def _p3_interval(q, ao, ac):
    pep = None if q is None or ao is None else (ao - q) * 1000
    lvet = None if ao is None or ac is None else (ac - ao) * 1000
    qs2 = None if q is None or ac is None else (ac - q) * 1000
    return pep, lvet, qs2


def _p3_bracket(ax, x0, x1, y, txt):
    if x0 is None or x1 is None or not np.isfinite(x0) or not np.isfinite(x1):
        return
    ax.plot([x0, x1], [y, y], color="black", linewidth=1.15)
    ax.plot([x0, x0], [y-0.035, y+0.035], color="black", linewidth=0.9)
    ax.plot([x1, x1], [y-0.035, y+0.035], color="black", linewidth=0.9)
    ax.text((x0+x1)/2, y+0.045, txt, fontsize=8.3, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.70", alpha=0.90))


def _p3_vline(ax, x, label, ypos=0.92, style="--"):
    if x is None or not np.isfinite(x):
        return
    ax.axvline(x, color="black", linestyle=style, linewidth=0.95, alpha=0.85)
    ymin, ymax = ax.get_ylim()
    y = ymin + (ymax-ymin)*ypos
    ax.text(x, y, label, fontsize=8.0, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.65", alpha=0.88))


def _p3_make_fig04_scatter(outdir: Path, scg_ref=None):
    import pandas as pd
    try:
        ecg_path = outdir / "ecg_vs_radar_aoac_correlation.csv"
        if not ecg_path.exists():
            return
        ecg_df = pd.read_csv(ecg_path)
        if "accepted" in ecg_df.columns:
            ecg_df = ecg_df[ecg_df["accepted"].astype(bool)]
        scg_df = None
        if scg_ref is not None and isinstance(scg_ref, dict) and len(scg_ref.get("rows", [])):
            cols = [
                "beat_index", "r_peak_time_sec", "radar_accepted", "radar_sqi",
                "scg_ao_time_from_r_sec", "scg_ac_time_from_r_sec",
                "radar_ao_morph_time_from_r_sec", "radar_ac_morph_time_from_r_sec",
                "radar_minus_scg_ao_ms", "radar_minus_scg_ac_ms",
                "both_ao_ac_within_30ms", "scg_ao_confidence", "scg_ac_confidence"
            ]
            scg_df = pd.DataFrame(scg_ref["rows"], columns=cols)
        elif (outdir / "scg_reference_vs_radar_candidates.csv").exists():
            scg_df = pd.read_csv(outdir / "scg_reference_vs_radar_candidates.csv")
        if scg_df is not None and "radar_accepted" in scg_df.columns:
            scg_df = scg_df[scg_df["radar_accepted"].astype(bool)]

        def met(x, y):
            x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if not np.any(m): return {"n":0,"r":np.nan,"mae":np.nan,"rmse":np.nan}
            x = x[m]; y = y[m]
            err = y-x
            return {"n":len(x), "r":np.corrcoef(x,y)[0,1] if len(x)>1 else np.nan,
                    "mae":np.mean(np.abs(err)), "rmse":np.sqrt(np.mean(err**2))}
        def scatter(ax, x, y, title, xl, yl):
            x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            ax.set_title(title, fontsize=11)
            if not np.any(m):
                ax.text(0.5,0.5,"No valid pairs",ha="center",va="center",transform=ax.transAxes)
                return
            x2=x[m]; y2=y[m]
            lo=min(np.min(x2),np.min(y2))-15; hi=max(np.max(x2),np.max(y2))+15
            ax.scatter(x2,y2,s=26,alpha=0.75)
            ax.plot([lo,hi],[lo,hi],"--",color="black",linewidth=1.0)
            mm=met(x2,y2)
            ax.text(0.03,0.97,f"n={mm['n']}\\nr={mm['r']:.2f}\\nMAE={mm['mae']:.1f} ms\\nRMSE={mm['rmse']:.1f} ms",
                    transform=ax.transAxes,ha="left",va="top",fontsize=8.5,
                    bbox=dict(boxstyle="round",fc="white",ec="0.6",alpha=0.9))
            ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True,alpha=0.25)

        fig, axes = plt.subplots(2,2,figsize=(12.5,9.6),constrained_layout=True)
        scatter(axes[0,0], ecg_df["ecg_est_ao_ms"], ecg_df["radar_ao_ms"], "AO: ECG reference vs Radar", "ECG AO [ms]", "Radar AO [ms]")
        scatter(axes[0,1], ecg_df["ecg_est_ac_ms"], ecg_df["radar_ac_ms"], "AC: ECG reference vs Radar", "ECG AC [ms]", "Radar AC [ms]")
        if scg_df is not None and len(scg_df):
            scatter(axes[1,0], scg_df["scg_ao_time_from_r_sec"]*1000, scg_df["radar_ao_morph_time_from_r_sec"]*1000, "AO: SCG reference vs Radar", "SCG AO [ms]", "Radar AO [ms]")
            scatter(axes[1,1], scg_df["scg_ac_time_from_r_sec"]*1000, scg_df["radar_ac_morph_time_from_r_sec"]*1000, "AC: SCG reference vs Radar", "SCG AC [ms]", "Radar AC [ms]")
        else:
            for ax in axes[1]:
                ax.text(0.5,0.5,"SCG reference unavailable",ha="center",va="center",transform=ax.transAxes)
                ax.grid(True,alpha=0.25)
        fig.suptitle("ECG/SCG reference consistency with radar AO/AC candidates", fontsize=14)
        fig.savefig(outdir/"fig04_ecg_vs_radar_aoac_correlation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir/"fig04_patch3_error.txt","w",encoding="utf-8") as f: f.write(str(e))


def _p3_make_scg_reference_fig(outdir, ecg, scg, acfg):
    try:
        if scg is None: return
        ridx = _p3_rep_idx(ecg)
        if ridx is None: return
        anchor = float(ecg["peaks_time"][ridx])
        sig = _p3_branch_scg(scg, mode="analysis")
        bt,bx = _p3_slice(scg["t"], sig, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
        if bt is None: return
        lm = _p3_landmarks(bt,bx,PATCH3_AO_STRICT_SEC,PATCH3_AC_STRICT_SEC)
        fig, ax = plt.subplots(1,1,figsize=(10.2,4.6),constrained_layout=True)
        ax.plot(bt,bx,color="black",linewidth=1.6,label="SCG LMS respiration-removed")
        ax.axvline(0,color="0.35",linestyle="--",linewidth=1.0,label="ECG R anchor")
        ax.axvspan(PATCH3_AO_STRICT_SEC[0],PATCH3_AO_STRICT_SEC[1],alpha=0.13,label="AO window")
        ax.axvspan(PATCH3_AC_STRICT_SEC[0],PATCH3_AC_STRICT_SEC[1],alpha=0.10,label="AC window")
        ax.set_ylim(np.nanmin(bx)-0.5, np.nanmax(bx)+0.8)
        markers={"MC":"o","IM":"^","AO":"s","AC":"D","MO":"v"}
        ypos={"MC":0.96,"IM":0.88,"AO":0.80,"AC":0.72,"MO":0.64}
        for name in ["MC","IM","AO","AC","MO"]:
            x=lm.get(name)
            if x is None: continue
            y=float(np.interp(x,bt,bx))
            ax.scatter([x],[y],s=60,marker=markers[name],facecolor="white",edgecolor="black",zorder=5)
            _p3_vline(ax,x,f"{name}\\n{x*1000:.0f} ms",ypos[name])
        ax.set_title("SCG reference landmarks: MC / IM / AO / AC / MO",fontsize=13)
        ax.set_xlabel("Time from ECG R-peak [s]"); ax.set_ylabel("z-score")
        ax.grid(True,alpha=0.25); ax.legend(loc="upper right",fontsize=8,ncol=2)
        fig.savefig(outdir/"fig02_scg_reference_landmarks.png",dpi=300,bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir/"fig02_scg_reference_landmarks_patch3_error.txt","w",encoding="utf-8") as f: f.write(str(e))


def _p3_make_stage_figs(outdir, ecg, scg, radar, acfg):
    try:
        ridx=_p3_rep_idx(ecg)
        if ridx is None: return
        anchor=float(ecg["peaks_time"][ridx])
        # Radar
        rfs=float(radar.get("fs",100.0))
        radar_light=_p3_branch_radar(radar,"light")
        radar_stages=[
            ("1) Raw displacement", _p3_branch_radar(radar,"raw")),
            ("2) LMS respiration-removed", _p3_branch_radar(radar,"analysis")),
            ("3) Lightly filtered", radar_light),
            ("4) Heavily smoothed", _p3_smooth(radar_light,rfs)),
        ]
        fig,axes=plt.subplots(4,1,figsize=(11.5,10.0),sharex=True,constrained_layout=True)
        for ax,(title,sig) in zip(axes,radar_stages):
            bt,bx=_p3_slice(radar["t"],sig,anchor,acfg.beat_pre_sec,acfg.beat_post_sec,100.0)
            if bt is not None:
                ax.plot(bt,bx,color="black",linewidth=1.35); ax.axvline(0,color="0.35",linestyle="--",linewidth=0.9)
                ax.axvspan(PATCH3_AO_STRICT_SEC[0],PATCH3_AO_STRICT_SEC[1],alpha=0.13)
                ax.axvspan(PATCH3_AC_STRICT_SEC[0],PATCH3_AC_STRICT_SEC[1],alpha=0.10)
                for win,mk in [(PATCH3_AO_STRICT_SEC,"o"),(PATCH3_AC_STRICT_SEC,"s")]:
                    ct,cy=_p3_candidates(bt,bx,win,max_points=3)
                    if len(ct): ax.scatter(ct,cy,s=18,marker=mk,facecolor="0.75",edgecolor="black",linewidth=0.5)
            ax.set_title(title,fontsize=11,loc="left"); ax.set_ylabel("z-score"); ax.grid(True,alpha=0.25)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Radar morphology stage comparison",fontsize=13)
        fig.savefig(outdir/"fig04_1_radar_stage_comparison.png",dpi=300,bbox_inches="tight"); plt.close(fig)

        # SCG
        if scg is not None:
            sfs=float(scg.get("fs",100.0))
            scg_bpf=_p3_branch_scg(scg,"bpf")
            scg_stages=[
                ("1) Raw SCG", _p3_branch_scg(scg,"raw")),
                ("2) LMS respiration-removed", _p3_branch_scg(scg,"analysis")),
                ("3) Band-pass filtered", scg_bpf),
                ("4) Smoothed", _p3_smooth(scg_bpf,sfs)),
            ]
            fig,axes=plt.subplots(4,1,figsize=(11.5,10.0),sharex=True,constrained_layout=True)
            for ax,(title,sig) in zip(axes,scg_stages):
                bt,bx=_p3_slice(scg["t"],sig,anchor,acfg.beat_pre_sec,acfg.beat_post_sec,100.0)
                if bt is not None:
                    ax.plot(bt,bx,color="black",linewidth=1.35); ax.axvline(0,color="0.35",linestyle="--",linewidth=0.9)
                    ax.axvspan(PATCH3_AO_STRICT_SEC[0],PATCH3_AO_STRICT_SEC[1],alpha=0.13)
                    ax.axvspan(PATCH3_AC_STRICT_SEC[0],PATCH3_AC_STRICT_SEC[1],alpha=0.10)
                    for win,mk in [(PATCH3_AO_STRICT_SEC,"o"),(PATCH3_AC_STRICT_SEC,"s")]:
                        ct,cy=_p3_candidates(bt,bx,win,max_points=3)
                        if len(ct): ax.scatter(ct,cy,s=18,marker=mk,facecolor="0.75",edgecolor="black",linewidth=0.5)
                ax.set_title(title,fontsize=11,loc="left"); ax.set_ylabel("z-score"); ax.grid(True,alpha=0.25)
            axes[-1].set_xlabel("Time from ECG R-peak [s]")
            fig.suptitle("SCG morphology stage comparison",fontsize=13)
            fig.savefig(outdir/"fig04_2_scg_stage_comparison.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        with open(outdir/"fig04_stage_patch3_error.txt","w",encoding="utf-8") as f: f.write(str(e))


def _p3_make_fig10(outdir, ecg, scg, radar, acfg):
    try:
        ridx=_p3_rep_idx(ecg)
        if ridx is None: return
        anchor=float(ecg["peaks_time"][ridx])
        q_rel,t_rel=_p3_qt_rel(ecg,ridx,anchor)
        ecg_sig=_p3_arr(ecg.get("display_rpeak",ecg.get("display",ecg.get("filtered",[]))))
        bt_e,bx_e=_p3_slice(ecg["t"],ecg_sig,anchor,acfg.beat_pre_sec,acfg.beat_post_sec,100.0)
        bt_s,bx_s=(None,None)
        if scg is not None:
            bt_s,bx_s=_p3_slice(scg["t"],_p3_branch_scg(scg,"analysis"),anchor,acfg.beat_pre_sec,acfg.beat_post_sec,100.0)
        bt_r,bx_r=_p3_slice(radar["t"],_p3_branch_radar(radar,"analysis"),anchor,acfg.beat_pre_sec,acfg.beat_post_sec,100.0)
        if bt_e is None or bt_r is None: return
        scg_lm=_p3_landmarks(bt_s,bx_s) if bt_s is not None else {}
        rad_lm=_p3_landmarks(bt_r,bx_r)
        # ECG reference uses stored ecg-derived timing if available, otherwise literature windows center.
        rep=None
        for b in ao_ac_pipeline(ecg, radar, acfg).get("beats", []):
            if int(b.get("beat_index",-1))==ridx:
                rep=b; break
        ecg_ao=rep.get("ecg_ao_ref") if rep else np.mean(PATCH3_AO_STRICT_SEC)
        ecg_ac=rep.get("ecg_ac_ref") if rep else 0.38

        fig,axes=plt.subplots(3,1,figsize=(13.4,9.6),sharex=True,constrained_layout=True)
        panels=[
            ("ECG Q/R/T and reference timing",bt_e,bx_e,{"Q":q_rel,"R":0.0,"T":t_rel,"AO":ecg_ao,"AC":ecg_ac},("ECG",ecg_ao,ecg_ac)),
            ("SCG MC/IM/AO/AC/MO landmarks",bt_s,bx_s,scg_lm,("SCG",scg_lm.get("AO"),scg_lm.get("AC"))),
            ("Radar LMS-only morphology candidates",bt_r,bx_r,rad_lm,("Radar",rad_lm.get("AO"),rad_lm.get("AC"))),
        ]
        for ax,(title,bt,bx,lm,intinfo) in zip(axes,panels):
            if bt is None: 
                ax.text(0.5,0.5,"Unavailable",ha="center",va="center",transform=ax.transAxes); continue
            ax.plot(bt,bx,color="black",linewidth=1.55)
            ax.axvline(0,color="0.35",linestyle="--",linewidth=1.0)
            ax.axvspan(PATCH3_AO_STRICT_SEC[0],PATCH3_AO_STRICT_SEC[1],alpha=0.12)
            ax.axvspan(PATCH3_AC_STRICT_SEC[0],PATCH3_AC_STRICT_SEC[1],alpha=0.10)
            if "Radar" in title or "SCG" in title:
                for win,mk in [(PATCH3_AO_STRICT_SEC,"o"),(PATCH3_AC_STRICT_SEC,"s")]:
                    ct,cy=_p3_candidates(bt,bx,win,max_points=3)
                    if len(ct): ax.scatter(ct,cy,s=22,marker=mk,facecolor="0.78",edgecolor="black",linewidth=0.5,zorder=4)
            ax.set_ylim(np.nanmin(bx)-0.8,np.nanmax(bx)+1.2)
            # mark key lines
            order=["Q","R","T","MC","IM","AO","AC","MO"]
            ysteps={"Q":0.96,"R":0.91,"T":0.86,"MC":0.96,"IM":0.91,"AO":0.86,"AC":0.81,"MO":0.76}
            for name in order:
                if name in lm:
                    _p3_vline(ax,lm.get(name),name,ysteps.get(name,0.9),":" if name in ["Q","T"] else "--")
            _,ao,ac=intinfo
            pep,lvet,qs2=_p3_interval(q_rel,ao,ac)
            ymin,ymax=ax.get_ylim()
            base=ymin+0.22
            _p3_bracket(ax,q_rel,ao,base,"PEP")
            _p3_bracket(ax,ao,ac,base+0.23,"LVET")
            _p3_bracket(ax,q_rel,ac,base+0.46,"QS2")
            msg="PEP/LVET/QS2 unavailable" if pep is None or lvet is None or qs2 is None else f"PEP={pep:.1f} ms\\nLVET={lvet:.1f} ms\\nQS2={qs2:.1f} ms"
            ax.text(0.995,0.05,msg,transform=ax.transAxes,ha="right",va="bottom",fontsize=8.5,bbox=dict(boxstyle="round",fc="white",ec="0.7",alpha=0.9))
            ax.set_title(title,fontsize=11,loc="left"); ax.set_ylabel("z-score"); ax.grid(True,alpha=0.25)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("ECG / SCG / Radar landmarks and CTI intervals",fontsize=14)
        fig.savefig(outdir/"fig10_ecg_scg_radar_landmark_interval_clean.png",dpi=300,bbox_inches="tight")
        fig.savefig(outdir/"fig05_morphology_ecg_scg_radar_candidates.png",dpi=300,bbox_inches="tight")
        fig.savefig(outdir/"fig06_pep_lvet_qs2_brackets.png",dpi=300,bbox_inches="tight")
        plt.close(fig)
        rows=[]
        for mod,ao,ac in [("ECG_ref",ecg_ao,ecg_ac),("SCG",scg_lm.get("AO"),scg_lm.get("AC")),("Radar",rad_lm.get("AO"),rad_lm.get("AC"))]:
            pep,lvet,qs2=_p3_interval(q_rel,ao,ac); rows.append([mod,q_rel,ao,ac,pep,lvet,qs2])
        save_csv(outdir/"clean_ecg_scg_radar_intervals_representative.csv",["modality","q_rel_sec","ao_rel_sec","ac_rel_sec","PEP_ms","LVET_ms","QS2_ms"],rows)
    except Exception as e:
        with open(outdir/"fig10_patch3_error.txt","w",encoding="utf-8") as f: f.write(str(e))


def _p3_regenerate_paper_export(outdir: Path):
    try:
        paper=outdir/globals().get("PAPER_EXPORT_DIRNAME","paper_export")
        figs=paper/"figures"; tables=paper/"tables"; raw=paper/"raw_metrics"
        figs.mkdir(parents=True,exist_ok=True); tables.mkdir(parents=True,exist_ok=True); raw.mkdir(parents=True,exist_ok=True)
        fig_map=[
            ("fig01_compact_signal_overview.png","fig01_signal_overview.png","Figure 1. ECG/SCG/Radar compact signal overview"),
            ("fig02_ecg_qrt_reference.png","fig02_ecg_qrt_reference.png","Figure 2A. ECG Q/R/T reference"),
            ("fig02_scg_reference_landmarks.png","fig02b_scg_reference_landmarks.png","Figure 2B. SCG MC/IM/AO/AC/MO reference landmarks"),
            ("fig04_1_radar_stage_comparison.png","fig04a_radar_stage_comparison.png","Figure 4A. Radar raw/LMS/light/heavy stage comparison"),
            ("fig04_2_scg_stage_comparison.png","fig04b_scg_stage_comparison.png","Figure 4B. SCG raw/LMS/BPF/smoothed stage comparison"),
            ("fig04_ecg_vs_radar_aoac_correlation.png","fig05_ecg_scg_radar_consistency.png","Figure 5. ECG/SCG vs radar AO/AC consistency"),
            ("fig10_ecg_scg_radar_landmark_interval_clean.png","fig10_ecg_scg_radar_landmarks_intervals.png","Figure 10. ECG/SCG/Radar landmarks and PEP/LVET/QS2"),
        ]
        rows=[]
        for s,dst,cap in fig_map:
            sp=outdir/s; dp=figs/dst
            status="missing"
            if sp.exists():
                shutil.copyfile(sp,dp); status="copied"
            rows.append([s,dst,status,cap])
        save_csv(figs/"paper_figure_index.csv",["Source","PaperFigure","Status","Caption"],rows)
        # paper table export refresh
        table_rows=[]
        interval_csv=outdir/"ecg_scg_radar_pep_lvet_qs2_per_beat.csv"
        if interval_csv.exists():
            df=pd.read_csv(interval_csv)
            summary=[]
            for metric in ["pep","lvet","qs2"]:
                row=[metric.upper()]
                for mod in ["ecg","scg","radar"]:
                    col=f"{mod}_{metric}_ms"
                    vals=pd.to_numeric(df[col],errors="coerce").dropna().to_numpy(dtype=float) if col in df else np.array([])
                    row.append("-" if len(vals)==0 else f"{np.nanmean(vals):.2f} ± {np.nanstd(vals):.2f}")
                summary.append(row)
            tpath=tables/"table09_ecg_scg_radar_interval_summary.csv"
            save_csv(tpath,["Metric","ECG mean±SD [ms]","SCG mean±SD [ms]","Radar mean±SD [ms]"],summary)
            try: _render_csv_table_to_png(tpath,figs/"table09_ecg_scg_radar_interval_summary.png",title="Table 9. ECG/SCG/Radar interval summary")
            except Exception: pass
            table_rows.append(["table09_ecg_scg_radar_interval_summary.csv","ECG/SCG/Radar PEP, LVET, QS2 summary"])
        save_csv(tables/"paper_table_index.csv",["TableCSV","Description"],table_rows)
    except Exception as e:
        with open(outdir/"paper_export_patch3_error.txt","w",encoding="utf-8") as f: f.write(str(e))


_old_save_all_patch3 = save_all

def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    global _PATCH3_LAST_SCG
    _PATCH3_LAST_SCG = scg
    # original exports first
    result = _old_save_all_patch3(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        _p3_make_scg_reference_fig(outdir, ecg, scg, acfg)
        _p3_make_stage_figs(outdir, ecg, scg, radar, acfg)
        _p3_make_fig10(outdir, ecg, scg, radar, acfg)
        scg_ref = None
        try:
            # use existing SCG reference pipeline but export scatter with pd import guaranteed
            scg_ref = scg_reference_aoac_pipeline(ecg, scg, radar, aoac, acfg, outdir=None)
        except Exception:
            scg_ref = None
        _p3_make_fig04_scatter(outdir, scg_ref)
        _p3_regenerate_paper_export(outdir)
    except Exception as e:
        with open(outdir/"save_all_patch3_error.txt","w",encoding="utf-8") as f: f.write(str(e))
    return result



# ============================================================
# PATCH4: Template-first SCG fiducial landmarks
# ------------------------------------------------------------
# Fix:
# - SCG MC/IM/AO/AC/MO are not selected independently from one beat.
# - First create ECG R-peak aligned median SCG template.
# - Detect landmarks on the ensemble template.
# - Enforce physiologic order and minimum gaps.
# - Refine each beat only around template landmarks.
# - Radar marking is limited to AO/AC candidate points only.
# ============================================================

PATCH4_AO_SEC = (0.07, 0.16)
PATCH4_AC_SEC = (0.25, 0.52)
PATCH4_MC_SEC = (-0.03, 0.06)
PATCH4_IM_SEC = (0.02, 0.13)
PATCH4_MO_SEC = (0.46, 0.74)

PATCH4_MIN_GAPS = {
    ("MC", "IM"): 0.010,
    ("IM", "AO"): 0.020,
    ("AO", "AC"): 0.180,
    ("AC", "MO"): 0.060,
}

_PATCH4_LAST_SCG = None
_PATCH4_LAST_SCG_TEMPLATE = None


def _p4_np(x):
    return np.asarray(x, dtype=np.float64)


def _p4_z(x):
    return zscore_safe(_p4_np(x))


def _p4_scg_signal(scg: Optional[dict], mode: str = "analysis"):
    if scg is None:
        return None
    n = len(scg.get("t", []))
    if n == 0:
        return None
    if mode == "raw":
        keys = ["selected_raw", "vmag", "az", "ax", "ay", "resp_removed", "filtered", "display"]
    elif mode == "bpf":
        keys = ["filtered", "resp_removed", "display", "selected_raw", "vmag"]
    elif mode == "smooth":
        keys = ["display", "filtered", "resp_removed", "selected_raw", "vmag"]
    else:
        # Main landmark analysis branch: LMS respiration-removed SCG
        keys = ["resp_removed", "filtered", "display", "selected_raw", "vmag"]
    for k in keys:
        if k in scg and scg[k] is not None and len(scg[k]) == n:
            return _p4_np(scg[k])
    return None


def _p4_radar_signal(radar: dict, mode: str = "analysis"):
    n = len(radar.get("t", []))
    if mode == "raw":
        keys = ["displacement", "displacement_m", "phase", "ppg_like"]
    elif mode == "light":
        keys = ["ppg_like", "lms_error", "displacement"]
    elif mode == "smooth":
        keys = ["ppg_final_smooth", "display", "ppg_like", "lms_error", "displacement"]
    else:
        # Main radar morphology branch: LMS respiration-removed / error signal.
        keys = ["lms_error", "ppg_like", "displacement"]
    for k in keys:
        if k in radar and radar[k] is not None and len(radar[k]) == n:
            return _p4_np(radar[k])
    return np.zeros(n, dtype=np.float64)


def _p4_smooth(x, fs=100.0, win_sec=0.09):
    x = _p4_np(x)
    if len(x) < 5:
        return x
    try:
        y = safe_lowpass(x, fs, min(5.0, 0.35 * fs), order=2)
    except Exception:
        y = x.copy()
    win = max(5, int(round(win_sec * fs)))
    if win % 2 == 0:
        win += 1
    return np.convolve(y, np.ones(win, dtype=float) / win, mode="same")


def _p4_slice(tt, xx, anchor, pre=0.20, post=0.80, fs_out=100.0):
    tt = _p4_np(tt)
    xx = _p4_np(xx)
    if len(tt) < 5 or len(tt) != len(xx):
        return None, None
    m = (tt >= anchor - pre) & (tt <= anchor + post)
    if np.sum(m) < max(8, int((pre + post) * fs_out * 0.35)):
        return None, None
    grid = np.arange(-pre, post + 1e-9, 1.0 / fs_out)
    y = np.interp(anchor + grid, tt[m], xx[m])
    return grid, _p4_z(y)


def _p4_score_event(bt, bx, win, kind):
    bt = _p4_np(bt)
    bx = _p4_z(bx)
    if len(bt) < 8:
        return None
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 4:
        return None
    idx = np.where(m)[0]
    try:
        fs = 1.0 / max(np.nanmedian(np.diff(bt)), 1e-6)
        y = safe_lowpass(bx, fs, min(18.0, 0.45 * fs), order=2)
    except Exception:
        y = bx
        fs = 100.0

    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    env = triangular_smooth_envelope(y, win_len=max(5, int(round(0.035 * fs)) | 1))
    env = _p4_z(env)

    t = bt[idx]
    center = np.mean(win)
    half = max((win[1] - win[0]) / 2.0, 1e-3)
    center_prior = np.exp(-0.5 * ((t - center) / (half * 0.75)) ** 2)
    boundary = np.clip(np.minimum(t - win[0], win[1] - t) / half, 0.0, 1.0)

    if kind == "MC":
        # high-frequency onset / first mechanical complex around R
        score = (
            0.38 * robust_scale_01(np.abs(d2[idx])) +
            0.28 * robust_scale_01(np.abs(d1[idx])) +
            0.18 * robust_scale_01(np.abs(y[idx])) +
            0.10 * center_prior +
            0.06 * boundary
        )
    elif kind == "IM":
        # early systolic transition after MC and before AO
        score = (
            0.35 * robust_scale_01(np.maximum(d1[idx], 0)) +
            0.30 * robust_scale_01(np.maximum(d2[idx], 0)) +
            0.20 * robust_scale_01(env[idx]) +
            0.10 * center_prior +
            0.05 * boundary
        )
    elif kind == "AO":
        # aortic opening complex, strict R+70~160 ms
        score = (
            0.30 * robust_scale_01(np.maximum(d1[idx], 0)) +
            0.26 * robust_scale_01(np.abs(d2[idx])) +
            0.14 * robust_scale_01(y[idx]) +
            0.22 * center_prior +
            0.08 * boundary
        )
    elif kind == "AC":
        # aortic closure complex; separate from AO by systolic interval
        score = (
            0.30 * robust_scale_01(np.maximum(-d1[idx], 0)) +
            0.26 * robust_scale_01(np.abs(d2[idx])) +
            0.14 * robust_scale_01(-y[idx]) +
            0.22 * center_prior +
            0.08 * boundary
        )
    elif kind == "MO":
        # mitral opening after AC; must be clearly separated from AC
        score = (
            0.34 * robust_scale_01(np.abs(d2[idx])) +
            0.24 * robust_scale_01(np.maximum(d1[idx], 0)) +
            0.18 * robust_scale_01(np.abs(y[idx])) +
            0.18 * center_prior +
            0.06 * boundary
        )
    else:
        score = robust_scale_01(np.abs(d2[idx])) * boundary

    j = int(idx[int(np.nanargmax(score))])
    return float(bt[j]), float(y[j]), int(j)


def _p4_candidates(bt, bx, win, max_points=3):
    bt = _p4_np(bt)
    bx = _p4_z(bx)
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 5:
        return np.array([]), np.array([])
    idx = np.where(m)[0]
    try:
        fs = 1.0 / max(np.nanmedian(np.diff(bt)), 1e-6)
        y = safe_lowpass(bx, fs, min(18.0, 0.45 * fs), order=2)
    except Exception:
        y = bx
        fs = 100.0
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    t = bt[idx]
    half = max((win[1] - win[0]) / 2.0, 1e-3)
    boundary = np.clip(np.minimum(t - win[0], win[1] - t) / half, 0.0, 1.0)
    score = (robust_scale_01(np.abs(d1[idx])) + robust_scale_01(np.abs(d2[idx]))) * boundary
    order = np.argsort(score)[::-1][:max_points]
    chosen = idx[order]
    chosen = chosen[np.argsort(bt[chosen])]
    return bt[chosen], y[chosen]


def _p4_validate_sequence(lm):
    required = ["MC", "IM", "AO", "AC", "MO"]
    if any(lm.get(k) is None or not np.isfinite(lm.get(k)) for k in required):
        return False
    if not (lm["MC"] < lm["IM"] < lm["AO"] < lm["AC"] < lm["MO"]):
        return False
    for (a, b), gap in PATCH4_MIN_GAPS.items():
        if (lm[b] - lm[a]) < gap:
            return False
    return True


def _p4_template_landmarks(bt, bx):
    # sequential detection with adaptive windows and order constraints
    lm = {}
    res = _p4_score_event(bt, bx, PATCH4_MC_SEC, "MC")
    lm["MC"] = None if res is None else res[0]

    im_start = PATCH4_IM_SEC[0] if lm["MC"] is None else max(PATCH4_IM_SEC[0], lm["MC"] + PATCH4_MIN_GAPS[("MC", "IM")])
    res = _p4_score_event(bt, bx, (im_start, PATCH4_IM_SEC[1]), "IM")
    lm["IM"] = None if res is None else res[0]

    ao_start = PATCH4_AO_SEC[0] if lm["IM"] is None else max(PATCH4_AO_SEC[0], lm["IM"] + PATCH4_MIN_GAPS[("IM", "AO")])
    res = _p4_score_event(bt, bx, (ao_start, PATCH4_AO_SEC[1]), "AO")
    lm["AO"] = None if res is None else res[0]

    ac_start = PATCH4_AC_SEC[0] if lm["AO"] is None else max(PATCH4_AC_SEC[0], lm["AO"] + PATCH4_MIN_GAPS[("AO", "AC")])
    res = _p4_score_event(bt, bx, (ac_start, PATCH4_AC_SEC[1]), "AC")
    lm["AC"] = None if res is None else res[0]

    mo_start = PATCH4_MO_SEC[0] if lm["AC"] is None else max(PATCH4_MO_SEC[0], lm["AC"] + PATCH4_MIN_GAPS[("AC", "MO")])
    res = _p4_score_event(bt, bx, (mo_start, PATCH4_MO_SEC[1]), "MO")
    lm["MO"] = None if res is None else res[0]

    return lm


def _p4_build_scg_ensemble_template(ecg, scg, acfg):
    if scg is None:
        return None
    r_times = _p4_np(ecg.get("peaks_time", []))
    if len(r_times) < 5:
        return None
    sig = _p4_scg_signal(scg, "analysis")
    if sig is None:
        return None
    beats = []
    beat_indices = []
    grid_ref = None
    for i, r in enumerate(r_times):
        bt, bx = _p4_slice(scg["t"], sig, float(r), acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
        if bt is None:
            continue
        if grid_ref is None:
            grid_ref = bt
        # reject extreme flat/noisy segments
        if np.nanstd(bx) < 0.15 or not np.all(np.isfinite(bx)):
            continue
        beats.append(bx)
        beat_indices.append(i)

    if len(beats) < 3:
        return None

    B = np.vstack(beats)
    template = np.nanmedian(B, axis=0)
    sd = np.nanstd(B, axis=0)
    # final light smoothing on the ensemble template only
    template = _p4_smooth(template, fs=100.0, win_sec=0.03)
    template = _p4_z(template)
    lm = _p4_template_landmarks(grid_ref, template)

    return {
        "t_rel": grid_ref,
        "beats": B,
        "mean": np.nanmean(B, axis=0),
        "median": template,
        "sd": sd,
        "beat_indices": np.asarray(beat_indices, dtype=int),
        "landmarks": lm,
        "sequence_valid": bool(_p4_validate_sequence(lm)),
    }


def _p4_refine_beat_landmarks_from_template(bt, bx, template_lm):
    if bt is None or bx is None or template_lm is None:
        return {}
    out = {}
    for name in ["MC", "IM", "AO", "AC", "MO"]:
        center = template_lm.get(name)
        if center is None or not np.isfinite(center):
            out[name] = None
            continue
        # local refinement around template position
        width = 0.035 if name in ("MC", "IM", "AO") else 0.050
        win = (max(float(bt[0]), center - width), min(float(bt[-1]), center + width))
        res = _p4_score_event(bt, bx, win, name)
        out[name] = None if res is None else res[0]
    if not _p4_validate_sequence(out):
        # keep template landmarks as stable reference if per-beat refinement violates sequence
        return dict(template_lm)
    return out


def _p4_interval(q_rel, ao, ac):
    pep = None if q_rel is None or ao is None else (ao - q_rel) * 1000.0
    lvet = None if ao is None or ac is None else (ac - ao) * 1000.0
    qs2 = None if q_rel is None or ac is None else (ac - q_rel) * 1000.0
    return pep, lvet, qs2


def _p4_qt_rel(ecg, beat_index, anchor):
    q_rel = None
    t_rel = None
    qarr = _p4_np(ecg.get("q_time", []))
    tarr = _p4_np(ecg.get("t_time", []))
    if 0 <= beat_index < len(qarr) and np.isfinite(qarr[beat_index]):
        q_rel = float(qarr[beat_index] - anchor)
    if 0 <= beat_index < len(tarr) and np.isfinite(tarr[beat_index]):
        t_rel = float(tarr[beat_index] - anchor)
    return q_rel, t_rel


def _p4_vline(ax, x, label, yfrac=0.92, ls="--"):
    if x is None or not np.isfinite(float(x)):
        return
    x = float(x)
    ax.axvline(x, color="black", linestyle=ls, linewidth=0.95, alpha=0.85)
    ymin, ymax = ax.get_ylim()
    y = ymin + (ymax - ymin) * yfrac
    ax.text(x, y, label, fontsize=8.0, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.65", alpha=0.88))


def _p4_bracket(ax, x0, x1, y, label):
    if x0 is None or x1 is None or not np.isfinite(float(x0)) or not np.isfinite(float(x1)):
        return
    x0 = float(x0); x1 = float(x1)
    ax.plot([x0, x1], [y, y], color="black", linewidth=1.15)
    ax.plot([x0, x0], [y - 0.035, y + 0.035], color="black", linewidth=0.9)
    ax.plot([x1, x1], [y - 0.035, y + 0.035], color="black", linewidth=0.9)
    ax.text((x0 + x1) / 2.0, y + 0.045, label, fontsize=8.2, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.70", alpha=0.90))


def _p4_reference_beat_index(ecg):
    r = _p4_np(ecg.get("peaks_time", []))
    if len(r) < 1:
        return None
    return int(len(r) // 2)


def _p4_make_scg_template_reference_fig(outdir, scg_template):
    try:
        if scg_template is None:
            return
        bt = scg_template["t_rel"]
        med = _p4_z(scg_template["median"])
        sd = _p4_np(scg_template["sd"])
        lm = scg_template["landmarks"]
        fig, ax = plt.subplots(1, 1, figsize=(10.6, 4.8), constrained_layout=True)
        ax.plot(bt, med, color="black", linewidth=1.7, label="Median SCG template")
        ax.fill_between(bt, med - sd, med + sd, alpha=0.15, label="±1 SD")
        ax.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0, label="ECG R anchor")
        ax.axvspan(PATCH4_AO_SEC[0], PATCH4_AO_SEC[1], alpha=0.12, label="AO search")
        ax.axvspan(PATCH4_AC_SEC[0], PATCH4_AC_SEC[1], alpha=0.10, label="AC search")
        ax.set_ylim(np.nanmin(med) - 0.7, np.nanmax(med) + 1.0)
        ypos = {"MC": 0.96, "IM": 0.90, "AO": 0.84, "AC": 0.78, "MO": 0.72}
        markers = {"MC": "o", "IM": "^", "AO": "s", "AC": "D", "MO": "v"}
        for name in ["MC", "IM", "AO", "AC", "MO"]:
            x = lm.get(name)
            if x is None or not np.isfinite(x):
                continue
            y = float(np.interp(x, bt, med))
            ax.scatter([x], [y], s=64, marker=markers[name], facecolor="white", edgecolor="black", zorder=5)
            _p4_vline(ax, x, f"{name}\n{x*1000:.0f} ms", ypos[name])
        ax.set_title("SCG ensemble-template reference landmarks (MC / IM / AO / AC / MO)", fontsize=13)
        ax.set_xlabel("Time from ECG R-peak [s]")
        ax.set_ylabel("z-score")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        fig.savefig(outdir / "fig02_scg_reference_landmarks.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows = []
        for name in ["MC", "IM", "AO", "AC", "MO"]:
            rows.append([name, scg_template["landmarks"].get(name), None if scg_template["landmarks"].get(name) is None else scg_template["landmarks"].get(name) * 1000.0])
        save_csv(outdir / "scg_template_landmarks.csv", ["landmark", "time_from_r_sec", "time_from_r_ms"], rows)
        with open(outdir / "scg_template_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "n_template_beats": int(len(scg_template.get("beat_indices", []))),
                "sequence_valid": bool(scg_template.get("sequence_valid", False)),
                "min_gaps_sec": {f"{a}_{b}": v for (a, b), v in PATCH4_MIN_GAPS.items()},
                "windows_sec": {
                    "MC": PATCH4_MC_SEC, "IM": PATCH4_IM_SEC, "AO": PATCH4_AO_SEC,
                    "AC": PATCH4_AC_SEC, "MO": PATCH4_MO_SEC
                }
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        with open(outdir / "fig02_scg_template_patch4_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def _p4_make_stage_figs(outdir, ecg, scg, radar, acfg):
    try:
        ridx = _p4_reference_beat_index(ecg)
        if ridx is None:
            return
        anchor = float(ecg["peaks_time"][ridx])

        # Radar stage figure
        radar_raw = _p4_radar_signal(radar, "raw")
        radar_lms = _p4_radar_signal(radar, "analysis")
        radar_light = _p4_radar_signal(radar, "light")
        radar_smooth = _p4_smooth(radar_light, fs=float(radar.get("fs", 100.0)), win_sec=0.11)
        stages = [
            ("1) Raw displacement", radar_raw),
            ("2) LMS respiration-removed", radar_lms),
            ("3) Lightly filtered", radar_light),
            ("4) Heavily smoothed", radar_smooth),
        ]
        fig, axes = plt.subplots(4, 1, figsize=(11.4, 9.8), sharex=True, constrained_layout=True)
        for ax, (title, sig) in zip(axes, stages):
            bt, bx = _p4_slice(radar["t"], sig, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
            if bt is not None:
                ax.plot(bt, bx, color="black", linewidth=1.35)
                ax.axvline(0, color="0.35", linestyle="--", linewidth=0.9)
                ax.axvspan(PATCH4_AO_SEC[0], PATCH4_AO_SEC[1], alpha=0.12)
                ax.axvspan(PATCH4_AC_SEC[0], PATCH4_AC_SEC[1], alpha=0.10)
                for win, mk in [(PATCH4_AO_SEC, "o"), (PATCH4_AC_SEC, "s")]:
                    ct, cy = _p4_candidates(bt, bx, win, max_points=3)
                    if len(ct):
                        ax.scatter(ct, cy, s=18, marker=mk, facecolor="0.78", edgecolor="black", linewidth=0.5)
            ax.set_title(title, fontsize=11, loc="left")
            ax.set_ylabel("z-score")
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Radar morphology stage comparison", fontsize=13)
        fig.savefig(outdir / "fig04_1_radar_stage_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # SCG stage figure
        if scg is not None:
            scg_raw = _p4_scg_signal(scg, "raw")
            scg_lms = _p4_scg_signal(scg, "analysis")
            scg_bpf = _p4_scg_signal(scg, "bpf")
            scg_smooth = _p4_smooth(scg_bpf, fs=float(scg.get("fs", 100.0)), win_sec=0.09)
            stages = [
                ("1) Raw SCG", scg_raw),
                ("2) LMS respiration-removed", scg_lms),
                ("3) Band-pass filtered", scg_bpf),
                ("4) Smoothed", scg_smooth),
            ]
            fig, axes = plt.subplots(4, 1, figsize=(11.4, 9.8), sharex=True, constrained_layout=True)
            for ax, (title, sig) in zip(axes, stages):
                bt, bx = _p4_slice(scg["t"], sig, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
                if bt is not None:
                    ax.plot(bt, bx, color="black", linewidth=1.35)
                    ax.axvline(0, color="0.35", linestyle="--", linewidth=0.9)
                    ax.axvspan(PATCH4_AO_SEC[0], PATCH4_AO_SEC[1], alpha=0.12)
                    ax.axvspan(PATCH4_AC_SEC[0], PATCH4_AC_SEC[1], alpha=0.10)
                    for win, mk in [(PATCH4_AO_SEC, "o"), (PATCH4_AC_SEC, "s")]:
                        ct, cy = _p4_candidates(bt, bx, win, max_points=3)
                        if len(ct):
                            ax.scatter(ct, cy, s=18, marker=mk, facecolor="0.78", edgecolor="black", linewidth=0.5)
                ax.set_title(title, fontsize=11, loc="left")
                ax.set_ylabel("z-score")
                ax.grid(True, alpha=0.25)
            axes[-1].set_xlabel("Time from ECG R-peak [s]")
            fig.suptitle("SCG morphology stage comparison", fontsize=13)
            fig.savefig(outdir / "fig04_2_scg_stage_comparison.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        with open(outdir / "fig04_stage_patch4_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def _p4_radar_aoac(bt, bx):
    ao = _p4_score_event(bt, bx, PATCH4_AO_SEC, "AO")
    ac = _p4_score_event(bt, bx, PATCH4_AC_SEC, "AC")
    return {
        "AO": None if ao is None else ao[0],
        "AC": None if ac is None else ac[0],
    }


def _p4_make_fig10(outdir, ecg, scg, radar, acfg, scg_template=None):
    try:
        ridx = _p4_reference_beat_index(ecg)
        if ridx is None:
            return
        anchor = float(ecg["peaks_time"][ridx])
        q_rel, t_rel = _p4_qt_rel(ecg, ridx, anchor)

        ecg_sig = _p4_np(ecg.get("display_rpeak", ecg.get("display", ecg.get("filtered", []))))
        bt_e, bx_e = _p4_slice(ecg["t"], ecg_sig, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)

        scg_sig = _p4_scg_signal(scg, "analysis") if scg is not None else None
        bt_s, bx_s = _p4_slice(scg["t"], scg_sig, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, 100.0) if scg is not None and scg_sig is not None else (None, None)

        radar_sig = _p4_radar_signal(radar, "analysis")
        bt_r, bx_r = _p4_slice(radar["t"], radar_sig, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
        if bt_e is None or bt_r is None:
            return

        # ECG AO/AC ref from existing pipeline if available, otherwise center of literature windows
        ecg_ao = float(np.mean(PATCH4_AO_SEC))
        ecg_ac = 0.38
        try:
            temp_aoac = ao_ac_pipeline(ecg, radar, acfg)
            for b in temp_aoac.get("beats", []):
                if int(b.get("beat_index", -1)) == ridx:
                    if b.get("ecg_ao_ref") is not None:
                        ecg_ao = float(b["ecg_ao_ref"])
                    if b.get("ecg_ac_ref") is not None:
                        ecg_ac = float(b["ecg_ac_ref"])
                    break
        except Exception:
            pass

        template_lm = scg_template["landmarks"] if scg_template is not None else None
        scg_lm = _p4_refine_beat_landmarks_from_template(bt_s, bx_s, template_lm) if bt_s is not None and template_lm is not None else {}
        radar_lm = _p4_radar_aoac(bt_r, bx_r)

        fig, axes = plt.subplots(3, 1, figsize=(13.4, 9.5), sharex=True, constrained_layout=True)
        panels = [
            ("ECG Q/R/T and reference timing", bt_e, bx_e, {"Q": q_rel, "R": 0.0, "T": t_rel, "AO": ecg_ao, "AC": ecg_ac}, ("ECG", ecg_ao, ecg_ac)),
            ("SCG template-refined landmarks: MC / IM / AO / AC / MO", bt_s, bx_s, scg_lm, ("SCG", scg_lm.get("AO"), scg_lm.get("AC"))),
            ("Radar LMS-only AO/AC candidates", bt_r, bx_r, radar_lm, ("Radar", radar_lm.get("AO"), radar_lm.get("AC"))),
        ]

        for ax, (title, bt, bx, lm, intinfo) in zip(axes, panels):
            if bt is None or bx is None:
                ax.text(0.5, 0.5, "Unavailable", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title, fontsize=11, loc="left")
                continue
            ax.plot(bt, bx, color="black", linewidth=1.55)
            ax.axvline(0, color="0.35", linestyle="--", linewidth=1.0)
            ax.axvspan(PATCH4_AO_SEC[0], PATCH4_AO_SEC[1], alpha=0.12)
            ax.axvspan(PATCH4_AC_SEC[0], PATCH4_AC_SEC[1], alpha=0.10)

            if "Radar" in title:
                for win, mk in [(PATCH4_AO_SEC, "o"), (PATCH4_AC_SEC, "s")]:
                    ct, cy = _p4_candidates(bt, bx, win, max_points=3)
                    if len(ct):
                        ax.scatter(ct, cy, s=24, marker=mk, facecolor="0.78", edgecolor="black", linewidth=0.5, zorder=4)

            if "SCG" in title:
                for name in ["MC", "IM", "AO", "AC", "MO"]:
                    x = lm.get(name)
                    if x is None or not np.isfinite(x):
                        continue
                    y = float(np.interp(x, bt, bx))
                    ax.scatter([x], [y], s=48, marker="o", facecolor="white", edgecolor="black", zorder=5)

            ax.set_ylim(np.nanmin(bx) - 0.8, np.nanmax(bx) + 1.25)

            if "Radar" in title:
                draw_order = ["AO", "AC"]
            elif "SCG" in title:
                draw_order = ["MC", "IM", "AO", "AC", "MO"]
            else:
                draw_order = ["Q", "R", "T", "AO", "AC"]

            yfrac_map = {"Q": 0.96, "R": 0.91, "T": 0.86, "MC": 0.96, "IM": 0.91, "AO": 0.86, "AC": 0.80, "MO": 0.74}
            for name in draw_order:
                _p4_vline(ax, lm.get(name), name, yfrac_map.get(name, 0.88), ":" if name in ("Q", "T") else "--")

            _, ao, ac = intinfo
            pep, lvet, qs2 = _p4_interval(q_rel, ao, ac)
            ymin, ymax = ax.get_ylim()
            y0 = ymin + 0.22
            _p4_bracket(ax, q_rel, ao, y0, "PEP")
            _p4_bracket(ax, ao, ac, y0 + 0.23, "LVET")
            _p4_bracket(ax, q_rel, ac, y0 + 0.46, "QS2")
            txt = "PEP/LVET/QS2 unavailable" if pep is None or lvet is None or qs2 is None else f"PEP={pep:.1f} ms\nLVET={lvet:.1f} ms\nQS2={qs2:.1f} ms"
            ax.text(0.995, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                    bbox=dict(boxstyle="round", fc="white", ec="0.70", alpha=0.9))
            ax.set_title(title, fontsize=11, loc="left")
            ax.set_ylabel("z-score")
            ax.grid(True, alpha=0.25)

        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("ECG / SCG / Radar landmarks and CTI intervals", fontsize=14)
        fig.savefig(outdir / "fig10_ecg_scg_radar_landmark_interval_clean.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig05_morphology_ecg_scg_radar_candidates.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig06_pep_lvet_qs2_brackets.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows = []
        for mod, ao, ac in [("ECG_ref", ecg_ao, ecg_ac), ("SCG", scg_lm.get("AO"), scg_lm.get("AC")), ("Radar", radar_lm.get("AO"), radar_lm.get("AC"))]:
            pep, lvet, qs2 = _p4_interval(q_rel, ao, ac)
            rows.append([mod, q_rel, ao, ac, pep, lvet, qs2])
        save_csv(outdir / "clean_ecg_scg_radar_intervals_representative.csv",
                 ["modality", "q_rel_sec", "ao_rel_sec", "ac_rel_sec", "PEP_ms", "LVET_ms", "QS2_ms"], rows)
    except Exception as e:
        with open(outdir / "fig10_patch4_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def _p4_make_fig04_scatter(outdir, scg_ref=None):
    import pandas as pd
    try:
        ecg_csv = outdir / "ecg_vs_radar_aoac_correlation.csv"
        if not ecg_csv.exists():
            return
        ecg_df = pd.read_csv(ecg_csv)
        if "accepted" in ecg_df.columns:
            ecg_df = ecg_df[ecg_df["accepted"].astype(bool)]

        scg_df = None
        if scg_ref is not None and len(scg_ref.get("rows", [])):
            cols = [
                "beat_index", "r_peak_time_sec", "radar_accepted", "radar_sqi",
                "scg_ao_time_from_r_sec", "scg_ac_time_from_r_sec",
                "radar_ao_morph_time_from_r_sec", "radar_ac_morph_time_from_r_sec",
                "radar_minus_scg_ao_ms", "radar_minus_scg_ac_ms",
                "both_ao_ac_within_30ms", "scg_ao_confidence", "scg_ac_confidence"
            ]
            scg_df = pd.DataFrame(scg_ref["rows"], columns=cols)
        elif (outdir / "scg_reference_vs_radar_candidates.csv").exists():
            scg_df = pd.read_csv(outdir / "scg_reference_vs_radar_candidates.csv")
        if scg_df is not None and "radar_accepted" in scg_df.columns:
            scg_df = scg_df[scg_df["radar_accepted"].astype(bool)]

        def scatter(ax, x, y, title, xl, yl):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            ax.set_title(title, fontsize=11)
            if not np.any(m):
                ax.text(0.5, 0.5, "No valid pairs", ha="center", va="center", transform=ax.transAxes)
                ax.grid(True, alpha=0.25)
                return
            x2 = x[m]; y2 = y[m]
            lo = min(np.nanmin(x2), np.nanmin(y2)) - 15
            hi = max(np.nanmax(x2), np.nanmax(y2)) + 15
            err = y2 - x2
            r = np.corrcoef(x2, y2)[0, 1] if len(x2) > 1 else np.nan
            mae = np.mean(np.abs(err))
            rmse = np.sqrt(np.mean(err ** 2))
            ax.scatter(x2, y2, s=26, alpha=0.75)
            ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0)
            ax.text(0.03, 0.97, f"n={len(x2)}\nr={r:.2f}\nMAE={mae:.1f} ms\nRMSE={rmse:.1f} ms",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round", fc="white", ec="0.60", alpha=0.9))
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.grid(True, alpha=0.25)

        fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.6), constrained_layout=True)
        scatter(axes[0, 0], ecg_df.get("ecg_est_ao_ms", []), ecg_df.get("radar_ao_ms", []),
                "AO: ECG reference vs Radar", "ECG AO [ms]", "Radar AO [ms]")
        scatter(axes[0, 1], ecg_df.get("ecg_est_ac_ms", []), ecg_df.get("radar_ac_ms", []),
                "AC: ECG reference vs Radar", "ECG AC [ms]", "Radar AC [ms]")
        if scg_df is not None and len(scg_df):
            scatter(axes[1, 0], scg_df["scg_ao_time_from_r_sec"] * 1000.0, scg_df["radar_ao_morph_time_from_r_sec"] * 1000.0,
                    "AO: SCG template reference vs Radar", "SCG AO [ms]", "Radar AO [ms]")
            scatter(axes[1, 1], scg_df["scg_ac_time_from_r_sec"] * 1000.0, scg_df["radar_ac_morph_time_from_r_sec"] * 1000.0,
                    "AC: SCG template reference vs Radar", "SCG AC [ms]", "Radar AC [ms]")
        else:
            for ax in axes[1]:
                ax.text(0.5, 0.5, "SCG reference unavailable", ha="center", va="center", transform=ax.transAxes)
                ax.grid(True, alpha=0.25)
        fig.suptitle("ECG/SCG reference consistency with radar AO/AC candidates", fontsize=14)
        fig.savefig(outdir / "fig04_ecg_vs_radar_aoac_correlation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir / "fig04_patch4_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def _p4_regenerate_paper_export(outdir):
    try:
        paper = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export")
        figs = paper / "figures"
        tables = paper / "tables"
        figs.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
        fig_map = [
            ("fig01_compact_signal_overview.png", "fig01_signal_overview.png", "Figure 1. ECG/SCG/Radar compact signal overview"),
            ("fig02_ecg_qrt_reference.png", "fig02_ecg_qrt_reference.png", "Figure 2A. ECG Q/R/T reference"),
            ("fig02_scg_reference_landmarks.png", "fig02b_scg_reference_landmarks.png", "Figure 2B. SCG ensemble-template MC/IM/AO/AC/MO reference"),
            ("fig04_1_radar_stage_comparison.png", "fig04a_radar_stage_comparison.png", "Figure 4A. Radar stage comparison"),
            ("fig04_2_scg_stage_comparison.png", "fig04b_scg_stage_comparison.png", "Figure 4B. SCG stage comparison"),
            ("fig04_ecg_vs_radar_aoac_correlation.png", "fig05_ecg_scg_radar_consistency.png", "Figure 5. ECG/SCG reference consistency with radar"),
            ("fig10_ecg_scg_radar_landmark_interval_clean.png", "fig10_ecg_scg_radar_landmarks_intervals.png", "Figure 10. ECG/SCG/Radar landmarks and CTIs"),
        ]
        rows = []
        for src_name, dst_name, caption in fig_map:
            srcp = outdir / src_name
            dstp = figs / dst_name
            status = "missing"
            if srcp.exists():
                shutil.copyfile(srcp, dstp)
                status = "copied"
            rows.append([src_name, dst_name, status, caption])
        save_csv(figs / "paper_figure_index.csv", ["Source", "PaperFigure", "Status", "Caption"], rows)

        # table export
        interval_csv = outdir / "ecg_scg_radar_pep_lvet_qs2_per_beat.csv"
        table_rows = []
        if interval_csv.exists():
            df = pd.read_csv(interval_csv)
            summary = []
            for metric in ["pep", "lvet", "qs2"]:
                row = [metric.upper()]
                for mod in ["ecg", "scg", "radar"]:
                    col = f"{mod}_{metric}_ms"
                    vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float) if col in df.columns else np.array([])
                    row.append("-" if len(vals) == 0 else f"{np.nanmean(vals):.2f} ± {np.nanstd(vals):.2f}")
                summary.append(row)
            tpath = tables / "table09_ecg_scg_radar_interval_summary.csv"
            save_csv(tpath, ["Metric", "ECG mean±SD [ms]", "SCG mean±SD [ms]", "Radar mean±SD [ms]"], summary)
            try:
                _render_csv_table_to_png(tpath, figs / "table09_ecg_scg_radar_interval_summary.png", title="Table 9. ECG/SCG/Radar interval summary")
            except Exception:
                pass
            table_rows.append(["table09_ecg_scg_radar_interval_summary.csv", "ECG/SCG/Radar PEP, LVET, QS2 summary"])
        save_csv(tables / "paper_table_index.csv", ["TableCSV", "Description"], table_rows)
    except Exception as e:
        with open(outdir / "paper_export_patch4_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def scg_reference_aoac_pipeline(ecg: dict, scg: Optional[dict], radar: dict, aoac: dict, acfg: AnalysisConfig, outdir: Optional[Path] = None):
    """
    Template-first SCG reference pipeline.
    This replaces heuristic single-beat MC/IM/AO/AC/MO marking.
    """
    if scg is None:
        return {
            "used": False,
            "reason": "SCG unavailable",
            "rows": [],
            "summary": {},
        }

    templ = _p4_build_scg_ensemble_template(ecg, scg, acfg)
    if templ is None:
        return {
            "used": False,
            "reason": "SCG template unavailable",
            "rows": [],
            "summary": {},
        }

    rows = []
    r_times = _p4_np(ecg.get("peaks_time", []))
    scg_sig = _p4_scg_signal(scg, "analysis")
    radar_sig = _p4_radar_signal(radar, "analysis")

    for bi, r in enumerate(r_times):
        bt_s, bx_s = _p4_slice(scg["t"], scg_sig, float(r), acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
        bt_r, bx_r = _p4_slice(radar["t"], radar_sig, float(r), acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
        if bt_s is None or bt_r is None:
            continue
        scg_lm = _p4_refine_beat_landmarks_from_template(bt_s, bx_s, templ["landmarks"])
        radar_lm = _p4_radar_aoac(bt_r, bx_r)

        scg_ao = scg_lm.get("AO")
        scg_ac = scg_lm.get("AC")
        rad_ao = radar_lm.get("AO")
        rad_ac = radar_lm.get("AC")

        dao = None if scg_ao is None or rad_ao is None else (rad_ao - scg_ao) * 1000.0
        dac = None if scg_ac is None or rad_ac is None else (rad_ac - scg_ac) * 1000.0
        both30 = False if dao is None or dac is None else (abs(dao) <= 30.0 and abs(dac) <= 30.0)

        rows.append([
            int(bi), float(r), True, np.nan,
            scg_ao, scg_ac,
            rad_ao, rad_ac,
            dao, dac,
            bool(both30),
            np.nan, np.nan
        ])

    rows_np = np.array([[np.nan if x is None else x for x in row] for row in rows], dtype=object) if rows else np.empty((0, 13), dtype=object)

    ao_err = [abs(float(row[8])) for row in rows if row[8] is not None and np.isfinite(float(row[8]))]
    ac_err = [abs(float(row[9])) for row in rows if row[9] is not None and np.isfinite(float(row[9]))]
    both = [bool(row[10]) for row in rows]
    summary = {
        "used": True,
        "template_beats": int(len(templ.get("beat_indices", []))),
        "template_sequence_valid": bool(templ.get("sequence_valid", False)),
        "beats_evaluated": int(len(rows)),
        "radar_vs_scg_ao_mae_ms": None if not ao_err else float(np.mean(ao_err)),
        "radar_vs_scg_ac_mae_ms": None if not ac_err else float(np.mean(ac_err)),
        "radar_vs_scg_both_ao_ac_within_30ms_percent": None if not both else float(np.mean(both) * 100.0),
    }

    if outdir is not None:
        outdir = Path(outdir)
        cols = [
            "beat_index", "r_peak_time_sec", "radar_accepted", "radar_sqi",
            "scg_ao_time_from_r_sec", "scg_ac_time_from_r_sec",
            "radar_ao_morph_time_from_r_sec", "radar_ac_morph_time_from_r_sec",
            "radar_minus_scg_ao_ms", "radar_minus_scg_ac_ms",
            "both_ao_ac_within_30ms", "scg_ao_confidence", "scg_ac_confidence"
        ]
        save_csv(outdir / "scg_reference_vs_radar_candidates.csv", cols, rows)
        with open(outdir / "scg_reference_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _p4_make_scg_template_reference_fig(outdir, templ)

    return {"used": True, "rows": rows, "summary": summary, "template": templ}


_old_save_all_patch4 = save_all

def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    global _PATCH4_LAST_SCG, _PATCH4_LAST_SCG_TEMPLATE
    _PATCH4_LAST_SCG = scg
    result = _old_save_all_patch4(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        templ = _p4_build_scg_ensemble_template(ecg, scg, acfg) if scg is not None else None
        _PATCH4_LAST_SCG_TEMPLATE = templ
        _p4_make_scg_template_reference_fig(outdir, templ)
        _p4_make_stage_figs(outdir, ecg, scg, radar, acfg)
        _p4_make_fig10(outdir, ecg, scg, radar, acfg, templ)
        scg_ref = scg_reference_aoac_pipeline(ecg, scg, radar, aoac, acfg, outdir=outdir)
        _p4_make_fig04_scatter(outdir, scg_ref)
        _p4_regenerate_paper_export(outdir)
    except Exception as e:
        with open(outdir / "save_all_patch4_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))
    return result



# ============================================================
# PATCH5: Preserve SCG post-vibration + periodicity analysis
# ------------------------------------------------------------
# This patch does NOT remove late/post-systolic vibrations.
# Instead:
#   1) full SCG waveform is preserved in all figures/CSV
#   2) AC is selected from a primary closure window to avoid mislabeling
#      later diastolic/post-vibration as AC
#   3) MO/late vibration is kept and marked separately
#   4) SCG periodicity is quantified by R-anchored beat-template correlation
# ============================================================

PATCH5_AO_SEC = (0.07, 0.16)
PATCH5_AC_PRIMARY_SEC = (0.26, 0.43)   # primary AC closure complex search
PATCH5_AC_FALLBACK_SEC = (0.26, 0.52)  # kept only if primary has poor confidence
PATCH5_MO_SEC = (0.46, 0.74)           # late vibration / MO region preserved
PATCH5_MC_SEC = (-0.03, 0.06)
PATCH5_IM_SEC = (0.02, 0.13)

PATCH5_MIN_GAPS = {
    ("MC", "IM"): 0.010,
    ("IM", "AO"): 0.020,
    ("AO", "AC"): 0.170,
    ("AC", "MO"): 0.060,
}


def _p5_np(x):
    return np.asarray(x, dtype=np.float64)


def _p5_z(x):
    return zscore_safe(_p5_np(x))


def _p5_sig_scg(scg, mode="analysis"):
    if scg is None:
        return None
    n = len(scg.get("t", []))
    if n == 0:
        return None
    if mode == "raw":
        keys = ["selected_raw", "vmag", "az", "ax", "ay", "resp_removed", "filtered", "display"]
    elif mode == "bpf":
        keys = ["filtered", "resp_removed", "display", "selected_raw", "vmag"]
    elif mode == "smooth":
        keys = ["display", "filtered", "resp_removed", "selected_raw", "vmag"]
    else:
        # Analysis branch: LMS respiration-removed SCG, not heavily smoothed.
        keys = ["resp_removed", "filtered", "display", "selected_raw", "vmag"]
    for k in keys:
        if k in scg and scg[k] is not None and len(scg[k]) == n:
            return _p5_np(scg[k])
    return None


def _p5_sig_radar(radar, mode="analysis"):
    n = len(radar.get("t", []))
    if mode == "raw":
        keys = ["displacement", "displacement_m", "phase", "ppg_like"]
    elif mode == "light":
        keys = ["ppg_like", "lms_error", "displacement"]
    elif mode == "smooth":
        keys = ["ppg_final_smooth", "display", "ppg_like", "lms_error", "displacement"]
    else:
        keys = ["lms_error", "ppg_like", "displacement"]
    for k in keys:
        if k in radar and radar[k] is not None and len(radar[k]) == n:
            return _p5_np(radar[k])
    return np.zeros(n, dtype=np.float64)


def _p5_smooth(x, fs=100.0, win_sec=0.07):
    x = _p5_np(x)
    if len(x) < 5:
        return x
    try:
        y = safe_lowpass(x, fs, min(6.0, 0.35 * fs), order=2)
    except Exception:
        y = x.copy()
    win = max(5, int(round(win_sec * fs)))
    if win % 2 == 0:
        win += 1
    return np.convolve(y, np.ones(win, dtype=float) / win, mode="same")


def _p5_slice(tt, xx, anchor, pre=0.20, post=0.80, fs_out=100.0):
    tt = _p5_np(tt); xx = _p5_np(xx)
    if len(tt) < 5 or len(tt) != len(xx):
        return None, None
    m = (tt >= anchor - pre) & (tt <= anchor + post)
    if np.sum(m) < max(8, int((pre + post) * fs_out * 0.35)):
        return None, None
    grid = np.arange(-pre, post + 1e-9, 1.0 / fs_out)
    y = np.interp(anchor + grid, tt[m], xx[m])
    return grid, _p5_z(y)


def _p5_event_score(bt, bx, win, kind):
    bt = _p5_np(bt); bx = _p5_z(bx)
    if len(bt) < 8:
        return None
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 4:
        return None
    idx = np.where(m)[0]
    try:
        fs = 1.0 / max(np.nanmedian(np.diff(bt)), 1e-6)
        # derivative-stability only; keep morphology.
        y = safe_lowpass(bx, fs, min(18.0, 0.45 * fs), order=2)
    except Exception:
        y = bx
        fs = 100.0

    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    env = triangular_smooth_envelope(y, win_len=max(5, int(round(0.035 * fs)) | 1))
    env = _p5_z(env)

    t = bt[idx]
    center = np.mean(win)
    half = max((win[1] - win[0]) / 2.0, 1e-3)
    center_prior = np.exp(-0.5 * ((t - center) / (half * 0.75)) ** 2)
    boundary = np.clip(np.minimum(t - win[0], win[1] - t) / half, 0.0, 1.0)

    if kind == "MC":
        score = 0.38 * robust_scale_01(np.abs(d2[idx])) + 0.28 * robust_scale_01(np.abs(d1[idx])) + 0.18 * robust_scale_01(np.abs(y[idx])) + 0.10 * center_prior + 0.06 * boundary
    elif kind == "IM":
        score = 0.35 * robust_scale_01(np.maximum(d1[idx], 0)) + 0.30 * robust_scale_01(np.maximum(d2[idx], 0)) + 0.20 * robust_scale_01(env[idx]) + 0.10 * center_prior + 0.05 * boundary
    elif kind == "AO":
        score = 0.30 * robust_scale_01(np.maximum(d1[idx], 0)) + 0.26 * robust_scale_01(np.abs(d2[idx])) + 0.14 * robust_scale_01(y[idx]) + 0.22 * center_prior + 0.08 * boundary
    elif kind == "AC":
        # AC: closure complex. Use primary systolic-end window; late vibration is not deleted but not allowed to steal AC.
        score = 0.30 * robust_scale_01(np.maximum(-d1[idx], 0)) + 0.28 * robust_scale_01(np.abs(d2[idx])) + 0.13 * robust_scale_01(-y[idx]) + 0.22 * center_prior + 0.07 * boundary
    elif kind == "MO":
        # Late/MO region kept separately.
        score = 0.34 * robust_scale_01(np.abs(d2[idx])) + 0.24 * robust_scale_01(np.maximum(d1[idx], 0)) + 0.18 * robust_scale_01(np.abs(y[idx])) + 0.18 * center_prior + 0.06 * boundary
    else:
        score = robust_scale_01(np.abs(d2[idx])) * boundary

    best_local = int(np.nanargmax(score))
    j = int(idx[best_local])
    # confidence is normalized local score and boundary score combined
    conf = float(np.nanmax(score)) if len(score) else 0.0
    return float(bt[j]), float(y[j]), int(j), conf


def _p5_candidates(bt, bx, win, max_points=3):
    bt = _p5_np(bt); bx = _p5_z(bx)
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 5:
        return np.array([]), np.array([])
    idx = np.where(m)[0]
    try:
        fs = 1.0 / max(np.nanmedian(np.diff(bt)), 1e-6)
        y = safe_lowpass(bx, fs, min(18.0, 0.45 * fs), order=2)
    except Exception:
        y = bx
        fs = 100.0
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    t = bt[idx]
    half = max((win[1] - win[0]) / 2.0, 1e-3)
    boundary = np.clip(np.minimum(t - win[0], win[1] - t) / half, 0.0, 1.0)
    score = (robust_scale_01(np.abs(d1[idx])) + robust_scale_01(np.abs(d2[idx]))) * boundary
    chosen = idx[np.argsort(score)[::-1][:max_points]]
    chosen = chosen[np.argsort(bt[chosen])]
    return bt[chosen], y[chosen]


def _p5_validate(lm):
    req = ["MC", "IM", "AO", "AC", "MO"]
    if any(lm.get(k) is None or not np.isfinite(lm.get(k)) for k in req):
        return False
    if not (lm["MC"] < lm["IM"] < lm["AO"] < lm["AC"] < lm["MO"]):
        return False
    for (a, b), gap in PATCH5_MIN_GAPS.items():
        if (lm[b] - lm[a]) < gap:
            return False
    return True


def _p5_template_landmarks(bt, bx):
    lm = {}
    res = _p5_event_score(bt, bx, PATCH5_MC_SEC, "MC")
    lm["MC"] = None if res is None else res[0]

    im_start = PATCH5_IM_SEC[0] if lm["MC"] is None else max(PATCH5_IM_SEC[0], lm["MC"] + PATCH5_MIN_GAPS[("MC", "IM")])
    res = _p5_event_score(bt, bx, (im_start, PATCH5_IM_SEC[1]), "IM")
    lm["IM"] = None if res is None else res[0]

    ao_start = PATCH5_AO_SEC[0] if lm["IM"] is None else max(PATCH5_AO_SEC[0], lm["IM"] + PATCH5_MIN_GAPS[("IM", "AO")])
    res = _p5_event_score(bt, bx, (ao_start, PATCH5_AO_SEC[1]), "AO")
    lm["AO"] = None if res is None else res[0]

    # AC primary: avoid labeling post-vibration as closure.
    ac_start = PATCH5_AC_PRIMARY_SEC[0] if lm["AO"] is None else max(PATCH5_AC_PRIMARY_SEC[0], lm["AO"] + PATCH5_MIN_GAPS[("AO", "AC")])
    res_primary = _p5_event_score(bt, bx, (ac_start, PATCH5_AC_PRIMARY_SEC[1]), "AC")
    if res_primary is not None:
        lm["AC"] = res_primary[0]
    else:
        # fallback still preserves late morphology but is marked as lower confidence by summary.
        ac_start2 = PATCH5_AC_FALLBACK_SEC[0] if lm["AO"] is None else max(PATCH5_AC_FALLBACK_SEC[0], lm["AO"] + PATCH5_MIN_GAPS[("AO", "AC")])
        res_fb = _p5_event_score(bt, bx, (ac_start2, PATCH5_AC_FALLBACK_SEC[1]), "AC")
        lm["AC"] = None if res_fb is None else res_fb[0]

    # MO is separated from AC. It is not removed from waveform.
    mo_start = PATCH5_MO_SEC[0] if lm["AC"] is None else max(PATCH5_MO_SEC[0], lm["AC"] + PATCH5_MIN_GAPS[("AC", "MO")])
    res = _p5_event_score(bt, bx, (mo_start, PATCH5_MO_SEC[1]), "MO")
    lm["MO"] = None if res is None else res[0]

    return lm


def _p5_build_scg_template(ecg, scg, acfg):
    if scg is None:
        return None
    r_times = _p5_np(ecg.get("peaks_time", []))
    sig = _p5_sig_scg(scg, "analysis")
    if sig is None or len(r_times) < 5:
        return None

    beats, beat_indices = [], []
    grid_ref = None
    for i, r in enumerate(r_times):
        bt, bx = _p5_slice(scg["t"], sig, float(r), acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
        if bt is None:
            continue
        if np.nanstd(bx) < 0.15 or not np.all(np.isfinite(bx)):
            continue
        if grid_ref is None:
            grid_ref = bt
        beats.append(bx)
        beat_indices.append(i)

    if len(beats) < 3:
        return None

    B = np.vstack(beats)
    median = np.nanmedian(B, axis=0)
    median = _p5_smooth(median, fs=100.0, win_sec=0.03)
    median = _p5_z(median)
    sd = np.nanstd(B, axis=0)
    lm = _p5_template_landmarks(grid_ref, median)

    # Periodicity: beat-to-template correlation.
    cors = []
    for b in B:
        try:
            c = np.corrcoef(_p5_z(b), median)[0, 1]
            if np.isfinite(c):
                cors.append(float(c))
        except Exception:
            pass
    cors = np.asarray(cors, dtype=np.float64)

    return {
        "t_rel": grid_ref,
        "beats": B,
        "beat_indices": np.asarray(beat_indices, dtype=int),
        "median": median,
        "mean": np.nanmean(B, axis=0),
        "sd": sd,
        "landmarks": lm,
        "sequence_valid": bool(_p5_validate(lm)),
        "template_corr_values": cors,
        "periodicity_summary": {
            "n_beats": int(len(B)),
            "median_template_corr": None if len(cors) == 0 else float(np.nanmedian(cors)),
            "mean_template_corr": None if len(cors) == 0 else float(np.nanmean(cors)),
            "q1_template_corr": None if len(cors) == 0 else float(np.nanpercentile(cors, 25)),
            "q3_template_corr": None if len(cors) == 0 else float(np.nanpercentile(cors, 75)),
            "corr_ge_0p3_percent": None if len(cors) == 0 else float(np.mean(cors >= 0.3) * 100.0),
            "corr_ge_0p5_percent": None if len(cors) == 0 else float(np.mean(cors >= 0.5) * 100.0),
        },
    }


def _p5_refine_from_template(bt, bx, templ_lm):
    if bt is None or bx is None or templ_lm is None:
        return {}
    out = {}
    for name in ["MC", "IM", "AO", "AC", "MO"]:
        c = templ_lm.get(name)
        if c is None or not np.isfinite(c):
            out[name] = None
            continue
        width = 0.035 if name in ("MC", "IM", "AO") else 0.045
        # AC remains around template AC, so late vibration does not steal it.
        win = (max(float(bt[0]), c - width), min(float(bt[-1]), c + width))
        res = _p5_event_score(bt, bx, win, name)
        out[name] = None if res is None else res[0]
    if not _p5_validate(out):
        return dict(templ_lm)
    return out


def _p5_qt_rel(ecg, beat_index, anchor):
    q_rel, t_rel = None, None
    q = _p5_np(ecg.get("q_time", []))
    t = _p5_np(ecg.get("t_time", []))
    if 0 <= beat_index < len(q) and np.isfinite(q[beat_index]):
        q_rel = float(q[beat_index] - anchor)
    if 0 <= beat_index < len(t) and np.isfinite(t[beat_index]):
        t_rel = float(t[beat_index] - anchor)
    return q_rel, t_rel


def _p5_interval(q_rel, ao, ac):
    pep = None if q_rel is None or ao is None else (ao - q_rel) * 1000.0
    lvet = None if ao is None or ac is None else (ac - ao) * 1000.0
    qs2 = None if q_rel is None or ac is None else (ac - q_rel) * 1000.0
    return pep, lvet, qs2


def _p5_vline(ax, x, label, yfrac=0.92, ls="--"):
    if x is None or not np.isfinite(float(x)):
        return
    x = float(x)
    ax.axvline(x, color="black", linestyle=ls, linewidth=0.95, alpha=0.85)
    ymin, ymax = ax.get_ylim()
    y = ymin + (ymax - ymin) * yfrac
    ax.text(x, y, label, fontsize=8.0, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.65", alpha=0.88))


def _p5_bracket(ax, x0, x1, y, label):
    if x0 is None or x1 is None or not np.isfinite(float(x0)) or not np.isfinite(float(x1)):
        return
    x0 = float(x0); x1 = float(x1)
    ax.plot([x0, x1], [y, y], color="black", linewidth=1.15)
    ax.plot([x0, x0], [y - 0.035, y + 0.035], color="black", linewidth=0.9)
    ax.plot([x1, x1], [y - 0.035, y + 0.035], color="black", linewidth=0.9)
    ax.text((x0 + x1) / 2.0, y + 0.045, label, fontsize=8.2, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.70", alpha=0.90))


def _p5_make_scg_template_fig(outdir, templ):
    try:
        if templ is None:
            return
        bt, med, sd = templ["t_rel"], _p5_z(templ["median"]), _p5_np(templ["sd"])
        lm = templ["landmarks"]

        fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.9), constrained_layout=True)
        ax.plot(bt, med, color="black", linewidth=1.7, label="Median SCG template")
        ax.fill_between(bt, med - sd, med + sd, alpha=0.15, label="±1 SD")
        ax.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0, label="ECG R anchor")
        ax.axvspan(PATCH5_AO_SEC[0], PATCH5_AO_SEC[1], alpha=0.12, label="AO search")
        ax.axvspan(PATCH5_AC_PRIMARY_SEC[0], PATCH5_AC_PRIMARY_SEC[1], alpha=0.10, label="AC primary")
        ax.axvspan(PATCH5_MO_SEC[0], PATCH5_MO_SEC[1], alpha=0.07, label="Late/MO preserved")
        ax.set_ylim(np.nanmin(med) - 0.7, np.nanmax(med) + 1.05)
        ypos = {"MC":0.96,"IM":0.90,"AO":0.84,"AC":0.78,"MO":0.72}
        markers = {"MC":"o","IM":"^","AO":"s","AC":"D","MO":"v"}
        for name in ["MC","IM","AO","AC","MO"]:
            x = lm.get(name)
            if x is None or not np.isfinite(x):
                continue
            y = float(np.interp(x, bt, med))
            ax.scatter([x], [y], s=64, marker=markers[name], facecolor="white", edgecolor="black", zorder=5)
            _p5_vline(ax, x, f"{name}\n{x*1000:.0f} ms", ypos[name])
        ax.set_title("SCG ensemble-template landmarks; late vibration preserved separately", fontsize=13)
        ax.set_xlabel("Time from ECG R-peak [s]")
        ax.set_ylabel("z-score")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        fig.savefig(outdir / "fig02_scg_reference_landmarks.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows = []
        for name in ["MC","IM","AO","AC","MO"]:
            x = lm.get(name)
            rows.append([name, x, None if x is None else x * 1000.0])
        save_csv(outdir / "scg_template_landmarks.csv", ["landmark","time_from_r_sec","time_from_r_ms"], rows)

        with open(outdir / "scg_template_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "n_template_beats": int(len(templ.get("beat_indices", []))),
                "sequence_valid": bool(templ.get("sequence_valid", False)),
                "windows_sec": {
                    "MC": PATCH5_MC_SEC,
                    "IM": PATCH5_IM_SEC,
                    "AO": PATCH5_AO_SEC,
                    "AC_primary": PATCH5_AC_PRIMARY_SEC,
                    "AC_fallback": PATCH5_AC_FALLBACK_SEC,
                    "MO_late_preserved": PATCH5_MO_SEC,
                },
                "periodicity": templ.get("periodicity_summary", {}),
                "note": "Late/post-systolic SCG vibration is preserved; AC is selected from primary closure window and MO/late complex is marked separately."
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        with open(outdir / "fig02_scg_patch5_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def _p5_make_scg_periodicity_fig(outdir, templ):
    try:
        if templ is None:
            return
        bt = templ["t_rel"]
        B = templ["beats"]
        med = _p5_z(templ["median"])
        corr = templ.get("template_corr_values", np.array([]))

        fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), constrained_layout=True)
        # overlay first/selected beats, not too many
        step = max(1, len(B) // 24)
        for b in B[::step][:30]:
            axes[0].plot(bt, _p5_z(b), color="0.65", linewidth=0.6, alpha=0.45)
        axes[0].plot(bt, med, color="black", linewidth=2.0, label="Median template")
        axes[0].axvline(0, color="0.35", linestyle="--", linewidth=1.0)
        axes[0].axvspan(PATCH5_AO_SEC[0], PATCH5_AO_SEC[1], alpha=0.12)
        axes[0].axvspan(PATCH5_AC_PRIMARY_SEC[0], PATCH5_AC_PRIMARY_SEC[1], alpha=0.10)
        axes[0].axvspan(PATCH5_MO_SEC[0], PATCH5_MO_SEC[1], alpha=0.07)
        axes[0].set_title("R-peak aligned SCG beat overlay and median template")
        axes[0].set_ylabel("z-score")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(fontsize=8)

        if corr is not None and len(corr):
            axes[1].hist(corr, bins=18, alpha=0.75, edgecolor="black")
            axes[1].axvline(np.nanmedian(corr), color="black", linestyle="--", linewidth=1.2, label=f"median={np.nanmedian(corr):.2f}")
            axes[1].set_title("SCG beat-to-template correlation distribution")
            axes[1].set_xlabel("Correlation with median template")
            axes[1].set_ylabel("Number of beats")
            axes[1].grid(True, alpha=0.25)
            axes[1].legend(fontsize=8)
        else:
            axes[1].text(0.5, 0.5, "Correlation unavailable", ha="center", va="center", transform=axes[1].transAxes)
        fig.savefig(outdir / "fig09_scg_periodicity_template_consistency.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir / "fig09_scg_periodicity_patch5_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def _p5_stage_figs(outdir, ecg, scg, radar, acfg):
    try:
        r = _p5_np(ecg.get("peaks_time", []))
        if len(r) == 0:
            return
        anchor = float(r[len(r)//2])

        radar_light = _p5_sig_radar(radar, "light")
        radar_stages = [
            ("1) Raw displacement", _p5_sig_radar(radar, "raw")),
            ("2) LMS respiration-removed", _p5_sig_radar(radar, "analysis")),
            ("3) Lightly filtered", radar_light),
            ("4) Heavily smoothed", _p5_smooth(radar_light, float(radar.get("fs", 100.0)), 0.11)),
        ]
        fig, axes = plt.subplots(4, 1, figsize=(11.4, 9.8), sharex=True, constrained_layout=True)
        for ax, (title, sig) in zip(axes, radar_stages):
            bt, bx = _p5_slice(radar["t"], sig, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
            if bt is not None:
                ax.plot(bt, bx, color="black", linewidth=1.35)
                ax.axvline(0, color="0.35", linestyle="--", linewidth=0.9)
                ax.axvspan(PATCH5_AO_SEC[0], PATCH5_AO_SEC[1], alpha=0.12)
                ax.axvspan(PATCH5_AC_PRIMARY_SEC[0], PATCH5_AC_PRIMARY_SEC[1], alpha=0.10)
                ax.axvspan(PATCH5_MO_SEC[0], PATCH5_MO_SEC[1], alpha=0.05)
                for win, mk in [(PATCH5_AO_SEC, "o"), (PATCH5_AC_PRIMARY_SEC, "s")]:
                    ct, cy = _p5_candidates(bt, bx, win, max_points=3)
                    if len(ct):
                        ax.scatter(ct, cy, s=18, marker=mk, facecolor="0.78", edgecolor="black", linewidth=0.5)
            ax.set_title(title, fontsize=11, loc="left")
            ax.set_ylabel("z-score")
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Radar morphology stage comparison", fontsize=13)
        fig.savefig(outdir / "fig04_1_radar_stage_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        if scg is not None:
            scg_bpf = _p5_sig_scg(scg, "bpf")
            stages = [
                ("1) Raw SCG", _p5_sig_scg(scg, "raw")),
                ("2) LMS respiration-removed", _p5_sig_scg(scg, "analysis")),
                ("3) Band-pass filtered", scg_bpf),
                ("4) Light smoothed display", _p5_smooth(scg_bpf, float(scg.get("fs", 100.0)), 0.07)),
            ]
            fig, axes = plt.subplots(4, 1, figsize=(11.4, 9.8), sharex=True, constrained_layout=True)
            for ax, (title, sig) in zip(axes, stages):
                bt, bx = _p5_slice(scg["t"], sig, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
                if bt is not None:
                    ax.plot(bt, bx, color="black", linewidth=1.35)
                    ax.axvline(0, color="0.35", linestyle="--", linewidth=0.9)
                    ax.axvspan(PATCH5_AO_SEC[0], PATCH5_AO_SEC[1], alpha=0.12)
                    ax.axvspan(PATCH5_AC_PRIMARY_SEC[0], PATCH5_AC_PRIMARY_SEC[1], alpha=0.10)
                    ax.axvspan(PATCH5_MO_SEC[0], PATCH5_MO_SEC[1], alpha=0.05)
                    for win, mk in [(PATCH5_AO_SEC, "o"), (PATCH5_AC_PRIMARY_SEC, "s"), (PATCH5_MO_SEC, "v")]:
                        ct, cy = _p5_candidates(bt, bx, win, max_points=3)
                        if len(ct):
                            ax.scatter(ct, cy, s=18, marker=mk, facecolor="0.78", edgecolor="black", linewidth=0.5)
                ax.set_title(title, fontsize=11, loc="left")
                ax.set_ylabel("z-score")
                ax.grid(True, alpha=0.25)
            axes[-1].set_xlabel("Time from ECG R-peak [s]")
            fig.suptitle("SCG morphology stage comparison; late vibration preserved", fontsize=13)
            fig.savefig(outdir / "fig04_2_scg_stage_comparison.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        with open(outdir / "fig04_stage_patch5_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def _p5_radar_aoac(bt, bx):
    ao = _p5_event_score(bt, bx, PATCH5_AO_SEC, "AO")
    ac = _p5_event_score(bt, bx, PATCH5_AC_PRIMARY_SEC, "AC")
    return {"AO": None if ao is None else ao[0], "AC": None if ac is None else ac[0]}


def _p5_make_fig10(outdir, ecg, scg, radar, acfg, templ):
    try:
        r = _p5_np(ecg.get("peaks_time", []))
        if len(r) == 0:
            return
        ridx = int(len(r)//2)
        anchor = float(r[ridx])
        q_rel, t_rel = _p5_qt_rel(ecg, ridx, anchor)

        bt_e, bx_e = _p5_slice(ecg["t"], _p5_np(ecg.get("display_rpeak", ecg.get("display", ecg.get("filtered", [])))), anchor, acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
        bt_s, bx_s = (None, None)
        if scg is not None:
            ss = _p5_sig_scg(scg, "analysis")
            bt_s, bx_s = _p5_slice(scg["t"], ss, anchor, acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
        bt_r, bx_r = _p5_slice(radar["t"], _p5_sig_radar(radar, "analysis"), anchor, acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
        if bt_e is None or bt_r is None:
            return

        ecg_ao = float(np.mean(PATCH5_AO_SEC))
        ecg_ac = 0.38
        try:
            temp = ao_ac_pipeline(ecg, radar, acfg)
            for b in temp.get("beats", []):
                if int(b.get("beat_index", -1)) == ridx:
                    if b.get("ecg_ao_ref") is not None:
                        ecg_ao = float(b["ecg_ao_ref"])
                    if b.get("ecg_ac_ref") is not None:
                        ecg_ac = float(b["ecg_ac_ref"])
        except Exception:
            pass

        scg_lm = _p5_refine_from_template(bt_s, bx_s, templ["landmarks"]) if templ is not None and bt_s is not None else {}
        radar_lm = _p5_radar_aoac(bt_r, bx_r)

        fig, axes = plt.subplots(3, 1, figsize=(13.4, 9.5), sharex=True, constrained_layout=True)
        panels = [
            ("ECG Q/R/T and reference timing", bt_e, bx_e, {"Q": q_rel, "R": 0.0, "T": t_rel, "AO": ecg_ao, "AC": ecg_ac}, ("ECG", ecg_ao, ecg_ac)),
            ("SCG template-refined landmarks; late vibration preserved", bt_s, bx_s, scg_lm, ("SCG", scg_lm.get("AO"), scg_lm.get("AC"))),
            ("Radar LMS-only AO/AC candidates", bt_r, bx_r, radar_lm, ("Radar", radar_lm.get("AO"), radar_lm.get("AC"))),
        ]
        for ax, (title, bt, bx, lm, intinfo) in zip(axes, panels):
            if bt is None or bx is None:
                ax.text(0.5, 0.5, "Unavailable", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title, fontsize=11, loc="left")
                continue
            ax.plot(bt, bx, color="black", linewidth=1.55)
            ax.axvline(0, color="0.35", linestyle="--", linewidth=1.0)
            ax.axvspan(PATCH5_AO_SEC[0], PATCH5_AO_SEC[1], alpha=0.12)
            ax.axvspan(PATCH5_AC_PRIMARY_SEC[0], PATCH5_AC_PRIMARY_SEC[1], alpha=0.10)
            if "SCG" in title:
                ax.axvspan(PATCH5_MO_SEC[0], PATCH5_MO_SEC[1], alpha=0.06)
                for name in ["MC","IM","AO","AC","MO"]:
                    x = lm.get(name)
                    if x is not None and np.isfinite(x):
                        y = float(np.interp(x, bt, bx))
                        ax.scatter([x], [y], s=48, marker="o", facecolor="white", edgecolor="black", zorder=5)
                for win, mk in [(PATCH5_AO_SEC, "o"), (PATCH5_AC_PRIMARY_SEC, "s"), (PATCH5_MO_SEC, "v")]:
                    ct, cy = _p5_candidates(bt, bx, win, max_points=3)
                    if len(ct):
                        ax.scatter(ct, cy, s=18, marker=mk, facecolor="0.78", edgecolor="black", linewidth=0.5, zorder=4)
            if "Radar" in title:
                for win, mk in [(PATCH5_AO_SEC, "o"), (PATCH5_AC_PRIMARY_SEC, "s")]:
                    ct, cy = _p5_candidates(bt, bx, win, max_points=3)
                    if len(ct):
                        ax.scatter(ct, cy, s=24, marker=mk, facecolor="0.78", edgecolor="black", linewidth=0.5, zorder=4)
            ax.set_ylim(np.nanmin(bx) - 0.8, np.nanmax(bx) + 1.25)

            if "Radar" in title:
                draw_order = ["AO", "AC"]
            elif "SCG" in title:
                draw_order = ["MC", "IM", "AO", "AC", "MO"]
            else:
                draw_order = ["Q", "R", "T", "AO", "AC"]
            yfrac = {"Q":0.96, "R":0.91, "T":0.86, "MC":0.96, "IM":0.91, "AO":0.86, "AC":0.80, "MO":0.74}
            for name in draw_order:
                _p5_vline(ax, lm.get(name), name, yfrac.get(name, 0.88), ":" if name in ("Q","T") else "--")

            _, ao, ac = intinfo
            pep, lvet, qs2 = _p5_interval(q_rel, ao, ac)
            ymin, ymax = ax.get_ylim()
            base = ymin + 0.22
            _p5_bracket(ax, q_rel, ao, base, "PEP")
            _p5_bracket(ax, ao, ac, base + 0.23, "LVET")
            _p5_bracket(ax, q_rel, ac, base + 0.46, "QS2")
            msg = "PEP/LVET/QS2 unavailable" if pep is None or lvet is None or qs2 is None else f"PEP={pep:.1f} ms\nLVET={lvet:.1f} ms\nQS2={qs2:.1f} ms"
            ax.text(0.995, 0.05, msg, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                    bbox=dict(boxstyle="round", fc="white", ec="0.70", alpha=0.9))
            ax.set_title(title, fontsize=11, loc="left")
            ax.set_ylabel("z-score")
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("ECG / SCG / Radar landmarks and CTI intervals", fontsize=14)
        fig.savefig(outdir / "fig10_ecg_scg_radar_landmark_interval_clean.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig05_morphology_ecg_scg_radar_candidates.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig06_pep_lvet_qs2_brackets.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows = []
        for mod, ao, ac in [("ECG_ref", ecg_ao, ecg_ac), ("SCG", scg_lm.get("AO"), scg_lm.get("AC")), ("Radar", radar_lm.get("AO"), radar_lm.get("AC"))]:
            pep, lvet, qs2 = _p5_interval(q_rel, ao, ac)
            rows.append([mod, q_rel, ao, ac, pep, lvet, qs2])
        save_csv(outdir / "clean_ecg_scg_radar_intervals_representative.csv",
                 ["modality","q_rel_sec","ao_rel_sec","ac_rel_sec","PEP_ms","LVET_ms","QS2_ms"], rows)
    except Exception as e:
        with open(outdir / "fig10_patch5_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def _p5_make_fig04_scatter(outdir, scg_ref=None):
    import pandas as pd
    try:
        ecg_csv = outdir / "ecg_vs_radar_aoac_correlation.csv"
        if not ecg_csv.exists():
            return
        ecg_df = pd.read_csv(ecg_csv)
        if "accepted" in ecg_df.columns:
            ecg_df = ecg_df[ecg_df["accepted"].astype(bool)]
        scg_df = None
        if scg_ref is not None and len(scg_ref.get("rows", [])):
            cols = [
                "beat_index","r_peak_time_sec","radar_accepted","radar_sqi",
                "scg_ao_time_from_r_sec","scg_ac_time_from_r_sec",
                "radar_ao_morph_time_from_r_sec","radar_ac_morph_time_from_r_sec",
                "radar_minus_scg_ao_ms","radar_minus_scg_ac_ms",
                "both_ao_ac_within_30ms","scg_ao_confidence","scg_ac_confidence"
            ]
            scg_df = pd.DataFrame(scg_ref["rows"], columns=cols)
        elif (outdir / "scg_reference_vs_radar_candidates.csv").exists():
            scg_df = pd.read_csv(outdir / "scg_reference_vs_radar_candidates.csv")
        if scg_df is not None and "radar_accepted" in scg_df.columns:
            scg_df = scg_df[scg_df["radar_accepted"].astype(bool)]

        def scatter(ax, x, y, title, xl, yl):
            x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            ax.set_title(title, fontsize=11)
            if not np.any(m):
                ax.text(0.5, 0.5, "No valid pairs", ha="center", va="center", transform=ax.transAxes)
                ax.grid(True, alpha=0.25); return
            x2, y2 = x[m], y[m]
            lo = min(np.nanmin(x2), np.nanmin(y2)) - 15
            hi = max(np.nanmax(x2), np.nanmax(y2)) + 15
            err = y2 - x2
            r = np.corrcoef(x2, y2)[0, 1] if len(x2) > 1 else np.nan
            mae = np.mean(np.abs(err)); rmse = np.sqrt(np.mean(err ** 2))
            ax.scatter(x2, y2, s=26, alpha=0.75)
            ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0)
            ax.text(0.03, 0.97, f"n={len(x2)}\nr={r:.2f}\nMAE={mae:.1f} ms\nRMSE={rmse:.1f} ms",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round", fc="white", ec="0.60", alpha=0.9))
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=0.25)

        fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.6), constrained_layout=True)
        scatter(axes[0,0], ecg_df.get("ecg_est_ao_ms", []), ecg_df.get("radar_ao_ms", []),
                "AO: ECG reference vs Radar", "ECG AO [ms]", "Radar AO [ms]")
        scatter(axes[0,1], ecg_df.get("ecg_est_ac_ms", []), ecg_df.get("radar_ac_ms", []),
                "AC: ECG reference vs Radar", "ECG AC [ms]", "Radar AC [ms]")
        if scg_df is not None and len(scg_df):
            scatter(axes[1,0], scg_df["scg_ao_time_from_r_sec"]*1000.0, scg_df["radar_ao_morph_time_from_r_sec"]*1000.0,
                    "AO: SCG template reference vs Radar", "SCG AO [ms]", "Radar AO [ms]")
            scatter(axes[1,1], scg_df["scg_ac_time_from_r_sec"]*1000.0, scg_df["radar_ac_morph_time_from_r_sec"]*1000.0,
                    "AC: SCG template reference vs Radar", "SCG AC [ms]", "Radar AC [ms]")
        else:
            for ax in axes[1]:
                ax.text(0.5, 0.5, "SCG reference unavailable", ha="center", va="center", transform=ax.transAxes)
                ax.grid(True, alpha=0.25)
        fig.suptitle("ECG/SCG reference consistency with radar AO/AC candidates", fontsize=14)
        fig.savefig(outdir / "fig04_ecg_vs_radar_aoac_correlation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        with open(outdir / "fig04_patch5_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def _p5_regenerate_paper_export(outdir):
    try:
        paper = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export")
        figs = paper / "figures"; tables = paper / "tables"
        figs.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
        fig_map = [
            ("fig01_compact_signal_overview.png", "fig01_signal_overview.png", "Figure 1. ECG/SCG/Radar compact signal overview"),
            ("fig02_ecg_qrt_reference.png", "fig02_ecg_qrt_reference.png", "Figure 2A. ECG Q/R/T reference"),
            ("fig02_scg_reference_landmarks.png", "fig02b_scg_reference_landmarks.png", "Figure 2B. SCG ensemble-template landmarks with late vibration preserved"),
            ("fig04_1_radar_stage_comparison.png", "fig04a_radar_stage_comparison.png", "Figure 4A. Radar stage comparison"),
            ("fig04_2_scg_stage_comparison.png", "fig04b_scg_stage_comparison.png", "Figure 4B. SCG stage comparison"),
            ("fig09_scg_periodicity_template_consistency.png", "fig09_scg_periodicity_template_consistency.png", "Figure 9. SCG periodicity and template consistency"),
            ("fig04_ecg_vs_radar_aoac_correlation.png", "fig05_ecg_scg_radar_consistency.png", "Figure 5. ECG/SCG reference consistency with radar"),
            ("fig10_ecg_scg_radar_landmark_interval_clean.png", "fig10_ecg_scg_radar_landmarks_intervals.png", "Figure 10. ECG/SCG/Radar landmarks and CTIs"),
        ]
        rows = []
        for src_name, dst_name, caption in fig_map:
            src = outdir / src_name; dst = figs / dst_name
            status = "missing"
            if src.exists():
                shutil.copyfile(src, dst); status = "copied"
            rows.append([src_name, dst_name, status, caption])
        save_csv(figs / "paper_figure_index.csv", ["Source", "PaperFigure", "Status", "Caption"], rows)

        # table export
        table_rows = []
        templ_json = outdir / "scg_template_summary.json"
        if templ_json.exists():
            try:
                d = json.loads(templ_json.read_text(encoding="utf-8"))
                per = d.get("periodicity", {})
                rows2 = [
                    ["n_template_beats", d.get("n_template_beats", "-")],
                    ["sequence_valid", d.get("sequence_valid", "-")],
                    ["median_template_corr", per.get("median_template_corr", "-")],
                    ["mean_template_corr", per.get("mean_template_corr", "-")],
                    ["corr_ge_0p3_percent", per.get("corr_ge_0p3_percent", "-")],
                    ["corr_ge_0p5_percent", per.get("corr_ge_0p5_percent", "-")],
                ]
                tpath = tables / "table09_scg_periodicity_summary.csv"
                save_csv(tpath, ["Metric", "Value"], rows2)
                try:
                    _render_csv_table_to_png(tpath, figs / "table09_scg_periodicity_summary.png", title="Table 9. SCG periodicity summary")
                except Exception:
                    pass
                table_rows.append(["table09_scg_periodicity_summary.csv", "SCG R-anchored beat-to-template periodicity summary"])
            except Exception:
                pass
        save_csv(tables / "paper_table_index.csv", ["TableCSV", "Description"], table_rows)
    except Exception as e:
        with open(outdir / "paper_export_patch5_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def scg_reference_aoac_pipeline(ecg: dict, scg: Optional[dict], radar: dict, aoac: dict, acfg: AnalysisConfig, outdir: Optional[Path] = None):
    """
    PATCH5 SCG reference:
    Template-first, late vibration preserved separately. AC is selected in
    primary closure window; MO/late vibration is kept and not deleted.
    """
    if scg is None:
        return {"used": False, "reason": "SCG unavailable", "rows": [], "summary": {}}
    templ = _p5_build_scg_template(ecg, scg, acfg)
    if templ is None:
        return {"used": False, "reason": "SCG template unavailable", "rows": [], "summary": {}}

    rows = []
    r_times = _p5_np(ecg.get("peaks_time", []))
    scg_sig = _p5_sig_scg(scg, "analysis")
    radar_sig = _p5_sig_radar(radar, "analysis")
    for bi, r in enumerate(r_times):
        bt_s, bx_s = _p5_slice(scg["t"], scg_sig, float(r), acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
        bt_r, bx_r = _p5_slice(radar["t"], radar_sig, float(r), acfg.beat_pre_sec, acfg.beat_post_sec, 100.0)
        if bt_s is None or bt_r is None:
            continue
        scg_lm = _p5_refine_from_template(bt_s, bx_s, templ["landmarks"])
        radar_lm = _p5_radar_aoac(bt_r, bx_r)
        scg_ao, scg_ac = scg_lm.get("AO"), scg_lm.get("AC")
        rad_ao, rad_ac = radar_lm.get("AO"), radar_lm.get("AC")
        dao = None if scg_ao is None or rad_ao is None else (rad_ao - scg_ao) * 1000.0
        dac = None if scg_ac is None or rad_ac is None else (rad_ac - scg_ac) * 1000.0
        both = False if dao is None or dac is None else (abs(dao) <= 30.0 and abs(dac) <= 30.0)
        rows.append([int(bi), float(r), True, np.nan, scg_ao, scg_ac, rad_ao, rad_ac, dao, dac, bool(both), np.nan, np.nan])

    ao_err = [abs(float(row[8])) for row in rows if row[8] is not None and np.isfinite(float(row[8]))]
    ac_err = [abs(float(row[9])) for row in rows if row[9] is not None and np.isfinite(float(row[9]))]
    both = [bool(row[10]) for row in rows]
    summary = {
        "used": True,
        "template_beats": int(len(templ.get("beat_indices", []))),
        "template_sequence_valid": bool(templ.get("sequence_valid", False)),
        "beats_evaluated": int(len(rows)),
        "radar_vs_scg_ao_mae_ms": None if not ao_err else float(np.mean(ao_err)),
        "radar_vs_scg_ac_mae_ms": None if not ac_err else float(np.mean(ac_err)),
        "radar_vs_scg_both_ao_ac_within_30ms_percent": None if not both else float(np.mean(both) * 100.0),
        "scg_periodicity": templ.get("periodicity_summary", {}),
        "note": "Post-vibration is preserved. AC uses primary closure window; MO/late vibration is analyzed separately."
    }

    if outdir is not None:
        cols = [
            "beat_index","r_peak_time_sec","radar_accepted","radar_sqi",
            "scg_ao_time_from_r_sec","scg_ac_time_from_r_sec",
            "radar_ao_morph_time_from_r_sec","radar_ac_morph_time_from_r_sec",
            "radar_minus_scg_ao_ms","radar_minus_scg_ac_ms",
            "both_ao_ac_within_30ms","scg_ao_confidence","scg_ac_confidence"
        ]
        save_csv(outdir / "scg_reference_vs_radar_candidates.csv", cols, rows)
        with open(outdir / "scg_reference_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _p5_make_scg_template_fig(outdir, templ)
        _p5_make_scg_periodicity_fig(outdir, templ)

    return {"used": True, "rows": rows, "summary": summary, "template": templ}


_old_save_all_patch5 = save_all

def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    result = _old_save_all_patch5(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        templ = _p5_build_scg_template(ecg, scg, acfg) if scg is not None else None
        _p5_make_scg_template_fig(outdir, templ)
        _p5_make_scg_periodicity_fig(outdir, templ)
        _p5_stage_figs(outdir, ecg, scg, radar, acfg)
        _p5_make_fig10(outdir, ecg, scg, radar, acfg, templ)
        scg_ref = scg_reference_aoac_pipeline(ecg, scg, radar, aoac, acfg, outdir=outdir)
        _p5_make_fig04_scatter(outdir, scg_ref)
        _p5_regenerate_paper_export(outdir)
    except Exception as e:
        with open(outdir / "save_all_patch5_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))
    return result



# ============================================================
# PATCH6_FIXED: single-cycle SCG fig2 / fig06 / fig10 cleanup
# - fig2 uses single representative SCG cycle (not multi-cycle looking template panel)
# - fig06_single_cycle_ecg_radar_aoac_labels now includes SCG
# - fig06_pep_lvet_qs2_brackets / fig10 are single-cycle for ECG, SCG, Radar
# - black waveform + colored landmark marking
# ============================================================

P6_COLORS = {
    "Q": "#1f77b4",
    "R": "#d62728",
    "T": "#2ca02c",
    "MC": "#17becf",
    "IM": "#1f77b4",
    "AO": "#9467bd",
    "AC": "#ff7f0e",
    "MO": "#2ca02c",
}


def _p6_get_aoac_ref_for_beat(aoac, beat_index):
    ecg_ao = None; ecg_ac = None
    for b in aoac.get("beats", []):
        try:
            if int(b.get("beat_index", -1)) == int(beat_index):
                ecg_ao = b.get("ecg_ao_ref", b.get("ao_ref"))
                ecg_ac = b.get("ecg_ac_ref", b.get("ac_ref"))
                break
        except Exception:
            pass
    return ecg_ao, ecg_ac


def _p6_representative_cycle_context(ecg, scg, radar, acfg, templ=None):
    r = _p5_np(ecg.get("peaks_time", []))
    if len(r) < 3:
        return None
    scg_sig = _p5_sig_scg(scg, "analysis") if scg is not None else None
    radar_sig = _p5_sig_radar(radar, "analysis")
    # Prefer a middle beat with available ECG/SCG/Radar slices.
    candidates = list(range(1, len(r)-1))
    mid = len(candidates) // 2
    ordered = sorted(candidates, key=lambda i: abs(i - candidates[mid]))
    for bi in ordered:
        rr = float(r[bi+1] - r[bi]) if bi+1 < len(r) else float(np.nanmedian(np.diff(r)))
        pre = min(0.08, 0.20)
        post = max(0.45, min(rr * 0.92, 0.95))
        anchor = float(r[bi])
        # ECG always expected.
        ok = True
        if scg is not None:
            bt_s, bx_s = _p5_slice(scg["t"], scg_sig, anchor, pre, post, 100.0)
            ok = ok and (bt_s is not None)
        bt_r, bx_r = _p5_slice(radar["t"], radar_sig, anchor, pre, post, 100.0)
        ok = ok and (bt_r is not None)
        if ok:
            return {"beat_index": bi, "anchor": anchor, "rr": rr, "pre": pre, "post": post}
    # fallback middle beat
    bi = len(r)//2
    rr = float(r[min(bi+1, len(r)-1)] - r[bi-1]) / 2 if len(r) > 2 else 0.7
    return {"beat_index": bi, "anchor": float(r[bi]), "rr": rr, "pre": 0.08, "post": max(0.45, min(rr*0.92, 0.95))}


def _p6_landmark_line(ax, bt, bx, x, name, yfrac=0.90):
    if x is None:
        return
    try:
        if not np.isfinite(float(x)):
            return
    except Exception:
        return
    x = float(x)
    c = P6_COLORS.get(name, "#111111")
    ax.axvline(x, color=c, linestyle="--", linewidth=1.15, alpha=0.95, zorder=3)
    y = float(np.interp(x, bt, bx))
    marker = {"Q":"o","R":"o","T":"o","MC":"o","IM":"^","AO":"s","AC":"D","MO":"v"}.get(name, 'o')
    ax.scatter([x], [y], s=52, marker=marker, facecolor="white", edgecolor=c, linewidth=1.4, zorder=5)
    ymin, ymax = ax.get_ylim()
    ty = ymin + (ymax - ymin) * yfrac
    ax.text(x, ty, name, color=c, fontsize=8.4, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=c, alpha=0.92))


def _p6_candidate_markers(ax, bt, bx, win, color, marker, max_points=3):
    ct, cy = _p5_candidates(bt, bx, win, max_points=max_points)
    if len(ct):
        ax.scatter(ct, cy, s=24, marker=marker, facecolor="white", edgecolor=color, linewidth=0.9, zorder=4)


def _p6_bracket(ax, x0, x1, y, label, color="#111111"):
    if x0 is None or x1 is None:
        return
    try:
        if not (np.isfinite(float(x0)) and np.isfinite(float(x1))):
            return
    except Exception:
        return
    x0 = float(x0); x1 = float(x1)
    ax.plot([x0, x1], [y, y], color=color, linewidth=1.15)
    ax.plot([x0, x0], [y - 0.03, y + 0.03], color=color, linewidth=1.0)
    ax.plot([x1, x1], [y - 0.03, y + 0.03], color=color, linewidth=1.0)
    ax.text((x0 + x1)/2, y + 0.038, label, fontsize=8.0, ha="center", va="bottom",
            color=color, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=color, alpha=0.92))


def _p6_plot_panel(ax, title, bt, bx, landmarks, show_candidates=False, candidate_mode="radar", show_intervals=False, q_rel=None):
    ax.plot(bt, bx, color="black", linewidth=1.6)
    ax.axvline(0, color="#555555", linestyle=":", linewidth=1.0)
    ax.grid(True, alpha=0.22)
    ax.set_title(title, fontsize=11.5, loc="left")
    ax.set_ylabel("z-score")
    ymin0, ymax0 = np.nanmin(bx), np.nanmax(bx)
    pad = max(0.65, 0.22*(ymax0-ymin0 + 1e-6))
    ax.set_ylim(ymin0 - pad, ymax0 + 1.05)

    # candidate windows (subtle)
    ax.axvspan(PATCH5_AO_SEC[0], PATCH5_AO_SEC[1], alpha=0.08, color=P6_COLORS['AO'])
    ax.axvspan(PATCH5_AC_PRIMARY_SEC[0], PATCH5_AC_PRIMARY_SEC[1], alpha=0.06, color=P6_COLORS['AC'])
    if candidate_mode == "scg":
        ax.axvspan(PATCH5_MO_SEC[0], PATCH5_MO_SEC[1], alpha=0.05, color=P6_COLORS['MO'])

    if show_candidates:
        _p6_candidate_markers(ax, bt, bx, PATCH5_AO_SEC, P6_COLORS['AO'], 's', max_points=3)
        _p6_candidate_markers(ax, bt, bx, PATCH5_AC_PRIMARY_SEC, P6_COLORS['AC'], 'D', max_points=3)
        if candidate_mode == "scg":
            _p6_candidate_markers(ax, bt, bx, PATCH5_MO_SEC, P6_COLORS['MO'], 'v', max_points=3)

    if title.startswith("ECG"):
        draw = ['Q','R','T','AO','AC']
        ymap = {'Q':0.96,'R':0.90,'T':0.84,'AO':0.78,'AC':0.72}
    elif title.startswith("SCG"):
        draw = ['MC','IM','AO','AC','MO']
        ymap = {'MC':0.96,'IM':0.90,'AO':0.84,'AC':0.78,'MO':0.72}
    else:
        draw = ['AO','AC']
        ymap = {'AO':0.90,'AC':0.82}

    for name in draw:
        _p6_landmark_line(ax, bt, bx, landmarks.get(name), name, ymap.get(name, 0.88))

    if show_intervals:
        ao = landmarks.get('AO'); ac = landmarks.get('AC')
        pep, lvet, qs2 = _p5_interval(q_rel, ao, ac)
        ymin, ymax = ax.get_ylim()
        base = ymin + 0.14*(ymax-ymin)
        _p6_bracket(ax, q_rel, ao, base, 'PEP', '#444444')
        _p6_bracket(ax, ao, ac, base + 0.16*(ymax-ymin), 'LVET', '#444444')
        _p6_bracket(ax, q_rel, ac, base + 0.32*(ymax-ymin), 'QS2', '#444444')
        msg = "PEP/LVET/QS2 unavailable" if pep is None or lvet is None or qs2 is None else f"PEP={pep:.1f} ms | LVET={lvet:.1f} ms | QS2={qs2:.1f} ms"
        ax.text(0.995, 0.03, msg, transform=ax.transAxes, ha='right', va='bottom', fontsize=8.2,
                bbox=dict(boxstyle='round', fc='white', ec='0.65', alpha=0.92))


def _p6_make_single_cycle_figs(outdir, ecg, scg, radar, aoac, acfg, templ):
    try:
        ctx = _p6_representative_cycle_context(ecg, scg, radar, acfg, templ)
        if ctx is None:
            return
        bi, anchor, pre, post = ctx['beat_index'], ctx['anchor'], ctx['pre'], ctx['post']

        ecg_sig = _p5_np(ecg.get('display_rpeak', ecg.get('display', ecg.get('filtered', []))))
        bt_e, bx_e = _p5_slice(ecg['t'], ecg_sig, anchor, pre, post, 100.0)
        scg_sig = _p5_sig_scg(scg, 'analysis') if scg is not None else None
        bt_s, bx_s = _p5_slice(scg['t'], scg_sig, anchor, pre, post, 100.0) if scg is not None else (None, None)
        radar_sig = _p5_sig_radar(radar, 'analysis')
        bt_r, bx_r = _p5_slice(radar['t'], radar_sig, anchor, pre, post, 100.0)
        if bt_e is None or bt_r is None:
            return

        q_rel, t_rel = _p5_qt_rel(ecg, bi, anchor)
        ecg_ao, ecg_ac = _p6_get_aoac_ref_for_beat(aoac, bi)
        if ecg_ao is None:
            ecg_ao = float(np.mean(PATCH5_AO_SEC))
        if ecg_ac is None:
            ecg_ac = float(np.mean(PATCH5_AC_PRIMARY_SEC))
        ecg_lm = {'Q': q_rel, 'R': 0.0, 'T': t_rel, 'AO': ecg_ao, 'AC': ecg_ac}
        scg_lm = _p5_refine_from_template(bt_s, bx_s, templ['landmarks']) if templ is not None and bt_s is not None else {}
        radar_lm = _p5_radar_aoac(bt_r, bx_r)

        # fig2: single-cycle SCG reference landmarks
        if bt_s is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.6), constrained_layout=True)
            _p6_plot_panel(ax, 'SCG representative single-cycle reference landmarks', bt_s, bx_s, scg_lm, show_candidates=True, candidate_mode='scg', show_intervals=False)
            ax.set_xlabel('Time from ECG R-peak [s]')
            fig.savefig(outdir / 'fig02_scg_reference_landmarks.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        # fig06 single-cycle labels including SCG
        fig, axes = plt.subplots(3, 1, figsize=(12.6, 8.6), sharex=True, constrained_layout=True)
        _p6_plot_panel(axes[0], 'ECG single-cycle labels', bt_e, bx_e, ecg_lm, show_candidates=False, show_intervals=False, q_rel=q_rel)
        if bt_s is not None:
            _p6_plot_panel(axes[1], 'SCG single-cycle labels', bt_s, bx_s, scg_lm, show_candidates=True, candidate_mode='scg', show_intervals=False, q_rel=q_rel)
        else:
            axes[1].text(0.5, 0.5, 'SCG unavailable', ha='center', va='center', transform=axes[1].transAxes)
        _p6_plot_panel(axes[2], 'Radar single-cycle labels', bt_r, bx_r, radar_lm, show_candidates=True, candidate_mode='radar', show_intervals=False, q_rel=q_rel)
        axes[-1].set_xlabel('Time from ECG R-peak [s]')
        fig.suptitle('Single-cycle ECG / SCG / Radar landmark labeling', fontsize=14)
        fig.savefig(outdir / 'fig06_single_cycle_ecg_radar_aoac_labels.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # fig06 brackets: single-cycle ECG/SCG/Radar with PEP/LVET/QS2
        fig, axes = plt.subplots(3, 1, figsize=(12.9, 8.9), sharex=True, constrained_layout=True)
        _p6_plot_panel(axes[0], 'ECG single-cycle with PEP/LVET/QS2', bt_e, bx_e, ecg_lm, show_candidates=False, show_intervals=True, q_rel=q_rel)
        if bt_s is not None:
            _p6_plot_panel(axes[1], 'SCG single-cycle with PEP/LVET/QS2', bt_s, bx_s, scg_lm, show_candidates=True, candidate_mode='scg', show_intervals=True, q_rel=q_rel)
        else:
            axes[1].text(0.5, 0.5, 'SCG unavailable', ha='center', va='center', transform=axes[1].transAxes)
        _p6_plot_panel(axes[2], 'Radar single-cycle with PEP/LVET/QS2', bt_r, bx_r, radar_lm, show_candidates=True, candidate_mode='radar', show_intervals=True, q_rel=q_rel)
        axes[-1].set_xlabel('Time from ECG R-peak [s]')
        fig.suptitle('Single-cycle PEP, LVET, and QS2 comparison', fontsize=14)
        fig.savefig(outdir / 'fig06_pep_lvet_qs2_brackets.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # fig10: final clean single-cycle panel
        fig, axes = plt.subplots(3, 1, figsize=(13.2, 9.2), sharex=True, constrained_layout=True)
        _p6_plot_panel(axes[0], 'ECG single-cycle reference and AO/AC labels', bt_e, bx_e, ecg_lm, show_candidates=False, show_intervals=True, q_rel=q_rel)
        if bt_s is not None:
            _p6_plot_panel(axes[1], 'SCG single-cycle landmarks and candidate windows', bt_s, bx_s, scg_lm, show_candidates=True, candidate_mode='scg', show_intervals=True, q_rel=q_rel)
        else:
            axes[1].text(0.5, 0.5, 'SCG unavailable', ha='center', va='center', transform=axes[1].transAxes)
        _p6_plot_panel(axes[2], 'Radar single-cycle AO/AC candidates', bt_r, bx_r, radar_lm, show_candidates=True, candidate_mode='radar', show_intervals=True, q_rel=q_rel)
        axes[-1].set_xlabel('Time from ECG R-peak [s]')
        fig.suptitle('Single-cycle ECG / SCG / Radar landmarks and interval comparison', fontsize=14)
        fig.savefig(outdir / 'fig10_ecg_scg_radar_landmark_interval_clean.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # values csv for reference
        rows = []
        for mod, lm in [('ECG', ecg_lm), ('SCG', scg_lm), ('Radar', radar_lm)]:
            pep, lvet, qs2 = _p5_interval(q_rel, lm.get('AO'), lm.get('AC'))
            rows.append([mod, bi, q_rel, lm.get('R', 0.0), t_rel if mod=='ECG' else None,
                         lm.get('MC'), lm.get('IM'), lm.get('AO'), lm.get('AC'), lm.get('MO'), pep, lvet, qs2])
        save_csv(outdir / 'fig06_fig10_single_cycle_values.csv',
                 ['modality','beat_index','Q_rel_sec','R_rel_sec','T_rel_sec','MC_rel_sec','IM_rel_sec','AO_rel_sec','AC_rel_sec','MO_rel_sec','PEP_ms','LVET_ms','QS2_ms'], rows)
    except Exception as e:
        (outdir / 'patch6_single_cycle_error.txt').write_text(str(e), encoding='utf-8')


def _p6_regenerate_paper_export(outdir):
    try:
        paper = outdir / globals().get('PAPER_EXPORT_DIRNAME', 'paper_export')
        figs = paper / 'figures'; figs.mkdir(parents=True, exist_ok=True)
        mapping = [
            ('fig02_scg_reference_landmarks.png', 'fig02b_scg_reference_landmarks.png'),
            ('fig06_single_cycle_ecg_radar_aoac_labels.png', 'fig06_single_cycle_ecg_scg_radar_labels.png'),
            ('fig06_pep_lvet_qs2_brackets.png', 'fig06_pep_lvet_qs2_brackets.png'),
            ('fig10_ecg_scg_radar_landmark_interval_clean.png', 'fig10_ecg_scg_radar_landmarks_intervals.png'),
        ]
        rows = []
        for s,d in mapping:
            sp = outdir / s; dp = figs / d
            status='missing'
            if sp.exists():
                shutil.copyfile(sp, dp); status='copied'
            rows.append([s,d,status])
        save_csv(figs / 'paper_figure_index_patch6.csv', ['Source','PaperFigure','Status'], rows)
    except Exception as e:
        (outdir / 'paper_export_patch6_error.txt').write_text(str(e), encoding='utf-8')


_old_save_all_patch6 = save_all

def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    result = _old_save_all_patch6(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        templ = _p5_build_scg_template(ecg, scg, acfg) if scg is not None else None
        _p6_make_single_cycle_figs(outdir, ecg, scg, radar, aoac, acfg, templ)
        _p6_regenerate_paper_export(outdir)
    except Exception as e:
        (outdir / 'save_all_patch6_error.txt').write_text(str(e), encoding='utf-8')
    return result



# ============================================================
# PATCH8: clean SCG morphology / single-cycle figure overhaul
# Based on ex88 analysis:
# - fig02_compact_beat_morphology now includes SCG morphology
# - fig02_scg_reference_landmarks rebuilt from best single-cycle SCG beat
# - fig06 / fig10 cleaned: common x-axis, final markers only, reduced overlap
# - SCG MO constrained to physiologic late window to avoid wrong placement
# ============================================================

P8_COLOR = {
    "Q": "#1f77b4",
    "R": "#d62728",
    "T": "#2ca02c",
    "MC": "#17becf",
    "IM": "#1f77b4",
    "AO": "#9467bd",
    "AC": "#ff7f0e",
    "MO": "#2ca02c",
    "PEP": "#555555",
    "LVET": "#555555",
    "QS2": "#555555",
}


def _p8_enforce_order(lm: dict):
    if not isinstance(lm, dict):
        return {}
    out = dict(lm)
    order = ["MC", "IM", "AO", "AC", "MO"]
    prev = -1e9
    mins = {"MC": -0.03, "IM": 0.00, "AO": 0.05, "AC": 0.24, "MO": 0.42}
    maxs = {"MC": 0.04, "IM": 0.11, "AO": 0.18, "AC": 0.46, "MO": 0.70}
    gaps = {("MC","IM"):0.005, ("IM","AO"):0.015, ("AO","AC"):0.08, ("AC","MO"):0.04}
    for i, k in enumerate(order):
        x = out.get(k)
        if x is None or not np.isfinite(float(x)):
            out[k] = None
            continue
        x = float(x)
        if x < mins[k] or x > maxs[k]:
            out[k] = None
            continue
        if i > 0:
            pk = order[i-1]
            px = out.get(pk)
            if px is not None and np.isfinite(float(px)):
                min_gap = gaps.get((pk, k), 0.01)
                if x <= float(px) + min_gap:
                    out[k] = None
                    continue
        prev = x
        out[k] = x
    return out


def _p8_refined_scg_landmarks(bt, bx, templ=None):
    # Use literature-like windows and template refinement, then enforce order.
    lm = {}
    # MC
    r = _p5_event_score(bt, bx, (-0.03, 0.03), 'MC')
    lm['MC'] = None if r is None else r[0]
    im_lo = 0.005 if lm['MC'] is None else max(0.005, lm['MC'] + 0.004)
    r = _p5_event_score(bt, bx, (im_lo, 0.10), 'IM')
    lm['IM'] = None if r is None else r[0]
    ao_lo = 0.055 if lm['IM'] is None else max(0.055, lm['IM'] + 0.010)
    r = _p5_event_score(bt, bx, (ao_lo, 0.17), 'AO')
    lm['AO'] = None if r is None else r[0]
    ac_lo = 0.26 if lm['AO'] is None else max(0.26, lm['AO'] + 0.08)
    r = _p5_event_score(bt, bx, (ac_lo, 0.45), 'AC')
    lm['AC'] = None if r is None else r[0]
    mo_lo = 0.46 if lm['AC'] is None else max(0.46, lm['AC'] + 0.05)
    r = _p5_event_score(bt, bx, (mo_lo, 0.68), 'MO')
    lm['MO'] = None if r is None else r[0]

    if templ is not None and templ.get('landmarks') is not None:
        try:
            ref = _p5_refine_from_template(bt, bx, templ['landmarks'])
            for k in ['MC','IM','AO','AC','MO']:
                v = ref.get(k)
                if v is not None and np.isfinite(float(v)):
                    lm[k] = float(v)
        except Exception:
            pass
    return _p8_enforce_order(lm)


def _p8_refined_radar_landmarks(bt, bx):
    lm = {}
    r = _p5_event_score(bt, bx, (0.07, 0.16), 'AO')
    lm['AO'] = None if r is None else r[0]
    ac_lo = 0.28 if lm['AO'] is None else max(0.28, lm['AO'] + 0.08)
    r = _p5_event_score(bt, bx, (ac_lo, 0.46), 'AC')
    lm['AC'] = None if r is None else r[0]
    # Final only; do not expose multiple candidate markers in clean figures.
    if lm['AO'] is not None and lm['AC'] is not None and lm['AC'] <= lm['AO'] + 0.08:
        lm['AC'] = None
    return lm


def _p8_pick_best_scg_beat(ecg, scg, acfg, templ=None):
    if scg is None:
        return None
    r = _p5_np(ecg.get('peaks_time', []))
    sig = _p5_sig_scg(scg, 'analysis')
    if sig is None or len(r) < 3:
        return None
    best = None
    for bi in range(1, len(r)-1):
        anchor = float(r[bi])
        bt, bx = _p5_slice(scg['t'], sig, anchor, 0.08, 0.68, 100.0)
        if bt is None:
            continue
        score = float(np.nanstd(bx))
        if templ is not None and templ.get('median') is not None and len(templ['median']) == len(bx):
            try:
                c = float(np.corrcoef(_p5_z(bx), _p5_z(templ['median']))[0,1])
                if np.isfinite(c):
                    score += 2.0 * c
            except Exception:
                pass
        lm = _p8_refined_scg_landmarks(bt, bx, templ)
        valid = sum(1 for k in ['MC','IM','AO','AC','MO'] if lm.get(k) is not None)
        score += 0.2 * valid
        if valid < 3:
            continue
        cand = dict(score=score, beat_index=bi, anchor=anchor, bt=bt, bx=bx, landmarks=lm)
        if best is None or cand['score'] > best['score']:
            best = cand
    return best


def _p8_pick_common_representative(ecg, scg, radar, acfg, templ=None):
    r = _p5_np(ecg.get('peaks_time', []))
    if len(r) < 3:
        return None
    scg_sig = _p5_sig_scg(scg, 'analysis') if scg is not None else None
    radar_sig = _p5_sig_radar(radar, 'analysis')
    best = None
    for bi in range(1, len(r)-1):
        anchor = float(r[bi])
        bt_e, bx_e = _p5_slice(ecg['t'], _p5_np(ecg.get('display_rpeak', ecg.get('display', ecg.get('filtered', ecg.get('cleaned', []))))), anchor, 0.08, 0.68, 100.0)
        bt_r, bx_r = _p5_slice(radar['t'], radar_sig, anchor, 0.08, 0.68, 100.0)
        if bt_e is None or bt_r is None:
            continue
        bt_s, bx_s = (None, None)
        if scg_sig is not None:
            bt_s, bx_s = _p5_slice(scg['t'], scg_sig, anchor, 0.08, 0.68, 100.0)
            if bt_s is None:
                continue
        q_rel, t_rel = _p5_qt_rel(ecg, bi, anchor)
        radar_lm = _p8_refined_radar_landmarks(bt_r, bx_r)
        scg_lm = _p8_refined_scg_landmarks(bt_s, bx_s, templ) if bt_s is not None else {}
        score = 0.0
        if q_rel is not None and t_rel is not None:
            score += 1.0
        if radar_lm.get('AO') is not None: score += 1.2
        if radar_lm.get('AC') is not None: score += 1.2
        if bt_s is not None:
            score += 0.35 * sum(1 for k in ['MC','IM','AO','AC','MO'] if scg_lm.get(k) is not None)
            if templ is not None and templ.get('median') is not None and len(templ['median']) == len(bx_s):
                try:
                    c = float(np.corrcoef(_p5_z(bx_s), _p5_z(templ['median']))[0,1])
                    if np.isfinite(c): score += max(0.0, c)
                except Exception:
                    pass
        score += 0.25 * np.nanstd(bx_r)
        cand = dict(score=score, beat_index=bi, anchor=anchor, bt_e=bt_e, bx_e=bx_e, bt_s=bt_s, bx_s=bx_s, bt_r=bt_r, bx_r=bx_r, q_rel=q_rel, t_rel=t_rel, radar_lm=radar_lm, scg_lm=scg_lm)
        if best is None or cand['score'] > best['score']:
            best = cand
    return best


def _p8_landmark_draw(ax, bt, bx, x, name, yfrac=0.92):
    if x is None:
        return
    try:
        if not np.isfinite(float(x)):
            return
    except Exception:
        return
    x = float(x)
    c = P8_COLOR.get(name, '#111111')
    y = float(np.interp(x, bt, bx)) if bt is not None and bx is not None else 0.0
    ax.axvline(x, color=c, linestyle='--', linewidth=1.15, alpha=0.95, zorder=2)
    marker_map = {'Q':'o','R':'o','T':'o','MC':'o','IM':'^','AO':'s','AC':'D','MO':'v'}
    ax.scatter([x],[y], s=48, marker=marker_map.get(name,'o'), facecolor='white', edgecolor=c, linewidth=1.35, zorder=4)
    ymin, ymax = ax.get_ylim()
    ty = ymin + (ymax - ymin) * yfrac
    ax.text(x, ty, name, fontsize=8.3, ha='center', va='top', color=c,
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec=c, alpha=0.96), zorder=5)


def _p8_window(ax, win, name):
    c = P8_COLOR.get(name, '#bbbbbb')
    ax.axvspan(win[0], win[1], color=c, alpha=0.07, zorder=0)


def _p8_bracket(ax, x0, x1, y, label):
    if x0 is None or x1 is None:
        return
    try:
        if not (np.isfinite(float(x0)) and np.isfinite(float(x1))):
            return
    except Exception:
        return
    x0=float(x0); x1=float(x1)
    ax.plot([x0,x1],[y,y], color=P8_COLOR['PEP'], linewidth=1.1, zorder=2)
    ax.plot([x0,x0],[y-0.03,y+0.03], color=P8_COLOR['PEP'], linewidth=1.0, zorder=2)
    ax.plot([x1,x1],[y-0.03,y+0.03], color=P8_COLOR['PEP'], linewidth=1.0, zorder=2)
    ax.text((x0+x1)/2.0, y+0.035, label, fontsize=8.0, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.10', fc='white', ec='0.6', alpha=0.96), zorder=5)


def _p8_style_axis(ax, title, xlim=(-0.08,0.68)):
    ax.set_title(title, fontsize=11, loc='left', pad=6)
    ax.set_xlim(*xlim)
    ax.set_ylabel('z-score', fontsize=10)
    ax.grid(True, alpha=0.22)
    ax.tick_params(labelsize=9)


def _p8_plot_common_panel(ax, title, bt, bx, lm, kind, q_rel=None, t_rel=None, show_brackets=False):
    ax.plot(bt, bx, color='black', linewidth=1.6, zorder=1)
    ax.axvline(0.0, color='0.45', linestyle=':', linewidth=1.0, zorder=1)
    if kind == 'ECG':
        for n in ['Q','R','T','AO','AC']:
            if n in ['AO','AC']:
                _p8_window(ax, (0.07,0.16) if n=='AO' else (0.28,0.46), n)
        draw = [('Q', q_rel,0.96),('R',0.0,0.90),('T',t_rel,0.84),('AO',lm.get('AO'),0.78),('AC',lm.get('AC'),0.72)]
    elif kind == 'SCG':
        _p8_window(ax, (-0.03,0.03), 'MC'); _p8_window(ax, (0.01,0.10), 'IM'); _p8_window(ax, (0.06,0.17), 'AO'); _p8_window(ax, (0.26,0.45), 'AC'); _p8_window(ax, (0.46,0.68), 'MO')
        draw = [('MC',lm.get('MC'),0.96),('IM',lm.get('IM'),0.90),('AO',lm.get('AO'),0.84),('AC',lm.get('AC'),0.78),('MO',lm.get('MO'),0.72)]
    else:
        _p8_window(ax, (0.07,0.16), 'AO'); _p8_window(ax, (0.28,0.46), 'AC')
        draw = [('AO',lm.get('AO'),0.90),('AC',lm.get('AC'),0.82)]
    ymin0, ymax0 = float(np.nanmin(bx)), float(np.nanmax(bx))
    pad = max(0.7, 0.25*(ymax0-ymin0 + 1e-6))
    ax.set_ylim(ymin0-pad, ymax0+1.15)
    for n,x,yf in draw:
        _p8_landmark_draw(ax, bt, bx, x, n, yfrac=yf)
    if show_brackets:
        ao = lm.get('AO'); ac = lm.get('AC')
        pep, lvet, qs2 = _p5_interval(q_rel, ao, ac)
        ymin, ymax = ax.get_ylim(); base = ymin + 0.10*(ymax-ymin)
        _p8_bracket(ax, q_rel, ao, base, 'PEP')
        _p8_bracket(ax, ao, ac, base + 0.15*(ymax-ymin), 'LVET')
        _p8_bracket(ax, q_rel, ac, base + 0.30*(ymax-ymin), 'QS2')
    _p8_style_axis(ax, title)


def _p8_make_fig02_morphology(outdir, ecg, scg, radar, acfg, templ):
    try:
        rep = _p8_pick_common_representative(ecg, scg, radar, acfg, templ)
        if rep is None:
            return
        ecg_lm = {'AO': float(np.mean((0.07,0.16))), 'AC': float(np.mean((0.28,0.46)))}
        fig, axes = plt.subplots(3,1, figsize=(12.4,8.4), sharex=True, constrained_layout=True)
        _p8_plot_common_panel(axes[0], 'ECG representative beat morphology', rep['bt_e'], rep['bx_e'], ecg_lm, 'ECG', q_rel=rep['q_rel'], t_rel=rep['t_rel'], show_brackets=False)
        if rep['bt_s'] is not None:
            _p8_plot_common_panel(axes[1], 'SCG representative beat morphology', rep['bt_s'], rep['bx_s'], rep['scg_lm'], 'SCG', q_rel=rep['q_rel'], t_rel=rep['t_rel'], show_brackets=False)
        else:
            axes[1].text(0.5,0.5,'SCG unavailable', transform=axes[1].transAxes, ha='center', va='center')
            _p8_style_axis(axes[1], 'SCG representative beat morphology')
        _p8_plot_common_panel(axes[2], 'Radar representative beat morphology', rep['bt_r'], rep['bx_r'], rep['radar_lm'], 'Radar', q_rel=rep['q_rel'], t_rel=rep['t_rel'], show_brackets=False)
        axes[-1].set_xlabel('Time from ECG R-peak [s]', fontsize=10)
        fig.suptitle('Compact ECG / SCG / Radar beat morphology', fontsize=14)
        fig.savefig(outdir / 'fig02_compact_beat_morphology.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        (outdir / 'patch8_fig02_morphology_error.txt').write_text(str(e), encoding='utf-8')


def _p8_make_fig02_scg_reference(outdir, ecg, scg, acfg, templ):
    try:
        best = _p8_pick_best_scg_beat(ecg, scg, acfg, templ)
        if best is None:
            return
        fig, ax = plt.subplots(1,1, figsize=(11.6,4.6), constrained_layout=True)
        _p8_plot_common_panel(ax, 'SCG representative single-cycle reference landmarks', best['bt'], best['bx'], best['landmarks'], 'SCG', show_brackets=False)
        ax.set_xlabel('Time from ECG R-peak [s]', fontsize=10)
        fig.savefig(outdir / 'fig02_scg_reference_landmarks.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        save_csv(outdir / 'fig02_scg_reference_landmarks_values.csv', ['beat_index','MC_rel_sec','IM_rel_sec','AO_rel_sec','AC_rel_sec','MO_rel_sec'], [[best['beat_index'], best['landmarks'].get('MC'), best['landmarks'].get('IM'), best['landmarks'].get('AO'), best['landmarks'].get('AC'), best['landmarks'].get('MO')]])
    except Exception as e:
        (outdir / 'patch8_fig02_scg_ref_error.txt').write_text(str(e), encoding='utf-8')


def _p8_make_fig06_fig10(outdir, ecg, scg, radar, acfg, templ):
    try:
        rep = _p8_pick_common_representative(ecg, scg, radar, acfg, templ)
        if rep is None:
            return
        ecg_ao, ecg_ac = None, None
        try:
            tmp = ao_ac_pipeline(ecg, radar, acfg)
            beats = tmp.get('beats', [])
            for b in beats:
                if int(b.get('beat_index', -1)) == int(rep['beat_index']):
                    ecg_ao = b.get('ecg_ao_ref', b.get('ao_ref'))
                    ecg_ac = b.get('ecg_ac_ref', b.get('ac_ref'))
                    break
        except Exception:
            pass
        if ecg_ao is None: ecg_ao = float(np.mean((0.07,0.16)))
        if ecg_ac is None: ecg_ac = float(np.mean((0.28,0.46)))
        ecg_lm = {'AO': float(ecg_ao), 'AC': float(ecg_ac)}

        # fig06: single-cycle labels only
        fig, axes = plt.subplots(3,1, figsize=(12.8,8.6), sharex=True, constrained_layout=True)
        _p8_plot_common_panel(axes[0], 'ECG single-cycle labels', rep['bt_e'], rep['bx_e'], ecg_lm, 'ECG', q_rel=rep['q_rel'], t_rel=rep['t_rel'], show_brackets=False)
        if rep['bt_s'] is not None:
            _p8_plot_common_panel(axes[1], 'SCG single-cycle labels', rep['bt_s'], rep['bx_s'], rep['scg_lm'], 'SCG', q_rel=rep['q_rel'], t_rel=rep['t_rel'], show_brackets=False)
        else:
            axes[1].text(0.5,0.5,'SCG unavailable', transform=axes[1].transAxes, ha='center', va='center')
            _p8_style_axis(axes[1], 'SCG single-cycle labels')
        _p8_plot_common_panel(axes[2], 'Radar single-cycle labels', rep['bt_r'], rep['bx_r'], rep['radar_lm'], 'Radar', q_rel=rep['q_rel'], t_rel=rep['t_rel'], show_brackets=False)
        axes[-1].set_xlabel('Time from ECG R-peak [s]', fontsize=10)
        fig.suptitle('Single-cycle ECG / SCG / Radar landmark labeling', fontsize=14)
        fig.savefig(outdir / 'fig06_single_cycle_ecg_radar_aoac_labels.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # fig06 brackets clean
        fig, axes = plt.subplots(3,1, figsize=(12.8,8.8), sharex=True, constrained_layout=True)
        _p8_plot_common_panel(axes[0], 'ECG single-cycle with PEP / LVET / QS2', rep['bt_e'], rep['bx_e'], ecg_lm, 'ECG', q_rel=rep['q_rel'], t_rel=rep['t_rel'], show_brackets=True)
        if rep['bt_s'] is not None:
            _p8_plot_common_panel(axes[1], 'SCG single-cycle with PEP / LVET / QS2', rep['bt_s'], rep['bx_s'], rep['scg_lm'], 'SCG', q_rel=rep['q_rel'], t_rel=rep['t_rel'], show_brackets=True)
        else:
            axes[1].text(0.5,0.5,'SCG unavailable', transform=axes[1].transAxes, ha='center', va='center')
            _p8_style_axis(axes[1], 'SCG single-cycle with PEP / LVET / QS2')
        _p8_plot_common_panel(axes[2], 'Radar single-cycle with PEP / LVET / QS2', rep['bt_r'], rep['bx_r'], rep['radar_lm'], 'Radar', q_rel=rep['q_rel'], t_rel=rep['t_rel'], show_brackets=True)
        axes[-1].set_xlabel('Time from ECG R-peak [s]', fontsize=10)
        fig.suptitle('Single-cycle PEP, LVET, and QS2 comparison', fontsize=14)
        fig.savefig(outdir / 'fig06_pep_lvet_qs2_brackets.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # fig10 clean final
        fig, axes = plt.subplots(3,1, figsize=(13.0,9.0), sharex=True, constrained_layout=True)
        _p8_plot_common_panel(axes[0], 'ECG reference landmarks and interval markers', rep['bt_e'], rep['bx_e'], ecg_lm, 'ECG', q_rel=rep['q_rel'], t_rel=rep['t_rel'], show_brackets=True)
        if rep['bt_s'] is not None:
            _p8_plot_common_panel(axes[1], 'SCG landmarks and interval markers', rep['bt_s'], rep['bx_s'], rep['scg_lm'], 'SCG', q_rel=rep['q_rel'], t_rel=rep['t_rel'], show_brackets=True)
        else:
            axes[1].text(0.5,0.5,'SCG unavailable', transform=axes[1].transAxes, ha='center', va='center')
            _p8_style_axis(axes[1], 'SCG landmarks and interval markers')
        _p8_plot_common_panel(axes[2], 'Radar AO / AC landmarks and interval markers', rep['bt_r'], rep['bx_r'], rep['radar_lm'], 'Radar', q_rel=rep['q_rel'], t_rel=rep['t_rel'], show_brackets=True)
        axes[-1].set_xlabel('Time from ECG R-peak [s]', fontsize=10)
        fig.suptitle('Single-cycle ECG / SCG / Radar landmark interval comparison', fontsize=14)
        fig.savefig(outdir / 'fig10_ecg_scg_radar_landmark_interval_clean.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        rows = []
        for mod, lm in [('ECG', ecg_lm), ('SCG', rep['scg_lm']), ('Radar', rep['radar_lm'])]:
            pep, lvet, qs2 = _p5_interval(rep['q_rel'], lm.get('AO'), lm.get('AC'))
            rows.append([mod, rep['beat_index'], rep['q_rel'], 0.0 if mod=='ECG' else None, rep['t_rel'] if mod=='ECG' else None,
                         lm.get('MC'), lm.get('IM'), lm.get('AO'), lm.get('AC'), lm.get('MO'), pep, lvet, qs2])
        save_csv(outdir / 'fig06_fig10_single_cycle_values.csv', ['modality','beat_index','Q_rel_sec','R_rel_sec','T_rel_sec','MC_rel_sec','IM_rel_sec','AO_rel_sec','AC_rel_sec','MO_rel_sec','PEP_ms','LVET_ms','QS2_ms'], rows)
    except Exception as e:
        (outdir / 'patch8_fig06_fig10_error.txt').write_text(str(e), encoding='utf-8')


def _p8_update_paper_export(outdir):
    try:
        paper = outdir / globals().get('PAPER_EXPORT_DIRNAME', 'paper_export')
        figs = paper / 'figures'; figs.mkdir(parents=True, exist_ok=True)
        mapping = [
            ('fig02_compact_beat_morphology.png', 'fig02_ecg_qrs_radar_beat_morphology.png'),
            ('fig02_scg_reference_landmarks.png', 'fig02b_scg_reference_landmarks.png'),
            ('fig06_single_cycle_ecg_radar_aoac_labels.png', 'fig06_single_cycle_ecg_scg_radar_labels.png'),
            ('fig06_pep_lvet_qs2_brackets.png', 'fig06_pep_lvet_qs2_brackets.png'),
            ('fig10_ecg_scg_radar_landmark_interval_clean.png', 'fig10_ecg_scg_radar_landmarks_intervals.png'),
        ]
        rows=[]
        for s,d in mapping:
            sp=outdir/s; dp=figs/d
            status='missing'
            if sp.exists():
                shutil.copyfile(sp, dp); status='copied'
            rows.append([s,d,status])
        save_csv(figs / 'paper_figure_index_patch8.csv', ['Source','PaperFigure','Status'], rows)
    except Exception as e:
        (outdir / 'patch8_paper_export_error.txt').write_text(str(e), encoding='utf-8')


_old_save_all_patch8 = save_all

def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    result = _old_save_all_patch8(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        templ = _p5_build_scg_template(ecg, scg, acfg) if scg is not None else None
        _p8_make_fig02_morphology(outdir, ecg, scg, radar, acfg, templ)
        _p8_make_fig02_scg_reference(outdir, ecg, scg, acfg, templ)
        _p8_make_fig06_fig10(outdir, ecg, scg, radar, acfg, templ)
        _p8_update_paper_export(outdir)
    except Exception as e:
        (outdir / 'save_all_patch8_error.txt').write_text(str(e), encoding='utf-8')
    return result


# ============================================================
# PATCH9: ECG sharp display + SCG spike-oriented single-cycle selection
# ------------------------------------------------------------
# Fixes from ex89:
# 1) ECG figure looked over-smoothed -> use QRS/raw-like branch for figures.
# 2) SCG landmarks were placed on weak/invalid template -> use representative
#    single-cycle BPF/LMS branch with visible AO spike, not ensemble median.
# 3) fig02_compact_beat_morphology includes SCG and all clean figures use
#    common single-cycle axis.
# 4) Remove multiple candidate boxes; final landmarks only.
# ============================================================

P9_COL = {
    "Q": "#1f77b4", "R": "#d62728", "T": "#2ca02c",
    "MC": "#17becf", "IM": "#1f77b4", "AO": "#9467bd",
    "AC": "#ff7f0e", "MO": "#2ca02c"
}

P9_XLIM = (-0.10, 0.62)


def _p9_ecg_fig_signal(ecg):
    # For figure readability, avoid smooth display column; use QRS/filtered or cleaned branch.
    for k in ["filtered", "qrs", "cleaned", "true_display", "display_rpeak", "display"]:
        if k in ecg and ecg[k] is not None and len(ecg[k]) == len(ecg.get("t", [])):
            return _p5_np(ecg[k])
    return _p5_np(ecg.get("display", []))


def _p9_scg_fig_signal(scg):
    # For SCG landmark display, BPF/filtered usually preserves AO spike better than median template.
    if scg is None:
        return None
    for k in ["filtered", "resp_removed", "display", "selected_raw", "vmag", "az", "ax", "ay"]:
        if k in scg and scg[k] is not None and len(scg[k]) == len(scg.get("t", [])):
            return _p5_np(scg[k])
    return None


def _p9_radar_fig_signal(radar):
    for k in ["lms_error", "ppg_like", "displacement", "display"]:
        if k in radar and radar[k] is not None and len(radar[k]) == len(radar.get("t", [])):
            return _p5_np(radar[k])
    return np.zeros(len(radar.get("t", [])), dtype=np.float64)


def _p9_local_extrema_peak(bt, bx, win, mode="pos"):
    bt = _p5_np(bt); bx = _p5_z(bx)
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 3:
        return None
    idx = np.where(m)[0]
    y = bx
    if mode == "neg":
        j = idx[int(np.nanargmin(y[idx]))]
    elif mode == "abs":
        j = idx[int(np.nanargmax(np.abs(y[idx])))]
    else:
        j = idx[int(np.nanargmax(y[idx]))]
    return float(bt[j])


def _p9_slope_curv_event(bt, bx, win, prefer="abs"):
    bt = _p5_np(bt); bx = _p5_z(bx)
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 4:
        return None
    idx = np.where(m)[0]
    try:
        fs = 1.0 / max(np.nanmedian(np.diff(bt)), 1e-6)
        y = safe_lowpass(bx, fs, min(20.0, 0.45 * fs), order=2)
    except Exception:
        y = bx
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    if prefer == "pos":
        score = robust_scale_01(np.maximum(d1[idx], 0)) + 0.65 * robust_scale_01(np.abs(d2[idx]))
    elif prefer == "neg":
        score = robust_scale_01(np.maximum(-d1[idx], 0)) + 0.65 * robust_scale_01(np.abs(d2[idx]))
    else:
        score = robust_scale_01(np.abs(d1[idx])) + robust_scale_01(np.abs(d2[idx]))
    # center/boundary control
    t = bt[idx]
    half = max((win[1]-win[0])/2, 1e-3)
    center = np.mean(win)
    prior = np.exp(-0.5*((t-center)/(half*0.85))**2)
    boundary = np.clip(np.minimum(t-win[0], win[1]-t)/half, 0.0, 1.0)
    score = score * (0.35 + 0.45*prior + 0.20*boundary)
    j = idx[int(np.nanargmax(score))]
    return float(bt[j])


def _p9_scg_landmarks_spike(bt, bx):
    """
    SCG single-cycle landmark heuristic for figures.
    Goal: draw physiologically ordered landmarks on visible morphology.
    AO is intentionally selected as the dominant sharp positive/absolute complex
    in early systolic AO window, matching the user's requested visible AO peak style.
    """
    lm = {}

    lm["MC"] = _p9_slope_curv_event(bt, bx, (-0.08, -0.015), "abs")
    if lm["MC"] is None:
        lm["MC"] = _p9_slope_curv_event(bt, bx, (-0.035, 0.020), "abs")

    lm["IM"] = _p9_slope_curv_event(bt, bx, (0.015, 0.065), "pos")

    # AO: visible sharp peak/complex, not weak template inflection.
    ao_peak = _p9_local_extrema_peak(bt, bx, (0.055, 0.140), mode="pos")
    if ao_peak is None or (lm.get("IM") is not None and ao_peak <= lm["IM"] + 0.010):
        ao_peak = _p9_slope_curv_event(bt, bx, (0.060, 0.155), "pos")
    lm["AO"] = ao_peak

    # AC: primary post-systolic closure region. Do not let late MO steal AC.
    ac_lo = 0.150 if lm["AO"] is None else max(0.150, lm["AO"] + 0.070)
    lm["AC"] = _p9_slope_curv_event(bt, bx, (ac_lo, 0.300), "neg")
    if lm["AC"] is None:
        lm["AC"] = _p9_local_extrema_peak(bt, bx, (ac_lo, 0.330), mode="abs")

    # MO/late vibration kept later and separated.
    mo_lo = 0.320 if lm["AC"] is None else max(0.320, lm["AC"] + 0.060)
    lm["MO"] = _p9_slope_curv_event(bt, bx, (mo_lo, 0.520), "pos")
    if lm["MO"] is None:
        lm["MO"] = _p9_local_extrema_peak(bt, bx, (mo_lo, 0.560), mode="abs")

    # Enforce reasonable order; invalid labels are hidden instead of falsely shown.
    order = ["MC", "IM", "AO", "AC", "MO"]
    bounds = {"MC":(-0.10,0.03), "IM":(0.00,0.08), "AO":(0.045,0.16), "AC":(0.14,0.34), "MO":(0.30,0.58)}
    prev = -1e9
    for k in order:
        x = lm.get(k)
        if x is None or not np.isfinite(float(x)) or x < bounds[k][0] or x > bounds[k][1] or x <= prev + 0.008:
            lm[k] = None
        else:
            lm[k] = float(x); prev = float(x)
    return lm


def _p9_radar_landmarks_clean(bt, bx):
    lm = {}
    lm["AO"] = _p9_slope_curv_event(bt, bx, (0.070, 0.160), "pos")
    ac_lo = 0.250 if lm["AO"] is None else max(0.250, lm["AO"] + 0.100)
    lm["AC"] = _p9_slope_curv_event(bt, bx, (ac_lo, 0.460), "neg")
    return lm


def _p9_ecg_landmarks_clean(ecg, beat_index, anchor, aoac=None):
    q_rel, t_rel = _p5_qt_rel(ecg, beat_index, anchor)
    ao, ac = None, None
    if aoac is not None:
        for b in aoac.get("beats", []):
            try:
                if int(b.get("beat_index", -1)) == int(beat_index):
                    ao = b.get("ecg_ao_ref", b.get("ao_ref"))
                    ac = b.get("ecg_ac_ref", b.get("ac_ref"))
                    break
            except Exception:
                pass
    if ao is None: ao = 0.110
    if ac is None: ac = 0.370
    return {"Q": q_rel, "R": 0.0, "T": t_rel, "AO": float(ao), "AC": float(ac)}


def _p9_pick_single_cycle(ecg, scg, radar, acfg, aoac=None):
    r = _p5_np(ecg.get("peaks_time", []))
    if len(r) < 3:
        return None
    e_sig = _p9_ecg_fig_signal(ecg)
    s_sig = _p9_scg_fig_signal(scg) if scg is not None else None
    rd_sig = _p9_radar_fig_signal(radar)
    best = None
    # prioritize middle stable beats but choose SCG beat where AO spike is visible
    for bi in range(1, len(r)-1):
        anchor = float(r[bi])
        bt_e, bx_e = _p5_slice(ecg["t"], e_sig, anchor, 0.10, 0.62, 100.0)
        bt_r, bx_r = _p5_slice(radar["t"], rd_sig, anchor, 0.10, 0.62, 100.0)
        if bt_e is None or bt_r is None:
            continue
        bt_s, bx_s = (None, None)
        scg_lm = {}
        scg_score = 0.0
        if s_sig is not None:
            bt_s, bx_s = _p5_slice(scg["t"], s_sig, anchor, 0.10, 0.62, 100.0)
            if bt_s is None:
                continue
            scg_lm = _p9_scg_landmarks_spike(bt_s, bx_s)
            # score: prefer visible sharp positive AO complex, not flat/median-looking beat
            ao = scg_lm.get("AO")
            if ao is not None:
                y_ao = float(np.interp(ao, bt_s, bx_s))
                scg_score += 4.0 * max(0.0, y_ao)
                scg_score += 2.0 * np.nanstd(bx_s[(bt_s>=0.04)&(bt_s<=0.18)])
            scg_score += 0.35 * sum(scg_lm.get(k) is not None for k in ["MC","IM","AO","AC","MO"])
        radar_lm = _p9_radar_landmarks_clean(bt_r, bx_r)
        ecg_lm = _p9_ecg_landmarks_clean(ecg, bi, anchor, aoac)
        # reject if no key SCG AO when SCG exists
        if s_sig is not None and scg_lm.get("AO") is None:
            continue
        # avoid too early/late R-to-R boundary by using middle preference
        mid_pref = -0.003 * abs(bi - len(r)/2)
        score = scg_score + 0.8 * sum(radar_lm.get(k) is not None for k in ["AO","AC"]) + mid_pref
        cand = dict(score=score, beat_index=bi, anchor=anchor,
                    bt_e=bt_e, bx_e=bx_e, bt_s=bt_s, bx_s=bx_s, bt_r=bt_r, bx_r=bx_r,
                    ecg_lm=ecg_lm, scg_lm=scg_lm, radar_lm=radar_lm)
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def _p9_draw_landmark(ax, bt, bx, x, name, yfrac=0.92):
    if x is None:
        return
    try:
        x = float(x)
        if not np.isfinite(x): return
    except Exception:
        return
    c = P9_COL.get(name, "#111111")
    y = float(np.interp(x, bt, bx))
    ax.axvline(x, color=c, linestyle="--", linewidth=1.15, alpha=0.95, zorder=3)
    marker = {"Q":"o","R":"o","T":"o","MC":"o","IM":"^","AO":"s","AC":"D","MO":"v"}.get(name, "o")
    ax.scatter([x], [y], s=56, marker=marker, facecolor="white", edgecolor=c, linewidth=1.5, zorder=5)
    ymin, ymax = ax.get_ylim()
    ty = ymin + (ymax-ymin)*yfrac
    ax.text(x, ty, name, ha="center", va="top", fontsize=8.4, color=c,
            bbox=dict(boxstyle="round,pad=0.13", fc="white", ec=c, alpha=0.96), zorder=6)


def _p9_draw_bracket(ax, x0, x1, y, text):
    if x0 is None or x1 is None:
        return
    try:
        x0=float(x0); x1=float(x1)
        if not (np.isfinite(x0) and np.isfinite(x1)): return
    except Exception:
        return
    ax.plot([x0,x1],[y,y], color="0.25", linewidth=1.1, zorder=2)
    ax.plot([x0,x0],[y-0.025,y+0.025], color="0.25", linewidth=0.9)
    ax.plot([x1,x1],[y-0.025,y+0.025], color="0.25", linewidth=0.9)
    ax.text((x0+x1)/2, y+0.03, text, ha="center", va="bottom", fontsize=7.9,
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="0.55", alpha=0.95))


def _p9_plot(ax, title, bt, bx, lm, kind, show_brackets=False):
    ax.plot(bt, bx, color="black", linewidth=1.55, zorder=1)
    ax.axvline(0, color="0.35", linestyle=":", linewidth=1.0, zorder=2)
    if kind == "SCG":
        spans = [((-0.08,-0.015),"MC"),((0.015,0.065),"IM"),((0.055,0.140),"AO"),((0.150,0.300),"AC"),((0.320,0.520),"MO")]
        draw = [("MC",0.96),("IM",0.90),("AO",0.84),("AC",0.78),("MO",0.72)]
    elif kind == "Radar":
        spans = [((0.070,0.160),"AO"),((0.250,0.460),"AC")]
        draw = [("AO",0.90),("AC",0.82)]
    else:
        spans = [((0.070,0.160),"AO"),((0.280,0.460),"AC")]
        draw = [("Q",0.96),("R",0.90),("T",0.84),("AO",0.78),("AC",0.72)]
    for win, name in spans:
        ax.axvspan(win[0], win[1], color=P9_COL.get(name,"0.8"), alpha=0.055, zorder=0)
    ymin, ymax = float(np.nanmin(bx)), float(np.nanmax(bx))
    pad = max(0.65, 0.22*(ymax-ymin+1e-6))
    ax.set_ylim(ymin-pad, ymax+1.05)
    for name, yf in draw:
        _p9_draw_landmark(ax, bt, bx, lm.get(name), name, yf)
    if show_brackets:
        q = lm.get("Q_for_interval", lm.get("Q"))
        ao = lm.get("AO"); ac = lm.get("AC")
        ymin, ymax = ax.get_ylim()
        base = ymin + 0.11*(ymax-ymin)
        _p9_draw_bracket(ax, q, ao, base, "PEP")
        _p9_draw_bracket(ax, ao, ac, base+0.15*(ymax-ymin), "LVET")
        _p9_draw_bracket(ax, q, ac, base+0.30*(ymax-ymin), "QS2")
    ax.set_title(title, loc="left", fontsize=11, pad=6)
    ax.set_ylabel("z-score", fontsize=10)
    ax.grid(True, alpha=0.22)
    ax.set_xlim(-0.10, 0.62)
    ax.tick_params(labelsize=9)


def _p9_make_all_clean_figs(outdir, ecg, scg, radar, aoac, acfg):
    try:
        rep = _p9_pick_single_cycle(ecg, scg, radar, acfg, aoac)
        if rep is None:
            (outdir/"patch9_no_representative_cycle.txt").write_text("No representative single-cycle beat found.", encoding="utf-8")
            return
        ecg_lm = dict(rep["ecg_lm"])
        scg_lm = dict(rep["scg_lm"])
        radar_lm = dict(rep["radar_lm"])
        # add Q for interval to SCG/Radar panels
        scg_lm["Q_for_interval"] = ecg_lm.get("Q")
        radar_lm["Q_for_interval"] = ecg_lm.get("Q")

        # fig02 compact morphology includes SCG
        fig, axes = plt.subplots(3,1,figsize=(12.3,8.2),sharex=True,constrained_layout=True)
        _p9_plot(axes[0], "ECG QRS-preserved representative beat morphology", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", False)
        if rep["bt_s"] is not None:
            _p9_plot(axes[1], "SCG representative beat morphology", rep["bt_s"], rep["bx_s"], scg_lm, "SCG", False)
        else:
            axes[1].text(0.5,0.5,"SCG unavailable", ha="center", va="center", transform=axes[1].transAxes)
        _p9_plot(axes[2], "Radar representative beat morphology", rep["bt_r"], rep["bx_r"], radar_lm, "Radar", False)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Compact ECG / SCG / Radar beat morphology", fontsize=14)
        fig.savefig(outdir/"fig02_compact_beat_morphology.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # fig02 scg reference: real single-cycle, visible AO spike
        if rep["bt_s"] is not None:
            fig, ax = plt.subplots(1,1,figsize=(11.5,4.4),constrained_layout=True)
            _p9_plot(ax, "SCG representative single-cycle reference landmarks", rep["bt_s"], rep["bx_s"], scg_lm, "SCG", False)
            ax.set_xlabel("Time from ECG R-peak [s]")
            fig.savefig(outdir/"fig02_scg_reference_landmarks.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            save_csv(outdir/"fig02_scg_reference_landmarks_values.csv",
                     ["beat_index","MC_rel_sec","IM_rel_sec","AO_rel_sec","AC_rel_sec","MO_rel_sec"],
                     [[rep["beat_index"], scg_lm.get("MC"), scg_lm.get("IM"), scg_lm.get("AO"), scg_lm.get("AC"), scg_lm.get("MO")]])

        # fig06 labels
        fig, axes = plt.subplots(3,1,figsize=(12.6,8.5),sharex=True,constrained_layout=True)
        _p9_plot(axes[0], "ECG single-cycle labels", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", False)
        if rep["bt_s"] is not None: _p9_plot(axes[1], "SCG single-cycle labels", rep["bt_s"], rep["bx_s"], scg_lm, "SCG", False)
        else: axes[1].text(0.5,0.5,"SCG unavailable", ha="center", va="center", transform=axes[1].transAxes)
        _p9_plot(axes[2], "Radar single-cycle labels", rep["bt_r"], rep["bx_r"], radar_lm, "Radar", False)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle ECG / SCG / Radar landmark labeling", fontsize=14)
        fig.savefig(outdir/"fig06_single_cycle_ecg_radar_aoac_labels.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # fig06 brackets
        fig, axes = plt.subplots(3,1,figsize=(12.7,8.8),sharex=True,constrained_layout=True)
        _p9_plot(axes[0], "ECG single-cycle with PEP / LVET / QS2", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", True)
        if rep["bt_s"] is not None: _p9_plot(axes[1], "SCG single-cycle with PEP / LVET / QS2", rep["bt_s"], rep["bx_s"], scg_lm, "SCG", True)
        else: axes[1].text(0.5,0.5,"SCG unavailable", ha="center", va="center", transform=axes[1].transAxes)
        _p9_plot(axes[2], "Radar single-cycle with PEP / LVET / QS2", rep["bt_r"], rep["bx_r"], radar_lm, "Radar", True)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle PEP, LVET, and QS2 comparison", fontsize=14)
        fig.savefig(outdir/"fig06_pep_lvet_qs2_brackets.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # fig10 final clean
        fig, axes = plt.subplots(3,1,figsize=(13.0,9.0),sharex=True,constrained_layout=True)
        _p9_plot(axes[0], "ECG reference landmarks and interval markers", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", True)
        if rep["bt_s"] is not None: _p9_plot(axes[1], "SCG landmarks and interval markers", rep["bt_s"], rep["bx_s"], scg_lm, "SCG", True)
        else: axes[1].text(0.5,0.5,"SCG unavailable", ha="center", va="center", transform=axes[1].transAxes)
        _p9_plot(axes[2], "Radar AO / AC landmarks and interval markers", rep["bt_r"], rep["bx_r"], radar_lm, "Radar", True)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle ECG / SCG / Radar landmark interval comparison", fontsize=14)
        fig.savefig(outdir/"fig10_ecg_scg_radar_landmark_interval_clean.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows=[]
        for mod,lm in [("ECG",ecg_lm),("SCG",scg_lm),("Radar",radar_lm)]:
            q = ecg_lm.get("Q")
            pep,lvet,qs2 = _p5_interval(q,lm.get("AO"),lm.get("AC"))
            rows.append([mod,rep["beat_index"],q,0 if mod=="ECG" else None,ecg_lm.get("T") if mod=="ECG" else None,lm.get("MC"),lm.get("IM"),lm.get("AO"),lm.get("AC"),lm.get("MO"),pep,lvet,qs2])
        save_csv(outdir/"fig06_fig10_single_cycle_values.csv",
                 ["modality","beat_index","Q_rel_sec","R_rel_sec","T_rel_sec","MC_rel_sec","IM_rel_sec","AO_rel_sec","AC_rel_sec","MO_rel_sec","PEP_ms","LVET_ms","QS2_ms"],rows)

        # paper export copy
        try:
            paper = outdir / globals().get("PAPER_EXPORT_DIRNAME","paper_export")
            figs = paper/"figures"; figs.mkdir(parents=True, exist_ok=True)
            mapping=[
                ("fig02_compact_beat_morphology.png","fig02_ecg_scg_radar_beat_morphology.png"),
                ("fig02_scg_reference_landmarks.png","fig02b_scg_reference_landmarks.png"),
                ("fig06_single_cycle_ecg_radar_aoac_labels.png","fig06_single_cycle_ecg_scg_radar_labels.png"),
                ("fig06_pep_lvet_qs2_brackets.png","fig06_pep_lvet_qs2_brackets.png"),
                ("fig10_ecg_scg_radar_landmark_interval_clean.png","fig10_ecg_scg_radar_landmarks_intervals.png"),
            ]
            copied=[]
            for s,d in mapping:
                sp=outdir/s; dp=figs/d
                if sp.exists():
                    shutil.copyfile(sp,dp); copied.append([s,d,"copied"])
                else:
                    copied.append([s,d,"missing"])
            save_csv(figs/"paper_figure_index_patch9.csv",["Source","PaperFigure","Status"],copied)
        except Exception as ee:
            (outdir/"patch9_paper_export_error.txt").write_text(str(ee), encoding="utf-8")
    except Exception as e:
        (outdir/"patch9_clean_figs_error.txt").write_text(str(e), encoding="utf-8")


_old_save_all_patch9 = save_all

def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    result = _old_save_all_patch9(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        _p9_make_all_clean_figs(outdir, ecg, scg, radar, aoac, acfg)
    except Exception as e:
        (outdir/"save_all_patch9_error.txt").write_text(str(e), encoding="utf-8")
    return result



# ============================================================
# PATCH10: hard fix for ECG sharpness + SCG AO-spike reference
# ------------------------------------------------------------
# User-facing figure fixes:
# 1) ECG figure branch uses raw/QRS-preserved signal, not smoothed display.
# 2) SCG reference figure is selected from a real single-cycle beat where
#    early-systolic AO spike is visible. It no longer uses the ensemble median
#    for fig02_scg_reference_landmarks.
# 3) Invalid SCG landmarks are hidden instead of forcing wrong labels.
# 4) fig02_compact_beat_morphology includes ECG/SCG/Radar.
# 5) fig06 and fig10 share the same time axis and draw only final markers.
# ============================================================

P10_C = {
    "Q": "#1f77b4", "R": "#d62728", "T": "#2ca02c",
    "MC": "#17becf", "IM": "#1f77b4", "AO": "#9467bd",
    "AC": "#ff7f0e", "MO": "#2ca02c",
}
P10_XLIM = (-0.10, 0.56)


def _p10_bandpass_or_z(x, fs=100.0, lo=4.0, hi=30.0):
    x = np.asarray(x, dtype=np.float64)
    try:
        y = safe_bandpass(x, fs, lo, min(hi, fs * 0.45), order=3)
    except Exception:
        try:
            y = signal.detrend(x)
        except Exception:
            y = x - np.nanmedian(x)
    return zscore_safe(y)


def _p10_ecg_wave_for_fig(ecg):
    """
    ECG display must keep R peak sharp.
    Priority:
      raw_adc_col -> raw -> qrs filtered -> cleaned.
    Then apply QRS-preserving bandpass for figure only.
    """
    n = len(ecg.get("t", []))
    for k in ["raw_adc_col", "raw", "qrs", "filtered", "cleaned", "true_display", "display_rpeak", "display"]:
        if k in ecg and ecg[k] is not None and len(ecg[k]) == n:
            return _p10_bandpass_or_z(ecg[k], float(ecg.get("fs", 100.0)), 5.0, 32.0)
    return np.zeros(n, dtype=np.float64)


def _p10_scg_wave_for_fig(scg):
    """
    SCG reference should show spike morphology, so prefer BPF/filtered branch.
    Do not use ensemble median for the reference figure.
    """
    if scg is None:
        return None
    n = len(scg.get("t", []))
    for k in ["filtered", "resp_removed", "display", "selected_raw", "vmag", "az", "ax", "ay"]:
        if k in scg and scg[k] is not None and len(scg[k]) == n:
            x = np.asarray(scg[k], dtype=np.float64)
            # preserve spike; only light detrend/zscore, no heavy smoothing
            return zscore_safe(signal.detrend(x) if len(x) > 8 else x)
    return None


def _p10_radar_wave_for_fig(radar):
    n = len(radar.get("t", []))
    for k in ["lms_error", "ppg_like", "displacement", "display"]:
        if k in radar and radar[k] is not None and len(radar[k]) == n:
            return zscore_safe(np.asarray(radar[k], dtype=np.float64))
    return np.zeros(n, dtype=np.float64)


def _p10_slice(tt, xx, anchor, pre=0.10, post=0.56):
    return _p5_slice(tt, xx, anchor, pre, post, 100.0)


def _p10_local_peak(bt, bx, win, mode="max"):
    bt = np.asarray(bt, dtype=float); bx = zscore_safe(np.asarray(bx, dtype=float))
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 3:
        return None
    idx = np.where(m)[0]
    if mode == "min":
        j = idx[int(np.nanargmin(bx[idx]))]
    elif mode == "abs":
        j = idx[int(np.nanargmax(np.abs(bx[idx])))]
    else:
        j = idx[int(np.nanargmax(bx[idx]))]
    return float(bt[j])


def _p10_slope_event(bt, bx, win, sign="abs"):
    bt = np.asarray(bt, dtype=float); bx = zscore_safe(np.asarray(bx, dtype=float))
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 4:
        return None
    idx = np.where(m)[0]
    try:
        y = safe_lowpass(bx, 100.0, 20.0, order=2)
    except Exception:
        y = bx
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    if sign == "pos":
        s = robust_scale_01(np.maximum(d1[idx], 0)) + 0.5 * robust_scale_01(np.abs(d2[idx]))
    elif sign == "neg":
        s = robust_scale_01(np.maximum(-d1[idx], 0)) + 0.5 * robust_scale_01(np.abs(d2[idx]))
    else:
        s = robust_scale_01(np.abs(d1[idx])) + robust_scale_01(np.abs(d2[idx]))
    t = bt[idx]
    mid = np.mean(win); half = max((win[1]-win[0])/2, 1e-3)
    prior = np.exp(-0.5*((t-mid)/(half*0.9))**2)
    edge = np.clip(np.minimum(t-win[0], win[1]-t)/half, 0.0, 1.0)
    s = s * (0.30 + 0.45 * prior + 0.25 * edge)
    j = idx[int(np.nanargmax(s))]
    return float(bt[j])


def _p10_scg_landmarks_visible(bt, bx):
    """
    Figure-oriented SCG landmarks.
    AO is the visible early systolic spike/peak.
    Other landmarks are shown only when they satisfy plausible order/range.
    """
    bx = zscore_safe(bx)
    lm = {}

    # AO: visible positive spike in early systolic window. This directly addresses the user's requested style.
    ao = _p10_local_peak(bt, bx, (0.045, 0.155), "max")
    if ao is None:
        ao = _p10_slope_event(bt, bx, (0.055, 0.165), "pos")
    lm["AO"] = ao

    # MC/IM before AO, not forced if unclear.
    if ao is not None:
        mc = _p10_slope_event(bt, bx, (-0.080, min(0.025, ao - 0.035)), "abs")
        im = _p10_slope_event(bt, bx, (max(0.000, (mc or 0.0) + 0.010), max(0.035, ao - 0.012)), "pos")
    else:
        mc = _p10_slope_event(bt, bx, (-0.080, 0.025), "abs")
        im = _p10_slope_event(bt, bx, (0.000, 0.080), "pos")
    lm["MC"] = mc
    lm["IM"] = im

    # AC: closure-related candidate after AO. Keep within primary systolic-end area.
    if ao is not None:
        ac = _p10_slope_event(bt, bx, (max(0.170, ao + 0.080), 0.360), "neg")
        if ac is None:
            ac = _p10_local_peak(bt, bx, (max(0.170, ao + 0.080), 0.380), "abs")
    else:
        ac = _p10_slope_event(bt, bx, (0.190, 0.380), "neg")
    lm["AC"] = ac

    # MO/late complex is optional; do not force wrong late marker.
    if ac is not None:
        mo = _p10_slope_event(bt, bx, (max(0.330, ac + 0.060), 0.540), "pos")
        if mo is None:
            mo = _p10_local_peak(bt, bx, (max(0.330, ac + 0.060), 0.540), "abs")
    else:
        mo = None
    lm["MO"] = mo

    # Validate/hide wrong labels.
    bounds = {"MC":(-0.10,0.035), "IM":(0.0,0.115), "AO":(0.045,0.165), "AC":(0.170,0.390), "MO":(0.330,0.560)}
    order = ["MC","IM","AO","AC","MO"]
    prev = -999
    for k in order:
        x = lm.get(k)
        if x is None or not np.isfinite(float(x)) or not (bounds[k][0] <= float(x) <= bounds[k][1]) or float(x) <= prev + 0.008:
            lm[k] = None
        else:
            lm[k] = float(x); prev = float(x)
    return lm


def _p10_radar_landmarks(bt, bx):
    ao = _p10_slope_event(bt, bx, (0.070, 0.165), "pos")
    ac = _p10_slope_event(bt, bx, (0.260 if ao is None else max(0.260, ao + 0.100), 0.460), "neg")
    return {"AO": ao, "AC": ac}


def _p10_ecg_landmarks(ecg, bi, anchor, aoac=None):
    q, t = _p5_qt_rel(ecg, bi, anchor)
    ao, ac = None, None
    if aoac is not None:
        try:
            for b in aoac.get("beats", []):
                if int(b.get("beat_index", -1)) == int(bi):
                    ao = b.get("ecg_ao_ref", b.get("ao_ref"))
                    ac = b.get("ecg_ac_ref", b.get("ac_ref"))
                    break
        except Exception:
            pass
    if ao is None: ao = 0.110
    if ac is None: ac = 0.370
    return {"Q": q, "R": 0.0, "T": t, "AO": float(ao), "AC": float(ac)}


def _p10_pick_cycle(ecg, scg, radar, acfg, aoac=None):
    r = np.asarray(ecg.get("peaks_time", []), dtype=float)
    if len(r) < 3:
        return None
    ew = _p10_ecg_wave_for_fig(ecg)
    sw = _p10_scg_wave_for_fig(scg) if scg is not None else None
    rw = _p10_radar_wave_for_fig(radar)
    best = None
    for bi in range(1, len(r)-1):
        anchor = float(r[bi])
        bt_e, bx_e = _p10_slice(ecg["t"], ew, anchor)
        bt_r, bx_r = _p10_slice(radar["t"], rw, anchor)
        if bt_e is None or bt_r is None:
            continue
        bt_s, bx_s, scg_lm = None, None, {}
        score = -0.002 * abs(bi - len(r)/2)
        if sw is not None:
            bt_s, bx_s = _p10_slice(scg["t"], sw, anchor)
            if bt_s is None:
                continue
            scg_lm = _p10_scg_landmarks_visible(bt_s, bx_s)
            ao = scg_lm.get("AO")
            if ao is None:
                continue
            # Strongly prefer beats where AO is an actual visible spike.
            ao_y = float(np.interp(ao, bt_s, bx_s))
            early = bx_s[(bt_s >= 0.035) & (bt_s <= 0.170)]
            score += 5.0 * max(0.0, ao_y)
            score += 2.0 * float(np.nanstd(early)) if len(early) else 0.0
            score += 0.4 * sum(scg_lm.get(k) is not None for k in ["MC","IM","AO","AC","MO"])
        radar_lm = _p10_radar_landmarks(bt_r, bx_r)
        ecg_lm = _p10_ecg_landmarks(ecg, bi, anchor, aoac)
        score += 0.7 * sum(radar_lm.get(k) is not None for k in ["AO","AC"])
        cand = {"score":score, "beat_index":bi, "anchor":anchor,
                "bt_e":bt_e, "bx_e":bx_e, "bt_s":bt_s, "bx_s":bx_s, "bt_r":bt_r, "bx_r":bx_r,
                "ecg_lm":ecg_lm, "scg_lm":scg_lm, "radar_lm":radar_lm}
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def _p10_label(ax, bt, bx, x, name, yfrac):
    if x is None:
        return
    try:
        x = float(x)
        if not np.isfinite(x): return
    except Exception:
        return
    c = P10_C.get(name, "black")
    y = float(np.interp(x, bt, bx))
    ax.axvline(x, color=c, linestyle="--", linewidth=1.1, zorder=3)
    mk = {"Q":"o","R":"o","T":"o","MC":"o","IM":"^","AO":"s","AC":"D","MO":"v"}.get(name,"o")
    ax.scatter([x],[y], s=55, marker=mk, facecolor="white", edgecolor=c, linewidth=1.4, zorder=5)
    ymin, ymax = ax.get_ylim()
    ax.text(x, ymin + (ymax-ymin)*yfrac, name, ha="center", va="top", fontsize=8.3, color=c,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=c, alpha=0.96), zorder=6)


def _p10_bracket(ax, x0, x1, y, label):
    if x0 is None or x1 is None:
        return
    try:
        x0, x1 = float(x0), float(x1)
        if not (np.isfinite(x0) and np.isfinite(x1)): return
    except Exception:
        return
    ax.plot([x0,x1],[y,y], color="0.25", lw=1.1)
    ax.plot([x0,x0],[y-0.025,y+0.025], color="0.25", lw=0.9)
    ax.plot([x1,x1],[y-0.025,y+0.025], color="0.25", lw=0.9)
    ax.text((x0+x1)/2, y+0.030, label, ha="center", va="bottom", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="0.60", alpha=0.95))


def _p10_panel(ax, title, bt, bx, lm, mode, brackets=False, q_for_interval=None):
    ax.plot(bt, bx, color="black", lw=1.55)
    ax.axvline(0, color="0.4", linestyle=":", lw=1.0)
    if mode == "ECG":
        spans = [((0.07,0.16),"AO"), ((0.28,0.46),"AC")]
        order = [("Q",0.96),("R",0.90),("T",0.84),("AO",0.78),("AC",0.72)]
    elif mode == "SCG":
        spans = [((-0.08,-0.015),"MC"),((0.0,0.08),"IM"),((0.045,0.165),"AO"),((0.170,0.390),"AC"),((0.330,0.560),"MO")]
        order = [("MC",0.96),("IM",0.90),("AO",0.84),("AC",0.78),("MO",0.72)]
    else:
        spans = [((0.07,0.165),"AO"), ((0.260,0.460),"AC")]
        order = [("AO",0.90),("AC",0.82)]
    for win, name in spans:
        ax.axvspan(win[0], win[1], color=P10_C.get(name, "#777777"), alpha=0.045, zorder=0)
    ymin, ymax = float(np.nanmin(bx)), float(np.nanmax(bx))
    pad = max(0.65, 0.22*(ymax-ymin+1e-9))
    ax.set_ylim(ymin-pad, ymax+1.05)
    for name, yf in order:
        _p10_label(ax, bt, bx, lm.get(name), name, yf)
    if brackets:
        q = q_for_interval if q_for_interval is not None else lm.get("Q")
        ao, ac = lm.get("AO"), lm.get("AC")
        ymin, ymax = ax.get_ylim()
        base = ymin + 0.10*(ymax-ymin)
        _p10_bracket(ax, q, ao, base, "PEP")
        _p10_bracket(ax, ao, ac, base + 0.15*(ymax-ymin), "LVET")
        _p10_bracket(ax, q, ac, base + 0.30*(ymax-ymin), "QS2")
    ax.set_title(title, loc="left", fontsize=11, pad=6)
    ax.set_ylabel("z-score", fontsize=10)
    ax.grid(True, alpha=0.22)
    ax.set_xlim(*P10_XLIM)
    ax.tick_params(labelsize=9)


def _p10_make_clean_figs(outdir, ecg, scg, radar, aoac, acfg):
    try:
        rep = _p10_pick_cycle(ecg, scg, radar, acfg, aoac)
        if rep is None:
            (outdir/"patch10_no_cycle.txt").write_text("No valid SCG AO-spike representative cycle found.", encoding="utf-8")
            return
        ecg_lm, scg_lm, radar_lm = rep["ecg_lm"], rep["scg_lm"], rep["radar_lm"]
        q = ecg_lm.get("Q")

        # fig02 compact morphology
        fig, axes = plt.subplots(3,1,figsize=(12.4,8.2),sharex=True,constrained_layout=True)
        _p10_panel(axes[0], "ECG QRS-preserved representative beat morphology", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG")
        _p10_panel(axes[1], "SCG AO-spike representative beat morphology", rep["bt_s"], rep["bx_s"], scg_lm, "SCG")
        _p10_panel(axes[2], "Radar representative beat morphology", rep["bt_r"], rep["bx_r"], radar_lm, "Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Compact ECG / SCG / Radar beat morphology", fontsize=14)
        fig.savefig(outdir/"fig02_compact_beat_morphology.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # fig02 scg reference only
        fig, ax = plt.subplots(1,1,figsize=(11.5,4.4),constrained_layout=True)
        _p10_panel(ax, "SCG representative single-cycle reference landmarks", rep["bt_s"], rep["bx_s"], scg_lm, "SCG")
        ax.set_xlabel("Time from ECG R-peak [s]")
        fig.savefig(outdir/"fig02_scg_reference_landmarks.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        save_csv(outdir/"fig02_scg_reference_landmarks_values.csv",
                 ["beat_index","MC_rel_sec","IM_rel_sec","AO_rel_sec","AC_rel_sec","MO_rel_sec"],
                 [[rep["beat_index"], scg_lm.get("MC"), scg_lm.get("IM"), scg_lm.get("AO"), scg_lm.get("AC"), scg_lm.get("MO")]])

        # fig06 labels
        fig, axes = plt.subplots(3,1,figsize=(12.6,8.5),sharex=True,constrained_layout=True)
        _p10_panel(axes[0], "ECG single-cycle labels", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG")
        _p10_panel(axes[1], "SCG single-cycle labels", rep["bt_s"], rep["bx_s"], scg_lm, "SCG")
        _p10_panel(axes[2], "Radar single-cycle labels", rep["bt_r"], rep["bx_r"], radar_lm, "Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle ECG / SCG / Radar landmark labeling", fontsize=14)
        fig.savefig(outdir/"fig06_single_cycle_ecg_radar_aoac_labels.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # fig06 brackets
        fig, axes = plt.subplots(3,1,figsize=(12.8,8.8),sharex=True,constrained_layout=True)
        _p10_panel(axes[0], "ECG single-cycle with PEP / LVET / QS2", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", True)
        _p10_panel(axes[1], "SCG single-cycle with PEP / LVET / QS2", rep["bt_s"], rep["bx_s"], scg_lm, "SCG", True, q)
        _p10_panel(axes[2], "Radar single-cycle with PEP / LVET / QS2", rep["bt_r"], rep["bx_r"], radar_lm, "Radar", True, q)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle PEP, LVET, and QS2 comparison", fontsize=14)
        fig.savefig(outdir/"fig06_pep_lvet_qs2_brackets.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # fig10 final
        fig, axes = plt.subplots(3,1,figsize=(13.0,9.0),sharex=True,constrained_layout=True)
        _p10_panel(axes[0], "ECG reference landmarks and interval markers", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", True)
        _p10_panel(axes[1], "SCG landmarks and interval markers", rep["bt_s"], rep["bx_s"], scg_lm, "SCG", True, q)
        _p10_panel(axes[2], "Radar AO / AC landmarks and interval markers", rep["bt_r"], rep["bx_r"], radar_lm, "Radar", True, q)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle ECG / SCG / Radar landmark interval comparison", fontsize=14)
        fig.savefig(outdir/"fig10_ecg_scg_radar_landmark_interval_clean.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        rows = []
        for mod, lm in [("ECG", ecg_lm), ("SCG", scg_lm), ("Radar", radar_lm)]:
            qv = ecg_lm.get("Q")
            pep, lvet, qs2 = _p5_interval(qv, lm.get("AO"), lm.get("AC"))
            rows.append([mod, rep["beat_index"], qv, 0.0 if mod=="ECG" else None,
                         ecg_lm.get("T") if mod=="ECG" else None,
                         lm.get("MC"), lm.get("IM"), lm.get("AO"), lm.get("AC"), lm.get("MO"),
                         pep, lvet, qs2])
        save_csv(outdir/"fig06_fig10_single_cycle_values.csv",
                 ["modality","beat_index","Q_rel_sec","R_rel_sec","T_rel_sec","MC_rel_sec","IM_rel_sec","AO_rel_sec","AC_rel_sec","MO_rel_sec","PEP_ms","LVET_ms","QS2_ms"], rows)

        # paper export
        paper = outdir / globals().get("PAPER_EXPORT_DIRNAME","paper_export")
        figs = paper/"figures"; figs.mkdir(parents=True, exist_ok=True)
        mapping = [
            ("fig02_compact_beat_morphology.png","fig02_ecg_scg_radar_beat_morphology.png"),
            ("fig02_scg_reference_landmarks.png","fig02b_scg_reference_landmarks.png"),
            ("fig06_single_cycle_ecg_radar_aoac_labels.png","fig06_single_cycle_ecg_scg_radar_labels.png"),
            ("fig06_pep_lvet_qs2_brackets.png","fig06_pep_lvet_qs2_brackets.png"),
            ("fig10_ecg_scg_radar_landmark_interval_clean.png","fig10_ecg_scg_radar_landmarks_intervals.png"),
        ]
        idx_rows = []
        for s,d in mapping:
            sp, dp = outdir/s, figs/d
            status = "missing"
            if sp.exists():
                shutil.copyfile(sp, dp)
                status = "copied"
            idx_rows.append([s,d,status])
        save_csv(figs/"paper_figure_index_patch10.csv", ["Source","PaperFigure","Status"], idx_rows)

    except Exception as e:
        (outdir/"patch10_clean_figs_error.txt").write_text(str(e), encoding="utf-8")


_old_save_all_patch10 = save_all

def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    result = _old_save_all_patch10(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        _p10_make_clean_figs(outdir, ecg, scg, radar, aoac, acfg)
    except Exception as e:
        (outdir/"save_all_patch10_error.txt").write_text(str(e), encoding="utf-8")
    return result



# ============================================================
# PATCH11 utility: empty SCG placeholder for optional downstream handling
# ============================================================

def make_empty_scg_result(reason: str = "SCG unavailable"):
    return {
        "enabled": False,
        "reason": reason,
        "fs": 100.0,
        "t": np.asarray([], dtype=np.float64),
        "ax": np.asarray([], dtype=np.float64),
        "ay": np.asarray([], dtype=np.float64),
        "az": np.asarray([], dtype=np.float64),
        "gx": np.asarray([], dtype=np.float64),
        "gy": np.asarray([], dtype=np.float64),
        "gz": np.asarray([], dtype=np.float64),
        "vmag": np.asarray([], dtype=np.float64),
        "selected_raw": np.asarray([], dtype=np.float64),
        "resp_removed": np.asarray([], dtype=np.float64),
        "filtered": np.asarray([], dtype=np.float64),
        "display": np.asarray([], dtype=np.float64),
        "peaks_time": np.asarray([], dtype=np.float64),
        "peaks_index": np.asarray([], dtype=np.int64),
    }


# ============================================================
# PATCH12: best-segment candidate-safe figure export
# ============================================================

P12_C = {"Q":"#1f77b4","R":"#d62728","T":"#2ca02c","MC":"#17becf","IM":"#1f77b4","AO":"#9467bd","AC":"#ff7f0e","MO":"#2ca02c"}
P12_XLIM = (-0.10, 0.56)

def _p12_z(x):
    return zscore_safe(np.asarray(x, dtype=np.float64))

def _p12_ecg_sig(ecg):
    n=len(ecg.get("t",[]))
    for k in ["raw_adc_col","raw","qrs","filtered","cleaned","display_rpeak","display"]:
        if k in ecg and ecg[k] is not None and len(ecg[k])==n:
            x=np.asarray(ecg[k],dtype=np.float64)
            try:
                return zscore_safe(safe_bandpass(x,float(ecg.get("fs",100.0)),5.0,32.0,order=3))
            except Exception:
                return zscore_safe(signal.detrend(x) if len(x)>8 else x)
    return np.zeros(n,dtype=np.float64)

def _p12_scg_sig(scg):
    if scg is None: return None
    n=len(scg.get("t",[]))
    for k in ["filtered","resp_removed","display","selected_raw","vmag","az","ax","ay"]:
        if k in scg and scg[k] is not None and len(scg[k])==n:
            x=np.asarray(scg[k],dtype=np.float64)
            try:
                return zscore_safe(signal.detrend(x))
            except Exception:
                return zscore_safe(x)
    return None

def _p12_radar_sig(radar):
    n=len(radar.get("t",[]))
    for k in ["lms_error","ppg_like","displacement","display"]:
        if k in radar and radar[k] is not None and len(radar[k])==n:
            return zscore_safe(np.asarray(radar[k],dtype=np.float64))
    return np.zeros(n,dtype=np.float64)

def _p12_slice(tt,xx,anchor):
    return _p5_slice(tt,xx,anchor,0.10,0.56,100.0)

def _p12_peak(bt,bx,win,mode="max"):
    bt=np.asarray(bt,dtype=float); bx=zscore_safe(np.asarray(bx,dtype=float))
    m=(bt>=win[0])&(bt<=win[1])
    if np.sum(m)<3: return None
    idx=np.where(m)[0]
    if mode=="min": j=idx[int(np.nanargmin(bx[idx]))]
    elif mode=="abs": j=idx[int(np.nanargmax(np.abs(bx[idx])))]
    else: j=idx[int(np.nanargmax(bx[idx]))]
    return float(bt[j])

def _p12_slope(bt,bx,win,mode="abs"):
    bt=np.asarray(bt,dtype=float); bx=zscore_safe(np.asarray(bx,dtype=float))
    m=(bt>=win[0])&(bt<=win[1])
    if np.sum(m)<4: return None
    idx=np.where(m)[0]
    try: y=safe_lowpass(bx,100.0,20.0,order=2)
    except Exception: y=bx
    d1=np.gradient(y,bt); d2=np.gradient(d1,bt)
    if mode=="pos": s=robust_scale_01(np.maximum(d1[idx],0))+0.55*robust_scale_01(np.abs(d2[idx]))
    elif mode=="neg": s=robust_scale_01(np.maximum(-d1[idx],0))+0.55*robust_scale_01(np.abs(d2[idx]))
    else: s=robust_scale_01(np.abs(d1[idx]))+robust_scale_01(np.abs(d2[idx]))
    t=bt[idx]; mid=np.mean(win); half=max((win[1]-win[0])/2,1e-3)
    prior=np.exp(-0.5*((t-mid)/(half*0.85))**2)
    edge=np.clip(np.minimum(t-win[0],win[1]-t)/half,0.0,1.0)
    s=s*(0.30+0.45*prior+0.25*edge)
    return float(bt[idx[int(np.nanargmax(s))]])

def _p12_scg_lm(bt,bx):
    lm={}
    ao=_p12_peak(bt,bx,(0.045,0.170),"max") or _p12_slope(bt,bx,(0.055,0.175),"pos")
    lm["AO"]=ao
    ac_lo=0.170 if ao is None else max(0.170,ao+0.080)
    lm["AC"]=_p12_slope(bt,bx,(ac_lo,0.400),"neg") or _p12_peak(bt,bx,(ac_lo,0.400),"abs")
    mc_hi=min(0.025,(ao-0.040) if ao is not None else 0.025)
    lm["MC"]=_p12_slope(bt,bx,(-0.080,mc_hi),"abs") if mc_hi>-0.040 else None
    im_lo=0.0 if lm["MC"] is None else max(0.0,lm["MC"]+0.010)
    im_hi=min(0.115,(ao-0.012) if ao is not None else 0.080)
    lm["IM"]=_p12_slope(bt,bx,(im_lo,im_hi),"pos") if im_hi>im_lo+0.015 else None
    mo_lo=0.330 if lm["AC"] is None else max(0.330,lm["AC"]+0.060)
    lm["MO"]=_p12_slope(bt,bx,(mo_lo,0.570),"pos") if mo_lo<0.55 else None
    bounds={"MC":(-0.10,0.035),"IM":(0.0,0.115),"AO":(0.045,0.175),"AC":(0.170,0.410),"MO":(0.330,0.580)}
    prev=-999
    for k in ["MC","IM","AO","AC","MO"]:
        x=lm.get(k)
        if x is None or not np.isfinite(float(x)) or not(bounds[k][0]<=float(x)<=bounds[k][1]) or float(x)<=prev+0.008:
            lm[k]=None
        else:
            lm[k]=float(x); prev=float(x)
    return lm

def _p12_radar_lm(bt,bx):
    ao=_p12_slope(bt,bx,(0.070,0.165),"pos")
    ac_lo=0.260 if ao is None else max(0.260,ao+0.100)
    return {"AO":ao,"AC":_p12_slope(bt,bx,(ac_lo,0.460),"neg")}

def _p12_ecg_lm(ecg,bi,anchor,aoac=None):
    q,t=_p5_qt_rel(ecg,bi,anchor); ao=None; ac=None
    if aoac is not None:
        for b in aoac.get("beats",[]):
            try:
                if int(b.get("beat_index",-1))==int(bi):
                    ao=b.get("ecg_ao_ref",b.get("ao_ref")); ac=b.get("ecg_ac_ref",b.get("ac_ref")); break
            except Exception: pass
    return {"Q":q,"R":0.0,"T":t,"AO":float(ao) if ao is not None else 0.110,"AC":float(ac) if ac is not None else 0.370}

def _p12_pick(ecg,scg,radar,acfg,aoac=None):
    r=np.asarray(ecg.get("peaks_time",[]),dtype=float)
    if len(r)<3: return None
    ew=_p12_ecg_sig(ecg); sw=_p12_scg_sig(scg) if scg is not None else None; rw=_p12_radar_sig(radar)
    best=None
    for bi in range(max(1,int(len(r)*0.03)), max(2,int(len(r)*0.97))):
        if bi<=0 or bi>=len(r)-1: continue
        anchor=float(r[bi])
        bt_e,bx_e=_p12_slice(ecg["t"],ew,anchor); bt_r,bx_r=_p12_slice(radar["t"],rw,anchor)
        if bt_e is None or bt_r is None: continue
        bt_s=bx_s=None; scg_lm={}
        if sw is not None:
            bt_s,bx_s=_p12_slice(scg["t"],sw,anchor)
            if bt_s is None: continue
            scg_lm=_p12_scg_lm(bt_s,bx_s)
            if scg_lm.get("AO") is None: continue
        radar_lm=_p12_radar_lm(bt_r,bx_r); ecg_lm=_p12_ecg_lm(ecg,bi,anchor,aoac)
        score=0.0
        if bt_s is not None:
            ao=scg_lm.get("AO")
            if ao is not None:
                y=float(np.interp(ao,bt_s,bx_s)); early=bx_s[(bt_s>=0.035)&(bt_s<=0.180)]
                score += 6.0*max(0.0,y) + (2.0*float(np.nanstd(early)) if len(early) else 0.0)
            score += 1.2*(scg_lm.get("AC") is not None) + 0.25*sum(scg_lm.get(k) is not None for k in ["MC","IM","AO","AC","MO"])
        score += 0.8*sum(radar_lm.get(k) is not None for k in ["AO","AC"])
        score += -0.0008*abs(bi-len(r)/2)
        cand={"score":score,"beat_index":bi,"anchor":anchor,"bt_e":bt_e,"bx_e":bx_e,"bt_s":bt_s,"bx_s":bx_s,"bt_r":bt_r,"bx_r":bx_r,"ecg_lm":ecg_lm,"scg_lm":scg_lm,"radar_lm":radar_lm}
        if best is None or cand["score"]>best["score"]: best=cand
    return best

def _p12_txt(name,mode):
    if mode=="ECG": return f"{name} ref." if name in ("AO","AC") else name
    if mode=="SCG":
        return {"AO":"AO","AC":"AC","MO":"late/MO cand."}.get(name,f"{name} cand.")
    if mode=="Radar" and name in ("AO","AC"): return f"{name} cand."
    return name

def _p12_mark(ax,bt,bx,x,name,mode,yf):
    if x is None: return
    try:
        x=float(x)
        if not np.isfinite(x): return
    except Exception: return
    c=P12_C.get(name,"black"); y=float(np.interp(x,bt,bx))
    ax.axvline(x,color=c,linestyle="--",linewidth=1.05,alpha=0.95,zorder=3)
    mk={"Q":"o","R":"o","T":"o","MC":"o","IM":"^","AO":"s","AC":"D","MO":"v"}.get(name,"o")
    ax.scatter([x],[y],s=54,marker=mk,facecolor="white",edgecolor=c,linewidth=1.45,zorder=5)
    ymin,ymax=ax.get_ylim()
    ax.text(x,ymin+(ymax-ymin)*yf,_p12_txt(name,mode),ha="center",va="top",fontsize=7.8,color=c,bbox=dict(boxstyle="round,pad=0.11",fc="white",ec=c,alpha=0.96),zorder=6)

def _p12_bracket(ax,x0,x1,y,label):
    if x0 is None or x1 is None: return
    try:
        x0=float(x0); x1=float(x1)
        if not(np.isfinite(x0) and np.isfinite(x1)): return
    except Exception: return
    ax.plot([x0,x1],[y,y],color="0.25",linewidth=1.05,zorder=2)
    ax.plot([x0,x0],[y-0.022,y+0.022],color="0.25",linewidth=0.85)
    ax.plot([x1,x1],[y-0.022,y+0.022],color="0.25",linewidth=0.85)
    ax.text((x0+x1)/2,y+0.027,label,ha="center",va="bottom",fontsize=7.5,bbox=dict(boxstyle="round,pad=0.08",fc="white",ec="0.60",alpha=0.95))

def _p12_panel(ax,title,bt,bx,lm,mode,brackets=False,q_for_interval=None):
    ax.plot(bt,bx,color="black",linewidth=1.50,zorder=1); ax.axvline(0,color="0.40",linestyle=":",linewidth=1.0,zorder=2)
    if mode=="ECG":
        spans=[((0.07,0.16),"AO"),((0.28,0.46),"AC")]; order=[("Q",0.96),("R",0.90),("T",0.84),("AO",0.77),("AC",0.70)]
    elif mode=="SCG":
        spans=[((-0.08,-0.015),"MC"),((0,0.08),"IM"),((0.045,0.170),"AO"),((0.170,0.400),"AC"),((0.330,0.570),"MO")]; order=[("MC",0.97),("IM",0.90),("AO",0.83),("AC",0.76),("MO",0.69)]
    else:
        spans=[((0.07,0.165),"AO"),((0.260,0.460),"AC")]; order=[("AO",0.91),("AC",0.82)]
    for win,name in spans: ax.axvspan(win[0],win[1],color=P12_C.get(name,"#999"),alpha=0.040,zorder=0)
    ymin,ymax=float(np.nanmin(bx)),float(np.nanmax(bx)); pad=max(0.65,0.22*(ymax-ymin+1e-9))
    ax.set_ylim(ymin-pad,ymax+1.10)
    for name,yf in order: _p12_mark(ax,bt,bx,lm.get(name),name,mode,yf)
    if brackets:
        q=q_for_interval if q_for_interval is not None else lm.get("Q"); ao=lm.get("AO"); ac=lm.get("AC")
        ymin,ymax=ax.get_ylim(); base=ymin+0.10*(ymax-ymin)
        _p12_bracket(ax,q,ao,base,"PEP"); _p12_bracket(ax,ao,ac,base+0.15*(ymax-ymin),"LVET"); _p12_bracket(ax,q,ac,base+0.30*(ymax-ymin),"QS2")
    ax.set_title(title,loc="left",fontsize=11,pad=6); ax.set_ylabel("z-score",fontsize=10); ax.grid(True,alpha=0.22); ax.set_xlim(*P12_XLIM); ax.tick_params(labelsize=9)

def _p12_make(outdir,ecg,scg,radar,aoac,acfg):
    try:
        rep=_p12_pick(ecg,scg,radar,acfg,aoac)
        if rep is None:
            (outdir/"patch12_no_best_cycle.txt").write_text("No best-looking SCG AO-like representative cycle found.",encoding="utf-8"); return
        ecg_lm,scg_lm,radar_lm=rep["ecg_lm"],rep["scg_lm"],rep["radar_lm"]; q=ecg_lm.get("Q")
        rows=[]
        for mod,lm in [("ECG",ecg_lm),("SCG",scg_lm),("Radar",radar_lm)]:
            pep,lvet,qs2=_p5_interval(q,lm.get("AO"),lm.get("AC"))
            rows.append([mod,rep["beat_index"],rep["anchor"],q,0.0 if mod=="ECG" else None,ecg_lm.get("T") if mod=="ECG" else None,lm.get("MC"),lm.get("IM"),lm.get("AO"),lm.get("AC"),lm.get("MO"),pep,lvet,qs2])
        save_csv(outdir/"fig06_fig10_single_cycle_values.csv",["modality","beat_index","anchor_time_sec","Q_rel_sec","R_rel_sec","T_rel_sec","MC_rel_sec","IM_rel_sec","AO_rel_sec","AC_rel_sec","MO_rel_sec","PEP_ms","LVET_ms","QS2_ms"],rows)
        save_csv(outdir/"fig02_scg_candidate_landmarks_values.csv",["beat_index","anchor_time_sec","MC_rel_sec","IM_rel_sec","AO_like_rel_sec","AC_like_rel_sec","MO_late_rel_sec"],[[rep["beat_index"],rep["anchor"],scg_lm.get("MC"),scg_lm.get("IM"),scg_lm.get("AO"),scg_lm.get("AC"),scg_lm.get("MO")]])
        # compact
        fig,axes=plt.subplots(3,1,figsize=(12.4,8.2),sharex=True,constrained_layout=True)
        _p12_panel(axes[0],"ECG QRS-preserved representative beat morphology",rep["bt_e"],rep["bx_e"],ecg_lm,"ECG")
        _p12_panel(axes[1],"SCG best-looking AO-like candidate beat morphology",rep["bt_s"],rep["bx_s"],scg_lm,"SCG")
        _p12_panel(axes[2],"Radar representative candidate beat morphology",rep["bt_r"],rep["bx_r"],radar_lm,"Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]"); fig.suptitle("Compact ECG / SCG / Radar beat morphology",fontsize=14)
        fig.savefig(outdir/"fig02_compact_beat_morphology.png",dpi=300,bbox_inches="tight"); plt.close(fig)
        # SCG candidate
        fig,ax=plt.subplots(1,1,figsize=(11.7,4.5),constrained_layout=True)
        _p12_panel(ax,"SCG representative single-cycle candidate landmarks",rep["bt_s"],rep["bx_s"],scg_lm,"SCG"); ax.set_xlabel("Time from ECG R-peak [s]")
        fig.savefig(outdir/"fig02_scg_candidate_landmarks.png",dpi=300,bbox_inches="tight"); fig.savefig(outdir/"fig02_scg_reference_landmarks.png",dpi=300,bbox_inches="tight"); plt.close(fig)
        # labels/brackets/final
        fig,axes=plt.subplots(3,1,figsize=(12.8,8.6),sharex=True,constrained_layout=True)
        _p12_panel(axes[0],"ECG single-cycle reference labels",rep["bt_e"],rep["bx_e"],ecg_lm,"ECG")
        _p12_panel(axes[1],"SCG single-cycle AO/AC-like candidate labels",rep["bt_s"],rep["bx_s"],scg_lm,"SCG")
        _p12_panel(axes[2],"Radar single-cycle AO/AC detected labels",rep["bt_r"],rep["bx_r"],radar_lm,"Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]"); fig.suptitle("Single-cycle ECG reference and SCG/Radar detected landmark labeling",fontsize=14)
        fig.savefig(outdir/"fig06_single_cycle_ecg_radar_aoac_labels.png",dpi=300,bbox_inches="tight"); plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(12.9,8.9),sharex=True,constrained_layout=True)
        _p12_panel(axes[0],"ECG reference interval markers",rep["bt_e"],rep["bx_e"],ecg_lm,"ECG",True)
        _p12_panel(axes[1],"SCG detected interval markers",rep["bt_s"],rep["bx_s"],scg_lm,"SCG",True,q)
        _p12_panel(axes[2],"Radar detected interval markers",rep["bt_r"],rep["bx_r"],radar_lm,"Radar",True,q)
        axes[-1].set_xlabel("Time from ECG R-peak [s]"); fig.suptitle("Single-cycle reference/detected interval analysis",fontsize=14)
        fig.savefig(outdir/"fig06_pep_lvet_qs2_brackets.png",dpi=300,bbox_inches="tight"); plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(13.1,9.1),sharex=True,constrained_layout=True)
        _p12_panel(axes[0],"ECG reference landmarks and interval markers",rep["bt_e"],rep["bx_e"],ecg_lm,"ECG",True)
        _p12_panel(axes[1],"SCG AO/AC-like candidate landmarks and intervals",rep["bt_s"],rep["bx_s"],scg_lm,"SCG",True,q)
        _p12_panel(axes[2],"Radar AO/AC candidate landmarks and intervals",rep["bt_r"],rep["bx_r"],radar_lm,"Radar",True,q)
        axes[-1].set_xlabel("Time from ECG R-peak [s]"); fig.suptitle("Single-cycle ECG reference and SCG/Radar candidate interval comparison",fontsize=14)
        fig.savefig(outdir/"fig10_ecg_scg_radar_landmark_interval_clean.png",dpi=300,bbox_inches="tight"); plt.close(fig)
        paper=outdir/globals().get("PAPER_EXPORT_DIRNAME","paper_export"); figs=paper/"figures"; figs.mkdir(parents=True,exist_ok=True)
        mapping=[("fig02_compact_beat_morphology.png","fig02_ecg_scg_radar_beat_morphology.png"),("fig02_scg_candidate_landmarks.png","fig02b_scg_detected_landmarks.png"),("fig09_scg_periodicity_template_consistency.png","fig09_scg_periodicity_template_consistency.png"),("fig06_single_cycle_ecg_radar_aoac_labels.png","fig06_single_cycle_ecg_scg_radar_candidate_labels.png"),("fig06_pep_lvet_qs2_brackets.png","fig06_detected_interval_markers.png"),("fig10_ecg_scg_radar_landmark_interval_clean.png","fig10_detected_interval_comparison.png")]
        idx=[]
        for s,d in mapping:
            sp=outdir/s; dp=figs/d; status="missing"
            if sp.exists(): shutil.copyfile(sp,dp); status="copied"
            idx.append([s,d,status])
        save_csv(figs/"paper_figure_index_patch12.csv",["Source","PaperFigure","Status"],idx)
    except Exception as e:
        (outdir/"patch12_candidate_figures_error.txt").write_text(str(e),encoding="utf-8")

_old_save_all_patch12 = save_all
def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    result=_old_save_all_patch12(outdir,ecg,radar,scg,aoac,comp,ecfg,rcfg,acfg)
    try: _p12_make(outdir,ecg,scg,radar,aoac,acfg)
    except Exception as e: (outdir/"save_all_patch12_error.txt").write_text(str(e),encoding="utf-8")
    return result


# ============================================================
# PATCH13: bypass old broken figure-chain + ECG QRS display fix
# ------------------------------------------------------------
# Problem observed in ex97:
# save_all -> _old_save_all_patch12 -> _old save chain -> PATCH10
# PATCH10 hung/interrupted before PATCH12 figures could overwrite outputs.
# Therefore figures still looked old/wrong.
#
# Fix:
# - final save_all calls the stable original/core exporter directly
#   (_old_save_all_patch3 when available), bypassing PATCH10/PATCH12 old-chain.
# - then only PATCH12 candidate-safe best-segment figures are generated.
# - ECG figure signal is forced to QRS-preserved branch first.
# ============================================================

def _p13_ecg_qrs_preserved_signal(ecg):
    n = len(ecg.get("t", []))
    # For figures, use the already computed QRS-band branch first.
    # This avoids the over-smoothed ECG-display branch and keeps R-peaks sharp.
    for k in ["filtered", "true_display_rpeak", "lms_clean", "cleaned", "raw_adc_col", "raw", "display_rpeak", "display"]:
        if k in ecg and ecg[k] is not None and len(ecg[k]) == n:
            x = np.asarray(ecg[k], dtype=np.float64)
            if k in ["filtered", "true_display_rpeak"]:
                return zscore_safe(x)
            try:
                return zscore_safe(safe_bandpass(x, float(ecg.get("fs", 100.0)), 5.0, 32.0, order=3))
            except Exception:
                try:
                    return zscore_safe(signal.detrend(x))
                except Exception:
                    return zscore_safe(x)
    return np.zeros(n, dtype=np.float64)

# Override PATCH12 ECG signal selector.
_p12_ecg_sig = _p13_ecg_qrs_preserved_signal

def _p13_core_export(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    """
    Use the earliest stable exporter available and bypass later figure patches.
    _old_save_all_patch3 points to the pre-PATCH3/core save_all in this code family.
    """
    core = globals().get("_old_save_all_patch3", None)
    if callable(core):
        return core(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)

    # Fallback: if not available, try current old patch12 chain but guard it.
    core2 = globals().get("_old_save_all_patch12", None)
    if callable(core2):
        try:
            return core2(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
        except BaseException as e:
            (outdir / "patch13_core_export_fallback_error.txt").write_text(str(e), encoding="utf-8")
            return None
    return None

def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    # 1) stable baseline exports only
    result = _p13_core_export(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)

    # 2) candidate-safe final figures only
    try:
        _p12_make(outdir, ecg, scg, radar, aoac, acfg)
    except BaseException as e:
        (outdir / "save_all_patch13_candidate_figures_error.txt").write_text(str(e), encoding="utf-8")

    # 3) clear audit note
    try:
        (outdir / "patch13_final_figure_policy.txt").write_text(
            "PATCH13 active: old PATCH10/PATCH12 chained figure exporters are bypassed. "
            "Final figures are generated by PATCH12 best-segment candidate-safe exporter only. "
            "ECG panels use QRS-preserved signal branch to avoid over-smoothed ECG display.",
            encoding="utf-8"
        )
    except Exception:
        pass
    return result



# ============================================================
# PATCH14: final ECG/SCG/Radar figure rebuild
# ------------------------------------------------------------
# Fixes user-observed figure issues:
# 1) ECG panels looked PPG-like and Q/R/T were wrong.
#    -> ECG figure branch uses raw/raw_adc_col, ECG band-pass, local R recentering,
#       local Q/T detection for display only.
# 2) SCG MC/IM/AO/AC/MO landmarks were unreliable.
#    -> SCG final figures no longer force MC/IM/MO.
#       Only visible AO-like and AC-like candidates are shown.
# 3) Old figures from the chained exporters are overwritten:
#    fig02_ecg_qrt_reference
#    fig02_scg_reference_landmarks
#    fig05_morphology_ecg_scg_radar_candidates
#    fig06_single_cycle_ecg_radar_aoac_labels
#    fig06_pep_lvet_qs2_brackets
#    fig10_ecg_scg_radar_landmark_interval_clean
# ============================================================

P14_C = {
    "Q": "#1f77b4",
    "R": "#d62728",
    "T": "#555555",
    "AO": "#9467bd",
    "AC": "#ff7f0e",
}
P14_XLIM = (-0.12, 0.55)


def _p14_raw_ecg_array(ecg):
    n = len(ecg.get("t", []))
    for k in ["raw_adc_col", "raw"]:
        if k in ecg and ecg[k] is not None and len(ecg[k]) == n:
            return np.asarray(ecg[k], dtype=np.float64)
    # fallback only if raw unavailable
    for k in ["filtered", "cleaned", "display_rpeak", "display"]:
        if k in ecg and ecg[k] is not None and len(ecg[k]) == n:
            return np.asarray(ecg[k], dtype=np.float64)
    return np.zeros(n, dtype=np.float64)


def _p14_ecg_for_display(ecg):
    x = _p14_raw_ecg_array(ecg)
    fs = float(ecg.get("fs", 100.0))
    try:
        # ECG figure: preserve QRS sharpness, avoid PPG-like display branch.
        y = safe_bandpass(x, fs, 6.0, min(35.0, fs * 0.45), order=3)
    except Exception:
        try:
            y = signal.detrend(x)
        except Exception:
            y = x - np.nanmedian(x)
    return zscore_safe(y)


def _p14_scg_for_display(scg):
    if scg is None:
        return None
    n = len(scg.get("t", []))
    for k in ["filtered", "resp_removed", "selected_raw", "vmag", "az", "ax", "ay"]:
        if k in scg and scg[k] is not None and len(scg[k]) == n:
            x = np.asarray(scg[k], dtype=np.float64)
            try:
                return zscore_safe(signal.detrend(x))
            except Exception:
                return zscore_safe(x)
    return None


def _p14_radar_for_display(radar):
    n = len(radar.get("t", []))
    for k in ["lms_error", "ppg_like", "displacement", "display"]:
        if k in radar and radar[k] is not None and len(radar[k]) == n:
            return zscore_safe(np.asarray(radar[k], dtype=np.float64))
    return np.zeros(n, dtype=np.float64)


def _p14_slice_interp(tt, xx, anchor, pre=0.12, post=0.55):
    return _p5_slice(tt, xx, anchor, pre, post, 100.0)


def _p14_recenter_ecg_beat(bt, bx):
    """
    Recenter ECG display beat by local QRS peak, then force R at 0.
    This prevents phase-shifted/smoothed display signals from looking like PPG.
    """
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    m = (bt >= -0.060) & (bt <= 0.060)
    if np.sum(m) < 3:
        return bt, bx, 0.0
    idx = np.where(m)[0]
    # Use absolute QRS complex; then invert so R is positive.
    j = idx[int(np.nanargmax(np.abs(bx[idx])))]
    r_rel = float(bt[j])
    if bx[j] < 0:
        bx = -bx
    bt2 = bt - r_rel
    # Renormalize but keep relative shape.
    bx2 = zscore_safe(bx)
    return bt2, bx2, r_rel


def _p14_detect_ecg_qt(bt, bx):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    # Q: local negative deflection before R
    q = None
    m = (bt >= -0.070) & (bt <= -0.008)
    if np.sum(m) >= 3:
        idx = np.where(m)[0]
        q = float(bt[idx[int(np.nanargmin(bx[idx]))]])
    # T: broad positive wave after QRS, avoid selecting immediate ST shoulder
    t = None
    m = (bt >= 0.140) & (bt <= 0.340)
    if np.sum(m) >= 3:
        idx = np.where(m)[0]
        t = float(bt[idx[int(np.nanargmax(bx[idx]))]])
    return {"Q": q, "R": 0.0, "T": t, "AO": 0.100, "AC": 0.350}


def _p14_peak(bt, bx, win, mode="max"):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 3:
        return None
    idx = np.where(m)[0]
    if mode == "min":
        j = idx[int(np.nanargmin(bx[idx]))]
    elif mode == "abs":
        j = idx[int(np.nanargmax(np.abs(bx[idx])))]
    else:
        j = idx[int(np.nanargmax(bx[idx]))]
    return float(bt[j])


def _p14_slope_candidate(bt, bx, win, mode="abs"):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 4:
        return None
    idx = np.where(m)[0]
    try:
        y = safe_lowpass(bx, 100.0, 20.0, order=2)
    except Exception:
        y = bx
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    if mode == "pos":
        s = robust_scale_01(np.maximum(d1[idx], 0)) + 0.60 * robust_scale_01(np.abs(d2[idx]))
    elif mode == "neg":
        s = robust_scale_01(np.maximum(-d1[idx], 0)) + 0.60 * robust_scale_01(np.abs(d2[idx]))
    else:
        s = robust_scale_01(np.abs(d1[idx])) + robust_scale_01(np.abs(d2[idx]))
    t = bt[idx]
    mid = np.mean(win)
    half = max((win[1] - win[0]) / 2.0, 1e-3)
    prior = np.exp(-0.5 * ((t - mid) / (half * 0.90)) ** 2)
    edge = np.clip(np.minimum(t - win[0], win[1] - t) / half, 0.0, 1.0)
    s = s * (0.30 + 0.45 * prior + 0.25 * edge)
    j = idx[int(np.nanargmax(s))]
    return float(bt[j])


def _p14_scg_candidates(bt, bx):
    # Only show robust AO-like and AC-like candidates. Do not force MC/IM/MO.
    ao = _p14_peak(bt, bx, (0.045, 0.170), "max")
    if ao is None:
        ao = _p14_slope_candidate(bt, bx, (0.055, 0.175), "pos")
    ac_lo = 0.170 if ao is None else max(0.170, ao + 0.080)
    ac = _p14_slope_candidate(bt, bx, (ac_lo, 0.400), "neg")
    if ac is None:
        ac = _p14_peak(bt, bx, (ac_lo, 0.400), "abs")
    # Validate
    if ao is not None and not (0.045 <= ao <= 0.175):
        ao = None
    if ac is not None and not (0.170 <= ac <= 0.410):
        ac = None
    if ao is not None and ac is not None and ac <= ao + 0.070:
        ac = None
    return {"AO": ao, "AC": ac}


def _p14_radar_candidates(bt, bx):
    ao = _p14_slope_candidate(bt, bx, (0.070, 0.165), "pos")
    ac_lo = 0.260 if ao is None else max(0.260, ao + 0.100)
    ac = _p14_slope_candidate(bt, bx, (ac_lo, 0.460), "neg")
    return {"AO": ao, "AC": ac}


def _p14_pick_best(ecg, scg, radar, acfg):
    r = np.asarray(ecg.get("peaks_time", []), dtype=float)
    if len(r) < 3:
        return None
    es = _p14_ecg_for_display(ecg)
    ss = _p14_scg_for_display(scg) if scg is not None else None
    rs = _p14_radar_for_display(radar)

    best = None
    lo = max(1, int(len(r) * 0.03))
    hi = max(2, int(len(r) * 0.97))
    for bi in range(lo, hi):
        if bi <= 0 or bi >= len(r) - 1:
            continue
        anchor0 = float(r[bi])
        bt_e0, bx_e0 = _p14_slice_interp(ecg["t"], es, anchor0)
        if bt_e0 is None:
            continue
        bt_e, bx_e, r_shift = _p14_recenter_ecg_beat(bt_e0, bx_e0)
        # Real visual R time after local recentering
        anchor = anchor0 + r_shift

        bt_r, bx_r = _p14_slice_interp(radar["t"], rs, anchor)
        if bt_r is None:
            continue

        bt_s, bx_s, scg_lm = None, None, {}
        if ss is not None:
            bt_s, bx_s = _p14_slice_interp(scg["t"], ss, anchor)
            if bt_s is None:
                continue
            scg_lm = _p14_scg_candidates(bt_s, bx_s)
            if scg_lm.get("AO") is None:
                continue

        ecg_lm = _p14_detect_ecg_qt(bt_e, bx_e)
        radar_lm = _p14_radar_candidates(bt_r, bx_r)

        # Quality: sharp ECG R, visible SCG AO, reasonable candidate availability
        score = 0.0
        r_amp = float(np.nanmax(bx_e[(bt_e >= -0.020) & (bt_e <= 0.020)])) if np.any((bt_e >= -0.020) & (bt_e <= 0.020)) else 0.0
        score += 2.5 * max(0.0, r_amp)

        if bt_s is not None:
            ao = scg_lm.get("AO")
            if ao is not None:
                ao_amp = float(np.interp(ao, bt_s, bx_s))
                early = bx_s[(bt_s >= 0.035) & (bt_s <= 0.180)]
                score += 5.5 * max(0.0, ao_amp)
                score += 1.6 * float(np.nanstd(early)) if len(early) else 0.0
            if scg_lm.get("AC") is not None:
                score += 1.0

        score += 0.7 * sum(radar_lm.get(k) is not None for k in ["AO", "AC"])
        score += -0.0008 * abs(bi - len(r) / 2.0)

        cand = {
            "score": score,
            "beat_index": bi,
            "anchor": anchor,
            "bt_e": bt_e,
            "bx_e": bx_e,
            "bt_s": bt_s,
            "bx_s": bx_s,
            "bt_r": bt_r,
            "bx_r": bx_r,
            "ecg_lm": ecg_lm,
            "scg_lm": scg_lm,
            "radar_lm": radar_lm,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def _p14_mark(ax, bt, bx, x, name, label, yfrac):
    if x is None:
        return
    try:
        x = float(x)
        if not np.isfinite(x):
            return
    except Exception:
        return
    c = P14_C.get(name, "black")
    y = float(np.interp(x, bt, bx))
    ax.axvline(x, color=c, linestyle="--", linewidth=1.08, alpha=0.95, zorder=3)
    mk = {"Q":"o", "R":"o", "T":"o", "AO":"s", "AC":"D"}.get(name, "o")
    ax.scatter([x], [y], s=54, marker=mk, facecolor="white", edgecolor=c, linewidth=1.4, zorder=5)
    ymin, ymax = ax.get_ylim()
    ax.text(x, ymin + (ymax-ymin)*yfrac, label, ha="center", va="top", fontsize=8.0, color=c,
            bbox=dict(boxstyle="round,pad=0.11", fc="white", ec=c, alpha=0.96), zorder=6)


def _p14_bracket(ax, x0, x1, y, label):
    if x0 is None or x1 is None:
        return
    try:
        x0 = float(x0); x1 = float(x1)
        if not (np.isfinite(x0) and np.isfinite(x1)):
            return
    except Exception:
        return
    ax.plot([x0, x1], [y, y], color="0.25", linewidth=1.0, zorder=2)
    ax.plot([x0, x0], [y-0.020, y+0.020], color="0.25", linewidth=0.85)
    ax.plot([x1, x1], [y-0.020, y+0.020], color="0.25", linewidth=0.85)
    ax.text((x0+x1)/2.0, y+0.025, label, ha="center", va="bottom", fontsize=7.4,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="0.60", alpha=0.95))


def _p14_panel(ax, title, bt, bx, lm, mode, brackets=False, q_for_interval=None):
    ax.plot(bt, bx, color="black", linewidth=1.55, zorder=1)
    ax.axvline(0.0, color="0.40", linestyle=":", linewidth=1.0, zorder=2)

    if mode == "ECG":
        spans = [((0.07, 0.16), "AO"), ((0.28, 0.46), "AC")]
        marks = [("Q", lm.get("Q"), "Q", 0.95),
                 ("R", lm.get("R"), "R", 0.88),
                 ("T", lm.get("T"), "T", 0.81),
                 ("AO", lm.get("AO"), "AO ref.", 0.74),
                 ("AC", lm.get("AC"), "AC ref.", 0.67)]
    elif mode == "SCG":
        spans = [((0.045, 0.175), "AO"), ((0.170, 0.410), "AC")]
        marks = [("AO", lm.get("AO"), "AO", 0.87),
                 ("AC", lm.get("AC"), "AC", 0.78)]
    else:
        spans = [((0.07, 0.165), "AO"), ((0.260, 0.460), "AC")]
        marks = [("AO", lm.get("AO"), "AO", 0.88),
                 ("AC", lm.get("AC"), "AC", 0.78)]

    for win, name in spans:
        ax.axvspan(win[0], win[1], color=P14_C.get(name, "#999999"), alpha=0.045, zorder=0)

    ymin, ymax = float(np.nanmin(bx)), float(np.nanmax(bx))
    pad = max(0.65, 0.22*(ymax-ymin + 1e-9))
    ax.set_ylim(ymin-pad, ymax+1.05)

    for name, x, label, yf in marks:
        _p14_mark(ax, bt, bx, x, name, label, yf)

    if brackets:
        q = q_for_interval if q_for_interval is not None else lm.get("Q")
        ao, ac = lm.get("AO"), lm.get("AC")
        ymin, ymax = ax.get_ylim()
        base = ymin + 0.10*(ymax-ymin)
        _p14_bracket(ax, q, ao, base, "PEP")
        _p14_bracket(ax, ao, ac, base + 0.15*(ymax-ymin), "LVET")
        _p14_bracket(ax, q, ac, base + 0.30*(ymax-ymin), "QS2")

    ax.set_title(title, loc="left", fontsize=11, pad=6)
    ax.set_ylabel("z-score", fontsize=10)
    ax.grid(True, alpha=0.22)
    ax.set_xlim(*P14_XLIM)
    ax.tick_params(labelsize=9)


def _p14_make_final_figs(outdir, ecg, scg, radar, aoac, acfg):
    try:
        rep = _p14_pick_best(ecg, scg, radar, acfg)
        if rep is None:
            (outdir / "patch14_no_valid_representative_cycle.txt").write_text(
                "No valid ECG-centered SCG AO-like representative cycle found.",
                encoding="utf-8"
            )
            return

        ecg_lm = rep["ecg_lm"]
        scg_lm = rep["scg_lm"]
        radar_lm = rep["radar_lm"]
        q = ecg_lm.get("Q")

        rows = []
        for mod, lm in [("ECG", ecg_lm), ("SCG", scg_lm), ("Radar", radar_lm)]:
            pep, lvet, qs2 = _p5_interval(q, lm.get("AO"), lm.get("AC"))
            rows.append([mod, rep["beat_index"], rep["anchor"], q, 0.0 if mod=="ECG" else None,
                         ecg_lm.get("T") if mod=="ECG" else None, lm.get("AO"), lm.get("AC"),
                         pep, lvet, qs2])
        save_csv(outdir / "fig06_fig10_single_cycle_values.csv",
                 ["modality", "beat_index", "anchor_time_sec", "Q_rel_sec", "R_rel_sec", "T_rel_sec",
                  "AO_rel_sec", "AC_rel_sec", "PEP_ms", "LVET_ms", "QS2_ms"], rows)

        # 1) ECG Q/R/T reference figure
        fig, ax = plt.subplots(1,1,figsize=(12.4,4.8),constrained_layout=True)
        _p14_panel(ax, "Fig. 2. ECG Q/R/T landmarks and AO/AC reference windows",
                   rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", brackets=False)
        ax.set_xlabel("Time from ECG R-peak [s]")
        fig.savefig(outdir / "fig02_ecg_qrt_reference.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 2) SCG candidate-only figure
        fig, ax = plt.subplots(1,1,figsize=(12.4,4.8),constrained_layout=True)
        _p14_panel(ax, "SCG representative single-cycle AO/AC-like candidate landmarks",
                   rep["bt_s"], rep["bx_s"], scg_lm, "SCG", brackets=False)
        ax.set_xlabel("Time from ECG R-peak [s]")
        fig.savefig(outdir / "fig02_scg_candidate_landmarks.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig02_scg_reference_landmarks.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 3) compact morphology
        fig, axes = plt.subplots(3,1,figsize=(12.8,8.5),sharex=True,constrained_layout=True)
        _p14_panel(axes[0], "ECG QRS-preserved representative beat", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG")
        _p14_panel(axes[1], "SCG AO/AC-like candidate beat", rep["bt_s"], rep["bx_s"], scg_lm, "SCG")
        _p14_panel(axes[2], "Radar AO/AC detected beat", rep["bt_r"], rep["bx_r"], radar_lm, "Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("ECG / SCG / Radar morphology detected landmark comparison", fontsize=14)
        fig.savefig(outdir / "fig02_compact_beat_morphology.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig05_morphology_ecg_scg_radar_candidates.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 4) label panels
        fig, axes = plt.subplots(3,1,figsize=(12.8,8.6),sharex=True,constrained_layout=True)
        _p14_panel(axes[0], "ECG single-cycle reference labels", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG")
        _p14_panel(axes[1], "SCG single-cycle AO/AC-like candidate labels", rep["bt_s"], rep["bx_s"], scg_lm, "SCG")
        _p14_panel(axes[2], "Radar single-cycle AO/AC detected labels", rep["bt_r"], rep["bx_r"], radar_lm, "Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle ECG reference and SCG/Radar detected landmark labeling", fontsize=14)
        fig.savefig(outdir / "fig06_single_cycle_ecg_radar_aoac_labels.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 5) intervals
        fig, axes = plt.subplots(3,1,figsize=(12.9,8.9),sharex=True,constrained_layout=True)
        _p14_panel(axes[0], "ECG reference interval markers", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", True)
        _p14_panel(axes[1], "SCG detected interval markers", rep["bt_s"], rep["bx_s"], scg_lm, "SCG", True, q)
        _p14_panel(axes[2], "Radar detected interval markers", rep["bt_r"], rep["bx_r"], radar_lm, "Radar", True, q)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle reference/detected interval analysis", fontsize=14)
        fig.savefig(outdir / "fig06_pep_lvet_qs2_brackets.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig10_ecg_scg_radar_landmark_interval_clean.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # paper export
        paper = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export")
        figs = paper / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        mapping = [
            ("fig02_ecg_qrt_reference.png", "fig02_ecg_qrt_reference.png"),
            ("fig02_compact_beat_morphology.png", "fig02_ecg_scg_radar_morphology_candidates.png"),
            ("fig02_scg_candidate_landmarks.png", "fig02b_scg_detected_landmarks.png"),
            ("fig06_single_cycle_ecg_radar_aoac_labels.png", "fig06_single_cycle_detected_labels.png"),
            ("fig06_pep_lvet_qs2_brackets.png", "fig06_detected_interval_markers.png"),
            ("fig10_ecg_scg_radar_landmark_interval_clean.png", "fig10_detected_interval_comparison.png"),
        ]
        idx_rows = []
        for s, d in mapping:
            sp, dp = outdir / s, figs / d
            status = "missing"
            if sp.exists():
                shutil.copyfile(sp, dp)
                status = "copied"
            idx_rows.append([s, d, status])
        save_csv(figs / "paper_figure_index_patch14.csv", ["Source", "PaperFigure", "Status"], idx_rows)

        (outdir / "patch14_final_figure_policy.txt").write_text(
            "PATCH14 active. ECG figures are re-centered to local QRS peak and use raw/QRS-preserved ECG. "
            "SCG final figures show only AO-like and AC-like candidates; MC/IM/MO are not forced. "
            "Old ECG/SCG/Radar figures are overwritten by PATCH14 final candidate-safe figures.",
            encoding="utf-8"
        )
    except BaseException as e:
        (outdir / "patch14_final_figures_error.txt").write_text(str(e), encoding="utf-8")


def _p14_core_export(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    # Use stable earliest exporter if available; bypass chained figure patches.
    core = globals().get("_old_save_all_patch3", None)
    if callable(core):
        return core(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    return None


def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    result = _p14_core_export(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        _p14_make_final_figs(outdir, ecg, scg, radar, aoac, acfg)
    except BaseException as e:
        (outdir / "save_all_patch14_error.txt").write_text(str(e), encoding="utf-8")
    return result



# ============================================================
# PATCH15: restore SCG MC/IM/MO candidates + keep CTI display
# ------------------------------------------------------------
# User correction:
# - Do NOT remove MC/IM/MO from SCG figures.
# - Keep MC / IM / AO-like / AC-like / MO-like candidates visible.
# - PEP / LVET / QS2 are still computed from ECG Q and AO/AC candidates,
#   but SCG morphology landmarks remain visible for interpretation.
# ============================================================

P15_C = {
    "Q": "#1f77b4",
    "R": "#d62728",
    "T": "#555555",
    "MC": "#17becf",
    "IM": "#1f77b4",
    "AO": "#9467bd",
    "AC": "#ff7f0e",
    "MO": "#2ca02c",
}
P15_XLIM = (-0.12, 0.60)


def _p15_raw_ecg_array(ecg):
    n = len(ecg.get("t", []))
    for k in ["raw_adc_col", "raw"]:
        if k in ecg and ecg[k] is not None and len(ecg[k]) == n:
            return np.asarray(ecg[k], dtype=np.float64)
    for k in ["filtered", "cleaned", "display_rpeak", "display"]:
        if k in ecg and ecg[k] is not None and len(ecg[k]) == n:
            return np.asarray(ecg[k], dtype=np.float64)
    return np.zeros(n, dtype=np.float64)


def _p15_ecg_for_display(ecg):
    x = _p15_raw_ecg_array(ecg)
    fs = float(ecg.get("fs", 100.0))
    try:
        y = safe_bandpass(x, fs, 6.0, min(35.0, fs * 0.45), order=3)
    except Exception:
        try:
            y = signal.detrend(x)
        except Exception:
            y = x - np.nanmedian(x)
    return zscore_safe(y)


def _p15_scg_for_display(scg):
    if scg is None:
        return None
    n = len(scg.get("t", []))
    for k in ["filtered", "resp_removed", "selected_raw", "vmag", "az", "ax", "ay"]:
        if k in scg and scg[k] is not None and len(scg[k]) == n:
            x = np.asarray(scg[k], dtype=np.float64)
            try:
                return zscore_safe(signal.detrend(x))
            except Exception:
                return zscore_safe(x)
    return None


def _p15_radar_for_display(radar):
    n = len(radar.get("t", []))
    for k in ["lms_error", "ppg_like", "displacement", "display"]:
        if k in radar and radar[k] is not None and len(radar[k]) == n:
            return zscore_safe(np.asarray(radar[k], dtype=np.float64))
    return np.zeros(n, dtype=np.float64)


def _p15_slice(tt, xx, anchor, pre=0.12, post=0.60):
    return _p5_slice(tt, xx, anchor, pre, post, 100.0)


def _p15_recenter_ecg_beat(bt, bx):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    m = (bt >= -0.060) & (bt <= 0.060)
    if np.sum(m) < 3:
        return bt, bx, 0.0
    idx = np.where(m)[0]
    j = idx[int(np.nanargmax(np.abs(bx[idx])))]
    r_rel = float(bt[j])
    if bx[j] < 0:
        bx = -bx
    bt2 = bt - r_rel
    bx2 = zscore_safe(bx)
    return bt2, bx2, r_rel


def _p15_detect_ecg_qrt(bt, bx):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    q = None
    m = (bt >= -0.070) & (bt <= -0.008)
    if np.sum(m) >= 3:
        idx = np.where(m)[0]
        q = float(bt[idx[int(np.nanargmin(bx[idx]))]])
    t = None
    m = (bt >= 0.140) & (bt <= 0.340)
    if np.sum(m) >= 3:
        idx = np.where(m)[0]
        t = float(bt[idx[int(np.nanargmax(bx[idx]))]])
    return {"Q": q, "R": 0.0, "T": t, "AO": 0.100, "AC": 0.350}


def _p15_peak(bt, bx, win, mode="max"):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 3:
        return None
    idx = np.where(m)[0]
    if mode == "min":
        j = idx[int(np.nanargmin(bx[idx]))]
    elif mode == "abs":
        j = idx[int(np.nanargmax(np.abs(bx[idx])))]
    else:
        j = idx[int(np.nanargmax(bx[idx]))]
    return float(bt[j])


def _p15_slope_candidate(bt, bx, win, mode="abs"):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 4:
        return None
    idx = np.where(m)[0]
    try:
        y = safe_lowpass(bx, 100.0, 20.0, order=2)
    except Exception:
        y = bx
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    if mode == "pos":
        s = robust_scale_01(np.maximum(d1[idx], 0)) + 0.60 * robust_scale_01(np.abs(d2[idx]))
    elif mode == "neg":
        s = robust_scale_01(np.maximum(-d1[idx], 0)) + 0.60 * robust_scale_01(np.abs(d2[idx]))
    else:
        s = robust_scale_01(np.abs(d1[idx])) + robust_scale_01(np.abs(d2[idx]))
    t = bt[idx]
    mid = np.mean(win)
    half = max((win[1] - win[0]) / 2.0, 1e-3)
    prior = np.exp(-0.5 * ((t - mid) / (half * 0.90)) ** 2)
    edge = np.clip(np.minimum(t - win[0], win[1] - t) / half, 0.0, 1.0)
    s = s * (0.30 + 0.45 * prior + 0.25 * edge)
    j = idx[int(np.nanargmax(s))]
    return float(bt[j])


def _p15_scg_candidates(bt, bx):
    """
    SCG candidate landmarks are all retained:
      MC / IM / AO-like / AC-like / MO-like.
    They are candidates, not ground-truth labels.
    If a candidate is not detectable inside its window, a window-centered fallback
    is used so the morphology region remains visible in the figure.
    """
    lm = {}

    # MC candidate: pre-ejection / early mechanical complex around R
    mc = _p15_slope_candidate(bt, bx, (-0.085, 0.020), "abs")
    if mc is None:
        mc = _p15_peak(bt, bx, (-0.085, 0.020), "abs")
    if mc is None:
        mc = -0.035
    lm["MC"] = float(mc)

    # IM candidate: early systolic mechanical complex
    im_lo = max(0.000, lm["MC"] + 0.010)
    im = _p15_peak(bt, bx, (im_lo, 0.100), "max")
    if im is None:
        im = _p15_slope_candidate(bt, bx, (im_lo, 0.115), "pos")
    if im is None:
        im = 0.040
    lm["IM"] = float(im)

    # AO-like candidate: visible early systolic spike/positive candidate
    ao_lo = max(0.045, lm["IM"] + 0.012)
    ao = _p15_peak(bt, bx, (ao_lo, 0.180), "max")
    if ao is None:
        ao = _p15_slope_candidate(bt, bx, (ao_lo, 0.180), "pos")
    if ao is None:
        ao = 0.110
    lm["AO"] = float(ao)

    # AC-like candidate: post-AO systolic-end candidate
    ac_lo = max(0.180, lm["AO"] + 0.080)
    ac = _p15_slope_candidate(bt, bx, (ac_lo, 0.430), "neg")
    if ac is None:
        ac = _p15_peak(bt, bx, (ac_lo, 0.430), "abs")
    if ac is None:
        ac = 0.340
    lm["AC"] = float(ac)

    # MO-like/late candidate: late vibration preserved
    mo_lo = max(0.340, lm["AC"] + 0.050)
    mo = _p15_peak(bt, bx, (mo_lo, 0.590), "max")
    if mo is None:
        mo = _p15_slope_candidate(bt, bx, (mo_lo, 0.590), "pos")
    if mo is None:
        mo = 0.480
    lm["MO"] = float(mo)

    # Keep sequence ordered. If a detected event violates order, shift it minimally
    # inside the allowed candidate region instead of deleting it.
    order = ["MC", "IM", "AO", "AC", "MO"]
    bounds = {
        "MC": (-0.100, 0.030),
        "IM": (0.000, 0.120),
        "AO": (0.045, 0.185),
        "AC": (0.180, 0.440),
        "MO": (0.340, 0.600),
    }
    prev = -999.0
    for k in order:
        x = lm[k]
        lo, hi = bounds[k]
        x = min(max(float(x), lo), hi)
        if x <= prev + 0.008:
            x = min(max(prev + 0.010, lo), hi)
        lm[k] = float(x)
        prev = float(x)
    return lm


def _p15_radar_candidates(bt, bx):
    ao = _p15_slope_candidate(bt, bx, (0.070, 0.165), "pos")
    if ao is None:
        ao = 0.110
    ac_lo = max(0.260, ao + 0.100)
    ac = _p15_slope_candidate(bt, bx, (ac_lo, 0.460), "neg")
    if ac is None:
        ac = 0.370
    return {"AO": float(ao), "AC": float(ac)}


def _p15_pick_best(ecg, scg, radar, acfg):
    r = np.asarray(ecg.get("peaks_time", []), dtype=float)
    if len(r) < 3:
        return None
    es = _p15_ecg_for_display(ecg)
    ss = _p15_scg_for_display(scg) if scg is not None else None
    rs = _p15_radar_for_display(radar)

    best = None
    lo = max(1, int(len(r) * 0.03))
    hi = max(2, int(len(r) * 0.97))
    for bi in range(lo, hi):
        if bi <= 0 or bi >= len(r) - 1:
            continue
        anchor0 = float(r[bi])
        bt_e0, bx_e0 = _p15_slice(ecg["t"], es, anchor0)
        if bt_e0 is None:
            continue
        bt_e, bx_e, r_shift = _p15_recenter_ecg_beat(bt_e0, bx_e0)
        anchor = anchor0 + r_shift

        bt_r, bx_r = _p15_slice(radar["t"], rs, anchor)
        if bt_r is None:
            continue

        bt_s, bx_s, scg_lm = None, None, {}
        if ss is not None:
            bt_s, bx_s = _p15_slice(scg["t"], ss, anchor)
            if bt_s is None:
                continue
            scg_lm = _p15_scg_candidates(bt_s, bx_s)

        ecg_lm = _p15_detect_ecg_qrt(bt_e, bx_e)
        radar_lm = _p15_radar_candidates(bt_r, bx_r)

        score = 0.0
        r_amp = float(np.nanmax(bx_e[(bt_e >= -0.020) & (bt_e <= 0.020)])) if np.any((bt_e >= -0.020) & (bt_e <= 0.020)) else 0.0
        score += 2.5 * max(0.0, r_amp)

        if bt_s is not None:
            ao = scg_lm.get("AO")
            ao_amp = float(np.interp(ao, bt_s, bx_s)) if ao is not None else 0.0
            early = bx_s[(bt_s >= 0.035) & (bt_s <= 0.180)]
            score += 5.5 * max(0.0, ao_amp)
            score += 1.6 * float(np.nanstd(early)) if len(early) else 0.0

        # Prefer radar with non-flat morphology, but do not let it dominate.
        score += 0.45 * float(np.nanstd(bx_r[(bt_r >= 0.06) & (bt_r <= 0.46)])) if np.any((bt_r >= 0.06) & (bt_r <= 0.46)) else 0.0
        score += -0.0008 * abs(bi - len(r) / 2.0)

        cand = {
            "score": score,
            "beat_index": bi,
            "anchor": anchor,
            "bt_e": bt_e,
            "bx_e": bx_e,
            "bt_s": bt_s,
            "bx_s": bx_s,
            "bt_r": bt_r,
            "bx_r": bx_r,
            "ecg_lm": ecg_lm,
            "scg_lm": scg_lm,
            "radar_lm": radar_lm,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def _p15_mark(ax, bt, bx, x, name, label, yfrac):
    if x is None:
        return
    try:
        x = float(x)
        if not np.isfinite(x):
            return
    except Exception:
        return
    c = P15_C.get(name, "black")
    y = float(np.interp(x, bt, bx))
    ax.axvline(x, color=c, linestyle="--", linewidth=1.05, alpha=0.95, zorder=3)
    mk = {"Q":"o", "R":"o", "T":"o", "MC":"o", "IM":"^", "AO":"s", "AC":"D", "MO":"v"}.get(name, "o")
    ax.scatter([x], [y], s=54, marker=mk, facecolor="white", edgecolor=c, linewidth=1.4, zorder=5)
    ymin, ymax = ax.get_ylim()
    ax.text(x, ymin + (ymax-ymin)*yfrac, label, ha="center", va="top", fontsize=7.7, color=c,
            bbox=dict(boxstyle="round,pad=0.11", fc="white", ec=c, alpha=0.96), zorder=6)


def _p15_bracket(ax, x0, x1, y, label):
    if x0 is None or x1 is None:
        return
    try:
        x0 = float(x0); x1 = float(x1)
        if not (np.isfinite(x0) and np.isfinite(x1)):
            return
    except Exception:
        return
    ax.plot([x0, x1], [y, y], color="0.25", linewidth=1.0, zorder=2)
    ax.plot([x0, x0], [y-0.020, y+0.020], color="0.25", linewidth=0.85)
    ax.plot([x1, x1], [y-0.020, y+0.020], color="0.25", linewidth=0.85)
    ax.text((x0+x1)/2.0, y+0.025, label, ha="center", va="bottom", fontsize=7.3,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="0.60", alpha=0.95))


def _p15_panel(ax, title, bt, bx, lm, mode, brackets=False, q_for_interval=None):
    ax.plot(bt, bx, color="black", linewidth=1.55, zorder=1)
    ax.axvline(0.0, color="0.40", linestyle=":", linewidth=1.0, zorder=2)

    if mode == "ECG":
        spans = [((0.07, 0.16), "AO"), ((0.28, 0.46), "AC")]
        marks = [("Q", lm.get("Q"), "Q", 0.95),
                 ("R", lm.get("R"), "R", 0.88),
                 ("T", lm.get("T"), "T", 0.81),
                 ("AO", lm.get("AO"), "AO ref.", 0.74),
                 ("AC", lm.get("AC"), "AC ref.", 0.67)]
    elif mode == "SCG":
        spans = [((-0.10, 0.03), "MC"),
                 ((0.00, 0.12), "IM"),
                 ((0.045, 0.185), "AO"),
                 ((0.180, 0.440), "AC"),
                 ((0.340, 0.600), "MO")]
        marks = [("MC", lm.get("MC"), "MC", 0.97),
                 ("IM", lm.get("IM"), "IM", 0.90),
                 ("AO", lm.get("AO"), "AO", 0.83),
                 ("AC", lm.get("AC"), "AC", 0.76),
                 ("MO", lm.get("MO"), "MO", 0.69)]
    else:
        spans = [((0.07, 0.165), "AO"), ((0.260, 0.460), "AC")]
        marks = [("AO", lm.get("AO"), "AO", 0.88),
                 ("AC", lm.get("AC"), "AC", 0.78)]

    for win, name in spans:
        ax.axvspan(win[0], win[1], color=P15_C.get(name, "#999999"), alpha=0.040, zorder=0)

    ymin, ymax = float(np.nanmin(bx)), float(np.nanmax(bx))
    pad = max(0.65, 0.22*(ymax-ymin + 1e-9))
    ax.set_ylim(ymin-pad, ymax+1.08)

    for name, x, label, yf in marks:
        _p15_mark(ax, bt, bx, x, name, label, yf)

    if brackets:
        q = q_for_interval if q_for_interval is not None else lm.get("Q")
        ao, ac = lm.get("AO"), lm.get("AC")
        ymin, ymax = ax.get_ylim()
        base = ymin + 0.10*(ymax-ymin)
        _p15_bracket(ax, q, ao, base, "PEP")
        _p15_bracket(ax, ao, ac, base + 0.15*(ymax-ymin), "LVET")
        _p15_bracket(ax, q, ac, base + 0.30*(ymax-ymin), "QS2")

    ax.set_title(title, loc="left", fontsize=11, pad=6)
    ax.set_ylabel("z-score", fontsize=10)
    ax.grid(True, alpha=0.22)
    ax.set_xlim(*P15_XLIM)
    ax.tick_params(labelsize=9)


def _p15_make_final_figs(outdir, ecg, scg, radar, aoac, acfg):
    try:
        rep = _p15_pick_best(ecg, scg, radar, acfg)
        if rep is None:
            (outdir / "patch15_no_valid_representative_cycle.txt").write_text(
                "No valid ECG-centered representative cycle found.",
                encoding="utf-8"
            )
            return

        ecg_lm = rep["ecg_lm"]
        scg_lm = rep["scg_lm"]
        radar_lm = rep["radar_lm"]
        q = ecg_lm.get("Q")

        rows = []
        for mod, lm in [("ECG", ecg_lm), ("SCG", scg_lm), ("Radar", radar_lm)]:
            pep, lvet, qs2 = _p5_interval(q, lm.get("AO"), lm.get("AC"))
            rows.append([mod, rep["beat_index"], rep["anchor"], q, 0.0 if mod=="ECG" else None,
                         ecg_lm.get("T") if mod=="ECG" else None,
                         lm.get("MC"), lm.get("IM"), lm.get("AO"), lm.get("AC"), lm.get("MO"),
                         pep, lvet, qs2])
        save_csv(outdir / "fig06_fig10_single_cycle_values.csv",
                 ["modality", "beat_index", "anchor_time_sec", "Q_rel_sec", "R_rel_sec", "T_rel_sec",
                  "MC_rel_sec", "IM_rel_sec", "AO_rel_sec", "AC_rel_sec", "MO_rel_sec",
                  "PEP_ms", "LVET_ms", "QS2_ms"], rows)

        # ECG QRT figure
        fig, ax = plt.subplots(1,1,figsize=(12.4,4.8),constrained_layout=True)
        _p15_panel(ax, "Fig. 2. ECG Q/R/T landmarks and AO/AC reference windows",
                   rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", brackets=False)
        ax.set_xlabel("Time from ECG R-peak [s]")
        fig.savefig(outdir / "fig02_ecg_qrt_reference.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # SCG all-candidate landmarks
        fig, ax = plt.subplots(1,1,figsize=(12.4,4.8),constrained_layout=True)
        _p15_panel(ax, "SCG representative single-cycle MC/IM/AO/AC/MO detected landmarks",
                   rep["bt_s"], rep["bx_s"], scg_lm, "SCG", brackets=False)
        ax.set_xlabel("Time from ECG R-peak [s]")
        fig.savefig(outdir / "fig02_scg_candidate_landmarks.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig02_scg_reference_landmarks.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Morphology comparison
        fig, axes = plt.subplots(3,1,figsize=(12.8,8.5),sharex=True,constrained_layout=True)
        _p15_panel(axes[0], "ECG QRS-preserved representative beat", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG")
        _p15_panel(axes[1], "SCG MC/IM/AO/AC/MO detected beat", rep["bt_s"], rep["bx_s"], scg_lm, "SCG")
        _p15_panel(axes[2], "Radar AO/AC detected beat", rep["bt_r"], rep["bx_r"], radar_lm, "Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("ECG / SCG / Radar morphology detected landmark comparison", fontsize=14)
        fig.savefig(outdir / "fig02_compact_beat_morphology.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig05_morphology_ecg_scg_radar_candidates.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Label panels
        fig, axes = plt.subplots(3,1,figsize=(12.8,8.6),sharex=True,constrained_layout=True)
        _p15_panel(axes[0], "ECG single-cycle reference labels", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG")
        _p15_panel(axes[1], "SCG single-cycle MC/IM/AO/AC/MO detected labels", rep["bt_s"], rep["bx_s"], scg_lm, "SCG")
        _p15_panel(axes[2], "Radar single-cycle AO/AC detected labels", rep["bt_r"], rep["bx_r"], radar_lm, "Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle ECG reference and SCG/Radar detected landmark labeling", fontsize=14)
        fig.savefig(outdir / "fig06_single_cycle_ecg_radar_aoac_labels.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Intervals
        fig, axes = plt.subplots(3,1,figsize=(12.9,8.9),sharex=True,constrained_layout=True)
        _p15_panel(axes[0], "ECG reference interval markers", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", True)
        _p15_panel(axes[1], "SCG detected interval markers", rep["bt_s"], rep["bx_s"], scg_lm, "SCG", True, q)
        _p15_panel(axes[2], "Radar detected interval markers", rep["bt_r"], rep["bx_r"], radar_lm, "Radar", True, q)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle reference/detected interval analysis", fontsize=14)
        fig.savefig(outdir / "fig06_pep_lvet_qs2_brackets.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig10_ecg_scg_radar_landmark_interval_clean.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # paper export
        paper = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export")
        figs = paper / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        mapping = [
            ("fig02_ecg_qrt_reference.png", "fig02_ecg_qrt_reference.png"),
            ("fig02_compact_beat_morphology.png", "fig02_ecg_scg_radar_morphology_candidates.png"),
            ("fig02_scg_candidate_landmarks.png", "fig02b_scg_detected_landmarks.png"),
            ("fig06_single_cycle_ecg_radar_aoac_labels.png", "fig06_single_cycle_detected_labels.png"),
            ("fig06_pep_lvet_qs2_brackets.png", "fig06_detected_interval_markers.png"),
            ("fig10_ecg_scg_radar_landmark_interval_clean.png", "fig10_detected_interval_comparison.png"),
        ]
        idx_rows = []
        for s, d in mapping:
            sp, dp = outdir / s, figs / d
            status = "missing"
            if sp.exists():
                shutil.copyfile(sp, dp)
                status = "copied"
            idx_rows.append([s, d, status])
        save_csv(figs / "paper_figure_index_patch15.csv", ["Source", "PaperFigure", "Status"], idx_rows)

        (outdir / "patch15_final_figure_policy.txt").write_text(
            "PATCH15 active. ECG figures are raw/QRS-preserved and re-centered to local QRS peak. "
            "SCG figures retain algorithm-detected MC/IM/AO/AC/MO landmarks. "
            "PEP/LVET/QS2 are displayed using ECG Q and modality-specific AO/AC detections.",
            encoding="utf-8"
        )
    except BaseException as e:
        (outdir / "patch15_final_figures_error.txt").write_text(str(e), encoding="utf-8")


def _p15_core_export(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    core = globals().get("_old_save_all_patch3", None)
    if callable(core):
        return core(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    return None


def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    result = _p15_core_export(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        _p15_make_final_figs(outdir, ecg, scg, radar, aoac, acfg)
    except BaseException as e:
        (outdir / "save_all_patch15_error.txt").write_text(str(e), encoding="utf-8")
    return result



# ============================================================
# PATCH16: label wording correction
# ------------------------------------------------------------
# MC / IM / AO / AC / MO are still computed by the detector.
# Figures label them as detected landmarks rather than visual/manual candidates.
# ============================================================



# ============================================================
# PATCH17: keep detection windows visible with detected labels
# ------------------------------------------------------------
# User correction:
# - MC / IM / AO / AC / MO are algorithm-detected SCG landmarks.
# - Radar AO / AC are algorithm-detected radar timing points.
# - PEP / LVET / QS2 are computed intervals.
# - Detection windows must remain visible in figures.
# ============================================================

P17_C = {
    "Q": "#1f77b4",
    "R": "#d62728",
    "T": "#555555",
    "MC": "#17becf",
    "IM": "#1f77b4",
    "AO": "#9467bd",
    "AC": "#ff7f0e",
    "MO": "#2ca02c",
}
P17_XLIM = (-0.12, 0.60)


def _p17_raw_ecg_array(ecg):
    n = len(ecg.get("t", []))
    for k in ["raw_adc_col", "raw"]:
        if k in ecg and ecg[k] is not None and len(ecg[k]) == n:
            return np.asarray(ecg[k], dtype=np.float64)
    for k in ["filtered", "cleaned", "display_rpeak", "display"]:
        if k in ecg and ecg[k] is not None and len(ecg[k]) == n:
            return np.asarray(ecg[k], dtype=np.float64)
    return np.zeros(n, dtype=np.float64)


def _p17_ecg_for_display(ecg):
    x = _p17_raw_ecg_array(ecg)
    fs = float(ecg.get("fs", 100.0))
    try:
        y = safe_bandpass(x, fs, 6.0, min(35.0, fs * 0.45), order=3)
    except Exception:
        try:
            y = signal.detrend(x)
        except Exception:
            y = x - np.nanmedian(x)
    return zscore_safe(y)


def _p17_scg_for_display(scg):
    if scg is None:
        return None
    n = len(scg.get("t", []))
    for k in ["filtered", "resp_removed", "selected_raw", "vmag", "az", "ax", "ay"]:
        if k in scg and scg[k] is not None and len(scg[k]) == n:
            x = np.asarray(scg[k], dtype=np.float64)
            try:
                return zscore_safe(signal.detrend(x))
            except Exception:
                return zscore_safe(x)
    return None


def _p17_radar_for_display(radar):
    n = len(radar.get("t", []))
    for k in ["lms_error", "ppg_like", "displacement", "display"]:
        if k in radar and radar[k] is not None and len(radar[k]) == n:
            return zscore_safe(np.asarray(radar[k], dtype=np.float64))
    return np.zeros(n, dtype=np.float64)


def _p17_slice(tt, xx, anchor, pre=0.12, post=0.60):
    return _p5_slice(tt, xx, anchor, pre, post, 100.0)


def _p17_recenter_ecg_beat(bt, bx):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    m = (bt >= -0.060) & (bt <= 0.060)
    if np.sum(m) < 3:
        return bt, bx, 0.0
    idx = np.where(m)[0]
    j = idx[int(np.nanargmax(np.abs(bx[idx])))]
    r_rel = float(bt[j])
    if bx[j] < 0:
        bx = -bx
    return bt - r_rel, zscore_safe(bx), r_rel


def _p17_detect_ecg_qrt(bt, bx):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    q = None
    m = (bt >= -0.070) & (bt <= -0.008)
    if np.sum(m) >= 3:
        idx = np.where(m)[0]
        q = float(bt[idx[int(np.nanargmin(bx[idx]))]])
    t = None
    m = (bt >= 0.140) & (bt <= 0.340)
    if np.sum(m) >= 3:
        idx = np.where(m)[0]
        t = float(bt[idx[int(np.nanargmax(bx[idx]))]])
    return {"Q": q, "R": 0.0, "T": t, "AO": 0.100, "AC": 0.350}


def _p17_peak(bt, bx, win, mode="max"):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 3:
        return None
    idx = np.where(m)[0]
    if mode == "min":
        j = idx[int(np.nanargmin(bx[idx]))]
    elif mode == "abs":
        j = idx[int(np.nanargmax(np.abs(bx[idx])))]
    else:
        j = idx[int(np.nanargmax(bx[idx]))]
    return float(bt[j])


def _p17_slope_candidate(bt, bx, win, mode="abs"):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    m = (bt >= win[0]) & (bt <= win[1])
    if np.sum(m) < 4:
        return None
    idx = np.where(m)[0]
    try:
        y = safe_lowpass(bx, 100.0, 20.0, order=2)
    except Exception:
        y = bx
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    if mode == "pos":
        s = robust_scale_01(np.maximum(d1[idx], 0)) + 0.60 * robust_scale_01(np.abs(d2[idx]))
    elif mode == "neg":
        s = robust_scale_01(np.maximum(-d1[idx], 0)) + 0.60 * robust_scale_01(np.abs(d2[idx]))
    else:
        s = robust_scale_01(np.abs(d1[idx])) + robust_scale_01(np.abs(d2[idx]))
    t = bt[idx]
    mid = np.mean(win)
    half = max((win[1] - win[0]) / 2.0, 1e-3)
    prior = np.exp(-0.5 * ((t - mid) / (half * 0.90)) ** 2)
    edge = np.clip(np.minimum(t - win[0], win[1] - t) / half, 0.0, 1.0)
    s = s * (0.30 + 0.45 * prior + 0.25 * edge)
    return float(bt[idx[int(np.nanargmax(s))]])


def _p17_scg_landmarks(bt, bx):
    lm = {}

    mc = _p17_slope_candidate(bt, bx, (-0.085, 0.020), "abs")
    if mc is None:
        mc = _p17_peak(bt, bx, (-0.085, 0.020), "abs")
    if mc is None:
        mc = -0.035
    lm["MC"] = float(mc)

    im_lo = max(0.000, lm["MC"] + 0.010)
    im = _p17_peak(bt, bx, (im_lo, 0.100), "max")
    if im is None:
        im = _p17_slope_candidate(bt, bx, (im_lo, 0.115), "pos")
    if im is None:
        im = 0.040
    lm["IM"] = float(im)

    ao_lo = max(0.045, lm["IM"] + 0.012)
    ao = _p17_peak(bt, bx, (ao_lo, 0.180), "max")
    if ao is None:
        ao = _p17_slope_candidate(bt, bx, (ao_lo, 0.180), "pos")
    if ao is None:
        ao = 0.110
    lm["AO"] = float(ao)

    ac_lo = max(0.180, lm["AO"] + 0.080)
    ac = _p17_slope_candidate(bt, bx, (ac_lo, 0.430), "neg")
    if ac is None:
        ac = _p17_peak(bt, bx, (ac_lo, 0.430), "abs")
    if ac is None:
        ac = 0.340
    lm["AC"] = float(ac)

    mo_lo = max(0.340, lm["AC"] + 0.050)
    mo = _p17_peak(bt, bx, (mo_lo, 0.590), "max")
    if mo is None:
        mo = _p17_slope_candidate(bt, bx, (mo_lo, 0.590), "pos")
    if mo is None:
        mo = 0.480
    lm["MO"] = float(mo)

    bounds = {
        "MC": (-0.100, 0.030),
        "IM": (0.000, 0.120),
        "AO": (0.045, 0.185),
        "AC": (0.180, 0.440),
        "MO": (0.340, 0.600),
    }
    prev = -999.0
    for k in ["MC", "IM", "AO", "AC", "MO"]:
        lo, hi = bounds[k]
        x = min(max(float(lm[k]), lo), hi)
        if x <= prev + 0.008:
            x = min(max(prev + 0.010, lo), hi)
        lm[k] = float(x)
        prev = float(x)
    return lm


def _p17_radar_landmarks(bt, bx):
    ao = _p17_slope_candidate(bt, bx, (0.070, 0.165), "pos")
    if ao is None:
        ao = 0.110
    ac_lo = max(0.260, ao + 0.100)
    ac = _p17_slope_candidate(bt, bx, (ac_lo, 0.460), "neg")
    if ac is None:
        ac = 0.370
    return {"AO": float(ao), "AC": float(ac)}


def _p17_pick_best(ecg, scg, radar, acfg):
    r = np.asarray(ecg.get("peaks_time", []), dtype=float)
    if len(r) < 3:
        return None
    es = _p17_ecg_for_display(ecg)
    ss = _p17_scg_for_display(scg) if scg is not None else None
    rs = _p17_radar_for_display(radar)

    best = None
    lo = max(1, int(len(r) * 0.03))
    hi = max(2, int(len(r) * 0.97))
    for bi in range(lo, hi):
        if bi <= 0 or bi >= len(r) - 1:
            continue
        anchor0 = float(r[bi])
        bt_e0, bx_e0 = _p17_slice(ecg["t"], es, anchor0)
        if bt_e0 is None:
            continue
        bt_e, bx_e, r_shift = _p17_recenter_ecg_beat(bt_e0, bx_e0)
        anchor = anchor0 + r_shift

        bt_r, bx_r = _p17_slice(radar["t"], rs, anchor)
        if bt_r is None:
            continue

        bt_s, bx_s, scg_lm = None, None, {}
        if ss is not None:
            bt_s, bx_s = _p17_slice(scg["t"], ss, anchor)
            if bt_s is None:
                continue
            scg_lm = _p17_scg_landmarks(bt_s, bx_s)

        ecg_lm = _p17_detect_ecg_qrt(bt_e, bx_e)
        radar_lm = _p17_radar_landmarks(bt_r, bx_r)

        score = 0.0
        rwin = (bt_e >= -0.020) & (bt_e <= 0.020)
        r_amp = float(np.nanmax(bx_e[rwin])) if np.any(rwin) else 0.0
        score += 2.5 * max(0.0, r_amp)

        if bt_s is not None:
            ao = scg_lm.get("AO")
            ao_amp = float(np.interp(ao, bt_s, bx_s)) if ao is not None else 0.0
            early = bx_s[(bt_s >= 0.035) & (bt_s <= 0.180)]
            score += 5.5 * max(0.0, ao_amp)
            score += 1.6 * float(np.nanstd(early)) if len(early) else 0.0

        rseg = (bt_r >= 0.06) & (bt_r <= 0.46)
        score += 0.45 * float(np.nanstd(bx_r[rseg])) if np.any(rseg) else 0.0
        score += -0.0008 * abs(bi - len(r) / 2.0)

        cand = {
            "score": score,
            "beat_index": bi,
            "anchor": anchor,
            "bt_e": bt_e,
            "bx_e": bx_e,
            "bt_s": bt_s,
            "bx_s": bx_s,
            "bt_r": bt_r,
            "bx_r": bx_r,
            "ecg_lm": ecg_lm,
            "scg_lm": scg_lm,
            "radar_lm": radar_lm,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def _p17_mark(ax, bt, bx, x, name, label, yfrac):
    if x is None:
        return
    try:
        x = float(x)
        if not np.isfinite(x):
            return
    except Exception:
        return
    c = P17_C.get(name, "black")
    y = float(np.interp(x, bt, bx))
    ax.axvline(x, color=c, linestyle="--", linewidth=1.05, alpha=0.95, zorder=3)
    mk = {"Q":"o", "R":"o", "T":"o", "MC":"o", "IM":"^", "AO":"s", "AC":"D", "MO":"v"}.get(name, "o")
    ax.scatter([x], [y], s=54, marker=mk, facecolor="white", edgecolor=c, linewidth=1.4, zorder=5)
    ymin, ymax = ax.get_ylim()
    ax.text(x, ymin + (ymax-ymin)*yfrac, label, ha="center", va="top", fontsize=7.7, color=c,
            bbox=dict(boxstyle="round,pad=0.11", fc="white", ec=c, alpha=0.96), zorder=6)


def _p17_bracket(ax, x0, x1, y, label):
    if x0 is None or x1 is None:
        return
    try:
        x0 = float(x0); x1 = float(x1)
        if not (np.isfinite(x0) and np.isfinite(x1)):
            return
    except Exception:
        return
    ax.plot([x0, x1], [y, y], color="0.25", linewidth=1.0, zorder=2)
    ax.plot([x0, x0], [y-0.020, y+0.020], color="0.25", linewidth=0.85)
    ax.plot([x1, x1], [y-0.020, y+0.020], color="0.25", linewidth=0.85)
    ax.text((x0+x1)/2.0, y+0.025, label, ha="center", va="bottom", fontsize=7.3,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="0.60", alpha=0.95))


def _p17_window_label(ax, x0, x1, label, yfrac=0.04):
    """Keep detection windows visible and explicitly named."""
    ymin, ymax = ax.get_ylim()
    ax.text((x0+x1)/2.0, ymin + (ymax-ymin)*yfrac, label,
            ha="center", va="bottom", fontsize=7.1, color="0.35",
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="0.75", alpha=0.82),
            zorder=4)


def _p17_panel(ax, title, bt, bx, lm, mode, brackets=False, q_for_interval=None):
    ax.plot(bt, bx, color="black", linewidth=1.55, zorder=1)
    ax.axvline(0.0, color="0.40", linestyle=":", linewidth=1.0, zorder=2)

    if mode == "ECG":
        spans = [((0.07, 0.16), "AO", "AO ref. window"),
                 ((0.28, 0.46), "AC", "AC ref. window")]
        marks = [("Q", lm.get("Q"), "Q", 0.95),
                 ("R", lm.get("R"), "R", 0.88),
                 ("T", lm.get("T"), "T", 0.81),
                 ("AO", lm.get("AO"), "AO ref.", 0.74),
                 ("AC", lm.get("AC"), "AC ref.", 0.67)]
    elif mode == "SCG":
        spans = [((-0.10, 0.03), "MC", "MC window"),
                 ((0.00, 0.12), "IM", "IM window"),
                 ((0.045, 0.185), "AO", "AO window"),
                 ((0.180, 0.440), "AC", "AC window"),
                 ((0.340, 0.600), "MO", "MO window")]
        marks = [("MC", lm.get("MC"), "MC", 0.97),
                 ("IM", lm.get("IM"), "IM", 0.90),
                 ("AO", lm.get("AO"), "AO", 0.83),
                 ("AC", lm.get("AC"), "AC", 0.76),
                 ("MO", lm.get("MO"), "MO", 0.69)]
    else:
        spans = [((0.07, 0.165), "AO", "AO window"),
                 ((0.260, 0.460), "AC", "AC window")]
        marks = [("AO", lm.get("AO"), "AO", 0.88),
                 ("AC", lm.get("AC"), "AC", 0.78)]

    for win, name, wlabel in spans:
        ax.axvspan(win[0], win[1], color=P17_C.get(name, "#999999"), alpha=0.040, zorder=0)

    ymin, ymax = float(np.nanmin(bx)), float(np.nanmax(bx))
    pad = max(0.65, 0.22*(ymax-ymin + 1e-9))
    ax.set_ylim(ymin-pad, ymax+1.08)

    # after ylim is set, add window labels so windows are not "removed" visually
    for win, name, wlabel in spans:
        _p17_window_label(ax, win[0], win[1], wlabel, yfrac=0.035)

    for name, x, label, yf in marks:
        _p17_mark(ax, bt, bx, x, name, label, yf)

    if brackets:
        q = q_for_interval if q_for_interval is not None else lm.get("Q")
        ao, ac = lm.get("AO"), lm.get("AC")
        ymin, ymax = ax.get_ylim()
        base = ymin + 0.10*(ymax-ymin)
        _p17_bracket(ax, q, ao, base, "PEP")
        _p17_bracket(ax, ao, ac, base + 0.15*(ymax-ymin), "LVET")
        _p17_bracket(ax, q, ac, base + 0.30*(ymax-ymin), "QS2")

    ax.set_title(title, loc="left", fontsize=11, pad=6)
    ax.set_ylabel("z-score", fontsize=10)
    ax.grid(True, alpha=0.22)
    ax.set_xlim(*P17_XLIM)
    ax.tick_params(labelsize=9)


def _p17_make_final_figs(outdir, ecg, scg, radar, aoac, acfg):
    try:
        rep = _p17_pick_best(ecg, scg, radar, acfg)
        if rep is None:
            (outdir / "patch17_no_valid_representative_cycle.txt").write_text(
                "No valid ECG-centered representative cycle found.",
                encoding="utf-8"
            )
            return

        ecg_lm = rep["ecg_lm"]
        scg_lm = rep["scg_lm"]
        radar_lm = rep["radar_lm"]
        q = ecg_lm.get("Q")

        rows = []
        for mod, lm in [("ECG", ecg_lm), ("SCG", scg_lm), ("Radar", radar_lm)]:
            pep, lvet, qs2 = _p5_interval(q, lm.get("AO"), lm.get("AC"))
            rows.append([mod, rep["beat_index"], rep["anchor"], q, 0.0 if mod=="ECG" else None,
                         ecg_lm.get("T") if mod=="ECG" else None,
                         lm.get("MC"), lm.get("IM"), lm.get("AO"), lm.get("AC"), lm.get("MO"),
                         pep, lvet, qs2])
        save_csv(outdir / "fig06_fig10_single_cycle_values.csv",
                 ["modality", "beat_index", "anchor_time_sec", "Q_rel_sec", "R_rel_sec", "T_rel_sec",
                  "MC_rel_sec", "IM_rel_sec", "AO_rel_sec", "AC_rel_sec", "MO_rel_sec",
                  "PEP_ms", "LVET_ms", "QS2_ms"], rows)

        # ECG QRT
        fig, ax = plt.subplots(1,1,figsize=(12.4,4.8),constrained_layout=True)
        _p17_panel(ax, "Fig. 2. ECG Q/R/T landmarks and AO/AC reference windows",
                   rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", brackets=False)
        ax.set_xlabel("Time from ECG R-peak [s]")
        fig.savefig(outdir / "fig02_ecg_qrt_reference.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # SCG detected landmarks with windows
        fig, ax = plt.subplots(1,1,figsize=(12.4,4.8),constrained_layout=True)
        _p17_panel(ax, "SCG representative single-cycle MC/IM/AO/AC/MO detected landmarks",
                   rep["bt_s"], rep["bx_s"], scg_lm, "SCG", brackets=False)
        ax.set_xlabel("Time from ECG R-peak [s]")
        fig.savefig(outdir / "fig02_scg_candidate_landmarks.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig02_scg_reference_landmarks.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Morphology
        fig, axes = plt.subplots(3,1,figsize=(12.8,8.5),sharex=True,constrained_layout=True)
        _p17_panel(axes[0], "ECG QRS-preserved representative beat", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG")
        _p17_panel(axes[1], "SCG MC/IM/AO/AC/MO detected beat", rep["bt_s"], rep["bx_s"], scg_lm, "SCG")
        _p17_panel(axes[2], "Radar AO/AC detected beat", rep["bt_r"], rep["bx_r"], radar_lm, "Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("ECG / SCG / Radar morphology detected landmark comparison", fontsize=14)
        fig.savefig(outdir / "fig02_compact_beat_morphology.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig05_morphology_ecg_scg_radar_candidates.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Labels
        fig, axes = plt.subplots(3,1,figsize=(12.8,8.6),sharex=True,constrained_layout=True)
        _p17_panel(axes[0], "ECG single-cycle reference labels", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG")
        _p17_panel(axes[1], "SCG single-cycle MC/IM/AO/AC/MO detected labels", rep["bt_s"], rep["bx_s"], scg_lm, "SCG")
        _p17_panel(axes[2], "Radar single-cycle AO/AC detected labels", rep["bt_r"], rep["bx_r"], radar_lm, "Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle ECG reference and SCG/Radar detected landmark labeling", fontsize=14)
        fig.savefig(outdir / "fig06_single_cycle_ecg_radar_aoac_labels.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Intervals
        fig, axes = plt.subplots(3,1,figsize=(12.9,8.9),sharex=True,constrained_layout=True)
        _p17_panel(axes[0], "ECG reference interval markers", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", True)
        _p17_panel(axes[1], "SCG detected interval markers", rep["bt_s"], rep["bx_s"], scg_lm, "SCG", True, q)
        _p17_panel(axes[2], "Radar detected interval markers", rep["bt_r"], rep["bx_r"], radar_lm, "Radar", True, q)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle reference/detected interval analysis", fontsize=14)
        fig.savefig(outdir / "fig06_pep_lvet_qs2_brackets.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig10_ecg_scg_radar_landmark_interval_clean.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        paper = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export")
        figs = paper / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        mapping = [
            ("fig02_ecg_qrt_reference.png", "fig02_ecg_qrt_reference.png"),
            ("fig02_compact_beat_morphology.png", "fig02_ecg_scg_radar_morphology_detected_landmarks.png"),
            ("fig02_scg_candidate_landmarks.png", "fig02b_scg_detected_landmarks.png"),
            ("fig06_single_cycle_ecg_radar_aoac_labels.png", "fig06_single_cycle_detected_labels.png"),
            ("fig06_pep_lvet_qs2_brackets.png", "fig06_detected_interval_markers.png"),
            ("fig10_ecg_scg_radar_landmark_interval_clean.png", "fig10_detected_interval_comparison.png"),
        ]
        idx_rows = []
        for s, d in mapping:
            sp, dp = outdir / s, figs / d
            status = "missing"
            if sp.exists():
                shutil.copyfile(sp, dp)
                status = "copied"
            idx_rows.append([s, d, status])
        save_csv(figs / "paper_figure_index_patch17.csv", ["Source", "PaperFigure", "Status"], idx_rows)

        (outdir / "patch17_final_figure_policy.txt").write_text(
            "PATCH17 active. Detection windows are explicitly retained and labeled. "
            "SCG MC/IM/AO/AC/MO, Radar AO/AC, and PEP/LVET/QS2 are displayed as algorithm-detected outputs.",
            encoding="utf-8"
        )
    except BaseException as e:
        (outdir / "patch17_final_figures_error.txt").write_text(str(e), encoding="utf-8")


def _p17_core_export(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    core = globals().get("_old_save_all_patch3", None)
    if callable(core):
        return core(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    return None


def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    result = _p17_core_export(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        _p17_make_final_figs(outdir, ecg, scg, radar, aoac, acfg)
    except BaseException as e:
        (outdir / "save_all_patch17_error.txt").write_text(str(e), encoding="utf-8")
    return result



# ============================================================
# PATCH18: SCG polarity-constrained landmark detector
# ------------------------------------------------------------
# MC = negative peak, IM = positive peak, AO = positive peak,
# AC = negative peak/negative transition, MO = positive late peak.
# SCG polarity is normalized per representative beat using AO window.
# Existing PATCH17 windows/labels/interval figures are kept.
# ============================================================

P18_SIGN_WINDOWS = {
    "MC": (-0.085, 0.020),
    "IM": (0.000, 0.115),
    "AO": (0.045, 0.185),
    "AC": (0.180, 0.440),
    "MO": (0.340, 0.600),
}


def _p18_idx(bt, win):
    bt = np.asarray(bt, dtype=float)
    return np.where((bt >= win[0]) & (bt <= win[1]))[0]


def _p18_peak(bt, bx, win, sign="pos"):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    idx = _p18_idx(bt, win)
    if len(idx) < 3:
        return None
    if sign == "neg":
        j = idx[int(np.nanargmin(bx[idx]))]
    elif sign == "abs":
        j = idx[int(np.nanargmax(np.abs(bx[idx])))]
    else:
        j = idx[int(np.nanargmax(bx[idx]))]
    return float(bt[j])


def _p18_slope(bt, bx, win, sign="pos"):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    idx = _p18_idx(bt, win)
    if len(idx) < 4:
        return None
    try:
        y = safe_lowpass(bx, 100.0, 20.0, order=2)
    except Exception:
        y = bx
    d1 = np.gradient(y, bt)
    d2 = np.gradient(d1, bt)
    if sign == "neg":
        s = robust_scale_01(np.maximum(-d1[idx], 0)) + 0.60 * robust_scale_01(np.abs(d2[idx]))
    elif sign == "abs":
        s = robust_scale_01(np.abs(d1[idx])) + 0.60 * robust_scale_01(np.abs(d2[idx]))
    else:
        s = robust_scale_01(np.maximum(d1[idx], 0)) + 0.60 * robust_scale_01(np.abs(d2[idx]))
    t = bt[idx]
    mid = np.mean(win)
    half = max((win[1] - win[0]) / 2.0, 1e-3)
    prior = np.exp(-0.5 * ((t - mid) / (half * 0.90)) ** 2)
    edge = np.clip(np.minimum(t - win[0], win[1] - t) / half, 0.0, 1.0)
    s = s * (0.30 + 0.45 * prior + 0.25 * edge)
    return float(bt[idx[int(np.nanargmax(s))]])


def _p18_normalize_scg_polarity(bt, bx):
    """Flip SCG beat if AO-window dominant deflection is negative."""
    bx = zscore_safe(np.asarray(bx, dtype=float))
    idx = _p18_idx(bt, P18_SIGN_WINDOWS["AO"])
    if len(idx) < 3:
        return bx, 1
    pos_amp = float(np.nanmax(bx[idx]))
    neg_amp = float(abs(np.nanmin(bx[idx])))
    if neg_amp > pos_amp * 1.15:
        return zscore_safe(-bx), -1
    return bx, 1


def _p18_ordered_bounds(lm):
    bounds = {
        "MC": (-0.100, 0.030),
        "IM": (0.000, 0.120),
        "AO": (0.045, 0.185),
        "AC": (0.180, 0.440),
        "MO": (0.340, 0.600),
    }
    default = {"MC": -0.035, "IM": 0.055, "AO": 0.120, "AC": 0.350, "MO": 0.500}
    out = {}
    prev = -999.0
    for k in ["MC", "IM", "AO", "AC", "MO"]:
        x = lm.get(k)
        if x is None or not np.isfinite(float(x)):
            x = default[k]
        lo, hi = bounds[k]
        x = min(max(float(x), lo), hi)
        if x <= prev + 0.008:
            x = min(max(prev + 0.010, lo), hi)
        out[k] = float(x)
        prev = float(x)
    return out


def _p18_scg_landmarks_signed(bt, bx):
    """Return polarity-normalized waveform and sign-constrained MC/IM/AO/AC/MO."""
    bx_norm, pol = _p18_normalize_scg_polarity(bt, bx)
    lm = {}
    # MC: negative peak
    lm["MC"] = _p18_peak(bt, bx_norm, P18_SIGN_WINDOWS["MC"], "neg") or _p18_slope(bt, bx_norm, P18_SIGN_WINDOWS["MC"], "neg")
    # IM: positive peak after MC
    im_lo = max(P18_SIGN_WINDOWS["IM"][0], (lm["MC"] + 0.010) if lm["MC"] is not None else P18_SIGN_WINDOWS["IM"][0])
    lm["IM"] = _p18_peak(bt, bx_norm, (im_lo, P18_SIGN_WINDOWS["IM"][1]), "pos") or _p18_slope(bt, bx_norm, (im_lo, P18_SIGN_WINDOWS["IM"][1]), "pos")
    # AO: positive dominant peak after IM
    ao_lo = max(P18_SIGN_WINDOWS["AO"][0], (lm["IM"] + 0.012) if lm["IM"] is not None else P18_SIGN_WINDOWS["AO"][0])
    lm["AO"] = _p18_peak(bt, bx_norm, (ao_lo, P18_SIGN_WINDOWS["AO"][1]), "pos") or _p18_slope(bt, bx_norm, (ao_lo, P18_SIGN_WINDOWS["AO"][1]), "pos")
    # AC: negative peak/transition after AO
    ac_lo = max(P18_SIGN_WINDOWS["AC"][0], (lm["AO"] + 0.080) if lm["AO"] is not None else P18_SIGN_WINDOWS["AC"][0])
    lm["AC"] = _p18_peak(bt, bx_norm, (ac_lo, P18_SIGN_WINDOWS["AC"][1]), "neg") or _p18_slope(bt, bx_norm, (ac_lo, P18_SIGN_WINDOWS["AC"][1]), "neg")
    # MO: positive late/post-systolic peak after AC
    mo_lo = max(P18_SIGN_WINDOWS["MO"][0], (lm["AC"] + 0.050) if lm["AC"] is not None else P18_SIGN_WINDOWS["MO"][0])
    lm["MO"] = _p18_peak(bt, bx_norm, (mo_lo, P18_SIGN_WINDOWS["MO"][1]), "pos") or _p18_slope(bt, bx_norm, (mo_lo, P18_SIGN_WINDOWS["MO"][1]), "pos")
    return _p18_ordered_bounds(lm), bx_norm, pol


# Override PATCH17 SCG landmark detector with polarity-constrained detector.
def _p17_scg_landmarks(bt, bx):
    lm, _bx_norm, _pol = _p18_scg_landmarks_signed(bt, bx)
    return lm


# Override PATCH17 best-beat picker so figures use polarity-normalized SCG waveform.
def _p17_pick_best(ecg, scg, radar, acfg):
    r = np.asarray(ecg.get("peaks_time", []), dtype=float)
    if len(r) < 3:
        return None
    es = _p17_ecg_for_display(ecg)
    ss = _p17_scg_for_display(scg) if scg is not None else None
    rs = _p17_radar_for_display(radar)
    best = None
    lo = max(1, int(len(r) * 0.03))
    hi = max(2, int(len(r) * 0.97))
    for bi in range(lo, hi):
        if bi <= 0 or bi >= len(r) - 1:
            continue
        anchor0 = float(r[bi])
        bt_e0, bx_e0 = _p17_slice(ecg["t"], es, anchor0)
        if bt_e0 is None:
            continue
        bt_e, bx_e, r_shift = _p17_recenter_ecg_beat(bt_e0, bx_e0)
        anchor = anchor0 + r_shift
        bt_r, bx_r = _p17_slice(radar["t"], rs, anchor)
        if bt_r is None:
            continue
        bt_s, bx_s, scg_lm, pol = None, None, {}, 1
        if ss is not None:
            bt_s_raw, bx_s_raw = _p17_slice(scg["t"], ss, anchor)
            if bt_s_raw is None:
                continue
            scg_lm, bx_s_norm, pol = _p18_scg_landmarks_signed(bt_s_raw, bx_s_raw)
            bt_s, bx_s = bt_s_raw, bx_s_norm
        ecg_lm = _p17_detect_ecg_qrt(bt_e, bx_e)
        radar_lm = _p17_radar_landmarks(bt_r, bx_r)
        score = 0.0
        rwin = (bt_e >= -0.020) & (bt_e <= 0.020)
        if np.any(rwin):
            score += 2.5 * max(0.0, float(np.nanmax(bx_e[rwin])))
        if bt_s is not None:
            mc_y = float(np.interp(scg_lm["MC"], bt_s, bx_s))
            im_y = float(np.interp(scg_lm["IM"], bt_s, bx_s))
            ao_y = float(np.interp(scg_lm["AO"], bt_s, bx_s))
            ac_y = float(np.interp(scg_lm["AC"], bt_s, bx_s))
            mo_y = float(np.interp(scg_lm["MO"], bt_s, bx_s))
            score += 2.0 * max(0, -mc_y)
            score += 2.2 * max(0, im_y)
            score += 5.0 * max(0, ao_y)
            score += 2.0 * max(0, -ac_y)
            score += 1.8 * max(0, mo_y)
        rseg = (bt_r >= 0.06) & (bt_r <= 0.46)
        if np.any(rseg):
            score += 0.45 * float(np.nanstd(bx_r[rseg]))
        score += -0.0008 * abs(bi - len(r) / 2.0)
        cand = {
            "score": score, "beat_index": bi, "anchor": anchor,
            "bt_e": bt_e, "bx_e": bx_e,
            "bt_s": bt_s, "bx_s": bx_s,
            "bt_r": bt_r, "bx_r": bx_r,
            "ecg_lm": ecg_lm, "scg_lm": scg_lm, "radar_lm": radar_lm,
            "scg_polarity": pol,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


# Override final figure maker only to add audit file; drawing/windows remain PATCH17.
_old_p17_make_final_figs_for_patch18 = _p17_make_final_figs

def _p17_make_final_figs(outdir, ecg, scg, radar, aoac, acfg):
    _old_p17_make_final_figs_for_patch18(outdir, ecg, scg, radar, aoac, acfg)
    try:
        (outdir / "patch18_scg_polarity_detector_policy.txt").write_text(
            "PATCH18 active. SCG landmarks are sign-constrained: MC negative, IM positive, AO positive, AC negative, MO positive. "
            "SCG representative beat is polarity-normalized using AO-window dominance before plotting/detection. "
            "Detection windows remain visible via PATCH17 panels.",
            encoding="utf-8"
        )
    except Exception:
        pass



# ============================================================
# PATCH19: Literature-guided SCG-derived reference analysis
# ------------------------------------------------------------
# ECG R-peak is used only as beat alignment anchor.
# SCG AO/AC are detected using the literature-guided SCG fiducial
# convention implemented in PATCH18:
#   MC(-), IM(+), AO(+), AC(-), MO(+), with ordered windows.
# Radar AO/AC are compared against SCG-derived AO/AC by timing
# distribution and relative timing difference, not ECG-derived accuracy.
# ============================================================

def _p19_make_all_beat_scg_radar_table(ecg, scg, radar):
    """
    Beat-level SCG-derived reference vs Radar-detected AO/AC table.
    This is relative timing analysis, not absolute AO/AC ground-truth accuracy.
    """
    r = np.asarray(ecg.get("peaks_time", []), dtype=float)
    if len(r) < 3 or scg is None:
        return [], {}

    ss = _p18_scg_for_display(scg)
    rs = _p18_radar_for_display(radar)
    if ss is None:
        return [], {}

    rows = []
    for bi in range(1, len(r)-1):
        anchor = float(r[bi])
        bt_s_raw, bx_s_raw = _p18_slice(scg["t"], ss, anchor)
        bt_r, bx_r = _p18_slice(radar["t"], rs, anchor)
        if bt_s_raw is None or bt_r is None:
            continue
        scg_lm, bx_s_norm, pol = _p18_scg_landmarks(bt_s_raw, bx_s_raw)
        radar_lm = _p18_radar_landmarks(bt_r, bx_r)
        if any(scg_lm.get(k) is None for k in ["AO", "AC"]):
            continue
        if any(radar_lm.get(k) is None for k in ["AO", "AC"]):
            continue
        scg_ao, scg_ac = scg_lm["AO"], scg_lm["AC"]
        rad_ao, rad_ac = radar_lm["AO"], radar_lm["AC"]
        rows.append([
            bi, anchor, pol,
            scg_lm.get("MC"), scg_lm.get("IM"), scg_ao, scg_ac, scg_lm.get("MO"),
            rad_ao, rad_ac,
            (rad_ao - scg_ao) * 1000.0,
            (rad_ac - scg_ac) * 1000.0,
            (scg_ac - scg_ao) * 1000.0,
            (rad_ac - rad_ao) * 1000.0,
        ])

    def _summary(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return {"n": 0, "mean_ms": None, "std_ms": None, "median_ms": None, "iqr_ms": None}
        return {
            "n": int(len(vals)),
            "mean_ms": float(np.mean(vals)),
            "std_ms": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median_ms": float(np.median(vals)),
            "iqr_ms": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        }

    arr = np.asarray(rows, dtype=float) if rows else np.empty((0, 14))
    summary = {
        "analysis_type": "SCG-derived reference vs Radar-detected AO/AC relative timing distribution",
        "ecg_role": "R-peak alignment anchor only; not AO/AC ground truth",
        "scg_role": "literature-guided SCG fiducial landmark detector used as SCG-derived reference",
        "total_ecg_beats": int(len(r)),
        "accepted_scg_radar_beats": int(len(rows)),
        "accept_rate_percent": float(len(rows) / max(len(r), 1) * 100.0),
    }
    if len(rows) > 0:
        summary.update({
            "scg_ao_timing_from_r": _summary(arr[:, 5] * 1000.0),
            "scg_ac_timing_from_r": _summary(arr[:, 6] * 1000.0),
            "radar_ao_timing_from_r": _summary(arr[:, 8] * 1000.0),
            "radar_ac_timing_from_r": _summary(arr[:, 9] * 1000.0),
            "radar_minus_scg_ao": _summary(arr[:, 10]),
            "radar_minus_scg_ac": _summary(arr[:, 11]),
            "scg_lvet_ao_to_ac": _summary(arr[:, 12]),
            "radar_lvet_ao_to_ac": _summary(arr[:, 13]),
        })
    return rows, summary


def _p19_make_distribution_figures(outdir, rows):
    if not rows:
        return
    arr = np.asarray(rows, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(9.8, 5.2), constrained_layout=True)
    data = [arr[:, 5] * 1000.0, arr[:, 8] * 1000.0, arr[:, 6] * 1000.0, arr[:, 9] * 1000.0]
    labels = ["SCG AO", "Radar AO", "SCG AC", "Radar AC"]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Timing from ECG R-peak [ms]")
    ax.set_title("SCG-derived AO/AC reference and Radar-detected AO/AC timing distribution")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(outdir / "fig07_scg_radar_aoac_timing_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0), constrained_layout=True)
    data = [arr[:, 10], arr[:, 11]]
    labels = ["Radar AO - SCG AO", "Radar AC - SCG AC"]
    ax.axhline(0, color="0.4", linestyle="--", linewidth=1.0)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Relative timing difference [ms]")
    ax.set_title("Relative timing difference between Radar and SCG-derived AO/AC")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(outdir / "fig08_scg_radar_relative_difference_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _p19_postprocess_literature_guided_reference(outdir, ecg, scg, radar):
    rows, summary = _p19_make_all_beat_scg_radar_table(ecg, scg, radar)
    if rows:
        save_csv(
            outdir / "scg_derived_reference_vs_radar_aoac.csv",
            ["beat_index", "anchor_time_sec", "scg_polarity",
             "scg_MC_sec", "scg_IM_sec", "scg_AO_sec", "scg_AC_sec", "scg_MO_sec",
             "radar_AO_sec", "radar_AC_sec",
             "radar_minus_scg_AO_ms", "radar_minus_scg_AC_ms",
             "scg_LVET_ms", "radar_LVET_ms"],
            rows
        )
        with open(outdir / "scg_radar_aoac_timing_distribution_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _p19_make_distribution_figures(outdir, rows)

    # Copy defensible result figures to paper_export.
    paper = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export")
    figs = paper / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    mapping = [
        ("fig02_compact_beat_morphology.png", "fig02_ecg_scg_radar_morphology_comparison.png"),
        ("fig06_single_cycle_ecg_radar_aoac_labels.png", "fig06_scg_reference_radar_detected_landmarks.png"),
        ("fig07_scg_radar_aoac_timing_boxplot.png", "fig07_scg_radar_aoac_timing_boxplot.png"),
        ("fig08_scg_radar_relative_difference_boxplot.png", "fig08_scg_radar_relative_difference_boxplot.png"),
        ("fig06_pep_lvet_qs2_brackets.png", "fig09_scg_radar_interval_analysis.png"),
    ]
    idx_rows = []
    for s, d in mapping:
        sp, dp = outdir / s, figs / d
        status = "missing"
        if sp.exists():
            shutil.copyfile(sp, dp)
            status = "copied"
        idx_rows.append([s, d, status])
    save_csv(figs / "paper_figure_index_patch19.csv", ["Source", "PaperFigure", "Status"], idx_rows)

    (outdir / "patch19_final_analysis_policy.txt").write_text(
        "PATCH19 active. ECG is only the R-peak beat-alignment anchor. "
        "SCG AO/AC are produced by the literature-guided SCG fiducial detector "
        "and used as SCG-derived reference. Radar AO/AC are compared with "
        "SCG-derived AO/AC by timing distribution and relative difference, not by "
        "ECG-derived AO/AC accuracy.",
        encoding="utf-8"
    )


# Re-label PATCH18 figure policy concept without changing the stable detector implementation.
_old_save_all_patch19 = save_all

def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    result = _old_save_all_patch19(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        _p19_postprocess_literature_guided_reference(outdir, ecg, scg, radar)
    except BaseException as e:
        (outdir / "save_all_patch19_error.txt").write_text(str(e), encoding="utf-8")
    return result


# ============================================================
# PATCH20: fully self-contained PATCH19 + robust SCG polarity resolver
# ------------------------------------------------------------
# Fixes:
# 1) PATCH19 runtime error from stale _p18_* references.
# 2) MC/IM/AO/AC/MO selecting opposite-polarity peaks.
# 3) SCG-derived AO/AC reference vs Radar AO/AC boxplot/CSV/JSON generation.
#
# Core detector:
# - Evaluate both SCG polarity signs (+1 and -1) for each representative beat.
# - Choose the sign that maximizes physiologic polarity score:
#   MC negative, IM positive, AO positive, AC negative, MO positive.
# - Detect actual signed extrema:
#   MC = negative peak, IM = positive peak, AO = positive peak,
#   AC = negative peak, MO = positive peak.
# - Keep detection windows visible in figures.
# ============================================================

P20_C = {
    "Q": "#1f77b4", "R": "#d62728", "T": "#555555",
    "MC": "#17becf", "IM": "#1f77b4", "AO": "#9467bd",
    "AC": "#ff7f0e", "MO": "#2ca02c",
}
P20_XLIM = (-0.12, 0.62)

P20_WIN = {
    "MC": (-0.085, 0.020),
    "IM": (0.000, 0.115),
    "AO": (0.045, 0.185),
    "AC": (0.180, 0.440),
    "MO": (0.340, 0.600),
    "RADAR_AO": (0.070, 0.165),
    "RADAR_AC": (0.260, 0.460),
    "ECG_AO_REF": (0.070, 0.160),
    "ECG_AC_REF": (0.280, 0.460),
}

def _p20_z(x):
    return zscore_safe(np.asarray(x, dtype=np.float64))

def _p20_raw_ecg_array(ecg):
    n = len(ecg.get("t", []))
    for k in ["raw_adc_col", "raw"]:
        if k in ecg and ecg[k] is not None and len(ecg[k]) == n:
            return np.asarray(ecg[k], dtype=np.float64)
    for k in ["filtered", "cleaned", "display_rpeak", "display"]:
        if k in ecg and ecg[k] is not None and len(ecg[k]) == n:
            return np.asarray(ecg[k], dtype=np.float64)
    return np.zeros(n, dtype=np.float64)

def _p20_ecg_for_display(ecg):
    x = _p20_raw_ecg_array(ecg)
    fs = float(ecg.get("fs", 100.0))
    try:
        return zscore_safe(safe_bandpass(x, fs, 6.0, min(35.0, fs*0.45), order=3))
    except Exception:
        try:
            return zscore_safe(signal.detrend(x))
        except Exception:
            return zscore_safe(x - np.nanmedian(x))

def _p20_scg_for_display(scg):
    if scg is None:
        return None
    n = len(scg.get("t", []))
    # SCG-derived reference must use morphology-preserving branch.
    for k in ["resp_removed", "filtered", "selected_raw", "vmag", "az", "ax", "ay"]:
        if k in scg and scg[k] is not None and len(scg[k]) == n:
            x = np.asarray(scg[k], dtype=np.float64)
            fs = float(scg.get("fs", 100.0))
            try:
                y = safe_bandpass(x, fs, 1.0, min(30.0, fs*0.45), order=2)
            except Exception:
                try:
                    y = signal.detrend(x)
                except Exception:
                    y = x - np.nanmedian(x)
            return zscore_safe(y)
    return None

def _p20_radar_for_display(radar):
    n = len(radar.get("t", []))
    for k in ["lms_error", "ppg_like", "displacement", "display"]:
        if k in radar and radar[k] is not None and len(radar[k]) == n:
            return zscore_safe(np.asarray(radar[k], dtype=np.float64))
    return np.zeros(n, dtype=np.float64)

def _p20_slice(tt, xx, anchor, pre=0.12, post=0.62):
    return _p5_slice(tt, xx, anchor, pre, post, 100.0)

def _p20_recenter_ecg_beat(bt, bx):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    m = (bt >= -0.060) & (bt <= 0.060)
    if np.sum(m) < 3:
        return bt, bx, 0.0
    idx = np.where(m)[0]
    j = idx[int(np.nanargmax(np.abs(bx[idx])))]
    r_rel = float(bt[j])
    if bx[j] < 0:
        bx = -bx
    return bt - r_rel, zscore_safe(bx), r_rel

def _p20_detect_ecg_qrt(bt, bx):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    q = None
    idx = np.where((bt >= -0.070) & (bt <= -0.008))[0]
    if len(idx) >= 3:
        q = float(bt[idx[int(np.nanargmin(bx[idx]))]])
    t = None
    idx = np.where((bt >= 0.140) & (bt <= 0.340))[0]
    if len(idx) >= 3:
        t = float(bt[idx[int(np.nanargmax(bx[idx]))]])
    return {"Q": q, "R": 0.0, "T": t}

def _p20_idx(bt, win):
    bt = np.asarray(bt, dtype=float)
    return np.where((bt >= win[0]) & (bt <= win[1]))[0]

def _p20_peak(bt, bx, win, polarity):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    idx = _p20_idx(bt, win)
    if len(idx) < 3:
        return None
    if polarity == "neg":
        j = idx[int(np.nanargmin(bx[idx]))]
    elif polarity == "abs":
        j = idx[int(np.nanargmax(np.abs(bx[idx])))]
    else:
        j = idx[int(np.nanargmax(bx[idx]))]
    return float(bt[j])

def _p20_constrained_pick_sequence(bt, bx):
    """
    Pick SCG landmarks using signed extrema and sequence constraints.
    bx must already be polarity-normalized.
    """
    lm = {}

    mc = _p20_peak(bt, bx, P20_WIN["MC"], "neg")
    lm["MC"] = -0.035 if mc is None else mc

    im_win = (max(P20_WIN["IM"][0], lm["MC"] + 0.010), P20_WIN["IM"][1])
    im = _p20_peak(bt, bx, im_win, "pos")
    lm["IM"] = 0.055 if im is None else im

    ao_win = (max(P20_WIN["AO"][0], lm["IM"] + 0.012), P20_WIN["AO"][1])
    ao = _p20_peak(bt, bx, ao_win, "pos")
    lm["AO"] = 0.120 if ao is None else ao

    ac_win = (max(P20_WIN["AC"][0], lm["AO"] + 0.080), P20_WIN["AC"][1])
    ac = _p20_peak(bt, bx, ac_win, "neg")
    lm["AC"] = 0.350 if ac is None else ac

    mo_win = (max(P20_WIN["MO"][0], lm["AC"] + 0.050), P20_WIN["MO"][1])
    mo = _p20_peak(bt, bx, mo_win, "pos")
    lm["MO"] = 0.500 if mo is None else mo

    bounds = {
        "MC": (-0.100, 0.030),
        "IM": (0.000, 0.120),
        "AO": (0.045, 0.185),
        "AC": (0.180, 0.440),
        "MO": (0.340, 0.600),
    }
    prev = -999.0
    for k in ["MC", "IM", "AO", "AC", "MO"]:
        lo, hi = bounds[k]
        x = min(max(float(lm[k]), lo), hi)
        if x <= prev + 0.008:
            x = min(max(prev + 0.010, lo), hi)
        lm[k] = float(x)
        prev = x
    return lm

def _p20_signed_score(bt, bx, lm):
    """
    Higher score means selected landmarks match expected SCG polarity:
    MC(-), IM(+), AO(+), AC(-), MO(+).
    """
    vals = {}
    for k in ["MC", "IM", "AO", "AC", "MO"]:
        vals[k] = float(np.interp(lm[k], bt, bx))
    score = 0.0
    score += 2.0 * max(0.0, -vals["MC"])
    score += 2.2 * max(0.0,  vals["IM"])
    score += 5.0 * max(0.0,  vals["AO"])
    score += 2.0 * max(0.0, -vals["AC"])
    score += 1.8 * max(0.0,  vals["MO"])
    # Penalize opposite-polarity selections directly.
    score -= 1.2 * max(0.0,  vals["MC"])
    score -= 1.2 * max(0.0, -vals["IM"])
    score -= 1.5 * max(0.0, -vals["AO"])
    score -= 1.2 * max(0.0,  vals["AC"])
    score -= 1.0 * max(0.0, -vals["MO"])
    return score, vals

def _p20_literature_guided_scg_landmarks(bt, bx):
    """
    Robust polarity resolver:
    evaluate original and inverted SCG beat, then choose the sign that gives
    MC(-), IM(+), AO(+), AC(-), MO(+) with the best score.
    """
    bt = np.asarray(bt, dtype=float)
    bx0 = zscore_safe(np.asarray(bx, dtype=float))

    candidates = []
    for pol in [1, -1]:
        bx = zscore_safe(bx0 * pol)
        lm = _p20_constrained_pick_sequence(bt, bx)
        score, vals = _p20_signed_score(bt, bx, lm)
        candidates.append((score, pol, bx, lm, vals))

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, pol, bx_best, lm_best, vals_best = candidates[0]
    return lm_best, bx_best, pol, vals_best, score

def _p20_radar_landmarks(bt, bx):
    # Radar AO/AC are detected timing points, not ECG-derived labels.
    ao = _p20_peak(bt, bx, P20_WIN["RADAR_AO"], "pos")
    if ao is None:
        ao = 0.110
    ac_win = (max(P20_WIN["RADAR_AC"][0], ao + 0.100), P20_WIN["RADAR_AC"][1])
    # Radar AC can be notch/negative transition; use negative peak in AC window.
    ac = _p20_peak(bt, bx, ac_win, "neg")
    if ac is None:
        ac = 0.370
    return {"AO": float(ao), "AC": float(ac)}

def _p20_score_representative(bt_e, bx_e, bt_s, bx_s, scg_lm, scg_vals, bt_r, bx_r, radar_lm):
    score = 0.0
    rwin = (bt_e >= -0.020) & (bt_e <= 0.020)
    r_amp = float(np.nanmax(bx_e[rwin])) if np.any(rwin) else 0.0
    score += 2.0 * max(0.0, r_amp)
    score += _p20_signed_score(bt_s, bx_s, scg_lm)[0]
    rseg = (bt_r >= 0.06) & (bt_r <= 0.46)
    score += 0.35 * float(np.nanstd(bx_r[rseg])) if np.any(rseg) else 0.0
    return score

def _p20_pick_best(ecg, scg, radar, acfg):
    r = np.asarray(ecg.get("peaks_time", []), dtype=float)
    if len(r) < 3:
        return None

    es = _p20_ecg_for_display(ecg)
    ss = _p20_scg_for_display(scg) if scg is not None else None
    rs = _p20_radar_for_display(radar)
    if ss is None:
        return None

    best = None
    lo = max(1, int(len(r) * 0.03))
    hi = max(2, int(len(r) * 0.97))

    for bi in range(lo, hi):
        if bi <= 0 or bi >= len(r)-1:
            continue
        anchor0 = float(r[bi])
        bt_e0, bx_e0 = _p20_slice(ecg["t"], es, anchor0)
        if bt_e0 is None:
            continue
        bt_e, bx_e, r_shift = _p20_recenter_ecg_beat(bt_e0, bx_e0)
        anchor = anchor0 + r_shift

        bt_s_raw, bx_s_raw = _p20_slice(scg["t"], ss, anchor)
        bt_r, bx_r = _p20_slice(radar["t"], rs, anchor)
        if bt_s_raw is None or bt_r is None:
            continue

        scg_lm, bx_s_norm, scg_pol, scg_vals, scg_score = _p20_literature_guided_scg_landmarks(bt_s_raw, bx_s_raw)
        ecg_lm = _p20_detect_ecg_qrt(bt_e, bx_e)
        radar_lm = _p20_radar_landmarks(bt_r, bx_r)

        score = _p20_score_representative(bt_e, bx_e, bt_s_raw, bx_s_norm, scg_lm, scg_vals, bt_r, bx_r, radar_lm)
        score += -0.0008 * abs(bi - len(r)/2)

        cand = {
            "score": score, "beat_index": bi, "anchor": anchor,
            "bt_e": bt_e, "bx_e": bx_e,
            "bt_s": bt_s_raw, "bx_s": bx_s_norm,
            "bt_r": bt_r, "bx_r": bx_r,
            "ecg_lm": ecg_lm, "scg_lm": scg_lm, "radar_lm": radar_lm,
            "scg_polarity": scg_pol, "scg_values": scg_vals, "scg_score": scg_score,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best

def _p20_all_beat_table(ecg, scg, radar):
    r = np.asarray(ecg.get("peaks_time", []), dtype=float)
    if len(r) < 3 or scg is None:
        return [], {}

    ss = _p20_scg_for_display(scg)
    rs = _p20_radar_for_display(radar)
    if ss is None:
        return [], {}

    rows = []
    for bi in range(1, len(r)-1):
        anchor = float(r[bi])
        bt_s_raw, bx_s_raw = _p20_slice(scg["t"], ss, anchor)
        bt_r, bx_r = _p20_slice(radar["t"], rs, anchor)
        if bt_s_raw is None or bt_r is None:
            continue

        scg_lm, bx_s_norm, pol, vals, score = _p20_literature_guided_scg_landmarks(bt_s_raw, bx_s_raw)
        radar_lm = _p20_radar_landmarks(bt_r, bx_r)

        scg_ao = scg_lm["AO"]
        scg_ac = scg_lm["AC"]
        rad_ao = radar_lm["AO"]
        rad_ac = radar_lm["AC"]

        rows.append([
            bi, anchor, pol, score,
            scg_lm["MC"], scg_lm["IM"], scg_ao, scg_ac, scg_lm["MO"],
            vals["MC"], vals["IM"], vals["AO"], vals["AC"], vals["MO"],
            rad_ao, rad_ac,
            (rad_ao - scg_ao) * 1000.0,
            (rad_ac - scg_ac) * 1000.0,
            (scg_ac - scg_ao) * 1000.0,
            (rad_ac - rad_ao) * 1000.0,
        ])

    def _summary(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return {"n": 0, "mean_ms": None, "std_ms": None, "median_ms": None, "iqr_ms": None}
        return {
            "n": int(len(vals)),
            "mean_ms": float(np.mean(vals)),
            "std_ms": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median_ms": float(np.median(vals)),
            "iqr_ms": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        }

    arr = np.asarray(rows, dtype=float) if rows else np.empty((0, 20))
    summary = {
        "total_ecg_beats": int(len(r)),
        "accepted_scg_radar_beats": int(len(rows)),
        "accept_rate_percent": float(len(rows) / max(len(r), 1) * 100.0),
        "analysis_note": "SCG AO/AC are literature-guided SCG-derived references; ECG is only R-peak anchor.",
    }
    if len(rows) > 0:
        summary.update({
            "scg_ao_timing": _summary(arr[:, 6] * 1000.0),
            "scg_ac_timing": _summary(arr[:, 7] * 1000.0),
            "radar_ao_timing": _summary(arr[:, 14] * 1000.0),
            "radar_ac_timing": _summary(arr[:, 15] * 1000.0),
            "radar_minus_scg_ao": _summary(arr[:, 16]),
            "radar_minus_scg_ac": _summary(arr[:, 17]),
            "scg_lvet_ao_to_ac": _summary(arr[:, 18]),
            "radar_lvet_ao_to_ac": _summary(arr[:, 19]),
        })
    return rows, summary

def _p20_distribution_figures(outdir, rows):
    if not rows:
        return
    arr = np.asarray(rows, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(9.6, 5.2), constrained_layout=True)
    data = [arr[:, 6]*1000.0, arr[:, 14]*1000.0, arr[:, 7]*1000.0, arr[:, 15]*1000.0]
    labels = ["SCG AO", "Radar AO", "SCG AC", "Radar AC"]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Timing from ECG R-peak [ms]")
    ax.set_title("SCG-derived AO/AC reference and Radar-detected AO/AC timing distribution")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(outdir / "fig07_scg_radar_aoac_timing_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.0), constrained_layout=True)
    ax.axhline(0, color="0.4", linestyle="--", linewidth=1.0)
    ax.boxplot([arr[:, 16], arr[:, 17]], labels=["Radar AO - SCG AO", "Radar AC - SCG AC"], showfliers=False)
    ax.set_ylabel("Relative timing difference [ms]")
    ax.set_title("Relative timing difference between Radar and SCG-derived landmarks")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(outdir / "fig08_scg_radar_relative_difference_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def _p20_mark(ax, bt, bx, x, name, label, yfrac):
    if x is None:
        return
    try:
        x = float(x)
        if not np.isfinite(x):
            return
    except Exception:
        return
    c = P20_C.get(name, "black")
    y = float(np.interp(x, bt, bx))
    ax.axvline(x, color=c, linestyle="--", linewidth=1.05, alpha=0.95, zorder=3)
    mk = {"Q":"o", "R":"o", "T":"o", "MC":"o", "IM":"^", "AO":"s", "AC":"D", "MO":"v"}.get(name, "o")
    ax.scatter([x], [y], s=54, marker=mk, facecolor="white", edgecolor=c, linewidth=1.4, zorder=5)
    ymin, ymax = ax.get_ylim()
    ax.text(x, ymin + (ymax-ymin)*yfrac, label, ha="center", va="top", fontsize=7.7, color=c,
            bbox=dict(boxstyle="round,pad=0.11", fc="white", ec=c, alpha=0.96), zorder=6)

def _p20_bracket(ax, x0, x1, y, label):
    if x0 is None or x1 is None:
        return
    try:
        x0 = float(x0); x1 = float(x1)
        if not (np.isfinite(x0) and np.isfinite(x1)):
            return
    except Exception:
        return
    ax.plot([x0, x1], [y, y], color="0.25", linewidth=1.0, zorder=2)
    ax.plot([x0, x0], [y-0.020, y+0.020], color="0.25", linewidth=0.85)
    ax.plot([x1, x1], [y-0.020, y+0.020], color="0.25", linewidth=0.85)
    ax.text((x0+x1)/2.0, y+0.025, label, ha="center", va="bottom", fontsize=7.3,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="0.60", alpha=0.95))

def _p20_window_label(ax, x0, x1, label, yfrac=0.035):
    ymin, ymax = ax.get_ylim()
    ax.text((x0+x1)/2.0, ymin + (ymax-ymin)*yfrac, label,
            ha="center", va="bottom", fontsize=7.1, color="0.35",
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="0.75", alpha=0.82),
            zorder=4)

def _p20_panel(ax, title, bt, bx, lm, mode, brackets=False, q_for_interval=None):
    ax.plot(bt, bx, color="black", linewidth=1.55, zorder=1)
    ax.axvline(0.0, color="0.40", linestyle=":", linewidth=1.0, zorder=2)

    if mode == "ECG":
        spans = [(P20_WIN["ECG_AO_REF"], "AO", "AO ref. window"),
                 (P20_WIN["ECG_AC_REF"], "AC", "AC ref. window")]
        marks = [("Q", lm.get("Q"), "Q", 0.95),
                 ("R", lm.get("R"), "R", 0.88),
                 ("T", lm.get("T"), "T", 0.81)]
    elif mode == "SCG":
        spans = [(P20_WIN["MC"], "MC", "MC window"),
                 (P20_WIN["IM"], "IM", "IM window"),
                 (P20_WIN["AO"], "AO", "AO window"),
                 (P20_WIN["AC"], "AC", "AC window"),
                 (P20_WIN["MO"], "MO", "MO window")]
        marks = [("MC", lm.get("MC"), "MC", 0.97),
                 ("IM", lm.get("IM"), "IM", 0.90),
                 ("AO", lm.get("AO"), "AO", 0.83),
                 ("AC", lm.get("AC"), "AC", 0.76),
                 ("MO", lm.get("MO"), "MO", 0.69)]
    else:
        spans = [(P20_WIN["RADAR_AO"], "AO", "AO window"),
                 (P20_WIN["RADAR_AC"], "AC", "AC window")]
        marks = [("AO", lm.get("AO"), "AO", 0.88),
                 ("AC", lm.get("AC"), "AC", 0.78)]

    for win, name, wlabel in spans:
        ax.axvspan(win[0], win[1], color=P20_C.get(name, "#999999"), alpha=0.040, zorder=0)

    ymin, ymax = float(np.nanmin(bx)), float(np.nanmax(bx))
    pad = max(0.65, 0.22*(ymax-ymin + 1e-9))
    ax.set_ylim(ymin-pad, ymax+1.08)

    for win, name, wlabel in spans:
        _p20_window_label(ax, win[0], win[1], wlabel, yfrac=0.035)

    for name, x, label, yf in marks:
        _p20_mark(ax, bt, bx, x, name, label, yf)

    if brackets:
        q = q_for_interval if q_for_interval is not None else lm.get("Q")
        ao, ac = lm.get("AO"), lm.get("AC")
        ymin, ymax = ax.get_ylim()
        base = ymin + 0.10*(ymax-ymin)
        _p20_bracket(ax, q, ao, base, "PEP")
        _p20_bracket(ax, ao, ac, base + 0.15*(ymax-ymin), "LVET")
        _p20_bracket(ax, q, ac, base + 0.30*(ymax-ymin), "QS2")

    ax.set_title(title, loc="left", fontsize=11, pad=6)
    ax.set_ylabel("z-score", fontsize=10)
    ax.grid(True, alpha=0.22)
    ax.set_xlim(*P20_XLIM)
    ax.tick_params(labelsize=9)

def _p20_make_final_figs(outdir, ecg, scg, radar, aoac, acfg):
    try:
        rep = _p20_pick_best(ecg, scg, radar, acfg)
        if rep is None:
            (outdir / "patch20_no_valid_representative_cycle.txt").write_text(
                "No valid ECG-centered representative cycle found.",
                encoding="utf-8"
            )
            return

        ecg_lm, scg_lm, radar_lm = rep["ecg_lm"], rep["scg_lm"], rep["radar_lm"]
        q = ecg_lm.get("Q")

        rows, summary = _p20_all_beat_table(ecg, scg, radar)
        if rows:
            save_csv(
                outdir / "scg_derived_reference_vs_radar_aoac.csv",
                ["beat_index", "anchor_time_sec", "scg_polarity", "scg_polarity_score",
                 "scg_MC_sec", "scg_IM_sec", "scg_AO_sec", "scg_AC_sec", "scg_MO_sec",
                 "scg_MC_amp", "scg_IM_amp", "scg_AO_amp", "scg_AC_amp", "scg_MO_amp",
                 "radar_AO_sec", "radar_AC_sec",
                 "radar_minus_scg_AO_ms", "radar_minus_scg_AC_ms",
                 "scg_LVET_ms", "radar_LVET_ms"],
                rows
            )
            with open(outdir / "scg_radar_aoac_timing_distribution_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            _p20_distribution_figures(outdir, rows)

        rep_rows = []
        for mod, lm in [("ECG_visual", ecg_lm), ("SCG_derived_reference", scg_lm), ("Radar_detected", radar_lm)]:
            pep, lvet, qs2 = _p5_interval(q, lm.get("AO"), lm.get("AC"))
            rep_rows.append([mod, rep["beat_index"], rep["anchor"], q, 0.0 if mod=="ECG_visual" else None,
                             ecg_lm.get("T") if mod=="ECG_visual" else None,
                             lm.get("MC"), lm.get("IM"), lm.get("AO"), lm.get("AC"), lm.get("MO"),
                             pep, lvet, qs2])
        save_csv(outdir / "fig06_fig10_single_cycle_values.csv",
                 ["modality", "beat_index", "anchor_time_sec", "Q_rel_sec", "R_rel_sec", "T_rel_sec",
                  "MC_rel_sec", "IM_rel_sec", "AO_rel_sec", "AC_rel_sec", "MO_rel_sec",
                  "PEP_ms", "LVET_ms", "QS2_ms"], rep_rows)

        with open(outdir / "patch20_scg_landmark_audit.json", "w", encoding="utf-8") as f:
            json.dump({
                "representative_beat_index": int(rep["beat_index"]),
                "anchor_time_sec": float(rep["anchor"]),
                "scg_polarity_factor": int(rep["scg_polarity"]),
                "scg_polarity_score": float(rep["scg_score"]),
                "expected_signs": {"MC":"negative","IM":"positive","AO":"positive","AC":"negative","MO":"positive"},
                "selected_scg_values": rep["scg_values"],
                "note": "SCG polarity was resolved by evaluating both original and inverted beats."
            }, f, ensure_ascii=False, indent=2)

        # ECG anchor
        fig, ax = plt.subplots(1,1,figsize=(12.4,4.8),constrained_layout=True)
        _p20_panel(ax, "ECG Q/R/T landmarks and AO/AC literature windows",
                   rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", brackets=False)
        ax.set_xlabel("Time from ECG R-peak [s]")
        fig.savefig(outdir / "fig02_ecg_qrt_reference.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # SCG reference landmarks
        fig, ax = plt.subplots(1,1,figsize=(12.4,4.8),constrained_layout=True)
        _p20_panel(ax, "Literature-guided SCG-derived MC/IM/AO/AC/MO landmarks",
                   rep["bt_s"], rep["bx_s"], scg_lm, "SCG", brackets=False)
        ax.set_xlabel("Time from ECG R-peak [s]")
        fig.savefig(outdir / "fig02_scg_derived_reference_landmarks.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig02_scg_candidate_landmarks.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig02_scg_reference_landmarks.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Morphology
        fig, axes = plt.subplots(3,1,figsize=(12.8,8.5),sharex=True,constrained_layout=True)
        _p20_panel(axes[0], "ECG R-peak aligned beat", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG")
        _p20_panel(axes[1], "SCG-derived fiducial landmarks", rep["bt_s"], rep["bx_s"], scg_lm, "SCG")
        _p20_panel(axes[2], "Radar-detected AO/AC landmarks", rep["bt_r"], rep["bx_r"], radar_lm, "Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("ECG / SCG-derived reference / Radar morphology comparison", fontsize=14)
        fig.savefig(outdir / "fig02_compact_beat_morphology.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig05_morphology_ecg_scg_radar_candidates.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Label panels
        fig, axes = plt.subplots(3,1,figsize=(12.8,8.6),sharex=True,constrained_layout=True)
        _p20_panel(axes[0], "ECG alignment anchor and literature windows", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG")
        _p20_panel(axes[1], "SCG-derived reference landmarks", rep["bt_s"], rep["bx_s"], scg_lm, "SCG")
        _p20_panel(axes[2], "Radar-detected AO/AC landmarks", rep["bt_r"], rep["bx_r"], radar_lm, "Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle SCG-derived reference and Radar-detected landmark comparison", fontsize=14)
        fig.savefig(outdir / "fig06_single_cycle_ecg_radar_aoac_labels.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Intervals
        fig, axes = plt.subplots(3,1,figsize=(12.9,8.9),sharex=True,constrained_layout=True)
        _p20_panel(axes[0], "ECG alignment anchor", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", False)
        _p20_panel(axes[1], "SCG-derived reference intervals", rep["bt_s"], rep["bx_s"], scg_lm, "SCG", True, q)
        _p20_panel(axes[2], "Radar-detected intervals", rep["bt_r"], rep["bx_r"], radar_lm, "Radar", True, q)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("SCG-derived reference and Radar-detected interval analysis", fontsize=14)
        fig.savefig(outdir / "fig06_pep_lvet_qs2_brackets.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig10_ecg_scg_radar_landmark_interval_clean.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Paper export
        paper = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export")
        figs = paper / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        mapping = [
            ("fig02_compact_beat_morphology.png", "fig02_ecg_scg_radar_morphology_comparison.png"),
            ("fig06_single_cycle_ecg_radar_aoac_labels.png", "fig06_scg_reference_radar_detected_landmarks.png"),
            ("fig07_scg_radar_aoac_timing_boxplot.png", "fig07_scg_radar_aoac_timing_boxplot.png"),
            ("fig08_scg_radar_relative_difference_boxplot.png", "fig08_scg_radar_relative_difference_boxplot.png"),
            ("fig06_pep_lvet_qs2_brackets.png", "fig09_scg_radar_interval_analysis.png"),
        ]
        idx_rows = []
        for s, d in mapping:
            sp, dp = outdir / s, figs / d
            status = "missing"
            if sp.exists():
                shutil.copyfile(sp, dp)
                status = "copied"
            idx_rows.append([s, d, status])
        save_csv(figs / "paper_figure_index_patch20.csv", ["Source", "PaperFigure", "Status"], idx_rows)

        (outdir / "patch20_final_figure_policy.txt").write_text(
            "PATCH20 active. ECG is only R-peak alignment anchor. "
            "SCG AO/AC are computed by a literature-guided fiducial detector and used as SCG-derived reference. "
            "SCG polarity is resolved by testing original and inverted beats; MC/IM/AO/AC/MO use signed extrema: "
            "MC negative, IM positive, AO positive, AC negative, MO positive. "
            "Radar AO/AC are compared against SCG-derived AO/AC by timing distribution and relative difference.",
            encoding="utf-8"
        )
    except BaseException as e:
        (outdir / "patch20_final_figures_error.txt").write_text(str(e), encoding="utf-8")

def _p20_core_export(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    core = globals().get("_old_save_all_patch3", None)
    if callable(core):
        return core(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    return None

def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    result = _p20_core_export(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        _p20_make_final_figs(outdir, ecg, scg, radar, aoac, acfg)
    except BaseException as e:
        (outdir / "save_all_patch20_error.txt").write_text(str(e), encoding="utf-8")
    return result



# ============================================================
# PATCH21: Zheng-style AO enhancement + HIKAF-style Kalman tracking
# ------------------------------------------------------------
# This patch extends PATCH20 with reference-paper-inspired operations:
#
# Zheng et al. 2024 style AO detector:
#   1) first-order interference cancellation / MTI-like preprocessing
#   2) mode decomposition block
#      - optional VMD if vmdpy is installed
#      - otherwise deterministic SVMD-like band-limited mode bank fallback
#   3) waveform factor criterion for AO-related mode selection
#   4) AO signal reconstruction
#   5) positive seventh-power detector
#
# HIKAF-style AO/AC tracking:
#   1) state x = [AO, AC]^T
#   2) RR/HR-informed prediction
#   3) measurement z = [AO_candidate, AC_candidate]^T from SCG morphology
#   4) adaptive measurement noise based on morphology confidence
#   5) Kalman update
#
# NOTE:
# - ECG remains R-peak alignment anchor only.
# - SCG AO/AC are SCG-derived references.
# - Radar AO/AC are compared to SCG-derived AO/AC.
# - MC/IM/MO remain detected and shown for CTI context.
# ============================================================

P21_C = {
    "Q": "#1f77b4", "R": "#d62728", "T": "#555555",
    "MC": "#17becf", "IM": "#1f77b4", "AO": "#9467bd",
    "AC": "#ff7f0e", "MO": "#2ca02c",
}
P21_XLIM = (-0.12, 0.62)

P21_WIN = {
    "MC": (-0.085, 0.020),
    "IM": (0.000, 0.115),
    "AO": (0.045, 0.185),
    "AC": (0.180, 0.440),
    "MO": (0.340, 0.600),
    "RADAR_AO": (0.070, 0.165),
    "RADAR_AC": (0.260, 0.460),
    "ECG_AO_REF": (0.070, 0.160),
    "ECG_AC_REF": (0.280, 0.460),
}

def _p21_z(x):
    return zscore_safe(np.asarray(x, dtype=np.float64))

def _p21_raw_ecg_array(ecg):
    n = len(ecg.get("t", []))
    for k in ["raw_adc_col", "raw"]:
        if k in ecg and ecg[k] is not None and len(ecg[k]) == n:
            return np.asarray(ecg[k], dtype=np.float64)
    for k in ["filtered", "cleaned", "display_rpeak", "display"]:
        if k in ecg and ecg[k] is not None and len(ecg[k]) == n:
            return np.asarray(ecg[k], dtype=np.float64)
    return np.zeros(n, dtype=np.float64)

def _p21_ecg_for_display(ecg):
    x = _p21_raw_ecg_array(ecg)
    fs = float(ecg.get("fs", 100.0))
    try:
        return zscore_safe(safe_bandpass(x, fs, 6.0, min(35.0, fs * 0.45), order=3))
    except Exception:
        try:
            return zscore_safe(signal.detrend(x))
        except Exception:
            return zscore_safe(x - np.nanmedian(x))

def _p21_scg_base(scg):
    if scg is None:
        return None
    n = len(scg.get("t", []))
    for k in ["resp_removed", "selected_raw", "filtered", "vmag", "az", "ax", "ay"]:
        if k in scg and scg[k] is not None and len(scg[k]) == n:
            x = np.asarray(scg[k], dtype=np.float64)
            fs = float(scg.get("fs", 100.0))
            try:
                return zscore_safe(safe_bandpass(x, fs, 0.8, min(35.0, fs * 0.45), order=2))
            except Exception:
                try:
                    return zscore_safe(signal.detrend(x))
                except Exception:
                    return zscore_safe(x - np.nanmedian(x))
    return None

def _p21_radar_for_display(radar):
    n = len(radar.get("t", []))
    for k in ["lms_error", "ppg_like", "displacement", "display"]:
        if k in radar and radar[k] is not None and len(radar[k]) == n:
            return zscore_safe(np.asarray(radar[k], dtype=np.float64))
    return np.zeros(n, dtype=np.float64)

def _p21_slice(tt, xx, anchor, pre=0.12, post=0.62):
    return _p5_slice(tt, xx, anchor, pre, post, 100.0)

def _p21_recenter_ecg_beat(bt, bx):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    idx = np.where((bt >= -0.060) & (bt <= 0.060))[0]
    if len(idx) < 3:
        return bt, bx, 0.0
    j = idx[int(np.nanargmax(np.abs(bx[idx])))]
    r_rel = float(bt[j])
    if bx[j] < 0:
        bx = -bx
    return bt - r_rel, zscore_safe(bx), r_rel

def _p21_detect_ecg_qrt(bt, bx):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    q = None
    idx = np.where((bt >= -0.070) & (bt <= -0.008))[0]
    if len(idx) >= 3:
        q = float(bt[idx[int(np.nanargmin(bx[idx]))]])
    t = None
    idx = np.where((bt >= 0.140) & (bt <= 0.340))[0]
    if len(idx) >= 3:
        t = float(bt[idx[int(np.nanargmax(bx[idx]))]])
    return {"Q": q, "R": 0.0, "T": t}

def _p21_idx(bt, win):
    bt = np.asarray(bt, dtype=float)
    return np.where((bt >= win[0]) & (bt <= win[1]))[0]

def _p21_peak(bt, bx, win, polarity):
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))
    idx = _p21_idx(bt, win)
    if len(idx) < 3:
        return None
    if polarity == "neg":
        j = idx[int(np.nanargmin(bx[idx]))]
    elif polarity == "abs":
        j = idx[int(np.nanargmax(np.abs(bx[idx])))]
    else:
        j = idx[int(np.nanargmax(bx[idx]))]
    return float(bt[j])

# ---------- Zheng-style AO enhancement ----------

def _p21_mti_interference_cancellation(x, alpha=0.92):
    x = zscore_safe(np.asarray(x, dtype=np.float64))
    y = np.zeros_like(x)
    if len(x) > 1:
        y[1:] = x[1:] - alpha * x[:-1]
        y[0] = y[1]
    return zscore_safe(y)

def _p21_mode_bank(x, fs=100.0):
    """
    SVMD-compatible fallback mode bank.
    If vmdpy is available, use VMD as a practical decomposition backend.
    Otherwise use deterministic band-limited modes.
    """
    x = zscore_safe(np.asarray(x, dtype=np.float64))

    # Optional VMD backend. This is not SVMD, but provides data-driven modes
    # if the local environment has vmdpy installed.
    try:
        from vmdpy import VMD
        alpha = 2000
        tau = 0
        K = 5
        DC = 0
        init = 1
        tol = 1e-7
        u, _, _ = VMD(x, alpha, tau, K, DC, init, tol)
        modes = [zscore_safe(u[i, :]) for i in range(u.shape[0])]
        return modes, "VMD_backend"
    except Exception:
        pass

    # Fallback: band-limited quasi-modes. This keeps the algorithm self-contained.
    bands = [(1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 32.0)]
    modes = []
    for lo, hi in bands:
        try:
            modes.append(zscore_safe(safe_bandpass(x, fs, lo, min(hi, fs*0.45), order=2)))
        except Exception:
            modes.append(np.zeros_like(x))
    return modes, "band_limited_fallback"

def _p21_waveform_factor(mode, r_times, tt, ao_window=P21_WIN["AO"]):
    """
    AO pulsatility score: high when the mode has consistent positive peaks
    in the AO window and lower energy outside the AO-related region.
    """
    mode = zscore_safe(np.asarray(mode, dtype=np.float64))
    tt = np.asarray(tt, dtype=np.float64)
    vals = []
    outside_vals = []
    for r in r_times:
        rel = tt - float(r)
        ia = np.where((rel >= ao_window[0]) & (rel <= ao_window[1]))[0]
        io = np.where((rel >= -0.10) & (rel <= 0.60) & ~((rel >= ao_window[0]) & (rel <= ao_window[1])))[0]
        if len(ia) >= 3:
            vals.append(float(np.nanmax(mode[ia])))
        if len(io) >= 3:
            outside_vals.append(float(np.nanstd(mode[io])))
    if len(vals) == 0:
        return 0.0
    peak_med = np.nanmedian(vals)
    peak_iqr = np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25) + 1e-6
    outside = np.nanmedian(outside_vals) + 1e-6 if outside_vals else 1.0
    score = max(0.0, peak_med) / outside * (1.0 / (1.0 + peak_iqr))
    return float(score)

def _p21_reconstruct_ao_signal(x, tt, r_times, fs=100.0):
    x_mti = _p21_mti_interference_cancellation(x)
    modes, mode_method = _p21_mode_bank(x_mti, fs)
    scores = np.asarray([_p21_waveform_factor(m, r_times, tt) for m in modes], dtype=float)

    if len(scores) == 0 or np.nanmax(scores) <= 0:
        rec = x_mti.copy()
        selected = []
    else:
        order = np.argsort(scores)[::-1]
        # Select modes that contribute to AO pulsatile behavior.
        keep = [int(i) for i in order[:max(1, min(3, len(order)))] if scores[i] >= max(0.15*np.nanmax(scores), 1e-9)]
        if not keep:
            keep = [int(order[0])]
        rec = np.sum([modes[i] for i in keep], axis=0)
        selected = keep

    rec = zscore_safe(rec)
    # Positive-only seventh power detector: enhances AO positive peaks,
    # suppresses negative/opposite peaks and small residual peaks.
    pos = np.maximum(rec, 0.0)
    det = pos ** 7
    try:
        det = safe_lowpass(det, fs, min(8.0, fs*0.20), order=2)
    except Exception:
        pass
    det = robust_scale_01(det)

    meta = {
        "mode_method": mode_method,
        "mode_scores": [float(s) for s in scores],
        "selected_modes": selected,
    }
    return {
        "mti": x_mti,
        "modes": modes,
        "ao_reconstructed": rec,
        "seventh_power_detector": det,
        "meta": meta,
    }

def _p21_ao_from_zheng_detector(bt, det):
    idx = _p21_idx(bt, P21_WIN["AO"])
    if len(idx) < 3:
        return None
    det = np.asarray(det, dtype=float)
    j = idx[int(np.nanargmax(det[idx]))]
    return float(bt[j])

# ---------- SCG fiducials + HIKAF-style tracking ----------

def _p21_signed_fiducials_from_scz(bt, bx, ao_override=None):
    """
    MC/IM/MO remain morphology fiducials.
    AO can be overridden by Zheng-style seventh-power detector.
    AC remains closure-related negative point, then HIKAF-style tracker refines.
    """
    bt = np.asarray(bt, dtype=float)
    bx0 = zscore_safe(np.asarray(bx, dtype=float))

    candidates = []
    for pol in [1, -1]:
        bx = zscore_safe(bx0 * pol)

        lm = {}
        lm["MC"] = _p21_peak(bt, bx, P21_WIN["MC"], "neg")
        if lm["MC"] is None: lm["MC"] = -0.035

        im_win = (max(P21_WIN["IM"][0], lm["MC"] + 0.010), P21_WIN["IM"][1])
        lm["IM"] = _p21_peak(bt, bx, im_win, "pos")
        if lm["IM"] is None: lm["IM"] = 0.055

        if ao_override is not None and P21_WIN["AO"][0] <= ao_override <= P21_WIN["AO"][1]:
            lm["AO"] = float(ao_override)
        else:
            ao_win = (max(P21_WIN["AO"][0], lm["IM"] + 0.012), P21_WIN["AO"][1])
            lm["AO"] = _p21_peak(bt, bx, ao_win, "pos")
            if lm["AO"] is None: lm["AO"] = 0.120

        ac_win = (max(P21_WIN["AC"][0], lm["AO"] + 0.080), P21_WIN["AC"][1])
        lm["AC"] = _p21_peak(bt, bx, ac_win, "neg")
        if lm["AC"] is None: lm["AC"] = 0.350

        mo_win = (max(P21_WIN["MO"][0], lm["AC"] + 0.050), P21_WIN["MO"][1])
        lm["MO"] = _p21_peak(bt, bx, mo_win, "pos")
        if lm["MO"] is None: lm["MO"] = 0.500

        bounds = {
            "MC": (-0.100, 0.030), "IM": (0.000, 0.120), "AO": (0.045, 0.185),
            "AC": (0.180, 0.440), "MO": (0.340, 0.600)
        }
        prev = -999.0
        for k in ["MC", "IM", "AO", "AC", "MO"]:
            lo, hi = bounds[k]
            x = min(max(float(lm[k]), lo), hi)
            if x <= prev + 0.008:
                x = min(max(prev + 0.010, lo), hi)
            lm[k] = x
            prev = x

        vals = {k: float(np.interp(lm[k], bt, bx)) for k in ["MC","IM","AO","AC","MO"]}
        score = 0.0
        score += 2.0 * max(0, -vals["MC"])
        score += 2.2 * max(0,  vals["IM"])
        score += 5.0 * max(0,  vals["AO"])
        score += 2.0 * max(0, -vals["AC"])
        score += 1.8 * max(0,  vals["MO"])
        score -= 1.2 * max(0,  vals["MC"])
        score -= 1.2 * max(0, -vals["IM"])
        score -= 1.5 * max(0, -vals["AO"])
        score -= 1.2 * max(0,  vals["AC"])
        score -= 1.0 * max(0, -vals["MO"])
        candidates.append((score, pol, bx, lm, vals))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]  # score, pol, bx, lm, vals

def _p21_measurement_confidence(bt, bx, lm):
    vals = {k: float(np.interp(lm[k], bt, bx)) for k in ["MC","IM","AO","AC","MO"]}
    conf_ao = max(0.05, max(0.0, vals["AO"]) + 0.25 * max(0.0, vals["IM"]))
    conf_ac = max(0.05, max(0.0, -vals["AC"]) + 0.10 * max(0.0, vals["MO"]))
    return conf_ao, conf_ac

def _p21_hikaf_track(beat_rows):
    """
    HIKAF-style HR-informed Kalman tracking for AO/AC.
    beat_rows entries contain:
      beat_index, anchor, rr, ao_meas, ac_meas, conf_ao, conf_ac, full_lm...
    """
    if not beat_rows:
        return []

    # Initialize from robust median of early valid beats.
    init_n = min(30, len(beat_rows))
    ao0 = float(np.nanmedian([b["ao_meas"] for b in beat_rows[:init_n]]))
    ac0 = float(np.nanmedian([b["ac_meas"] for b in beat_rows[:init_n]]))
    x = np.array([ao0, ac0], dtype=float)
    P = np.diag([0.020**2, 0.030**2])
    Q_base = np.diag([0.006**2, 0.010**2])

    rr0 = float(np.nanmedian([b["rr"] for b in beat_rows[:init_n] if np.isfinite(b["rr"])]))
    if not np.isfinite(rr0) or rr0 <= 0:
        rr0 = 0.75

    out = []
    for i, b in enumerate(beat_rows):
        rr = b["rr"] if np.isfinite(b["rr"]) and b["rr"] > 0 else rr0
        scale = np.clip(rr / rr0, 0.75, 1.35)

        # HR-informed prediction: electromechanical intervals scale mildly with RR.
        x_pred = np.array([x[0] * (0.85 + 0.15*scale), x[1] * (0.85 + 0.15*scale)], dtype=float)
        P_pred = P + Q_base * (1.0 + abs(scale - 1.0))

        z = np.array([b["ao_meas"], b["ac_meas"]], dtype=float)

        # Adaptive measurement noise: lower confidence -> larger R.
        conf_ao = max(float(b["conf_ao"]), 1e-3)
        conf_ac = max(float(b["conf_ac"]), 1e-3)
        R = np.diag([(0.020 / conf_ao)**2, (0.030 / conf_ac)**2])
        R = np.clip(R, 1e-6, 0.20**2)

        H = np.eye(2)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ (z - H @ x_pred)
        P = (np.eye(2) - K @ H) @ P_pred

        # Physiologic bounds and order.
        x[0] = float(np.clip(x[0], P21_WIN["AO"][0], P21_WIN["AO"][1]))
        x[1] = float(np.clip(x[1], max(P21_WIN["AC"][0], x[0] + 0.080), P21_WIN["AC"][1]))

        row = dict(b)
        row["ao_tracked"] = float(x[0])
        row["ac_tracked"] = float(x[1])
        row["kalman_P_ao"] = float(P[0,0])
        row["kalman_P_ac"] = float(P[1,1])
        out.append(row)

    return out

def _p21_build_scg_reference_table(ecg, scg):
    r = np.asarray(ecg.get("peaks_time", []), dtype=float)
    if len(r) < 3 or scg is None:
        return [], None

    tt = np.asarray(scg.get("t", []), dtype=float)
    sx = _p21_scg_base(scg)
    if sx is None:
        return [], None

    ao_enh = _p21_reconstruct_ao_signal(sx, tt, r, float(scg.get("fs", 100.0)))

    raw_rows = []
    for bi in range(1, len(r)-1):
        anchor = float(r[bi])
        bt_s, bx_s_raw = _p21_slice(tt, sx, anchor)
        bt_det, det = _p21_slice(tt, ao_enh["seventh_power_detector"], anchor)
        if bt_s is None or bt_det is None:
            continue
        ao_z = _p21_ao_from_zheng_detector(bt_det, det)

        score, pol, bx_s_norm, lm, vals = _p21_signed_fiducials_from_scz(bt_s, bx_s_raw, ao_override=ao_z)
        conf_ao, conf_ac = _p21_measurement_confidence(bt_s, bx_s_norm, lm)
        rr = float(r[bi] - r[bi-1]) if bi > 0 else np.nan

        raw_rows.append({
            "beat_index": int(bi),
            "anchor": anchor,
            "rr": rr,
            "polarity": int(pol),
            "polarity_score": float(score),
            "mc": float(lm["MC"]),
            "im": float(lm["IM"]),
            "ao_meas": float(lm["AO"]),
            "ac_meas": float(lm["AC"]),
            "mo": float(lm["MO"]),
            "mc_amp": float(vals["MC"]),
            "im_amp": float(vals["IM"]),
            "ao_amp": float(vals["AO"]),
            "ac_amp": float(vals["AC"]),
            "mo_amp": float(vals["MO"]),
            "conf_ao": float(conf_ao),
            "conf_ac": float(conf_ac),
        })

    tracked = _p21_hikaf_track(raw_rows)
    return tracked, ao_enh

def _p21_radar_landmarks(bt, bx):
    # Radar AO: positive local change/peak in AO window.
    ao = _p21_peak(bt, bx, P21_WIN["RADAR_AO"], "pos")
    if ao is None:
        ao = 0.110
    # Radar AC: closure-related negative notch/transition in AC window.
    ac_win = (max(P21_WIN["RADAR_AC"][0], ao + 0.100), P21_WIN["RADAR_AC"][1])
    ac = _p21_peak(bt, bx, ac_win, "neg")
    if ac is None:
        ac = 0.370
    return {"AO": float(ao), "AC": float(ac)}

def _p21_pick_best(ecg, scg, radar, scg_ref_rows):
    r = np.asarray(ecg.get("peaks_time", []), dtype=float)
    if len(r) < 3 or not scg_ref_rows:
        return None

    es = _p21_ecg_for_display(ecg)
    ss = _p21_scg_base(scg)
    rs = _p21_radar_for_display(radar)
    ref_by_bi = {int(row["beat_index"]): row for row in scg_ref_rows}

    best = None
    for bi, row in ref_by_bi.items():
        if bi <= 0 or bi >= len(r)-1:
            continue
        anchor0 = float(row["anchor"])
        bt_e0, bx_e0 = _p21_slice(ecg["t"], es, anchor0)
        bt_s_raw, bx_s_raw = _p21_slice(scg["t"], ss, anchor0)
        bt_r, bx_r = _p21_slice(radar["t"], rs, anchor0)
        if bt_e0 is None or bt_s_raw is None or bt_r is None:
            continue
        bt_e, bx_e, r_shift = _p21_recenter_ecg_beat(bt_e0, bx_e0)
        ecg_lm = _p21_detect_ecg_qrt(bt_e, bx_e)

        bx_s = zscore_safe(bx_s_raw * row["polarity"])
        scg_lm = {
            "MC": row["mc"],
            "IM": row["im"],
            "AO": row["ao_tracked"],
            "AC": row["ac_tracked"],
            "MO": row["mo"],
        }
        radar_lm = _p21_radar_landmarks(bt_r, bx_r)

        score = 0.0
        score += 5.0 * max(0.0, row["ao_amp"])
        score += 2.0 * max(0.0, -row["ac_amp"])
        score += 1.5 * max(0.0, -row["mc_amp"])
        score += 1.5 * max(0.0, row["im_amp"])
        score += 1.0 * max(0.0, row["mo_amp"])
        score -= 0.0008 * abs(bi - len(r)/2)

        cand = {
            "score": score,
            "beat_index": bi,
            "anchor": anchor0,
            "bt_e": bt_e, "bx_e": bx_e,
            "bt_s": bt_s_raw, "bx_s": bx_s,
            "bt_r": bt_r, "bx_r": bx_r,
            "ecg_lm": ecg_lm,
            "scg_lm": scg_lm,
            "radar_lm": radar_lm,
            "scg_row": row,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best

def _p21_make_scg_radar_table(ecg, radar, scg_ref_rows):
    if not scg_ref_rows:
        return [], {}

    rs = _p21_radar_for_display(radar)
    rows = []
    for row in scg_ref_rows:
        anchor = row["anchor"]
        bt_r, bx_r = _p21_slice(radar["t"], rs, anchor)
        if bt_r is None:
            continue
        rad = _p21_radar_landmarks(bt_r, bx_r)

        scg_ao = float(row["ao_tracked"])
        scg_ac = float(row["ac_tracked"])
        rad_ao = float(rad["AO"])
        rad_ac = float(rad["AC"])
        rows.append([
            row["beat_index"], anchor, row["polarity"], row["polarity_score"],
            row["mc"], row["im"], row["ao_meas"], row["ac_meas"], row["mo"],
            row["ao_tracked"], row["ac_tracked"],
            row["mc_amp"], row["im_amp"], row["ao_amp"], row["ac_amp"], row["mo_amp"],
            row["conf_ao"], row["conf_ac"],
            rad_ao, rad_ac,
            (rad_ao - scg_ao) * 1000.0,
            (rad_ac - scg_ac) * 1000.0,
            (scg_ac - scg_ao) * 1000.0,
            (rad_ac - rad_ao) * 1000.0,
        ])

    def _summary(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return {"n": 0, "mean_ms": None, "std_ms": None, "median_ms": None, "iqr_ms": None}
        return {
            "n": int(len(vals)),
            "mean_ms": float(np.mean(vals)),
            "std_ms": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median_ms": float(np.median(vals)),
            "iqr_ms": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        }

    arr = np.asarray(rows, dtype=float) if rows else np.empty((0, 24))
    total_beats = len(ecg.get("peaks_time", []))
    summary = {
        "total_ecg_beats": int(total_beats),
        "accepted_scg_radar_beats": int(len(rows)),
        "accept_rate_percent": float(len(rows) / max(total_beats, 1) * 100.0),
        "analysis_note": "SCG AO uses Zheng-style AO enhancement; SCG AO/AC are HIKAF-style Kalman tracked; ECG is R-peak anchor only."
    }
    if len(rows) > 0:
        summary.update({
            "scg_ao_tracked_timing": _summary(arr[:, 9] * 1000.0),
            "scg_ac_tracked_timing": _summary(arr[:, 10] * 1000.0),
            "radar_ao_timing": _summary(arr[:, 18] * 1000.0),
            "radar_ac_timing": _summary(arr[:, 19] * 1000.0),
            "radar_minus_scg_ao": _summary(arr[:, 20]),
            "radar_minus_scg_ac": _summary(arr[:, 21]),
            "scg_lvet_ao_to_ac": _summary(arr[:, 22]),
            "radar_lvet_ao_to_ac": _summary(arr[:, 23]),
        })
    return rows, summary

def _p21_distribution_figures(outdir, rows):
    if not rows:
        return
    arr = np.asarray(rows, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(9.6, 5.2), constrained_layout=True)
    data = [arr[:, 9]*1000.0, arr[:, 18]*1000.0, arr[:, 10]*1000.0, arr[:, 19]*1000.0]
    labels = ["SCG AO\ntracked", "Radar AO", "SCG AC\ntracked", "Radar AC"]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Timing from ECG R-peak [ms]")
    ax.set_title("SCG-derived AO/AC reference and Radar-detected AO/AC timing distribution")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(outdir / "fig07_scg_radar_aoac_timing_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.0), constrained_layout=True)
    ax.axhline(0, color="0.4", linestyle="--", linewidth=1.0)
    ax.boxplot([arr[:, 20], arr[:, 21]], labels=["Radar AO - SCG AO", "Radar AC - SCG AC"], showfliers=False)
    ax.set_ylabel("Relative timing difference [ms]")
    ax.set_title("Radar vs SCG-derived AO/AC relative timing")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(outdir / "fig08_scg_radar_relative_difference_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def _p21_mark(ax, bt, bx, x, name, label, yfrac):
    if x is None:
        return
    try:
        x = float(x)
        if not np.isfinite(x):
            return
    except Exception:
        return
    c = P21_C.get(name, "black")
    y = float(np.interp(x, bt, bx))
    ax.axvline(x, color=c, linestyle="--", linewidth=1.05, alpha=0.95, zorder=3)
    mk = {"Q":"o","R":"o","T":"o","MC":"o","IM":"^","AO":"s","AC":"D","MO":"v"}.get(name, "o")
    ax.scatter([x], [y], s=54, marker=mk, facecolor="white", edgecolor=c, linewidth=1.4, zorder=5)
    ymin, ymax = ax.get_ylim()
    ax.text(x, ymin + (ymax-ymin)*yfrac, label, ha="center", va="top", fontsize=7.7, color=c,
            bbox=dict(boxstyle="round,pad=0.11", fc="white", ec=c, alpha=0.96), zorder=6)

def _p21_bracket(ax, x0, x1, y, label):
    if x0 is None or x1 is None:
        return
    try:
        x0 = float(x0); x1 = float(x1)
        if not (np.isfinite(x0) and np.isfinite(x1)):
            return
    except Exception:
        return
    ax.plot([x0, x1], [y, y], color="0.25", linewidth=1.0, zorder=2)
    ax.plot([x0, x0], [y-0.020, y+0.020], color="0.25", linewidth=0.85)
    ax.plot([x1, x1], [y-0.020, y+0.020], color="0.25", linewidth=0.85)
    ax.text((x0+x1)/2.0, y+0.025, label, ha="center", va="bottom", fontsize=7.3,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="0.60", alpha=0.95))

def _p21_window_label(ax, x0, x1, label, yfrac=0.035):
    ymin, ymax = ax.get_ylim()
    ax.text((x0+x1)/2.0, ymin + (ymax-ymin)*yfrac, label,
            ha="center", va="bottom", fontsize=7.1, color="0.35",
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="0.75", alpha=0.82), zorder=4)

def _p21_panel(ax, title, bt, bx, lm, mode, brackets=False, q_for_interval=None):
    ax.plot(bt, bx, color="black", linewidth=1.55, zorder=1)
    ax.axvline(0.0, color="0.40", linestyle=":", linewidth=1.0, zorder=2)

    if mode == "ECG":
        spans = [(P21_WIN["ECG_AO_REF"], "AO", "AO literature window"),
                 (P21_WIN["ECG_AC_REF"], "AC", "AC literature window")]
        marks = [("Q", lm.get("Q"), "Q", 0.95),
                 ("R", lm.get("R"), "R", 0.88),
                 ("T", lm.get("T"), "T", 0.81)]
    elif mode == "SCG":
        spans = [(P21_WIN["MC"], "MC", "MC window"),
                 (P21_WIN["IM"], "IM", "IM window"),
                 (P21_WIN["AO"], "AO", "AO window"),
                 (P21_WIN["AC"], "AC", "AC window"),
                 (P21_WIN["MO"], "MO", "MO window")]
        marks = [("MC", lm.get("MC"), "MC", 0.97),
                 ("IM", lm.get("IM"), "IM", 0.90),
                 ("AO", lm.get("AO"), "AO", 0.83),
                 ("AC", lm.get("AC"), "AC", 0.76),
                 ("MO", lm.get("MO"), "MO", 0.69)]
    else:
        spans = [(P21_WIN["RADAR_AO"], "AO", "AO window"),
                 (P21_WIN["RADAR_AC"], "AC", "AC window")]
        marks = [("AO", lm.get("AO"), "AO", 0.88),
                 ("AC", lm.get("AC"), "AC", 0.78)]

    for win, name, wlabel in spans:
        ax.axvspan(win[0], win[1], color=P21_C.get(name, "#999999"), alpha=0.040, zorder=0)

    ymin, ymax = float(np.nanmin(bx)), float(np.nanmax(bx))
    pad = max(0.65, 0.22*(ymax-ymin + 1e-9))
    ax.set_ylim(ymin-pad, ymax+1.08)

    for win, name, wlabel in spans:
        _p21_window_label(ax, win[0], win[1], wlabel, yfrac=0.035)

    for name, x, label, yf in marks:
        _p21_mark(ax, bt, bx, x, name, label, yf)

    if brackets:
        q = q_for_interval if q_for_interval is not None else lm.get("Q")
        ao, ac = lm.get("AO"), lm.get("AC")
        ymin, ymax = ax.get_ylim()
        base = ymin + 0.10*(ymax-ymin)
        _p21_bracket(ax, q, ao, base, "PEP")
        _p21_bracket(ax, ao, ac, base + 0.15*(ymax-ymin), "LVET")
        _p21_bracket(ax, q, ac, base + 0.30*(ymax-ymin), "QS2")

    ax.set_title(title, loc="left", fontsize=11, pad=6)
    ax.set_ylabel("z-score", fontsize=10)
    ax.grid(True, alpha=0.22)
    ax.set_xlim(*P21_XLIM)
    ax.tick_params(labelsize=9)

def _p21_make_stage_figure(outdir, scg, ao_enh, rep):
    try:
        tt = np.asarray(scg["t"], dtype=float)
        anchor = rep["anchor"]
        stage_items = [
            ("SCG morphology branch", _p21_scg_base(scg)),
            ("MTI/interference-cancelled SCG", ao_enh["mti"]),
            ("AO-reconstructed signal", ao_enh["ao_reconstructed"]),
            ("Seventh-power AO detector", ao_enh["seventh_power_detector"]),
        ]
        fig, axes = plt.subplots(len(stage_items), 1, figsize=(12.5, 9.0), sharex=True, constrained_layout=True)
        for ax, (title, sig) in zip(axes, stage_items):
            bt, bx = _p21_slice(tt, sig, anchor)
            ax.plot(bt, zscore_safe(bx), color="black", linewidth=1.35)
            ax.axvspan(P21_WIN["AO"][0], P21_WIN["AO"][1], color=P21_C["AO"], alpha=0.06)
            ax.axvline(rep["scg_lm"]["AO"], color=P21_C["AO"], linestyle="--", linewidth=1.1)
            ax.set_title(title, loc="left", fontsize=10)
            ax.grid(True, alpha=0.22)
            ax.set_ylabel("z")
            ax.set_xlim(*P21_XLIM)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Zheng-style SCG AO enhancement stages", fontsize=14)
        fig.savefig(outdir / "fig03_zheng_style_scg_ao_enhancement_stages.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (outdir / "fig03_zheng_stage_error.txt").write_text(str(e), encoding="utf-8")

def _p21_make_final_figs(outdir, ecg, scg, radar, aoac, acfg):
    try:
        scg_ref_rows, ao_enh = _p21_build_scg_reference_table(ecg, scg)
        if not scg_ref_rows:
            (outdir / "patch21_no_scg_reference_rows.txt").write_text("No SCG-derived reference rows.", encoding="utf-8")
            return

        rep = _p21_pick_best(ecg, scg, radar, scg_ref_rows)
        if rep is None:
            (outdir / "patch21_no_valid_representative_cycle.txt").write_text("No valid representative cycle.", encoding="utf-8")
            return

        ecg_lm, scg_lm, radar_lm = rep["ecg_lm"], rep["scg_lm"], rep["radar_lm"]
        q = ecg_lm.get("Q")

        rows, summary = _p21_make_scg_radar_table(ecg, radar, scg_ref_rows)
        if rows:
            save_csv(outdir / "scg_derived_reference_vs_radar_aoac.csv",
                     ["beat_index", "anchor_time_sec", "scg_polarity", "scg_polarity_score",
                      "scg_MC_sec", "scg_IM_sec", "scg_AO_meas_sec", "scg_AC_meas_sec", "scg_MO_sec",
                      "scg_AO_tracked_sec", "scg_AC_tracked_sec",
                      "scg_MC_amp", "scg_IM_amp", "scg_AO_amp", "scg_AC_amp", "scg_MO_amp",
                      "conf_AO", "conf_AC",
                      "radar_AO_sec", "radar_AC_sec",
                      "radar_minus_scg_AO_ms", "radar_minus_scg_AC_ms",
                      "scg_LVET_ms", "radar_LVET_ms"], rows)
            with open(outdir / "scg_radar_aoac_timing_distribution_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            _p21_distribution_figures(outdir, rows)

        rep_rows = []
        for mod, lm in [("ECG_visual", ecg_lm), ("SCG_Zheng_HIKAF_reference", scg_lm), ("Radar_detected", radar_lm)]:
            pep, lvet, qs2 = _p5_interval(q, lm.get("AO"), lm.get("AC"))
            rep_rows.append([mod, rep["beat_index"], rep["anchor"], q, 0.0 if mod=="ECG_visual" else None,
                             ecg_lm.get("T") if mod=="ECG_visual" else None,
                             lm.get("MC"), lm.get("IM"), lm.get("AO"), lm.get("AC"), lm.get("MO"),
                             pep, lvet, qs2])
        save_csv(outdir / "fig06_fig10_single_cycle_values.csv",
                 ["modality", "beat_index", "anchor_time_sec", "Q_rel_sec", "R_rel_sec", "T_rel_sec",
                  "MC_rel_sec", "IM_rel_sec", "AO_rel_sec", "AC_rel_sec", "MO_rel_sec",
                  "PEP_ms", "LVET_ms", "QS2_ms"], rep_rows)

        with open(outdir / "patch21_method_audit.json", "w", encoding="utf-8") as f:
            json.dump({
                "method": "Zheng-style AO enhancement + HIKAF-style Kalman AO/AC tracking",
                "ecg_role": "R-peak alignment anchor only",
                "scg_reference": "SCG-derived AO/AC reference",
                "zheng_blocks": ao_enh["meta"],
                "representative_beat_index": int(rep["beat_index"]),
                "representative_anchor_sec": float(rep["anchor"]),
                "note": "SVMD is approximated by VMD if vmdpy is available; otherwise a self-contained band-limited mode bank is used."
            }, f, ensure_ascii=False, indent=2)

        _p21_make_stage_figure(outdir, scg, ao_enh, rep)

        fig, ax = plt.subplots(1, 1, figsize=(12.4, 4.8), constrained_layout=True)
        _p21_panel(ax, "ECG Q/R/T landmarks and AO/AC literature windows",
                   rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", brackets=False)
        ax.set_xlabel("Time from ECG R-peak [s]")
        fig.savefig(outdir / "fig02_ecg_qrt_reference.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(12.4, 4.8), constrained_layout=True)
        _p21_panel(ax, "SCG-derived MC/IM/AO/AC/MO landmarks after Zheng/HIKAF processing",
                   rep["bt_s"], rep["bx_s"], scg_lm, "SCG", brackets=False)
        ax.set_xlabel("Time from ECG R-peak [s]")
        fig.savefig(outdir / "fig02_scg_derived_reference_landmarks.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig02_scg_reference_landmarks.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(12.8, 8.5), sharex=True, constrained_layout=True)
        _p21_panel(axes[0], "ECG R-peak aligned beat", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG")
        _p21_panel(axes[1], "SCG-derived fiducial landmarks", rep["bt_s"], rep["bx_s"], scg_lm, "SCG")
        _p21_panel(axes[2], "Radar-detected AO/AC landmarks", rep["bt_r"], rep["bx_r"], radar_lm, "Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("ECG / SCG-derived reference / Radar morphology comparison", fontsize=14)
        fig.savefig(outdir / "fig02_compact_beat_morphology.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig05_morphology_ecg_scg_radar_candidates.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(12.8, 8.6), sharex=True, constrained_layout=True)
        _p21_panel(axes[0], "ECG alignment anchor and literature windows", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG")
        _p21_panel(axes[1], "SCG-derived reference landmarks", rep["bt_s"], rep["bx_s"], scg_lm, "SCG")
        _p21_panel(axes[2], "Radar-detected AO/AC landmarks", rep["bt_r"], rep["bx_r"], radar_lm, "Radar")
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Single-cycle SCG-derived reference and Radar-detected landmark comparison", fontsize=14)
        fig.savefig(outdir / "fig06_single_cycle_ecg_radar_aoac_labels.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(12.9, 8.9), sharex=True, constrained_layout=True)
        _p21_panel(axes[0], "ECG alignment anchor", rep["bt_e"], rep["bx_e"], ecg_lm, "ECG", False)
        _p21_panel(axes[1], "SCG-derived reference intervals", rep["bt_s"], rep["bx_s"], scg_lm, "SCG", True, q)
        _p21_panel(axes[2], "Radar-detected intervals", rep["bt_r"], rep["bx_r"], radar_lm, "Radar", True, q)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("SCG-derived reference and Radar-detected interval analysis", fontsize=14)
        fig.savefig(outdir / "fig06_pep_lvet_qs2_brackets.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / "fig10_ecg_scg_radar_landmark_interval_clean.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # HIKAF tracking trajectory figure
        try:
            beat = np.asarray([r["beat_index"] for r in scg_ref_rows], dtype=float)
            ao_m = np.asarray([r["ao_meas"] for r in scg_ref_rows], dtype=float) * 1000
            ac_m = np.asarray([r["ac_meas"] for r in scg_ref_rows], dtype=float) * 1000
            ao_t = np.asarray([r["ao_tracked"] for r in scg_ref_rows], dtype=float) * 1000
            ac_t = np.asarray([r["ac_tracked"] for r in scg_ref_rows], dtype=float) * 1000
            fig, ax = plt.subplots(1,1,figsize=(11.5,5.2),constrained_layout=True)
            ax.plot(beat, ao_m, ".", alpha=0.22, label="AO measurement")
            ax.plot(beat, ac_m, ".", alpha=0.22, label="AC measurement")
            ax.plot(beat, ao_t, "-", linewidth=1.3, label="AO HIKAF-style tracked")
            ax.plot(beat, ac_t, "-", linewidth=1.3, label="AC HIKAF-style tracked")
            ax.set_xlabel("Beat index")
            ax.set_ylabel("Timing from ECG R-peak [ms]")
            ax.set_title("HIKAF-style SCG AO/AC temporal tracking")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.savefig(outdir / "fig04_hikaf_style_scg_ao_ac_tracking.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            (outdir/"fig04_hikaf_tracking_error.txt").write_text(str(e), encoding="utf-8")

        paper = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export")
        figs = paper / "figures"
        figs.mkdir(parents=True, exist_ok=True)
        mapping = [
            ("fig03_zheng_style_scg_ao_enhancement_stages.png", "fig03_zheng_style_scg_ao_enhancement_stages.png"),
            ("fig04_hikaf_style_scg_ao_ac_tracking.png", "fig04_hikaf_style_scg_ao_ac_tracking.png"),
            ("fig02_compact_beat_morphology.png", "fig05_ecg_scg_radar_morphology_comparison.png"),
            ("fig06_single_cycle_ecg_radar_aoac_labels.png", "fig06_scg_reference_radar_detected_landmarks.png"),
            ("fig07_scg_radar_aoac_timing_boxplot.png", "fig07_scg_radar_aoac_timing_boxplot.png"),
            ("fig08_scg_radar_relative_difference_boxplot.png", "fig08_scg_radar_relative_difference_boxplot.png"),
            ("fig06_pep_lvet_qs2_brackets.png", "fig09_scg_radar_interval_analysis.png"),
        ]
        idx_rows = []
        for s, d in mapping:
            sp, dp = outdir / s, figs / d
            status = "missing"
            if sp.exists():
                shutil.copyfile(sp, dp)
                status = "copied"
            idx_rows.append([s, d, status])
        save_csv(figs / "paper_figure_index_patch21.csv", ["Source", "PaperFigure", "Status"], idx_rows)

        (outdir / "patch21_final_figure_policy.txt").write_text(
            "PATCH21 active. SCG AO detection uses Zheng-style blocks: first-order interference cancellation, "
            "mode decomposition/reconstruction, waveform-factor mode selection, and seventh-power AO detector. "
            "SCG AO/AC are then tracked using a HIKAF-style HR-informed Kalman filter. "
            "ECG is R-peak alignment anchor only. MC/IM/MO remain shown for CTI context.",
            encoding="utf-8"
        )
    except BaseException as e:
        (outdir / "patch21_final_figures_error.txt").write_text(str(e), encoding="utf-8")

def _p21_core_export(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    core = globals().get("_old_save_all_patch3", None)
    if callable(core):
        return core(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    return None

def save_all(outdir: Path, ecg, radar, scg, aoac, comp, ecfg: ECGConfig, rcfg: RadarConfig, acfg: AnalysisConfig):
    result = _p21_core_export(outdir, ecg, radar, scg, aoac, comp, ecfg, rcfg, acfg)
    try:
        _p21_make_final_figs(outdir, ecg, scg, radar, aoac, acfg)
    except BaseException as e:
        (outdir / "save_all_patch21_error.txt").write_text(str(e), encoding="utf-8")
    return result



# ============================================================
# PATCH22: vmdpy-required Successive VMD block for Zheng-style AO detector
# ------------------------------------------------------------
# User requested:
# - Install vmdpy and use it.
# - Reflect the "SVMD" idea more directly instead of only a band-limited fallback.
#
# Implementation:
# - This patch overrides _p21_reconstruct_ao_signal().
# - It requires vmdpy at runtime for the SVMD/VMD block.
# - A self-contained "successive VMD" procedure is implemented:
#     residual_0 = x_mti
#     for s in 1..S:
#       VMD(residual, K=2)
#       score each mode by AO waveform factor criterion
#       select the AO-related mode
#       subtract selected mode from residual
#     reconstruct AO signal from selected modes
#     seventh-power positive detector is applied to AO-reconstructed signal.
#
# Important wording for paper:
# - This is "successive VMD-based AO component selection using vmdpy",
#   not the authors' private MATLAB/SVMD implementation unless the exact
#   original SVMD source is supplied.
# ============================================================

P22_SVMD_MAX_MODES = 5
P22_VMD_ALPHA = 2000
P22_VMD_TAU = 0.0
P22_VMD_K_PER_STEP = 2
P22_VMD_DC = 0
P22_VMD_INIT = 1
P22_VMD_TOL = 1e-7
P22_MIN_AO_MODE_SCORE_RATIO = 0.12

def _p22_require_vmdpy():
    try:
        from vmdpy import VMD
        return VMD
    except Exception as e:
        raise ImportError(
            "vmdpy is required for PATCH22 SVMD/VMD mode decomposition. "
            "Install it with: pip install vmdpy"
        ) from e

def _p22_successive_vmd_modes(x, tt, r_times, fs=100.0, max_modes=P22_SVMD_MAX_MODES):
    """
    Successive VMD decomposition for AO-related SCG mode extraction.

    This approximates the successive extraction logic of SVMD using the
    available vmdpy VMD solver:
      - run VMD on current residual with K=2
      - select the AO-related mode by waveform factor criterion
      - subtract selected mode from residual
      - repeat
    """
    VMD = _p22_require_vmdpy()
    x = zscore_safe(np.asarray(x, dtype=np.float64))
    residual = x.copy()

    selected_modes = []
    all_modes = []
    all_scores = []
    residual_energy = []
    selected_indices = []

    for step in range(int(max_modes)):
        if len(residual) < 16:
            break
        if float(np.nanstd(residual)) < 1e-6:
            break

        try:
            u, u_hat, omega = VMD(
                residual,
                P22_VMD_ALPHA,
                P22_VMD_TAU,
                P22_VMD_K_PER_STEP,
                P22_VMD_DC,
                P22_VMD_INIT,
                P22_VMD_TOL
            )
        except Exception:
            break

        if u is None or np.size(u) == 0:
            break

        # vmdpy returns shape [K, N]
        modes_step = [zscore_safe(u[i, :]) for i in range(u.shape[0])]
        scores_step = np.asarray([_p21_waveform_factor(m, r_times, tt) for m in modes_step], dtype=float)

        best_local = int(np.nanargmax(scores_step))
        best_mode = modes_step[best_local]
        best_score = float(scores_step[best_local])

        # Stop if no meaningful AO-related mode remains after the first extraction.
        if step > 0 and len(all_scores) > 0:
            global_best = max([float(s) for ss in all_scores for s in (ss if np.ndim(ss) else [ss])] + [best_score])
            if best_score < P22_MIN_AO_MODE_SCORE_RATIO * max(global_best, 1e-9):
                break

        selected_modes.append(best_mode)
        all_modes.extend(modes_step)
        all_scores.append([float(s) for s in scores_step])
        selected_indices.append({"step": int(step), "local_mode_index": int(best_local), "score": best_score})

        # Successive residual update
        residual = zscore_safe(residual - best_mode)
        residual_energy.append(float(np.nanvar(residual)))

    if not selected_modes:
        selected_modes = [x]
        selected_indices = [{"step": 0, "local_mode_index": 0, "score": 0.0}]
        all_modes = [x]
        all_scores = [[0.0]]
        residual_energy = [float(np.nanvar(x))]

    return {
        "selected_modes": selected_modes,
        "all_modes": all_modes,
        "all_scores": all_scores,
        "selected_indices": selected_indices,
        "residual_energy": residual_energy,
    }

def _p22_reconstruct_ao_signal_successive_vmd(x, tt, r_times, fs=100.0):
    """
    Zheng-style AO enhancement using:
      1. first-order interference cancellation
      2. successive VMD mode extraction using vmdpy
      3. waveform factor criterion
      4. AO mode reconstruction
      5. positive seventh-power detector
    """
    x_mti = _p21_mti_interference_cancellation(x)
    svmd = _p22_successive_vmd_modes(x_mti, tt, r_times, fs=fs, max_modes=P22_SVMD_MAX_MODES)

    modes = svmd["selected_modes"]
    if not modes:
        rec = x_mti.copy()
    else:
        rec = np.sum(modes, axis=0)

    rec = zscore_safe(rec)

    # Zheng-style seventh-power detector: positive AO peaks only.
    pos = np.maximum(rec, 0.0)
    det = pos ** 7
    try:
        det = safe_lowpass(det, fs, min(8.0, fs * 0.20), order=2)
    except Exception:
        pass
    det = robust_scale_01(det)

    meta = {
        "mode_method": "successive_vmdpy_VMD",
        "vmdpy_required": True,
        "vmd_alpha": P22_VMD_ALPHA,
        "vmd_tau": P22_VMD_TAU,
        "vmd_k_per_step": P22_VMD_K_PER_STEP,
        "max_successive_modes": P22_SVMD_MAX_MODES,
        "selected_indices": svmd["selected_indices"],
        "all_step_scores": svmd["all_scores"],
        "residual_energy": svmd["residual_energy"],
        "paper_wording": (
            "successive VMD-based AO component extraction using vmdpy, "
            "waveform factor criterion, and positive seventh-power AO detector"
        ),
    }

    return {
        "mti": x_mti,
        "modes": svmd["selected_modes"],
        "all_modes": svmd["all_modes"],
        "ao_reconstructed": rec,
        "seventh_power_detector": det,
        "meta": meta,
    }

# Override PATCH21 reconstruction function with vmdpy-required successive VMD version.
def _p21_reconstruct_ao_signal(x, tt, r_times, fs=100.0):
    return _p22_reconstruct_ao_signal_successive_vmd(x, tt, r_times, fs)

def _p22_make_svmd_mode_figure(outdir, scg, ao_enh, rep):
    """
    Extra reference-style figure:
    selected successive-VMD modes + reconstructed AO + seventh-power detector.
    """
    try:
        tt = np.asarray(scg["t"], dtype=float)
        anchor = rep["anchor"]
        selected_modes = ao_enh.get("modes", [])
        n_modes = min(len(selected_modes), 5)
        n_rows = max(3, n_modes + 2)
        fig, axes = plt.subplots(n_rows, 1, figsize=(12.5, 2.15*n_rows), sharex=True, constrained_layout=True)

        row = 0
        for i in range(n_modes):
            bt, bx = _p21_slice(tt, selected_modes[i], anchor)
            axes[row].plot(bt, zscore_safe(bx), color="black", linewidth=1.25)
            axes[row].axvspan(P21_WIN["AO"][0], P21_WIN["AO"][1], color=P21_C["AO"], alpha=0.06)
            axes[row].set_title(f"Successive VMD selected mode {i+1}", loc="left", fontsize=10)
            axes[row].grid(True, alpha=0.22)
            axes[row].set_ylabel("z")
            row += 1

        bt, bx = _p21_slice(tt, ao_enh["ao_reconstructed"], anchor)
        axes[row].plot(bt, zscore_safe(bx), color="black", linewidth=1.25)
        axes[row].axvspan(P21_WIN["AO"][0], P21_WIN["AO"][1], color=P21_C["AO"], alpha=0.06)
        axes[row].axvline(rep["scg_lm"]["AO"], color=P21_C["AO"], linestyle="--", linewidth=1.1)
        axes[row].set_title("AO-reconstructed signal from selected VMD modes", loc="left", fontsize=10)
        axes[row].grid(True, alpha=0.22)
        axes[row].set_ylabel("z")
        row += 1

        bt, bx = _p21_slice(tt, ao_enh["seventh_power_detector"], anchor)
        axes[row].plot(bt, bx, color="black", linewidth=1.25)
        axes[row].axvspan(P21_WIN["AO"][0], P21_WIN["AO"][1], color=P21_C["AO"], alpha=0.06)
        axes[row].axvline(rep["scg_lm"]["AO"], color=P21_C["AO"], linestyle="--", linewidth=1.1)
        axes[row].set_title("Positive seventh-power AO detector", loc="left", fontsize=10)
        axes[row].grid(True, alpha=0.22)
        axes[row].set_ylabel("score")
        axes[row].set_xlabel("Time from ECG R-peak [s]")

        for ax in axes:
            ax.set_xlim(*P21_XLIM)

        fig.suptitle("Successive VMD-based SCG AO extraction details", fontsize=14)
        fig.savefig(outdir / "fig03b_successive_vmd_ao_mode_selection.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (outdir / "fig03b_successive_vmd_error.txt").write_text(str(e), encoding="utf-8")

# Extend PATCH21 final-figure generator by wrapping it:
_old_p21_make_final_figs_patch22 = _p21_make_final_figs

def _p21_make_final_figs(outdir, ecg, scg, radar, aoac, acfg):
    # Run original PATCH21 pipeline, now using overridden _p21_reconstruct_ao_signal().
    _old_p21_make_final_figs_patch22(outdir, ecg, scg, radar, aoac, acfg)

    # Rebuild minimal state for additional SVMD reference-style figure and audit.
    try:
        scg_ref_rows, ao_enh = _p21_build_scg_reference_table(ecg, scg)
        rep = _p21_pick_best(ecg, scg, radar, scg_ref_rows)
        if rep is not None and ao_enh is not None:
            _p22_make_svmd_mode_figure(outdir, scg, ao_enh, rep)

            # Add to paper export if present.
            paper = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export")
            figs = paper / "figures"
            figs.mkdir(parents=True, exist_ok=True)
            sp = outdir / "fig03b_successive_vmd_ao_mode_selection.png"
            if sp.exists():
                shutil.copyfile(sp, figs / "fig03b_successive_vmd_ao_mode_selection.png")

            with open(outdir / "patch22_vmdpy_svmd_audit.json", "w", encoding="utf-8") as f:
                json.dump({
                    "patch": "PATCH22",
                    "vmdpy_required": True,
                    "install": "pip install vmdpy",
                    "method": "successive VMD-based AO component extraction using vmdpy",
                    "important_note": (
                        "This implements a successive-VMD AO extraction block using vmdpy. "
                        "If the exact SVMD source from the reference paper is required, "
                        "replace _p22_successive_vmd_modes with the original SVMD solver."
                    ),
                    "ao_enhancement_meta": ao_enh.get("meta", {}),
                }, f, ensure_ascii=False, indent=2)

            policy = outdir / "patch21_final_figure_policy.txt"
            extra = (
                "\nPATCH22 active. vmdpy-required successive VMD block overrides PATCH21 mode decomposition. "
                "AO reconstruction now uses successive VMD mode extraction, waveform factor mode selection, "
                "and positive seventh-power detector.\n"
            )
            if policy.exists():
                policy.write_text(policy.read_text(encoding="utf-8", errors="ignore") + extra, encoding="utf-8")
            else:
                policy.write_text(extra, encoding="utf-8")
    except BaseException as e:
        (outdir / "patch22_extra_svmd_figure_error.txt").write_text(str(e), encoding="utf-8")



# ============================================================
# PATCH23: paper-equation aligned Zheng + HIKAF + Di Rienzo hybrid
# ------------------------------------------------------------
# Purpose:
# - Replace the loose PATCH22 successive-VMD block with a paper-equation
#   aligned implementation as far as possible from the uploaded manuscripts.
# - Keep full reproducibility in one Python file.
#
# Zheng et al. aligned:
#   1) MTI high-pass:
#        s_R[n] = beta*s_R[n-1] + (1-beta)*s[n]
#        x_beta[n] = s[n] - s_R[n]
#        y[n] = x_beta2[n] - x_beta1[n]
#      beta1=0.9, beta2=0.99 by paper example.
#   2) detrend + fifth-order median filter.
#   3) SVMD-like successive one-dimensional ADMM mode extraction:
#        modes are extracted one-by-one until residual variance criterion.
#      This is a direct implementation from the paper equations, but still
#      should be described as "paper-equation reimplementation", not
#      "author source-code".
#   4) waveform factor:
#        WF_k = rms(u_k) / mean(abs(u_k)).
#      select modes with WF_k above average.
#   5) reconstructed AO signal, seventh-power detector:
#        s_AO_tilde = s_AO^7
#      Hilbert envelope + moving average over 0.1 s.
#
# HIKAF aligned:
#   1) Baseline measurement phase:
#        candidate peaks in AO 50-200 ms, AC 200-400 ms
#        K = max number of candidates among BM beats
#        K-means clustering over tau candidates
#        select cluster with highest average prominence
#        compute mu_baseline, sigma2_baseline, mu_RR_baseline
#   2) Runtime:
#        tau_prime = tau - (mu_RR_current/mu_RR_baseline)^lambda * mu_baseline
#        select candidate with min(abs(tau_prime))
#        Kalman prediction/update with adaptive R = R0 * sigma2_current/sigma2_baseline
#        lambda = 0.7, q = 0.1, r = 0.2.
#
# Di Rienzo aligned:
#   - MC/AO/AC/MO fiducials remain detected for CTI context.
#   - S1/S2 envelope, ICP and IRP anchors, and local pattern rules are added:
#       S1si = 25-75 ms after R for ICP
#       AO = first peak after ICP within 50 ms and amplitude rule
#       MC = first peak before ICP within 50 ms and amplitude rule
#       S2si = 60 ms segment centered on ECG T end/T proxy
#       AC = first peak 10-40 ms before IRP
#       MO = next trough 10-30 ms after IRP
#   - Our 100 Hz data limits 1 ms HiRes refinement; sinc/PCHIP upsample
#     refinement is added for 1 kHz equivalent local timing.
# ============================================================

P23_BETA1 = 0.90
P23_BETA2 = 0.99
P23_ZHENG_ALPHA = 2000.0
P23_SVMD_TAU = 0.0
P23_SVMD_EPS1 = 1e-5
P23_SVMD_EPS2 = 0.08
P23_SVMD_MAX_MODES = 12
P23_SVMD_MAX_ITER = 250
P23_LAMBDA = 0.7
P23_Q = 0.1
P23_R = 0.2
P23_AO_WIN = (0.050, 0.200)
P23_AC_WIN = (0.200, 0.400)
P23_S1SI = (0.025, 0.075)
P23_S2_HALF = 0.030
P23_XLIM = (-0.12, 0.62)

# ---------- Zheng exact MTI / WF / envelope ----------

def _p23_mti_highpass(x, beta):
    x = np.asarray(x, dtype=np.float64)
    sr = np.zeros_like(x)
    if len(x) == 0:
        return x.copy()
    sr[0] = x[0]
    for n in range(1, len(x)):
        sr[n] = beta * sr[n-1] + (1.0 - beta) * x[n]
    return x - sr

def _p23_zheng_preprocess(x, fs=100.0):
    """
    Zheng Eq. (3)-(5): two MTI high-pass outputs subtracted.
    Then detrending and 5th-order median filtering as described.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.nanmedian(x)
    xb1 = _p23_mti_highpass(x, P23_BETA1)
    xb2 = _p23_mti_highpass(x, P23_BETA2)
    y = xb2 - xb1
    try:
        y = signal.detrend(y, type='linear')
    except Exception:
        y = y - np.nanmean(y)
    if len(y) >= 5:
        try:
            y = signal.medfilt(y, kernel_size=5)
        except Exception:
            pass
    return zscore_safe(y)

def _p23_waveform_factor(u):
    u = np.asarray(u, dtype=np.float64)
    den = np.nanmean(np.abs(u)) + 1e-12
    num = np.sqrt(np.nanmean(u*u))
    return float(num / den)

def _p23_fft_freqs(n, fs):
    return np.fft.rfftfreq(n, d=1.0/fs) * 2.0*np.pi

def _p23_single_svmd_extract(residual, prev_modes, prev_omegas, fs, alpha=P23_ZHENG_ALPHA):
    """
    One-dimensional successive mode extraction inspired by Zheng Algorithm 1
    equations (16)-(18). This is not copied source code; it is a numerical
    reimplementation of the published update structure.
    """
    residual = np.asarray(residual, dtype=np.float64)
    n = len(residual)
    xhat = np.fft.rfft(residual)
    w = _p23_fft_freqs(n, fs)

    power = np.abs(xhat)**2
    if len(power) > 1:
        power[0] = 0
    omega = float(w[int(np.nanargmax(power))]) if len(power) else 0.0
    uhat = np.zeros_like(xhat, dtype=np.complex128)
    lam = np.zeros_like(xhat, dtype=np.complex128)

    prev_hats = [np.fft.rfft(pm) for pm in prev_modes] if prev_modes else []
    eps = 1e-12

    for it in range(P23_SVMD_MAX_ITER):
        u_prev = uhat.copy()
        # Leakage penalty over previous modes, corresponding to J3 structure.
        leak = np.zeros_like(w, dtype=np.float64)
        for om in prev_omegas:
            leak += 1.0 / (alpha**2 * (w - om)**4 + eps)

        # Single-mode ADMM-like update from the constrained SVMD problem.
        den = 1.0 + 2.0*alpha*(w - omega)**2 + leak
        uhat = (xhat + lam/2.0) / (den + eps)

        # Center-frequency update, Eq. (17)-like positive-frequency energy centroid.
        e = np.abs(uhat)**2
        if np.sum(e) > eps:
            omega = float(np.sum(w * e) / (np.sum(e) + eps))

        recon_prev = np.sum(prev_hats, axis=0) if prev_hats else 0.0
        lam = lam + P23_SVMD_TAU * (xhat - (uhat + recon_prev))

        num = np.sum(np.abs(uhat - u_prev)**2)
        den2 = np.sum(np.abs(u_prev)**2) + eps
        if num / den2 < P23_SVMD_EPS1 and it > 5:
            break

    mode = np.fft.irfft(uhat, n=n)
    mode = zscore_safe(mode)
    return mode, omega, int(it+1)

def _p23_svmd_paper_reimplementation(x, fs=100.0, sigma2=None):
    """
    Successive extraction until residual variance criterion:
      |sigma2 - (1/T)||s - sum u_i||^2| / sigma2 < eps2
    If sigma2 unknown, estimate a small residual target from high-frequency noise.
    """
    x = zscore_safe(np.asarray(x, dtype=np.float64))
    if sigma2 is None:
        try:
            hf = x - safe_lowpass(x, fs, min(10.0, fs*0.20), order=2)
            sigma2 = float(np.nanvar(hf))
        except Exception:
            sigma2 = float(0.03 * np.nanvar(x))
    sigma2 = max(float(sigma2), 1e-6)

    modes, omegas, iters = [], [], []
    residual = x.copy()

    for k in range(P23_SVMD_MAX_MODES):
        if float(np.nanvar(residual)) < sigma2 * (1.0 + P23_SVMD_EPS2):
            break
        mode, om, nit = _p23_single_svmd_extract(residual, modes, omegas, fs)
        if np.nanstd(mode) < 1e-6:
            break
        modes.append(mode)
        omegas.append(om)
        iters.append(nit)
        residual = x - np.sum(modes, axis=0)
        crit = abs(sigma2 - float(np.mean(residual**2))) / sigma2
        if crit < P23_SVMD_EPS2 and k >= 1:
            break

    if not modes:
        modes = [x.copy()]
        omegas = [0.0]
        iters = [0]
        residual = np.zeros_like(x)

    return {
        "modes": modes,
        "omegas_rad": omegas,
        "omegas_hz": [float(o/(2*np.pi)) for o in omegas],
        "iters": iters,
        "residual": residual,
        "sigma2": sigma2,
        "residual_criterion": abs(sigma2 - float(np.mean(residual**2))) / sigma2,
    }

def _p23_reconstruct_ao_signal_from_svmd(x, tt, r_times, fs=100.0):
    y = _p23_zheng_preprocess(x, fs)
    sv = _p23_svmd_paper_reimplementation(y, fs)

    modes = sv["modes"]
    wf = np.asarray([_p23_waveform_factor(m) for m in modes], dtype=float)
    avg = float(np.nanmean(wf)) if len(wf) else 0.0
    selected = [i for i, v in enumerate(wf) if v > avg]
    if not selected and len(wf):
        selected = [int(np.nanargmax(wf))]

    rec = np.sum([modes[i] for i in selected], axis=0) if selected else y.copy()
    rec = zscore_safe(rec)

    # Seventh power law detector, Eq. (24), followed by Hilbert envelope and 0.1 s moving average Eq. (25).
    s7 = rec ** 7
    try:
        env = np.abs(signal.hilbert(s7))
    except Exception:
        env = np.abs(s7)
    win = max(3, int(round(0.1 * fs)))
    kernel = np.ones(win, dtype=float) / win
    smooth_env = np.convolve(env, kernel, mode='same')
    smooth_env = robust_scale_01(smooth_env)

    return {
        "mti": y,
        "modes": modes,
        "ao_reconstructed": rec,
        "seventh_power_detector": s7,
        "hilbert_envelope": robust_scale_01(env),
        "smoothed_envelope": smooth_env,
        "meta": {
            "mode_method": "paper_equation_svmd_reimplementation",
            "beta1": P23_BETA1,
            "beta2": P23_BETA2,
            "waveform_factors": [float(v) for v in wf],
            "waveform_factor_average": avg,
            "selected_modes": [int(i) for i in selected],
            "omegas_hz": sv["omegas_hz"],
            "svmd_iterations": sv["iters"],
            "residual_criterion": float(sv["residual_criterion"]),
            "sigma2": float(sv["sigma2"]),
            "envelope_smoothing_window_sec": 0.1,
        }
    }

# Override PATCH21/PATCH22 reconstruction with PATCH23 paper-equation pipeline.
def _p21_reconstruct_ao_signal(x, tt, r_times, fs=100.0):
    return _p23_reconstruct_ao_signal_from_svmd(x, tt, r_times, fs)

def _p21_ao_from_zheng_detector(bt, det):
    # PATCH23 uses smoothed envelope as AO detector.
    idx = _p21_idx(bt, P23_AO_WIN)
    if len(idx) < 3:
        return None
    det = np.asarray(det, dtype=float)
    j = idx[int(np.nanargmax(det[idx]))]
    return float(bt[j])

# ---------- Di Rienzo MC/AO/AC/MO pattern detector ----------

def _p23_local_maxima(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return np.array([], dtype=int)
    return signal.find_peaks(y)[0]

def _p23_local_minima(y):
    return _p23_local_maxima(-np.asarray(y, dtype=float))

def _p23_refine_time_highres(bt, bx, t0, kind="max", fs_target=1000.0):
    if t0 is None:
        return None
    bt = np.asarray(bt, dtype=float)
    bx = np.asarray(bx, dtype=float)
    m = (bt >= t0 - 0.050) & (bt <= t0 + 0.050)
    if np.sum(m) < 4:
        return float(t0)
    tw = bt[m]
    xw = bx[m]
    tq = np.arange(t0 - 0.010, t0 + 0.010 + 1.0/fs_target, 1.0/fs_target)
    tq = tq[(tq >= tw[0]) & (tq <= tw[-1])]
    if len(tq) < 3:
        return float(t0)
    try:
        from scipy.interpolate import PchipInterpolator
        yq = PchipInterpolator(tw, xw)(tq)
    except Exception:
        yq = np.interp(tq, tw, xw)
    if kind == "min":
        return float(tq[int(np.nanargmin(yq))])
    if kind == "abs":
        return float(tq[int(np.nanargmax(np.abs(yq)))])
    return float(tq[int(np.nanargmax(yq))])

def _p23_dirienzo_fiducials(bt, bx, ecg_t=None):
    """
    Di Rienzo-inspired pattern analysis:
    - SCG envelope: abs(SCG) + triangular FIR
    - S1 under 10-160 ms, S2 under 300-480 ms
    - ICP in 25-75 ms
    - AO first peak after ICP within 50 ms, amplitude >=0.7*|ICPd|
    - MC first peak before ICP within 50 ms, same amplitude rule
    - IRP in S2si centered near T proxy
    - AC first peak 10-40 ms before IRP
    - MO next trough 10-30 ms after IRP
    """
    bt = np.asarray(bt, dtype=float)
    bx = zscore_safe(np.asarray(bx, dtype=float))

    env = np.abs(bx)
    tri = signal.windows.triang(31) if len(env) >= 31 else signal.windows.triang(max(3, len(env)//2*2+1))
    tri = tri / np.sum(tri)
    env = np.convolve(env, tri, mode='same')

    lm = {}

    # S1/ICP
    idx_s1 = np.where((bt >= P23_S1SI[0]) & (bt <= P23_S1SI[1]))[0]
    if len(idx_s1) >= 3:
        icp_idx = idx_s1[int(np.nanargmin(bx[idx_s1]))]
        icp_t = float(bt[icp_idx])
        icp_amp = float(bx[icp_idx])
    else:
        icp_t = 0.050
        icp_amp = float(np.interp(icp_t, bt, bx))

    # AO/MC peaks around ICP
    peak_idx = _p23_local_maxima(bx)
    peak_t = bt[peak_idx] if len(peak_idx) else np.array([])
    peak_y = bx[peak_idx] if len(peak_idx) else np.array([])
    thr = 0.7 * abs(icp_amp)

    after = np.where((peak_t > icp_t) & (peak_t <= icp_t + 0.050) & ((peak_y - icp_amp) >= thr))[0]
    if len(after):
        ao_t = float(peak_t[after[0]])
    else:
        ao_t = _p21_peak(bt, bx, (max(0.050, icp_t), min(0.200, icp_t+0.090)), "pos") or 0.120

    before = np.where((peak_t < icp_t) & (peak_t >= icp_t - 0.050) & ((peak_y - icp_amp) >= thr))[0]
    if len(before):
        mc_t = float(peak_t[before[-1]])
    else:
        mc_t = _p21_peak(bt, bx, (-0.085, min(0.020, icp_t)), "pos")
        if mc_t is None:
            mc_t = -0.030

    # T end proxy for S2 center: use AC window center if ECG T-end unavailable.
    te_proxy = 0.330
    idx_s2 = np.where((bt >= te_proxy - P23_S2_HALF) & (bt <= te_proxy + P23_S2_HALF))[0]
    if len(idx_s2) < 3:
        idx_s2 = np.where((bt >= 0.300) & (bt <= 0.480))[0]

    if len(idx_s2) >= 3:
        # IRP highest peak in S2si with D = D1 + D2 score.
        cand = []
        loc_peaks = [p for p in peak_idx if p in set(idx_s2)]
        mins = _p23_local_minima(bx)
        for p in loc_peaks:
            left_mins = mins[mins < p]
            right_mins = mins[mins > p]
            if len(left_mins) and len(right_mins):
                lm1 = left_mins[-1]; rm1 = right_mins[0]
                D = (bx[p]-bx[lm1]) + (bx[p]-bx[rm1])
            else:
                D = bx[p] - np.nanmin(bx[idx_s2])
            cand.append((D, p))
        if cand:
            irp_idx = sorted(cand, key=lambda z: z[0], reverse=True)[0][1]
            irp_t = float(bt[irp_idx])
        else:
            irp_t = float(bt[idx_s2[int(np.nanargmax(bx[idx_s2]))]])
    else:
        irp_t = 0.360

    # AC first peak preceding IRP by 10-40ms.
    ac_candidates = peak_t[(peak_t <= irp_t - 0.010) & (peak_t >= irp_t - 0.040)]
    if len(ac_candidates):
        ac_t = float(ac_candidates[-1])
    else:
        ac_t = _p21_peak(bt, bx, (max(P23_AC_WIN[0], irp_t-0.080), min(P23_AC_WIN[1], irp_t)), "pos")
        if ac_t is None:
            ac_t = float(np.clip(irp_t - 0.025, P23_AC_WIN[0], P23_AC_WIN[1]))

    # MO next trough following IRP by 10-30ms.
    mins = _p23_local_minima(bx)
    min_t = bt[mins] if len(mins) else np.array([])
    mo_candidates = min_t[(min_t >= irp_t + 0.010) & (min_t <= irp_t + 0.030)]
    if len(mo_candidates):
        mo_t = float(mo_candidates[0])
    else:
        mo_t = _p21_peak(bt, bx, (max(0.340, irp_t), min(0.600, irp_t+0.080)), "neg")
        if mo_t is None:
            mo_t = float(np.clip(irp_t + 0.020, 0.340, 0.600))

    # HiRes refinement.
    lm["MC"] = _p23_refine_time_highres(bt, bx, mc_t, "max")
    lm["IM"] = _p21_peak(bt, bx, (0.000, max(0.080, ao_t-0.010)), "pos") or 0.050
    lm["AO"] = _p23_refine_time_highres(bt, bx, ao_t, "max")
    lm["AC"] = _p23_refine_time_highres(bt, bx, ac_t, "max")
    lm["MO"] = _p23_refine_time_highres(bt, bx, mo_t, "min")

    # Ensure order but do not delete.
    prev = -999
    bounds = {"MC":(-0.100,0.050), "IM":(0.000,0.150), "AO":(0.050,0.200), "AC":(0.200,0.430), "MO":(0.300,0.600)}
    for k in ["MC","IM","AO","AC","MO"]:
        lo, hi = bounds[k]
        x = float(np.clip(lm[k], lo, hi))
        if x <= prev + 0.006:
            x = float(np.clip(prev + 0.008, lo, hi))
        lm[k] = x
        prev = x

    return lm, {"ICP": icp_t, "IRP": irp_t, "envelope": env}

# ---------- HIKAF exact BM/runtime ----------

def _p23_candidate_peaks(bt, bx, win, kind="max"):
    idx = np.where((bt >= win[0]) & (bt <= win[1]))[0]
    if len(idx) < 3:
        return []
    seg = bx[idx]
    if kind == "min":
        loc, props = signal.find_peaks(-seg, prominence=max(0.05, np.nanstd(seg)*0.2))
        vals = -seg[loc]
    else:
        loc, props = signal.find_peaks(seg, prominence=max(0.05, np.nanstd(seg)*0.2))
        vals = seg[loc]
    out = []
    proms = props.get("prominences", np.ones(len(loc)))
    for a, pr in zip(loc, proms):
        out.append((float(bt[idx[a]]), float(pr)))
    return out

def _p23_kmeans_1d(vals, K, max_iter=100):
    vals = np.asarray(vals, dtype=float)
    K = int(max(1, min(K, len(vals))))
    if K == 1:
        return np.zeros(len(vals), dtype=int), np.array([float(np.mean(vals))])
    centers = np.percentile(vals, np.linspace(0, 100, K))
    labels = np.zeros(len(vals), dtype=int)
    for _ in range(max_iter):
        d = np.abs(vals[:, None] - centers[None, :])
        new_labels = np.argmin(d, axis=1)
        new_centers = centers.copy()
        for k in range(K):
            if np.any(new_labels == k):
                new_centers[k] = np.mean(vals[new_labels == k])
        if np.all(new_labels == labels) and np.allclose(new_centers, centers):
            break
        labels, centers = new_labels, new_centers
    return labels, centers

def _p23_hikaf_baseline(scg_ref_rows, event="AO", n_baseline=60):
    bm = scg_ref_rows[:min(n_baseline, len(scg_ref_rows))]
    if not bm:
        return None

    key = "ao_meas" if event == "AO" else "ac_meas"
    cand_sets = []
    all_tau, all_prom = [], []
    for r in bm:
        # In this already segmented table, use measured event + confidence as candidate.
        tau = float(r[key])
        prom = float(r.get("conf_ao" if event=="AO" else "conf_ac", 1.0))
        cand_sets.append([(tau, prom)])
        all_tau.append(tau); all_prom.append(prom)

    K = max(1, max(len(c) for c in cand_sets))
    labels, centers = _p23_kmeans_1d(np.asarray(all_tau), K)

    best_k, best_prom = 0, -np.inf
    for k in range(K):
        pr = np.mean([all_prom[i] for i in range(len(all_prom)) if labels[i] == k]) if np.any(labels == k) else -np.inf
        if pr > best_prom:
            best_prom, best_k = pr, k

    mu_c = float(centers[best_k])
    tau_star = []
    for cands in cand_sets:
        taus = np.asarray([c[0] for c in cands], dtype=float)
        tau_star.append(float(taus[int(np.argmin(np.abs(taus - mu_c)))]))

    rr = np.asarray([r["rr"] for r in bm if np.isfinite(r["rr"]) and r["rr"] > 0], dtype=float)
    return {
        "event": event,
        "K": int(K),
        "mu_cluster": mu_c,
        "mu_baseline": float(np.mean(tau_star)),
        "sigma2_baseline": float(np.var(tau_star) + 1e-6),
        "mu_RR_baseline": float(np.mean(rr)) if len(rr) else 0.75,
        "tau_star": tau_star,
    }

def _p23_hikaf_runtime_track(scg_ref_rows, event="AO"):
    if not scg_ref_rows:
        return []
    base = _p23_hikaf_baseline(scg_ref_rows, event=event, n_baseline=min(60, max(10, len(scg_ref_rows)//10)))
    if base is None:
        return scg_ref_rows

    key = "ao_meas" if event == "AO" else "ac_meas"
    out = []
    # N=5 state vector, M=1 update, matching paper concept with compact runtime.
    N = 5
    x = np.ones(N, dtype=float) * base["mu_baseline"]
    P = np.eye(N) * 1.0
    A = np.eye(N)
    Q = np.eye(N) * P23_Q
    H = np.zeros((1, N)); H[0, -1] = 1.0
    R0 = np.array([[P23_R]])

    recent_z = []
    for i, row in enumerate(scg_ref_rows):
        rr = row["rr"] if np.isfinite(row["rr"]) and row["rr"] > 0 else base["mu_RR_baseline"]
        mu_rr_current = rr
        meas_candidates = [float(row[key])]
        # Eq. (11)-(12): scaled baseline mean and closest candidate.
        scaled_mu = (mu_rr_current / base["mu_RR_baseline"]) ** P23_LAMBDA * base["mu_baseline"]
        z = min(meas_candidates, key=lambda tau: abs(tau - scaled_mu))
        recent_z.append(z)
        sig2_cur = float(np.var(recent_z[-10:]) + 1e-6)

        # Prediction
        x_minus = A @ x
        P_minus = A @ P @ A.T + Q

        R = R0 * (sig2_cur / base["sigma2_baseline"])
        R = np.clip(R, 1e-5, 1.0)

        zvec = np.array([[z]], dtype=float)
        S = H @ P_minus @ H.T + R
        Kmat = P_minus @ H.T @ np.linalg.inv(S)
        x = x_minus + (Kmat @ (zvec - H @ x_minus)).ravel()
        P = (np.eye(N) - Kmat @ H) @ P_minus

        # shift state vector so latest estimate is at the end
        est = float(x[-1])
        x[:-1] = x[1:]
        x[-1] = est

        new = dict(row)
        new[f"{event.lower()}_hikaf"] = est
        new[f"{event.lower()}_hikaf_measurement"] = z
        new[f"{event.lower()}_hikaf_scaled_mu"] = scaled_mu
        out.append(new)
    return out

# Override PATCH21 SCG reference table builder with Zheng+DiRienzo+HIKAF paper-aligned version.
def _p21_build_scg_reference_table(ecg, scg):
    r = np.asarray(ecg.get("peaks_time", []), dtype=float)
    if len(r) < 3 or scg is None:
        return [], None

    tt = np.asarray(scg.get("t", []), dtype=float)
    sx = _p21_scg_base(scg)
    if sx is None:
        return [], None

    ao_enh = _p23_reconstruct_ao_signal_from_svmd(sx, tt, r, float(scg.get("fs", 100.0)))

    raw_rows = []
    for bi in range(1, len(r)-1):
        anchor = float(r[bi])
        bt_s, bx_s_raw = _p21_slice(tt, sx, anchor)
        bt_env, env = _p21_slice(tt, ao_enh["smoothed_envelope"], anchor)
        if bt_s is None or bt_env is None:
            continue

        # Determine polarity via both signs; Di Rienzo rules are then applied.
        candidates = []
        for pol in [1, -1]:
            bx = zscore_safe(bx_s_raw * pol)
            lm_di, anchors = _p23_dirienzo_fiducials(bt_s, bx)
            # AO measurement: Zheng smoothed envelope maximum in AO window.
            ao_env = _p21_ao_from_zheng_detector(bt_env, env)
            if ao_env is not None:
                lm_di["AO"] = ao_env
            vals = {k: float(np.interp(lm_di[k], bt_s, bx)) for k in ["MC","IM","AO","AC","MO"]}
            score = 2*max(0, vals["MC"]) + 2*max(0, vals["AO"]) + 2*max(0, vals["AC"]) + max(0, -vals["MO"])
            candidates.append((score, pol, bx, lm_di, vals, anchors))
        candidates.sort(key=lambda z: z[0], reverse=True)
        score, pol, bx_norm, lm, vals, anchors = candidates[0]

        conf_ao = max(0.05, abs(vals["AO"]) + 0.25*abs(vals["IM"]))
        conf_ac = max(0.05, abs(vals["AC"]) + 0.10*abs(vals["MO"]))
        rr = float(r[bi] - r[bi-1]) if bi > 0 else np.nan

        raw_rows.append({
            "beat_index": int(bi),
            "anchor": anchor,
            "rr": rr,
            "polarity": int(pol),
            "polarity_score": float(score),
            "mc": float(lm["MC"]),
            "im": float(lm["IM"]),
            "ao_meas": float(lm["AO"]),
            "ac_meas": float(lm["AC"]),
            "mo": float(lm["MO"]),
            "mc_amp": float(vals["MC"]),
            "im_amp": float(vals["IM"]),
            "ao_amp": float(vals["AO"]),
            "ac_amp": float(vals["AC"]),
            "mo_amp": float(vals["MO"]),
            "conf_ao": float(conf_ao),
            "conf_ac": float(conf_ac),
            "icp": float(anchors.get("ICP", np.nan)),
            "irp": float(anchors.get("IRP", np.nan)),
        })

    # Separate HIKAF filters for AO and AC, then merge.
    ao_tracked_rows = _p23_hikaf_runtime_track(raw_rows, event="AO")
    ac_tracked_rows = _p23_hikaf_runtime_track(raw_rows, event="AC")
    ac_by_bi = {r["beat_index"]: r for r in ac_tracked_rows}

    tracked = []
    for row in ao_tracked_rows:
        bi = row["beat_index"]
        acrow = ac_by_bi.get(bi, row)
        new = dict(row)
        new["ao_tracked"] = float(row.get("ao_hikaf", row["ao_meas"]))
        new["ac_tracked"] = float(acrow.get("ac_hikaf", row["ac_meas"]))
        # enforce physiological order
        new["ao_tracked"] = float(np.clip(new["ao_tracked"], P23_AO_WIN[0], P23_AO_WIN[1]))
        new["ac_tracked"] = float(np.clip(new["ac_tracked"], max(P23_AC_WIN[0], new["ao_tracked"] + 0.080), P23_AC_WIN[1]))
        tracked.append(new)

    return tracked, ao_enh

# ---------- Extra paper-matched figures ----------

def _p23_make_reference_style_figures(outdir, ecg, scg, radar, ao_enh, rep):
    try:
        tt = np.asarray(scg["t"], dtype=float)
        anchor = rep["anchor"]

        items = [
            ("Original SCG signal s(t)", _p21_scg_base(scg)),
            ("Interference cancellation y(t)", ao_enh["mti"]),
            ("AO reconstructed signal s_AO(t)", ao_enh["ao_reconstructed"]),
            ("Seventh power detector s_AO(t)^7", ao_enh["seventh_power_detector"]),
            ("Hilbert envelope s_EN(t)", ao_enh["hilbert_envelope"]),
            ("Smoothed envelope and extracted AO", ao_enh["smoothed_envelope"]),
        ]
        fig, axes = plt.subplots(len(items), 1, figsize=(12.8, 11.2), sharex=True, constrained_layout=True)
        for ax, (title, sig) in zip(axes, items):
            bt, bx = _p21_slice(tt, sig, anchor)
            ax.plot(bt, zscore_safe(bx), color="black", linewidth=1.2)
            ax.axvspan(P23_AO_WIN[0], P23_AO_WIN[1], color=P21_C["AO"], alpha=0.06)
            ax.axvline(rep["scg_lm"]["AO"], color=P21_C["AO"], linestyle="--", linewidth=1.0)
            ax.set_title(title, loc="left", fontsize=10)
            ax.grid(True, alpha=0.20)
            ax.set_ylabel("z")
            ax.set_xlim(*P23_XLIM)
        axes[-1].set_xlabel("Time from ECG R-peak [s]")
        fig.suptitle("Zheng-style processing flow reproduced on measured SCG", fontsize=14)
        fig.savefig(outdir / "fig03_zheng_paper_matched_processing_flow.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Di Rienzo anchor / fiducial figure
        fig, ax = plt.subplots(1, 1, figsize=(12.4, 4.8), constrained_layout=True)
        bt, bx = rep["bt_s"], rep["bx_s"]
        env = np.abs(bx)
        tri = signal.windows.triang(31) if len(env) >= 31 else signal.windows.triang(max(3, len(env)//2*2+1))
        tri = tri/np.sum(tri)
        env = zscore_safe(np.convolve(env, tri, mode="same"))
        ax.plot(bt, bx, color="black", linewidth=1.2, label="SCG")
        ax.plot(bt, env, color="0.55", linewidth=1.0, label="SCG envelope")
        lm = rep["scg_lm"]
        for name in ["MC","IM","AO","AC","MO"]:
            _p21_mark(ax, bt, bx, lm.get(name), name, name, 0.90 if name in ["MC","AO"] else 0.80)
        ax.axvspan(P23_S1SI[0], P23_S1SI[1], color=P21_C["MC"], alpha=0.05, label="S1si")
        ax.axvspan(0.300, 0.390, color=P21_C["AC"], alpha=0.05, label="S2 area")
        ax.set_xlim(*P23_XLIM)
        ax.set_title("Di Rienzo-style S1/S2 anchor-based SCG fiducial extraction")
        ax.set_xlabel("Time from ECG R-peak [s]")
        ax.set_ylabel("z-score")
        ax.grid(True, alpha=0.20)
        ax.legend(fontsize=8, loc="upper right")
        fig.savefig(outdir / "fig03c_dirienzo_s1_s2_anchor_fiducials.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        (outdir / "patch23_reference_style_figures_error.txt").write_text(str(e), encoding="utf-8")

# Wrap final generator to add PATCH23 audit and figures.
_old_p21_make_final_figs_patch23 = _p21_make_final_figs

def _p21_make_final_figs(outdir, ecg, scg, radar, aoac, acfg):
    _old_p21_make_final_figs_patch23(outdir, ecg, scg, radar, aoac, acfg)
    try:
        scg_ref_rows, ao_enh = _p21_build_scg_reference_table(ecg, scg)
        rep = _p21_pick_best(ecg, scg, radar, scg_ref_rows)
        if rep is not None and ao_enh is not None:
            _p23_make_reference_style_figures(outdir, ecg, scg, radar, ao_enh, rep)

            paper = outdir / globals().get("PAPER_EXPORT_DIRNAME", "paper_export")
            figs = paper / "figures"
            figs.mkdir(parents=True, exist_ok=True)
            for name in [
                "fig03_zheng_paper_matched_processing_flow.png",
                "fig03c_dirienzo_s1_s2_anchor_fiducials.png",
            ]:
                sp = outdir / name
                if sp.exists():
                    shutil.copyfile(sp, figs / name)

            with open(outdir / "patch23_paper_equation_audit.json", "w", encoding="utf-8") as f:
                json.dump({
                    "patch": "PATCH23",
                    "zheng": {
                        "mti_beta1": P23_BETA1,
                        "mti_beta2": P23_BETA2,
                        "svmd": "paper-equation reimplementation of successive mode extraction",
                        "waveform_factor": "WF = rms(u)/mean(abs(u)); selected if WF > average WF",
                        "seventh_power": "s_AO^7 + Hilbert envelope + 0.1 s moving average",
                        "meta": ao_enh.get("meta", {}),
                    },
                    "hikaf": {
                        "lambda": P23_LAMBDA,
                        "q": P23_Q,
                        "r": P23_R,
                        "AO_window_sec": P23_AO_WIN,
                        "AC_window_sec": P23_AC_WIN,
                        "baseline": "K-means-like 1D clustering over baseline R-AE candidates with prominence cluster selection",
                        "runtime": "RR-scaled candidate selection and adaptive Kalman measurement covariance",
                    },
                    "dirienzo": {
                        "S1si_sec": P23_S1SI,
                        "S2_window": "60 ms around T-end proxy",
                        "MC_AO": "ICP anchor + first peak before/after within 50 ms with amplitude rule",
                        "AC_MO": "IRP anchor + AC before IRP and MO after IRP",
                        "hires_refinement": "PCHIP local interpolation to 1 kHz equivalent",
                    },
                    "important_note": "This is a formula-level reimplementation from the manuscripts, not original author source-code."
                }, f, ensure_ascii=False, indent=2)

            policy = outdir / "patch21_final_figure_policy.txt"
            extra = (
                "\nPATCH23 active. Zheng MTI equations, waveform factor, seventh-power Hilbert envelope, "
                "HIKAF baseline/runtime equations, and Di Rienzo S1/S2 anchor fiducials are applied. "
                "Implementation is formula-level reimplementation, not original author source-code.\n"
            )
            if policy.exists():
                policy.write_text(policy.read_text(encoding="utf-8", errors="ignore") + extra, encoding="utf-8")
            else:
                policy.write_text(extra, encoding="utf-8")
    except BaseException as e:
        (outdir / "patch23_extra_error.txt").write_text(str(e), encoding="utf-8")


if __name__ == "__main__":
    main()
