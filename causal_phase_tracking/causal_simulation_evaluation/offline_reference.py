"""Offline-only ground-truth reference helpers for the 0.5-1.25 Hz SO band.

These functions are deliberately non-causal.  They must never be imported into
the online estimator, trigger, adapter, or host wrapper.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt


SO_BAND = (0.5, 1.25)


def noncausal_reference_resample(
    signal: np.ndarray,
    *,
    input_rate_hz: float = 100.0,
    output_rate_hz: float = 250.0,
) -> np.ndarray:
    """High-quality offline rational resampling, independent of the causal adapter."""

    if not (math.isclose(input_rate_hz, 100.0) and math.isclose(output_rate_hz, 250.0)):
        raise ValueError("reference resample supports only 100 -> 250 Hz")
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("reference resample expects a 1-D signal")
    return resample_poly(x, up=5, down=2).astype(np.float64)


def zero_phase_so_bandpass(
    signal: np.ndarray,
    sfreq: float,
    *,
    l_freq: float = SO_BAND[0],
    h_freq: float = SO_BAND[1],
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass in the SO band, offline reference only."""

    if not (0.0 < l_freq < h_freq < sfreq / 2.0):
        raise ValueError("invalid SO band for the given sampling rate")
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("reference bandpass expects a 1-D signal")
    sos = butter(order, [l_freq, h_freq], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x).astype(np.float64)


def hilbert_phase_rad(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Hilbert phase expects a 1-D signal")
    return np.angle(hilbert(x)).astype(np.float64)


def circular_mean_bias(angles: np.ndarray) -> float:
    values = np.asarray(angles, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("circular bias requires non-empty finite angles")
    return float(np.angle(np.mean(np.exp(1j * values))))


def circular_sd_from_resultant(resultant_length: Real) -> float:
    r = float(resultant_length)
    if not (0.0 < r <= 1.0):
        raise ValueError("resultant length must be in (0, 1]")
    return math.sqrt(-2.0 * math.log(r))
