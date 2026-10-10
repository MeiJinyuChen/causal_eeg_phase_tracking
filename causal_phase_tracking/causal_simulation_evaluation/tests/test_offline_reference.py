from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.signal import hilbert


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_simulation_evaluation.offline_reference import (
    circular_mean_bias,
    circular_sd_from_resultant,
    hilbert_phase_rad,
    noncausal_reference_resample,
    zero_phase_so_bandpass,
)


def test_non_causal_reference_resample_has_exact_rational_length() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal(200)
    y = noncausal_reference_resample(x, input_rate_hz=100.0, output_rate_hz=250.0)
    assert y.size == 500
    assert np.isfinite(y).all()


def test_zero_phase_bandpass_tracks_slow_oscillation_phase() -> None:
    fs = 250.0
    t = np.arange(9000, dtype=np.float64) / fs
    x = np.sin(2.0 * np.pi * 0.8 * t)
    filtered = zero_phase_so_bandpass(x, fs, l_freq=0.5, h_freq=1.25)
    phase = hilbert_phase_rad(filtered)
    expected = np.unwrap(np.angle(hilbert(x)))
    # Discard generous filter transients on both edges.
    interior = slice(2000, 7000)
    error = np.angle(np.exp(1j * (phase[interior] - expected[interior])))
    assert abs(float(np.mean(error))) < 0.05


def test_full_channel_filter_then_slice_matches_guard_extract() -> None:
    fs = 250.0
    t = np.arange(40000, dtype=np.float64) / fs
    x = np.sin(2.0 * np.pi * 0.9 * t) + 0.3 * np.sin(2.0 * np.pi * 1.1 * t)
    full = zero_phase_so_bandpass(x, fs, l_freq=0.5, h_freq=1.25)
    full_phase = hilbert_phase_rad(full)

    start_s = 60.0
    end_s = 90.0
    start_idx = int(start_s * fs)
    end_idx = int(end_s * fs)
    for guard_s in (30.0, 60.0):
        left = max(0, start_idx - int(guard_s * fs))
        right = min(x.size, end_idx + int(guard_s * fs))
        guarded = zero_phase_so_bandpass(x[left:right], fs, l_freq=0.5, h_freq=1.25)
        guarded_phase = hilbert_phase_rad(guarded)[start_idx - left : end_idx - left]
        error = np.angle(np.exp(1j * (guarded_phase - full_phase[start_idx:end_idx])))
        assert abs(float(np.mean(error))) < 0.02


def test_circular_statistics_use_wrapped_angles() -> None:
    bias = circular_mean_bias(np.array([-1.0, 1.0]))
    assert abs(bias) < 1e-9
    r = float(np.abs(np.mean(np.exp(1j * np.array([-3.0, 3.0])))))
    assert circular_sd_from_resultant(r) == pytest.approx(
        math.sqrt(-2.0 * math.log(r))
    )
