from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_simulation_evaluation.edf_replay import (
    load_fpz_cz_100hz,
    qa_fpz_cz_signal,
    run_adapted_stream,
)
from causal_phase_tracking.causal_simulation_evaluation.rate_adapter import (
    RationalRateAdapterConfig,
)


PSG = ROOT / "data" / "sleep_edf" / "default" / "SC4001E0-PSG.edf"


def adapter_config() -> RationalRateAdapterConfig:
    return RationalRateAdapterConfig(
        input_rate_hz=100.0,
        output_rate_hz=250.0,
        upsample=5,
        downsample=2,
        kernel_rate_hz=500.0,
        cutoff_hz=40.0,
        stopband_hz=50.0,
        kernel_taps=33,
        max_input_samples=4096,
        config_version="replay-test",
    )


def test_edf_loader_returns_fpz_cz_at_100_hz() -> None:
    if not PSG.is_file():
        pytest.skip("Sleep-EDF file is not present")
    signal = load_fpz_cz_100hz(PSG, start_s=0.0, stop_s=10.0)
    assert signal.size == 1000
    assert signal.dtype == np.float64
    assert np.isfinite(signal).all()


def test_qa_flags_flat_runs_nonfinite_and_clipping() -> None:
    rng = np.random.default_rng(9)
    signal = rng.standard_normal(2000) * 10.0
    summary = qa_fpz_cz_signal(signal, sampling_rate_hz=100.0)
    assert summary["longest_flat_run_s"] == pytest.approx(0.01, abs=0.02)
    assert summary["nonfinite_count"] == 0
    assert summary["rail_count"] == 0

    corrupted = signal.copy()
    corrupted[500:700] = 0.0
    corrupted[800] = np.nan
    summary = qa_fpz_cz_signal(corrupted, sampling_rate_hz=100.0)
    assert summary["nonfinite_count"] == 1
    assert summary["longest_flat_run_s"] == pytest.approx(2.0, abs=0.02)


def test_adapted_stream_is_chunk_invariant_and_finite() -> None:
    rng = np.random.default_rng(42)
    samples = rng.standard_normal(400)
    whole = run_adapted_stream(samples, adapter_config(), [400])
    split = run_adapted_stream(samples, adapter_config(), [1, 7, 25, 100, 17, 50, 200])
    np.testing.assert_allclose(split.values, whole.values, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(split.output_index, whole.output_index)
    assert np.isfinite(whole.values).all()
