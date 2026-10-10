from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_cecHT.pseudo_trigger import (
    PseudoTriggerConfig,
)
from causal_phase_tracking.causal_cecHT.streaming_core import CecHTCoreConfig
from causal_phase_tracking.causal_simulation_evaluation.rate_adapter import (
    RationalRateAdapterConfig,
)
from causal_phase_tracking.causal_simulation_evaluation.edf_replay import run_adapted_stream
from causal_phase_tracking.causal_simulation_evaluation.evaluation_pipeline import (
    run_causal_pipeline,
)


def core_config() -> CecHTCoreConfig:
    return CecHTCoreConfig(
        sampling_rate_hz=250.0,
        input_dtype=np.dtype(np.float64),
        numeric_unit_code=1,
        channel_code=7,
        polarity_code=1,
        window_samples=150,
        l_freq=0.5,
        h_freq=1.25,
        filter_order=2,
        filter_type="butter",
        n_fft=150,
        fft_mode="exact",
        calibrate=False,
        f0=None,
        parameter_version="evaluation-pipeline-test",
        config_version="evaluation-pipeline-v1",
        max_block_samples=1024,
    )


def trigger_config() -> PseudoTriggerConfig:
    return PseudoTriggerConfig(
        sampling_rate_hz=250.0,
        target_phase_rad=0.0,
        refractory_s=0.5,
        parameter_version="evaluation-pipeline-test",
        config_version="evaluation-pipeline-v1",
        max_block_samples=1024,
    )


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
        max_input_samples=8192,
        config_version="evaluation-pipeline-test",
    )


def test_full_pipeline_is_chunk_invariant_and_emits_events() -> None:
    fs = 100.0
    t = np.arange(2400, dtype=np.float64) / fs
    samples = 80.0 * np.sin(2.0 * np.pi * 0.8 * t)
    stream = run_adapted_stream(samples, adapter_config(), [2400])
    whole = run_causal_pipeline(
        stream.values,
        stream.valid,
        stream.output_index,
        core_config(),
        trigger_config(),
        [1024],
    )

    split_stream = run_adapted_stream(samples, adapter_config(), [1, 7, 25, 100, 17, 50, 200])
    split = run_causal_pipeline(
        split_stream.values,
        split_stream.valid,
        split_stream.output_index,
        core_config(),
        trigger_config(),
        [10, 40, 100, 17, 250, 3],
    )

    np.testing.assert_allclose(split.phase_rad, whole.phase_rad, rtol=0.0, atol=2e-12)
    np.testing.assert_array_equal(split.valid, whole.valid)
    np.testing.assert_array_equal(split.ready, whole.ready)
    np.testing.assert_array_equal(split.event, whole.event)
    np.testing.assert_allclose(split.crossing_time_s, whole.crossing_time_s, atol=1e-12)
    np.testing.assert_allclose(split.decision_time_s, whole.decision_time_s, atol=1e-12)
    assert whole.ready.sum() > 0
    assert whole.event.sum() > 0
