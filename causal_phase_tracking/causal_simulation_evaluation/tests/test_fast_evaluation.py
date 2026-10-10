from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_cecHT.pseudo_trigger import (
    CausalCecHTTrigger,
    PseudoTriggerConfig,
    PseudoTriggerOutputBuffers,
)
from causal_phase_tracking.causal_cecHT.streaming_core import (
    CausalCecHTCore,
    CecHTBlockContext,
    CecHTCoreConfig,
    CecHTOutputBuffers,
    CecHTVerifiedField,
)
from causal_phase_tracking.causal_simulation_evaluation.fast_evaluation import (
    endpoint_sequence,
    pseudo_events_vectorized,
)


def core_config() -> CecHTCoreConfig:
    return CecHTCoreConfig(
        sampling_rate_hz=250.0,
        input_dtype=np.dtype(np.float64),
        numeric_unit_code=1,
        channel_code=7,
        polarity_code=1,
        window_samples=17,
        l_freq=0.5,
        h_freq=1.25,
        filter_order=2,
        filter_type="butter",
        n_fft=17,
        fft_mode="exact",
        calibrate=False,
        f0=None,
        parameter_version="fast-eval-test",
        config_version="fast-eval-v1",
        max_block_samples=1024,
    )


def test_fast_endpoint_sequence_matches_streaming_core() -> None:
    config = core_config()
    core = CausalCecHTCore(config)
    rng = np.random.default_rng(11)
    samples = rng.standard_normal(120)
    outputs = CecHTOutputBuffers.allocate(samples.size)
    core.process_block_into(
        samples,
        CecHTBlockContext(
            stream_epoch=1,
            first_relative_index=0,
            declared_sampling_rate_hz=250.0,
            unit_code=1,
            channel_code=7,
            polarity_code=1,
            verified_fields_mask=CecHTVerifiedField.ALL,
        ),
        outputs,
    )
    fast = endpoint_sequence(samples, core.endpoint_kernel)
    np.testing.assert_allclose(fast, outputs.phase_rad, rtol=0.0, atol=2e-12)


def test_fast_pseudo_events_match_streaming_trigger() -> None:
    fs = 250.0
    t = np.arange(4000, dtype=np.float64) / fs
    phase = np.unwrap(np.angle(np.exp(1j * (2.0 * np.pi * 0.8 * t + 0.05 * np.sin(2.0 * np.pi * 0.2 * t)))))
    trigger = CausalCecHTTrigger(
        PseudoTriggerConfig(
            sampling_rate_hz=fs,
            target_phase_rad=0.0,
            refractory_s=0.5,
            parameter_version="fast-eval-test",
            config_version="fast-eval-v1",
            max_block_samples=4000,
        )
    )
    index = np.arange(phase.size, dtype=np.int64)
    valid = np.ones(phase.size, dtype=np.bool_)
    outputs = PseudoTriggerOutputBuffers.allocate(phase.size)
    trigger.process_block_into(index, phase, valid, valid, outputs)

    fast_event, fast_crossing = pseudo_events_vectorized(
        phase, fs=fs, target_rad=0.0, refractory_s=0.5
    )
    np.testing.assert_array_equal(fast_event, outputs.event)
    np.testing.assert_allclose(
        fast_crossing[fast_event],
        outputs.crossing_time_s[outputs.event],
        rtol=0.0,
        atol=1e-12,
    )
