from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.signal import firwin, upfirdn


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_simulation_evaluation.rate_adapter import (
    Causal100To250Adapter,
    RationalRateAdapterConfig,
    RationalRateAdapterStatus,
    RationalRateReasonCode,
)


def make_config(*, max_input_samples: int = 4096) -> RationalRateAdapterConfig:
    return RationalRateAdapterConfig(
        input_rate_hz=100.0,
        output_rate_hz=250.0,
        upsample=5,
        downsample=2,
        kernel_rate_hz=500.0,
        cutoff_hz=40.0,
        stopband_hz=50.0,
        kernel_taps=33,
        max_input_samples=max_input_samples,
        config_version="adapter-v1",
    )


def make_adapter(config: RationalRateAdapterConfig) -> Causal100To250Adapter:
    return Causal100To250Adapter(config)


def expected_kernel(config: RationalRateAdapterConfig) -> np.ndarray:
    return firwin(
        config.kernel_taps,
        config.cutoff_hz,
        fs=config.kernel_rate_hz,
        pass_zero="lowpass",
    ).astype(np.float64)


def run_all(
    adapter: Causal100To250Adapter,
    samples: np.ndarray,
    *,
    first_index: int = 0,
):
    count = adapter.output_count_for_range(first_index, samples.size)
    outputs = adapter.allocate_outputs(max(1, count))
    status = adapter.process_block_into(
        samples, first_index, outputs
    )
    return status, outputs


def test_constructor_validates_rates_and_kernel_parameters() -> None:
    config = make_config()
    assert config.upsample == 5
    assert config.downsample == 2
    assert config.kernel_taps == 33
    with pytest.raises(ValueError, match="input_rate_hz"):
        RationalRateAdapterConfig(
            input_rate_hz=200.0,
            output_rate_hz=250.0,
            upsample=5,
            downsample=2,
            kernel_rate_hz=500.0,
            cutoff_hz=40.0,
            stopband_hz=50.0,
            kernel_taps=33,
            max_input_samples=32,
            config_version="v",
        )
    with pytest.raises(ValueError, match="kernel_taps"):
        RationalRateAdapterConfig(
            input_rate_hz=100.0,
            output_rate_hz=250.0,
            upsample=5,
            downsample=2,
            kernel_rate_hz=500.0,
            cutoff_hz=40.0,
            stopband_hz=50.0,
            kernel_taps=32,
            max_input_samples=32,
            config_version="v",
        )


def test_group_delay_and_output_count_formula_are_exact() -> None:
    config = make_config()
    adapter = make_adapter(config)
    assert adapter.group_delay_output_samples == 8
    assert adapter.group_delay_s == pytest.approx(8 / 250.0)
    assert adapter.output_count_for(0) == 0
    assert adapter.output_count_for(2) == 5
    assert adapter.output_count_for(4) == 10


def test_impulse_responses_match_scipy_upfirdn_in_steady_state() -> None:
    config = make_config()
    adapter = make_adapter(config)
    for position in range(12):
        samples = np.zeros(24, dtype=np.float64)
        samples[position] = 1.0
        status, outputs = run_all(adapter, samples)
        assert status is RationalRateAdapterStatus.OK
        oracle = upfirdn(expected_kernel(config), samples, up=5, down=2)
        valid = outputs.valid
        for idx in range(8, outputs.resampled.size):
            assert bool(outputs.valid[idx]) is True
            assert outputs.resampled[idx] == pytest.approx(float(oracle[idx]), abs=1e-15)
            assert outputs.output_index[idx] == idx
        assert not valid[:8].any()
        adapter.reset(RationalRateReasonCode.EXPLICIT_RESET)


def test_random_sequence_matches_scipy_upfirdn_after_group_delay() -> None:
    config = make_config(max_input_samples=4096)
    adapter = make_adapter(config)
    rng = np.random.default_rng(20260910)
    samples = rng.standard_normal(1024)
    status, outputs = run_all(adapter, samples)
    assert status is RationalRateAdapterStatus.OK
    oracle = upfirdn(expected_kernel(config), samples, up=5, down=2)
    compare_end = outputs.resampled.size
    np.testing.assert_allclose(
        outputs.resampled[8:compare_end],
        oracle[8:compare_end],
        rtol=0.0,
        atol=1e-14,
    )


def test_arbitrary_chunking_is_invariant() -> None:
    config = make_config(max_input_samples=8192)
    rng = np.random.default_rng(77)
    samples = rng.standard_normal(500)
    whole_adapter = make_adapter(config)
    _, whole = run_all(whole_adapter, samples)

    chunks = [1, 7, 25, 100, 17, 50, 3, 100, 197]
    split_adapter = make_adapter(config)
    parts = []
    cursor = 0
    for width in chunks:
        block = samples[cursor : cursor + width]
        if block.size == 0:
            break
        _, part = run_all(split_adapter, block, first_index=cursor)
        parts.append(part)
        cursor += width
    split_resampled = np.concatenate([p.resampled for p in parts])
    split_index = np.concatenate([p.output_index for p in parts])
    np.testing.assert_allclose(split_resampled, whole.resampled, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(split_index, whole.output_index)


def test_prefix_outputs_do_not_change_when_future_samples_are_appended() -> None:
    config = make_config(max_input_samples=8192)
    rng = np.random.default_rng(108)
    samples = rng.standard_normal(300)
    prefix_adapter = make_adapter(config)
    _, prefix = run_all(prefix_adapter, samples[:180])
    future_adapter = make_adapter(config)
    _, future = run_all(future_adapter, samples)
    np.testing.assert_allclose(
        prefix.resampled,
        future.resampled[: prefix.resampled.size],
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_array_equal(prefix.output_index, future.output_index[: prefix.output_index.size])
    np.testing.assert_array_equal(prefix.valid, future.valid[: prefix.valid.size])


def test_gap_nan_and_recovery_are_fail_closed() -> None:
    config = make_config(max_input_samples=64)
    adapter = make_adapter(config)
    adapter.process_block_into(np.ones(4, dtype=np.float64), 0, adapter.allocate_outputs(10))
    gap_outputs = adapter.allocate_outputs(10)
    status = adapter.process_block_into(np.ones(4, dtype=np.float64), 8, gap_outputs)
    assert status is RationalRateAdapterStatus.STREAM_FAULT
    assert adapter.last_reason_code is RationalRateReasonCode.INPUT_GAP
    assert not gap_outputs.valid.any()
    assert adapter.sample_count == 0

    recovery = adapter.allocate_outputs(5)
    status = adapter.process_block_into(np.ones(2, dtype=np.float64), 100, recovery)
    assert status is RationalRateAdapterStatus.OK
    assert recovery.valid.tolist() == [False, False, False, False, False]
    assert adapter.sample_count == 2

    nan_outputs = adapter.allocate_outputs(10)
    status = adapter.process_block_into(np.array([1.0, np.nan]), 0, nan_outputs)
    assert status is RationalRateAdapterStatus.STREAM_FAULT
    assert adapter.last_reason_code is RationalRateReasonCode.NONFINITE_INPUT


def test_structural_rejects_cover_schema_capacity_and_alias() -> None:
    config = make_config(max_input_samples=4)
    adapter = make_adapter(config)
    outputs = adapter.allocate_outputs(10)
    assert adapter.process_block_into([1.0], 0, outputs) is (
        RationalRateAdapterStatus.STRUCTURAL_REJECT
    )
    assert adapter.process_block_into(np.ones(5, dtype=np.float64), 0, outputs) is (
        RationalRateAdapterStatus.STRUCTURAL_REJECT
    )
    alias = adapter.allocate_outputs(10)
    alias.resampled = np.ones(1, dtype=np.float64)
    assert adapter.process_block_into(np.ones(1, dtype=np.float64), 0, alias) is (
        RationalRateAdapterStatus.STRUCTURAL_REJECT
    )


def test_empty_block_preserves_state() -> None:
    config = make_config(max_input_samples=64)
    adapter = make_adapter(config)
    adapter.process_block_into(np.ones(2, dtype=np.float64), 0, adapter.allocate_outputs(5))
    outputs = adapter.allocate_outputs(1)
    outputs.resampled[0] = 99.0
    status = adapter.process_block_into(np.empty(0, dtype=np.float64), 99, outputs)
    assert status is RationalRateAdapterStatus.EMPTY
    assert outputs.resampled[0] == 99.0
    assert adapter.expected_input_index == 2
