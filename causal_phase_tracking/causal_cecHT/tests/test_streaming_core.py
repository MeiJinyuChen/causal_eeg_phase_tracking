from __future__ import annotations

import ast
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_cecHT.streaming_core import (
    CausalCecHTCore,
    CecHTBlockContext,
    CecHTCoreConfig,
    CecHTOutputBuffers,
    CecHTProcessStatus,
    CecHTReasonCode,
    CecHTVerifiedField,
    wrap_phase_rad,
)
from causal_phase_tracking.causal_cecHT.vendor.eikeosmers_cecht.phase import ECHT


ENDPOINT_ATOL = 2e-12


def make_config(
    *,
    window_samples: int = 8,
    max_block_samples: int = 32,
    input_dtype: np.dtype = np.dtype(np.float64),
    sampling_rate_hz: float = 250.0,
) -> CecHTCoreConfig:
    return CecHTCoreConfig(
        sampling_rate_hz=sampling_rate_hz,
        input_dtype=input_dtype,
        numeric_unit_code=1,
        channel_code=7,
        polarity_code=1,
        window_samples=window_samples,
        l_freq=1.0,
        h_freq=20.0,
        filter_order=1,
        filter_type="butter",
        n_fft=window_samples,
        fft_mode="exact",
        calibrate=False,
        f0=None,
        parameter_version="step4-stream-test",
        config_version="step4-stream-v1",
        max_block_samples=max_block_samples,
    )


def make_context(
    *,
    first_relative_index: int,
    stream_epoch: int = 1,
) -> CecHTBlockContext:
    return CecHTBlockContext(
        stream_epoch=stream_epoch,
        first_relative_index=first_relative_index,
        declared_sampling_rate_hz=250.0,
        unit_code=1,
        channel_code=7,
        polarity_code=1,
        verified_fields_mask=CecHTVerifiedField.ALL,
    )


def test_constructor_accepts_exact_250_hz_and_rejects_100_hz() -> None:
    config = make_config(sampling_rate_hz=250.0)

    assert config.sampling_rate_hz == 250.0
    with pytest.raises(ValueError, match="sampling_rate_hz must be exactly 250.0"):
        make_config(sampling_rate_hz=100.0)


def make_oracle(config: CecHTCoreConfig) -> ECHT:
    oracle = ECHT(
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        sfreq=config.sampling_rate_hz,
        n_fft=config.n_fft,
        fft_mode=config.fft_mode,
        filt_order=config.filter_order,
        filter_type=config.filter_type,
        calibrate=config.calibrate,
        f0=config.f0,
    )
    oracle.fit(np.zeros(config.window_samples, dtype=config.input_dtype))
    return oracle


def test_fixed_storage_and_warmup_align_the_nth_sample_to_the_endpoint() -> None:
    config = make_config()
    core = CausalCecHTCore(config)
    ring_id = id(core._ring)
    finite_id = id(core._finite_scratch)
    magnitude_id = id(core._magnitude_scratch)
    dot_ids = tuple(id(item) for item in core._dot_scratch)
    samples = np.linspace(-1.0, 1.0, config.window_samples, dtype=np.float64)
    first = CecHTOutputBuffers.allocate(config.window_samples - 1)

    status = core.process_block_into(
        samples[:-1], make_context(first_relative_index=0), first
    )

    assert status is CecHTProcessStatus.OK
    assert np.all(first.ready is False) if first.ready.ndim == 0 else not first.ready.any()
    assert not first.valid.any()
    assert np.all(first.reason_code == CecHTReasonCode.WARMUP)
    np.testing.assert_array_equal(first.phase_rad, 0.0)
    np.testing.assert_array_equal(first.amplitude, 0.0)
    np.testing.assert_array_equal(
        first.relative_sample_index, np.arange(config.window_samples - 1)
    )

    endpoint = CecHTOutputBuffers.allocate(1)
    status = core.process_block_into(
        samples[-1:],
        make_context(first_relative_index=config.window_samples - 1),
        endpoint,
    )

    expected = np.asarray(make_oracle(config).transform(samples)).ravel()[-1]
    assert status is CecHTProcessStatus.OK
    assert endpoint.ready.tolist() == [True]
    assert endpoint.valid.tolist() == [True]
    assert endpoint.reason_code.tolist() == [CecHTReasonCode.NONE]
    assert endpoint.relative_sample_index.tolist() == [config.window_samples - 1]
    np.testing.assert_allclose(endpoint.phase_rad[0], np.angle(expected), atol=ENDPOINT_ATOL)
    np.testing.assert_allclose(endpoint.amplitude[0], abs(expected), atol=ENDPOINT_ATOL)

    assert core._ring.dtype == config.input_dtype
    assert core._ring.shape == (config.window_samples,)
    assert core._finite_scratch.dtype == np.dtype(np.bool_)
    assert core._finite_scratch.shape == (config.max_block_samples,)
    assert core._magnitude_scratch.dtype == config.input_dtype
    assert core._magnitude_scratch.shape == (config.max_block_samples,)
    assert len(core._dot_scratch) >= 2
    assert all(item.shape == () and item.dtype == np.dtype(np.complex128) for item in core._dot_scratch)
    assert id(core._ring) == ring_id
    assert id(core._finite_scratch) == finite_id
    assert id(core._magnitude_scratch) == magnitude_id
    assert tuple(id(item) for item in core._dot_scratch) == dot_ids


def test_every_ring_write_position_matches_the_pinned_trailing_window_endpoint() -> None:
    config = make_config(window_samples=9)
    core = CausalCecHTCore(config)
    oracle = make_oracle(config)
    samples = np.sin(np.arange(27, dtype=np.float64) * 0.19) + 0.2 * np.cos(
        np.arange(27, dtype=np.float64) * 0.07
    )
    outputs = CecHTOutputBuffers.allocate(1)

    for index, sample in enumerate(samples):
        status = core.process_block_into(
            samples[index : index + 1],
            make_context(first_relative_index=index),
            outputs,
        )
        assert status is CecHTProcessStatus.OK
        if index < config.window_samples - 1:
            assert outputs.ready.tolist() == [False]
            assert outputs.reason_code.tolist() == [CecHTReasonCode.WARMUP]
            continue

        window = samples[index - config.window_samples + 1 : index + 1]
        expected = np.asarray(oracle.transform(window)).ravel()[-1]
        assert outputs.ready.tolist() == [True]
        np.testing.assert_allclose(outputs.phase_rad[0], np.angle(expected), atol=ENDPOINT_ATOL)
        np.testing.assert_allclose(outputs.amplitude[0], abs(expected), atol=ENDPOINT_ATOL)


def test_ready_processing_never_calls_pinned_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_config()
    core = CausalCecHTCore(config)

    def forbidden_transform(*args, **kwargs):
        raise AssertionError("transform is forbidden in the streaming hot path")

    monkeypatch.setattr(core._echt, "transform", forbidden_transform)
    samples = np.arange(config.window_samples + 3, dtype=np.float64)
    outputs = CecHTOutputBuffers.allocate(samples.size)

    status = core.process_block_into(samples, make_context(first_relative_index=0), outputs)

    assert status is CecHTProcessStatus.OK
    assert outputs.ready.sum() == 4


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(math.pi, -math.pi, id="positive-pi"),
        pytest.param(-math.pi, -math.pi, id="negative-pi"),
        pytest.param(3.0 * math.pi, -math.pi, id="three-pi"),
        pytest.param(0.0, 0.0, id="zero"),
    ],
)
def test_phase_normalization_is_half_open(value: float, expected: float) -> None:
    assert wrap_phase_rad(value) == expected


def test_process_does_not_modify_input() -> None:
    config = make_config()
    core = CausalCecHTCore(config)
    samples = np.linspace(-2.0, 2.0, 12, dtype=np.float64)
    frozen = samples.copy()

    status = core.process_block_into(
        samples, make_context(first_relative_index=0), CecHTOutputBuffers.allocate(12)
    )

    assert status is CecHTProcessStatus.OK
    np.testing.assert_array_equal(samples, frozen)


@pytest.mark.parametrize(
    ("samples", "reason"),
    [
        pytest.param([1.0, 2.0], CecHTReasonCode.STRUCTURAL_INPUT_TYPE, id="list"),
        pytest.param(
            np.ones((2, 2), dtype=np.float64),
            CecHTReasonCode.STRUCTURAL_INPUT_NDIM,
            id="two-dimensional",
        ),
        pytest.param(
            np.ones(4, dtype=np.float32),
            CecHTReasonCode.STRUCTURAL_INPUT_DTYPE,
            id="dtype-mismatch",
        ),
        pytest.param(
            np.ones(8, dtype=np.float64)[::2],
            CecHTReasonCode.STRUCTURAL_INPUT_CONTIGUITY,
            id="non-contiguous",
        ),
        pytest.param(
            np.ones(4, dtype=np.complex128),
            CecHTReasonCode.STRUCTURAL_INPUT_DTYPE,
            id="complex",
        ),
    ],
)
def test_structural_input_reject_writes_zero_slots_and_clears_state(
    samples, reason: CecHTReasonCode
) -> None:
    config = make_config()
    core = CausalCecHTCore(config)
    seeded = CecHTOutputBuffers.allocate(2)
    core.process_block_into(
        np.ones(2, dtype=np.float64), make_context(first_relative_index=0), seeded
    )
    outputs = CecHTOutputBuffers.allocate(8)
    outputs.phase_rad[:] = 123.0
    outputs.reason_code[:] = 99

    status = core.process_block_into(samples, make_context(first_relative_index=2), outputs)

    assert status is CecHTProcessStatus.STRUCTURAL_REJECT
    assert core.last_reason_code is reason
    np.testing.assert_array_equal(outputs.phase_rad, 123.0)
    np.testing.assert_array_equal(outputs.reason_code, 99)
    assert core.sample_count == 0


def test_wrong_output_schema_alias_and_capacity_are_structural_rejects() -> None:
    config = make_config(max_block_samples=4)
    samples = np.ones(4, dtype=np.float64)
    context = make_context(first_relative_index=0)

    wrong_dtype = CecHTOutputBuffers.allocate(4)
    wrong_dtype.phase_rad = np.zeros(4, dtype=np.float32)
    core = CausalCecHTCore(config)
    assert core.process_block_into(samples, context, wrong_dtype) is CecHTProcessStatus.STRUCTURAL_REJECT
    assert core.last_reason_code is CecHTReasonCode.STRUCTURAL_OUTPUT_SCHEMA

    alias = CecHTOutputBuffers.allocate(4)
    alias.phase_rad = samples
    core = CausalCecHTCore(config)
    assert core.process_block_into(samples, context, alias) is CecHTProcessStatus.STRUCTURAL_REJECT
    assert core.last_reason_code is CecHTReasonCode.STRUCTURAL_ALIAS

    shared_bool = np.zeros(4, dtype=np.bool_)
    output_alias = CecHTOutputBuffers.allocate(4)
    output_alias.valid = shared_bool
    output_alias.ready = shared_bool
    core = CausalCecHTCore(config)
    assert core.process_block_into(samples, context, output_alias) is CecHTProcessStatus.STRUCTURAL_REJECT
    assert core.last_reason_code is CecHTReasonCode.STRUCTURAL_ALIAS

    oversize = np.ones(5, dtype=np.float64)
    core = CausalCecHTCore(config)
    assert core.process_block_into(
        oversize, context, CecHTOutputBuffers.allocate(5)
    ) is CecHTProcessStatus.STRUCTURAL_REJECT
    assert core.last_reason_code is CecHTReasonCode.STRUCTURAL_CAPACITY

    undersized = CecHTOutputBuffers.allocate(3)
    core = CausalCecHTCore(config)
    assert core.process_block_into(
        samples, context, undersized
    ) is CecHTProcessStatus.STRUCTURAL_REJECT
    assert core.last_reason_code is CecHTReasonCode.STRUCTURAL_CAPACITY


@pytest.mark.parametrize(
    "field_name",
    [
        "phase_rad",
        "amplitude",
        "valid",
        "ready",
        "reason_code",
        "relative_sample_index",
    ],
)
def test_read_only_output_array_is_rejected_before_state_mutation(
    field_name: str,
) -> None:
    config = make_config(max_block_samples=4)
    core = CausalCecHTCore(config)
    core.process_block_into(
        np.ones(1, dtype=np.float64),
        make_context(first_relative_index=0),
        CecHTOutputBuffers.allocate(1),
    )
    outputs = CecHTOutputBuffers.allocate(2)
    outputs.phase_rad.fill(11.25)
    outputs.amplitude.fill(22.5)
    outputs.valid.fill(True)
    outputs.ready.fill(True)
    outputs.reason_code.fill(123)
    outputs.relative_sample_index.fill(456)
    before = {
        name: getattr(outputs, name).copy()
        for name in (
            "phase_rad",
            "amplitude",
            "valid",
            "ready",
            "reason_code",
            "relative_sample_index",
        )
    }
    target = getattr(outputs, field_name)
    target.flags.writeable = False

    status = core.process_block_into(
        np.ones(2, dtype=np.float64),
        make_context(first_relative_index=1),
        outputs,
    )

    assert status is CecHTProcessStatus.STRUCTURAL_REJECT
    assert core.last_reason_code is CecHTReasonCode.STRUCTURAL_OUTPUT_SCHEMA
    assert core.sample_count == 0
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(outputs, name), expected)


def assert_output_prefix_equal(
    left: CecHTOutputBuffers, right: CecHTOutputBuffers, size: int
) -> None:
    np.testing.assert_allclose(
        left.phase_rad[:size], right.phase_rad[:size], rtol=0.0, atol=ENDPOINT_ATOL
    )
    np.testing.assert_allclose(
        left.amplitude[:size], right.amplitude[:size], rtol=0.0, atol=ENDPOINT_ATOL
    )
    np.testing.assert_array_equal(left.valid[:size], right.valid[:size])
    np.testing.assert_array_equal(left.ready[:size], right.ready[:size])
    np.testing.assert_array_equal(left.reason_code[:size], right.reason_code[:size])
    np.testing.assert_array_equal(
        left.relative_sample_index[:size], right.relative_sample_index[:size]
    )


def snapshot_dynamic_state(core: CausalCecHTCore) -> tuple:
    return (
        core._ring.copy(),
        core._write_pos,
        core.sample_count,
        core._stream_epoch,
        core._expected_relative_index,
        core.last_reason_code,
    )


def test_empty_block_writes_nothing_and_preserves_dynamic_state() -> None:
    config = make_config()
    core = CausalCecHTCore(config)
    core.process_block_into(
        np.ones(3, dtype=np.float64),
        make_context(first_relative_index=0),
        CecHTOutputBuffers.allocate(3),
    )
    before = snapshot_dynamic_state(core)
    outputs = CecHTOutputBuffers.allocate(2)
    outputs.phase_rad[:] = 91.0

    status = core.process_block_into(
        np.empty(0, dtype=np.float64),
        make_context(first_relative_index=999, stream_epoch=999),
        outputs,
    )

    assert status is CecHTProcessStatus.EMPTY
    after = snapshot_dynamic_state(core)
    np.testing.assert_array_equal(after[0], before[0])
    assert after[1:] == before[1:]
    np.testing.assert_array_equal(outputs.phase_rad, 91.0)


def test_epoch_change_resets_then_ingests_the_new_block() -> None:
    config = make_config()
    restarted = CausalCecHTCore(config)
    restarted.process_block_into(
        np.arange(5, dtype=np.float64),
        make_context(first_relative_index=0, stream_epoch=10),
        CecHTOutputBuffers.allocate(5),
    )
    new_samples = np.linspace(3.0, 4.0, 10, dtype=np.float64)
    restarted_outputs = CecHTOutputBuffers.allocate(new_samples.size)

    status = restarted.process_block_into(
        new_samples,
        make_context(first_relative_index=200, stream_epoch=11),
        restarted_outputs,
    )

    fresh = CausalCecHTCore(config)
    fresh_outputs = CecHTOutputBuffers.allocate(new_samples.size)
    fresh_status = fresh.process_block_into(
        new_samples,
        make_context(first_relative_index=200, stream_epoch=11),
        fresh_outputs,
    )
    assert status is CecHTProcessStatus.EPOCH_RESTARTED
    assert fresh_status is CecHTProcessStatus.OK
    assert_output_prefix_equal(restarted_outputs, fresh_outputs, new_samples.size)
    restarted_state = snapshot_dynamic_state(restarted)
    fresh_state = snapshot_dynamic_state(fresh)
    np.testing.assert_array_equal(restarted_state[0], fresh_state[0])
    assert restarted_state[1:] == fresh_state[1:]


def test_explicit_reset_then_ingest_matches_a_fresh_core() -> None:
    config = make_config()
    reset_core = CausalCecHTCore(config)
    reset_core.process_block_into(
        np.arange(7, dtype=np.float64),
        make_context(first_relative_index=0),
        CecHTOutputBuffers.allocate(7),
    )
    reset_core.reset(CecHTReasonCode.EXPLICIT_RESET)
    samples = np.linspace(-0.5, 0.5, 9, dtype=np.float64)
    reset_outputs = CecHTOutputBuffers.allocate(9)
    fresh_outputs = CecHTOutputBuffers.allocate(9)

    reset_status = reset_core.process_block_into(
        samples, make_context(first_relative_index=100), reset_outputs
    )
    fresh_core = CausalCecHTCore(config)
    fresh_status = fresh_core.process_block_into(
        samples, make_context(first_relative_index=100), fresh_outputs
    )

    assert reset_status is fresh_status is CecHTProcessStatus.OK
    assert_output_prefix_equal(reset_outputs, fresh_outputs, samples.size)


def stream_fault_cases() -> list[pytest.ParameterSet]:
    full = CecHTVerifiedField.ALL
    return [
        pytest.param(
            np.array([1.0, np.nan, 3.0]),
            make_context(first_relative_index=3),
            CecHTReasonCode.NONFINITE_INPUT,
            id="nan",
        ),
        pytest.param(
            np.array([1.0, np.inf, 3.0]),
            make_context(first_relative_index=3),
            CecHTReasonCode.NONFINITE_INPUT,
            id="inf",
        ),
        pytest.param(
            np.ones(3),
            make_context(first_relative_index=4),
            CecHTReasonCode.SAMPLE_GAP,
            id="gap",
        ),
        pytest.param(
            np.ones(1),
            make_context(first_relative_index=2),
            CecHTReasonCode.DUPLICATE_SAMPLE,
            id="duplicate",
        ),
        pytest.param(
            np.ones(3),
            make_context(first_relative_index=1),
            CecHTReasonCode.OUT_OF_ORDER_SAMPLE,
            id="out-of-order",
        ),
        pytest.param(
            np.ones(3),
            CecHTBlockContext(1, 3, 250.0, 1, 7, 1, full & ~CecHTVerifiedField.SAMPLING_RATE),
            CecHTReasonCode.SAMPLING_RATE_UNKNOWN,
            id="rate-unknown",
        ),
        pytest.param(
            np.ones(3),
            CecHTBlockContext(1, 3, 100.0, 1, 7, 1, full),
            CecHTReasonCode.SAMPLING_RATE_MISMATCH,
            id="rate-mismatch",
        ),
        pytest.param(
            np.ones(3),
            CecHTBlockContext(1, 3, 250.0, 1, 7, 1, full & ~CecHTVerifiedField.UNIT),
            CecHTReasonCode.UNIT_UNKNOWN,
            id="unit-unknown",
        ),
        pytest.param(
            np.ones(3),
            CecHTBlockContext(1, 3, 250.0, 2, 7, 1, full),
            CecHTReasonCode.UNIT_MISMATCH,
            id="unit-mismatch",
        ),
        pytest.param(
            np.ones(3),
            CecHTBlockContext(1, 3, 250.0, 1, 7, 1, full & ~CecHTVerifiedField.CHANNEL),
            CecHTReasonCode.CHANNEL_UNKNOWN,
            id="channel-unknown",
        ),
        pytest.param(
            np.ones(3),
            CecHTBlockContext(1, 3, 250.0, 1, 8, 1, full),
            CecHTReasonCode.CHANNEL_MISMATCH,
            id="channel-mismatch",
        ),
        pytest.param(
            np.ones(3),
            CecHTBlockContext(1, 3, 250.0, 1, 7, 1, full & ~CecHTVerifiedField.POLARITY),
            CecHTReasonCode.POLARITY_UNKNOWN,
            id="polarity-unknown",
        ),
        pytest.param(
            np.ones(3),
            CecHTBlockContext(1, 3, 250.0, 1, 7, 2, full),
            CecHTReasonCode.POLARITY_MISMATCH,
            id="polarity-mismatch",
        ),
    ]


@pytest.mark.parametrize(("fault_samples", "fault_context", "reason"), stream_fault_cases())
def test_stream_fault_invalidates_the_whole_block_without_prefix_state(
    fault_samples: np.ndarray,
    fault_context: CecHTBlockContext,
    reason: CecHTReasonCode,
) -> None:
    config = make_config()
    core = CausalCecHTCore(config)
    core.process_block_into(
        np.ones(3, dtype=np.float64),
        make_context(first_relative_index=0),
        CecHTOutputBuffers.allocate(3),
    )
    outputs = CecHTOutputBuffers.allocate(max(4, fault_samples.size))
    outputs.phase_rad[:] = 44.0
    outputs.amplitude[:] = 55.0
    outputs.valid[:] = True
    outputs.ready[:] = True
    outputs.reason_code[:] = 99

    status = core.process_block_into(fault_samples, fault_context, outputs)

    size = fault_samples.size
    assert status is CecHTProcessStatus.STREAM_FAULT
    assert core.last_reason_code is reason
    np.testing.assert_array_equal(outputs.phase_rad[:size], 0.0)
    np.testing.assert_array_equal(outputs.amplitude[:size], 0.0)
    assert not outputs.valid[:size].any()
    assert not outputs.ready[:size].any()
    np.testing.assert_array_equal(outputs.reason_code[:size], reason)
    assert core.sample_count == 0
    assert core._write_pos == 0
    assert core._stream_epoch is None
    assert core._expected_relative_index is None
    np.testing.assert_array_equal(core._ring, 0.0)

    recovery = CecHTOutputBuffers.allocate(config.window_samples)
    recovered = core.process_block_into(
        np.ones(config.window_samples, dtype=np.float64),
        make_context(first_relative_index=100),
        recovery,
    )
    assert recovered is CecHTProcessStatus.OK
    assert recovery.ready.tolist() == [False] * (config.window_samples - 1) + [True]


def test_malformed_context_is_a_zero_slot_structural_reject() -> None:
    config = make_config()
    core = CausalCecHTCore(config)
    outputs = CecHTOutputBuffers.allocate(2)
    outputs.phase_rad[:] = 17.0

    status = core.process_block_into(
        np.ones(2, dtype=np.float64), object(), outputs
    )

    assert status is CecHTProcessStatus.STRUCTURAL_REJECT
    assert core.last_reason_code is CecHTReasonCode.STRUCTURAL_CONTEXT_SCHEMA
    np.testing.assert_array_equal(outputs.phase_rad, 17.0)
    assert core.sample_count == 0


def process_with_chunks(
    samples: np.ndarray,
    chunks: list[int],
    *,
    config: CecHTCoreConfig,
    stream_epoch: int = 1,
    first_index: int = 0,
) -> tuple[CausalCecHTCore, dict[str, np.ndarray]]:
    core = CausalCecHTCore(config)
    captured = {
        "phase": [],
        "amplitude": [],
        "valid": [],
        "ready": [],
        "reason": [],
        "index": [],
    }
    cursor = 0
    chunk_cursor = 0
    while cursor < samples.size:
        width = min(chunks[chunk_cursor % len(chunks)], samples.size - cursor)
        outputs = CecHTOutputBuffers.allocate(width)
        status = core.process_block_into(
            samples[cursor : cursor + width],
            make_context(
                first_relative_index=first_index + cursor, stream_epoch=stream_epoch
            ),
            outputs,
        )
        assert status is CecHTProcessStatus.OK
        captured["phase"].append(outputs.phase_rad.copy())
        captured["amplitude"].append(outputs.amplitude.copy())
        captured["valid"].append(outputs.valid.copy())
        captured["ready"].append(outputs.ready.copy())
        captured["reason"].append(outputs.reason_code.copy())
        captured["index"].append(outputs.relative_sample_index.copy())
        cursor += width
        chunk_cursor += 1
    return core, {key: np.concatenate(parts) for key, parts in captured.items()}


@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param([1], id="singleton"),
        pytest.param([1, 7, 25, 100, 17, 50], id="declared-variable"),
        pytest.param([18, 3, 11, 29], id="fixed-pattern"),
        pytest.param(
            np.random.default_rng(20260807).integers(1, 32, size=25).tolist(),
            id="seeded-random",
        ),
    ],
)
def test_legal_chunking_preserves_all_outputs_and_final_state(chunks: list[int]) -> None:
    config = make_config(window_samples=17, max_block_samples=256)
    x = np.arange(240, dtype=np.float64)
    samples = np.sin(2.0 * np.pi * 0.8 * x / 250.0) + 0.1 * np.cos(
        2.0 * np.pi * 1.1 * x / 250.0
    )
    whole_core, whole = process_with_chunks(samples, [samples.size], config=config)
    split_core, split = process_with_chunks(samples, chunks, config=config)

    for key in ("phase", "amplitude"):
        np.testing.assert_allclose(split[key], whole[key], rtol=0.0, atol=ENDPOINT_ATOL)
    for key in ("valid", "ready", "reason", "index"):
        np.testing.assert_array_equal(split[key], whole[key])
    whole_state = snapshot_dynamic_state(whole_core)
    split_state = snapshot_dynamic_state(split_core)
    np.testing.assert_array_equal(split_state[0], whole_state[0])
    assert split_state[1:] == whole_state[1:]


def test_legal_prefix_is_invariant_to_future_samples_in_the_same_block() -> None:
    config = make_config(window_samples=13, max_block_samples=128)
    samples = np.random.default_rng(108).standard_normal(80)
    prefix_size = 37
    full_core = CausalCecHTCore(config)
    full_outputs = CecHTOutputBuffers.allocate(samples.size)
    prefix_core = CausalCecHTCore(config)
    prefix_outputs = CecHTOutputBuffers.allocate(prefix_size)

    full_core.process_block_into(samples, make_context(first_relative_index=0), full_outputs)
    prefix_core.process_block_into(
        samples[:prefix_size], make_context(first_relative_index=0), prefix_outputs
    )

    assert_output_prefix_equal(full_outputs, prefix_outputs, prefix_size)


def test_fault_boundary_recovery_is_independent_of_legal_chunking() -> None:
    config = make_config(window_samples=8, max_block_samples=32)
    before = np.linspace(-1.0, 0.0, 15, dtype=np.float64)
    fault = np.array([0.1, np.nan, 0.2], dtype=np.float64)
    after = np.linspace(0.0, 1.0, 15, dtype=np.float64)

    def run(pre_chunks: list[int], post_chunks: list[int]) -> tuple[dict, tuple]:
        core, pre = process_with_chunks(before, pre_chunks, config=config)
        fault_outputs = CecHTOutputBuffers.allocate(fault.size)
        status = core.process_block_into(
            fault, make_context(first_relative_index=before.size), fault_outputs
        )
        assert status is CecHTProcessStatus.STREAM_FAULT

        captured = {key: [value] for key, value in pre.items()}
        captured["phase"].append(fault_outputs.phase_rad.copy())
        captured["amplitude"].append(fault_outputs.amplitude.copy())
        captured["valid"].append(fault_outputs.valid.copy())
        captured["ready"].append(fault_outputs.ready.copy())
        captured["reason"].append(fault_outputs.reason_code.copy())
        captured["index"].append(fault_outputs.relative_sample_index.copy())

        cursor = 0
        chunk_cursor = 0
        while cursor < after.size:
            width = min(post_chunks[chunk_cursor % len(post_chunks)], after.size - cursor)
            outputs = CecHTOutputBuffers.allocate(width)
            status = core.process_block_into(
                after[cursor : cursor + width],
                make_context(first_relative_index=before.size + fault.size + cursor),
                outputs,
            )
            assert status is CecHTProcessStatus.OK
            captured["phase"].append(outputs.phase_rad.copy())
            captured["amplitude"].append(outputs.amplitude.copy())
            captured["valid"].append(outputs.valid.copy())
            captured["ready"].append(outputs.ready.copy())
            captured["reason"].append(outputs.reason_code.copy())
            captured["index"].append(outputs.relative_sample_index.copy())
            cursor += width
            chunk_cursor += 1
        joined = {key: np.concatenate(parts) for key, parts in captured.items()}
        return joined, snapshot_dynamic_state(core)

    one, one_state = run([15], [15])
    split, split_state = run([1, 7, 3], [2, 5, 1])
    for key in ("phase", "amplitude"):
        np.testing.assert_allclose(split[key], one[key], rtol=0.0, atol=ENDPOINT_ATOL)
    for key in ("valid", "ready", "reason", "index"):
        np.testing.assert_array_equal(split[key], one[key])
    np.testing.assert_array_equal(split_state[0], one_state[0])
    assert split_state[1:] == one_state[1:]


def test_separate_instances_are_isolated_and_sequential_runs_are_deterministic() -> None:
    config = make_config(window_samples=9, max_block_samples=32)
    samples_a = np.arange(25, dtype=np.float64)
    samples_b = -np.arange(25, dtype=np.float64)
    first = CausalCecHTCore(config)
    second = CausalCecHTCore(config)
    first_outputs = CecHTOutputBuffers.allocate(25)
    second_outputs = CecHTOutputBuffers.allocate(25)

    first.process_block_into(samples_a, make_context(first_relative_index=0), first_outputs)
    first_state_before_second = snapshot_dynamic_state(first)
    second.process_block_into(samples_b, make_context(first_relative_index=0), second_outputs)
    first_state_after_second = snapshot_dynamic_state(first)

    np.testing.assert_array_equal(first_state_after_second[0], first_state_before_second[0])
    assert first_state_after_second[1:] == first_state_before_second[1:]
    assert not np.array_equal(first._ring, second._ring)

    repeated = CausalCecHTCore(config)
    repeated_outputs = CecHTOutputBuffers.allocate(25)
    repeated.process_block_into(
        samples_a, make_context(first_relative_index=0), repeated_outputs
    )
    assert_output_prefix_equal(first_outputs, repeated_outputs, samples_a.size)


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"sampling_rate_hz": 100.0}, id="not-250-hz"),
        pytest.param({"input_dtype": np.dtype(np.int16)}, id="integer-dtype"),
        pytest.param({"input_dtype": np.dtype(np.float16)}, id="float16-dtype"),
        pytest.param({"numeric_unit_code": 0}, id="unit-code-not-positive"),
        pytest.param({"channel_code": "Fpz-Cz"}, id="channel-code-not-integral"),
        pytest.param({"polarity_code": False}, id="polarity-code-bool"),
        pytest.param({"window_samples": 0}, id="empty-window"),
        pytest.param({"l_freq": 0.0}, id="nonpositive-low-cutoff"),
        pytest.param({"h_freq": 125.0}, id="high-cutoff-at-nyquist"),
        pytest.param({"l_freq": 2.0, "h_freq": 1.0}, id="reversed-band"),
        pytest.param({"filter_order": 0}, id="nonpositive-filter-order"),
        pytest.param({"filter_type": "unknown"}, id="unknown-filter"),
        pytest.param({"fft_mode": "other"}, id="unknown-fft-mode"),
        pytest.param({"n_fft": True}, id="boolean-fft-length"),
        pytest.param({"parameter_version": ""}, id="empty-parameter-version"),
        pytest.param({"config_version": ""}, id="empty-config-version"),
        pytest.param({"max_block_samples": 0}, id="empty-block-capacity"),
        pytest.param({"method_id": "other"}, id="wrong-method-id"),
        pytest.param({"calibrate": True, "f0": None}, id="missing-calibration-f0"),
        pytest.param({"calibrate": False, "f0": 0.875}, id="unused-calibration-f0"),
    ],
)
def test_constructor_rejects_invalid_or_implicit_configuration(changes: dict) -> None:
    values = {
        "sampling_rate_hz": 250.0,
        "input_dtype": np.dtype(np.float64),
        "numeric_unit_code": 1,
        "channel_code": 7,
        "polarity_code": 1,
        "window_samples": 8,
        "l_freq": 1.0,
        "h_freq": 20.0,
        "filter_order": 1,
        "filter_type": "butter",
        "n_fft": 8,
        "fft_mode": "exact",
        "calibrate": False,
        "f0": None,
        "parameter_version": "step4-stream-test",
        "config_version": "step4-stream-v1",
        "max_block_samples": 32,
        "method_id": "cecHT",
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError)):
        CausalCecHTCore(CecHTCoreConfig(**values))


def test_float32_input_uses_a_float32_ring_without_coercion() -> None:
    config = make_config(input_dtype=np.dtype(np.float32))
    core = CausalCecHTCore(config)
    samples = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    outputs = CecHTOutputBuffers.allocate(samples.size)

    status = core.process_block_into(
        samples, make_context(first_relative_index=0), outputs
    )

    assert status is CecHTProcessStatus.OK
    assert core._ring.dtype == np.dtype(np.float32)
    assert outputs.ready.sum() == samples.size - config.window_samples + 1


def test_noncontiguous_or_multidimensional_output_is_a_structural_reject() -> None:
    config = make_config()
    samples = np.ones(2, dtype=np.float64)
    context = make_context(first_relative_index=0)

    noncontiguous = CecHTOutputBuffers.allocate(4)
    noncontiguous.phase_rad = np.zeros(8, dtype=np.float64)[::2]
    core = CausalCecHTCore(config)
    assert core.process_block_into(
        samples, context, noncontiguous
    ) is CecHTProcessStatus.STRUCTURAL_REJECT
    assert core.last_reason_code is CecHTReasonCode.STRUCTURAL_OUTPUT_SCHEMA

    multidimensional = CecHTOutputBuffers.allocate(4)
    multidimensional.reason_code = np.zeros((2, 2), dtype=np.uint8)
    core = CausalCecHTCore(config)
    assert core.process_block_into(
        samples, context, multidimensional
    ) is CecHTProcessStatus.STRUCTURAL_REJECT
    assert core.last_reason_code is CecHTReasonCode.STRUCTURAL_OUTPUT_SCHEMA


def test_streaming_hot_call_graph_has_no_forbidden_fft_conversion_growth_or_io() -> None:
    source = (ROOT / "causal_phase_tracking" / "causal_cecHT" / "streaming_core.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    hot_functions = {
        "process_block_into",
        "_stream_fault_reason",
        "_write_invalid_block",
        "_endpoint_from_ring",
        "_structural_reason",
        "_valid_output_array",
        "_output_schema_is_valid",
        "_has_alias",
        "_context_schema_is_valid",
        "wrap_phase_rad",
    }
    forbidden_calls = {
        "fft",
        "ifft",
        "fftshift",
        "ifftshift",
        "_constructor_ifft",
        "_constructor_ifftshift",
        "transform",
        "fit",
        "fit_transform",
        "asarray",
        "astype",
        "append",
        "concatenate",
        "roll",
        "resize",
        "open",
        "print",
    }
    violations = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        candidates = node.body if isinstance(node, ast.ClassDef) else [node]
        for candidate in candidates:
            if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if candidate.name not in hot_functions:
                continue
            for child in ast.walk(candidate):
                if not isinstance(child, ast.Call):
                    continue
                if isinstance(child.func, ast.Attribute):
                    call_name = child.func.attr
                elif isinstance(child.func, ast.Name):
                    call_name = child.func.id
                else:
                    continue
                if call_name in forbidden_calls:
                    violations.append((candidate.name, call_name, child.lineno))

    assert violations == []


def test_nonfinite_endpoint_from_finite_extreme_input_fails_the_whole_block_closed() -> None:
    config = make_config()
    core = CausalCecHTCore(config)
    signs = np.where(core.endpoint_kernel.real >= 0.0, 1.0, -1.0)
    samples = signs.astype(np.float64) * np.finfo(np.float64).max
    outputs = CecHTOutputBuffers.allocate(samples.size)
    outputs.phase_rad[:] = 71.0
    outputs.amplitude[:] = 72.0
    outputs.valid[:] = True
    outputs.ready[:] = True

    with np.errstate(over="ignore", invalid="ignore"):
        status = core.process_block_into(
            samples, make_context(first_relative_index=0), outputs
        )

    assert status is CecHTProcessStatus.STREAM_FAULT
    assert core.last_reason_code is CecHTReasonCode.NONFINITE_ENDPOINT
    np.testing.assert_array_equal(outputs.phase_rad, 0.0)
    np.testing.assert_array_equal(outputs.amplitude, 0.0)
    assert not outputs.valid.any()
    assert not outputs.ready.any()
    np.testing.assert_array_equal(
        outputs.reason_code, CecHTReasonCode.NONFINITE_ENDPOINT
    )
    assert core.sample_count == 0
    np.testing.assert_array_equal(core._ring, 0.0)


def test_relative_index_that_cannot_fit_all_output_slots_is_structurally_rejected() -> None:
    config = make_config()
    core = CausalCecHTCore(config)
    outputs = CecHTOutputBuffers.allocate(2)
    outputs.relative_sample_index[:] = 41
    context = make_context(first_relative_index=np.iinfo(np.int64).max)

    status = core.process_block_into(np.ones(2, dtype=np.float64), context, outputs)

    assert status is CecHTProcessStatus.STRUCTURAL_REJECT
    assert core.last_reason_code is CecHTReasonCode.STRUCTURAL_CONTEXT_SCHEMA
    np.testing.assert_array_equal(outputs.relative_sample_index, 41)
    assert core.sample_count == 0
