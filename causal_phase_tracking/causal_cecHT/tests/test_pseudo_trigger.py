from __future__ import annotations

import ast
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_cecHT.pseudo_trigger import (
    CausalCecHTTrigger,
    PseudoTriggerConfig,
    PseudoTriggerOutputBuffers,
    PseudoTriggerReasonCode,
    PseudoTriggerStatus,
)


def make_config(
    *,
    target_phase_rad: float = 0.0,
    refractory_s: float = 0.5,
) -> PseudoTriggerConfig:
    return PseudoTriggerConfig(
        sampling_rate_hz=250.0,
        target_phase_rad=target_phase_rad,
        refractory_s=refractory_s,
        parameter_version="step6-trigger-test",
        config_version="step6-trigger-v1",
        max_block_samples=64,
    )


def make_inputs(
    phases: list[float],
    *,
    first_index: int = 0,
    ready: list[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = len(phases)
    index = np.arange(first_index, first_index + size, dtype=np.int64)
    phase = np.asarray(phases, dtype=np.float64)
    valid = np.ones(size, dtype=np.bool_)
    ready_array = (
        np.asarray(ready, dtype=np.bool_) if ready is not None else valid.copy()
    )
    return index, phase, valid, ready_array


def run_trigger(
    trigger: CausalCecHTTrigger,
    phases: list[float],
    *,
    first_index: int = 0,
    ready: list[bool] | None = None,
) -> tuple[PseudoTriggerStatus, PseudoTriggerOutputBuffers]:
    index, phase, valid, ready_array = make_inputs(
        phases, first_index=first_index, ready=ready
    )
    outputs = PseudoTriggerOutputBuffers.allocate(len(phases))
    status = trigger.process_block_into(
        index, phase, valid, ready_array, outputs
    )
    return status, outputs


def test_constructor_validates_rate_target_and_refractory() -> None:
    assert make_config().target_phase_rad == 0.0
    assert make_config().refractory_s == 0.5
    with pytest.raises(ValueError, match="exactly 250.0"):
        PseudoTriggerConfig(
            sampling_rate_hz=100.0,
            target_phase_rad=0.0,
            refractory_s=0.5,
            parameter_version="v",
            config_version="v",
            max_block_samples=8,
        )
    with pytest.raises(ValueError, match="target_phase_rad"):
        make_config(target_phase_rad=math.pi)
    with pytest.raises(ValueError, match="refractory_s"):
        make_config(refractory_s=-0.1)


def test_positive_crossing_fires_with_interpolated_and_decision_times() -> None:
    trigger = CausalCecHTTrigger(make_config())
    status, outputs = run_trigger(
        trigger, [-0.5, 0.5], first_index=40
    )

    assert status is PseudoTriggerStatus.OK
    assert outputs.event.tolist() == [False, True]
    assert outputs.decision_time_s[1] == pytest.approx(41 / 250.0)
    # Previous endpoint time is 40/250 s; crossing is half-way to sample 41.
    assert outputs.crossing_time_s[1] == pytest.approx((40 + 0.5) / 250.0)


def test_negative_or_stationary_phase_delta_does_not_fire() -> None:
    trigger = CausalCecHTTrigger(make_config())
    status, outputs = run_trigger(trigger, [-0.2, -0.8])
    assert status is PseudoTriggerStatus.OK
    assert not outputs.event.any()

    second = CausalCecHTTrigger(make_config())
    _, second_outputs = run_trigger(second, [0.2, 0.3])
    assert not second_outputs.event.any()


def test_positive_wraparound_crossing_at_minus_pi() -> None:
    trigger = CausalCecHTTrigger(make_config(target_phase_rad=-math.pi))
    status, outputs = run_trigger(trigger, [3.0, -3.0], first_index=100)

    assert status is PseudoTriggerStatus.OK
    assert outputs.event.tolist() == [False, True]
    assert outputs.crossing_time_s[1] == pytest.approx((100 + 0.5) / 250.0)


def test_refractory_suppresses_duplicate_crossing_jitter() -> None:
    trigger = CausalCecHTTrigger(make_config(refractory_s=0.5))
    status, outputs = run_trigger(trigger, [-0.5, 0.5, -0.5, 0.5])

    assert status is PseudoTriggerStatus.OK
    assert outputs.event.tolist() == [False, True, False, False]


def test_target_update_clears_comparator_and_applies_at_next_block() -> None:
    trigger = CausalCecHTTrigger(make_config(target_phase_rad=0.0))
    _, first = run_trigger(trigger, [-0.4], first_index=0)
    assert not first.event.any()

    trigger.set_target_phase_rad(0.5)
    _, second = run_trigger(trigger, [0.1], first_index=1)
    assert not second.event.any()

    _, third = run_trigger(trigger, [0.6], first_index=2)
    assert third.event.tolist() == [True]
    assert trigger.target_phase_rad == 0.5


def test_not_ready_gap_clears_comparator_without_carry_forward() -> None:
    trigger = CausalCecHTTrigger(make_config())
    status, outputs = run_trigger(
        trigger, [-0.5, 0.0, 0.5], ready=[True, False, True]
    )

    assert status is PseudoTriggerStatus.OK
    assert not outputs.event.any()


def test_empty_block_preserves_trigger_state() -> None:
    trigger = CausalCecHTTrigger(make_config())
    run_trigger(trigger, [-0.5], first_index=0)
    empty = PseudoTriggerOutputBuffers.allocate(1)
    empty.event[0] = True

    status = trigger.process_block_into(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.bool_),
        np.empty(0, dtype=np.bool_),
        empty,
    )

    assert status is PseudoTriggerStatus.EMPTY
    assert empty.event.tolist() == [True]
    _, outputs = run_trigger(trigger, [0.5], first_index=1)
    assert outputs.event.tolist() == [True]


def test_structural_rejects_cover_schema_dtype_alias_and_capacity() -> None:
    trigger = CausalCecHTTrigger(make_config())
    index, phase, valid, ready = make_inputs([-0.5, 0.5])
    outputs = PseudoTriggerOutputBuffers.allocate(2)

    bad_phase = phase.astype(np.float32)
    assert trigger.process_block_into(
        index, bad_phase, valid, ready, outputs
    ) is PseudoTriggerStatus.STRUCTURAL_REJECT

    alias = PseudoTriggerOutputBuffers.allocate(2)
    alias.crossing_time_s = phase
    assert trigger.process_block_into(
        index, phase, valid, ready, alias
    ) is PseudoTriggerStatus.STRUCTURAL_REJECT

    small = PseudoTriggerOutputBuffers.allocate(1)
    assert trigger.process_block_into(
        index, phase, valid, ready, small
    ) is PseudoTriggerStatus.STRUCTURAL_REJECT


def test_legal_chunking_produces_identical_event_sequence() -> None:
    phases = [-0.6, 0.2, -0.4, 0.4]
    whole_trigger = CausalCecHTTrigger(make_config())
    _, whole = run_trigger(whole_trigger, phases)

    split_trigger = CausalCecHTTrigger(make_config())
    _, left = run_trigger(split_trigger, phases[:2], first_index=0)
    _, right = run_trigger(split_trigger, phases[2:], first_index=2)

    np.testing.assert_array_equal(
        np.concatenate([left.event, right.event]), whole.event
    )
    np.testing.assert_allclose(
        np.concatenate([left.crossing_time_s, right.crossing_time_s]),
        whole.crossing_time_s,
    )
    np.testing.assert_allclose(
        np.concatenate([left.decision_time_s, right.decision_time_s]),
        whole.decision_time_s,
    )


def test_explicit_reset_clears_refractory_and_comparator() -> None:
    trigger = CausalCecHTTrigger(make_config())
    _, first = run_trigger(trigger, [-0.5, 0.5])
    assert first.event.tolist() == [False, True]

    trigger.reset(PseudoTriggerReasonCode.EXPLICIT_RESET)
    _, second = run_trigger(trigger, [-0.5, 0.5], first_index=100)
    assert second.event.tolist() == [False, True]


def test_trigger_hot_path_has_no_forbidden_conversion_or_io() -> None:
    source = (
        ROOT
        / "causal_phase_tracking"
        / "causal_cecHT"
        / "pseudo_trigger.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    hot_functions = {
        "process_block_into",
        "_crossing_slots",
        "_schema_reason",
    }
    forbidden_calls = {
        "asarray",
        "astype",
        "append",
        "concatenate",
        "resize",
        "open",
        "print",
    }
    violations = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in hot_functions:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            name = (
                child.func.attr
                if isinstance(child.func, ast.Attribute)
                else child.func.id
                if isinstance(child.func, ast.Name)
                else ""
            )
            if name in forbidden_calls:
                violations.append((node.name, name, child.lineno))
    assert violations == []
