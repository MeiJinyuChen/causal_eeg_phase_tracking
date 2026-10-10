"""Independent configurable positive-crossing pseudo-trigger for causal cecHT."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Integral, Real

import numpy as np

from .streaming_core import wrap_phase_rad


class PseudoTriggerStatus(IntEnum):
    OK = 0
    EMPTY = 1
    STRUCTURAL_REJECT = 2


class PseudoTriggerReasonCode(IntEnum):
    NONE = 0
    EXPLICIT_RESET = 1
    TARGET_UPDATED = 2
    NOT_READY = 3
    NONFINITE_PHASE = 4
    STRUCTURAL_INPUT_TYPE = 10
    STRUCTURAL_INPUT_NDIM = 11
    STRUCTURAL_INPUT_DTYPE = 12
    STRUCTURAL_INPUT_CONTIGUITY = 13
    STRUCTURAL_INPUT_LENGTH_MISMATCH = 14
    STRUCTURAL_OUTPUT_SCHEMA = 15
    STRUCTURAL_ALIAS = 16
    STRUCTURAL_CAPACITY = 17


@dataclass(frozen=True, slots=True)
class PseudoTriggerConfig:
    sampling_rate_hz: float
    target_phase_rad: float
    refractory_s: float
    parameter_version: str
    config_version: str
    max_block_samples: int

    def __post_init__(self) -> None:
        if isinstance(self.sampling_rate_hz, bool) or not isinstance(
            self.sampling_rate_hz, Real
        ):
            raise TypeError("sampling_rate_hz must be a real scalar")
        if not math.isclose(
            self.sampling_rate_hz, 250.0, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("sampling_rate_hz must be exactly 250.0")
        if isinstance(self.target_phase_rad, bool) or not isinstance(
            self.target_phase_rad, Real
        ):
            raise TypeError("target_phase_rad must be a real scalar")
        if not math.isfinite(self.target_phase_rad):
            raise ValueError("target_phase_rad must be finite")
        if not (-math.pi <= self.target_phase_rad < math.pi):
            raise ValueError("target_phase_rad must be in [-pi, pi)")
        if isinstance(self.refractory_s, bool) or not isinstance(
            self.refractory_s, Real
        ):
            raise TypeError("refractory_s must be a real scalar")
        if not math.isfinite(self.refractory_s) or self.refractory_s < 0.0:
            raise ValueError("refractory_s must be finite and non-negative")
        for name, value in (
            ("parameter_version", self.parameter_version),
            ("config_version", self.config_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.max_block_samples, bool) or not isinstance(
            self.max_block_samples, Integral
        ):
            raise TypeError("max_block_samples must be an integer")
        if self.max_block_samples <= 0:
            raise ValueError("max_block_samples must be positive")


@dataclass(slots=True)
class PseudoTriggerOutputBuffers:
    event: np.ndarray
    crossing_time_s: np.ndarray
    decision_time_s: np.ndarray

    @classmethod
    def allocate(cls, capacity: int) -> "PseudoTriggerOutputBuffers":
        if capacity < 0:
            raise ValueError("capacity must not be negative")
        return cls(
            event=np.zeros(capacity, dtype=np.bool_),
            crossing_time_s=np.zeros(capacity, dtype=np.float64),
            decision_time_s=np.zeros(capacity, dtype=np.float64),
        )


class CausalCecHTTrigger:
    """Stateful positive-crossing policy consuming valid/ready phase endpoints."""

    def __init__(self, config: PseudoTriggerConfig) -> None:
        self.config = config
        self._target_phase_rad = config.target_phase_rad
        self.reset(PseudoTriggerReasonCode.EXPLICIT_RESET)

    @property
    def target_phase_rad(self) -> float:
        return self._target_phase_rad

    @property
    def last_reason_code(self) -> PseudoTriggerReasonCode:
        return self._last_reason_code

    def set_target_phase_rad(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("target_phase_rad must be a real scalar")
        if not math.isfinite(value) or not (-math.pi <= value < math.pi):
            raise ValueError("target_phase_rad must be finite and in [-pi, pi)")
        self._target_phase_rad = float(value)
        self._clear_comparator(PseudoTriggerReasonCode.TARGET_UPDATED)

    def reset(self, reason: PseudoTriggerReasonCode) -> None:
        self._prev_phase_rad: float | None = None
        self._prev_time_s: float | None = None
        self._next_allowed_time_s = 0.0
        self._last_reason_code = PseudoTriggerReasonCode(reason)

    def process_block_into(
        self,
        relative_sample_index: np.ndarray,
        phase_rad: np.ndarray,
        valid: np.ndarray,
        ready: np.ndarray,
        outputs: PseudoTriggerOutputBuffers,
    ) -> PseudoTriggerStatus:
        reason = self._schema_reason(
            relative_sample_index, phase_rad, valid, ready, outputs
        )
        if reason is not PseudoTriggerReasonCode.NONE:
            self.reset(reason)
            return PseudoTriggerStatus.STRUCTURAL_REJECT

        sample_total = relative_sample_index.size
        if sample_total == 0:
            return PseudoTriggerStatus.EMPTY

        sample_interval_s = 1.0 / self.config.sampling_rate_hz
        for offset in range(sample_total):
            outputs.event[offset] = False
            outputs.crossing_time_s[offset] = 0.0
            outputs.decision_time_s[offset] = 0.0

            if not (valid[offset] and ready[offset]):
                self._clear_comparator(PseudoTriggerReasonCode.NOT_READY)
                continue

            phase = float(phase_rad[offset])
            if not math.isfinite(phase):
                self._clear_comparator(PseudoTriggerReasonCode.NONFINITE_PHASE)
                continue

            current_time_s = (
                float(relative_sample_index[offset])
                / self.config.sampling_rate_hz
            )
            if self._prev_phase_rad is None or self._prev_time_s is None:
                self._prev_phase_rad = phase
                self._prev_time_s = current_time_s
                continue

            delta = wrap_phase_rad(phase - self._prev_phase_rad)
            if delta <= 0.0:
                self._prev_phase_rad = phase
                self._prev_time_s = current_time_s
                continue

            target_unwrapped = self._prev_phase_rad + wrap_phase_rad(
                self._target_phase_rad - self._prev_phase_rad
            )
            crossed = (
                self._prev_phase_rad
                <= target_unwrapped
                < self._prev_phase_rad + delta
            )
            if not crossed:
                self._prev_phase_rad = phase
                self._prev_time_s = current_time_s
                continue

            fraction = (target_unwrapped - self._prev_phase_rad) / delta
            crossing_time_s = self._prev_time_s + fraction * sample_interval_s
            decision_time_s = current_time_s
            if crossing_time_s >= self._next_allowed_time_s:
                outputs.event[offset] = True
                outputs.crossing_time_s[offset] = crossing_time_s
                outputs.decision_time_s[offset] = decision_time_s
                self._next_allowed_time_s = (
                    crossing_time_s + self.config.refractory_s
                )

            self._prev_phase_rad = phase
            self._prev_time_s = current_time_s

        self._last_reason_code = PseudoTriggerReasonCode.NONE
        return PseudoTriggerStatus.OK

    def _clear_comparator(self, reason: PseudoTriggerReasonCode) -> None:
        self._prev_phase_rad = None
        self._prev_time_s = None
        self._last_reason_code = PseudoTriggerReasonCode(reason)

    def _schema_reason(
        self,
        index: object,
        phase: object,
        valid: object,
        ready: object,
        outputs: object,
    ) -> PseudoTriggerReasonCode:
        for value in (index, phase, valid, ready):
            if not isinstance(value, np.ndarray):
                return PseudoTriggerReasonCode.STRUCTURAL_INPUT_TYPE
        if index.ndim != 1 or phase.ndim != 1 or valid.ndim != 1 or ready.ndim != 1:
            return PseudoTriggerReasonCode.STRUCTURAL_INPUT_NDIM
        if index.dtype != np.dtype(np.int64):
            return PseudoTriggerReasonCode.STRUCTURAL_INPUT_DTYPE
        if phase.dtype != np.dtype(np.float64):
            return PseudoTriggerReasonCode.STRUCTURAL_INPUT_DTYPE
        if valid.dtype != np.dtype(np.bool_) or ready.dtype != np.dtype(np.bool_):
            return PseudoTriggerReasonCode.STRUCTURAL_INPUT_DTYPE
        if not (
            index.flags.c_contiguous
            and phase.flags.c_contiguous
            and valid.flags.c_contiguous
            and ready.flags.c_contiguous
        ):
            return PseudoTriggerReasonCode.STRUCTURAL_INPUT_CONTIGUITY
        if not (
            index.size == phase.size == valid.size == ready.size
        ):
            return PseudoTriggerReasonCode.STRUCTURAL_INPUT_LENGTH_MISMATCH
        if index.size > self.config.max_block_samples:
            return PseudoTriggerReasonCode.STRUCTURAL_CAPACITY
        if not isinstance(outputs, PseudoTriggerOutputBuffers):
            return PseudoTriggerReasonCode.STRUCTURAL_OUTPUT_SCHEMA
        if not self._output_schema_is_valid(outputs):
            return PseudoTriggerReasonCode.STRUCTURAL_OUTPUT_SCHEMA
        if outputs.event.size < index.size:
            return PseudoTriggerReasonCode.STRUCTURAL_CAPACITY
        if self._has_alias(index, phase, valid, ready, outputs):
            return PseudoTriggerReasonCode.STRUCTURAL_ALIAS
        return PseudoTriggerReasonCode.NONE

    @staticmethod
    def _output_schema_is_valid(outputs: PseudoTriggerOutputBuffers) -> bool:
        return (
            isinstance(outputs.event, np.ndarray)
            and outputs.event.ndim == 1
            and outputs.event.dtype == np.dtype(np.bool_)
            and outputs.event.flags.c_contiguous
            and outputs.event.flags.writeable
            and isinstance(outputs.crossing_time_s, np.ndarray)
            and outputs.crossing_time_s.ndim == 1
            and outputs.crossing_time_s.dtype == np.dtype(np.float64)
            and outputs.crossing_time_s.flags.c_contiguous
            and outputs.crossing_time_s.flags.writeable
            and isinstance(outputs.decision_time_s, np.ndarray)
            and outputs.decision_time_s.ndim == 1
            and outputs.decision_time_s.dtype == np.dtype(np.float64)
            and outputs.decision_time_s.flags.c_contiguous
            and outputs.decision_time_s.flags.writeable
            and outputs.event.size == outputs.crossing_time_s.size
            and outputs.event.size == outputs.decision_time_s.size
        )

    @staticmethod
    def _has_alias(
        index: np.ndarray,
        phase: np.ndarray,
        valid: np.ndarray,
        ready: np.ndarray,
        outputs: PseudoTriggerOutputBuffers,
    ) -> bool:
        inputs = (index, phase, valid, ready)
        output_arrays = (
            outputs.event,
            outputs.crossing_time_s,
            outputs.decision_time_s,
        )
        for left in inputs:
            for right in output_arrays:
                if np.shares_memory(left, right):
                    return True
        for left in range(len(output_arrays)):
            for right in range(left + 1, len(output_arrays)):
                if np.shares_memory(output_arrays[left], output_arrays[right]):
                    return True
        return False
