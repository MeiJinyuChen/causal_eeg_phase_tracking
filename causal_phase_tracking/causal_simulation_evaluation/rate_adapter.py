"""Strictly causal rational 100 Hz -> 250 Hz streaming adapter.

The adapter uses a fixed linear-phase FIR kernel in the upsampled domain and an
explicit polyphase decomposition.  Samples are never shifted internally: the
adapter only reports its exact group delay, which the offline reference and
event-time layer subtract once.  See HANDOFF S7-D1 to S7-D3.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Integral, Real

import numpy as np
from scipy.signal import firwin


class RationalRateAdapterStatus(IntEnum):
    OK = 0
    EMPTY = 1
    STREAM_FAULT = 2
    STRUCTURAL_REJECT = 3


class RationalRateReasonCode(IntEnum):
    NONE = 0
    EXPLICIT_RESET = 1
    WARMUP = 2
    NONFINITE_INPUT = 3
    INPUT_GAP = 4
    DUPLICATE_INPUT = 5
    OUT_OF_ORDER_INPUT = 6
    STRUCTURAL_INPUT_TYPE = 10
    STRUCTURAL_INPUT_NDIM = 11
    STRUCTURAL_INPUT_DTYPE = 12
    STRUCTURAL_INPUT_CONTIGUITY = 13
    STRUCTURAL_INPUT_INDEX = 14
    STRUCTURAL_OUTPUT_SCHEMA = 15
    STRUCTURAL_ALIAS = 16
    STRUCTURAL_CAPACITY = 17


@dataclass(frozen=True, slots=True)
class RationalRateAdapterConfig:
    input_rate_hz: float
    output_rate_hz: float
    upsample: int
    downsample: int
    kernel_rate_hz: float
    cutoff_hz: float
    stopband_hz: float
    kernel_taps: int
    max_input_samples: int
    config_version: str

    def __post_init__(self) -> None:
        if not math.isclose(
            self.input_rate_hz, 100.0, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("input_rate_hz must be exactly 100.0")
        if not math.isclose(
            self.output_rate_hz, 250.0, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("output_rate_hz must be exactly 250.0")
        if self.upsample != 5 or self.downsample != 2:
            raise ValueError("upsample/downsample must be exactly 5 and 2")
        if not math.isclose(
            self.kernel_rate_hz,
            self.input_rate_hz * self.upsample,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("kernel_rate_hz must equal input_rate_hz * upsample")
        if (
            isinstance(self.cutoff_hz, bool)
            or isinstance(self.stopband_hz, bool)
            or not math.isfinite(self.cutoff_hz)
            or not math.isfinite(self.stopband_hz)
        ):
            raise ValueError("cutoff_hz and stopband_hz must be finite scalars")
        if not (0.0 < self.cutoff_hz < self.stopband_hz < self.kernel_rate_hz / 2.0):
            raise ValueError("require 0 < cutoff_hz < stopband_hz < kernel Nyquist")
        if (
            isinstance(self.kernel_taps, bool)
            or not isinstance(self.kernel_taps, Integral)
            or self.kernel_taps <= 0
            or self.kernel_taps % 2 == 0
        ):
            raise ValueError("kernel_taps must be a positive odd integer")
        if (
            isinstance(self.max_input_samples, bool)
            or not isinstance(self.max_input_samples, Integral)
            or self.max_input_samples <= 0
        ):
            raise ValueError("max_input_samples must be a positive integer")
        if not isinstance(self.config_version, str) or not self.config_version.strip():
            raise ValueError("config_version must be a non-empty string")


@dataclass(slots=True)
class RationalRateOutputBuffers:
    resampled: np.ndarray
    valid: np.ndarray
    reason_code: np.ndarray
    output_index: np.ndarray

    @classmethod
    def allocate(cls, capacity: int) -> "RationalRateOutputBuffers":
        if capacity < 0:
            raise ValueError("capacity must not be negative")
        return cls(
            resampled=np.zeros(capacity, dtype=np.float64),
            valid=np.zeros(capacity, dtype=np.bool_),
            reason_code=np.zeros(capacity, dtype=np.uint8),
            output_index=np.zeros(capacity, dtype=np.int64),
        )


class Causal100To250Adapter:
    """One-owner, fixed-capacity, strictly causal 100 -> 250 resampler."""

    def __init__(self, config: RationalRateAdapterConfig) -> None:
        self.config = config
        kernel = firwin(
            config.kernel_taps,
            config.cutoff_hz,
            fs=config.kernel_rate_hz,
            pass_zero="lowpass",
        ).astype(np.float64)
        taps_per_phase = int(math.ceil(config.kernel_taps / config.upsample))
        self._phase_kernels = np.zeros(
            (config.upsample, taps_per_phase), dtype=np.float64
        )
        for phase in range(config.upsample):
            for tap in range(taps_per_phase):
                position = phase + tap * config.upsample
                if position < config.kernel_taps:
                    self._phase_kernels[phase, tap] = kernel[position]
        self._kernel = kernel
        self._history = np.zeros(taps_per_phase, dtype=np.float64)
        self._finite_scratch = np.empty(config.max_input_samples, dtype=np.bool_)
        self._dot_scratch = np.zeros((), dtype=np.float64)
        self.reset(RationalRateReasonCode.EXPLICIT_RESET)

    @property
    def group_delay_output_samples(self) -> int:
        return (self.config.kernel_taps - 1) // (2 * self.config.downsample)

    @property
    def group_delay_s(self) -> float:
        return self.group_delay_output_samples / self.config.output_rate_hz

    @property
    def last_reason_code(self) -> RationalRateReasonCode:
        return self._last_reason_code

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def expected_input_index(self) -> int | None:
        return self._expected_input_index

    @staticmethod
    def output_count_for(input_samples: int) -> int:
        return (5 * input_samples + 1) // 2

    @staticmethod
    def output_count_for_range(first_input_index: int, input_samples: int) -> int:
        start = (5 * first_input_index + 1) // 2
        end = (5 * (first_input_index + input_samples) + 1) // 2
        return end - start

    def allocate_outputs(self, capacity: int) -> RationalRateOutputBuffers:
        return RationalRateOutputBuffers.allocate(capacity)

    def reset(self, reason: RationalRateReasonCode) -> None:
        self._history.fill(0.0)
        self._output_index = 0
        self._outputs_since_reset = 0
        self._sample_count = 0
        self._expected_input_index: int | None = None
        self._last_reason_code = RationalRateReasonCode(reason)

    def process_block_into(
        self,
        samples: np.ndarray,
        first_input_index: int,
        outputs: RationalRateOutputBuffers,
    ) -> RationalRateAdapterStatus:
        structural = self._structural_reason(samples, first_input_index, outputs)
        if structural is not RationalRateReasonCode.NONE:
            self.reset(structural)
            return RationalRateAdapterStatus.STRUCTURAL_REJECT

        sample_total = samples.size
        if sample_total == 0:
            return RationalRateAdapterStatus.EMPTY

        fault = self._stream_fault_reason(samples, first_input_index)
        if fault is not RationalRateReasonCode.NONE:
            self._write_invalid_block(
                outputs, first_input_index, sample_total, fault
            )
            self.reset(fault)
            return RationalRateAdapterStatus.STREAM_FAULT

        cursor = 0
        for offset in range(sample_total):
            input_index = first_input_index + offset
            self._history[1:] = self._history[:-1]
            self._history[0] = samples[offset]
            m_start = (5 * input_index + 1) // 2
            m_end = (5 * (input_index + 1) + 1) // 2 - 1
            for output_index in range(m_start, m_end + 1):
                phase = (output_index * self.config.downsample) % self.config.upsample
                np.dot(
                    self._phase_kernels[phase],
                    self._history,
                    out=self._dot_scratch,
                )
                outputs.resampled[cursor] = float(self._dot_scratch)
                outputs.output_index[cursor] = output_index
                ready = self._outputs_since_reset >= self.group_delay_output_samples
                outputs.valid[cursor] = ready
                outputs.reason_code[cursor] = (
                    RationalRateReasonCode.NONE
                    if ready
                    else RationalRateReasonCode.WARMUP
                )
                cursor += 1
                self._outputs_since_reset += 1

        self._output_index += cursor
        self._sample_count += sample_total
        self._expected_input_index = first_input_index + sample_total
        self._last_reason_code = RationalRateReasonCode.NONE
        return RationalRateAdapterStatus.OK

    def _stream_fault_reason(
        self, samples: np.ndarray, first_input_index: int
    ) -> RationalRateReasonCode:
        np.isfinite(samples, out=self._finite_scratch[: samples.size])
        if not self._finite_scratch[: samples.size].all():
            return RationalRateReasonCode.NONFINITE_INPUT
        if self._expected_input_index is None:
            return RationalRateReasonCode.NONE
        if first_input_index == self._expected_input_index:
            return RationalRateReasonCode.NONE
        if first_input_index > self._expected_input_index:
            return RationalRateReasonCode.INPUT_GAP
        if first_input_index + samples.size == self._expected_input_index:
            return RationalRateReasonCode.DUPLICATE_INPUT
        return RationalRateReasonCode.OUT_OF_ORDER_INPUT

    def _write_invalid_block(
        self,
        outputs: RationalRateOutputBuffers,
        first_input_index: int,
        sample_total: int,
        reason: RationalRateReasonCode,
    ) -> None:
        count = self.output_count_for_range(first_input_index, sample_total)
        m_start = (5 * first_input_index + 1) // 2
        outputs.resampled[:count].fill(0.0)
        outputs.valid[:count].fill(False)
        outputs.reason_code[:count].fill(reason)
        for offset in range(count):
            outputs.output_index[offset] = m_start + offset

    def _structural_reason(
        self,
        samples: object,
        first_input_index: object,
        outputs: object,
    ) -> RationalRateReasonCode:
        if not isinstance(samples, np.ndarray):
            return RationalRateReasonCode.STRUCTURAL_INPUT_TYPE
        if samples.ndim != 1:
            return RationalRateReasonCode.STRUCTURAL_INPUT_NDIM
        if samples.dtype != np.dtype(np.float64):
            return RationalRateReasonCode.STRUCTURAL_INPUT_DTYPE
        if not samples.flags.c_contiguous:
            return RationalRateReasonCode.STRUCTURAL_INPUT_CONTIGUITY
        if (
            isinstance(first_input_index, bool)
            or not isinstance(first_input_index, Integral)
            or first_input_index < 0
        ):
            return RationalRateReasonCode.STRUCTURAL_INPUT_INDEX
        if samples.size > self.config.max_input_samples:
            return RationalRateReasonCode.STRUCTURAL_CAPACITY
        if not isinstance(outputs, RationalRateOutputBuffers):
            return RationalRateReasonCode.STRUCTURAL_OUTPUT_SCHEMA
        if not self._output_schema_is_valid(outputs):
            return RationalRateReasonCode.STRUCTURAL_OUTPUT_SCHEMA
        if outputs.resampled.size < self.output_count_for_range(
            first_input_index, samples.size
        ):
            return RationalRateReasonCode.STRUCTURAL_CAPACITY
        if self._has_alias(samples, outputs):
            return RationalRateReasonCode.STRUCTURAL_ALIAS
        return RationalRateReasonCode.NONE

    @staticmethod
    def _valid_array(array: object, dtype: np.dtype) -> bool:
        return (
            isinstance(array, np.ndarray)
            and array.ndim == 1
            and array.dtype == dtype
            and array.flags.c_contiguous
            and array.flags.writeable
        )

    def _output_schema_is_valid(self, outputs: RationalRateOutputBuffers) -> bool:
        return (
            self._valid_array(outputs.resampled, np.dtype(np.float64))
            and self._valid_array(outputs.valid, np.dtype(np.bool_))
            and self._valid_array(outputs.reason_code, np.dtype(np.uint8))
            and self._valid_array(outputs.output_index, np.dtype(np.int64))
            and outputs.resampled.size == outputs.valid.size
            and outputs.resampled.size == outputs.reason_code.size
            and outputs.resampled.size == outputs.output_index.size
        )

    @staticmethod
    def _has_alias(samples: np.ndarray, outputs: RationalRateOutputBuffers) -> bool:
        arrays = (
            outputs.resampled,
            outputs.valid,
            outputs.reason_code,
            outputs.output_index,
        )
        for array in arrays:
            if np.shares_memory(samples, array):
                return True
        for left in range(len(arrays)):
            for right in range(left + 1, len(arrays)):
                if np.shares_memory(arrays[left], arrays[right]):
                    return True
        return False
