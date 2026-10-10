"""Fixed-capacity, strictly causal cecHT streaming core."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import math
from numbers import Integral, Real

import numpy as np
from scipy.fft import ifft as _constructor_ifft
from scipy.fft import ifftshift as _constructor_ifftshift

from .vendor.eikeosmers_cecht.phase import ECHT


_INT64_MAX = int(np.iinfo(np.int64).max)


class CecHTReasonCode(IntEnum):
    NONE = 0
    EXPLICIT_RESET = 1
    WARMUP = 2
    NONFINITE_INPUT = 3
    SAMPLE_GAP = 4
    DUPLICATE_SAMPLE = 5
    OUT_OF_ORDER_SAMPLE = 6
    SAMPLING_RATE_UNKNOWN = 7
    SAMPLING_RATE_MISMATCH = 8
    UNIT_UNKNOWN = 9
    UNIT_MISMATCH = 10
    CHANNEL_UNKNOWN = 11
    CHANNEL_MISMATCH = 12
    POLARITY_UNKNOWN = 13
    POLARITY_MISMATCH = 14
    NONFINITE_ENDPOINT = 15
    STRUCTURAL_INPUT_TYPE = 20
    STRUCTURAL_INPUT_NDIM = 21
    STRUCTURAL_INPUT_DTYPE = 22
    STRUCTURAL_INPUT_CONTIGUITY = 23
    STRUCTURAL_OUTPUT_SCHEMA = 24
    STRUCTURAL_ALIAS = 25
    STRUCTURAL_CAPACITY = 26
    STRUCTURAL_CONTEXT_SCHEMA = 27


class CecHTProcessStatus(IntEnum):
    OK = 0
    EMPTY = 1
    EPOCH_RESTARTED = 2
    STREAM_FAULT = 3
    STRUCTURAL_REJECT = 4


class CecHTVerifiedField(IntFlag):
    NONE = 0
    SAMPLING_RATE = 1
    UNIT = 2
    CHANNEL = 4
    POLARITY = 8
    ALL = SAMPLING_RATE | UNIT | CHANNEL | POLARITY


def wrap_phase_rad(angle: float) -> float:
    """Normalize one phase angle to the contract's half-open interval."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(slots=True)
class CecHTBlockContext:
    stream_epoch: int
    first_relative_index: int
    declared_sampling_rate_hz: float
    unit_code: int
    channel_code: int
    polarity_code: int
    verified_fields_mask: CecHTVerifiedField


@dataclass(slots=True)
class CecHTOutputBuffers:
    phase_rad: np.ndarray
    amplitude: np.ndarray
    valid: np.ndarray
    ready: np.ndarray
    reason_code: np.ndarray
    relative_sample_index: np.ndarray

    @classmethod
    def allocate(cls, capacity: int) -> "CecHTOutputBuffers":
        if capacity < 0:
            raise ValueError("capacity must not be negative")
        return cls(
            phase_rad=np.zeros(capacity, dtype=np.float64),
            amplitude=np.zeros(capacity, dtype=np.float64),
            valid=np.zeros(capacity, dtype=np.bool_),
            ready=np.zeros(capacity, dtype=np.bool_),
            reason_code=np.zeros(capacity, dtype=np.uint8),
            relative_sample_index=np.zeros(capacity, dtype=np.int64),
        )


@dataclass(frozen=True, slots=True)
class CecHTCoreConfig:
    sampling_rate_hz: float
    input_dtype: np.dtype
    numeric_unit_code: int
    channel_code: int
    polarity_code: int
    window_samples: int
    l_freq: float
    h_freq: float
    filter_order: int
    filter_type: str
    n_fft: int | None
    fft_mode: str
    calibrate: bool
    f0: float | None
    parameter_version: str
    config_version: str
    max_block_samples: int
    method_id: str = "cecHT"

    def __post_init__(self) -> None:
        dtype = np.dtype(self.input_dtype)
        if not np.issubdtype(dtype, np.floating):
            raise TypeError("input_dtype must be a real floating NumPy dtype")
        if dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise ValueError("input_dtype must be float32 or float64")
        object.__setattr__(self, "input_dtype", dtype)

        if isinstance(self.sampling_rate_hz, bool) or not isinstance(
            self.sampling_rate_hz, Real
        ):
            raise TypeError("sampling_rate_hz must be a real scalar")
        if not math.isclose(self.sampling_rate_hz, 250.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("sampling_rate_hz must be exactly 250.0")
        for name, value in (
            ("numeric_unit_code", self.numeric_unit_code),
            ("channel_code", self.channel_code),
            ("polarity_code", self.polarity_code),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integral identity code")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.window_samples, bool) or not isinstance(
            self.window_samples, Integral
        ):
            raise TypeError("window_samples must be an integer")
        if self.window_samples <= 0:
            raise ValueError("window_samples must be positive")
        for name, value in (("l_freq", self.l_freq), ("h_freq", self.h_freq)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real scalar")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not (0.0 < self.l_freq < self.h_freq < self.sampling_rate_hz / 2.0):
            raise ValueError("band edges must satisfy 0 < l_freq < h_freq < Nyquist")
        if isinstance(self.filter_order, bool) or not isinstance(
            self.filter_order, Integral
        ):
            raise TypeError("filter_order must be an integer")
        if self.filter_order <= 0:
            raise ValueError("filter_order must be positive")
        if self.filter_type not in {"butter", "bessel", "cheby1", "cheby2", "ellip"}:
            raise ValueError("filter_type is not supported by the pinned ECHT")
        if self.fft_mode not in {"fast", "exact"}:
            raise ValueError("fft_mode must be 'fast' or 'exact'")
        if self.n_fft is not None and (
            isinstance(self.n_fft, bool) or not isinstance(self.n_fft, Integral)
        ):
            raise TypeError("n_fft must be an integer or None")
        if self.n_fft is not None and self.n_fft < self.window_samples:
            raise ValueError("n_fft must be greater than or equal to window_samples")
        if not isinstance(self.calibrate, bool):
            raise TypeError("calibrate must be bool")
        if self.calibrate:
            if isinstance(self.f0, bool) or not isinstance(self.f0, Real):
                raise ValueError("finite f0 is required when calibrate=True")
            if not math.isfinite(self.f0) or not (
                0.0 < self.f0 < self.sampling_rate_hz / 2.0
            ):
                raise ValueError("f0 must be finite and between DC and Nyquist")
        elif self.f0 is not None:
            raise ValueError("f0 must be None when calibrate=False")
        for name, value in (
            ("parameter_version", self.parameter_version),
            ("config_version", self.config_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.method_id != "cecHT":
            raise ValueError("method_id must be 'cecHT'")
        if isinstance(self.max_block_samples, bool) or not isinstance(
            self.max_block_samples, Integral
        ):
            raise TypeError("max_block_samples must be an integer")
        if self.max_block_samples <= 0:
            raise ValueError("max_block_samples must be positive")


class CausalCecHTCore:
    """One-owner state container with a constructor-derived endpoint kernel."""

    def __init__(self, config: CecHTCoreConfig) -> None:
        self.config = config
        self._echt = ECHT(
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
        self._echt.fit(np.zeros(config.window_samples, dtype=config.input_dtype))
        self._resolved_n_fft = int(self._echt.n_fft)
        if self._resolved_n_fft < config.window_samples:
            raise ValueError("resolved n_fft must be at least window_samples")
        self._endpoint_kernel = self._derive_endpoint_kernel()
        self._ring = np.zeros(config.window_samples, dtype=config.input_dtype)
        self._finite_scratch = np.empty(config.max_block_samples, dtype=np.bool_)
        self._magnitude_scratch = np.empty(
            config.max_block_samples, dtype=config.input_dtype
        )
        self._dot_scratch = (
            np.zeros((), dtype=np.complex128),
            np.zeros((), dtype=np.complex128),
        )
        self.reset(CecHTReasonCode.EXPLICIT_RESET)

    @property
    def endpoint_kernel(self) -> np.ndarray:
        inspection_copy = self._endpoint_kernel.copy()
        inspection_copy.flags.writeable = False
        return inspection_copy

    @property
    def resolved_n_fft(self) -> int:
        return self._resolved_n_fft

    @property
    def last_reason_code(self) -> CecHTReasonCode:
        return self._last_reason_code

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def reset(self, reason: CecHTReasonCode) -> None:
        self._ring.fill(0)
        self._write_pos = 0
        self._sample_count = 0
        self._stream_epoch: int | None = None
        self._expected_relative_index: int | None = None
        self._last_reason_code = CecHTReasonCode(reason)

    def process_block_into(
        self,
        samples: np.ndarray,
        block_context: CecHTBlockContext,
        output_buffers: CecHTOutputBuffers,
    ) -> CecHTProcessStatus:
        structural_reason = self._structural_reason(
            samples, block_context, output_buffers
        )
        if structural_reason is not CecHTReasonCode.NONE:
            self.reset(structural_reason)
            return CecHTProcessStatus.STRUCTURAL_REJECT

        sample_total = samples.size
        if sample_total == 0:
            return CecHTProcessStatus.EMPTY

        stream_reason = self._stream_fault_reason(samples, block_context)
        if stream_reason is not CecHTReasonCode.NONE:
            self._write_invalid_block(
                output_buffers, block_context.first_relative_index, sample_total, stream_reason
            )
            self.reset(stream_reason)
            return CecHTProcessStatus.STREAM_FAULT

        epoch_restarted = (
            self._stream_epoch is not None
            and self._stream_epoch != block_context.stream_epoch
        )
        if epoch_restarted:
            self.reset(CecHTReasonCode.EXPLICIT_RESET)

        self._stream_epoch = block_context.stream_epoch
        for offset in range(sample_total):
            relative_index = block_context.first_relative_index + offset
            self._ring[self._write_pos] = samples[offset]
            self._write_pos = (self._write_pos + 1) % self.config.window_samples
            if self._sample_count < self.config.window_samples:
                self._sample_count += 1

            output_buffers.relative_sample_index[offset] = relative_index
            if self._sample_count < self.config.window_samples:
                output_buffers.phase_rad[offset] = 0.0
                output_buffers.amplitude[offset] = 0.0
                output_buffers.valid[offset] = False
                output_buffers.ready[offset] = False
                output_buffers.reason_code[offset] = CecHTReasonCode.WARMUP
                continue

            endpoint = self._endpoint_from_ring()
            endpoint_real = float(endpoint.real)
            endpoint_imag = float(endpoint.imag)
            amplitude = math.hypot(endpoint_real, endpoint_imag)
            if not (
                math.isfinite(endpoint_real)
                and math.isfinite(endpoint_imag)
                and math.isfinite(amplitude)
            ):
                reason = CecHTReasonCode.NONFINITE_ENDPOINT
                self._write_invalid_block(
                    output_buffers,
                    block_context.first_relative_index,
                    sample_total,
                    reason,
                )
                self.reset(reason)
                return CecHTProcessStatus.STREAM_FAULT
            output_buffers.phase_rad[offset] = wrap_phase_rad(
                math.atan2(endpoint_imag, endpoint_real)
            )
            output_buffers.amplitude[offset] = amplitude
            output_buffers.valid[offset] = True
            output_buffers.ready[offset] = True
            output_buffers.reason_code[offset] = CecHTReasonCode.NONE

        self._expected_relative_index = block_context.first_relative_index + sample_total
        self._last_reason_code = CecHTReasonCode.NONE
        if epoch_restarted:
            return CecHTProcessStatus.EPOCH_RESTARTED
        return CecHTProcessStatus.OK

    def _stream_fault_reason(
        self, samples: np.ndarray, context: CecHTBlockContext
    ) -> CecHTReasonCode:
        mask = context.verified_fields_mask
        if not (mask & CecHTVerifiedField.SAMPLING_RATE):
            return CecHTReasonCode.SAMPLING_RATE_UNKNOWN
        if context.declared_sampling_rate_hz != self.config.sampling_rate_hz:
            return CecHTReasonCode.SAMPLING_RATE_MISMATCH
        if not (mask & CecHTVerifiedField.UNIT):
            return CecHTReasonCode.UNIT_UNKNOWN
        if context.unit_code != self.config.numeric_unit_code:
            return CecHTReasonCode.UNIT_MISMATCH
        if not (mask & CecHTVerifiedField.CHANNEL):
            return CecHTReasonCode.CHANNEL_UNKNOWN
        if context.channel_code != self.config.channel_code:
            return CecHTReasonCode.CHANNEL_MISMATCH
        if not (mask & CecHTVerifiedField.POLARITY):
            return CecHTReasonCode.POLARITY_UNKNOWN
        if context.polarity_code != self.config.polarity_code:
            return CecHTReasonCode.POLARITY_MISMATCH

        np.isfinite(samples, out=self._finite_scratch[: samples.size])
        if not self._finite_scratch[: samples.size].all():
            return CecHTReasonCode.NONFINITE_INPUT
        np.abs(samples, out=self._magnitude_scratch[: samples.size])
        if self._magnitude_scratch[: samples.size].max() > (
            self._max_safe_input_magnitude
        ):
            return CecHTReasonCode.NONFINITE_ENDPOINT

        if (
            self._stream_epoch is None
            or context.stream_epoch != self._stream_epoch
        ):
            return CecHTReasonCode.NONE
        expected = self._expected_relative_index
        if context.first_relative_index == expected:
            return CecHTReasonCode.NONE
        if context.first_relative_index > expected:
            return CecHTReasonCode.SAMPLE_GAP
        if context.first_relative_index + samples.size == expected:
            return CecHTReasonCode.DUPLICATE_SAMPLE
        return CecHTReasonCode.OUT_OF_ORDER_SAMPLE

    @staticmethod
    def _write_invalid_block(
        outputs: CecHTOutputBuffers,
        first_relative_index: int,
        sample_total: int,
        reason: CecHTReasonCode,
    ) -> None:
        outputs.phase_rad[:sample_total].fill(0.0)
        outputs.amplitude[:sample_total].fill(0.0)
        outputs.valid[:sample_total].fill(False)
        outputs.ready[:sample_total].fill(False)
        outputs.reason_code[:sample_total].fill(reason)
        for offset in range(sample_total):
            outputs.relative_sample_index[offset] = first_relative_index + offset

    def _endpoint_from_ring(self) -> complex:
        oldest = self._write_pos
        tail_size = self.config.window_samples - oldest
        np.dot(
            self._endpoint_kernel[:tail_size],
            self._ring[oldest:],
            out=self._dot_scratch[0],
        )
        if oldest:
            np.dot(
                self._endpoint_kernel[tail_size:],
                self._ring[:oldest],
                out=self._dot_scratch[1],
            )
            self._dot_scratch[0][...] += self._dot_scratch[1]
        else:
            self._dot_scratch[1][...] = 0.0j
        return complex(self._dot_scratch[0])

    def _structural_reason(
        self, samples: object, context: object, outputs: object
    ) -> CecHTReasonCode:
        if not isinstance(samples, np.ndarray):
            return CecHTReasonCode.STRUCTURAL_INPUT_TYPE
        if samples.ndim != 1:
            return CecHTReasonCode.STRUCTURAL_INPUT_NDIM
        if samples.dtype != self.config.input_dtype or not np.issubdtype(
            samples.dtype, np.floating
        ):
            return CecHTReasonCode.STRUCTURAL_INPUT_DTYPE
        if not samples.flags.c_contiguous:
            return CecHTReasonCode.STRUCTURAL_INPUT_CONTIGUITY
        if not isinstance(outputs, CecHTOutputBuffers):
            return CecHTReasonCode.STRUCTURAL_OUTPUT_SCHEMA
        if not self._output_schema_is_valid(outputs):
            return CecHTReasonCode.STRUCTURAL_OUTPUT_SCHEMA
        if samples.size > self.config.max_block_samples:
            return CecHTReasonCode.STRUCTURAL_CAPACITY
        if outputs.phase_rad.size < samples.size:
            return CecHTReasonCode.STRUCTURAL_CAPACITY
        if self._has_alias(samples, outputs):
            return CecHTReasonCode.STRUCTURAL_ALIAS
        if not self._context_schema_is_valid(context, samples.size):
            return CecHTReasonCode.STRUCTURAL_CONTEXT_SCHEMA
        return CecHTReasonCode.NONE

    @staticmethod
    def _context_schema_is_valid(context: object, sample_total: int) -> bool:
        if not isinstance(context, CecHTBlockContext):
            return False
        if (
            isinstance(context.stream_epoch, bool)
            or not isinstance(context.stream_epoch, Integral)
            or context.stream_epoch < 0
        ):
            return False
        if (
            isinstance(context.first_relative_index, bool)
            or not isinstance(context.first_relative_index, Integral)
            or context.first_relative_index < 0
        ):
            return False
        if context.first_relative_index > _INT64_MAX - max(0, sample_total - 1):
            return False
        if isinstance(context.declared_sampling_rate_hz, bool) or not isinstance(
            context.declared_sampling_rate_hz, Real
        ):
            return False
        for value in (context.unit_code, context.channel_code, context.polarity_code):
            if isinstance(value, bool) or not isinstance(value, Integral):
                return False
        if not isinstance(context.verified_fields_mask, CecHTVerifiedField):
            return False
        if int(context.verified_fields_mask) & ~int(CecHTVerifiedField.ALL):
            return False
        return True

    @staticmethod
    def _valid_output_array(array: object, dtype: np.dtype) -> bool:
        return (
            isinstance(array, np.ndarray)
            and array.ndim == 1
            and array.dtype == dtype
            and array.flags.c_contiguous
            and array.flags.writeable
        )

    def _output_schema_is_valid(self, outputs: CecHTOutputBuffers) -> bool:
        capacity = outputs.phase_rad.size
        return (
            self._valid_output_array(outputs.phase_rad, np.dtype(np.float64))
            and self._valid_output_array(outputs.amplitude, np.dtype(np.float64))
            and self._valid_output_array(outputs.valid, np.dtype(np.bool_))
            and self._valid_output_array(outputs.ready, np.dtype(np.bool_))
            and self._valid_output_array(outputs.reason_code, np.dtype(np.uint8))
            and self._valid_output_array(
                outputs.relative_sample_index, np.dtype(np.int64)
            )
            and outputs.amplitude.size == capacity
            and outputs.valid.size == capacity
            and outputs.ready.size == capacity
            and outputs.reason_code.size == capacity
            and outputs.relative_sample_index.size == capacity
        )

    @staticmethod
    def _has_alias(samples: np.ndarray, outputs: CecHTOutputBuffers) -> bool:
        if np.shares_memory(samples, outputs.phase_rad):
            return True
        if np.shares_memory(samples, outputs.amplitude):
            return True
        if np.shares_memory(samples, outputs.valid):
            return True
        if np.shares_memory(samples, outputs.ready):
            return True
        if np.shares_memory(samples, outputs.reason_code):
            return True
        if np.shares_memory(samples, outputs.relative_sample_index):
            return True

        arrays = (
            outputs.phase_rad,
            outputs.amplitude,
            outputs.valid,
            outputs.ready,
            outputs.reason_code,
            outputs.relative_sample_index,
        )
        for left in range(len(arrays)):
            for right in range(left + 1, len(arrays)):
                if np.shares_memory(arrays[left], arrays[right]):
                    return True
        return False

    def _derive_endpoint_kernel(self) -> np.ndarray:
        n_samples = self.config.window_samples
        spectrum = self._echt.h_ * _constructor_ifftshift(
            self._echt.coef_.ravel()
        )
        impulse_response = _constructor_ifft(spectrum)
        indices = (n_samples - 1 - np.arange(n_samples)) % self._resolved_n_fft
        kernel = np.asarray(impulse_response[indices], dtype=np.complex128)
        if self._echt.calibrate and self._echt.calib_gain_ is not None:
            kernel *= self._echt.calib_gain_
        kernel_l1 = float(np.sum(np.abs(kernel)))
        float64_max = np.finfo(np.float64).max
        if kernel_l1 > 1.0:
            self._max_safe_input_magnitude = np.nextafter(
                float64_max / kernel_l1, 0.0
            )
        else:
            self._max_safe_input_magnitude = np.nextafter(float64_max, 0.0)
        kernel.flags.writeable = False
        return kernel
