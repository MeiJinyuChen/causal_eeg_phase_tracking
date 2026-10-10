"""CLAS 因果 cecHT 实时相位估计（host 外部算法，仅估计相位，不发刺激）。

Generated single-file plugin.  Do not edit by hand: rebuild with
``realtime_deployment/deploy/PhaseTracking_cecHT/build_PhaseTracking_cecHT.py``.

Frozen deployment configuration: 250 Hz, 8 s window (2000 samples),
``n_fft = N`` with ``fft_mode='exact'``, Butterworth order 2 band 0.5-1.25 Hz,
calibrated at f0 = 0.8 Hz, pseudo-trigger target 0 rad with 0.5 s refractory.

The waveform returned to the host is the input block unchanged; the estimated
phase lives in the plugin's in-memory ring buffer and side-channel log, because
the host's ``process()`` contract only accepts a same-shape waveform.
"""

# ---------------------------------------------------------------------
# GENERATED FILE - do not edit by hand.
# Source SHA-256 at build time:
#   phase.py         49b85cfd75f382aa9b12e00fca4cbfd903bf4daad0101ccba42b162f2af27bea  causal_phase_tracking/causal_cecHT/vendor/eikeosmers_cecht/phase.py
#   streaming_core.py 641e2cf1f47a5c1426a2bddda98831f58b13ee06d88b01c810e60ba620b39525  causal_phase_tracking/causal_cecHT/streaming_core.py
#   pseudo_trigger.py be7fabcb3adf1d581082681a5f90910516707c2e46fc20aa78326b6fc91cc4f9  causal_phase_tracking/causal_cecHT/pseudo_trigger.py
#   wrapper          0cb5b04836a46708aa4792749c9ab55ea5aa2b419c9b09d955c2c3ef6fc0427c  realtime_deployment/deploy/PhaseTracking_cecHT/PhaseTracking_cecHT_wrapper.py
# ---------------------------------------------------------------------

from __future__ import annotations

# ---- merged imports (union of the inlined modules plus the wrapper) --------
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum, IntFlag
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from threading import Lock, get_ident
from time import perf_counter_ns
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift, next_fast_len
from scipy.signal import bessel, butter, cheby1, cheby2, ellip, sosfreqz


# ---------------------------------------------------------------------
# begin inlined vendor/eikeosmers_cecht/phase.py (BSD-3-Clause)
# ---------------------------------------------------------------------
"""
Endpoint Corrected Hilbert Transform (ECHT) with optional calibration.
===========================================

Implements a causal, band-limited, endpoint-corrected analytic transform.
This method can be used for example to estimate the real-time phase of a signal.

This code is based on MEEGkit's ECHT class
    https://nbara.github.io/python-meegkit/
    https://github.com/nbara/python-meegkit/tree/master
    BSD 3-Clause License
    Copyright (c) 2019, nbara

ECHT was originally invented by Schreglmann et al. (2021):
Schreglmann, S. R., Wang, D., Peach, R. L., Li, J., Zhang, X.,
       Latorre, A., ... & Grossman, N. (2021). Non-invasive suppression of
       essential tremor via phase-locked disruption of its temporal coherence.
       Nature communications, 12(1), 363.
"""

from functools import lru_cache
import numpy as np
from scipy.signal import butter, bessel, cheby1, cheby2, ellip, sosfreqz
from scipy.fft import fft, ifft, fftshift, ifftshift, next_fast_len


class ECHT:
    """
    Endpoint-Corrected Hilbert Transform (ECHT) with optional endpoint calibration.

    Parameters
    ----------
    l_freq : float or None
        Low cutoff of band-pass filter (Hz). If None, only high-pass is applied.
    h_freq : float or None
        High cutoff of band-pass filter (Hz). If None, only low-pass is applied.
    sfreq : float
        Sampling rate of the signal.
    n_fft : int or None, optional
        FFT length used internally. If None, determined by ``fft_mode``.
    fft_mode : {'fast', 'exact'}, optional
        Strategy for FFT length selection:
        - 'fast' (standard): use ``next_fast_len`` (faster, potentially less accurate)
        - 'exact': use ``n_samples`` (slower, exact window alignment)
    filt_order : int, optional
        IIR filter order (default 1).
        Note: scipy defines a bandpass of order 1 as (highpass + lowpass both order 1).
    filter_type : {'butter','bessel','cheby1','cheby2','ellip'}, optional
        Type of band-pass filter.
        butter: standard
    calibrate : bool, optional
        If True, apply complex-valued scalar gain to correct endpoint amplitude/phase.
    f0 : float or None, optional
        Center frequency (Hz) of the assumed cosine used for calibration.

    Attributes
    ----------
    h_ : ndarray
        Hilbert analytic-signal multiplier (freq-domain).
    coef_ : ndarray
        Frequency response of the band-pass filter (column vector).
    calib_gain_ : complex or None
        Global endpoint calibration gain.
    calib_err_ : dict or None
        Theoretical calibration error metrics.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(
            self,
            l_freq,
            h_freq,
            sfreq,
            n_fft=None,
            fft_mode="fast",
            filt_order=1,
            filter_type="butter",
            calibrate=False,
            f0=None,
    ):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq
        self.n_fft = n_fft
        self.fft_mode = str(fft_mode).lower()
        if self.fft_mode not in ("fast", "exact"):
            raise ValueError("fft_mode must be 'fast' or 'exact'")
        self.filt_order = filt_order
        self.filter_type = filter_type

        self.calibrate = bool(calibrate)
        self.f0 = f0

        # Runtime attributes set during fitting
        self.h_ = None          # analytic-signal multiplier
        self.coef_ = None       # band-pass frequency response (column vector)
        self.calib_gain_ = None # complex scalar calibration factor
        self.calib_err_ = None  # dict with theoretical error metrics


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    @lru_cache(maxsize=None)
    def _design_bandpass(l_freq, h_freq, sfreq, filt_order, n_fft, filter_type="butter"):
        """
        Construct Hilbert multiplier and centered band-pass frequency response.

        Returns
        -------
        h : ndarray
            Analytic-signal Hilbert multiplier (length ``n_fft``).
        H_center : ndarray
            Band-pass filter response on a centered FFT grid.
        """

        # ---------------------------
        # Hilbert analytic multiplier
        # ---------------------------
        h = np.zeros(n_fft, dtype=float)
        h[0] = 1
        if n_fft % 2 == 0:
            # even length: DC + positive + Nyquist
            h[1:n_fft // 2] = 2
            h[n_fft // 2] = 1
        else:
            # odd length: DC + positive
            h[1:(n_fft + 1) // 2] = 2

        # ---------------------------
        # Band-pass IIR filter design
        # ---------------------------
        Wn = [l_freq / (sfreq / 2), h_freq / (sfreq / 2)]

        if filter_type == "butter":
            sos = butter(filt_order, Wn, btype="band", output='sos')
        elif filter_type == "bessel":
            sos = bessel(filt_order, Wn, btype="band", norm="phase", output='sos')
        elif filter_type == "cheby1":
            sos = cheby1(filt_order, 1, Wn, btype="band", output='sos')  # 1 dB ripple
        elif filter_type == "cheby2":
            sos = cheby2(filt_order, 40, Wn, btype="band", output='sos')  # 40 dB stopband
        elif filter_type == "ellip":
            sos = ellip(filt_order, 1, 40, Wn, btype="band", output='sos')  # elliptic / Cauer
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

        # Centered frequency grid (Hz), length n_fft
        filt_freq = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1 / sfreq))

        _, H_center = sosfreqz(sos, worN=filt_freq, fs=sfreq)

        return h, H_center

    @staticmethod
    def _calibration(f0, sfreq, N, l_freq, h_freq, filt_order=1, filter_type="butter", L=None, fft_mode="fast"):
        """
        Compute theoretical endpoint error for a pure cosine input.
        For many applications this approximation is valid.

        Returns
        -------
        err : dict
            Contains:
            - 'Gplus'  : desired analytic-aligned component
            - 'Gminus' : residual image component
            - 'C_opt'  : complex calibration gain
            - 'J_opt'  : minimal mean-square endpoint error
        """
        if L is None:
            L = next_fast_len(N) if fft_mode == "fast" else N

        # ---- Dirichlet kernel (length-N cosine spectrum) ----
        def _dirichlet_N(alpha):
            alpha = np.asarray(alpha, dtype=float)
            D = np.empty(alpha.shape, dtype=np.complex128)
            small = np.abs(alpha) < 1e-12
            D[small] = N
            a_ns = alpha[~small]
            D[~small] = (
                np.exp(1j * a_ns * (N - 1) / 2)
                * np.sin(0.5 * N * a_ns)
                / np.sin(0.5 * a_ns)
            )
            return D

        k = np.arange(L)
        omega_k = 2 * np.pi * k / L  # rad/sample, DFT grid
        omega0 = 2 * np.pi * f0 / sfreq
        n = N - 1

        # Hilbert multiplier + centered band-pass response
        h, H_center = ECHT._design_bandpass(
            l_freq=l_freq, h_freq=h_freq, sfreq=sfreq, filt_order=filt_order, n_fft=L, filter_type=filter_type,
        )

        H_eff = ifftshift(H_center)
        G = h * H_eff

        D_plus = _dirichlet_N(omega0 - omega_k )
        D_minus = _dirichlet_N(-omega0 - omega_k)

        X_plus = 0.5 * D_plus
        X_minus = 0.5 * D_minus

        # Endpoint of the ECHT output for each spectral component.
        phase = np.exp(1j * omega_k * n)
        P = (G * X_plus * phase).sum() / L
        M = (G * X_minus * phase).sum() / L

        # Separate contributions aligned with the ideal analytic signal.
        Gplus = P * np.exp(-1j * omega0 * n)
        Gminus = M * np.exp(-1j * omega0 * n)

        # MSE-optimal complex calibration gain and corresponding minimal error
        denom = np.abs(Gplus) ** 2 + np.abs(Gminus) ** 2
        if denom == 0:
            C_opt = 1 + 0j
            J_opt = 0
        else:
            C_opt = np.conj(Gplus) / denom
            J_opt = np.abs(Gminus) ** 2 / denom

        return {
            "Gplus": Gplus,
            "Gminus": Gminus,
            "C_opt": C_opt,
            "J_opt": J_opt,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """
        Fit the ECHT transform to the input signal (FFT size, filters, calibration).

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels)
            The input signal to be transformed.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        if self.n_fft is None:
            self.n_fft = (
                next_fast_len(n_samples) if self.fft_mode == "fast" else n_samples
            )

        self.h_, H_center = self._design_bandpass(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            sfreq=self.sfreq,
            filt_order=self.filt_order,
            n_fft=self.n_fft,
            filter_type=self.filter_type,
        )
        self.coef_ = H_center[:, None]

        # Optional endpoint calibration
        self.calib_gain_ = None
        self.calib_err_ = None
        if self.calibrate:
            if self.f0 is None:
                raise ValueError("f0 (signal frequency) must be provided when calibrate=True.")

            err = self._calibration(
                f0=self.f0,
                sfreq=self.sfreq,
                N=n_samples,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                filt_order=self.filt_order,
                L=self.n_fft,
                fft_mode=self.fft_mode,
                filter_type=self.filter_type,
            )
            C_opt = err["C_opt"]

            if not (np.isfinite(C_opt.real) and np.isfinite(C_opt.imag)):
                raise RuntimeError(
                    "Non-finite calibration gain C_opt computed for ECHT."
                )

            self.calib_gain_ = C_opt
            self.calib_err_ = err

        return self

    def transform(self, X, **kwargs):
        X = np.asarray(X)
        if not np.isrealobj(X):
            X = np.real(X)

        if self.h_ is None or self.coef_ is None:
            self.fit(X)

        if X.ndim == 1:
            X = X[:, np.newaxis]

        n_samples = X.shape[0]

        Xf = fft(X, self.n_fft, axis=0)
        Xf = Xf * self.h_[:, None]
        Xf = fftshift(Xf, axes=0)
        Xf = Xf * self.coef_
        Xf = ifftshift(Xf, axes=0)
        Xf = ifft(Xf, axis=0)

        Xf = self._apply_calibration(Xf, **kwargs)

        return Xf[:n_samples, :]

    def _apply_calibration(self, Xf, **kwargs):
        if self.calibrate and self.calib_gain_ is not None:
            return Xf * self.calib_gain_
        return Xf

    def fit_transform(self, X, y=None):
        """Fit the ECHT transform to the input signal and transform it."""
        return self.fit(X).transform(X)

# ---------------------------------------------------------------------
# begin inlined causal_cecHT/streaming_core.py
# ---------------------------------------------------------------------
"""Fixed-capacity, strictly causal cecHT streaming core."""


from dataclasses import dataclass
from enum import IntEnum, IntFlag
import math
from numbers import Integral, Real

import numpy as np
from scipy.fft import ifft as _constructor_ifft
from scipy.fft import ifftshift as _constructor_ifftshift

# [build] removed relative import: from .vendor.eikeosmers_cecht.phase import ECHT


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

# ---------------------------------------------------------------------
# begin inlined causal_cecHT/pseudo_trigger.py
# ---------------------------------------------------------------------
"""Independent configurable positive-crossing pseudo-trigger for causal cecHT."""


from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Integral, Real

import numpy as np

# [build] removed relative import: from .streaming_core import wrap_phase_rad


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

# ---------------------------------------------------------------------
# begin generated golden constants (development-machine reference)
# config: {"calibrate": true, "f0_hz": 0.8, "filter_order": 2, "h_freq": 1.25, "l_freq": 0.5, "n_fft": 2000, "window_samples": 2000}
# ---------------------------------------------------------------------
GOLDEN_KERNEL = {
    "at": {
        "0": {
            "im": -0.0027829411966988292,
            "re": 1.9238219458147307e-05
        },
        "1": {
            "im": -0.0025310203024583155,
            "re": 1.7371120372483818e-05
        },
        "1000": {
            "im": 4.363729922963883e-05,
            "re": -0.00010361800270710107
        },
        "1999": {
            "im": -0.0031385479379442546,
            "re": 0.00011317764051728123
        }
    },
    "kernel_len": 2000,
    "l1": 2.6458647517767395,
    "max_abs": 0.008130532047694572
}

# =====================================================================
# Hand-written deployment wrapper.  Everything above this line is generated
# verbatim from the frozen sources; only this section is project glue.
# =====================================================================


import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, get_ident
from time import perf_counter_ns
from uuid import uuid4

import numpy as np


PLUGIN_VERSION = "phase-tracking-cecht-plugin-v1"

# Frozen canonical deployment configuration (approved 8 s / f0 = 0.8 Hz).
DEPLOY_FS_HZ = 250.0
DEPLOY_WINDOW_S = 8.0
DEPLOY_WINDOW_SAMPLES = 2000
DEPLOY_N_FFT = 2000
DEPLOY_F0_HZ = 0.8
DEPLOY_BAND_HZ = (0.5, 1.25)
DEPLOY_FILTER_ORDER = 2
DEPLOY_TARGET_PHASE_RAD = 0.0
DEPLOY_REFRACTORY_S = 0.5
MAX_BLOCK_SAMPLES = 1024
PHASE_RING_SAMPLES = 2500  # 10 s of history at 250 Hz

SELF_TEST_REL_TOL = 1e-9
SELF_TEST_ABS_TOL = 1e-12

_LOG_LOCK = Lock()
_LOG_ENV = "CLAS_CECHT_PHASE_LOG"
_DEFAULT_LOG = Path(__file__).resolve().parent / "probe_logs" / "PhaseTracking_cecHT.jsonl"
_SELF_TEST_CACHE: dict | None = None


def deploy_core_config():
    """The frozen estimator configuration used by every channel instance."""

    return CecHTCoreConfig(
        sampling_rate_hz=DEPLOY_FS_HZ,
        input_dtype=np.dtype(np.float64),
        numeric_unit_code=1,
        channel_code=7,
        polarity_code=1,
        window_samples=DEPLOY_WINDOW_SAMPLES,
        l_freq=DEPLOY_BAND_HZ[0],
        h_freq=DEPLOY_BAND_HZ[1],
        filter_order=DEPLOY_FILTER_ORDER,
        filter_type="butter",
        n_fft=DEPLOY_N_FFT,
        fft_mode="exact",
        calibrate=True,
        f0=DEPLOY_F0_HZ,
        parameter_version="deploy-canonical-8s-v1",
        config_version="deploy-canonical-8s-v1",
        max_block_samples=MAX_BLOCK_SAMPLES,
    )


def deploy_trigger_config():
    """Pseudo-trigger configuration; no real stimulation is ever commanded."""

    return PseudoTriggerConfig(
        sampling_rate_hz=DEPLOY_FS_HZ,
        target_phase_rad=DEPLOY_TARGET_PHASE_RAD,
        refractory_s=DEPLOY_REFRACTORY_S,
        parameter_version="deploy-canonical-8s-v1",
        config_version="deploy-canonical-8s-v1",
        max_block_samples=MAX_BLOCK_SAMPLES,
    )


def kernel_statistics(kernel: np.ndarray) -> dict:
    """Environment-sensitive numbers baked into GOLDEN_KERNEL by the builder."""

    values = np.asarray(kernel, dtype=np.complex128)
    return {
        "kernel_len": int(values.size),
        "max_abs": float(np.max(np.abs(values))),
        "l1": float(np.sum(np.abs(values))),
        "at": {
            str(index): {
                "re": float(values[index].real),
                "im": float(values[index].imag),
            }
            for index in (0, 1, values.size // 2, values.size - 1)
        },
    }


def run_kernel_self_test() -> dict:
    """Recompute the endpoint kernel in-host and compare against the golden.

    This is the deployment-time environment check: ``GOLDEN_KERNEL`` was
    produced on the development machine, so a mismatch means the host's
    numpy/scipy produced a numerically different kernel.
    """

    global _SELF_TEST_CACHE
    if _SELF_TEST_CACHE is not None:
        return _SELF_TEST_CACHE

    try:
        kernel = CausalCecHTCore(deploy_core_config()).endpoint_kernel
        observed = kernel_statistics(kernel)
        worst = 0.0
        worst_path = None
        failures = []

        def compare(reference, candidate, path):
            nonlocal worst, worst_path
            if isinstance(reference, dict):
                for key, value in reference.items():
                    compare(value, candidate.get(key), f"{path}.{key}")
                return
            difference = abs(float(candidate) - float(reference))
            relative = difference / abs(float(reference)) if reference else difference
            if difference > SELF_TEST_ABS_TOL and relative > SELF_TEST_REL_TOL:
                failures.append(path)
                if relative > worst:
                    worst, worst_path = relative, path

        compare(GOLDEN_KERNEL, observed, "kernel")
        _SELF_TEST_CACHE = {
            "status": "PASS" if not failures else "FAIL",
            "failed_paths": failures,
            "worst_relative_deviation": worst,
            "worst_path": worst_path,
            "observed": observed,
            "expected": GOLDEN_KERNEL,
            "rel_tol": SELF_TEST_REL_TOL,
            "abs_tol": SELF_TEST_ABS_TOL,
        }
    except BaseException as exc:  # noqa: BLE001 - diagnostics must never raise
        _SELF_TEST_CACHE = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return _SELF_TEST_CACHE


class CechtPhaseCore:
    """Causal cecHT phase estimator with a host-shaped block interface.

    Responsibilities:

    * stage each incoming block into a pre-allocated C-contiguous float64
      buffer (the host hands over strided views; the streaming core rejects
      non-contiguous input);
    * keep one absolute sample counter per instance so the core never sees a
      gap or a duplicate index;
    * run the pseudo-trigger on ``valid & ready`` phase endpoints;
    * hand the phase history to the companion side channel;
    * return the input waveform unchanged (the phase is not a waveform).
    """

    def __init__(self, fs: float) -> None:
        self.constructor_fs = fs
        self._instance_id = uuid4().hex
        configured_log = os.environ.get(_LOG_ENV)
        self._log_path = (
            Path(configured_log).expanduser().resolve() if configured_log else _DEFAULT_LOG
        )
        self._log_errors = 0
        self._initialized = False
        self._degraded = False
        self._degrade_reason = None
        self._core = None
        self._trigger = None
        self._self_test = None

        # Pre-allocated hot-path storage (no allocation inside ``process``).
        self._block = np.empty(MAX_BLOCK_SAMPLES, dtype=np.float64)
        self._relative_index = np.empty(MAX_BLOCK_SAMPLES, dtype=np.int64)
        self._offset = np.arange(MAX_BLOCK_SAMPLES, dtype=np.int64)
        self._core_outputs = CecHTOutputBuffers.allocate(MAX_BLOCK_SAMPLES)
        self._trigger_outputs = PseudoTriggerOutputBuffers.allocate(MAX_BLOCK_SAMPLES)
        self._phase_ring = np.zeros(PHASE_RING_SAMPLES, dtype=np.float64)
        self._index_ring = np.zeros(PHASE_RING_SAMPLES, dtype=np.int64)
        self._ready_ring = np.zeros(PHASE_RING_SAMPLES, dtype=np.bool_)

        self._zero_counters()
        if fs == DEPLOY_FS_HZ:
            try:
                self._core = CausalCecHTCore(deploy_core_config())
                self._trigger = CausalCecHTTrigger(deploy_trigger_config())
                self._self_test = run_kernel_self_test()
            except BaseException as exc:  # noqa: BLE001 - degrade instead of crashing
                self._degrade(f"setup_failed: {type(exc).__name__}: {exc}")
        else:
            self._degrade(f"unsupported_sampling_rate: {fs}")

        self._initialized = True
        self._write_event(
            "loaded",
            {
                "plugin_version": PLUGIN_VERSION,
                "constructor_fs_declared_by_host": fs,
                "module_name": __name__,
                "module_present_in_sys_modules": bool(
                    __name__ in sys.modules and sys.modules.get(__name__) is not None
                ),
                "module_path": str(Path(__file__).resolve()),
                "numpy_version": np.__version__,
                "scipy_version": _scipy_version(),
                "degraded": self._degraded,
                "degrade_reason": self._degrade_reason,
                "self_test": self._self_test,
                "config": {
                    "sampling_rate_hz": DEPLOY_FS_HZ,
                    "window_s": DEPLOY_WINDOW_S,
                    "window_samples": DEPLOY_WINDOW_SAMPLES,
                    "n_fft": DEPLOY_N_FFT,
                    "f0_hz": DEPLOY_F0_HZ,
                    "band_hz": list(DEPLOY_BAND_HZ),
                    "filter_order": DEPLOY_FILTER_ORDER,
                    "calibrated": True,
                    "target_phase_rad": DEPLOY_TARGET_PHASE_RAD,
                    "refractory_s": DEPLOY_REFRACTORY_S,
                },
                "stimulation_enabled": False,
                "process_file_io": False,
            },
        )

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        if self._initialized and self.attempts:
            self._write_event("reset_summary", self.snapshot())
        if self._core is not None:
            self._core.reset(CecHTReasonCode.EXPLICIT_RESET)
        if self._trigger is not None:
            self._trigger.reset(PseudoTriggerReasonCode.EXPLICIT_RESET)
        self._zero_counters()

    def _zero_counters(self) -> None:
        self.attempts = 0
        self.calls = 0
        self.samples = 0
        self.ready_samples = 0
        self.events = 0
        self.internal_errors = 0
        self.structural_rejects = 0
        self.stream_faults = 0
        self.oversize_blocks = 0
        self.max_process_ns = 0
        self.total_process_ns = 0
        self.epoch = 0
        self._sample_index = 0
        self._ring_pos = 0
        self._ring_filled = 0
        self._last_ready_phase = None
        self._last_ready_index = None
        self._last_event_index = None
        self._last_event_crossing = None
        self._first_fault_logged = False
        self._first_reject_logged = False

    def _degrade(self, reason: str) -> None:
        self._degraded = True
        self._degrade_reason = reason

    # -- hot path ----------------------------------------------------------

    def process(self, x, stim_label):
        started = perf_counter_ns()
        self.attempts += 1
        try:
            if self._degraded or self._core is None:
                return x
            try:
                block = x if isinstance(x, np.ndarray) else np.asarray(x)
            except BaseException:  # noqa: BLE001
                self.internal_errors += 1
                return x
            if block.ndim != 1:
                self.structural_rejects += 1
                self._degrade(f"unexpected_block_ndim: {block.ndim}")
                return x
            count = int(block.size)
            if count == 0:
                return x
            if count > MAX_BLOCK_SAMPLES:
                self.oversize_blocks += 1
                self._degrade(f"block_larger_than_capacity: {count}")
                return x

            # The host hands over a strided view; the core requires C-contiguous
            # float64 input, so stage it into our own pre-allocated buffer.
            self._block[:count] = block
            staged = self._block[:count]

            np.add(self._offset[:count], self._sample_index, out=self._relative_index[:count])
            context = CecHTBlockContext(
                stream_epoch=self.epoch,
                first_relative_index=self._sample_index,
                declared_sampling_rate_hz=DEPLOY_FS_HZ,
                unit_code=1,
                channel_code=7,
                polarity_code=1,
                verified_fields_mask=CecHTVerifiedField.ALL,
            )
            status = self._core.process_block_into(staged, context, self._core_outputs)
            self.calls += 1
            self.samples += count

            if status is CecHTProcessStatus.STRUCTURAL_REJECT:
                self.structural_rejects += 1
                if not self._first_reject_logged:
                    self._first_reject_logged = True
                    self._write_event(
                        "structural_reject",
                        {
                            "reason": str(self._core.last_reason_code),
                            "block_size": count,
                            "sample_index": self._sample_index,
                        },
                    )
                self._degrade(f"structural_reject: {self._core.last_reason_code}")
                self._sample_index += count
                return x
            if status is CecHTProcessStatus.STREAM_FAULT:
                self.stream_faults += 1
                if not self._first_fault_logged:
                    self._first_fault_logged = True
                    self._write_event(
                        "stream_fault",
                        {
                            "reason": str(self._core.last_reason_code),
                            "block_size": count,
                            "sample_index": self._sample_index,
                        },
                    )
                # The core has already reset itself; keep running so a single
                # transient glitch cannot disable the algorithm for the session.
                self._sample_index += count
                return x

            self._run_trigger(count)
            self._push_history(count)
            self._sample_index += count
        except BaseException as exc:  # noqa: BLE001 - never break the host callback
            self.internal_errors += 1
            if self.internal_errors == 1:
                self._write_event(
                    "internal_error",
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
        finally:
            elapsed = perf_counter_ns() - started
            self.total_process_ns += elapsed
            self.max_process_ns = max(self.max_process_ns, elapsed)
        return x

    def _run_trigger(self, count: int) -> None:
        if self._trigger is None:
            return
        status = self._trigger.process_block_into(
            self._relative_index[:count],
            self._core_outputs.phase_rad[:count],
            self._core_outputs.valid[:count],
            self._core_outputs.ready[:count],
            self._trigger_outputs,
        )
        if status is PseudoTriggerStatus.OK:
            fired = int(np.count_nonzero(self._trigger_outputs.event[:count]))
            if fired:
                self.events += fired
                positions = np.flatnonzero(self._trigger_outputs.event[:count])
                last = int(positions[-1])
                self._last_event_index = int(self._relative_index[last])
                self._last_event_crossing = float(
                    self._trigger_outputs.crossing_time_s[last]
                )

    def _push_history(self, count: int) -> None:
        ready = self._core_outputs.ready[:count]
        if not ready.any():
            return
        positions = np.flatnonzero(ready)
        self.ready_samples += int(positions.size)
        last = int(positions[-1])
        self._last_ready_phase = float(self._core_outputs.phase_rad[last])
        self._last_ready_index = int(self._relative_index[last])

        capacity = PHASE_RING_SAMPLES
        position = self._ring_pos
        end = position + count
        if end <= capacity:
            self._phase_ring[position:end] = self._core_outputs.phase_rad[:count]
            self._index_ring[position:end] = self._relative_index[:count]
            self._ready_ring[position:end] = ready
        else:
            head = capacity - position
            self._phase_ring[position:] = self._core_outputs.phase_rad[:head]
            self._index_ring[position:] = self._relative_index[:head]
            self._ready_ring[position:] = ready[:head]
            tail = count - head
            self._phase_ring[:tail] = self._core_outputs.phase_rad[head:count]
            self._index_ring[:tail] = self._relative_index[head:count]
            self._ready_ring[:tail] = ready[head:count]
        self._ring_pos = end % capacity
        self._ring_filled = min(capacity, self._ring_filled + count)

    # -- reporting ---------------------------------------------------------

    def latest_phase(self) -> dict:
        """Read-only view of the newest ready phase endpoint."""

        return {
            "ready": self._last_ready_phase is not None,
            "phase_rad": self._last_ready_phase,
            "relative_sample_index": self._last_ready_index,
            "ready_samples": self.ready_samples,
        }

    def phase_history(self) -> dict:
        """Snapshot of the phase ring for a companion app."""

        count = self._ring_filled
        if count == 0:
            return {
                "phase_rad": np.empty(0, dtype=np.float64),
                "relative_sample_index": np.empty(0, dtype=np.int64),
                "ready": np.empty(0, dtype=np.bool_),
            }
        capacity = PHASE_RING_SAMPLES
        if count < capacity:
            return {
                "phase_rad": self._phase_ring[:count].copy(),
                "relative_sample_index": self._index_ring[:count].copy(),
                "ready": self._ready_ring[:count].copy(),
            }
        order = np.concatenate(
            (
                np.arange(self._ring_pos, capacity),
                np.arange(0, self._ring_pos),
            )
        )
        return {
            "phase_rad": self._phase_ring[order],
            "relative_sample_index": self._index_ring[order],
            "ready": self._ready_ring[order],
        }

    def snapshot(self) -> dict:
        mean_ns = self.total_process_ns / self.attempts if self.attempts else None
        return {
            "plugin_version": PLUGIN_VERSION,
            "instance_id": self._instance_id,
            "constructor_fs_declared_by_host": self.constructor_fs,
            "degraded": self._degraded,
            "degrade_reason": self._degrade_reason,
            "self_test_status": (self._self_test or {}).get("status"),
            "attempts": self.attempts,
            "calls": self.calls,
            "samples": self.samples,
            "ready_samples": self.ready_samples,
            "events": self.events,
            "internal_errors": self.internal_errors,
            "structural_rejects": self.structural_rejects,
            "stream_faults": self.stream_faults,
            "oversize_blocks": self.oversize_blocks,
            "last_reason_code": (
                str(self._core.last_reason_code) if self._core is not None else None
            ),
            "last_trigger_reason_code": (
                str(self._trigger.last_reason_code) if self._trigger is not None else None
            ),
            "last_ready_phase_rad": self._last_ready_phase,
            "last_ready_index": self._last_ready_index,
            "last_event_index": self._last_event_index,
            "last_event_crossing_s": self._last_event_crossing,
            "mean_process_ns": mean_ns,
            "max_process_ns": self.max_process_ns if self.attempts else None,
            "raw_samples_retained": False,
            "returned_object_identity_preserved": True,
            "stimulation_enabled": False,
        }

    def _write_event(self, event: str, payload: dict) -> None:
        record = {
            "schema_version": 1,
            "event": event,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "instance_id": self._instance_id,
            "pid": os.getpid(),
            "thread_id": get_ident(),
            "payload": payload,
        }
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, default=str)
            with _LOG_LOCK, self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except BaseException:  # noqa: BLE001 - logging must not break the host
            self._log_errors += 1


def _scipy_version() -> str:
    try:
        import scipy

        return str(scipy.__version__)
    except BaseException:  # noqa: BLE001
        return "unavailable"


# NOTE: this class deliberately has no docstring.  The host builds each
# algorithm-list entry as '<file stem> (<first line of the class docstring>)',
# so a docstring here would be appended to the name shown in the GUI.
class CustomSignalAlgorithm(CechtPhaseCore):
    pass


__all__ = [
    "CustomSignalAlgorithm",
    "CechtPhaseCore",
    "deploy_core_config",
    "deploy_trigger_config",
    "kernel_statistics",
    "run_kernel_self_test",
    "PLUGIN_VERSION",
]
