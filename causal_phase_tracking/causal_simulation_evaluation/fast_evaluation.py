"""Fast, parity-verified offline evaluation mirrors used during tuning."""

from __future__ import annotations

import math

import numpy as np


def endpoint_sequence(samples: np.ndarray, endpoint_kernel: np.ndarray) -> np.ndarray:
    """Mirror CausalCecHTCore endpoints for a full continuous sequence."""

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("endpoint_sequence expects a 1-D signal")
    kernel = np.asarray(endpoint_kernel, dtype=np.complex128)
    window = kernel.size
    b = kernel[::-1]
    conv = np.convolve(x, b, mode="full")
    result = np.zeros(x.size, dtype=np.float64)
    if x.size >= window:
        result[window - 1 :] = np.angle(conv[window - 1 : x.size])
    return result


def endpoint_sequence_complex(
    samples: np.ndarray, endpoint_kernel: np.ndarray
) -> np.ndarray:
    """Complex analytic endpoint per sample with the same warm-up convention."""

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("endpoint_sequence_complex expects a 1-D signal")
    kernel = np.asarray(endpoint_kernel, dtype=np.complex128)
    window = kernel.size
    b = kernel[::-1]
    conv = np.convolve(x, b, mode="full")
    result = np.zeros(x.size, dtype=np.complex128)
    if x.size >= window:
        result[window - 1 :] = conv[window - 1 : x.size]
    return result


def pseudo_events_vectorized(
    phase_rad: np.ndarray,
    *,
    fs: float,
    target_rad: float = 0.0,
    refractory_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror CausalCecHTTrigger events for a complete phase sequence."""

    phase = np.asarray(phase_rad, dtype=np.float64)
    event = np.zeros(phase.size, dtype=np.bool_)
    crossing_time = np.zeros(phase.size, dtype=np.float64)
    dt = 1.0 / fs
    prev = phase[0]
    prev_time = 0.0
    next_allowed = 0.0
    for idx in range(1, phase.size):
        current = phase[idx]
        delta = (current - prev + math.pi) % (2.0 * math.pi) - math.pi
        current_time = idx * dt
        if delta > 0.0:
            target_unwrapped = prev + ((target_rad - prev + math.pi) % (2.0 * math.pi) - math.pi)
            if prev <= target_unwrapped < prev + delta:
                fraction = (target_unwrapped - prev) / delta
                crossing = prev_time + fraction * dt
                if crossing >= next_allowed:
                    event[idx] = True
                    crossing_time[idx] = crossing
                    next_allowed = crossing + refractory_s
        prev = current
        prev_time = current_time
    return event, crossing_time
