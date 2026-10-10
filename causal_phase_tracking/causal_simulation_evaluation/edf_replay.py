"""Deterministic Sleep-EDF loader, QA, and causal 100 -> 250 replay helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

from .rate_adapter import (
    Causal100To250Adapter,
    RationalRateAdapterConfig,
    RationalRateAdapterStatus,
)


PSG_CHANNEL = "EEG Fpz-Cz"
PSG_RATE_HZ = 100.0


def load_fpz_cz_100hz(
    path: str | Path,
    *,
    start_s: float | None = None,
    stop_s: float | None = None,
) -> np.ndarray:
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
    if raw.info["sfreq"] != PSG_RATE_HZ:
        raise ValueError(f"expected {PSG_RATE_HZ} Hz PSG, got {raw.info['sfreq']}")
    if PSG_CHANNEL not in raw.ch_names:
        raise ValueError(f"{PSG_CHANNEL} channel not found")
    start_sample = 0 if start_s is None else int(round(start_s * raw.info["sfreq"]))
    stop_sample = raw.n_times if stop_s is None else int(round(stop_s * raw.info["sfreq"]))
    data, _ = raw[PSG_CHANNEL, start_sample:stop_sample]
    return np.asarray(data, dtype=np.float64).ravel()


def qa_fpz_cz_signal(signal: np.ndarray, *, sampling_rate_hz: float) -> dict:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("QA expects a 1-D signal")
    nonfinite_count = int(np.sum(~np.isfinite(x)))
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            "nonfinite_count": nonfinite_count,
            "longest_flat_run_s": 0.0,
            "rail_count": 0,
            "derivative_mad_outlier_ratio": 0.0,
        }
    flat = np.diff(finite) == 0.0
    longest_run = 0
    current = 1
    for equal in flat:
        if equal:
            current += 1
            longest_run = max(longest_run, current)
        else:
            current = 1
    longest_flat_run_s = max(1, longest_run) / sampling_rate_hz
    rail_count = int(np.sum(np.abs(finite) >= 750000.0))
    derivative = np.diff(finite)
    mad = float(np.median(np.abs(derivative - np.median(derivative))))
    if mad > 0.0:
        ratio = float(np.mean(np.abs(derivative) > 10.0 * mad))
    else:
        ratio = 0.0
    return {
        "nonfinite_count": nonfinite_count,
        "longest_flat_run_s": longest_flat_run_s,
        "rail_count": rail_count,
        "derivative_mad_outlier_ratio": ratio,
    }


@dataclass(frozen=True, slots=True)
class AdaptedStreamResult:
    values: np.ndarray
    output_index: np.ndarray
    valid: np.ndarray
    reason_code: np.ndarray


def run_adapted_stream(
    samples: np.ndarray,
    config: RationalRateAdapterConfig,
    chunk_sizes: list[int],
) -> AdaptedStreamResult:
    adapter = Causal100To250Adapter(config)
    value_parts = []
    index_parts = []
    valid_parts = []
    reason_parts = []
    cursor = 0
    pattern_index = 0
    while cursor < samples.size:
        width = min(chunk_sizes[pattern_index % len(chunk_sizes)], samples.size - cursor)
        block = samples[cursor : cursor + width]
        count = adapter.output_count_for_range(cursor, width)
        outputs = adapter.allocate_outputs(count)
        status = adapter.process_block_into(block, cursor, outputs)
        if status is RationalRateAdapterStatus.STREAM_FAULT:
            raise RuntimeError("unexpected adapter stream fault in deterministic replay")
        if status is RationalRateAdapterStatus.STRUCTURAL_REJECT:
            raise RuntimeError("unexpected adapter structural reject in replay")
        value_parts.append(outputs.resampled)
        index_parts.append(outputs.output_index)
        valid_parts.append(outputs.valid)
        reason_parts.append(outputs.reason_code)
        cursor += width
        pattern_index += 1
    return AdaptedStreamResult(
        values=np.concatenate(value_parts),
        output_index=np.concatenate(index_parts),
        valid=np.concatenate(valid_parts),
        reason_code=np.concatenate(reason_parts),
    )
