"""Isolated alignment diagnostic on a known 0.8 Hz sine.

This module is read-only with respect to the causal core/trigger. It reuses only
the same adapter, fast endpoint mirror, and offline reference helpers as
``run_evaluation.py``, and prints per-sample MAE, plus-minus-pi/4 hit rate, and the
default-target trigger bias.
"""

from __future__ import annotations

import math

import numpy as np

from causal_phase_tracking.causal_cecHT.streaming_core import CecHTCoreConfig, CausalCecHTCore

from .fast_evaluation import endpoint_sequence_complex, pseudo_events_vectorized
from .offline_reference import (
    hilbert_phase_rad,
    noncausal_reference_resample,
    zero_phase_so_bandpass,
)
from .rate_adapter import RationalRateAdapterConfig, Causal100To250Adapter
from .evaluation_metrics import wrap


def run() -> None:
    fs_in = 100.0
    duration_s = 180.0
    n = int(duration_s * fs_in)
    t = np.arange(n, dtype=np.float64) / fs_in
    signal = 80.0 * np.sin(2.0 * np.pi * 0.8 * t)

    adapter_config = RationalRateAdapterConfig(
        input_rate_hz=100.0,
        output_rate_hz=250.0,
        upsample=5,
        downsample=2,
        kernel_rate_hz=500.0,
        cutoff_hz=40.0,
        stopband_hz=50.0,
        kernel_taps=33,
        max_input_samples=8192,
        config_version="alignment-diagnostic-v1",
    )
    adapter = Causal100To250Adapter(adapter_config)
    stream_values = []
    stream_index = []
    chunk_sizes = [10, 40, 100, 17, 250, 3]
    cursor = 0
    pattern = 0
    while cursor < n:
        width = min(chunk_sizes[pattern % len(chunk_sizes)], n - cursor)
        block = signal[cursor : cursor + width]
        count = adapter.output_count_for_range(cursor, width)
        outputs = adapter.allocate_outputs(count)
        adapter.process_block_into(block, cursor, outputs)
        stream_values.append(outputs.resampled)
        stream_index.append(outputs.output_index)
        cursor += width
        pattern += 1
    stream_values = np.concatenate(stream_values)
    stream_index = np.concatenate(stream_index)

    reference_250 = noncausal_reference_resample(signal, input_rate_hz=100.0, output_rate_hz=250.0)
    reference_band = zero_phase_so_bandpass(reference_250, 250.0, l_freq=0.5, h_freq=1.25)
    reference_phase = hilbert_phase_rad(reference_band)

    core = CausalCecHTCore(
        CecHTCoreConfig(
            sampling_rate_hz=250.0,
            input_dtype=np.dtype(np.float64),
            numeric_unit_code=1,
            channel_code=7,
            polarity_code=1,
            window_samples=2000,
            l_freq=0.5,
            h_freq=1.25,
            filter_order=2,
            filter_type="butter",
            n_fft=2000,
            fft_mode="exact",
            calibrate=True,
            f0=0.8,
            parameter_version="alignment-diagnostic-v1",
            config_version="alignment-diagnostic-v1",
            max_block_samples=1024,
        )
    )
    endpoints = endpoint_sequence_complex(stream_values, core.endpoint_kernel)
    phase_est = np.angle(endpoints)
    ready = stream_index >= (core.config.window_samples - 1)
    ref_idx = np.clip(stream_index - 8, 0, reference_phase.size - 1)
    error = wrap(phase_est[ready] - reference_phase[ref_idx[ready]])
    mae = float(np.mean(np.abs(error)))
    hit_pi4 = float(np.mean(np.abs(error) <= math.pi / 4))

    event, crossing = pseudo_events_vectorized(
        phase_est[ready], fs=250.0, target_rad=0.0, refractory_s=0.5
    )
    event_ref_idx = np.clip(stream_index[ready][event] - 8, 0, reference_phase.size - 1)
    event_error = wrap(reference_phase[event_ref_idx]) if event_ref_idx.size else np.empty(0)
    event_bias = (
        float(np.angle(np.mean(np.exp(1j * event_error)))) if event_error.size else None
    )
    print(
        {
            "synthetic_freq_hz": 0.8,
            "ready_samples": int(ready.sum()),
            "mean_abs_wrapped_error_rad": mae,
            "hit_rate_within_pi_over_4": hit_pi4,
            "pseudo_event_count": int(event.sum()),
            "event_circular_bias_rad": event_bias,
        }
    )


if __name__ == "__main__":
    run()
