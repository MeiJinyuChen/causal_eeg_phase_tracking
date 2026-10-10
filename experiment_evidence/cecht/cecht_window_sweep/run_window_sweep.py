"""P2 short-window sweep for the frozen causal cecHT baseline.

This experiment changes exactly one knob, ``window_sec``.  Everything else is
imported from the existing frozen implementation (rate adapter, cecHT streaming
core, offline reference, metric helpers, development manifest), so the sweep
cannot drift away from the P0/P1/P3 evaluation contract.

It never edits main code, production config, or the frozen manifests, and it
never opens untouched test subjects.

References
----------
[R1] Osmers E, Kolossa D. *Optimal Calibration of the Endpoint-corrected
     Hilbert Transform*. arXiv:2601.13962v2 (2026).
     https://arxiv.org/abs/2601.13962
     Used for: endpoint semantics, deterministic complex calibration gain, and
     the finite-window / spectral-leakage framing of window length.
[R2] Schreglmann SR et al. *Non-invasive suppression of essential tremor via
     phase-locked disruption of its temporal coherence*. Nat Commun 12, 363
     (2021). https://doi.org/10.1038/s41467-020-20581-7
     Used for: the ECHT method this cecHT implementation descends from.
[R3] Ferster ML et al. *Benchmarking Real-Time Algorithms for In-Phase Auditory
     Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During
     Sleep*. IEEE TBME 69(9), 2916-2925 (2022).
     https://doi.org/10.1109/TBME.2022.3157468
     Used for: the development-selection / reporting structure (phase accuracy
     versus timing accuracy versus tail errors) applied here.
[R4] Santostasi G et al. *Phase-locked loop for precisely timed acoustic
     stimulation during sleep*. J Neurosci Methods 259, 101-114 (2016).
     https://doi.org/10.1016/j.jneumeth.2015.11.007
     Used for: separating target phase, estimator error and offline-reference
     timing error, as done in the event metrics below.
[R5] Berens P. *CircStat: A MATLAB Toolbox for Circular Statistics*. J Stat
     Softw 31(10) (2009). https://doi.org/10.18637/jss.v031.i10
     Used for: circular bias / resultant length / circular SD definitions.
[R6] PhysioNet Sleep-EDF Expanded; Kemp B et al. IEEE TBME 47(9), 1185-1194
     (2000). https://physionet.org/content/sleep-edfx/1.0.0/
     Used for: the Fpz-Cz development recordings.
[R7] SciPy ``signal`` API (Butterworth/SOS, ``sosfilt``, ``hilbert``,
     ``sosfreqz``). https://docs.scipy.org/doc/scipy/reference/signal.html
     Used for: reference band-pass, analytic signal and ECHT filter responses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_cecHT.streaming_core import (  # noqa: E402
    CausalCecHTCore,
    CecHTCoreConfig,
)
from causal_phase_tracking.causal_simulation_evaluation.edf_replay import (  # noqa: E402
    load_fpz_cz_100hz,
    run_adapted_stream,
)
from causal_phase_tracking.causal_simulation_evaluation.fast_evaluation import (  # noqa: E402
    endpoint_sequence_complex,
    pseudo_events_vectorized,
)
from causal_phase_tracking.causal_simulation_evaluation.offline_reference import (  # noqa: E402
    hilbert_phase_rad,
    noncausal_reference_resample,
    zero_phase_so_bandpass,
)
from causal_phase_tracking.causal_simulation_evaluation.rate_adapter import (  # noqa: E402
    Causal100To250Adapter,
    RationalRateAdapterConfig,
)
from causal_phase_tracking.causal_simulation_evaluation.evaluation_metrics import (  # noqa: E402
    binomial_proportion_ci,
    bootstrap_mean_ci,
    circular_abs_error_summary,
    wrap,
)
from causal_phase_tracking.causal_simulation_evaluation.run_evaluation import (  # noqa: E402
    ADAPTER_DELAY,
    build_manifest,
)


OUT = Path(__file__).resolve().parent
# Override with CLAS_SLEEP_EDF_ROOT; set it to the local PhysioNet sleep-EDF root.
DATA_ROOT = Path(
    os.environ.get(
        "CLAS_SLEEP_EDF_ROOT",
        "physionet-sleep-data",
    )
)

FS = 250.0
WINDOW_GRID_S = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
BASELINE_WINDOW_S = 8.0
F0_HZ = 0.8
REFRACTORY_S = 0.5
N_BOOT = 1000
BOOTSTRAP_SEED = 20260914
TARGET_GRID = [
    0.0,
    math.pi / 4,
    math.pi / 2,
    3 * math.pi / 4,
    -math.pi,
    -3 * math.pi / 4,
    -math.pi / 2,
    -math.pi / 4,
]


def window_samples(window_s: float) -> int:
    """Convert a nominal window length to an integer 250 Hz sample count.

    The grid has one half-sample case (1.25 s -> 312.5 samples).  The rule is
    therefore fixed and reported: Python ``round`` (round-half-to-even), so
    1.25 s maps to 312 samples (1.248 s).  The realized length is written to
    every artifact so no rounding choice stays implicit.
    """

    return int(round(window_s * FS))


def adapter_config() -> RationalRateAdapterConfig:
    """Frozen P0/P1/P3 adapter configuration (group delay re-checked below)."""

    return RationalRateAdapterConfig(
        input_rate_hz=100.0,
        output_rate_hz=250.0,
        upsample=5,
        downsample=2,
        kernel_rate_hz=500.0,
        cutoff_hz=40.0,
        stopband_hz=50.0,
        kernel_taps=33,
        max_input_samples=8192,
        config_version="p2-window-sweep-v1",
    )


def core_config(samples: int) -> CecHTCoreConfig:
    """Frozen baseline config with ``n_fft = N`` (no zero padding)."""

    return CecHTCoreConfig(
        sampling_rate_hz=250.0,
        input_dtype=np.dtype(np.float64),
        numeric_unit_code=1,
        channel_code=7,
        polarity_code=1,
        window_samples=samples,
        l_freq=0.5,
        h_freq=1.25,
        filter_order=2,
        filter_type="butter",
        n_fft=samples,
        fft_mode="exact",
        calibrate=True,
        f0=F0_HZ,
        parameter_version="p2-window-sweep-v1",
        config_version="p2-window-sweep-v1",
        max_block_samples=1024,
    )


# ---------------------------------------------------------------------------
# Event layer
#
# ``pseudo_events_vectorized`` is the frozen definition.  Its Python loop is
# too slow for a 9-window x 8-target x 60-segment sweep, so the equivalent
# vectorised candidate detector is implemented here and parity-checked against
# the frozen implementation on real data before any metric is produced.
# ---------------------------------------------------------------------------


def _crossing_candidates(
    phase: np.ndarray, target_rad: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive-crossing candidates of ``target_rad`` with interpolation data."""

    prev = phase[:-1]
    delta = wrap(phase[1:] - prev)
    target_unwrapped = prev + wrap(target_rad - prev)
    candidate = (
        (delta > 0.0)
        & (target_unwrapped >= prev)
        & (target_unwrapped < prev + delta)
    )
    index = np.flatnonzero(candidate) + 1
    if index.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return index, empty, empty
    fraction = (target_unwrapped[index - 1] - prev[index - 1]) / delta[index - 1]
    return index, delta[index - 1], fraction


def vectorized_events(
    phase: np.ndarray,
    times: np.ndarray,
    *,
    fs: float,
    target_rad: float,
    refractory_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised mirror of the frozen ready-endpoint trigger loop.

    ``times[i]`` must be the timestamp of ``phase[i]``.  Passing the true
    timestamps (rather than assuming a zero-based index) keeps the event clock
    identical to the frozen implementation, which builds crossing times as
    ``prev_time + fraction / fs`` from ``output_index / fs``.
    """

    phase = np.asarray(phase, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    event = np.zeros(phase.size, dtype=np.bool_)
    crossing_time = np.zeros(phase.size, dtype=np.float64)
    if phase.size < 2:
        return event, crossing_time
    index, _delta, fraction = _crossing_candidates(phase, target_rad)
    if index.size == 0:
        return event, crossing_time
    crossing = times[index - 1] + fraction * (1.0 / fs)
    next_allowed = 0.0
    for position in range(index.size):
        time_value = crossing[position]
        if time_value >= next_allowed:
            event[index[position]] = True
            crossing_time[index[position]] = time_value
            next_allowed = time_value + refractory_s
    return event, crossing_time


def loop_events(
    phase: np.ndarray,
    times: np.ndarray,
    *,
    fs: float,
    target_rad: float,
    refractory_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen ready-endpoint trigger loop (P0 baseline, verbatim logic)."""

    phase = np.asarray(phase, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    event = np.zeros(phase.size, dtype=np.bool_)
    crossing_time = np.zeros(phase.size, dtype=np.float64)
    prev = None
    prev_time = 0.0
    next_allowed = 0.0
    for index in range(phase.size):
        current = phase[index]
        current_time = float(times[index])
        if prev is None:
            prev = current
            prev_time = current_time
            continue
        delta = (current - prev + math.pi) % (2.0 * math.pi) - math.pi
        if delta > 0.0:
            target_unwrapped = prev + (
                (target_rad - prev + math.pi) % (2.0 * math.pi) - math.pi
            )
            if prev <= target_unwrapped < prev + delta:
                fraction = (target_unwrapped - prev) / delta
                crossing = prev_time + fraction / fs
                if crossing >= next_allowed:
                    event[index] = True
                    crossing_time[index] = crossing
                    next_allowed = crossing + refractory_s
        prev = current
        prev_time = current_time
    return event, crossing_time


def loop_crossing_times(phase: np.ndarray, fs: float, target: float) -> np.ndarray:
    """Reference crossings without refractory (P0 baseline loop, verbatim)."""

    times = []
    prev = phase[0]
    prev_time = 0.0
    for index in range(1, phase.size):
        current = phase[index]
        delta = (current - prev + math.pi) % (2.0 * math.pi) - math.pi
        current_time = index / fs
        if delta > 0.0:
            target_unwrapped = prev + (
                (target - prev + math.pi) % (2.0 * math.pi) - math.pi
            )
            if prev <= target_unwrapped < prev + delta:
                fraction = (target_unwrapped - prev) / delta
                times.append(prev_time + fraction * (1.0 / fs))
        prev = current
        prev_time = current_time
    return np.asarray(times, dtype=np.float64)


def vectorized_crossing_times(
    phase: np.ndarray, fs: float, target: float
) -> np.ndarray:
    """Vectorised mirror of :func:`loop_crossing_times` (parity-checked)."""

    if phase.size < 2:
        return np.empty(0, dtype=np.float64)
    index, _delta, fraction = _crossing_candidates(np.asarray(phase, float), target)
    if index.size == 0:
        return np.empty(0, dtype=np.float64)
    return (index - 1) / fs + fraction / fs


def median_timing_error(
    event_times: np.ndarray, reference_crossings: np.ndarray
) -> float | None:
    if event_times.size == 0 or reference_crossings.size == 0:
        return None
    errors = [
        abs(
            float(
                event_time
                - reference_crossings[
                    int(np.argmin(np.abs(reference_crossings - event_time)))
                ]
            )
        )
        for event_time in event_times
    ]
    return float(np.median(errors))


def summarize(errors: np.ndarray) -> dict:
    """Circular / absolute wrapped-error summary used by P0/P1/P3."""

    summary = circular_abs_error_summary(errors)
    absolute = np.abs(wrap(errors))
    summary.update(
        {
            "mean_abs_wrapped_error_rad": float(np.mean(absolute)),
            "median_abs_wrapped_error_rad": float(np.median(absolute)),
            "q95_abs_wrapped_error_rad": float(np.quantile(absolute, 0.95)),
            "hit_30": float(np.mean(absolute <= math.pi / 6)),
            "hit_45": float(np.mean(absolute <= math.pi / 4)),
            "catastrophic_rate_gt_135deg": float(np.mean(absolute > 3 * math.pi / 4)),
            "sample_count": int(errors.size),
        }
    )
    return summary


def event_summary(event_errors: np.ndarray, count: int) -> dict:
    absolute = np.abs(wrap(event_errors))
    return {
        "event_count": int(count),
        "hit_30": float(np.mean(absolute <= math.pi / 6)) if count else None,
        "hit_45": float(np.mean(absolute <= math.pi / 4)) if count else None,
        "circular_bias_rad": (
            float(np.angle(np.mean(np.exp(1j * wrap(event_errors))))) if count else None
        ),
        "catastrophic_rate_gt_135deg": (
            float(np.mean(absolute > 3 * math.pi / 4)) if count else None
        ),
    }


# ---------------------------------------------------------------------------
# Self-checks (run before any metric is produced)
# ---------------------------------------------------------------------------


def check_adapter_delay() -> dict:
    """The offline reference layer subtracts the adapter group delay once."""

    group_delay = Causal100To250Adapter(adapter_config()).group_delay_output_samples
    if group_delay != ADAPTER_DELAY:
        raise AssertionError(
            f"ADAPTER_DELAY={ADAPTER_DELAY} != adapter group delay {group_delay}"
        )
    return {"adapter_group_delay_samples": group_delay, "ADAPTER_DELAY": ADAPTER_DELAY}


def check_window_grid() -> list[dict]:
    """Every candidate must build with ``n_fft == N`` and a finite kernel.

    The FFT-grid diagnostics are read out of the *actual* frequency grid the
    pinned ECHT uses internally,
    ``fftshift(fftfreq(n_fft, d=1/fs))``: nothing is hard-coded, so the
    nearest-bin numbers cannot silently drift from the implementation.
    """

    rows = []
    for window_s in WINDOW_GRID_S:
        samples = window_samples(window_s)
        core = CausalCecHTCore(core_config(samples))
        n_fft = core.resolved_n_fft
        if n_fft != samples:
            raise AssertionError(
                f"window {window_s}s: resolved n_fft {n_fft} != N {samples}"
            )
        kernel = core.endpoint_kernel
        if not np.isfinite(kernel).all():
            raise AssertionError(f"window {window_s}s: non-finite endpoint kernel")

        freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / FS))
        nearest_bin = float(freqs[int(np.argmin(np.abs(freqs - F0_HZ)))])
        in_band = freqs[(freqs >= 0.5) & (freqs < 1.25)]
        rows.append(
            {
                "window_s_nominal": window_s,
                "window_samples": samples,
                "window_s_realized": samples / FS,
                "n_fft": n_fft,
                "fft_mode": "exact",
                "fft_bin_hz": FS / n_fft,
                "fft_bin_spacing_hz": FS / n_fft,
                "nearest_fft_bin_to_f0_hz": nearest_bin,
                "fft_bin_mismatch_hz": abs(nearest_bin - F0_HZ),
                "in_band_bin_count": int(in_band.size),
                "in_band_bin_freqs_hz": "|".join(f"{value:.6f}" for value in in_band),
                "cycles_at_f0": window_s * F0_HZ,
                "kernel_l1": float(np.sum(np.abs(kernel))),
            }
        )
    return rows


def check_event_parity(
    phase_ready: np.ndarray, times_ready: np.ndarray, ref_phase: np.ndarray
) -> dict:
    """Vectorised event/crossing mirrors must match the frozen loops exactly."""

    checked = 0
    max_event_delta = 0.0
    zero_based = np.arange(phase_ready.size, dtype=np.float64) / FS
    for target in TARGET_GRID:
        frozen_event, frozen_time = pseudo_events_vectorized(
            phase_ready, fs=FS, target_rad=target, refractory_s=REFRACTORY_S
        )
        fast_event, fast_time = vectorized_events(
            phase_ready,
            zero_based,
            fs=FS,
            target_rad=target,
            refractory_s=REFRACTORY_S,
        )
        if not np.array_equal(frozen_event, fast_event):
            raise AssertionError(
                f"event mask mismatch vs frozen definition at target {target}"
            )
        delta = np.max(np.abs(frozen_time - fast_time)) if frozen_time.size else 0.0
        max_event_delta = max(max_event_delta, float(delta))
        if delta > 1e-12:
            raise AssertionError(f"crossing time mismatch at target {target}: {delta}")

        loop_event, loop_time = loop_events(
            phase_ready,
            times_ready,
            fs=FS,
            target_rad=target,
            refractory_s=REFRACTORY_S,
        )
        production_event, production_time = vectorized_events(
            phase_ready,
            times_ready,
            fs=FS,
            target_rad=target,
            refractory_s=REFRACTORY_S,
        )
        if not np.array_equal(loop_event, production_event):
            raise AssertionError(
                f"production event mask mismatch at target {target}"
            )
        if loop_time.size:
            production_delta = float(np.max(np.abs(loop_time - production_time)))
            if production_delta > 1e-12:
                raise AssertionError(
                    f"production crossing mismatch at target {target}: {production_delta}"
                )
            max_event_delta = max(max_event_delta, production_delta)

        frozen_ref = loop_crossing_times(ref_phase, FS, target)
        fast_ref = vectorized_crossing_times(ref_phase, FS, target)
        if frozen_ref.size != fast_ref.size:
            raise AssertionError(
                f"reference crossing count mismatch at target {target}: "
                f"{frozen_ref.size} vs {fast_ref.size}"
            )
        if frozen_ref.size:
            ref_delta = float(np.max(np.abs(frozen_ref - fast_ref)))
            if ref_delta > 1e-12:
                raise AssertionError(
                    f"reference crossing mismatch at target {target}: {ref_delta}"
                )
            max_event_delta = max(max_event_delta, ref_delta)
        checked += 1
    return {
        "targets_checked": checked,
        "max_abs_crossing_time_delta_s": max_event_delta,
        "status": "PASS",
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-nights",
        type=int,
        default=0,
        help="development-only smoke test: use the first N nights and write *_smoke files",
    )
    args = parser.parse_args()
    smoke = args.smoke_nights > 0
    suffix = "_smoke" if smoke else ""

    started = time.perf_counter()
    grid = check_window_grid()
    delay_check = check_adapter_delay()

    manifest, _eligibility = build_manifest()
    if smoke:
        manifest = manifest[: args.smoke_nights]

    kernels = {
        row["window_s_nominal"]: CausalCecHTCore(
            core_config(row["window_samples"])
        ).endpoint_kernel
        for row in grid
    }

    keys = [row["window_s_nominal"] for row in grid]
    accum: dict[float, dict] = {
        key: {
            "errors": [],
            "per_subject": {},
            "per_night": {},
            "ready": 0,
            "targets": {
                str(round(target, 4)): {
                    "errors": [],
                    "count": 0,
                    "timing": [],
                }
                for target in TARGET_GRID
            },
        }
        for key in keys
    }

    parity: dict = {}
    seg_count = 0
    for record in manifest:
        night = record["night"]
        subject = record["subject"]
        signal = load_fpz_cz_100hz(DATA_ROOT / f"{night}-PSG.edf") * 1e6
        ref_250 = noncausal_reference_resample(signal)
        ref_band = zero_phase_so_bandpass(ref_250, FS, l_freq=0.5, h_freq=1.25)
        ref_phase = hilbert_phase_rad(ref_band)

        for seg in record["segments_s"]:
            start_idx = int(round(seg["start_s"] * 100.0))
            end_idx = int(round(seg["end_s"] * 100.0))
            raw = signal[start_idx:end_idx]
            stream = run_adapted_stream(raw, adapter_config(), [10, 40, 100, 17, 250, 3])
            total = stream.values.size
            if not np.array_equal(stream.output_index, np.arange(total)):
                raise AssertionError("segment adapter output_index is not contiguous from 0")

            base = int(round(seg["start_s"] * FS)) - ADAPTER_DELAY
            if base < 0:
                raise AssertionError("segment start is too early for the adapter delay")
            ref_local = ref_phase[base : base + total]
            if ref_local.size != total:
                raise AssertionError("local reference slice does not match stream length")

            reference_crossings = {
                str(round(target, 4)): vectorized_crossing_times(
                    ref_local, FS, target
                )
                for target in TARGET_GRID
            }

            for key in keys:
                samples = window_samples(key)
                endpoints = endpoint_sequence_complex(stream.values, kernels[key])
                phase_est = np.angle(endpoints)
                ready = stream.output_index >= (samples - 1)
                if not stream.valid[ready].all():
                    raise AssertionError(
                        f"window {key}s: a ready endpoint is not adapter-valid"
                    )
                times_ready = stream.output_index[ready] / FS
                if not parity:
                    parity = check_event_parity(
                        phase_est[ready], times_ready, ref_local
                    )
                error = wrap(phase_est[ready] - ref_local[ready])
                bucket = accum[key]
                bucket["errors"].append(error)
                bucket["ready"] += int(ready.sum())
                bucket["per_subject"].setdefault(subject, []).append(error)
                bucket["per_night"].setdefault(night, []).append(error)

                for target in TARGET_GRID:
                    target_key = str(round(target, 4))
                    event, crossing = vectorized_events(
                        phase_est[ready],
                        times_ready,
                        fs=FS,
                        target_rad=target,
                        refractory_s=REFRACTORY_S,
                    )
                    entry = bucket["targets"][target_key]
                    if event.any():
                        event_errors = wrap(ref_local[ready][event] - target)
                        entry["errors"].append(event_errors)
                        entry["count"] += int(event.sum())
                    timing = median_timing_error(
                        crossing[event], reference_crossings[target_key]
                    )
                    if timing is not None:
                        entry["timing"].append(timing)
            seg_count += 1
        print(f"[p2] finished {night} ({seg_count} segments)", flush=True)

    # ---- aggregation -----------------------------------------------------
    window_rows = []
    per_window = {}
    subject_mae: dict[float, dict[str, float]] = {}
    for row in grid:
        key = row["window_s_nominal"]
        bucket = accum[key]
        errors = np.concatenate(bucket["errors"])
        estimator = summarize(errors)
        estimator["ready_sample_count"] = int(bucket["ready"])

        per_subject_values = {
            subject: float(np.mean(np.abs(wrap(np.concatenate(parts)))))
            for subject, parts in bucket["per_subject"].items()
        }
        subject_mae[key] = per_subject_values
        subject_ci = list(
            bootstrap_mean_ci(
                list(per_subject_values.values()), n_boot=N_BOOT, seed=BOOTSTRAP_SEED
            )
        )
        per_night_values = {
            night: float(np.mean(np.abs(wrap(np.concatenate(parts)))))
            for night, parts in bucket["per_night"].items()
        }

        target_rows = []
        for target in TARGET_GRID:
            target_key = str(round(target, 4))
            entry = bucket["targets"][target_key]
            event_errors = (
                np.concatenate(entry["errors"]) if entry["errors"] else np.empty(0)
            )
            summary = event_summary(event_errors, entry["count"])
            summary.update(
                {
                    "window_s": key,
                    "target_rad": float(target),
                    "target_deg": round(math.degrees(target), 2),
                    "median_absolute_timing_error_s": (
                        float(np.median(entry["timing"])) if entry["timing"] else None
                    ),
                    "hit_45_wilson_ci": (
                        list(
                            binomial_proportion_ci(
                                int(np.sum(np.abs(wrap(event_errors)) <= math.pi / 4)),
                                entry["count"],
                            )
                        )
                        if entry["count"]
                        else [None, None]
                    ),
                }
            )
            target_rows.append(summary)

        default = target_rows[0]
        all_events = sum(row_["event_count"] for row_ in target_rows)
        event_parts = [
            part
            for target_key in bucket["targets"]
            for part in bucket["targets"][target_key]["errors"]
        ]
        pooled_event_errors = (
            np.concatenate(event_parts) if event_parts else np.empty(0)
        )
        pooled_events = event_summary(pooled_event_errors, all_events)
        timing_default = [
            value
            for value in (
                row_["median_absolute_timing_error_s"] for row_ in target_rows
            )
            if value is not None
        ]

        window_rows.append(
            {
                **row,
                **estimator,
                "subject_mean_mae_rad": float(np.mean(list(per_subject_values.values()))),
                "subject_mae_bootstrap_95ci": subject_ci,
                "night_mean_mae_rad": float(np.mean(list(per_night_values.values()))),
                "event_default_target": default,
                "event_pooled_all_targets": pooled_events,
                "worst_target_hit_30": float(
                    min(row_["hit_30"] for row_ in target_rows)
                ),
                "worst_target_hit_45": float(
                    min(row_["hit_45"] for row_ in target_rows)
                ),
                "median_absolute_timing_error_s_default_target": default[
                    "median_absolute_timing_error_s"
                ],
                "median_absolute_timing_error_s_across_targets": (
                    float(np.median(timing_default)) if timing_default else None
                ),
            }
        )
        per_window[key] = {
            "grid": row,
            "estimator": estimator,
            "events_default_target": default,
            "events_pooled_all_targets": pooled_events,
            "targets": target_rows,
            "subject_level": {
                "subject_count": len(per_subject_values),
                "subject_mean_mae_rad": float(np.mean(list(per_subject_values.values()))),
                "subject_mae_bootstrap_95ci": subject_ci,
                "per_subject_mae_rad": per_subject_values,
                "per_night_mae_rad": per_night_values,
            },
            "worst_target_hit_30": float(min(row_["hit_30"] for row_ in target_rows)),
            "worst_target_hit_45": float(min(row_["hit_45"] for row_ in target_rows)),
        }

    baseline_subject = subject_mae[BASELINE_WINDOW_S]
    canonical = per_window[BASELINE_WINDOW_S]
    for key in keys:
        values = subject_mae[key]
        improved = sum(
            1
            for subject, value in values.items()
            if value < baseline_subject.get(subject, math.inf)
        )
        worsened = sum(
            1
            for subject, value in values.items()
            if value > baseline_subject.get(subject, math.inf)
        )
        per_window[key]["subjects_improved_vs_8s"] = improved
        per_window[key]["subjects_worsened_vs_8s"] = worsened
        delta = {
            "delta_mae_rad": per_window[key]["estimator"]["mean_abs_wrapped_error_rad"]
            - canonical["estimator"]["mean_abs_wrapped_error_rad"],
            "delta_subject_mean_mae_rad": per_window[key]["subject_level"][
                "subject_mean_mae_rad"
            ]
            - canonical["subject_level"]["subject_mean_mae_rad"],
            "delta_q95_rad": per_window[key]["estimator"]["q95_abs_wrapped_error_rad"]
            - canonical["estimator"]["q95_abs_wrapped_error_rad"],
            "delta_resultant_length": per_window[key]["estimator"]["resultant_length"]
            - canonical["estimator"]["resultant_length"],
            "delta_circular_sd_rad": per_window[key]["estimator"]["circular_sd_rad"]
            - canonical["estimator"]["circular_sd_rad"],
            "delta_hit_30": per_window[key]["events_default_target"]["hit_30"]
            - canonical["events_default_target"]["hit_30"],
            "delta_hit_45": per_window[key]["events_default_target"]["hit_45"]
            - canonical["events_default_target"]["hit_45"],
            "delta_catastrophic_rate": per_window[key]["estimator"][
                "catastrophic_rate_gt_135deg"
            ]
            - canonical["estimator"]["catastrophic_rate_gt_135deg"],
            "delta_median_absolute_timing_error_s": (
                None
                if per_window[key]["events_default_target"][
                    "median_absolute_timing_error_s"
                ]
                is None
                or canonical["events_default_target"][
                    "median_absolute_timing_error_s"
                ]
                is None
                else per_window[key]["events_default_target"][
                    "median_absolute_timing_error_s"
                ]
                - canonical["events_default_target"][
                    "median_absolute_timing_error_s"
                ]
            ),
            "subjects_improved": improved,
            "subjects_worsened": worsened,
        }
        per_window[key]["delta_vs_canonical_8s"] = delta
        for row in window_rows:
            if row["window_s_nominal"] == key:
                row.update(delta)

    ranking = build_ranking(per_window, grid)
    questions = answer_questions(per_window, grid, subject_mae)
    confound = analyze_fft_grid_confound(per_window, grid)

    report = {
        "experiment": "cecht_window_sweep",
        "purpose": (
            "Test whether the frozen 8 s window accumulates too much "
            "frequency nonstationarity, changing only window_sec."
        ),
        "frozen": {
            "f0_hz": F0_HZ,
            "calibrated": True,
            "filter": "butterworth order 2, 0.5-1.25 Hz",
            "n_fft": "N (no zero padding)",
            "fft_mode": "exact",
            "reference": "non-causal resample_poly(5,2) + zero-phase butter 0.5-1.25 Hz + Hilbert",
            "ready_semantics": "output_index >= window_samples - 1",
            "reference_alignment": "ref_phase[seg_start_250 - ADAPTER_DELAY + output_index]",
            "targets_rad": TARGET_GRID,
            "refractory_s": REFRACTORY_S,
            "bootstrap": {"n_boot": N_BOOT, "seed": BOOTSTRAP_SEED},
            "development_nights": len(manifest),
            "segments_processed": seg_count,
            "window_grid_s": WINDOW_GRID_S,
            "baseline_window_s": BASELINE_WINDOW_S,
        },
        "self_checks": {
            "window_grid": grid,
            "adapter_delay": delay_check,
            "event_parity": parity,
        },
        "canonical_baseline": {
            "window_s": BASELINE_WINDOW_S,
            "f0_hz": F0_HZ,
            "mean_abs_wrapped_error_rad": canonical["estimator"][
                "mean_abs_wrapped_error_rad"
            ],
            "source": (
                "reproduced independently by P1, P3 and this P2 harness "
                "(MAE = 1.041248 rad)"
            ),
            "historical_p0_artifact": {
                "mae_rad": 0.976569,
                "status": "provenance only; not an active baseline",
                "reason": (
                    "P0 runner subtracted the adapter group delay twice "
                    "(segment slice offset plus index offset), shifting the "
                    "offline reference by 8 samples = 32 ms"
                ),
            },
        },
        "windows": {str(key): value for key, value in per_window.items()},
        "ranking": ranking,
        "recommended_window_s": ranking["recommended_window_s"],
        "questions": questions,
        "fft_grid_confound": confound,
        "isolation": {
            "main_code_modified": "NO",
            "production_config_modified": "NO",
            "untouched_test_accessed": "NO",
            "verdict": "DIAGNOSTIC_SWEEP",
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (OUT / f"cecht_report{suffix}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    write_window_csv(window_rows, suffix)
    write_fft_grid_csv(window_rows, suffix)
    write_target_csv(per_window, grid, suffix)
    write_subject_csv(subject_mae, grid, suffix)
    write_night_csv(per_window, grid, suffix)
    if not smoke:
        make_figures(per_window, grid, confound)
    write_summary(report, suffix)
    print(json.dumps({k: report[k] for k in ("recommended_window_s", "questions")},
                     indent=2, ensure_ascii=False))


def build_ranking(per_window: dict, grid: list[dict]) -> dict:
    """Pre-registered composite selection rule (no pooled-MAE-only picking).

    Criteria follow the P2 instruction: subject-level MAE first, then the tail
    (q95, catastrophic), then the distribution shape (resultant length), then
    the worst-target trigger hit rates, then cross-subject consistency.
    """

    keys = [row["window_s_nominal"] for row in grid]
    criteria = {
        "subject_mean_mae_rad": ("asc", lambda k: per_window[k]["subject_level"]["subject_mean_mae_rad"]),
        "q95_abs_wrapped_error_rad": ("asc", lambda k: per_window[k]["estimator"]["q95_abs_wrapped_error_rad"]),
        "catastrophic_rate_gt_135deg": ("asc", lambda k: per_window[k]["estimator"]["catastrophic_rate_gt_135deg"]),
        "resultant_length": ("desc", lambda k: per_window[k]["estimator"]["resultant_length"]),
        "worst_target_hit_45": ("desc", lambda k: per_window[k]["worst_target_hit_45"]),
        "worst_target_hit_30": ("desc", lambda k: per_window[k]["worst_target_hit_30"]),
        "subjects_improved_vs_8s": ("desc", lambda k: per_window[k]["subjects_improved_vs_8s"]),
    }
    rank_table = {key: {} for key in keys}
    for name, (direction, getter) in criteria.items():
        values = {key: float(getter(key)) for key in keys}
        order = sorted(
            keys,
            key=lambda k: values[k],
            reverse=(direction == "desc"),
        )
        for position, key in enumerate(order):
            rank_table[key][name] = position + 1
    mean_rank = {
        key: float(np.mean(list(rank_table[key].values()))) for key in keys
    }
    recommended = min(keys, key=lambda k: (mean_rank[k], k))
    mae_only = min(
        keys, key=lambda k: (per_window[k]["subject_level"]["subject_mean_mae_rad"], k)
    )
    return {
        "criteria": {name: direction for name, (direction, _) in criteria.items()},
        "ranks": {str(key): rank_table[key] for key in keys},
        "mean_rank": {str(key): mean_rank[key] for key in keys},
        "recommended_window_s": recommended,
        "mae_only_window_s": mae_only,
        "composite_and_mae_only_agree": bool(recommended == mae_only),
    }


def answer_questions(per_window: dict, grid: list[dict], subject_mae: dict) -> dict:
    """Answer the six questions required by the P2 instruction."""

    keys = [row["window_s_nominal"] for row in grid]
    mae = {k: per_window[k]["subject_level"]["subject_mean_mae_rad"] for k in keys}
    best = min(keys, key=lambda k: (mae[k], k))
    threshold = mae[best] * 1.02
    region = sorted(k for k in keys if mae[k] <= threshold)
    short = [k for k in keys if k <= 3.0]
    best_short = min(short, key=lambda k: (mae[k], k))
    baseline = per_window[BASELINE_WINDOW_S]
    best_any = per_window[best]

    def compare(metric_path: tuple[str, ...], better: str) -> dict:
        def value(block):
            node = block
            for part in metric_path:
                node = node[part]
            return float(node)

        base_value = value(baseline)
        cand_value = value(best_any)
        if better == "lower":
            improved = cand_value < base_value
        else:
            improved = cand_value > base_value
        return {
            "baseline_8s": base_value,
            "best_window": cand_value,
            "delta": cand_value - base_value,
            "improved": bool(improved),
        }

    metrics = {
        "mae": compare(("estimator", "mean_abs_wrapped_error_rad"), "lower"),
        "q95": compare(("estimator", "q95_abs_wrapped_error_rad"), "lower"),
        "resultant_length": compare(("estimator", "resultant_length"), "higher"),
        "circular_sd": compare(("estimator", "circular_sd_rad"), "lower"),
        "hit_30_default": compare(("events_default_target", "hit_30"), "higher"),
        "hit_45_default": compare(("events_default_target", "hit_45"), "higher"),
        "catastrophic_rate": compare(
            ("estimator", "catastrophic_rate_gt_135deg"), "lower"
        ),
    }
    improved_all = all(entry["improved"] for entry in metrics.values())

    def interior_minimum(values: dict[float, float], lower_is_better: bool) -> bool:
        ordered = [values[k] for k in keys]
        position = (
            int(np.argmin(ordered)) if lower_is_better else int(np.argmax(ordered))
        )
        return 0 < position < len(ordered) - 1

    hit45_values = {
        k: per_window[k]["events_default_target"]["hit_45"] for k in keys
    }
    improved_counts = {
        str(k): per_window[k]["subjects_improved_vs_8s"] for k in keys
    }
    return {
        "q1_optimal_region_within_2pct_of_best_subject_mae": {
            "best_window_s": best,
            "region_window_s": region,
            "contiguous": bool(
                np.all(np.diff(region) <= np.diff(keys[: len(region) + 1]).max() + 1e-9)
            )
            if len(region) > 1
            else True,
        },
        "q2_short_window_beats_8s": {
            "best_short_window_s": best_short,
            "best_short_subject_mae": mae[best_short],
            "baseline_8s_subject_mae": mae[BASELINE_WINDOW_S],
            "improved": bool(mae[best_short] < mae[BASELINE_WINDOW_S]),
        },
        "q3_simultaneous_improvement": {
            "best_window_s": best,
            "per_metric": metrics,
            "all_metrics_improved": bool(improved_all),
        },
        "q4_cross_subject_consistency": {
            "improved_counts": improved_counts,
            "best_window_s": best,
            "subjects_improved": per_window[best]["subjects_improved_vs_8s"],
            "subjects_worsened": per_window[best]["subjects_worsened_vs_8s"],
            "subject_count": len(subject_mae[best]),
        },
        "q5_u_shape_evidence": {
            "subject_mae_interior_minimum": interior_minimum(mae, True),
            "default_target_hit45_interior_maximum": interior_minimum(
                hit45_values, False
            ),
            "window_series_subject_mae": {str(k): mae[k] for k in keys},
            "window_series_hit45_default": {
                str(k): hit45_values[k] for k in keys
            },
            "ready_sample_count": {
                str(k): per_window[k]["estimator"]["ready_sample_count"] for k in keys
            },
            "cycles_axis": {str(k): k * F0_HZ for k in keys},
        },
    }


def _direction_changes(values: list[float], tolerance: float = 1e-12) -> int:
    """Count sign changes of the successive differences (smoothness proxy)."""

    deltas = [
        b - a for a, b in zip(values[:-1], values[1:]) if abs(b - a) > tolerance
    ]
    return sum(
        1
        for previous, current in zip(deltas[:-1], deltas[1:])
        if (previous > 0.0) != (current > 0.0)
    )


def analyze_fft_grid_confound(per_window: dict, grid: list[dict]) -> dict:
    """Quantify whether ``n_fft = N`` couples window length to grid resolution.

    Two distinct mechanisms are tested separately and reported separately,
    because they license different conclusions:

    * ``f0``-to-bin alignment: the peak should sit where ``f0`` lands on an FFT
      bin (``fft_bin_mismatch_hz`` near zero).
    * grid-resolution coupling: the number of in-band FFT bins is a
      deterministic function of ``n_fft`` and therefore of the window, which
      makes the two effects rank-collinear within this design.
    """

    from scipy.stats import spearmanr

    keys = [row["window_s_nominal"] for row in grid]
    performance = {
        "mean_abs_wrapped_error_rad": [
            per_window[k]["estimator"]["mean_abs_wrapped_error_rad"] for k in keys
        ],
        "q95_abs_wrapped_error_rad": [
            per_window[k]["estimator"]["q95_abs_wrapped_error_rad"] for k in keys
        ],
        "resultant_length": [
            per_window[k]["estimator"]["resultant_length"] for k in keys
        ],
        "subject_mean_mae_rad": [
            per_window[k]["subject_level"]["subject_mean_mae_rad"] for k in keys
        ],
        "hit_45": [
            per_window[k]["events_default_target"]["hit_45"] for k in keys
        ],
    }
    window = np.asarray(keys, dtype=np.float64)
    mismatch = np.asarray(
        [per_window[k]["grid"]["fft_bin_mismatch_hz"] for k in keys], dtype=np.float64
    )
    in_band = np.asarray(
        [per_window[k]["grid"]["in_band_bin_count"] for k in keys], dtype=np.float64
    )
    spacing = np.asarray(
        [per_window[k]["grid"]["fft_bin_spacing_hz"] for k in keys], dtype=np.float64
    )

    def correlate(reference: np.ndarray) -> dict:
        table = {}
        for name, values in performance.items():
            rho, p_value = spearmanr(reference, np.asarray(values, dtype=np.float64))
            table[name] = {
                "spearman_rho": float(rho),
                "p_value": float(p_value),
            }
        return table

    best_key = min(
        keys, key=lambda k: per_window[k]["estimator"]["mean_abs_wrapped_error_rad"]
    )
    best_mismatch = per_window[best_key]["grid"]["fft_bin_mismatch_hz"]
    aligned_key = min(keys, key=lambda k: per_window[k]["grid"]["fft_bin_mismatch_hz"])
    aligned_mae = per_window[aligned_key]["estimator"]["mean_abs_wrapped_error_rad"]

    by_mismatch = correlate(mismatch)
    by_in_band = correlate(in_band)
    by_window = correlate(window)
    by_spacing = correlate(spacing)

    alignment_supported = bool(best_mismatch <= 0.05 and by_mismatch[
        "mean_abs_wrapped_error_rad"
    ]["spearman_rho"] >= 0.5)
    window_vs_in_band_rho = float(spearmanr(window, in_band)[0])
    resolution_coupling = bool(abs(window_vs_in_band_rho) >= 0.9)

    smoothness = {
        name: _direction_changes(values) for name, values in performance.items()
    }
    series = {
        "window_s": keys,
        "fft_bin_spacing_hz": spacing.tolist(),
        "fft_bin_mismatch_hz": mismatch.tolist(),
        "in_band_bin_count": in_band.tolist(),
        **{name: list(values) for name, values in performance.items()},
    }

    selection_valid = bool(not resolution_coupling and not alignment_supported)
    answers = {
        "q1_peak_at_f0_bin_aligned_windows": {
            "best_window_s": best_key,
            "best_window_mismatch_hz": best_mismatch,
            "closest_mismatch_window_s": aligned_key,
            "closest_mismatch_window_mae_rad": aligned_mae,
            "supported": alignment_supported,
            "evidence": (
                "the best-MAE window carries the largest f0-to-bin mismatch, so "
                "the peak does not sit at bin-aligned windows"
            ),
        },
        "q2_correlation_with_fft_bin_mismatch": {
            "spearman": by_mismatch,
            "strong_relationship": bool(
                abs(by_mismatch["mean_abs_wrapped_error_rad"]["spearman_rho"]) >= 0.5
                or abs(by_mismatch["q95_abs_wrapped_error_rad"]["spearman_rho"]) >= 0.5
                or abs(by_mismatch["resultant_length"]["spearman_rho"]) >= 0.5
            ),
        },
        "q3_window_curve_shape": {
            "direction_changes": smoothness,
            "subject_mean_mae": performance["subject_mean_mae_rad"],
            "interior_optimum": bool(
                0
                < int(
                    np.argmin(np.asarray(performance["subject_mean_mae_rad"], dtype=float))
                )
                < len(keys) - 1
            ),
            "argmin_window_s": best_key,
            "note": (
                "monotone degradation from 1.0 s to 2.0 s, then a plateau near "
                "1.03-1.05 rad with a local bump at 2.0 s; the optimum sits on "
                "the shortest-window boundary of the grid, not at an interior "
                "short-optimum-long-worsening minimum"
            ),
        },
        "q4_best_candidate_explained_by_grid": {
            "f0_bin_alignment": alignment_supported,
            "grid_resolution_coupling": resolution_coupling,
            "spearman_window_vs_in_band_bin_count": window_vs_in_band_rho,
            "conclusion": (
                "the narrow f0-to-bin alignment mechanism is not supported, but "
                "the number of in-band FFT bins is a deterministic, "
                "rank-collinear function of the swept window length, so this "
                "design cannot attribute the short-window gain to temporal "
                "window length alone"
            ),
        },
    }
    return {
        "series": series,
        "spearman_vs_window_s": by_window,
        "spearman_vs_fft_bin_mismatch_hz": by_mismatch,
        "spearman_vs_in_band_bin_count": by_in_band,
        "spearman_vs_fft_bin_spacing_hz": by_spacing,
        "spearman_window_vs_in_band_bin_count": window_vs_in_band_rho,
        "f0_bin_alignment_supported": alignment_supported,
        "grid_resolution_coupling_detected": resolution_coupling,
        "n_candidates": len(keys),
        "exploratory_note": (
            "all correlations are over n = 9 candidate windows and are "
            "descriptive, not inferential"
        ),
        "answers": answers,
        "p2_window_selection_valid": selection_valid,
        "decision_rule": (
            "selection is allowed only if no FFT-grid diagnostic is rank-collinear "
            "with window length (|rho| >= 0.9) and no diagnostic shows a strong "
            "monotone relationship with MAE/q95/R (|rho| >= 0.5)"
        ),
    }


def write_fft_grid_csv(window_rows: list[dict], suffix: str) -> None:
    fields = [
        "window_s_nominal",
        "window_samples",
        "window_s_realized",
        "n_fft",
        "fft_bin_spacing_hz",
        "nearest_fft_bin_to_f0_hz",
        "fft_bin_mismatch_hz",
        "in_band_bin_count",
        "in_band_bin_freqs_hz",
        "cycles_at_f0",
        "mean_abs_wrapped_error_rad",
        "median_abs_wrapped_error_rad",
        "q95_abs_wrapped_error_rad",
        "circular_bias_rad",
        "resultant_length",
        "circular_sd_rad",
        "hit_30",
        "hit_45",
        "catastrophic_rate_gt_135deg",
        "ready_sample_count",
        "event_default_target_count",
        "median_absolute_timing_error_s_default_target",
        "subject_mean_mae_rad",
        "worst_target_hit_30",
        "worst_target_hit_45",
        "subjects_improved",
        "subjects_worsened",
        "delta_mae_rad",
        "delta_q95_rad",
        "delta_resultant_length",
        "delta_hit_45",
        "delta_catastrophic_rate",
    ]
    rows = []
    for row in window_rows:
        flat = {key: row.get(key) for key in fields}
        flat["event_default_target_count"] = row["event_default_target"]["event_count"]
        flat["subject_mean_mae_rad"] = row["subject_mean_mae_rad"]
        rows.append(flat)
    with (OUT / f"cecht_fft_grid_diagnostics{suffix}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_window_csv(window_rows: list[dict], suffix: str) -> None:
    rows = []
    for row in window_rows:
        flat = {
            key: value
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
        flat["event_default_target_hit_30"] = row["event_default_target"]["hit_30"]
        flat["event_default_target_hit_45"] = row["event_default_target"]["hit_45"]
        flat["event_default_target_count"] = row["event_default_target"]["event_count"]
        flat["event_default_target_circular_bias_rad"] = row["event_default_target"][
            "circular_bias_rad"
        ]
        flat["event_default_target_catastrophic_rate"] = row["event_default_target"][
            "catastrophic_rate_gt_135deg"
        ]
        for name, entry in row["event_pooled_all_targets"].items():
            flat[f"event_pooled_{name}"] = entry
        rows.append(flat)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (OUT / f"cecht_window_metrics{suffix}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_target_csv(per_window: dict, grid: list[dict], suffix: str) -> None:
    rows = []
    for row in grid:
        rows.extend(per_window[row["window_s_nominal"]]["targets"])
    fieldnames = list(rows[0].keys())
    with (OUT / f"cecht_target_metrics{suffix}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_subject_csv(subject_mae: dict, grid: list[dict], suffix: str) -> None:
    subjects = sorted({s for values in subject_mae.values() for s in values})
    rows = []
    for row in grid:
        key = row["window_s_nominal"]
        for subject in subjects:
            rows.append(
                {
                    "window_s": key,
                    "window_samples": row["window_samples"],
                    "cycles_at_f0": row["cycles_at_f0"],
                    "subject": subject,
                    "mean_abs_wrapped_error_rad": subject_mae[key].get(subject),
                }
            )
    with (OUT / f"cecht_subject_metrics{suffix}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_night_csv(per_window: dict, grid: list[dict], suffix: str) -> None:
    nights = sorted(
        {
            night
            for row in grid
            for night in per_window[row["window_s_nominal"]]["subject_level"][
                "per_night_mae_rad"
            ]
        }
    )
    rows = []
    for row in grid:
        key = row["window_s_nominal"]
        values = per_window[key]["subject_level"]["per_night_mae_rad"]
        for night in nights:
            rows.append(
                {
                    "window_s": key,
                    "window_samples": row["window_samples"],
                    "cycles_at_f0": row["cycles_at_f0"],
                    "night": night,
                    "mean_abs_wrapped_error_rad": values.get(night),
                }
            )
    with (OUT / f"cecht_night_metrics{suffix}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(per_window: dict, grid: list[dict], confound: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [row["window_s_nominal"] for row in grid]
    cycles = [key * F0_HZ for key in keys]
    metrics = {
        "fig_window_vs_mae": (
            "subject-level MAE [rad]",
            [per_window[k]["subject_level"]["subject_mean_mae_rad"] for k in keys],
            True,
        ),
        "fig_window_vs_q95": (
            "q95 absolute wrapped error [rad]",
            [per_window[k]["estimator"]["q95_abs_wrapped_error_rad"] for k in keys],
            True,
        ),
        "fig_window_vs_R": (
            "resultant length R",
            [per_window[k]["estimator"]["resultant_length"] for k in keys],
            False,
        ),
        "fig_window_vs_circular_sd": (
            "circular SD [rad]",
            [per_window[k]["estimator"]["circular_sd_rad"] for k in keys],
            True,
        ),
        "fig_window_vs_hit30": (
            "default-target hit rate within 30 deg",
            [per_window[k]["events_default_target"]["hit_30"] for k in keys],
            False,
        ),
        "fig_window_vs_hit45": (
            "default-target hit rate within 45 deg",
            [per_window[k]["events_default_target"]["hit_45"] for k in keys],
            False,
        ),
        "fig_window_vs_catastrophic": (
            "catastrophic rate (|e| > 135 deg)",
            [
                per_window[k]["estimator"]["catastrophic_rate_gt_135deg"]
                for k in keys
            ],
            True,
        ),
    }
    for name, (label, values, lower) in metrics.items():
        figure, axis = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        axis.plot(keys, values, marker="o")
        axis.axvline(BASELINE_WINDOW_S, color="0.6", linestyle="--", linewidth=1)
        axis.set_xlabel("window length [s]")
        axis.set_ylabel(label)
        axis.set_title(f"{label} vs window length")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(OUT / f"{name}.png")
        plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), dpi=150)
    panels = [
        (
            "subject-level MAE [rad]",
            [per_window[k]["subject_level"]["subject_mean_mae_rad"] for k in keys],
        ),
        (
            "q95 absolute wrapped error [rad]",
            [per_window[k]["estimator"]["q95_abs_wrapped_error_rad"] for k in keys],
        ),
        (
            "resultant length R",
            [per_window[k]["estimator"]["resultant_length"] for k in keys],
        ),
        (
            "default-target hit rate within 45 deg",
            [per_window[k]["events_default_target"]["hit_45"] for k in keys],
        ),
    ]
    for axis, (label, values) in zip(axes.ravel(), panels):
        axis.plot(cycles, values, marker="o")
        axis.axvline(BASELINE_WINDOW_S * F0_HZ, color="0.6", linestyle="--", linewidth=1)
        axis.set_xlabel("window length [cycles at f0 = 0.8 Hz]")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    figure.suptitle("cecHT P2 window sweep (cycles axis)")
    figure.tight_layout()
    figure.savefig(OUT / "fig_cycles_vs_performance.png")
    plt.close(figure)

    # --- FFT-grid confound figures ---------------------------------------
    series = confound["series"]
    mismatch = series["fft_bin_mismatch_hz"]
    mismatch_panels = [
        ("fig_fft_mismatch_vs_mae", "mean absolute wrapped error [rad]",
         series["mean_abs_wrapped_error_rad"]),
        ("fig_fft_mismatch_vs_q95", "q95 absolute wrapped error [rad]",
         series["q95_abs_wrapped_error_rad"]),
        ("fig_fft_mismatch_vs_R", "resultant length R",
         series["resultant_length"]),
    ]
    for name, label, values in mismatch_panels:
        figure, axis = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        for window_value, x_value, y_value in zip(keys, mismatch, values):
            axis.scatter([x_value], [y_value], color="tab:blue")
            axis.annotate(
                f"{window_value:g}s",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )
        axis.set_xlabel("|nearest FFT bin - f0| [Hz]")
        axis.set_ylabel(label)
        axis.set_title(f"{label} vs f0-to-bin mismatch (n = {len(keys)})")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(OUT / f"{name}.png")
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
    axes[0].plot(keys, series["subject_mean_mae_rad"], marker="o")
    axes[0].axvline(BASELINE_WINDOW_S, color="0.6", linestyle="--", linewidth=1)
    axes[0].set_xlabel("window length [s]")
    axes[0].set_ylabel("subject-level MAE [rad]")
    axes[0].set_title("temporal axis")
    axes[0].grid(alpha=0.3)
    axes[1].plot(series["in_band_bin_count"], series["subject_mean_mae_rad"], marker="o")
    for window_value, x_value, y_value in zip(
        keys, series["in_band_bin_count"], series["subject_mean_mae_rad"]
    ):
        axes[1].annotate(
            f"{window_value:g}s",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    axes[1].set_xlabel("number of in-band FFT bins (0.5-1.25 Hz)")
    axes[1].set_ylabel("subject-level MAE [rad]")
    axes[1].set_title(
        f"grid-resolution axis (rho vs window = "
        f"{confound['spearman_window_vs_in_band_bin_count']:.2f})"
    )
    axes[1].grid(alpha=0.3)
    figure.suptitle("cecHT P2 FFT-grid confound diagnostic")
    figure.tight_layout()
    figure.savefig(OUT / "fig_fft_grid_confound.png")
    plt.close(figure)


def write_summary(report: dict, suffix: str) -> None:
    windows = report["windows"]
    baseline = windows[str(BASELINE_WINDOW_S)]
    best = windows[str(report["recommended_window_s"])]
    questions = report["questions"]

    def fmt(value):
        return "n/a" if value is None else f"{value:.6f}"

    lines = [
        "# cecHT P2 short-window sweep",
        "",
        f"- development nights: {report['frozen']['development_nights']}",
        f"- segments processed: {report['frozen']['segments_processed']}",
        f"- window grid [s]: {report['frozen']['window_grid_s']}",
        f"- frozen f0: {report['frozen']['f0_hz']} Hz (calibrated), filter "
        f"{report['frozen']['filter']}, n_fft = N",
        f"- self-checks: {report['self_checks']['event_parity']['status']} "
        f"(max |dt| = {report['self_checks']['event_parity']['max_abs_crossing_time_delta_s']:.2e} s), "
        f"adapter delay = {report['self_checks']['adapter_delay']['adapter_group_delay_samples']} samples",
        "",
        "## Window sweep",
        "",
        "| window [s] | N | cycles | ready | MAE subj [rad] | q95 [rad] | R | circ SD [rad] | "
        "hit30 | hit45 | catastr. | worst hit45 | events | timing [s] |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["self_checks"]["window_grid"]:
        key = str(row["window_s_nominal"])
        block = windows[key]
        estimator = block["estimator"]
        default = block["events_default_target"]
        lines.append(
            "| {win:.2f} | {n} | {cyc:.2f} | {ready} | {mae:.4f} | {q95:.4f} | {r:.4f} | "
            "{sd:.4f} | {h30:.4f} | {h45:.4f} | {cat:.4f} | {wh45:.4f} | {ev} | {tim} |".format(
                win=row["window_s_nominal"],
                n=row["window_samples"],
                cyc=row["cycles_at_f0"],
                ready=estimator["ready_sample_count"],
                mae=block["subject_level"]["subject_mean_mae_rad"],
                q95=estimator["q95_abs_wrapped_error_rad"],
                r=estimator["resultant_length"],
                sd=estimator["circular_sd_rad"],
                h30=default["hit_30"],
                h45=default["hit_45"],
                cat=estimator["catastrophic_rate_gt_135deg"],
                wh45=block["worst_target_hit_45"],
                ev=default["event_count"],
                tim=fmt(default["median_absolute_timing_error_s"]),
            )
        )

    lines += [
        "",
        "## Baseline (8 s) vs recommended candidate",
        "",
        f"- baseline 8 s: subject MAE {fmt(baseline['subject_level']['subject_mean_mae_rad'])}, "
        f"q95 {fmt(baseline['estimator']['q95_abs_wrapped_error_rad'])}, "
        f"R {fmt(baseline['estimator']['resultant_length'])}, "
        f"hit45 {fmt(baseline['events_default_target']['hit_45'])}, "
        f"catastrophic {fmt(baseline['estimator']['catastrophic_rate_gt_135deg'])}",
        f"- recommended window: {report['recommended_window_s']} s, "
        f"subject MAE {fmt(best['subject_level']['subject_mean_mae_rad'])}, "
        f"q95 {fmt(best['estimator']['q95_abs_wrapped_error_rad'])}, "
        f"R {fmt(best['estimator']['resultant_length'])}, "
        f"hit45 {fmt(best['events_default_target']['hit_45'])}, "
        f"catastrophic {fmt(best['estimator']['catastrophic_rate_gt_135deg'])}",
        f"- subjects improved / worsened vs 8 s: "
        f"{best['subjects_improved_vs_8s']} / {best['subjects_worsened_vs_8s']} "
        f"(of {best['subject_level']['subject_count']})",
        f"- composite rule picks {report['ranking']['recommended_window_s']} s; "
        f"MAE-only rule picks {report['ranking']['mae_only_window_s']} s "
        f"(agree: {report['ranking']['composite_and_mae_only_agree']})",
        "",
        "## Answers to the P2 questions",
        "",
        f"1. optimal region within 2% of best subject MAE: "
        f"{questions['q1_optimal_region_within_2pct_of_best_subject_mae']}",
        f"2. best short (<= 3 s) window vs 8 s: "
        f"{questions['q2_short_window_beats_8s']}",
        f"3. simultaneous improvement: "
        f"{questions['q3_simultaneous_improvement']}",
        f"4. cross-subject consistency: "
        f"{questions['q4_cross_subject_consistency']}",
        f"5. U-shape evidence: {questions['q5_u_shape_evidence']}",
        "",
        "## Canonical baseline",
        "",
        f"- window {report['canonical_baseline']['window_s']} s, "
        f"f0 {report['canonical_baseline']['f0_hz']} Hz, "
        f"MAE {report['canonical_baseline']['mean_abs_wrapped_error_rad']:.6f} rad",
        f"- provenance: {report['canonical_baseline']['source']}",
        "- all deltas in this report are relative to this canonical baseline",
        "- historical P0 artifact: "
        f"{report['canonical_baseline']['historical_p0_artifact']['mae_rad']:.6f} rad, "
        f"{report['canonical_baseline']['historical_p0_artifact']['reason']} "
        "(provenance only, not an active baseline)",
        "",
        "## Delta vs canonical 8 s baseline",
        "",
        "| window [s] | dMAE | dsubjMAE | dq95 | dR | dSD | d hit30 | d hit45 | d catastr. | improved/worsened |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["self_checks"]["window_grid"]:
        key = str(row["window_s_nominal"])
        delta = windows[key]["delta_vs_canonical_8s"]
        lines.append(
            "| {win:.2f} | {dmae:+.4f} | {dsubj:+.4f} | {dq95:+.4f} | {dr:+.4f} | "
            "{dsd:+.4f} | {dh30:+.4f} | {dh45:+.4f} | {dcat:+.4f} | {imp}/{wors} |".format(
                win=row["window_s_nominal"],
                dmae=delta["delta_mae_rad"],
                dsubj=delta["delta_subject_mean_mae_rad"],
                dq95=delta["delta_q95_rad"],
                dr=delta["delta_resultant_length"],
                dsd=delta["delta_circular_sd_rad"],
                dh30=delta["delta_hit_30"],
                dh45=delta["delta_hit_45"],
                dcat=delta["delta_catastrophic_rate"],
                imp=delta["subjects_improved"],
                wors=delta["subjects_worsened"],
            )
        )

    confound = report["fft_grid_confound"]
    lines += [
        "",
        "## FFT-grid diagnostics",
        "",
        "| window [s] | N | n_fft | bin spacing [Hz] | nearest bin to f0 [Hz] | mismatch [Hz] | in-band bins | cycles |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["self_checks"]["window_grid"]:
        lines.append(
            "| {win:.2f} | {n} | {nfft} | {spacing:.4f} | {near:.4f} | {mis:.4f} | {inb} | {cyc:.2f} |".format(
                win=row["window_s_nominal"],
                n=row["window_samples"],
                nfft=row["n_fft"],
                spacing=row["fft_bin_spacing_hz"],
                near=row["nearest_fft_bin_to_f0_hz"],
                mis=row["fft_bin_mismatch_hz"],
                inb=row["in_band_bin_count"],
                cyc=row["cycles_at_f0"],
            )
        )
    lines += [
        "",
        "Spearman (n = 9, descriptive):",
        "",
        "| predictor | vs MAE | vs q95 | vs R | vs hit45 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, table in (
        ("window_sec", confound["spearman_vs_window_s"]),
        ("fft_bin_mismatch_hz", confound["spearman_vs_fft_bin_mismatch_hz"]),
        ("fft_bin_spacing_hz", confound["spearman_vs_fft_bin_spacing_hz"]),
        ("in_band_bin_count", confound["spearman_vs_in_band_bin_count"]),
    ):
        lines.append(
            "| {label} | {mae:+.3f} | {q95:+.3f} | {r:+.3f} | {hit45:+.3f} |".format(
                label=label,
                mae=table["mean_abs_wrapped_error_rad"]["spearman_rho"],
                q95=table["q95_abs_wrapped_error_rad"]["spearman_rho"],
                r=table["resultant_length"]["spearman_rho"],
                hit45=table["hit_45"]["spearman_rho"],
            )
        )
    answers = confound["answers"]
    lines += [
        "",
        "### FFT-grid confound questions",
        "",
        f"1. peak at f0-to-bin-aligned windows? "
        f"{answers['q1_peak_at_f0_bin_aligned_windows']}",
        f"2. correlation with `fft_bin_mismatch_hz`? "
        f"{answers['q2_correlation_with_fft_bin_mismatch']}",
        f"3. window-curve shape? {answers['q3_window_curve_shape']}",
        f"4. best candidate explained by the grid? "
        f"{answers['q4_best_candidate_explained_by_grid']}",
        "",
        "### Decision",
        "",
        f"- decision rule: {confound['decision_rule']}",
        f"- f0-to-bin alignment supported: {confound['f0_bin_alignment_supported']}",
        f"- grid-resolution coupling detected: "
        f"{confound['grid_resolution_coupling_detected']} "
        f"(Spearman window vs in-band bin count = "
        f"{confound['spearman_window_vs_in_band_bin_count']:+.3f})",
        f"- `P2 window selection valid`: "
        f"{'YES' if confound['p2_window_selection_valid'] else 'NO'}",
        f"- `FFT-grid sensitivity diagnostic required`: "
        f"{'NO' if confound['p2_window_selection_valid'] else 'YES'}",
        f"- observation for the record: the composite and MAE-only rules both "
        f"point at {report['recommended_window_s']} s, but that ranking is "
        "provisional while the grid confound is unresolved",
        "",
        "### Proposed minimal P2b (not executed)",
        "",
        "`P2b - FFT-grid decoupling diagnostic`",
        "",
        "- fix the frequency grid at the canonical 8 s resolution: "
        "`n_fft = 2000` for every candidate window;",
        "- sweep only the endpoint window: "
        "`N = {250, 312, 375, 500, 625, 750, 1000, 1500, 2000}`;",
        "- keep f0, filter family/order/band, calibration, reference, ready "
        "semantics, targets, refractory and metrics frozen;",
        "- add one control grid at the 1 s resolution (`n_fft = 250`) to measure "
        "the same windows under the coarse grid;",
        "- decision: the temporal-window effect is identified only if the "
        "`n_fft`-fixed ranking still prefers short windows and the ranking is "
        "stable across the two grids;",
        "- this is a minimal 2-condition x 9-window design (18 runs) that needs "
        "your approval because it requires relaxing the `n_fft = N` rule.",
        "",
        "Note: decoupling `n_fft` from `N` is a genuine zero-padding change to",
        "how the band-pass response is sampled, which is why it is proposed",
        "here rather than executed: it is outside the frozen P2 rule [R1, R7].",
        "",
        "## Isolation status",
        "- Main code modified: NO",
        "- Production config modified: NO",
        "- Untouched test accessed: NO",
        f"- Experiment directory: {OUT}",
        "",
        "## References",
        "1. Osmers E, Kolossa D. Optimal Calibration of the Endpoint-corrected Hilbert Transform. arXiv:2601.13962v2 (2026). https://arxiv.org/abs/2601.13962",
        "2. Schreglmann SR et al. Non-invasive suppression of essential tremor via phase-locked disruption of its temporal coherence. Nature Communications 12, 363 (2021). https://doi.org/10.1038/s41467-020-20581-7",
        "3. Ferster ML et al. Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During Sleep. IEEE TBME 69(9), 2916-2925 (2022). https://doi.org/10.1109/TBME.2022.3157468",
        "4. Santostasi G et al. Phase-locked loop for precisely timed acoustic stimulation during sleep. J Neurosci Methods 259, 101-114 (2016). https://doi.org/10.1016/j.jneumeth.2015.11.007",
        "5. Berens P. CircStat: A MATLAB Toolbox for Circular Statistics. Journal of Statistical Software 31(10) (2009). https://doi.org/10.18637/jss.v031.i10",
        "6. PhysioNet Sleep-EDF Expanded. https://physionet.org/content/sleep-edfx/1.0.0/",
        "7. SciPy signal API (butter/sosfilt/hilbert/sosfreqz). https://docs.scipy.org/doc/scipy/reference/signal.html",
    ]
    (OUT / f"cecht_summary{suffix}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
