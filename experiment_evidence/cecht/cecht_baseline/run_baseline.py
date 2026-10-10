"""P0 frozen baseline for the selected causal cecHT configuration.

This experiment only imports existing modules; it does not modify estimator
code, production config, or frozen test manifests, and it does not read
untouched test subjects.
"""

from __future__ import annotations

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

from causal_phase_tracking.causal_cecHT.streaming_core import CecHTCoreConfig, CausalCecHTCore
from causal_phase_tracking.causal_simulation_evaluation.edf_replay import (
    load_fpz_cz_100hz,
    qa_fpz_cz_signal,
    run_adapted_stream,
)
from causal_phase_tracking.causal_simulation_evaluation.fast_evaluation import (
    endpoint_sequence_complex,
)
from causal_phase_tracking.causal_simulation_evaluation.offline_reference import (
    hilbert_phase_rad,
    noncausal_reference_resample,
    zero_phase_so_bandpass,
)
from causal_phase_tracking.causal_simulation_evaluation.rate_adapter import RationalRateAdapterConfig
from causal_phase_tracking.causal_simulation_evaluation.evaluation_metrics import (
    binomial_proportion_ci,
    bootstrap_mean_ci,
    circular_abs_error_summary,
    wrap,
)
from causal_phase_tracking.causal_simulation_evaluation.run_evaluation import build_manifest


OUT = Path(__file__).resolve().parent
# Override with CLAS_SLEEP_EDF_ROOT; set it to the local PhysioNet sleep-EDF root.
DATA_ROOT = Path(
    os.environ.get(
        "CLAS_SLEEP_EDF_ROOT",
        "physionet-sleep-data",
    )
)
SELECTED = {"branch": "calibrated", "window_s": 8.0, "f0_hz": 0.8}
ADAPTER_DELAY = 8
TARGET_GRID = [
    0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4,
    -math.pi, -3 * math.pi / 4, -math.pi / 2, -math.pi / 4,
]


def adapter_config() -> RationalRateAdapterConfig:
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
        config_version="p0-baseline-v1",
    )


def core_config() -> CecHTCoreConfig:
    window = int(round(SELECTED["window_s"] * 250.0))
    return CecHTCoreConfig(
        sampling_rate_hz=250.0,
        input_dtype=np.dtype(np.float64),
        numeric_unit_code=1,
        channel_code=7,
        polarity_code=1,
        window_samples=window,
        l_freq=0.5,
        h_freq=1.25,
        filter_order=2,
        filter_type="butter",
        n_fft=window,
        fft_mode="exact",
        calibrate=True,
        f0=SELECTED["f0_hz"],
        parameter_version="p0-baseline-v1",
        config_version="p0-baseline-v1",
        max_block_samples=1024,
    )


def positive_crossing_times(
    phase: np.ndarray, fs: float, target_rad: float
) -> np.ndarray:
    """All positive crossings of the given target, without refractory."""

    times = []
    prev = phase[0]
    prev_time = 0.0
    for idx in range(1, phase.size):
        current = phase[idx]
        delta = (current - prev + math.pi) % (2.0 * math.pi) - math.pi
        current_time = idx / fs
        if delta > 0.0:
            target_unwrapped = prev + (
                (target_rad - prev + math.pi) % (2.0 * math.pi) - math.pi
            )
            if prev <= target_unwrapped < prev + delta:
                fraction = (target_unwrapped - prev) / delta
                times.append(prev_time + fraction * (1.0 / fs))
        prev = current
        prev_time = current_time
    return np.asarray(times, dtype=np.float64)


def median_timing_error(
    event_times: np.ndarray, reference_crossing_times: np.ndarray
) -> float | None:
    if event_times.size == 0 or reference_crossing_times.size == 0:
        return None
    errors = []
    for event_time in event_times:
        nearest = np.argmin(np.abs(reference_crossing_times - event_time))
        errors.append(abs(float(event_time - reference_crossing_times[nearest])))
    return float(np.median(errors))


def main() -> None:
    started = time.perf_counter()
    manifest, eligibility = build_manifest()
    core = CausalCecHTCore(core_config())
    ready_count = 0
    pooled_errors: list[np.ndarray] = []
    per_subject: dict[str, list[float]] = {}
    per_night: dict[str, list[float]] = {}
    target_stats: dict[str, dict[str, list]] = {
        str(round(target, 4)): {
            "errors": [],
            "event_times": [],
            "timing_errors": [],
        }
        for target in TARGET_GRID
    }

    for record in manifest:
        night = record["night"]
        subject = record["subject"]
        psg = DATA_ROOT / f"{night}-PSG.edf"
        signal = load_fpz_cz_100hz(psg) * 1e6
        ref_250 = noncausal_reference_resample(signal)
        ref_band = zero_phase_so_bandpass(ref_250, 250.0, l_freq=0.5, h_freq=1.25)
        ref_phase = hilbert_phase_rad(ref_band)

        for seg in record["segments_s"]:
            start_idx = int(round(seg["start_s"] * 100.0))
            end_idx = int(round(seg["end_s"] * 100.0))
            raw = signal[start_idx:end_idx]
            stream = run_adapted_stream(raw, adapter_config(), [10, 40, 100, 17, 250, 3])
            base = int(round(seg["start_s"] * 250.0))
            total = stream.values.size
            ref_slice = slice(max(0, base - ADAPTER_DELAY), base - ADAPTER_DELAY + total)
            ref_phase_seg = ref_phase[ref_slice]

            endpoints = endpoint_sequence_complex(stream.values, core.endpoint_kernel)
            phase_est = np.angle(endpoints)
            ready = stream.output_index >= (core.config.window_samples - 1)
            ref_idx = np.clip(
                stream.output_index - ADAPTER_DELAY, 0, ref_phase_seg.size - 1
            )
            error = wrap(phase_est[ready] - ref_phase_seg[ref_idx[ready]])
            ready_count += int(ready.sum())
            pooled_errors.append(error)
            per_subject.setdefault(subject, []).extend(error.tolist())
            per_night.setdefault(night, []).extend(error.tolist())

            for target in TARGET_GRID:
                key = str(round(target, 4))
                event_times = []
                prev = None
                prev_time = 0.0
                next_allowed = 0.0
                for idx in np.flatnonzero(ready):
                    current = phase_est[idx]
                    current_time = stream.output_index[idx] / 250.0
                    if prev is None:
                        prev = current
                        prev_time = current_time
                        continue
                    delta = (current - prev + math.pi) % (2.0 * math.pi) - math.pi
                    if delta > 0.0:
                        target_unwrapped = prev + (
                            (target - prev + math.pi) % (2.0 * math.pi) - math.pi
                        )
                        if prev <= target_unwrapped < prev + delta:
                            fraction = (target_unwrapped - prev) / delta
                            crossing = prev_time + fraction / 250.0
                            if crossing >= next_allowed:
                                event_times.append(crossing)
                                next_allowed = crossing + 0.5
                                ref_event_idx = int(
                                    round((crossing - ADAPTER_DELAY / 250.0) * 250.0)
                                )
                                ref_event_idx = int(
                                    np.clip(ref_event_idx, 0, ref_phase_seg.size - 1)
                                )
                                target_stats[key]["errors"].append(
                                    wrap(ref_phase_seg[ref_event_idx] - target)
                                )
                    prev = current
                    prev_time = current_time
                event_times = np.asarray(event_times, dtype=np.float64)
                ref_crossings = positive_crossing_times(ref_phase_seg, 250.0, target)
                target_stats[key]["event_times"].extend(event_times.tolist())
                timing = median_timing_error(event_times, ref_crossings)
                if timing is not None:
                    target_stats[key]["timing_errors"].append(timing)

    pooled_error = np.concatenate(pooled_errors)
    estimator = circular_abs_error_summary(pooled_error)
    estimator["ready_sample_count"] = ready_count
    estimator["mean_abs_wrapped_error_rad"] = float(np.mean(np.abs(pooled_error)))
    estimator["median_abs_wrapped_error_rad"] = float(np.median(np.abs(pooled_error)))
    estimator["q95_abs_wrapped_error_rad"] = float(np.quantile(np.abs(pooled_error), 0.95))

    subject_means = [
        float(np.mean(np.abs(wrap(np.asarray(values)))))
        for values in per_subject.values()
        if values
    ]
    subject_ci = (
        list(bootstrap_mean_ci(subject_means, n_boot=1000, seed=20260914))
        if subject_means
        else [None, None]
    )

    target_rows = []
    for target in TARGET_GRID:
        key = str(round(target, 4))
        errors = np.asarray(target_stats[key]["errors"], dtype=np.float64)
        times = np.asarray(target_stats[key]["event_times"], dtype=np.float64)
        event_count = int(times.size)
        timing_errors = target_stats[key]["timing_errors"]
        target_rows.append(
            {
                "target_rad": float(target),
                "target_deg": round(math.degrees(target), 2),
                "event_count": event_count,
                "hit_30": float(np.mean(np.abs(errors) <= math.pi / 6)) if errors.size else None,
                "hit_45": float(np.mean(np.abs(errors) <= math.pi / 4)) if errors.size else None,
                "circular_bias_rad": (
                    float(np.angle(np.mean(np.exp(1j * errors)))) if errors.size else None
                ),
                "median_timing_error_s": (
                    float(np.median(timing_errors)) if timing_errors else None
                ),
                "hit_45_wilson_ci": (
                    list(
                        binomial_proportion_ci(
                            int(np.sum(np.abs(errors) <= math.pi / 4)), event_count
                        )
                    )
                    if event_count
                    else [None, None]
                ),
            }
        )

    subject_rows = []
    for subject, values in sorted(per_subject.items()):
        arr = np.asarray(values, dtype=np.float64)
        summary = circular_abs_error_summary(arr)
        subject_rows.append(
            {
                "subject": subject,
                "sample_count": int(arr.size),
                "mean_abs_wrapped_error_rad": float(np.mean(np.abs(wrap(arr)))),
                "median_abs_wrapped_error_rad": float(np.median(np.abs(wrap(arr)))),
                "q95_abs_wrapped_error_rad": float(np.quantile(np.abs(wrap(arr)), 0.95)),
                "circular_bias_rad": summary["circular_bias_rad"],
                "resultant_length": summary["resultant_length"],
                "circular_sd_rad": summary["circular_sd_rad"],
            }
        )

    output = {
        "status": "baseline",
        "selected": SELECTED,
        "manifest_night_count": len(manifest),
        "estimator": estimator,
        "subject_level": {
            "subject_count": len(per_subject),
            "subject_mean_mae_rad": float(np.mean(subject_means)) if subject_means else None,
            "subject_mae_bootstrap_95ci": subject_ci,
        },
        "targets": target_rows,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (OUT / "baseline_complete_metrics.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (OUT / "baseline_subject_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subject_rows[0].keys()))
        writer.writeheader()
        writer.writerows(subject_rows)
    with (OUT / "baseline_target_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(target_rows[0].keys()))
        writer.writeheader()
        writer.writerows(target_rows)

    lines = [
        "# P0 frozen baseline",
        "",
        f"- selected: {SELECTED}",
        f"- ready samples: {ready_count}",
        f"- circular bias: {estimator['circular_bias_rad']:.6f} rad",
        f"- R: {estimator['resultant_length']:.6f}",
        f"- circular SD: {estimator['circular_sd_rad']:.6f} rad",
        f"- mean abs error: {estimator['mean_abs_wrapped_error_rad']:.6f} rad",
        f"- median abs error: {estimator['median_abs_wrapped_error_rad']:.6f} rad",
        f"- q95 abs error: {estimator['q95_abs_wrapped_error_rad']:.6f} rad",
        f"- subject mean MAE CI: {subject_ci}",
        "",
        "## VERDICT",
        "BASELINE_ONLY (no candidate comparison)",
        "",
        "## Isolation status",
        "- Main code modified: NO",
        "- Production config modified: NO",
        "- Untouched test accessed: NO",
        f"- Experiment directory: {OUT}",
    ]
    (OUT / "baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
