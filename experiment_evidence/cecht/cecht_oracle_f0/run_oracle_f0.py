"""P3 oracle f0 ablation for the frozen 8 s causal cecHT baseline.

The cecHT endpoint kernel factorizes into an f0-independent base kernel and a
complex calibration gain. This experiment exploits that factorization to build
recording-level and local oracle f0 diagnostics without modifying estimator
code or production config.
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
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert, welch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_cecHT.streaming_core import CecHTCoreConfig, CausalCecHTCore
from causal_phase_tracking.causal_simulation_evaluation.edf_replay import (
    load_fpz_cz_100hz,
    run_adapted_stream,
)
from causal_phase_tracking.causal_simulation_evaluation.fast_evaluation import (
    endpoint_sequence_complex,
    pseudo_events_vectorized,
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
FIXED_F0 = 0.8
WINDOW_S = 8.0
ADAPTER_DELAY = 8
F0_GRID = np.round(np.arange(0.50, 1.2501, 0.01), 4)


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
        config_version="p3-oracle-v1",
    )


def core_config(f0: float | None) -> CecHTCoreConfig:
    window = int(round(WINDOW_S * 250.0))
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
        calibrate=f0 is not None,
        f0=f0,
        parameter_version="p3-oracle-v1",
        config_version="p3-oracle-v1",
        max_block_samples=1024,
    )


def build_gain_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_core = CausalCecHTCore(core_config(None))
    base_kernel = base_core.endpoint_kernel
    probe_index = int(np.argmax(np.abs(base_kernel)))
    gains = []
    for f0 in F0_GRID:
        calibrated = CausalCecHTCore(core_config(float(f0))).endpoint_kernel
        gain = calibrated[probe_index] / base_kernel[probe_index]
        gains.append(gain)
    gains = np.asarray(gains, dtype=np.complex128)
    # Audit the factorization for a subset of grid points.
    for index in (0, len(F0_GRID) // 3, 2 * len(F0_GRID) // 3, len(F0_GRID) - 1):
        reconstructed = base_kernel * gains[index]
        calibrated = CausalCecHTCore(core_config(float(F0_GRID[index]))).endpoint_kernel
        if not np.allclose(reconstructed, calibrated, rtol=0.0, atol=2e-12):
            raise AssertionError("base kernel x calibration gain factorization failed")
    return base_kernel, gains, F0_GRID


def gain_for(f0: np.ndarray | float, gains: np.ndarray) -> np.ndarray:
    values = np.atleast_1d(np.asarray(f0, dtype=np.float64))
    clipped = np.clip(values, F0_GRID[0], F0_GRID[-1])
    real = np.interp(clipped, F0_GRID, gains.real)
    imag = np.interp(clipped, F0_GRID, gains.imag)
    return real + 1j * imag


def instantaneous_frequency(phase: np.ndarray, fs: float) -> np.ndarray:
    unwrapped = np.unwrap(phase)
    diff = np.diff(unwrapped, prepend=unwrapped[0])
    values = diff * fs / (2.0 * math.pi)
    return uniform_filter1d(values, size=int(round(fs)), mode="nearest")


def dominant_frequency(signal: np.ndarray, fs: float) -> float:
    frequency, power = welch(signal, fs=fs, nperseg=min(8192, signal.size))
    mask = (frequency >= 0.5) & (frequency <= 1.25)
    if not mask.any():
        return FIXED_F0
    return float(frequency[mask][int(np.argmax(power[mask]))])


def crossing_times(phase: np.ndarray, fs: float, target: float) -> np.ndarray:
    times = []
    prev = phase[0]
    prev_time = 0.0
    for index in range(1, phase.size):
        current = phase[index]
        delta = (current - prev + math.pi) % (2.0 * math.pi) - math.pi
        current_time = index / fs
        if delta > 0.0:
            target_unwrapped = prev + ((target - prev + math.pi) % (2.0 * math.pi) - math.pi)
            if prev <= target_unwrapped < prev + delta:
                fraction = (target_unwrapped - prev) / delta
                times.append(prev_time + fraction / fs)
        prev = current
        prev_time = current_time
    return np.asarray(times, dtype=np.float64)


def median_timing(event_times: np.ndarray, reference_crossings: np.ndarray) -> float | None:
    if event_times.size == 0 or reference_crossings.size == 0:
        return None
    errors = [
        abs(float(event_time - reference_crossings[np.argmin(np.abs(reference_crossings - event_time))]))
        for event_time in event_times
    ]
    return float(np.median(errors))


def summary(errors: np.ndarray) -> dict:
    result = circular_abs_error_summary(errors)
    absolute = np.abs(wrap(errors))
    result.update(
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
    return result


def main() -> None:
    started = time.perf_counter()
    base_kernel, gains, _ = build_gain_table()
    manifest, _ = build_manifest()
    conditions = ("fixed", "recording_oracle", "local_oracle")
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in conditions}
    per_subject: dict[str, dict[str, list[float]]] = {name: {} for name in conditions}
    timing_default: dict[str, list[float]] = {name: [] for name in conditions}
    night_f0: dict[str, float] = {}

    for record in manifest:
        night = record["night"]
        subject = record["subject"]
        signal = load_fpz_cz_100hz(DATA_ROOT / f"{night}-PSG.edf") * 1e6
        ref_250 = noncausal_reference_resample(signal)
        ref_band = zero_phase_so_bandpass(ref_250, 250.0, l_freq=0.5, h_freq=1.25)
        ref_phase = hilbert_phase_rad(ref_band)
        f_inst = instantaneous_frequency(ref_phase, 250.0)
        recording_f0 = dominant_frequency(ref_band, 250.0)
        night_f0[night] = recording_f0

        for seg in record["segments_s"]:
            start_idx = int(round(seg["start_s"] * 100.0))
            end_idx = int(round(seg["end_s"] * 100.0))
            raw = signal[start_idx:end_idx]
            stream = run_adapted_stream(raw, adapter_config(), [10, 40, 100, 17, 250, 3])
            base_endpoint = endpoint_sequence_complex(stream.values, base_kernel)
            ready = stream.output_index >= (int(round(WINDOW_S * 250.0)) - 1)
            base_idx = int(round(seg["start_s"] * 250.0)) + stream.output_index - ADAPTER_DELAY
            ref_idx = np.clip(base_idx, 0, ref_phase.size - 1)
            local_f0 = np.clip(f_inst[ref_idx], 0.5, 1.25)
            reference = ref_phase[ref_idx]

            condition_phase = {
                "fixed": base_endpoint * gain_for(FIXED_F0, gains)[0],
                "recording_oracle": base_endpoint * gain_for(recording_f0, gains)[0],
                "local_oracle": base_endpoint * gain_for(local_f0, gains),
            }
            for name in conditions:
                phase = np.angle(condition_phase[name])
                error = wrap(phase[ready] - reference[ready])
                pooled[name].append(error)
                per_subject[name].setdefault(subject, []).extend(error.tolist())

            ref_ready = reference[ready]
            ref_crossings = crossing_times(ref_ready, 250.0, 0.0)
            for name in conditions:
                phase = np.angle(condition_phase[name][ready])
                event, crossing = pseudo_events_vectorized(
                    phase, fs=250.0, target_rad=0.0, refractory_s=0.5
                )
                timing = median_timing(crossing[event], ref_crossings)
                if timing is not None:
                    timing_default[name].append(timing)

    condition_summary = {}
    subject_values = {}
    for name in conditions:
        errors = np.concatenate(pooled[name])
        condition_summary[name] = summary(errors)
        subject_values[name] = [
            float(np.mean(np.abs(wrap(np.asarray(values)))))
            for values in per_subject[name].values()
        ]
        condition_summary[name]["median_absolute_timing_error_s"] = (
            float(np.median(timing_default[name])) if timing_default[name] else None
        )
        condition_summary[name]["subject_mean_mae_rad"] = float(np.mean(subject_values[name]))
        condition_summary[name]["subject_mae_bootstrap_95ci"] = list(
            bootstrap_mean_ci(subject_values[name], n_boot=1000, seed=20260914)
        )

    fixed = condition_summary["fixed"]
    deltas = {}
    for name in ("recording_oracle", "local_oracle"):
        candidate = condition_summary[name]
        deltas[f"delta_{name}"] = {
            "delta_mae_rad": fixed["mean_abs_wrapped_error_rad"]
            - candidate["mean_abs_wrapped_error_rad"],
            "delta_q95_rad": fixed["q95_abs_wrapped_error_rad"]
            - candidate["q95_abs_wrapped_error_rad"],
            "delta_resultant_length": candidate["resultant_length"]
            - fixed["resultant_length"],
            "delta_hit_30": candidate["hit_30"] - fixed["hit_30"],
            "delta_hit_45": candidate["hit_45"] - fixed["hit_45"],
            "delta_catastrophic_rate": candidate["catastrophic_rate_gt_135deg"]
            - fixed["catastrophic_rate_gt_135deg"],
        }
        fixed_subject = subject_values["fixed"]
        candidate_subject = subject_values[name]
        improved = sum(1 for a, b in zip(fixed_subject, candidate_subject) if b < a)
        worsened = sum(1 for a, b in zip(fixed_subject, candidate_subject) if b > a)
        deltas[f"delta_{name}"]["subjects_improved"] = improved
        deltas[f"delta_{name}"]["subjects_worsened"] = worsened

    report = {
        "fixed_f0_hz": FIXED_F0,
        "night_recording_oracle_f0_hz": night_f0,
        "conditions": condition_summary,
        "deltas": deltas,
        "isolation": {
            "main_code_modified": "NO",
            "production_config_modified": "NO",
            "untouched_test_accessed": "NO",
            "verdict": "DIAGNOSTIC",
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (OUT / "oracle_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (OUT / "condition_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["condition"] + list(condition_summary["fixed"].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in conditions:
            writer.writerow({"condition": name, **condition_summary[name]})

    recording_improvement = deltas["delta_recording_oracle"]["delta_mae_rad"]
    local_improvement = deltas["delta_local_oracle"]["delta_mae_rad"]
    personalization_sufficient = bool(
        recording_improvement > 0.05 and local_improvement - recording_improvement < 0.02
    )
    dynamic_justified = bool(local_improvement - recording_improvement > 0.05)
    lines = [
        "# P3 oracle f0 ablation",
        "",
        f"- fixed MAE: {fixed['mean_abs_wrapped_error_rad']:.6f} rad",
        f"- recording oracle MAE: {condition_summary['recording_oracle']['mean_abs_wrapped_error_rad']:.6f} rad",
        f"- local oracle MAE: {condition_summary['local_oracle']['mean_abs_wrapped_error_rad']:.6f} rad",
        f"- delta MAE recording: {recording_improvement:.6f} rad",
        f"- delta MAE local: {local_improvement:.6f} rad",
        f"- recording oracle improved subjects: {deltas['delta_recording_oracle']['subjects_improved']}",
        f"- local oracle improved subjects: {deltas['delta_local_oracle']['subjects_improved']}",
        f"- Recording-level personalization sufficient: {'YES' if personalization_sufficient else 'NO'}",
        f"- Dynamic f0 justified: {'YES' if dynamic_justified else 'NO'}",
        "",
        "## Isolation status",
        "- Main code modified: NO",
        "- Production config modified: NO",
        "- Untouched test accessed: NO",
        f"- Experiment directory: {OUT}",
        "",
        "## References",
        "1. Osmers E, Kolossa D. Optimal Calibration of the Endpoint-corrected Hilbert Transform. arXiv:2601.13962v2 (2026). https://arxiv.org/abs/2601.13962",
        "2. Santostasi G et al. Phase-locked loop for precisely timed acoustic stimulation during sleep. J Neurosci Methods 259, 101-114 (2016). https://doi.org/10.1016/j.jneumeth.2015.11.007",
        "3. Ferster ML et al. Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation. IEEE TBME 69(9), 2916-2925 (2022). https://doi.org/10.1109/TBME.2022.3157468",
        "4. scipy.signal.welch documentation for PSD peak estimation. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
