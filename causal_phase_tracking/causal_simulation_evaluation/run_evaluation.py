"""Development parameter-selection evaluation runner.

This is a development-only, shadow-only evaluation. It never opens untouched
test subjects and never uses cecHT output to choose the development segments.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.signal import fftconvolve

from causal_phase_tracking.causal_cecHT.streaming_core import CecHTCoreConfig

from .edf_replay import load_fpz_cz_100hz, qa_fpz_cz_signal, run_adapted_stream
from .fast_evaluation import endpoint_sequence_complex, pseudo_events_vectorized
from .offline_reference import (
    hilbert_phase_rad,
    noncausal_reference_resample,
    zero_phase_so_bandpass,
)
from .rate_adapter import RationalRateAdapterConfig
from .evaluation_manifest import choose_development_segments
from .evaluation_metrics import (
    binomial_proportion_ci,
    bootstrap_mean_ci,
    circular_abs_error_summary,
    subject_mean_metric,
    wrap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Folder holding Sleep-EDF Expanded night files (PSG + Hypnogram). Override with the
# PHASE_BENCHMARK_DATA_DIR environment variable to point at your own recording folder;
# the default stays inside the project so a handed-over copy carries no machine-specific
# absolute path.
DEFAULT_SLEEP_EDF_DIR = PROJECT_ROOT / "data" / "sleep_edf" / "default"
DATA_ROOT = Path(os.environ.get("PHASE_BENCHMARK_DATA_DIR", DEFAULT_SLEEP_EDF_DIR))

DEVELOPMENT_NIGHTS = [
    "SC4011E0", "SC4012E0", "SC4021E0", "SC4022E0", "SC4031E0", "SC4032E0",
    "SC4041E0", "SC4042E0", "SC4051E0", "SC4052E0", "SC4061E0", "SC4062E0",
    "SC4071E0", "SC4072E0", "SC4081E0", "SC4082E0", "SC4091E0", "SC4092E0",
    "SC4101E0", "SC4102E0",
]
SPARE_NIGHTS = [
    "SC4211E0", "SC4221E0", "SC4231E0", "SC4241E0", "SC4251E0", "SC4261F0",
    "SC4271F0", "SC4281G0", "SC4291G0", "SC4301E0", "SC4311E0", "SC4321E0",
    "SC4331F0", "SC4341F0", "SC4351F0", "SC4362F0", "SC4371F0", "SC4381F0",
    "SC4401E0", "SC4411E0", "SC4421E0", "SC4431E0", "SC4441E0", "SC4451F0",
    "SC4461F0", "SC4471F0", "SC4481F0", "SC4491G0", "SC4501E0", "SC4511E0",
    "SC4522E0", "SC4531E0", "SC4541F0", "SC4551F0", "SC4561F0", "SC4571F0",
    "SC4581G0", "SC4591G0", "SC4601E0", "SC4611E0", "SC4621E0", "SC4631E0",
    "SC4641E0", "SC4651E0", "SC4661E0", "SC4671G0", "SC4701E0", "SC4711E0",
    "SC4721E0", "SC4731E0",
]
SEGMENT_S = 180.0
SPACING_S = 30.0
SEGMENTS_PER_NIGHT = 3
ADAPTER_DELAY = 8
WINDOW_GRID_S = [4.0, 5.0, 6.0, 7.0, 8.0]
F0_GRID_HZ = [0.60, 0.70, 0.80, 0.875, 0.90, 1.00, 1.10, 1.20]
TARGET_GRID_RAD = [
    0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4,
    -math.pi, -3 * math.pi / 4, -math.pi / 2, -math.pi / 4,
]


def _hypnogram_for(night: str) -> Path | None:
    matches = sorted(DATA_ROOT.glob(f"{night[:6]}*Hypnogram.edf"))
    if len(matches) == 1:
        return matches[0]
    for path in matches:
        if path.name.startswith(night):
            return path
    return matches[0] if matches else None


def _stage4_blocks(path: Path) -> list[tuple[float, float, str]]:
    import mne

    annotations = mne.read_annotations(path)
    blocks = []
    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        if description in {"Sleep stage 3", "Sleep stage 4"}:
            blocks.append((float(onset), float(onset + duration), description))
    return blocks


def build_manifest() -> tuple[list[dict], dict]:
    nights = list(DEVELOPMENT_NIGHTS)
    fallback = list(SPARE_NIGHTS)
    manifest = []
    eligibility = {}
    while nights and len(manifest) < 20:
        night = nights.pop(0)
        hyp = _hypnogram_for(night)
        if hyp is None:
            eligibility[night] = {"eligible": False, "reason": "missing_hypnogram"}
            if not nights:
                nights = fallback
                fallback = []
            continue
        blocks = _stage4_blocks(hyp)
        record_end = max((end for _, end, _ in blocks), default=0.0)
        bounded_blocks = [
            (start, end, description)
            for start, end, description in blocks
            if start >= 300.0 and end <= record_end - 300.0
        ]
        segments = choose_development_segments(
            bounded_blocks,
            segment_s=SEGMENT_S,
            spacing_s=SPACING_S,
            count=SEGMENTS_PER_NIGHT,
        )
        if segments:
            manifest.append(
                {
                    "night": night,
                    "subject": night[:5],
                    "segments_s": [
                        {"start_s": round(start, 6), "end_s": round(end, 6)}
                        for start, end in segments
                    ],
                }
            )
            eligibility[night] = {"eligible": True, "segment_count": len(segments)}
        else:
            eligibility[night] = {"eligible": False, "reason": "fewer_than_3_stage4"}
            if not nights:
                nights = fallback
                fallback = []
    return manifest, eligibility


def _adapter_config() -> RationalRateAdapterConfig:
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
        config_version="evaluation-run-v1",
    )


def _core_config(window_s: float, f0: float | None) -> CecHTCoreConfig:
    window = int(round(window_s * 250.0))
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
        parameter_version="evaluation-grid-v1",
        config_version="evaluation-grid-v1",
        max_block_samples=1024,
    )


def _endpoint_complex(x: np.ndarray, core: "CausalCecHTCore") -> np.ndarray:
    window = core.config.window_samples
    kernel = core.endpoint_kernel
    b = kernel[::-1]
    conv = fftconvolve(x, b, mode="full")
    result = np.zeros(x.size, dtype=np.complex128)
    if x.size >= window:
        result[window - 1 :] = conv[window - 1 : x.size]
    return result


def _segment_candidate_metrics(
    stream_values: np.ndarray,
    stream_index: np.ndarray,
    reference_phase: np.ndarray,
    reference_analytic: np.ndarray,
    core: "CausalCecHTCore",
) -> dict:
    endpoints = _endpoint_complex(stream_values, core)
    ref_idx = np.clip(
        (stream_index.astype(np.int64) - ADAPTER_DELAY),
        0,
        reference_phase.size - 1,
    )
    ready = stream_index >= (core.config.window_samples - 1)
    phase_est = np.angle(endpoints)
    error = wrap(phase_est[ready] - reference_phase[ref_idx[ready]])
    summary = circular_abs_error_summary(error)
    summary["complex_mse_normalized"] = float(
        np.mean(
            np.abs(endpoints[ready] - reference_analytic[ref_idx[ready]]) ** 2
            / (np.abs(reference_analytic[ref_idx[ready]]) ** 2 + 1e-9)
        )
    )
    summary["hit_rate_within_30deg"] = float(np.mean(np.abs(error) <= math.pi / 6))
    return summary


def _trigger_metrics_for_targets(
    phase_est: np.ndarray,
    ready: np.ndarray,
    stream_index: np.ndarray,
    reference_phase: np.ndarray,
    targets: list[float],
) -> dict:
    ref_idx = np.clip(stream_index - ADAPTER_DELAY, 0, reference_phase.size - 1)
    result = {}
    hit_rates = []
    for target in targets:
        event = np.zeros(phase_est.size, dtype=np.bool_)
        prev = None
        prev_time = 0.0
        next_allowed = 0.0
        for idx in np.flatnonzero(ready):
            current = phase_est[idx]
            current_time = stream_index[idx] / 250.0
            if prev is None:
                prev = current
                prev_time = current_time
                continue
            delta = (current - prev + math.pi) % (2.0 * math.pi) - math.pi
            if delta > 0.0:
                target_unwrapped = prev + ((target - prev + math.pi) % (2.0 * math.pi) - math.pi)
                if prev <= target_unwrapped < prev + delta:
                    fraction = (target_unwrapped - prev) / delta
                    crossing = prev_time + fraction / 250.0
                    if crossing >= next_allowed:
                        event[idx] = True
                        next_allowed = crossing + 0.5
            prev = current
            prev_time = current_time
        errors = wrap(reference_phase[ref_idx[event]] - target) if event.any() else np.empty(0)
        hits = int(np.sum(np.abs(errors) <= math.pi / 6))
        total = errors.size
        hit_rates.append(hits / total if total else 1.0)
        result[str(round(target, 4))] = {
            "event_count": int(event.sum()),
            "hit_rate_within_30deg": hits / total if total else None,
            "circular_bias_rad": (
                circular_abs_error_summary(errors)["circular_bias_rad"] if total else None
            ),
        }
    result["worst_hit_rate"] = float(min(hit_rates))
    return result


def main() -> None:
    started = time.perf_counter()
    from causal_phase_tracking.causal_cecHT.streaming_core import CausalCecHTCore

    manifest, eligibility = build_manifest()
    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "development_selection.json"
    if len(manifest) < 20:
        blocked = {
            "status": "blocked_insufficient_eligible_nights",
            "eligible_night_count": len(manifest),
            "required_night_count": 20,
            "manifest": manifest,
            "eligibility": eligibility,
            "reason": (
                "Under the approved S8-D1 rule (at least 3 Stage 4 blocks >=180 s "
                "per night, no Stage 2 fill), fewer than 20 development nights are "
                "eligible in the available Sleep-EDF set."
            ),
        }
        out_path.write_text(
            json.dumps(blocked, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("wrote", out_path)
        print(json.dumps(blocked, indent=2, ensure_ascii=False))
        return
    subjects = sorted({record["subject"] for record in manifest})
    training_subjects = set(subjects[:-2])
    validation_subjects = set(subjects[-2:])
    coarse_candidates = []
    for window_s in WINDOW_GRID_S:
        coarse_candidates.append(("uncalibrated", window_s, None))
        for f0 in F0_GRID_HZ:
            coarse_candidates.append(("calibrated", window_s, f0))

    per_segment = {}
    excluded_segments = 0
    for record in manifest:
        night = record["night"]
        psg = DATA_ROOT / f"{night}-PSG.edf"
        signal = load_fpz_cz_100hz(psg) * 1e6
        ref_250 = noncausal_reference_resample(signal)
        ref_band = zero_phase_so_bandpass(ref_250, 250.0, l_freq=0.5, h_freq=1.25)
        ref_phase = hilbert_phase_rad(ref_band)
        from scipy.signal import hilbert as _scipy_hilbert

        ref_analytic = np.asarray(_scipy_hilbert(ref_band), dtype=np.complex128)
        for seg in record["segments_s"]:
            start_s = seg["start_s"]
            end_s = seg["end_s"]
            start_idx = int(round(start_s * 100.0))
            end_idx = int(round(end_s * 100.0))
            raw = signal[start_idx:end_idx]
            qa = qa_fpz_cz_signal(raw, sampling_rate_hz=100.0)
            p2p = float(np.ptp(raw[np.isfinite(raw)])) if np.isfinite(raw).any() else 0.0
            if not (
                qa["nonfinite_count"] == 0
                and qa["rail_count"] == 0
                and qa["longest_flat_run_s"] <= 1.0
                and qa["derivative_mad_outlier_ratio"] < 0.01
                and 20.0 <= p2p <= 400.0
            ):
                excluded_segments += 1
                continue
            stream = run_adapted_stream(raw, _adapter_config(), [10, 40, 100, 17, 250, 3])
            base = int(round(start_s * 250.0))
            total = stream.values.size
            ref_slice = slice(max(0, base - ADAPTER_DELAY), base - ADAPTER_DELAY + total)
            key = f"{night}_{int(start_s)}"
            per_segment[key] = {
                "qa": qa,
                "stream_values": stream.values,
                "stream_index": stream.output_index,
                "ref_phase": ref_phase[ref_slice],
                "ref_analytic": ref_analytic[ref_slice],
            }

    coarse_metrics = []
    for branch, window_s, f0 in coarse_candidates:
        core = CausalCecHTCore(_core_config(window_s, f0))
        subject_metrics = {}
        for record in manifest:
            subject = record["subject"]
            if subject not in training_subjects:
                continue
            values = []
            for seg in record["segments_s"]:
                key = f"{record['night']}_{int(seg['start_s'])}"
                if key not in per_segment:
                    continue
                item = per_segment[key]
                summary = _segment_candidate_metrics(
                    item["stream_values"],
                    item["stream_index"],
                    item["ref_phase"],
                    item["ref_analytic"],
                    core,
                )
                values.append(summary["mean_abs_wrapped_error_rad"])
            subject_metrics.setdefault(subject, []).extend(values)
        subject_values = [
            float(np.mean(values)) for values in subject_metrics.values() if values
        ]
        coarse_metrics.append(
            {
                "branch": branch,
                "window_s": window_s,
                "f0_hz": f0,
                "subject_mean_mae_rad": (
                    float(np.mean(subject_values)) if subject_values else math.inf
                ),
            }
        )

    finite_coarse = [item for item in coarse_metrics if math.isfinite(item["subject_mean_mae_rad"])]
    if not finite_coarse:
        blocked = {
            "status": "blocked_no_finite_candidates_after_qa",
            "manifest": manifest,
            "eligibility": eligibility,
            "excluded_segments_by_qa": excluded_segments,
            "reason": "All coarse candidates were excluded by the S8-D1 QA gate.",
        }
        out_path.write_text(
            json.dumps(blocked, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("wrote", out_path)
        print(json.dumps(blocked, indent=2, ensure_ascii=False))
        return
    ranked = sorted(finite_coarse, key=lambda item: item["subject_mean_mae_rad"])
    best_two_windows = sorted(
        {item["window_s"] for item in ranked[:6]}, key=lambda w: next(
            (m["subject_mean_mae_rad"] for m in ranked if m["window_s"] == w), math.inf
        )
    )[:2]
    best_two_f0 = []
    for item in ranked:
        if (
            item["branch"] == "calibrated"
            and item["f0_hz"] is not None
            and item["f0_hz"] not in best_two_f0
        ):
            best_two_f0.append(item["f0_hz"])
        if len(best_two_f0) == 2:
            break

    local_windows = sorted(set(best_two_windows))
    local_f0 = sorted(set(best_two_f0))
    if len(local_windows) == 2 and local_windows[1] - local_windows[0] == 1.0:
        local_windows = [local_windows[0], local_windows[0] + 0.5, local_windows[1]]
    if len(local_f0) == 2:
        midpoint = round((local_f0[0] + local_f0[1]) / 2.0, 3)
        local_f0 = sorted(set(local_f0 + [midpoint]))

    local_candidates = []
    for window_s in local_windows:
        local_candidates.append(("uncalibrated", window_s, None))
        for f0 in local_f0:
            local_candidates.append(("calibrated", window_s, f0))

    local_metrics = []
    for branch, window_s, f0 in local_candidates:
        core = CausalCecHTCore(_core_config(window_s, f0))
        per_subject = {}
        trigger_hit_rates = []
        trigger_biases = []
        for record in manifest:
            subject = record["subject"]
            if subject not in validation_subjects:
                continue
            vals = []
            for seg in record["segments_s"]:
                key = f"{record['night']}_{int(seg['start_s'])}"
                if key not in per_segment:
                    continue
                item = per_segment[key]
                summary = _segment_candidate_metrics(
                    item["stream_values"],
                    item["stream_index"],
                    item["ref_phase"],
                    item["ref_analytic"],
                    core,
                )
                vals.append(summary["mean_abs_wrapped_error_rad"])
                endpoints = _endpoint_complex(item["stream_values"], core)
                ready = item["stream_index"] >= (core.config.window_samples - 1)
                trigger = _trigger_metrics_for_targets(
                    np.angle(endpoints),
                    ready,
                    item["stream_index"],
                    item["ref_phase"],
                    TARGET_GRID_RAD,
                )
                trigger_hit_rates.append(trigger["worst_hit_rate"])
                default = trigger[str(0.0)]
                if default["circular_bias_rad"] is not None:
                    trigger_biases.append(default["circular_bias_rad"])
            per_subject.setdefault(subject, []).extend(vals)
        subject_values = [float(np.mean(v)) for v in per_subject.values() if v]
        local_metrics.append(
            {
                "branch": branch,
                "window_s": window_s,
                "f0_hz": f0,
                "subject_mean_mae_rad": float(np.mean(subject_values)),
                "subject_mae_bootstrap_95ci": list(
                    bootstrap_mean_ci(subject_values, n_boot=1000, seed=20260911)
                ) if subject_values else [None, None],
                "validation_trigger_worst_hit_rate": (
                    float(np.mean(trigger_hit_rates)) if trigger_hit_rates else None
                ),
                "validation_default_target_trigger_bias_rad": (
                    float(np.mean(trigger_biases)) if trigger_biases else None
                ),
            }
        )

    finite_local = [
        item
        for item in local_metrics
        if isinstance(item["subject_mean_mae_rad"], (int, float))
        and math.isfinite(item["subject_mean_mae_rad"])
    ]
    if not finite_local:
        blocked = {
            "status": "blocked_no_finite_local_candidates",
            "manifest": manifest,
            "eligibility": eligibility,
            "training_subjects": sorted(training_subjects),
            "validation_subjects": sorted(validation_subjects),
            "excluded_segments_by_qa": excluded_segments,
            "reason": "Local refinement candidates had no finite validation subject metrics.",
        }
        out_path.write_text(
            json.dumps(blocked, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("wrote", out_path)
        print(json.dumps(blocked, indent=2, ensure_ascii=False))
        return
    final_rank = sorted(finite_local, key=lambda item: item["subject_mean_mae_rad"])
    final = final_rank[0]
    output = {
        "manifest": manifest,
        "eligibility": eligibility,
        "excluded_segments_by_qa": excluded_segments,
        "training_subjects": sorted(training_subjects),
        "validation_subjects": sorted(validation_subjects),
        "coarse_candidate_count": len(coarse_candidates),
        "coarse_top_10": ranked[:10],
        "best_two_windows_s": best_two_windows,
        "best_two_f0_hz": best_two_f0,
        "local_grid": {"windows_s": local_windows, "f0_hz": local_f0},
        "local_metrics": local_metrics,
        "selected": final,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out_path)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
