"""PLL P0 baseline using the frozen cecHT evaluation protocol.

The PLL runs at its native 100 Hz contract while sharing the same development
subjects, segments, offline reference, warm-up semantics, target/refractory
definition, metric functions and subject-level aggregation as the cecHT
experiments. No main code, production config, or frozen test manifest is
modified.
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
from scipy.signal import butter, hilbert, sosfilt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# The reduced offline PLL module lives in the isolated benchmark tree, which is not
# importable as a repository-root package on its own.
sys.path.insert(0, str(ROOT / "offline_benchmark"))

from offline_src.phase_benchmark.pll import run_classic_pll_core
from causal_phase_tracking.causal_simulation_evaluation.edf_replay import load_fpz_cz_100hz
from causal_phase_tracking.causal_simulation_evaluation.offline_reference import (
    hilbert_phase_rad,
    noncausal_reference_resample,
    zero_phase_so_bandpass,
)
from causal_phase_tracking.causal_simulation_evaluation.evaluation_metrics import (
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
CECHT_P0 = ROOT / "experiment_evidence" / "cecht" / "cecht_baseline" / "baseline_complete_metrics.json"
CECHT_P0_SUBJECTS = (
    ROOT / "experiment_evidence" / "cecht" / "cecht_baseline" / "baseline_subject_metrics.csv"
)
CECHT_P3 = ROOT / "experiment_evidence" / "cecht" / "cecht_oracle_f0" / "oracle_report.json"
CECHT_P1 = (
    ROOT
    / "experiment_evidence"
    / "cecht"
    / "cecht_stratification"
    / "stratification_report.json"
)

FS = 100.0
WARMUP_S = 8.0
REFRACTORY_S = 0.5
TARGET = 0.0

# Classic-PLL configuration.
# This package is the result of a research internship in industry; the classic
# PLL module and its frozen tuning parameters are not included in this
# distribution for intellectual-property reasons.
PLL_WITHHELD = (
    "the classic PLL module and its frozen tuning parameters are not included "
    "in this distribution for intellectual-property reasons"
)
PLL_FC = None
PLL_KVCO = None
PLL_KA = None
PLL_LOOP_B = None
PLL_LOOP_A = None
PLL_BAND = None
PLL_FILTER_ORDER = None
PLL_NORMALIZE_PERCENTILE = None
PLL_PHASE_OFFSET = None


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


def event_metrics(phase: np.ndarray, fs: float, reference_phase_at_events: np.ndarray) -> dict:
    from causal_phase_tracking.causal_simulation_evaluation.fast_evaluation import (
        pseudo_events_vectorized,
    )

    event, crossing = pseudo_events_vectorized(
        phase, fs=fs, target_rad=TARGET, refractory_s=REFRACTORY_S
    )
    event_phase_errors = wrap(reference_phase_at_events[event]) if event.any() else np.empty(0)
    reference_crossings = crossing_times(reference_phase_at_events, fs, TARGET)
    timing_errors = []
    if reference_crossings.size:
        for event_time in crossing[event]:
            nearest = int(np.argmin(np.abs(reference_crossings - event_time)))
            timing_errors.append(abs(float(event_time - reference_crossings[nearest])))
    return {
        "event_count": int(event.sum()),
        "hit_30": float(np.mean(np.abs(event_phase_errors) <= math.pi / 6))
        if event_phase_errors.size
        else None,
        "hit_45": float(np.mean(np.abs(event_phase_errors) <= math.pi / 4))
        if event_phase_errors.size
        else None,
        "circular_bias_rad": float(np.angle(np.mean(np.exp(1j * event_phase_errors))))
        if event_phase_errors.size
        else None,
        "median_absolute_timing_error_s": float(np.median(timing_errors))
        if timing_errors
        else None,
    }


def summary(error: np.ndarray) -> dict:
    result = circular_abs_error_summary(error)
    absolute = np.abs(wrap(error))
    result.update(
        {
            "mean_abs_wrapped_error_rad": float(np.mean(absolute)),
            "median_abs_wrapped_error_rad": float(np.median(absolute)),
            "q95_abs_wrapped_error_rad": float(np.quantile(absolute, 0.95)),
            "hit_30": float(np.mean(absolute <= math.pi / 6)),
            "hit_45": float(np.mean(absolute <= math.pi / 4)),
            "catastrophic_rate_gt_135deg": float(np.mean(absolute > 3 * math.pi / 4)),
            "sample_count": int(error.size),
        }
    )
    return result


def main() -> None:
    raise NotImplementedError(PLL_WITHHELD)

    started = time.perf_counter()
    manifest, _ = build_manifest()
    pooled_errors = []
    pooled_errors_no_offset = []
    per_subject: dict[str, list[float]] = {}
    event_summaries = []
    frequency_stats = []
    control_stats = []

    for record in manifest:
        night = record["night"]
        subject = record["subject"]
        signal = load_fpz_cz_100hz(DATA_ROOT / f"{night}-PSG.edf") * 1e6
        ref_250 = noncausal_reference_resample(signal)
        ref_band = zero_phase_so_bandpass(ref_250, 250.0, l_freq=0.5, h_freq=1.25)
        ref_phase_250 = hilbert_phase_rad(ref_band)

        for seg in record["segments_s"]:
            start_idx = int(round(seg["start_s"] * 100.0))
            end_idx = int(round(seg["end_s"] * 100.0))
            raw = signal[start_idx:end_idx]
            sos = butter(
                PLL_FILTER_ORDER, PLL_BAND, btype="band", fs=FS, output="sos"
            )
            filtered = sosfilt(sos, raw)
            scale = float(np.nanpercentile(np.abs(filtered), PLL_NORMALIZE_PERCENTILE))
            if not np.isfinite(scale) or scale <= 0:
                scale = float(np.std(filtered)) or 1.0
            pll = run_classic_pll_core(
                filtered / scale,
                sfreq=FS,
                fc=PLL_FC,
                kvco=PLL_KVCO,
                ka=PLL_KA,
                loop_b=PLL_LOOP_B,
                loop_a=PLL_LOOP_A,
            )
            phase_internal = pll["phase"]
            phase_reference_convention = wrap(phase_internal + PLL_PHASE_OFFSET)
            n = raw.size
            times = (seg["start_s"] + np.arange(n) / FS)
            ref_idx = np.clip(np.rint(times * 250.0).astype(np.int64), 0, ref_phase_250.size - 1)
            reference_phase = ref_phase_250[ref_idx]
            ready = np.arange(n) >= int(WARMUP_S * FS)
            error = wrap(phase_reference_convention[ready] - reference_phase[ready])
            error_no_offset = wrap(phase_internal[ready] - reference_phase[ready])
            pooled_errors.append(error)
            pooled_errors_no_offset.append(error_no_offset)
            per_subject.setdefault(subject, []).extend(error.tolist())
            event_summaries.append(
                event_metrics(phase_reference_convention, FS, reference_phase)
            )
            frequency_stats.append(pll["freq_inst"][ready])
            control_stats.append(pll["control"][ready])

    errors = np.concatenate(pooled_errors)
    errors_no_offset = np.concatenate(pooled_errors_no_offset)
    estimator = summary(errors)
    estimator_no_offset = summary(errors_no_offset)
    estimator_no_offset["note"] = "raw PLL internal phase without historical phase_offset"

    subject_values = [
        float(np.mean(np.abs(wrap(np.asarray(values)))))
        for values in per_subject.values()
    ]
    subject_ci = list(bootstrap_mean_ci(subject_values, n_boot=1000, seed=20260914))

    events = {
        key: float(np.nanmean([item[key] for item in event_summaries if item[key] is not None]))
        if any(item[key] is not None for item in event_summaries)
        else None
        for key in ("hit_30", "hit_45", "circular_bias_rad", "median_absolute_timing_error_s")
    }
    events["event_count"] = int(sum(item["event_count"] for item in event_summaries))

    freq_all = np.concatenate(frequency_stats)
    ctrl_all = np.concatenate(control_stats)
    internal_state = {
        "estimated_frequency_mean_hz": float(np.mean(freq_all)),
        "estimated_frequency_sd_hz": float(np.std(freq_all)),
        "estimated_frequency_jitter_hz": float(np.std(np.diff(freq_all))),
        "fraction_abs_freq_minus_fc_gt_0p2": float(np.mean(np.abs(freq_all - PLL_FC) > 0.2)),
        "control_mean": float(np.mean(ctrl_all)),
        "control_sd": float(np.std(ctrl_all)),
    }

    cecHT_p0 = json.loads(CECHT_P0.read_text(encoding="utf-8"))
    cecHT_p3 = json.loads(CECHT_P3.read_text(encoding="utf-8"))
    cecHT_p1 = json.loads(CECHT_P1.read_text(encoding="utf-8"))
    cecHT_fixed = cecHT_p0["estimator"]
    cecHT_oracle = cecHT_p3["conditions"]["local_oracle"]
    cecHT_pooled = cecHT_p1["pooled"]

    delta_vs_fixed = {
        "delta_mae_rad": cecHT_fixed["mean_abs_wrapped_error_rad"]
        - estimator["mean_abs_wrapped_error_rad"],
        "delta_q95_rad": cecHT_fixed["q95_abs_wrapped_error_rad"]
        - estimator["q95_abs_wrapped_error_rad"],
        "delta_resultant_length": estimator["resultant_length"]
        - cecHT_fixed["resultant_length"],
        "delta_hit_30": estimator["hit_30"] - cecHT_pooled["hit_30"],
        "delta_hit_45": estimator["hit_45"] - cecHT_pooled["hit_45"],
        "delta_catastrophic_rate": estimator["catastrophic_rate_gt_135deg"]
        - cecHT_pooled["catastrophic_rate_gt_135deg"],
    }
    cecHT_subject_map = {}
    with CECHT_P0_SUBJECTS.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cecHT_subject_map[row["subject"]] = float(
                row["mean_abs_wrapped_error_rad"]
            )
    improved = None
    worsened = None
    common_subjects = sorted(set(cecHT_subject_map) & set(per_subject))
    if common_subjects:
        improved = sum(
            1
            for subject in common_subjects
            if float(np.mean(np.abs(wrap(np.asarray(per_subject[subject])))))
            < cecHT_subject_map[subject]
        )
        worsened = sum(
            1
            for subject in common_subjects
            if float(np.mean(np.abs(wrap(np.asarray(per_subject[subject])))))
            > cecHT_subject_map[subject]
        )

    gap_fixed = cecHT_fixed["mean_abs_wrapped_error_rad"]
    gap_oracle = cecHT_oracle["mean_abs_wrapped_error_rad"]
    gap_closed = None
    if gap_fixed > gap_oracle:
        gap_closed = float((gap_fixed - estimator["mean_abs_wrapped_error_rad"]) / (gap_fixed - gap_oracle))

    report = {
        "config": {
            "fs_hz": FS,
            "fc_hz": PLL_FC,
            "kvco": PLL_KVCO,
            "ka": PLL_KA,
            "loop_b": PLL_LOOP_B,
            "loop_a": PLL_LOOP_A,
            "band_hz": PLL_BAND,
            "filter_order": PLL_FILTER_ORDER,
            "normalize_percentile": PLL_NORMALIZE_PERCENTILE,
            "phase_offset": PLL_PHASE_OFFSET,
            "warmup_s": WARMUP_S,
            "refractory_s": REFRACTORY_S,
        },
        "estimator": estimator,
        "estimator_without_phase_offset": estimator_no_offset,
        "subject_level": {
            "subject_count": len(per_subject),
            "subject_mean_mae_rad": float(np.mean(subject_values)),
            "subject_mae_bootstrap_95ci": subject_ci,
        },
        "events": events,
        "internal_state": internal_state,
        "cecht_fixed_baseline_mae_rad": gap_fixed,
        "cecht_local_oracle_mae_rad": gap_oracle,
        "delta_vs_cecht_fixed": delta_vs_fixed,
        "fraction_of_fixed_to_oracle_gap_closed": gap_closed,
        "subjects_improved_vs_cecht": improved,
        "subjects_worsened_vs_cecht": worsened,
        "isolation": {
            "main_code_modified": "NO",
            "production_config_modified": "NO",
            "untouched_test_accessed": "NO",
            "verdict": "BASELINE_ONLY",
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (OUT / "pll_complete_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (OUT / "pll_subject_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "mean_abs_wrapped_error_rad", "sample_count"])
        for subject, values in sorted(per_subject.items()):
            writer.writerow([subject, float(np.mean(np.abs(wrap(np.asarray(values))))), len(values)])

    lines = [
        "# PLL P0 baseline",
        "",
        f"- PLL MAE: {estimator['mean_abs_wrapped_error_rad']:.6f} rad",
        f"- PLL q95: {estimator['q95_abs_wrapped_error_rad']:.6f} rad",
        f"- PLL R: {estimator['resultant_length']:.6f}",
        f"- PLL circular SD: {estimator['circular_sd_rad']:.6f} rad",
        f"- PLL ±45° hit: {estimator['hit_45']:.6f}",
        f"- cecHT fixed MAE: {gap_fixed:.6f} rad",
        f"- cecHT local-oracle MAE: {gap_oracle:.6f} rad",
        f"- delta MAE vs cecHT fixed: {delta_vs_fixed['delta_mae_rad']:.6f} rad",
        f"- fixed-to-oracle gap closed: {gap_closed if gap_closed is None else round(gap_closed, 4)}",
        f"- subjects improved vs cecHT: {improved}",
        f"- internal estimated frequency mean: {internal_state['estimated_frequency_mean_hz']:.6f} Hz",
        f"- internal frequency SD: {internal_state['estimated_frequency_sd_hz']:.6f} Hz",
        "",
        "## Isolation status",
        "- Main code modified: NO",
        "- Production config modified: NO",
        "- Untouched test accessed: NO",
        f"- Experiment directory: {OUT}",
        "",
        "## References",
        "1. Santostasi G et al. Phase-locked loop for precisely timed acoustic stimulation during sleep. J Neurosci Methods 259, 101-114 (2016). https://doi.org/10.1016/j.jneumeth.2015.11.007",
        "2. Ferster ML et al. Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation. IEEE TBME 69(9), 2916-2925 (2022). https://doi.org/10.1109/TBME.2022.3157468",
        "3. Berens P. CircStat: A MATLAB Toolbox for Circular Statistics. J Stat Softw 31(10) (2009). https://doi.org/10.18637/jss.v031.i10",
    ]
    (OUT / "pll_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
