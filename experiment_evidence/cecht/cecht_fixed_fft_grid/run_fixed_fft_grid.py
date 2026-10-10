"""P2b fixed-FFT-grid diagnostic for the causal cecHT window sweep.

P2 changed ``window_sec`` while the rule forced ``n_fft = N``, so window length
and the discretisation of the band-pass response moved together.  P2b breaks
that coupling by pinning ``n_fft`` and sweeping only the endpoint window:

* P2b-A   : ``window_sec in {0.75, 1.0, 1.25, 1.5, 2.0, 2.5}``, ``n_fft = 2000``
* P2b-A'  : same windows, ``n_fft = 4000`` (grid-density sensitivity check)
* anchor  : ``window_sec = 8.0``, ``n_fft = 2000`` = the canonical baseline

``n_fft > N`` is zero-padding.  It does **not** increase the true spectral
resolution of the windowed data; it is used only to sample the fixed
Butterworth response on a window-independent grid.  Every artifact records
``zero_padding_samples`` so the padding is explicit.

The estimator, filter, calibration, reference, ready/pseudo-event semantics,
targets, refractory and metrics are imported unchanged from the frozen
implementation; the parity-checked event helpers are reused from the P2 runner
so both stages share one definition.

References
----------
[R1] Osmers E, Kolossa D. *Optimal Calibration of the Endpoint-corrected
     Hilbert Transform*. arXiv:2601.13962v2 (2026).
     https://arxiv.org/abs/2601.13962
     Used for: endpoint/calibration semantics and the role of ``n_fft`` in
     sampling the band-pass response.
[R2] `eikeosmers/cecHT` pinned commit `e079c3115f202a4272cf64143bf33ddddc247778`
     (BSD-3-Clause), ``phase.py``.
     https://github.com/eikeosmers/cecHT/tree/e079c3115f202a4272cf64143bf33ddddc247778
     Used for: the vendored ECHT core and its ``_design_bandpass`` grid.
[R3] Ferster ML et al. *Benchmarking Real-Time Algorithms for In-Phase Auditory
     Stimulation ...* IEEE TBME 69(9), 2916-2925 (2022).
     https://doi.org/10.1109/TBME.2022.3157468
     Used for: development-only selection and hit/tail reporting structure.
[R4] Santostasi G et al. *Phase-locked loop for precisely timed acoustic
     stimulation during sleep*. J Neurosci Methods 259, 101-114 (2016).
     https://doi.org/10.1016/j.jneumeth.2015.11.007
     Used for: target-phase vs estimator-error vs timing-error separation.
[R5] Berens P. *CircStat: A MATLAB Toolbox for Circular Statistics*. J Stat
     Softw 31(10) (2009). https://doi.org/10.18637/jss.v031.i10
     Used for: circular bias / resultant length / circular SD.
[R6] SciPy ``signal`` API (butter, sosfiltfilt, hilbert, sosfreqz) and
     ``scipy.stats.spearmanr``. https://docs.scipy.org/doc/scipy/reference/signal.html
     Used for: reference band-pass, analytic signal, filter responses and the
     rank-consistency statistics.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
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
    wrap,
)
from causal_phase_tracking.causal_simulation_evaluation.run_evaluation import (  # noqa: E402
    ADAPTER_DELAY,
    build_manifest,
)


OUT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiment_evidence" / "cecht"
P2_RUNNER = EXPERIMENTS / "cecht_window_sweep" / "run_window_sweep.py"
P2_REPORT = EXPERIMENTS / "cecht_window_sweep" / "cecht_report.json"
# Override with CLAS_SLEEP_EDF_ROOT; set it to the local PhysioNet sleep-EDF root.
DATA_ROOT = Path(
    os.environ.get(
        "CLAS_SLEEP_EDF_ROOT",
        "physionet-sleep-data",
    )
)


def _load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


P2 = _load_module(P2_RUNNER, "cecht_runner")

FS = P2.FS
F0_HZ = P2.F0_HZ
REFRACTORY_S = P2.REFRACTORY_S
TARGET_GRID = P2.TARGET_GRID
N_BOOT = P2.N_BOOT
BOOTSTRAP_SEED = P2.BOOTSTRAP_SEED
BASELINE_WINDOW_S = P2.BASELINE_WINDOW_S

CANDIDATE_WINDOWS_S = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
N_FFT_GRID = [2000, 4000]
CANONICAL_MAE_RAD = 1.041248


def adapter_config() -> RationalRateAdapterConfig:
    """Frozen adapter parameters; only the version tag is P2b-specific."""

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
        config_version="p2b-fixed-grid-v1",
    )


def core_config(samples: int, n_fft: int) -> CecHTCoreConfig:
    """Frozen estimator config with an explicit, window-independent ``n_fft``."""

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
        n_fft=n_fft,
        fft_mode="exact",
        calibrate=True,
        f0=F0_HZ,
        parameter_version="p2b-fixed-grid-v1",
        config_version="p2b-fixed-grid-v1",
        max_block_samples=1024,
    )


def config_key(n_fft: int, window_s: float) -> str:
    return f"nfft{n_fft}_w{window_s:g}"


def check_grid() -> list[dict]:
    """Validate every (n_fft, window) pair and record its discretisation."""

    rows = []
    for n_fft in N_FFT_GRID:
        for window_s in CANDIDATE_WINDOWS_S + [BASELINE_WINDOW_S]:
            samples = P2.window_samples(window_s)
            if n_fft < samples:
                raise AssertionError(
                    f"n_fft {n_fft} < N {samples} would truncate the window"
                )
            core = CausalCecHTCore(core_config(samples, n_fft))
            if core.resolved_n_fft != n_fft:
                raise AssertionError(
                    f"resolved n_fft {core.resolved_n_fft} != requested {n_fft}"
                )
            if core.endpoint_kernel.size != samples:
                raise AssertionError("endpoint kernel length must equal N")
            if not np.isfinite(core.endpoint_kernel).all():
                raise AssertionError(f"non-finite kernel for {config_key(n_fft, window_s)}")
            freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / FS))
            in_band = freqs[(freqs >= 0.5) & (freqs < 1.25)]
            rows.append(
                {
                    "config": config_key(n_fft, window_s),
                    "n_fft": n_fft,
                    "window_s_nominal": window_s,
                    "window_samples": samples,
                    "window_s_realized": samples / FS,
                    "zero_padding_samples": n_fft - samples,
                    "fft_bin_spacing_hz": FS / n_fft,
                    "in_band_bin_count": int(in_band.size),
                    "cycles_at_f0": window_s * F0_HZ,
                    "role": (
                        "canonical_anchor"
                        if window_s == BASELINE_WINDOW_S and n_fft == 2000
                        else (
                            "anchor_sensitivity"
                            if window_s == BASELINE_WINDOW_S
                            else "candidate"
                        )
                    ),
                    "n_fft_rule": "n_fft = N" if n_fft == samples else "n_fft > N",
                }
            )
    return rows


def main() -> None:
    global OUT
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-nights",
        type=int,
        default=0,
        help="development-only smoke test: use the first N nights and write into ./smoke",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    grid = check_grid()
    delay_check = P2.check_adapter_delay()

    kernels: dict[str, np.ndarray] = {}
    for row in grid:
        kernels[row["config"]] = CausalCecHTCore(
            core_config(row["window_samples"], row["n_fft"])
        ).endpoint_kernel

    manifest, _eligibility = build_manifest()
    if args.smoke_nights > 0:
        manifest = manifest[: args.smoke_nights]
        OUT = OUT / "smoke"
        OUT.mkdir(parents=True, exist_ok=True)
    configs = [
        (row["n_fft"], row["window_s_nominal"], row["config"]) for row in grid
    ]
    accum: dict[str, dict] = {
        key: {
            "errors": [],
            "per_subject": {},
            "per_night": {},
            "ready": 0,
            "targets": {
                str(round(target, 4)): {"errors": [], "count": 0, "timing": []}
                for target in TARGET_GRID
            },
        }
        for _n_fft, _window, key in configs
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
                str(round(target, 4)): P2.vectorized_crossing_times(
                    ref_local, FS, target
                )
                for target in TARGET_GRID
            }

            for n_fft, window_s, key in configs:
                samples = P2.window_samples(window_s)
                endpoints = endpoint_sequence_complex(stream.values, kernels[key])
                phase_est = np.angle(endpoints)
                ready = stream.output_index >= (samples - 1)
                if not stream.valid[ready].all():
                    raise AssertionError(f"{key}: a ready endpoint is not adapter-valid")
                times_ready = stream.output_index[ready] / FS
                if not parity:
                    parity = P2.check_event_parity(phase_est[ready], times_ready, ref_local)
                error = wrap(phase_est[ready] - ref_local[ready])
                bucket = accum[key]
                bucket["errors"].append(error)
                bucket["ready"] += int(ready.sum())
                bucket["per_subject"].setdefault(subject, []).append(error)
                bucket["per_night"].setdefault(night, []).append(error)

                for target in TARGET_GRID:
                    target_key = str(round(target, 4))
                    event, crossing = P2.vectorized_events(
                        phase_est[ready],
                        times_ready,
                        fs=FS,
                        target_rad=target,
                        refractory_s=REFRACTORY_S,
                    )
                    entry = bucket["targets"][target_key]
                    if event.any():
                        entry["errors"].append(wrap(ref_local[ready][event] - target))
                        entry["count"] += int(event.sum())
                    timing = P2.median_timing_error(
                        crossing[event], reference_crossings[target_key]
                    )
                    if timing is not None:
                        entry["timing"].append(timing)
            seg_count += 1
        print(f"[p2b] finished {night} ({seg_count} segments)", flush=True)

    # ---- aggregation (same shape as P2 so the shared selectors apply) -----
    per_window: dict[float | str, dict] = {}
    subject_mae: dict[str, dict[str, float]] = {}
    metric_rows = []
    grid_by_key = {row["config"]: row for row in grid}
    anchor_key = config_key(2000, BASELINE_WINDOW_S)

    for n_fft, window_s, key in configs:
        bucket = accum[key]
        row = grid_by_key[key]
        errors = np.concatenate(bucket["errors"])
        estimator = P2.summarize(errors)
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
            summary = P2.event_summary(event_errors, entry["count"])
            summary.update(
                {
                    "config": key,
                    "n_fft": n_fft,
                    "window_s": window_s,
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
        event_parts = [
            part
            for target_key in bucket["targets"]
            for part in bucket["targets"][target_key]["errors"]
        ]
        pooled_events = P2.event_summary(
            np.concatenate(event_parts) if event_parts else np.empty(0),
            sum(row_["event_count"] for row_ in target_rows),
        )
        timing_values = [
            row_["median_absolute_timing_error_s"]
            for row_ in target_rows
            if row_["median_absolute_timing_error_s"] is not None
        ]
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
            "median_absolute_timing_error_s_default_target": default[
                "median_absolute_timing_error_s"
            ],
            "median_absolute_timing_error_s_across_targets": (
                float(np.median(timing_values)) if timing_values else None
            ),
        }
        metric_rows.append(
            {
                "config": key,
                "n_fft": n_fft,
                "window_s": window_s,
                "role": row["role"],
                **{name: row[name] for name in (
                    "window_samples",
                    "window_s_realized",
                    "zero_padding_samples",
                    "fft_bin_spacing_hz",
                    "in_band_bin_count",
                    "cycles_at_f0",
                )},
                **estimator,
                "subject_mean_mae_rad": per_window[key]["subject_level"][
                    "subject_mean_mae_rad"
                ],
                "event_default_target_count": default["event_count"],
                "event_default_target_hit_30": default["hit_30"],
                "event_default_target_hit_45": default["hit_45"],
                "median_absolute_timing_error_s": default[
                    "median_absolute_timing_error_s"
                ],
                "worst_target_hit_45": per_window[key]["worst_target_hit_45"],
            }
        )

    # canonical anchor must reproduce the frozen baseline exactly
    anchor = per_window[anchor_key]
    anchor_mae = anchor["estimator"]["mean_abs_wrapped_error_rad"]
    if args.smoke_nights == 0 and abs(anchor_mae - CANONICAL_MAE_RAD) > 5e-7:
        raise AssertionError(
            f"canonical anchor mismatch: {anchor_mae} vs {CANONICAL_MAE_RAD}"
        )

    fixed_keys = [config_key(2000, w) for w in CANDIDATE_WINDOWS_S]
    sensitivity_keys = [config_key(4000, w) for w in CANDIDATE_WINDOWS_S]
    metric_rows_by_key = {row["config"]: row for row in metric_rows}
    for key in fixed_keys + sensitivity_keys:
        values = subject_mae[key]
        reference = subject_mae[anchor_key]
        per_window[key]["subjects_improved_vs_8s"] = sum(
            1 for s, v in values.items() if v < reference.get(s, math.inf)
        )
        per_window[key]["subjects_worsened_vs_8s"] = sum(
            1 for s, v in values.items() if v > reference.get(s, math.inf)
        )
        per_window[key]["delta_vs_canonical"] = {
            "delta_mae_rad": per_window[key]["estimator"]["mean_abs_wrapped_error_rad"]
            - anchor_mae,
            "delta_subject_mean_mae_rad": per_window[key]["subject_level"][
                "subject_mean_mae_rad"
            ]
            - anchor["subject_level"]["subject_mean_mae_rad"],
            "delta_q95_rad": per_window[key]["estimator"]["q95_abs_wrapped_error_rad"]
            - anchor["estimator"]["q95_abs_wrapped_error_rad"],
            "delta_resultant_length": per_window[key]["estimator"]["resultant_length"]
            - anchor["estimator"]["resultant_length"],
            "delta_circular_sd_rad": per_window[key]["estimator"]["circular_sd_rad"]
            - anchor["estimator"]["circular_sd_rad"],
            "delta_hit_45": per_window[key]["events_default_target"]["hit_45"]
            - anchor["events_default_target"]["hit_45"],
            "delta_catastrophic_rate": per_window[key]["estimator"][
                "catastrophic_rate_gt_135deg"
            ]
            - anchor["estimator"]["catastrophic_rate_gt_135deg"],
            "subjects_improved": per_window[key]["subjects_improved_vs_8s"],
            "subjects_worsened": per_window[key]["subjects_worsened_vs_8s"],
        }
        metric_rows_by_key[key].update(per_window[key]["delta_vs_canonical"])

    comparison = compare_rankings(
        per_window, fixed_keys, sensitivity_keys, subject_mae, anchor_key
    )
    fixed_grid = [
        {
            "window_s_nominal": window,
            "window_samples": P2.window_samples(window),
            "cycles_at_f0": window * F0_HZ,
        }
        for window in CANDIDATE_WINDOWS_S
    ]
    fixed_ranking = P2.build_ranking(
        {w: per_window[config_key(2000, w)] for w in CANDIDATE_WINDOWS_S},
        fixed_grid,
    )
    sensitivity_ranking = P2.build_ranking(
        {w: per_window[config_key(4000, w)] for w in CANDIDATE_WINDOWS_S},
        fixed_grid,
    )

    report = {
        "experiment": "cecht_fixed_fft_grid",
        "purpose": (
            "Decouple window length from the FFT-grid discretisation of the "
            "band-pass response by fixing n_fft."
        ),
        "zero_padding_statement": (
            "zero-padding does not increase true spectral resolution and is "
            "used only to isolate FFT-grid discretization from temporal-window "
            "length"
        ),
        "canonical_baseline": {
            "window_s": BASELINE_WINDOW_S,
            "f0_hz": F0_HZ,
            "mean_abs_wrapped_error_rad": CANONICAL_MAE_RAD,
            "anchor_reproduced_rad": anchor_mae,
            "source": "P1 / P3 / P2 reproduction; P2b anchor re-checked in-harness",
        },
        "frozen": {
            "filter": "butterworth order 2, 0.5-1.25 Hz",
            "calibration": True,
            "reference": "non-causal resample_poly(5,2) + zero-phase butter 0.5-1.25 Hz + Hilbert",
            "reference_alignment": "ref_phase[seg_start_250 - ADAPTER_DELAY + output_index]",
            "ready_semantics": "output_index >= N - 1",
            "targets_rad": TARGET_GRID,
            "refractory_s": REFRACTORY_S,
            "bootstrap": {"n_boot": N_BOOT, "seed": BOOTSTRAP_SEED},
            "development_nights": len(manifest),
            "segments_processed": seg_count,
            "candidate_windows_s": CANDIDATE_WINDOWS_S,
            "n_fft_grid": N_FFT_GRID,
        },
        "self_checks": {
            "grid": grid,
            "adapter_delay": delay_check,
            "event_parity": parity,
            "canonical_anchor_mae_rad": anchor_mae,
            "p2_runner_sha256": hashlib.sha256(P2_RUNNER.read_bytes()).hexdigest(),
            "shared_event_definition": str(P2_RUNNER),
        },
        "windows": {key: value for key, value in per_window.items()},
        "ranking_fixed_grid_nfft2000": fixed_ranking,
        "ranking_sensitivity_nfft4000": sensitivity_ranking,
        "comparison": comparison,
        "isolation": {
            "main_code_modified": "NO",
            "production_config_modified": "NO",
            "untouched_test_accessed": "NO",
            "verdict": "DIAGNOSTIC",
        },
        "wall_clock_seconds": time.perf_counter() - started,
    }
    (OUT / "cecht_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_metrics_csv(metric_rows)
    write_subject_csv(subject_mae, grid)
    write_target_csv(per_window, grid)
    write_night_csv(per_window, grid)
    write_comparison_csv(comparison)
    make_figures(per_window, comparison)
    write_summary(report)
    print(json.dumps({
        "comparison": comparison,
        "fixed_ranking": fixed_ranking["mean_rank"],
    }, indent=2, ensure_ascii=False))


def _rank(values: list[float], lower_is_better: bool) -> list[int]:
    order = sorted(
        range(len(values)), key=lambda i: values[i], reverse=not lower_is_better
    )
    ranks = [0] * len(values)
    for position, index in enumerate(order):
        ranks[index] = position + 1
    return ranks


def _spearman(first: list[float], second: list[float]) -> float:
    from scipy.stats import spearmanr

    rho, _p = spearmanr(np.asarray(first, float), np.asarray(second, float))
    return float(rho)


def compare_rankings(
    per_window: dict,
    fixed_keys: list[str],
    sensitivity_keys: list[str],
    subject_mae: dict,
    anchor_key: str,
) -> dict:
    """Rank-window comparison between P2, P2b-A (2000) and P2b-A' (4000)."""

    def mae(key: str, level: str) -> float:
        if level == "pooled":
            return per_window[key]["estimator"]["mean_abs_wrapped_error_rad"]
        return per_window[key]["subject_level"]["subject_mean_mae_rad"]

    windows = CANDIDATE_WINDOWS_S
    fixed_subj = [mae(k, "subject") for k in fixed_keys]
    sens_subj = [mae(k, "subject") for k in sensitivity_keys]

    p2_shared_windows = [1.0, 1.25, 1.5, 2.0, 2.5]
    p2_values: list[float] = []
    p2_available = P2_REPORT.exists()
    if p2_available:
        p2_report = json.loads(P2_REPORT.read_text(encoding="utf-8"))
        p2_values = [
            p2_report["windows"][str(window)]["subject_level"]["subject_mean_mae_rad"]
            for window in p2_shared_windows
        ]
    fixed_shared = [
        per_window[config_key(2000, window)]["subject_level"]["subject_mean_mae_rad"]
        for window in p2_shared_windows
    ]
    sensitivity_shared = [
        per_window[config_key(4000, window)]["subject_level"]["subject_mean_mae_rad"]
        for window in p2_shared_windows
    ]

    anchor = per_window[anchor_key]
    canonical_subject_mae = anchor["subject_level"]["subject_mean_mae_rad"]
    best_index = int(np.argmin(fixed_subj))
    best_window = windows[best_index]
    best_key = fixed_keys[best_index]
    boundary_limited = bool(best_window == min(windows))
    improved = per_window[best_key]["subjects_improved_vs_8s"]
    worsened = per_window[best_key]["subjects_worsened_vs_8s"]

    retained = {}
    if p2_available:
        for window, p2_value in zip(p2_shared_windows, p2_values):
            p2_gain = canonical_subject_mae - p2_value
            fixed_gain = (
                canonical_subject_mae
                - per_window[config_key(2000, window)]["subject_level"][
                    "subject_mean_mae_rad"
                ]
            )
            retained[str(window)] = {
                "p2_gain_rad": p2_gain,
                "fixed_grid_gain_rad": fixed_gain,
                "fraction_of_p2_gain_retained": (
                    fixed_gain / p2_gain if abs(p2_gain) > 1e-12 else None
                ),
            }

    supported = bool(
        per_window[best_key]["subject_level"]["subject_mean_mae_rad"]
        < canonical_subject_mae
        and improved >= 10
    )
    ranking_stable = bool(
        _spearman(fixed_subj, sens_subj) >= 0.7
    )
    consistent_with_p2 = bool(
        p2_available and _spearman(p2_values, fixed_shared) >= 0.7
    )
    selection_valid = bool(supported and ranking_stable)

    answers = {
        "q1_short_window_still_beats_8s_under_fixed_grid": {
            "best_window_s": best_window,
            "best_n_fft_2000_subject_mae": fixed_subj[best_index],
            "canonical_subject_mae": canonical_subject_mae,
            "delta_subject_mae": fixed_subj[best_index] - canonical_subject_mae,
            "subjects_improved": improved,
            "subjects_worsened": worsened,
            "supported": supported,
        },
        "q2_optimum_still_near_1_to_2_cycles": {
            "best_window_s": best_window,
            "best_cycles": best_window * F0_HZ,
            "boundary_limited": boundary_limited,
            "window_series_subject_mae_nfft2000": {
                str(window): value for window, value in zip(windows, fixed_subj)
            },
            "window_series_subject_mae_nfft4000": {
                str(window): value for window, value in zip(windows, sens_subj)
            },
        },
        "q3_relative_ordering_of_short_windows": {
            "nfft2000_rank_by_subject_mae_best_first": {
                str(window): rank
                for window, rank in zip(
                    windows, _rank(fixed_subj, lower_is_better=True)
                )
            },
            "nfft4000_rank_by_subject_mae_best_first": {
                str(window): rank
                for window, rank in zip(
                    windows, _rank(sens_subj, lower_is_better=True)
                )
            },
            "spearman_nfft2000_vs_nfft4000": _spearman(fixed_subj, sens_subj),
        },
        "q4_how_much_of_p2_gain_was_grid_discretisation": {
            "per_window": retained,
            "spearman_p2_vs_fixed_grid_shared_windows": (
                _spearman(p2_values, fixed_shared) if p2_available else None
            ),
            "spearman_p2_vs_nfft4000_shared_windows": (
                _spearman(p2_values, sensitivity_shared) if p2_available else None
            ),
            "p2_report_used": str(P2_REPORT) if p2_available else None,
        },
        "q5_can_window_be_selected_now": {
            "supported": supported,
            "ranking_stable_across_grids": ranking_stable,
            "ranking_consistent_with_p2": consistent_with_p2,
            "boundary_limited": boundary_limited,
            "selection_valid": selection_valid,
        },
    }
    return {
        "windows": windows,
        "subject_mae_nfft2000": fixed_subj,
        "subject_mae_nfft4000": sens_subj,
        "pooled_mae_nfft2000": [mae(k, "pooled") for k in fixed_keys],
        "pooled_mae_nfft4000": [mae(k, "pooled") for k in sensitivity_keys],
        "p2_shared_windows": p2_shared_windows,
        "p2_subject_mae": p2_values if p2_available else None,
        "p2_available": p2_available,
        "canonical_subject_mae": canonical_subject_mae,
        "spearman_nfft2000_vs_nfft4000": _spearman(fixed_subj, sens_subj),
        "spearman_p2_vs_nfft2000": (
            _spearman(p2_values, fixed_shared) if p2_available else None
        ),
        "spearman_p2_vs_nfft4000": (
            _spearman(p2_values, sensitivity_shared) if p2_available else None
        ),
        "best_window_s": best_window,
        "best_config": best_key,
        "boundary_limited": boundary_limited,
        "tested_windows": windows,
        "temporal_short_window_effect_supported": supported,
        "ranking_stable_across_grids": ranking_stable,
        "ranking_consistent_with_p2": consistent_with_p2,
        "p2_window_selection_valid": selection_valid,
        "further_refinement_required": boundary_limited,
        "decision_rule": (
            "supported = best fixed-grid subject MAE beats the canonical 8 s "
            "baseline with >= 10 of 13 subjects improved; selection valid = "
            "supported AND Spearman(n_fft=2000, n_fft=4000) >= 0.7; the optimum "
            "is not declared final if it sits on the 0.75 s search boundary"
        ),
        "answers": answers,
    }


def write_metrics_csv(rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (OUT / "cecht_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_subject_csv(subject_mae: dict, grid: list[dict]) -> None:
    subjects = sorted({s for values in subject_mae.values() for s in values})
    rows = []
    for row in grid:
        key = row["config"]
        for subject in subjects:
            rows.append(
                {
                    "config": key,
                    "n_fft": row["n_fft"],
                    "window_s": row["window_s_nominal"],
                    "window_samples": row["window_samples"],
                    "zero_padding_samples": row["zero_padding_samples"],
                    "cycles_at_f0": row["cycles_at_f0"],
                    "role": row["role"],
                    "subject": subject,
                    "mean_abs_wrapped_error_rad": subject_mae[key].get(subject),
                }
            )
    with (OUT / "cecht_subject_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_target_csv(per_window: dict, grid: list[dict]) -> None:
    rows = []
    for row in grid:
        rows.extend(per_window[row["config"]]["targets"])
    with (OUT / "cecht_target_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_night_csv(per_window: dict, grid: list[dict]) -> None:
    nights = sorted(
        {
            night
            for row in grid
            for night in per_window[row["config"]]["subject_level"]["per_night_mae_rad"]
        }
    )
    rows = []
    for row in grid:
        key = row["config"]
        values = per_window[key]["subject_level"]["per_night_mae_rad"]
        for night in nights:
            rows.append(
                {
                    "config": key,
                    "n_fft": row["n_fft"],
                    "window_s": row["window_s_nominal"],
                    "night": night,
                    "mean_abs_wrapped_error_rad": values.get(night),
                }
            )
    with (OUT / "cecht_night_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_csv(comparison: dict) -> None:
    p2_by_window = {}
    if comparison["p2_subject_mae"] is not None:
        p2_by_window = {
            window: value
            for window, value in zip(
                comparison["p2_shared_windows"], comparison["p2_subject_mae"]
            )
        }
    rows = []
    for index, window in enumerate(comparison["windows"]):
        rows.append(
            {
                "window_s": window,
                "cycles_at_f0": window * F0_HZ,
                "subject_mae_nfft2000": comparison["subject_mae_nfft2000"][index],
                "subject_mae_nfft4000": comparison["subject_mae_nfft4000"][index],
                "pooled_mae_nfft2000": comparison["pooled_mae_nfft2000"][index],
                "pooled_mae_nfft4000": comparison["pooled_mae_nfft4000"][index],
                "p2_nfft_eq_N_subject_mae": p2_by_window.get(window),
                "canonical_subject_mae": comparison["canonical_subject_mae"],
            }
        )
    with (OUT / "cecht_ranking_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(per_window: dict, comparison: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = comparison["windows"]
    fixed_values = {
        "mae": comparison["subject_mae_nfft2000"],
        "q95": [
            per_window[config_key(2000, w)]["estimator"]["q95_abs_wrapped_error_rad"]
            for w in windows
        ],
        "R": [
            per_window[config_key(2000, w)]["estimator"]["resultant_length"]
            for w in windows
        ],
        "hit45": [
            per_window[config_key(2000, w)]["events_default_target"]["hit_45"]
            for w in windows
        ],
        "cat": [
            per_window[config_key(2000, w)]["estimator"][
                "catastrophic_rate_gt_135deg"
            ]
            for w in windows
        ],
    }
    sensitivity_values = {
        "mae": comparison["subject_mae_nfft4000"],
        "q95": [
            per_window[config_key(4000, w)]["estimator"]["q95_abs_wrapped_error_rad"]
            for w in windows
        ],
        "R": [
            per_window[config_key(4000, w)]["estimator"]["resultant_length"]
            for w in windows
        ],
        "hit45": [
            per_window[config_key(4000, w)]["events_default_target"]["hit_45"]
            for w in windows
        ],
        "cat": [
            per_window[config_key(4000, w)]["estimator"][
                "catastrophic_rate_gt_135deg"
            ]
            for w in windows
        ],
    }
    labels = {
        "mae": "subject-level MAE [rad]",
        "q95": "q95 absolute wrapped error [rad]",
        "R": "resultant length R",
        "hit45": "default-target hit rate within 45 deg",
        "cat": "catastrophic rate (|e| > 135 deg)",
    }
    anchor_block = per_window[config_key(2000, BASELINE_WINDOW_S)]
    canonical_levels = {
        "mae": comparison["canonical_subject_mae"],
        "q95": anchor_block["estimator"]["q95_abs_wrapped_error_rad"],
        "R": anchor_block["estimator"]["resultant_length"],
        "hit45": anchor_block["events_default_target"]["hit_45"],
        "cat": anchor_block["estimator"]["catastrophic_rate_gt_135deg"],
    }
    for name, label in labels.items():
        figure, axis = plt.subplots(figsize=(6.6, 4.2), dpi=150)
        axis.plot(windows, fixed_values[name], marker="o", label="P2b n_fft = 2000")
        axis.plot(
            windows,
            sensitivity_values[name],
            marker="s",
            linestyle="--",
            label="P2b n_fft = 4000",
        )
        axis.axhline(
            canonical_levels[name],
            color="0.55",
            linestyle=":",
            linewidth=1,
            label="canonical 8 s",
        )
        axis.set_xlabel("window length [s] (n_fft fixed)")
        axis.set_ylabel(label)
        axis.set_title(f"{label} vs window length, fixed FFT grid")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(OUT / f"fig_window_vs_{name}.png")
        plt.close(figure)

    # P2 vs P2b ranking comparison (shared windows)
    shared = comparison["p2_shared_windows"]
    if comparison["p2_subject_mae"] is not None:
        figure, axis = plt.subplots(figsize=(6.8, 4.4), dpi=150)
        axis.plot(
            shared,
            comparison["p2_subject_mae"],
            marker="^",
            label="P2 (n_fft = N)",
        )
        axis.plot(
            shared,
            [
                per_window[config_key(2000, w)]["subject_level"]["subject_mean_mae_rad"]
                for w in shared
            ],
            marker="o",
            label="P2b (n_fft = 2000)",
        )
        axis.plot(
            shared,
            [
                per_window[config_key(4000, w)]["subject_level"]["subject_mean_mae_rad"]
                for w in shared
            ],
            marker="s",
            linestyle="--",
            label="P2b (n_fft = 4000)",
        )
        axis.axhline(
            comparison["canonical_subject_mae"],
            color="0.55",
            linestyle=":",
            label="canonical 8 s",
        )
        axis.set_xlabel("window length [s]")
        axis.set_ylabel("subject-level MAE [rad]")
        axis.set_title("P2 vs P2b ranking comparison")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(OUT / "fig_ranking_comparison.png")
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    axis.plot(
        windows, comparison["subject_mae_nfft2000"], marker="o", label="n_fft = 2000"
    )
    axis.plot(
        windows,
        comparison["subject_mae_nfft4000"],
        marker="s",
        linestyle="--",
        label="n_fft = 4000",
    )
    axis.set_xlabel("window length [s]")
    axis.set_ylabel("subject-level MAE [rad]")
    axis.set_title(
        f"FFT-grid density check (Spearman rho = "
        f"{comparison['spearman_nfft2000_vs_nfft4000']:+.2f})"
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "fig_nfft2000_vs_4000.png")
    plt.close(figure)


def write_summary(report: dict) -> None:
    comparison = report["comparison"]
    answers = comparison["answers"]
    anchor_mae = report["self_checks"]["canonical_anchor_mae_rad"]

    def fmt(value):
        return "n/a" if value is None else f"{value:.6f}"

    lines = [
        "# cecHT P2b fixed-FFT-grid diagnostic",
        "",
        f"- development nights: {report['frozen']['development_nights']}",
        f"- segments processed: {report['frozen']['segments_processed']}",
        f"- candidate windows [s]: {report['frozen']['candidate_windows_s']}",
        f"- n_fft grid: {report['frozen']['n_fft_grid']} (fixed; n_fft > N is zero padding)",
        f"- zero padding: {report['zero_padding_statement']}",
        f"- canonical anchor (8 s, n_fft = 2000) reproduced MAE = {anchor_mae:.6f} rad",
        f"- self-checks: event parity {report['self_checks']['event_parity']['status']} "
        f"(max |dt| = {report['self_checks']['event_parity']['max_abs_crossing_time_delta_s']:.2e} s), "
        f"adapter delay = {report['self_checks']['adapter_delay']['adapter_group_delay_samples']} samples",
        "",
        "## Candidates",
        "",
        "| n_fft | window [s] | N | zero-pad | cycles | ready | pooled MAE | subj MAE | q95 | R | circ SD | hit30 | hit45 | catastr. | events | timing [s] | dMAE(pooled) | dMAE(subj) | imp/wors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["self_checks"]["grid"]:
        key = row["config"]
        block = report["windows"][key]
        estimator = block["estimator"]
        default = block["events_default_target"]
        delta = block.get("delta_vs_canonical", {})
        lines.append(
            "| {nfft} | {win:g} | {n} | {pad} | {cyc:.2f} | {ready} | {mae:.4f} | {smae:.4f} | "
            "{q95:.4f} | {r:.4f} | {sd:.4f} | {h30:.4f} | {h45:.4f} | {cat:.4f} | {ev} | {tim} | {dmae} | {dsubj} | {imp}/{wors} |".format(
                nfft=row["n_fft"],
                win=row["window_s_nominal"],
                n=row["window_samples"],
                pad=row["zero_padding_samples"],
                cyc=row["cycles_at_f0"],
                ready=estimator["ready_sample_count"],
                mae=estimator["mean_abs_wrapped_error_rad"],
                smae=block["subject_level"]["subject_mean_mae_rad"],
                q95=estimator["q95_abs_wrapped_error_rad"],
                r=estimator["resultant_length"],
                sd=estimator["circular_sd_rad"],
                h30=default["hit_30"],
                h45=default["hit_45"],
                cat=estimator["catastrophic_rate_gt_135deg"],
                ev=default["event_count"],
                tim=fmt(default["median_absolute_timing_error_s"]),
                dmae=(
                    f"{delta['delta_mae_rad']:+.4f}" if delta.get("delta_mae_rad") is not None else "n/a"
                ),
                dsubj=(
                    f"{delta['delta_subject_mean_mae_rad']:+.4f}"
                    if delta.get("delta_subject_mean_mae_rad") is not None
                    else "n/a"
                ),
                imp=delta.get("subjects_improved", "n/a"),
                wors=delta.get("subjects_worsened", "n/a"),
            )
        )

    lines += [
        "",
        "## Ranking comparison",
        "",
        f"- P2 (n_fft = N) shared-window subject MAE: "
        f"{comparison['p2_subject_mae']}",
        f"- P2b n_fft = 2000 subject MAE: {comparison['subject_mae_nfft2000']}",
        f"- P2b n_fft = 4000 subject MAE: {comparison['subject_mae_nfft4000']}",
        f"- Spearman P2 vs P2b(2000) on shared windows: "
        f"{comparison['spearman_p2_vs_nfft2000']}",
        f"- Spearman P2 vs P2b(4000) on shared windows: "
        f"{comparison['spearman_p2_vs_nfft4000']}",
        f"- Spearman P2b(2000) vs P2b(4000): "
        f"{comparison['spearman_nfft2000_vs_nfft4000']}",
        f"- best fixed-grid candidate: {comparison['best_window_s']} s "
        f"({comparison['best_config']}), boundary limited: {comparison['boundary_limited']}",
        "",
        "## Answers",
        "",
        f"1. short window still beats 8 s under a fixed grid? "
        f"{answers['q1_short_window_still_beats_8s_under_fixed_grid']}",
        f"2. optimum still near 1-2 cycles? "
        f"{answers['q2_optimum_still_near_1_to_2_cycles']}",
        f"3. relative ordering of short windows? "
        f"{answers['q3_relative_ordering_of_short_windows']}",
        f"4. how much of the P2 gain was grid discretisation? "
        f"{answers['q4_how_much_of_p2_gain_was_grid_discretisation']}",
        f"5. can a window be selected now? "
        f"{answers['q5_can_window_be_selected_now']}",
        "",
        "## Decision",
        "",
        f"- decision rule: {comparison['decision_rule']}",
        f"- `Temporal short-window effect supported`: "
        f"{'YES' if comparison['temporal_short_window_effect_supported'] else 'NO'}",
        f"- `P2 window selection valid`: "
        f"{'YES' if comparison['p2_window_selection_valid'] else 'NO'}",
        f"- ranking stable across FFT grids: {comparison['ranking_stable_across_grids']}",
        f"- ranking consistent with P2: {comparison['ranking_consistent_with_p2']}",
        f"- `Further refinement required`: "
        f"{'YES' if comparison['further_refinement_required'] else 'NO'}",
        "",
        "## Isolation status",
        "- Main code modified: NO",
        "- Production config modified: NO",
        "- Untouched test accessed: NO",
        "- Commit performed: NO",
        f"- Experiment directory: {OUT}",
        "",
        "## References",
        "1. Osmers E, Kolossa D. Optimal Calibration of the Endpoint-corrected Hilbert Transform. arXiv:2601.13962v2 (2026). https://arxiv.org/abs/2601.13962",
        "2. eikeosmers/cecHT pinned commit e079c3115f202a4272cf64143bf33ddddc247778 (BSD-3-Clause), phase.py. https://github.com/eikeosmers/cecHT/tree/e079c3115f202a4272cf64143bf33ddddc247778",
        "3. Ferster ML et al. Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During Sleep. IEEE TBME 69(9), 2916-2925 (2022). https://doi.org/10.1109/TBME.2022.3157468",
        "4. Santostasi G et al. Phase-locked loop for precisely timed acoustic stimulation during sleep. J Neurosci Methods 259, 101-114 (2016). https://doi.org/10.1016/j.jneumeth.2015.11.007",
        "5. Berens P. CircStat: A MATLAB Toolbox for Circular Statistics. Journal of Statistical Software 31(10) (2009). https://doi.org/10.18637/jss.v031.i10",
        "6. SciPy signal API and scipy.stats.spearmanr. https://docs.scipy.org/doc/scipy/reference/signal.html ; https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html",
    ]
    (OUT / "cecht_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
