"""P1 targeted error stratification diagnostic.

Offline-only diagnostic of the frozen P0 baseline. It does not modify main
code, production config, or frozen test manifests, and never reads untouched
test subjects.
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
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_cecHT.streaming_core import CecHTCoreConfig, CausalCecHTCore
from causal_phase_tracking.causal_simulation_evaluation.edf_replay import (
    load_fpz_cz_100hz,
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
F0 = 0.8
WINDOW_S = 8.0
ADAPTER_DELAY = 8
PHASE_BINS = 12


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
        config_version="p1-stratification-v1",
    )


def core_config() -> CecHTCoreConfig:
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
        calibrate=True,
        f0=F0,
        parameter_version="p1-stratification-v1",
        config_version="p1-stratification-v1",
        max_block_samples=1024,
    )


def _instantaneous_frequency(phase: np.ndarray, fs: float) -> np.ndarray:
    unwrapped = np.unwrap(phase)
    diff = np.diff(unwrapped, prepend=unwrapped[0])
    instantaneous = diff * fs / (2.0 * math.pi)
    return uniform_filter1d(instantaneous, size=int(round(fs)), mode="nearest")


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    mean = uniform_filter1d(values, size=window, mode="nearest")
    mean_square = uniform_filter1d(values * values, size=window, mode="nearest")
    return np.sqrt(np.maximum(mean_square - mean * mean, 0.0))


def collect_features() -> dict:
    manifest, _ = build_manifest()
    core = CausalCecHTCore(core_config())
    nights: dict[str, dict[str, np.ndarray]] = {}
    for record in manifest:
        night = record["night"]
        psg = DATA_ROOT / f"{night}-PSG.edf"
        signal = load_fpz_cz_100hz(psg) * 1e6
        ref_250 = noncausal_reference_resample(signal)
        ref_band = zero_phase_so_bandpass(ref_250, 250.0, l_freq=0.5, h_freq=1.25)
        ref_complex = hilbert(ref_band)
        ref_phase_full = np.angle(ref_complex)
        ref_amp_full = np.abs(ref_complex)
        f_inst_full = _instantaneous_frequency(ref_phase_full, 250.0)
        sigma_f_full = _rolling_std(
            f_inst_full, int(round(WINDOW_S * 250.0))
        )

        error_parts = []
        amplitude_parts = []
        delta_f_parts = []
        sigma_f_parts = []
        ref_phase_parts = []
        for seg in record["segments_s"]:
            start_idx = int(round(seg["start_s"] * 100.0))
            end_idx = int(round(seg["end_s"] * 100.0))
            raw = signal[start_idx:end_idx]
            stream = run_adapted_stream(raw, adapter_config(), [10, 40, 100, 17, 250, 3])
            endpoints = endpoint_sequence_complex(stream.values, core.endpoint_kernel)
            ready = stream.output_index >= (core.config.window_samples - 1)
            base = int(round(seg["start_s"] * 250.0))
            ref_idx = np.clip(
                base + stream.output_index - ADAPTER_DELAY,
                0,
                ref_phase_full.size - 1,
            )
            idx = ref_idx[ready]
            error_parts.append(
                wrap(np.angle(endpoints)[ready] - ref_phase_full[idx])
            )
            amplitude_parts.append(ref_amp_full[idx])
            delta_f_parts.append(np.abs(f_inst_full[idx] - F0))
            sigma_f_parts.append(sigma_f_full[idx])
            ref_phase_parts.append(ref_phase_full[idx])

        nights[night] = {
            "subject": np.asarray([record["subject"]] * sum(part.size for part in error_parts)),
            "error": np.concatenate(error_parts),
            "amplitude": np.concatenate(amplitude_parts),
            "abs_delta_f": np.concatenate(delta_f_parts),
            "sigma_f": np.concatenate(sigma_f_parts),
            "ref_phase": np.concatenate(ref_phase_parts),
        }
    return nights


def _summary(error: np.ndarray) -> dict:
    summary = circular_abs_error_summary(error)
    absolute = np.abs(wrap(error))
    summary["mean_abs_wrapped_error_rad"] = float(np.mean(absolute))
    summary["median_abs_wrapped_error_rad"] = float(np.median(absolute))
    summary["q95_abs_wrapped_error_rad"] = float(np.quantile(absolute, 0.95))
    summary["sample_count"] = int(error.size)
    summary["hit_30"] = float(np.mean(absolute <= math.pi / 6))
    summary["hit_45"] = float(np.mean(absolute <= math.pi / 4))
    summary["catastrophic_rate_gt_135deg"] = float(
        np.mean(absolute > 3.0 * math.pi / 4)
    )
    return summary


def stratified_metrics(
    nights: dict[str, dict[str, np.ndarray]],
    feature: str,
    *,
    quantiles: int = 5,
) -> list[dict]:
    rows = []
    layer_values: dict[str, list[np.ndarray]] = {f"Q{i+1}": [] for i in range(quantiles)}
    for night, data in nights.items():
        values = data[feature]
        edges = np.quantile(values, np.linspace(0.0, 1.0, quantiles + 1))
        labels = np.clip(np.digitize(values, edges[1:-1]), 0, quantiles - 1)
        for layer in range(quantiles):
            layer_values[f"Q{layer+1}"].append(data["error"][labels == layer])
    for layer, parts in layer_values.items():
        errors = np.concatenate([p for p in parts if p.size]) if any(p.size for p in parts) else np.empty(0)
        row = {"layer": layer, "feature": feature}
        row.update(_summary(errors) if errors.size else {})
        rows.append(row)
    return rows


def phase_binned_metrics(nights: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    edges = np.linspace(-math.pi, math.pi, PHASE_BINS + 1)
    rows = []
    for bin_index in range(PHASE_BINS):
        errors = []
        for data in nights.values():
            mask = (data["ref_phase"] >= edges[bin_index]) & (
                data["ref_phase"] < edges[bin_index + 1]
            )
            errors.append(data["error"][mask])
        joined = np.concatenate(errors) if any(part.size for part in errors) else np.empty(0)
        row = {
            "phase_bin_rad": float((edges[bin_index] + edges[bin_index + 1]) / 2.0),
            "phase_bin_deg": float(math.degrees((edges[bin_index] + edges[bin_index + 1]) / 2.0)),
        }
        if joined.size:
            summary = circular_abs_error_summary(joined)
            row.update(
                {
                    "mean_signed_error_rad": float(np.mean(wrap(joined))),
                    "mean_abs_error_rad": float(np.mean(np.abs(wrap(joined)))),
                    "resultant_length": summary["resultant_length"],
                    "sample_count": int(joined.size),
                }
            )
        rows.append(row)
    return rows


def interaction_matrix(
    nights: dict[str, dict[str, np.ndarray]],
    amplitude_quantiles: int = 5,
    delta_quantiles: int = 5,
) -> dict:
    mae = np.full((amplitude_quantiles, delta_quantiles), np.nan)
    catastrophic = np.full((amplitude_quantiles, delta_quantiles), np.nan)
    counts = np.zeros((amplitude_quantiles, delta_quantiles), dtype=int)
    for data in nights.values():
        a_edges = np.quantile(data["amplitude"], np.linspace(0, 1, amplitude_quantiles + 1))
        f_edges = np.quantile(data["abs_delta_f"], np.linspace(0, 1, delta_quantiles + 1))
        a_labels = np.clip(np.digitize(data["amplitude"], a_edges[1:-1]), 0, amplitude_quantiles - 1)
        f_labels = np.clip(np.digitize(data["abs_delta_f"], f_edges[1:-1]), 0, delta_quantiles - 1)
        absolute = np.abs(wrap(data["error"]))
        for a in range(amplitude_quantiles):
            for f in range(delta_quantiles):
                mask = (a_labels == a) & (f_labels == f)
                if not mask.any():
                    continue
                counts[a, f] += int(mask.sum())
                chunk = np.abs(wrap(data["error"][mask]))
                mae[a, f] = np.nanmean(
                    [mae[a, f], np.mean(chunk)] if not np.isnan(mae[a, f]) else [np.mean(chunk)]
                )
                catastrophic[a, f] = np.nanmean(
                    [catastrophic[a, f], np.mean(chunk > 3 * math.pi / 4)]
                    if not np.isnan(catastrophic[a, f])
                    else [np.mean(chunk > 3 * math.pi / 4)]
                )
    return {
        "mae_rad": mae.tolist(),
        "catastrophic_rate": catastrophic.tolist(),
        "counts": counts.tolist(),
    }


def catastrophic_comparison(nights: dict[str, dict[str, np.ndarray]]) -> dict:
    errors = np.concatenate([data["error"] for data in nights.values()])
    amplitude = np.concatenate([data["amplitude"] for data in nights.values()])
    delta_f = np.concatenate([data["abs_delta_f"] for data in nights.values()])
    sigma_f = np.concatenate([data["sigma_f"] for data in nights.values()])
    absolute = np.abs(wrap(errors))
    groups = {
        "moderate_lt_45deg": absolute < math.pi / 4,
        "large_gt_90deg": absolute > math.pi / 2,
        "catastrophic_gt_135deg": absolute > 3 * math.pi / 4,
    }
    result = {}
    for name, mask in groups.items():
        if not mask.any():
            result[name] = None
            continue
        result[name] = {
            "count": int(mask.sum()),
            "mean_amplitude": float(np.mean(amplitude[mask])),
            "mean_abs_delta_f": float(np.mean(delta_f[mask])),
            "mean_sigma_f": float(np.mean(sigma_f[mask])),
        }
    return result


def figures(nights: dict[str, dict[str, np.ndarray]], amplitude_rows, delta_rows, sigma_rows, phase_rows, interaction) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=180)
        plt.close(fig)

    amps = [row["mean_abs_wrapped_error_rad"] for row in amplitude_rows]
    amp_q95 = [row["q95_abs_wrapped_error_rad"] for row in amplitude_rows]
    amp_r = [row["resultant_length"] for row in amplitude_rows]
    fig, ax = plt.subplots()
    ax.plot(amps, label="MAE")
    ax.plot(amp_q95, label="q95")
    ax2 = ax.twinx()
    ax2.plot(amp_r, color="tab:green", label="R")
    ax.set_xlabel("amplitude quintile")
    ax.set_ylabel("rad")
    ax2.set_ylabel("R")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    save(fig, "fig1_amplitude_vs_error.png")

    fig, ax = plt.subplots()
    ax.plot([row["mean_abs_wrapped_error_rad"] for row in delta_rows], label="MAE")
    ax.plot([row["q95_abs_wrapped_error_rad"] for row in delta_rows], label="q95")
    ax.set_xlabel("|f_inst - f0| quantile")
    ax.set_ylabel("rad")
    ax.legend()
    save(fig, "fig2_frequency_mismatch_vs_error.png")

    fig, ax = plt.subplots()
    ax.plot([row["mean_abs_wrapped_error_rad"] for row in sigma_rows], label="MAE")
    ax.plot([row["q95_abs_wrapped_error_rad"] for row in sigma_rows], label="q95")
    ax.set_xlabel("window frequency-variability quintile")
    ax.set_ylabel("rad")
    ax.legend()
    save(fig, "fig3_frequency_variability_vs_error.png")

    fig, ax = plt.subplots()
    ax.plot([row["phase_bin_deg"] for row in phase_rows], [row.get("mean_signed_error_rad", np.nan) for row in phase_rows], marker="o")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("reference phase (deg)")
    ax.set_ylabel("mean signed error (rad)")
    save(fig, "fig4_reference_phase_vs_signed_error.png")

    fig, ax = plt.subplots()
    image = ax.imshow(interaction["mae_rad"], aspect="auto", origin="lower")
    fig.colorbar(image, ax=ax, label="MAE (rad)")
    ax.set_xlabel("|f_inst - f0| quantile")
    ax.set_ylabel("amplitude quantile")
    save(fig, "fig5_amplitude_frequency_mae_heatmap.png")

    fig, ax = plt.subplots()
    image = ax.imshow(interaction["catastrophic_rate"], aspect="auto", origin="lower")
    fig.colorbar(image, ax=ax, label="catastrophic rate")
    ax.set_xlabel("|f_inst - f0| quantile")
    ax.set_ylabel("amplitude quantile")
    save(fig, "fig6_amplitude_frequency_catastrophic_heatmap.png")

    comparison = catastrophic_comparison(nights)
    labels = list(comparison.keys())
    amp_values = [comparison[label]["mean_amplitude"] if comparison[label] else 0.0 for label in labels]
    delta_values = [comparison[label]["mean_abs_delta_f"] if comparison[label] else 0.0 for label in labels]
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(labels)) - 0.2, amp_values, width=0.4, label="amplitude")
    ax.bar(np.arange(len(labels)) + 0.2, delta_values, width=0.4, label="|delta f|")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    save(fig, "fig7_catastrophic_feature_comparison.png")

    subject_mae = {}
    for data in nights.values():
        subject = str(data["subject"][0])
        subject_mae.setdefault(subject, []).append(float(np.mean(np.abs(wrap(data["error"])))))
    fig, ax = plt.subplots()
    ax.bar(list(subject_mae.keys()), [np.mean(values) for values in subject_mae.values()])
    ax.set_ylabel("mean abs error (rad)")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "fig8_per_subject_trend.png")


def main() -> None:
    started = time.perf_counter()
    nights = collect_features()
    amplitude_rows = stratified_metrics(nights, "amplitude")
    delta_rows = stratified_metrics(nights, "abs_delta_f")
    sigma_rows = stratified_metrics(nights, "sigma_f")
    phase_rows = phase_binned_metrics(nights)
    interaction = interaction_matrix(nights)
    catastrophic = catastrophic_comparison(nights)

    subject_trends = {}
    for night, data in nights.items():
        subject = str(data["subject"][0])
        absolute = np.abs(wrap(data["error"]))

        def low_high(feature):
            edges = np.quantile(data[feature], [0.0, 0.2, 0.8, 1.0])
            low = absolute[data[feature] <= edges[1]]
            high = absolute[data[feature] >= edges[2]]
            return float(np.mean(low)), float(np.mean(high))

        amp_low, amp_high = low_high("amplitude")
        df_low, df_high = low_high("abs_delta_f")
        sigma_low, sigma_high = low_high("sigma_f")
        subject_trends.setdefault(subject, []).append(
            {
                "amplitude_low": amp_low,
                "amplitude_high": amp_high,
                "delta_f_low": df_low,
                "delta_f_high": df_high,
                "sigma_f_low": sigma_low,
                "sigma_f_high": sigma_high,
            }
        )

    subject_summary = {}
    for subject, rows in subject_trends.items():
        subject_summary[subject] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
    amplitude_expected = sum(
        1 for row in subject_summary.values() if row["amplitude_low"] > row["amplitude_high"]
    )
    delta_expected = sum(
        1 for row in subject_summary.values() if row["delta_f_high"] > row["delta_f_low"]
    )
    sigma_expected = sum(
        1 for row in subject_summary.values() if row["sigma_f_high"] > row["sigma_f_low"]
    )
    subject_summary["counts"] = {
        "subject_count": len(subject_trends),
        "amplitude_low_gt_high": amplitude_expected,
        "delta_f_high_gt_low": delta_expected,
        "sigma_f_high_gt_low": sigma_expected,
    }

    for rows, name in (
        (amplitude_rows, "amplitude_strata.csv"),
        (delta_rows, "frequency_mismatch_strata.csv"),
        (sigma_rows, "frequency_variability_strata.csv"),
        (phase_rows, "phase_binned_residuals.csv"),
    ):
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    (OUT / "interaction_matrices.json").write_text(
        json.dumps(interaction, indent=2), encoding="utf-8"
    )
    with (OUT / "subject_trends.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["subject"] + list(next(iter(subject_trends.values()))[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for subject, rows in sorted(subject_trends.items()):
            averaged = {
                key: float(np.mean([row[key] for row in rows])) for key in rows[0]
            }
            writer.writerow({"subject": subject, **averaged})
    figures(nights, amplitude_rows, delta_rows, sigma_rows, phase_rows, interaction)

    pooled_error = np.concatenate([data["error"] for data in nights.values()])
    pooled = _summary(pooled_error)
    report = {
        "frozen": {"window_s": WINDOW_S, "f0_hz": F0, "calibrated": True},
        "pooled": pooled,
        "amplitude_strata": amplitude_rows,
        "frequency_mismatch_strata": delta_rows,
        "frequency_variability_strata": sigma_rows,
        "phase_binned": phase_rows,
        "catastrophic_comparison": catastrophic,
        "subject_trend_summary": subject_summary,
        "interaction": interaction,
        "wall_clock_seconds": time.perf_counter() - started,
        "isolation": {
            "main_code_modified": "NO",
            "production_config_modified": "NO",
            "untouched_test_accessed": "NO",
            "verdict": "DIAGNOSTIC",
        },
    }
    (OUT / "stratification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# P1 error stratification report",
        "",
        f"- pooled MAE: {pooled['mean_abs_wrapped_error_rad']:.6f} rad",
        f"- pooled median: {pooled['median_abs_wrapped_error_rad']:.6f} rad",
        f"- pooled q95: {pooled['q95_abs_wrapped_error_rad']:.6f} rad",
        f"- pooled circular bias: {pooled['circular_bias_rad']:.6f} rad",
        f"- pooled R: {pooled['resultant_length']:.6f}",
        f"- pooled circular SD: {pooled['circular_sd_rad']:.6f} rad",
        f"- catastrophic rate >135 deg: {pooled['catastrophic_rate_gt_135deg']:.6f}",
        "",
        "## Isolation status",
        "- Main code modified: NO",
        "- Production config modified: NO",
        "- Untouched test accessed: NO",
        f"- Experiment directory: {OUT}",
        "",
        "## VERDICT",
        "DIAGNOSTIC",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["pooled"], indent=2, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
