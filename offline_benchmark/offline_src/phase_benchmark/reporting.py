from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from offline_src.phase_benchmark.pll import CLASSIC_PLL_WITHHELD_REASON


SIMPLICITY_ORDER = {
    "offline_hilbert": 0,
    "cecHT": 1,
    "ferster_pll": 2,
    "pll": 3,
    "ARHT": 4,
}


def select_best_realtime(results: pd.DataFrame) -> dict:
    realtime = results[results["method"] != "offline_hilbert"].copy()
    if realtime.empty:
        return {}
    realtime["simplicity_rank"] = realtime["method"].map(SIMPLICITY_ORDER).fillna(99)
    ranked = realtime.sort_values(
        ["accuracy_within_pi_4", "median_abs_phase_error", "circular_resultant_length", "simplicity_rank"],
        ascending=[False, True, False, True],
    )
    best = ranked.iloc[0].to_dict()
    close = ranked[
        ranked["accuracy_within_pi_4"] >= float(best["accuracy_within_pi_4"]) - 0.05
    ].sort_values(["simplicity_rank", "median_abs_phase_error"], ascending=[True, True])
    if not close.empty:
        best = close.iloc[0].to_dict()
    return best


def write_summary(results: pd.DataFrame, output_dir: Path, mode: str) -> dict:
    best_realtime = select_best_realtime(results)
    withheld_methods = [
        str(record["method"])
        for record in results.to_dict(orient="records")
        if record.get("withheld")
    ]
    offline = results[results["method"] == "offline_hilbert"]
    summary = {
        "mode": mode,
        "evaluation_edge_trim_sec": getattr(results, "attrs", {}).get("evaluation_edge_trim_sec"),
        "fieldtrip_buffer_sec": getattr(results, "attrs", {}).get("fieldtrip_buffer_sec"),
        "ranking_primary_metric": "accuracy_within_pi_4",
        "ranking_secondary_metric": "median_abs_phase_error",
        "ranking_tertiary_metric": "circular_resultant_length",
        "best_realtime_method": best_realtime,
        "offline_hilbert": offline.iloc[0].to_dict() if not offline.empty else None,
        "preliminary_preview": bool((results["n_triggers"] < 100).any()),
        "withheld_methods": withheld_methods,
        "withheld_reason": CLASSIC_PLL_WITHHELD_REASON if withheld_methods else None,
        "methods": results.to_dict(orient="records"),
    }
    path = output_dir / "phase_method_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_report(results: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    best = summary.get("best_realtime_method") or {}
    preliminary = "Yes" if summary.get("preliminary_preview") else "No"
    edge_trim_sec = summary.get("evaluation_edge_trim_sec")
    fieldtrip_buffer_sec = summary.get("fieldtrip_buffer_sec")
    columns = [
        "method",
        "n_triggers",
        "trigger_rate_per_min",
        "accuracy_within_pi_6",
        "accuracy_within_pi_4",
        "accuracy_within_pi_3",
        "median_abs_phase_error",
        "circular_resultant_length",
        "runtime_ms_per_epoch",
        "failure_rate",
    ]
    table_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in results[columns].iterrows():
        table_lines.append(
            "| "
            + " | ".join(
                f"{row[col]:.3f}" if isinstance(row[col], float) else str(row[col])
                for col in columns
            )
            + " |"
        )
    lines = [
        "# Phase Method Benchmark Report",
        "",
        "## Scope",
        "Independent quick benchmark of slow-oscillation phase targeting methods. The main pipeline is not modified.",
        "",
        "## Ground Truth",
        "Offline non-causal Hilbert phase in the 0.5-1.25 Hz SO band is used as the reference phase.",
        (
            f"Hilbert ground truth and phase methods are computed on epochs with "
            f"{fieldtrip_buffer_sec:.1f}s FieldTrip-style buffer on each side, then cropped to the original epoch."
            if fieldtrip_buffer_sec is not None and fieldtrip_buffer_sec > 0
            else "Hilbert ground truth and phase methods are computed on the original epoch without extra buffer."
        ),
        (
            f"Triggers are evaluated only inside the non-edge analysis window after trimming "
            f"{edge_trim_sec:.1f}s from each epoch edge."
            if edge_trim_sec is not None
            else "Triggers are evaluated only inside the configured non-edge analysis window."
        ),
        "",
        "## Methods",
        "- offline_hilbert: non-causal ideal upper bound, not a realtime method.",
        "- cecHT: calibrated endpoint-corrected Hilbert transform adapted from the cecHT source.",
        "- ferster_pll: first-order PLL baseline adapted to the SO band.",
        "- pll: withheld from this distribution (research-internship artifact; see pll.py).",
        "- ARHT: Opitz-style AR forward prediction + Hilbert method using past EEG and predicted future EEG.",
        "",
        "## Withheld Methods",
        (
            "- "
            + ", ".join(summary.get("withheld_methods") or [])
            + ": not runnable in this distribution. "
            + str(summary.get("withheld_reason"))
            if summary.get("withheld_methods")
            else "No method is withheld in this distribution."
        ),
        "",
        "## Preview Status",
        f"Preliminary preview due to low trigger count in at least one method: {preliminary}",
        "",
        "## Results",
        "\n".join(table_lines),
        "",
        "## Recommended Realtime Method",
        (
            f"{best.get('method', 'NA')} with accuracy_within_pi_4="
            f"{best.get('accuracy_within_pi_4', 0):.3f}, median_abs_phase_error="
            f"{best.get('median_abs_phase_error', 0):.3f}, runtime_ms_per_epoch="
            f"{best.get('runtime_ms_per_epoch', 0):.1f}."
        ),
        "",
        "## Selection Logic",
        "Primary metric is accuracy_within_pi_4. Secondary metric is median_abs_phase_error. "
        "If two realtime methods are within 0.05 accuracy, the simpler and faster method is preferred.",
        "",
        "## Limitations",
        "- This is a preview benchmark, not a final clinical validation.",
        "- Offline Hilbert is an upper bound and uses future signal.",
        "- cecHT uses a fixed slow-oscillation center frequency for calibration; subject- or epoch-specific frequency tracking is not yet enabled.",
        "- Opitz-style AR was originally designed for alpha-band EEG-TMS; this benchmark uses SO-scaled parameters rather than the original alpha-band constants.",
        "- The ferster_pll row is a simplified experimental first-order PLL baseline; the canonical PLL is withheld from this distribution.",
        "- The benchmark uses a limited number of N3/N4 epochs by default.",
    ]
    (output_dir / "phase_method_report.md").write_text("\n".join(lines), encoding="utf-8")
