from __future__ import annotations

import numpy as np
import pandas as pd

from offline_src.phase_benchmark.config import PhaseBenchmarkConfig
from offline_src.phase_benchmark.data import BenchmarkEpoch
from offline_src.phase_benchmark.methods import method_registry, true_phase_info


def _method_metrics(method_name: str, rows: list[dict], cfg: PhaseBenchmarkConfig) -> dict:
    # A method can legitimately produce no trigger at all (for example one whose
    # implementation is withheld from this distribution), so the concatenation
    # must tolerate an empty input instead of aborting the whole benchmark.
    error_arrays = [row["phase_errors"] for row in rows if len(row["phase_errors"])]
    true_phase_arrays = [row["true_trigger_phases"] for row in rows if len(row["true_trigger_phases"])]
    phase_errors = np.concatenate(error_arrays) if error_arrays else np.asarray([], dtype=float)
    true_phases = np.concatenate(true_phase_arrays) if true_phase_arrays else np.asarray([], dtype=float)
    n_epochs = len(rows)
    n_subjects = len({row["subject_id"] for row in rows})
    n_triggers = len(phase_errors)
    minutes = n_epochs * 30.0 / 60.0
    abs_errors = np.abs(phase_errors)

    if n_triggers:
        circular_mean_error = float(np.angle(np.mean(np.exp(1j * phase_errors))))
        circular_resultant_length = float(np.abs(np.mean(np.exp(1j * true_phases))))
        mean_abs_phase_error = float(np.mean(abs_errors))
        median_abs_phase_error = float(np.median(abs_errors))
        acc_pi_6 = float(np.mean(abs_errors <= cfg.success_windows["pi_6"]))
        acc_pi_4 = float(np.mean(abs_errors <= cfg.success_windows["pi_4"]))
        acc_pi_3 = float(np.mean(abs_errors <= cfg.success_windows["pi_3"]))
    else:
        circular_mean_error = 0.0
        circular_resultant_length = 0.0
        mean_abs_phase_error = 0.0
        median_abs_phase_error = 0.0
        acc_pi_6 = 0.0
        acc_pi_4 = 0.0
        acc_pi_3 = 0.0

    return {
        "method": method_name,
        "n_subjects": n_subjects,
        "n_epochs": n_epochs,
        "n_triggers": n_triggers,
        "trigger_rate_per_min": n_triggers / minutes if minutes else 0.0,
        "accuracy_within_pi_6": acc_pi_6,
        "accuracy_within_pi_4": acc_pi_4,
        "accuracy_within_pi_3": acc_pi_3,
        "mean_abs_phase_error": mean_abs_phase_error,
        "median_abs_phase_error": median_abs_phase_error,
        "circular_mean_error": circular_mean_error,
        "circular_resultant_length": circular_resultant_length,
        "runtime_ms_per_epoch": float(np.mean([row["runtime_ms"] for row in rows])) if rows else 0.0,
        "failure_rate": float(np.mean([row["failed"] for row in rows])) if rows else 0.0,
        "preliminary_preview": bool(n_triggers < cfg.preliminary_trigger_threshold),
        "withheld": bool(any(row.get("withheld") for row in rows)),
    }


def evaluate_methods(epochs: list[BenchmarkEpoch], cfg: PhaseBenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method_rows = []
    epoch_rows = []
    trigger_rows = []
    registry = method_registry(cfg.method_names)

    for epoch_idx, epoch in enumerate(epochs, start=1):
        print(
            f"[Phase benchmark epoch {epoch_idx}/{len(epochs)}] "
            f"subject={epoch.subject_id:02d} epoch={epoch.epoch_index}",
            flush=True,
        )
        phase_info = true_phase_info(epoch.signal, epoch.sfreq, cfg)
        phase_info["evaluation_crop_start_sample"] = int(epoch.crop_start_sample)
        phase_info["evaluation_crop_stop_sample"] = (
            int(epoch.crop_stop_sample) if epoch.crop_stop_sample is not None else len(epoch.signal)
        )
        for method_fn in registry:
            result = method_fn(epoch.signal, phase_info, epoch.sfreq, cfg)
            row = {
                "method": result["method"],
                "subject_id": epoch.subject_id,
                "recording_id": epoch.recording_id,
                "epoch_index": epoch.epoch_index,
                "n_triggers": int(len(result["phase_errors"])),
                "runtime_ms": float(result["runtime_ms"]),
                "failed": bool(result["failed"]),
                "withheld": bool(result.get("withheld", False)),
                "phase_errors": result["phase_errors"],
                "true_trigger_phases": result["true_trigger_phases"],
            }
            method_rows.append(row)
            epoch_rows.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"phase_errors", "true_trigger_phases"}
                }
            )
            for trigger_idx, (phase_error, true_phase) in enumerate(
                zip(result["phase_errors"], result["true_trigger_phases"])
            ):
                trigger_rows.append(
                    {
                        "method": result["method"],
                        "subject_id": epoch.subject_id,
                        "recording_id": epoch.recording_id,
                        "epoch_index": epoch.epoch_index,
                        "trigger_number": trigger_idx,
                        "phase_error": float(phase_error),
                        "abs_phase_error": float(abs(phase_error)),
                        "true_trigger_phase": float(true_phase),
                    }
                )

    results = []
    for method_name in sorted({row["method"] for row in method_rows}):
        rows = [row for row in method_rows if row["method"] == method_name]
        results.append(_method_metrics(method_name, rows, cfg))

    results_df = pd.DataFrame(results).sort_values(
        ["accuracy_within_pi_4", "median_abs_phase_error", "circular_resultant_length"],
        ascending=[False, True, False],
    )
    return results_df, pd.DataFrame(epoch_rows), pd.DataFrame(trigger_rows)
