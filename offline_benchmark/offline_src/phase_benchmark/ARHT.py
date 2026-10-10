from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, hilbert, sosfiltfilt

from offline_src.phase_benchmark.config import PhaseBenchmarkConfig
from offline_src.trigger_simulator import circular_distance


@dataclass(frozen=True)
class OpitzStyleARConfig:
    """Sleep-SO adaptation of the CL-phase AR-forward phase estimator.

    OpitzLab/CL-phase was designed for alpha-band EEG-TMS. This configuration
    keeps the important algorithmic ideas, but rescales window length, AR order,
    and edge trimming for 0.5-1.25 Hz slow oscillations sampled at 100 Hz.
    """

    past_window_sec: float
    ar_order: int
    edge_trim_sec: float
    prediction_horizon_ms: float
    hilbert_padding_ms: float
    update_step_ms: float
    phase_tolerance_rad: float
    amp_percentile: int
    technical_delay_ms: float
    refractory_sec: float
    target_phase: float
    band: tuple[float, float]


def config_from_benchmark(cfg: PhaseBenchmarkConfig) -> OpitzStyleARConfig:
    return OpitzStyleARConfig(
        past_window_sec=cfg.opitz_ar_past_window_sec,
        ar_order=cfg.opitz_ar_order,
        edge_trim_sec=cfg.opitz_ar_edge_trim_sec,
        prediction_horizon_ms=cfg.opitz_ar_prediction_horizon_ms,
        hilbert_padding_ms=cfg.opitz_ar_hilbert_padding_ms,
        update_step_ms=cfg.opitz_ar_update_step_ms,
        phase_tolerance_rad=cfg.opitz_ar_phase_tolerance_rad,
        amp_percentile=cfg.opitz_ar_amp_percentile,
        technical_delay_ms=cfg.opitz_ar_technical_delay_ms,
        refractory_sec=cfg.refractory_sec,
        target_phase=cfg.target_phase,
        band=cfg.so_band,
    )


def _zero_phase_bandpass_window(signal: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Bandpass one past-only window with forward-backward filtering.

    This is non-causal inside the already observed window, matching the
    CL-phase style of repeatedly filtering a rolling window without using
    samples beyond the current time.
    """
    high = min(float(band[1]), 0.5 * sfreq - 0.1)
    low = max(float(band[0]), 0.05)
    if high <= low:
        raise ValueError(f"Invalid bandpass for sfreq={sfreq}: {band}")
    sos = butter(2, [low, high], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, signal)


def _yule_walker_coefficients(signal: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Fit x[t] = sum(phi_i * x[t-i]) + mean using Yule-Walker equations."""
    if len(signal) <= order + 2:
        raise ValueError("AR fit window is too short.")

    mean = float(np.mean(signal))
    centered = np.asarray(signal, dtype=float) - mean
    autocorr = np.correlate(centered, centered, mode="full")[len(centered) - 1 : len(centered) + order]
    autocorr = autocorr / max(len(centered), 1)
    if not np.isfinite(autocorr).all() or autocorr[0] <= np.finfo(float).eps:
        raise ValueError("Degenerate AR fit window.")

    # Small diagonal loading protects solve_toeplitz from nearly periodic or
    # very flat windows without changing the estimator's Yule-Walker character.
    autocorr = autocorr.copy()
    autocorr[0] += 1e-8 * autocorr[0]
    coeff = solve_toeplitz((autocorr[:order], autocorr[:order]), autocorr[1 : order + 1])
    if not np.isfinite(coeff).all():
        raise ValueError("Non-finite AR coefficients.")
    return np.asarray(coeff, dtype=float), mean


def _predict_ar(history: np.ndarray, coeff: np.ndarray, mean: float, n_future: int) -> np.ndarray:
    centered_history = list(np.asarray(history, dtype=float) - mean)
    predictions = []
    order = len(coeff)
    for _ in range(n_future):
        recent = np.asarray(centered_history[-order:][::-1], dtype=float)
        next_centered = float(np.dot(coeff, recent))
        # Avoid unstable runaway forecasts from occasional bad windows.
        if not np.isfinite(next_centered) or abs(next_centered) > 20 * (np.std(centered_history) + 1e-12):
            raise ValueError("Unstable AR forecast.")
        centered_history.append(next_centered)
        predictions.append(next_centered + mean)
    return np.asarray(predictions, dtype=float)


def estimate_opitz_style_ar_phase(
    signal: np.ndarray,
    current_index: int,
    sfreq: float,
    cfg: OpitzStyleARConfig,
) -> dict[str, float] | None:
    past_samples = int(round(cfg.past_window_sec * sfreq))
    edge_samples = int(round(cfg.edge_trim_sec * sfreq))
    horizon_samples = max(1, int(round(cfg.prediction_horizon_ms / 1000.0 * sfreq)))
    technical_delay_samples = max(0, int(round(cfg.technical_delay_ms / 1000.0 * sfreq)))
    padding_samples = max(0, int(round(cfg.hilbert_padding_ms / 1000.0 * sfreq)))
    trigger_offset = horizon_samples + technical_delay_samples
    forecast_samples = trigger_offset + padding_samples
    start = current_index - past_samples
    trigger_index = current_index + trigger_offset
    if start < 0 or trigger_index >= len(signal):
        return None

    window = np.asarray(signal[start:current_index], dtype=float)
    if len(window) < cfg.ar_order + 2 * edge_samples + 3:
        return None

    try:
        filtered_window = _zero_phase_bandpass_window(window, sfreq, cfg.band)
        fit_signal = filtered_window[edge_samples:-edge_samples] if edge_samples else filtered_window
        if len(fit_signal) <= cfg.ar_order + 2:
            return None
        coeff, mean = _yule_walker_coefficients(fit_signal, cfg.ar_order)
        predicted_future = _predict_ar(fit_signal, coeff, mean, forecast_samples)
        hilbert_context = np.concatenate([fit_signal, predicted_future])
        analytic = hilbert(hilbert_context)
        phase_index = len(fit_signal) + trigger_offset - 1
        return {
            "trigger_index": float(trigger_index),
            "phase": float(np.angle(analytic[phase_index])),
            "amplitude": float(np.abs(analytic[phase_index])),
        }
    except (ValueError, np.linalg.LinAlgError):
        return None


def simulate_opitz_style_ar_triggers(
    signal: np.ndarray,
    true_phase: np.ndarray,
    sfreq: float,
    cfg: OpitzStyleARConfig,
) -> dict[str, np.ndarray | float | int]:
    past_samples = int(round(cfg.past_window_sec * sfreq))
    horizon_samples = max(1, int(round(cfg.prediction_horizon_ms / 1000.0 * sfreq)))
    technical_delay_samples = max(0, int(round(cfg.technical_delay_ms / 1000.0 * sfreq)))
    padding_samples = max(0, int(round(cfg.hilbert_padding_ms / 1000.0 * sfreq)))
    forecast_samples = horizon_samples + technical_delay_samples + padding_samples
    step_samples = max(1, int(round(cfg.update_step_ms / 1000.0 * sfreq)))
    refractory_samples = max(1, int(round(cfg.refractory_sec * sfreq)))
    stop = len(signal) - forecast_samples

    candidate_indices = []
    candidate_phases = []
    candidate_amplitudes = []
    failed_windows = 0
    total_windows = 0
    for current_index in range(past_samples, stop, step_samples):
        total_windows += 1
        result = estimate_opitz_style_ar_phase(signal, current_index, sfreq, cfg)
        if result is None:
            failed_windows += 1
            continue
        candidate_indices.append(int(result["trigger_index"]))
        candidate_phases.append(float(result["phase"]))
        candidate_amplitudes.append(float(result["amplitude"]))

    if not candidate_indices:
        return {
            "trigger_indices": np.asarray([], dtype=int),
            "estimated_trigger_phases": np.asarray([], dtype=float),
            "true_trigger_phases": np.asarray([], dtype=float),
            "phase_errors": np.asarray([], dtype=float),
            "candidate_failure_rate": 1.0 if total_windows else 0.0,
            "n_candidate_windows": total_windows,
        }

    candidate_indices_arr = np.asarray(candidate_indices, dtype=int)
    candidate_phases_arr = np.asarray(candidate_phases, dtype=float)
    candidate_amplitudes_arr = np.asarray(candidate_amplitudes, dtype=float)
    amp_threshold = float(np.percentile(candidate_amplitudes_arr, cfg.amp_percentile))

    trigger_indices = []
    estimated_trigger_phases = []
    true_trigger_phases = []
    phase_errors = []
    last_trigger = -refractory_samples
    for idx, phase, amp in zip(candidate_indices_arr, candidate_phases_arr, candidate_amplitudes_arr):
        if idx - last_trigger < refractory_samples:
            continue
        phase_error_to_target = float(circular_distance(np.array([phase]), cfg.target_phase)[0])
        if abs(phase_error_to_target) <= cfg.phase_tolerance_rad and amp >= amp_threshold:
            true_at_trigger = float(true_phase[idx])
            err = float(circular_distance(np.array([true_at_trigger]), cfg.target_phase)[0])
            trigger_indices.append(int(idx))
            estimated_trigger_phases.append(float(phase))
            true_trigger_phases.append(true_at_trigger)
            phase_errors.append(err)
            last_trigger = int(idx)

    return {
        "trigger_indices": np.asarray(trigger_indices, dtype=int),
        "estimated_trigger_phases": np.asarray(estimated_trigger_phases, dtype=float),
        "true_trigger_phases": np.asarray(true_trigger_phases, dtype=float),
        "phase_errors": np.asarray(phase_errors, dtype=float),
        "candidate_failure_rate": float(failed_windows / total_windows) if total_windows else 0.0,
        "n_candidate_windows": total_windows,
    }
