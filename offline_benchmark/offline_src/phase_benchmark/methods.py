from __future__ import annotations

import sys
from time import perf_counter

import numpy as np
from scipy.signal import bessel, butter, sosfilt

from offline_src.hilbert_phase import HilbertPhaseEstimator
from offline_src.phase_benchmark.cecHT import ECHT
from offline_src.phase_benchmark.config import PhaseBenchmarkConfig
from offline_src.phase_benchmark.ARHT import config_from_benchmark, simulate_opitz_style_ar_triggers
from offline_src.phase_benchmark.pll import (
    CLASSIC_PLL_IS_WITHHELD,
    CLASSIC_PLL_WITHHELD_REASON,
    run_classic_pll_core,
)
from offline_src.trigger_simulator import circular_distance


def _evaluation_mask(
    candidate_indices: np.ndarray,
    n_samples: int,
    sfreq: float,
    cfg: PhaseBenchmarkConfig,
    crop_start_sample: int = 0,
    crop_stop_sample: int | None = None,
) -> np.ndarray:
    trim_samples = int(round(cfg.evaluation_edge_trim_sec * sfreq))
    crop_start = max(0, int(crop_start_sample))
    crop_stop = n_samples if crop_stop_sample is None else min(n_samples, int(crop_stop_sample))
    start = crop_start + max(0, trim_samples)
    stop = crop_stop - max(0, trim_samples)
    if stop <= start:
        return np.zeros(len(candidate_indices), dtype=bool)
    return (candidate_indices >= start) & (candidate_indices < stop)


def _internal_target_phase(evaluation_target_phase: float, phase_offset: float) -> float:
    return float(np.angle(np.exp(1j * (evaluation_target_phase - phase_offset))))


def _simulate_candidate_triggers(
    candidate_indices: np.ndarray,
    estimated_phases: np.ndarray,
    estimated_amplitudes: np.ndarray,
    true_phase: np.ndarray,
    sfreq: float,
    cfg: PhaseBenchmarkConfig,
    evaluation_target_phase: float | None = None,
    evaluation_crop_start_sample: int = 0,
    evaluation_crop_stop_sample: int | None = None,
) -> dict:
    if len(candidate_indices) == 0:
        return _empty_result()

    valid_mask = _evaluation_mask(
        candidate_indices,
        len(true_phase),
        sfreq,
        cfg,
        crop_start_sample=evaluation_crop_start_sample,
        crop_stop_sample=evaluation_crop_stop_sample,
    )
    if not np.any(valid_mask):
        return _empty_result()

    amp_threshold = float(np.percentile(estimated_amplitudes[valid_mask], cfg.amp_percentile))
    refractory_samples = int(round(cfg.refractory_sec * sfreq))
    trigger_indices = []
    estimated_trigger_phases = []
    true_trigger_phases = []
    phase_errors = []
    last_trigger = -refractory_samples
    prev_diff = None
    eval_target = cfg.target_phase if evaluation_target_phase is None else float(evaluation_target_phase)
    valid_stop = len(true_phase) if evaluation_crop_stop_sample is None else int(evaluation_crop_stop_sample)
    valid_stop -= int(round(cfg.evaluation_edge_trim_sec * sfreq))

    for idx, phase, amplitude, valid in zip(candidate_indices, estimated_phases, estimated_amplitudes, valid_mask):
        if not valid:
            if idx >= valid_stop:
                break
            continue
        diff = float(circular_distance(np.array([phase]), cfg.target_phase)[0])
        if idx - last_trigger < refractory_samples:
            prev_diff = diff
            continue
        crossing = prev_diff is not None and prev_diff < 0 <= diff
        high_amp = amplitude >= amp_threshold
        if crossing and high_amp:
            true_at_trigger = float(true_phase[idx])
            err = float(circular_distance(np.array([true_at_trigger]), eval_target)[0])
            trigger_indices.append(int(idx))
            estimated_trigger_phases.append(float(phase))
            true_trigger_phases.append(true_at_trigger)
            phase_errors.append(err)
            last_trigger = int(idx)
        prev_diff = diff

    return {
        "trigger_indices": np.asarray(trigger_indices, dtype=int),
        "estimated_trigger_phases": np.asarray(estimated_trigger_phases, dtype=float),
        "true_trigger_phases": np.asarray(true_trigger_phases, dtype=float),
        "phase_errors": np.asarray(phase_errors, dtype=float),
    }


def _empty_result() -> dict:
    return {
        "trigger_indices": np.asarray([], dtype=int),
        "estimated_trigger_phases": np.asarray([], dtype=float),
        "true_trigger_phases": np.asarray([], dtype=float),
        "phase_errors": np.asarray([], dtype=float),
    }


def _cecht_center_frequency(cfg: PhaseBenchmarkConfig) -> float:
    return float(np.mean(cfg.so_band))


def _cecht_endpoint_phase(
    echt: ECHT,
    signal: np.ndarray,
    endpoint_index: int,
    sfreq: float,
    cfg: PhaseBenchmarkConfig,
) -> tuple[float, float]:
    window_samples = int(round(cfg.ec_window_sec * sfreq))
    start = endpoint_index - window_samples + 1
    if start < 0:
        raise ValueError("Not enough past samples for cecHT endpoint Hilbert.")
    segment = signal[start : endpoint_index + 1]
    endpoint = np.ravel(echt.transform(segment))[-1]
    return float(np.angle(endpoint)), float(np.abs(endpoint))


def run_offline_hilbert(signal: np.ndarray, true_phase_info: dict, sfreq: float, cfg: PhaseBenchmarkConfig) -> dict:
    start = perf_counter()
    candidate_indices = np.arange(len(signal), dtype=int)
    result = _simulate_candidate_triggers(
        candidate_indices,
        true_phase_info["phase"],
        true_phase_info["amplitude"],
        true_phase_info["phase"],
        sfreq,
        cfg,
        evaluation_crop_start_sample=int(true_phase_info.get("evaluation_crop_start_sample", 0)),
        evaluation_crop_stop_sample=true_phase_info.get("evaluation_crop_stop_sample"),
    )
    elapsed = perf_counter() - start
    return {
        "method": "offline_hilbert",
        **result,
        "runtime_ms": elapsed * 1000.0,
        "failed": False,
    }


def run_ar_forward_hilbert(
    signal: np.ndarray,
    true_phase_info: dict,
    sfreq: float,
    cfg: PhaseBenchmarkConfig,
) -> dict:
    start = perf_counter()
    ar_cfg = config_from_benchmark(cfg)
    result = simulate_opitz_style_ar_triggers(signal, true_phase_info["phase"], sfreq, ar_cfg)
    elapsed = perf_counter() - start
    return {
        "method": "ARHT",
        "trigger_indices": result["trigger_indices"],
        "true_trigger_phases": result["true_trigger_phases"],
        "phase_errors": result["phase_errors"],
        "runtime_ms": elapsed * 1000.0,
        "failed": False,
        "candidate_failure_rate": result["candidate_failure_rate"],
        "opitz_ar_past_window_sec": cfg.opitz_ar_past_window_sec,
        "opitz_ar_order": cfg.opitz_ar_order,
        "opitz_ar_edge_trim_sec": cfg.opitz_ar_edge_trim_sec,
        "opitz_ar_prediction_horizon_ms": cfg.opitz_ar_prediction_horizon_ms,
        "opitz_ar_hilbert_padding_ms": cfg.opitz_ar_hilbert_padding_ms,
        "opitz_ar_phase_tolerance_rad": cfg.opitz_ar_phase_tolerance_rad,
        "opitz_ar_technical_delay_ms": cfg.opitz_ar_technical_delay_ms,
    }


def _make_cecht(sfreq: float, cfg: PhaseBenchmarkConfig) -> ECHT:
    return ECHT(
        l_freq=cfg.so_band[0],
        h_freq=cfg.so_band[1],
        sfreq=sfreq,
        filt_order=cfg.cecht_filter_order,
        filter_type="butter",
        calibrate=cfg.cecht_calibrate,
        f0=_cecht_center_frequency(cfg),
    )


def _causal_bandpass(
    signal: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    order: int = 2,
    filter_type: str = "butter",
) -> np.ndarray:
    high = min(float(band[1]), 0.5 * sfreq - 0.1)
    low = max(float(band[0]), 0.05)
    if high <= low:
        raise ValueError(f"Invalid bandpass for sfreq={sfreq}: {band}")
    if filter_type == "butter":
        sos = butter(order, [low, high], btype="band", fs=sfreq, output="sos")
    elif filter_type == "bessel":
        sos = bessel(order, [low, high], btype="band", fs=sfreq, output="sos", norm="phase")
    else:
        raise ValueError(f"Unknown causal bandpass filter_type: {filter_type}")
    return sosfilt(sos, signal)


def _run_pll_candidates(
    signal: np.ndarray,
    sfreq: float,
    cfg: PhaseBenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reproduce the Ferster 2022 first-order PLL architecture for sample-level candidates.

    The phase detector multiplies the preprocessed EEG sample by the previous NCO
    cosine output. The resulting error controls the NCO phase directly through a
    single combined K_PLL gain, matching the paper's first-order simplification.
    """
    filtered = _causal_bandpass(signal, sfreq, cfg.ferster_pll_input_band)
    scale = float(np.nanpercentile(np.abs(filtered), 95))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(filtered))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    x = filtered / scale

    base_omega = 2.0 * np.pi * cfg.ferster_pll_center_frequency_hz / sfreq
    gain = float(cfg.ferster_pll_gain)
    sample_step = max(1, int(cfg.ferster_pll_sample_step))
    phase_state = 0.0
    candidate_indices = []
    phases = []
    amplitudes = []

    for idx, sample in enumerate(x):
        nco_previous = np.cos(phase_state)
        phase_detector_error = float(sample * nco_previous)
        phase_state = float(np.angle(np.exp(1j * (phase_state + base_omega - gain * phase_detector_error))))
        if idx % sample_step == 0:
            candidate_indices.append(idx)
            phases.append(phase_state)
            amplitudes.append(abs(float(filtered[idx])))

    return (
        np.asarray(candidate_indices, dtype=int),
        np.asarray(phases, dtype=float),
        np.asarray(amplitudes, dtype=float),
    )


def run_pll(
    signal: np.ndarray,
    true_phase_info: dict,
    sfreq: float,
    cfg: PhaseBenchmarkConfig,
) -> dict:
    start = perf_counter()
    try:
        candidate_indices, phases, amplitudes = _run_pll_candidates(signal, sfreq, cfg)
        internal_target = _internal_target_phase(cfg.target_phase, cfg.ferster_pll_phase_offset)
        trigger_cfg = PhaseBenchmarkConfig(
            **{
                **cfg.__dict__,
                "target_phase": internal_target,
                "amp_percentile": cfg.ferster_pll_amp_percentile,
                "refractory_sec": cfg.ferster_pll_refractory_sec,
            }
        )
        result = _simulate_candidate_triggers(
            candidate_indices,
            phases,
            amplitudes,
            true_phase_info["phase"],
            sfreq,
            trigger_cfg,
            evaluation_target_phase=cfg.target_phase,
            evaluation_crop_start_sample=int(true_phase_info.get("evaluation_crop_start_sample", 0)),
            evaluation_crop_stop_sample=true_phase_info.get("evaluation_crop_stop_sample"),
        )
        failed = False
    except Exception:
        result = _empty_result()
        failed = True
    elapsed = perf_counter() - start
    return {
        "method": "ferster_pll",
        **result,
        "runtime_ms": elapsed * 1000.0,
        "failed": failed,
        "ferster_pll_center_frequency_hz": cfg.ferster_pll_center_frequency_hz,
        "ferster_pll_gain": cfg.ferster_pll_gain,
        "ferster_pll_phase_offset": cfg.ferster_pll_phase_offset,
        "ferster_pll_internal_target_phase": _internal_target_phase(cfg.target_phase, cfg.ferster_pll_phase_offset),
        "ferster_pll_input_band_low": cfg.ferster_pll_input_band[0],
        "ferster_pll_input_band_high": cfg.ferster_pll_input_band[1],
        "ferster_pll_amp_percentile": cfg.ferster_pll_amp_percentile,
        "ferster_pll_refractory_sec": cfg.ferster_pll_refractory_sec,
    }


_WITHHELD_NOTICE_PRINTED = False


def _announce_withheld_once() -> None:
    """Print the withheld-implementation notice once per process."""
    global _WITHHELD_NOTICE_PRINTED
    if _WITHHELD_NOTICE_PRINTED:
        return
    _WITHHELD_NOTICE_PRINTED = True
    print(f"[withheld] method=pll: {CLASSIC_PLL_WITHHELD_REASON}", file=sys.stderr, flush=True)


def run_classic_pll(
    signal: np.ndarray,
    true_phase_info: dict,
    sfreq: float,
    cfg: PhaseBenchmarkConfig,
) -> dict:
    start = perf_counter()
    input_scale = 1.0
    try:
        filtered = _causal_bandpass(
            signal,
            sfreq,
            cfg.classic_pll_input_band,
            order=cfg.classic_pll_filter_order,
            filter_type=cfg.classic_pll_filter_type,
        )
        pll_input = filtered
        if cfg.classic_pll_normalize_input:
            input_scale = float(np.nanpercentile(np.abs(filtered), cfg.classic_pll_normalize_percentile))
            if not np.isfinite(input_scale) or input_scale <= 0:
                input_scale = float(np.std(filtered))
            if not np.isfinite(input_scale) or input_scale <= 0:
                input_scale = 1.0
            pll_input = filtered / input_scale
        pll = run_classic_pll_core(
            pll_input,
            sfreq=sfreq,
            fc=cfg.classic_pll_center_frequency_hz,
            kvco=cfg.classic_pll_kvco,
            ka=cfg.classic_pll_ka,
            loop_b=cfg.classic_pll_loop_filter_b,
            loop_a=cfg.classic_pll_loop_filter_a,
        )
        sample_step = max(1, int(cfg.classic_pll_sample_step))
        candidate_indices = np.arange(0, len(signal), sample_step, dtype=int)
        phases = pll["phase"][candidate_indices]
        amplitudes = np.abs(filtered[candidate_indices])
        internal_target = _internal_target_phase(cfg.target_phase, cfg.classic_pll_phase_offset)
        trigger_cfg = PhaseBenchmarkConfig(
            **{
                **cfg.__dict__,
                "target_phase": internal_target,
                "amp_percentile": cfg.classic_pll_amp_percentile,
                "refractory_sec": cfg.classic_pll_refractory_sec,
            }
        )
        result = _simulate_candidate_triggers(
            candidate_indices,
            phases,
            amplitudes,
            true_phase_info["phase"],
            sfreq,
            trigger_cfg,
            evaluation_target_phase=cfg.target_phase,
            evaluation_crop_start_sample=int(true_phase_info.get("evaluation_crop_start_sample", 0)),
            evaluation_crop_stop_sample=true_phase_info.get("evaluation_crop_stop_sample"),
        )
        failed = False
    except Exception as exc:
        elapsed = perf_counter() - start
        if CLASSIC_PLL_IS_WITHHELD:
            # pll.py is the result of a research internship in industry; the
            # canonical classic PLL implementation is withheld from this
            # distribution for intellectual-property reasons. Report that
            # explicitly instead of returning a silent empty row.
            _announce_withheld_once()
        else:
            print(
                f"[failed] method=pll: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        return {
            "method": "pll",
            **_empty_result(),
            "runtime_ms": elapsed * 1000.0,
            "failed": True,
            "withheld": CLASSIC_PLL_IS_WITHHELD,
            "withheld_reason": (
                CLASSIC_PLL_WITHHELD_REASON if CLASSIC_PLL_IS_WITHHELD else None
            ),
        }
    elapsed = perf_counter() - start
    return {
        "method": "pll",
        **result,
        "runtime_ms": elapsed * 1000.0,
        "failed": failed,
    }


def run_cecht_hilbert(
    signal: np.ndarray,
    true_phase_info: dict,
    sfreq: float,
    cfg: PhaseBenchmarkConfig,
) -> dict:
    start = perf_counter()
    step = max(1, int(round(cfg.update_step_ms / 1000.0 * sfreq)))
    window = int(round(cfg.ec_window_sec * sfreq))
    candidate_indices = []
    phases = []
    amplitudes = []
    failed = False
    if window >= len(signal):
        return {"method": "cecHT", **_empty_result(), "runtime_ms": 0.0, "failed": True}
    echt = _make_cecht(sfreq, cfg)
    echt.fit(signal[:window])
    for idx in range(window - 1, len(signal), step):
        try:
            phase, amp = _cecht_endpoint_phase(echt, signal, idx, sfreq, cfg)
        except Exception:
            failed = True
            continue
        candidate_indices.append(idx)
        phases.append(phase)
        amplitudes.append(amp)
    result = _simulate_candidate_triggers(
        np.asarray(candidate_indices, dtype=int),
        np.asarray(phases, dtype=float),
        np.asarray(amplitudes, dtype=float),
        true_phase_info["phase"],
        sfreq,
        cfg,
        evaluation_crop_start_sample=int(true_phase_info.get("evaluation_crop_start_sample", 0)),
        evaluation_crop_stop_sample=true_phase_info.get("evaluation_crop_stop_sample"),
    )
    elapsed = perf_counter() - start
    return {
        "method": "cecHT",
        **result,
        "runtime_ms": elapsed * 1000.0,
        "failed": failed,
        "cecht_calibrated": cfg.cecht_calibrate,
        "cecht_f0_hz": _cecht_center_frequency(cfg),
        "cecht_filter_order": cfg.cecht_filter_order,
    }


def method_registry(method_names: tuple[str, ...] | None = None):
    methods = {
        "offline_hilbert": run_offline_hilbert,
        "cecHT": run_cecht_hilbert,
        "ferster_pll": run_pll,
        "pll": run_classic_pll,
        "ARHT": run_ar_forward_hilbert,
    }
    if method_names is None:
        return list(methods.values())
    unknown = sorted(set(method_names) - set(methods))
    if unknown:
        raise ValueError(f"Unknown phase benchmark methods: {unknown}")
    return [methods[name] for name in method_names]


def true_phase_info(signal: np.ndarray, sfreq: float, cfg: PhaseBenchmarkConfig) -> dict:
    return HilbertPhaseEstimator(sfreq=sfreq, band=cfg.so_band).transform(signal)
