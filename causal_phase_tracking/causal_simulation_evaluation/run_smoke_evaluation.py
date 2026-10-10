"""Sleep-EDF smoke run: causal adapter + cecHT + trigger vs offline reference."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from causal_phase_tracking.causal_cecHT.pseudo_trigger import PseudoTriggerConfig
from causal_phase_tracking.causal_cecHT.streaming_core import CecHTCoreConfig

from .edf_replay import PSG_CHANNEL, load_fpz_cz_100hz, qa_fpz_cz_signal, run_adapted_stream
from .offline_reference import (
    circular_mean_bias,
    circular_sd_from_resultant,
    hilbert_phase_rad,
    noncausal_reference_resample,
    zero_phase_so_bandpass,
)
from .rate_adapter import RationalRateAdapterConfig
from .evaluation_pipeline import run_causal_pipeline


ROOT = Path(__file__).resolve().parents[2]
PSG = ROOT / "data" / "sleep_edf" / "default" / "SC4001E0-PSG.edf"
LOAD_START_S = 34230.0
LOAD_END_S = 34530.0
CENTRAL_START_S = 34290.0
CENTRAL_END_S = 34470.0
ADAPTER_GROUP_DELAY = 8


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def main() -> None:
    t0 = time.perf_counter()
    signal = load_fpz_cz_100hz(PSG)
    qa = qa_fpz_cz_signal(signal, sampling_rate_hz=100.0)

    reference_250 = noncausal_reference_resample(signal, input_rate_hz=100.0, output_rate_hz=250.0)
    reference_band = zero_phase_so_bandpass(reference_250, 250.0, l_freq=0.5, h_freq=1.25)
    reference_phase = hilbert_phase_rad(reference_band)

    load_signal = signal[int(LOAD_START_S * 100) : int(LOAD_END_S * 100)]
    adapter_config = RationalRateAdapterConfig(
        input_rate_hz=100.0,
        output_rate_hz=250.0,
        upsample=5,
        downsample=2,
        kernel_rate_hz=500.0,
        cutoff_hz=40.0,
        stopband_hz=50.0,
        kernel_taps=33,
        max_input_samples=8192,
        config_version="smoke-v1",
    )
    stream = run_adapted_stream(load_signal, adapter_config, [10, 40, 100, 17, 250, 3])

    core_config = CecHTCoreConfig(
        sampling_rate_hz=250.0,
        input_dtype=np.dtype(np.float64),
        numeric_unit_code=1,
        channel_code=7,
        polarity_code=1,
        window_samples=1500,
        l_freq=0.5,
        h_freq=1.25,
        filter_order=2,
        filter_type="butter",
        n_fft=1500,
        fft_mode="exact",
        calibrate=False,
        f0=None,
        parameter_version="smoke-v1",
        config_version="smoke-v1",
        max_block_samples=1024,
    )
    trigger_config = PseudoTriggerConfig(
        sampling_rate_hz=250.0,
        target_phase_rad=0.0,
        refractory_s=0.5,
        parameter_version="smoke-v1",
        config_version="smoke-v1",
        max_block_samples=1024,
    )
    result = run_causal_pipeline(
        stream.values,
        stream.valid,
        stream.output_index,
        core_config,
        trigger_config,
        [10, 40, 100, 17, 250, 3],
    )

    aligned_source_time = LOAD_START_S + (
        result.output_index.astype(np.float64) - ADAPTER_GROUP_DELAY
    ) / 250.0
    central = (
        (aligned_source_time >= CENTRAL_START_S)
        & (aligned_source_time <= CENTRAL_END_S)
        & result.ready
    )
    ref_indices = np.clip(
        np.rint((aligned_source_time[central]) * 250.0).astype(np.int64),
        0,
        reference_phase.size - 1,
    )
    estimator_error = wrap(result.phase_rad[central] - reference_phase[ref_indices])
    resultant = float(np.abs(np.mean(np.exp(1j * estimator_error))))

    event_slots = result.event
    event_ref_time = LOAD_START_S + (
        result.crossing_time_s[event_slots] - ADAPTER_GROUP_DELAY / 250.0
    )
    event_ref_idx = np.clip(
        np.rint(event_ref_time * 250.0).astype(np.int64), 0, reference_phase.size - 1
    )
    trigger_error = (
        wrap(reference_phase[event_ref_idx])
        if event_ref_idx.size
        else np.empty(0, dtype=np.float64)
    )

    metrics = {
        "qa": qa,
        "load_start_s": LOAD_START_S,
        "load_end_s": LOAD_END_S,
        "central_start_s": CENTRAL_START_S,
        "central_end_s": CENTRAL_END_S,
        "reference_definition": {
            "resample": "scipy.signal.resample_poly 5/2 non-causal",
            "bandpass": "0.5-1.25 Hz Butterworth order 2 sosfiltfilt zero-phase",
            "phase": "scipy.signal.hilbert angle",
        },
        "estimator_ready_central_samples": int(central.sum()),
        "estimator_circular_bias_rad": circular_mean_bias(estimator_error),
        "estimator_resultant_length": resultant,
        "estimator_circular_sd_rad": circular_sd_from_resultant(max(resultant, 1e-12)),
        "estimator_mean_abs_wrapped_error_rad": float(np.mean(np.abs(estimator_error))),
        "estimator_median_abs_wrapped_error_rad": float(np.median(np.abs(estimator_error))),
        "estimator_hit_rate_within_30deg": float(np.mean(np.abs(estimator_error) <= math.pi / 6)),
        "pseudo_event_count": int(event_slots.sum()),
        "pseudo_event_circular_bias_rad": (
            circular_mean_bias(trigger_error) if trigger_error.size else None
        ),
        "pseudo_event_mean_abs_wrapped_error_rad": (
            float(np.mean(np.abs(trigger_error))) if trigger_error.size else None
        ),
        "wall_clock_seconds": time.perf_counter() - t0,
    }

    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "smoke_report.md"
    metrics_path = artifact_dir / "smoke_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Sleep-EDF smoke report",
        "",
        f"- channel: {PSG_CHANNEL}",
        f"- load: {LOAD_START_S}-{LOAD_END_S} s",
        f"- central: {CENTRAL_START_S}-{CENTRAL_END_S} s",
        f"- QA: {metrics['qa']}",
        f"- ready central endpoints: {metrics['estimator_ready_central_samples']}",
        f"- circular bias: {metrics['estimator_circular_bias_rad']:.6f} rad",
        f"- resultant length: {metrics['estimator_resultant_length']:.6f}",
        f"- circular SD: {metrics['estimator_circular_sd_rad']:.6f} rad",
        f"- mean abs wrapped error: {metrics['estimator_mean_abs_wrapped_error_rad']:.6f} rad",
        f"- within 30 deg: {metrics['estimator_hit_rate_within_30deg']:.6f}",
        f"- pseudo events: {metrics['pseudo_event_count']}",
        f"- wall clock: {metrics['wall_clock_seconds']:.2f} s",
        "",
        "This is a development smoke result only. It is not a tuning or generalization claim.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", metrics_path)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
