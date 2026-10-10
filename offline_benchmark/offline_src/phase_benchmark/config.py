from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from offline_src.config import PROJECT_ROOT, SO_PHASE_BAND, TARGET_PHASE


@dataclass
class PhaseBenchmarkConfig:
    mode: str = "quick"
    n_subjects: int = 10
    epochs_per_subject: int = 5
    method_names: tuple[str, ...] = ("offline_hilbert", "cecHT", "ferster_pll", "ARHT")
    so_band: tuple[float, float] = SO_PHASE_BAND
    target_phase: float = TARGET_PHASE
    output_dir: Path = PROJECT_ROOT / "outputs" / "phase_method_benchmark"
    past_window_sec: float = 6.0
    prediction_horizon_ms: float = 200.0
    hilbert_padding_ms: float = 500.0
    update_step_ms: float = 50.0
    fieldtrip_buffer_sec: float = 3.0
    evaluation_edge_trim_sec: float = 3.0
    amp_percentile: int = 80
    refractory_sec: float = 1.2

    ec_window_sec: float = 6.0
    cecht_filter_order: int = 1
    cecht_calibrate: bool = True

    # First-order PLL tuning; not used by the current runners, and covered by the
    # same withholding as the classic PLL fields below.
    pll_loop_gain: float | None = None
    pll_center_frequency_hz: float | None = None
    ferster_pll_center_frequency_hz: float = 0.75
    ferster_pll_gain: float = 0.08
    # Equivalent to old ferster_pll_target_phase = 2.8 because internal target = target_phase - offset.
    ferster_pll_phase_offset: float = -2.8
    ferster_pll_input_band: tuple[float, float] = (0.5, 1.25)
    ferster_pll_amp_percentile: int = 80
    ferster_pll_refractory_sec: float = 0.8
    ferster_pll_sample_step: int = 1

    # Classic-PLL tuning parameters.
    # pll.py is the result of a research internship in industry; the canonical
    # classic PLL implementation and its frozen tuning parameters are not
    # included in this distribution for intellectual-property reasons. The
    # fields are kept as ``None`` so the configuration surface stays intact.
    classic_pll_center_frequency_hz: float | None = None
    classic_pll_kvco: float | None = None
    classic_pll_ka: float | None = None
    classic_pll_loop_filter_b: tuple[float, float] | None = None
    classic_pll_loop_filter_a: tuple[float, float] | None = None
    classic_pll_input_band: tuple[float, float] | None = None
    classic_pll_filter_type: str | None = None
    classic_pll_filter_order: int | None = None
    classic_pll_normalize_input: bool | None = None
    classic_pll_normalize_percentile: int | None = None
    classic_pll_phase_offset: float | None = None
    classic_pll_amp_percentile: int | None = None
    classic_pll_refractory_sec: float | None = None
    classic_pll_sample_step: int | None = None
    classic_pll_compute_plv: bool | None = None
    
    opitz_ar_past_window_sec: float = 8.0
    opitz_ar_order: int = 16
    opitz_ar_edge_trim_sec: float = 1.0
    opitz_ar_prediction_horizon_ms: float = 100.0
    opitz_ar_hilbert_padding_ms: float = 500.0
    opitz_ar_update_step_ms: float = 50.0
    opitz_ar_phase_tolerance_rad: float = np.pi / 8
    opitz_ar_amp_percentile: int = 80
    opitz_ar_technical_delay_ms: float = 0.0
    preliminary_trigger_threshold: int = 100

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def success_windows(self) -> dict[str, float]:
        return {
            "pi_6": np.pi / 6,
            "pi_4": np.pi / 4,
            "pi_3": np.pi / 3,
        }


def config_for_mode(mode: str) -> PhaseBenchmarkConfig:
    if mode == "quick":
        return PhaseBenchmarkConfig(mode=mode, n_subjects=10, epochs_per_subject=5)
    if mode == "single-subject":
        return PhaseBenchmarkConfig(
            mode=mode,
            n_subjects=1,
            epochs_per_subject=20,
            method_names=("cecHT", "pll"),
            output_dir=PROJECT_ROOT / "outputs" / "phase_method_benchmark_single_subject",
        )
    if mode == "full-preview":
        return PhaseBenchmarkConfig(mode=mode, n_subjects=20, epochs_per_subject=10)
    raise ValueError(f"Unknown phase benchmark mode: {mode}")
