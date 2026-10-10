"""Placeholder for the canonical classic PLL implementation.

This package is the result of a research internship in industry. The original implementation is withheld from this distribution for intellectual-property reasons.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The canonical classic PLL implementation is withheld from this distribution.
CLASSIC_PLL_IS_WITHHELD = True
CLASSIC_PLL_WITHHELD_REASON = (
    "pll.py is the result of a research internship in industry; the canonical "
    "classic PLL implementation and its frozen tuning parameters are not "
    "included in this distribution for intellectual-property reasons "
    "(reference MATLAB port omitted)"
)
_NOT_INCLUDED = CLASSIC_PLL_WITHHELD_REASON


@dataclass(frozen=True)
class ClassicPLLSettings:
    fc: float
    kvco: float


def classic_pll_settings(filter_type: str) -> ClassicPLLSettings:
    raise NotImplementedError(_NOT_INCLUDED)


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * phase))


def run_classic_pll_core(
    signal: np.ndarray,
    sfreq: float,
    fc: float,
    kvco: float,
    ka: float = 1.0,
    loop_b: tuple[float, float] | None = None,
    loop_a: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    raise NotImplementedError(_NOT_INCLUDED)


def phase_locking_value(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.signal import hilbert
    phase_x = np.angle(hilbert(np.asarray(x, dtype=float)))
    phase_y = np.angle(hilbert(np.asarray(y, dtype=float)))
    return float(np.abs(np.mean(np.exp(1j * (phase_y - phase_x)))))
