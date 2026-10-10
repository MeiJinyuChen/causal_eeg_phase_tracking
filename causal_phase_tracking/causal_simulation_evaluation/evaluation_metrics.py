"""Circular statistics, binomial intervals and subject-level summaries."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def wrap(angle: float | np.ndarray) -> np.ndarray:
    values = np.asarray(angle, dtype=np.float64)
    return (values + math.pi) % (2.0 * math.pi) - math.pi


def binomial_proportion_ci(
    hits: int, total: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Wilson score interval, a small-sample stable binomial CI."""

    if total <= 0:
        raise ValueError("total must be positive")
    z = 1.96  # two-sided 95% normal quantile; alpha is fixed by convention here
    p = hits / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 20260911,
    alpha: float = 0.05,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("values must not be empty")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        boot[idx] = np.mean(rng.choice(data, size=data.size, replace=True))
    return float(np.quantile(boot, alpha / 2.0)), float(np.quantile(boot, 1.0 - alpha / 2.0))


def circular_abs_error_summary(error_rad: np.ndarray) -> dict[str, float]:
    wrapped = wrap(error_rad)
    resultant_complex = np.mean(np.exp(1j * wrapped))
    r = float(abs(resultant_complex))
    abs_error = np.abs(wrapped)
    return {
        "circular_bias_rad": float(np.angle(resultant_complex)),
        "resultant_length": r,
        "circular_sd_rad": math.sqrt(-2.0 * math.log(max(r, 1e-12))),
        "mean_abs_wrapped_error_rad": float(np.mean(abs_error)),
        "median_abs_wrapped_error_rad": float(np.median(abs_error)),
        "q95_abs_wrapped_error_rad": float(np.quantile(abs_error, 0.95)),
    }


def subject_mean_metric(metric_by_subject: Mapping[str, Sequence[float]]) -> float:
    subject_means = [float(np.mean(values)) for values in metric_by_subject.values() if values]
    if not subject_means:
        raise ValueError("metric_by_subject must contain at least one non-empty subject")
    return float(np.mean(subject_means))
