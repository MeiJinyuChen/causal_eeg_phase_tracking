from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_simulation_evaluation.evaluation_metrics import (
    binomial_proportion_ci,
    bootstrap_mean_ci,
    circular_abs_error_summary,
    subject_mean_metric,
)


def test_bootstrap_confidence_interval_contains_mean() -> None:
    rng = np.random.default_rng(3)
    values = rng.standard_normal(50)
    low, high = bootstrap_mean_ci(values, n_boot=500, seed=7)
    assert low <= float(np.mean(values)) <= high
    assert low < high


def test_binomial_confidence_interval_is_symmetric_near_half() -> None:
    low, high = binomial_proportion_ci(hits=10, total=20, alpha=0.05)
    assert low < 0.5 < high
    assert 0.0 <= low <= high <= 1.0


def test_circular_abs_error_summary_returns_bias_plv_sd_and_quantiles() -> None:
    rng = np.random.default_rng(8)
    true = np.zeros(200)
    error = rng.normal(0.1, 0.3, 200)
    summary = circular_abs_error_summary(error)
    assert abs(summary["circular_bias_rad"] - 0.1) < 0.08
    assert 0.0 < summary["resultant_length"] <= 1.0
    assert summary["circular_sd_rad"] > 0.0
    assert summary["mean_abs_wrapped_error_rad"] > 0.0
    assert summary["q95_abs_wrapped_error_rad"] >= summary["median_abs_wrapped_error_rad"]


def test_subject_mean_metric_is_the_mean_of_subject_means() -> None:
    metric_by_subject = {"A": [0.1, 0.3], "B": [0.5, 0.5], "C": [0.9]}
    assert subject_mean_metric(metric_by_subject) == np.mean([0.2, 0.5, 0.9])
