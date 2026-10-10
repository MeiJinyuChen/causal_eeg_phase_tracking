from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
VENDOR_DIR = (
    ROOT
    / "causal_phase_tracking"
    / "causal_cecHT"
    / "vendor"
    / "eikeosmers_cecht"
)
PHASE_PATH = VENDOR_DIR / "phase.py"
PROVENANCE_PATH = VENDOR_DIR / "UPSTREAM_PROVENANCE.json"

PARAMETERS = {
    "sampling_rate_hz": 373,
    "duration_s": 1,
    "sample_count": 373,
    "test_frequency_hz": 2.5,
    "initial_phase_count": 8,
    "initial_phases_rad": [float(2 * math.pi * n / 8) for n in range(8)],
    "band_hz": [1.875, 3.125],
    "filter_order": 2,
    "filter_type": "butter",
    "n_fft": 373,
    "fft_mode": "exact",
}
GOLDEN_ATOL = 1e-12
MIN_CALIBRATED_ENDPOINT_ADVANTAGE_RAD = 0.01
EXPECTED_BRANCHES = {
    "uncalibrated": {
        "golden_endpoint_real": -1.0580374193202473,
        "golden_endpoint_imag": 0.17611524876424103,
        "mean_abs_endpoint_error_rad": 0.13135600617913523,
        "mean_abs_rejected_index_error_rad": 0.08802897169510443,
    },
    "calibrated": {
        "golden_endpoint_real": -0.9959222375666699,
        "golden_endpoint_imag": 0.0334620057244191,
        "mean_abs_endpoint_error_rad": 0.005982482493083197,
        "mean_abs_rejected_index_error_rad": 0.043327032711055646,
    },
}


def wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def load_echt_class():
    spec = importlib.util.spec_from_file_location("pinned_eikeosmers_cecht", PHASE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module.ECHT


def run_characterization() -> dict[str, dict[str, np.ndarray | complex]]:
    echt_class = load_echt_class()
    sfreq = PARAMETERS["sampling_rate_hz"]
    sample_count = PARAMETERS["sample_count"]
    frequency = PARAMETERS["test_frequency_hz"]
    t = np.arange(sample_count, dtype=float) / sfreq

    results: dict[str, dict[str, np.ndarray | complex]] = {}
    for branch, calibrate in (("uncalibrated", False), ("calibrated", True)):
        endpoint_errors = []
        first_index_errors = []
        golden_endpoint = None

        for initial_phase in PARAMETERS["initial_phases_rad"]:
            signal = np.cos(2 * np.pi * frequency * t + initial_phase)
            estimator = echt_class(
                l_freq=PARAMETERS["band_hz"][0],
                h_freq=PARAMETERS["band_hz"][1],
                sfreq=sfreq,
                n_fft=PARAMETERS["n_fft"],
                fft_mode=PARAMETERS["fft_mode"],
                filt_order=PARAMETERS["filter_order"],
                filter_type=PARAMETERS["filter_type"],
                calibrate=calibrate,
                f0=frequency if calibrate else None,
            )
            transformed = np.asarray(estimator.fit_transform(signal)).ravel()
            assert transformed.shape == (sample_count,)
            assert np.all(np.isfinite(transformed.real))
            assert np.all(np.isfinite(transformed.imag))

            true_endpoint_phase = wrap(
                2 * np.pi * frequency * t[-1] + initial_phase
            )
            endpoint_errors.append(wrap(np.angle(transformed[-1]) - true_endpoint_phase))
            first_index_errors.append(
                wrap(np.angle(transformed[0]) - true_endpoint_phase)
            )
            if initial_phase == 0.0:
                golden_endpoint = complex(transformed[-1])

        assert golden_endpoint is not None
        results[branch] = {
            "endpoint_errors": np.asarray(endpoint_errors),
            "first_index_errors": np.asarray(first_index_errors),
            "golden_endpoint": golden_endpoint,
        }

    return results


def test_characterization_manifest_freezes_the_approved_setup() -> None:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    characterization = provenance["endpoint_characterization"]

    assert characterization["parameters"] == PARAMETERS
    assert characterization["accepted_endpoint_index"] == -1
    assert characterization["rejected_readme_index"] == 0
    assert characterization["golden_atol"] == GOLDEN_ATOL
    assert characterization["minimum_calibrated_endpoint_advantage_rad"] == (
        MIN_CALIBRATED_ENDPOINT_ADVANTAGE_RAD
    )
    assert characterization["branches"] == EXPECTED_BRANCHES


def test_calibrated_latest_sample_is_better_aligned_than_readme_index() -> None:
    results = run_characterization()

    calibrated_endpoint_mae = float(
        np.mean(np.abs(results["calibrated"]["endpoint_errors"]))
    )
    calibrated_first_index_mae = float(
        np.mean(np.abs(results["calibrated"]["first_index_errors"]))
    )
    assert (
        calibrated_endpoint_mae + MIN_CALIBRATED_ENDPOINT_ADVANTAGE_RAD
        < calibrated_first_index_mae
    )

    uncalibrated_mae = float(
        np.mean(np.abs(results["uncalibrated"]["endpoint_errors"]))
    )
    assert calibrated_endpoint_mae < uncalibrated_mae


def test_fixed_complex_endpoint_goldens_detect_numeric_drift() -> None:
    results = run_characterization()

    for branch in ("uncalibrated", "calibrated"):
        actual = results[branch]["golden_endpoint"]
        frozen = EXPECTED_BRANCHES[branch]
        expected = complex(
            frozen["golden_endpoint_real"], frozen["golden_endpoint_imag"]
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=GOLDEN_ATOL)

        endpoint_mae = float(np.mean(np.abs(results[branch]["endpoint_errors"])))
        first_index_mae = float(
            np.mean(np.abs(results[branch]["first_index_errors"]))
        )
        assert math.isclose(
            endpoint_mae,
            frozen["mean_abs_endpoint_error_rad"],
            rel_tol=0.0,
            abs_tol=GOLDEN_ATOL,
        )
        assert math.isclose(
            first_index_mae,
            frozen["mean_abs_rejected_index_error_rad"],
            rel_tol=0.0,
            abs_tol=GOLDEN_ATOL,
        )
