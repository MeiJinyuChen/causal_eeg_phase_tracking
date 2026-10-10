from __future__ import annotations

import ast
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from importlib.metadata import version
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import pytest
import scipy


ROOT = Path(__file__).resolve().parents[3]
EVALUATION_DIR = ROOT / "causal_phase_tracking" / "causal_simulation_evaluation"
MANIFEST_PATH = (
    EVALUATION_DIR / "manifests" / "cecht_synthetic_cases.json"
)
HISTORICAL_MANIFEST_PATH = (
    EVALUATION_DIR / "manifests" / "cecht_synthetic_cases_100hz.json"
)
SYNTHETIC_PATH = EVALUATION_DIR / "synthetic_truth.py"
GOLDEN_PATH = (
    EVALUATION_DIR / "goldens" / "cecht_synthetic_goldens.json"
)
HISTORICAL_GOLDEN_PATH = (
    EVALUATION_DIR / "goldens" / "cecht_synthetic_goldens_100hz.json"
)
PHASE_PATH = (
    ROOT
    / "causal_phase_tracking"
    / "causal_cecHT"
    / "vendor"
    / "eikeosmers_cecht"
    / "phase.py"
)


def load_synthetic_truth():
    assert SYNTHETIC_PATH.is_file(), "independent synthetic truth module is missing"
    spec = importlib.util.spec_from_file_location("cecht_synthetic_truth", SYNTHETIC_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


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


def float_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def expand_case_descriptors(manifest: dict) -> list[dict]:
    descriptors: list[dict] = []
    phases = manifest["phase_grid_rad"]

    for profile in manifest["profiles"]:
        profile_id = profile["profile_id"]

        for frequency_hz in manifest["signals"]["pure_tone"]["frequencies_hz"]:
            condition_id = f"f_{float_token(frequency_hz)}"
            for phase_index, initial_phase_rad in enumerate(phases):
                descriptors.append(
                    {
                        "profile_id": profile_id,
                        "family": "pure_tone",
                        "condition_id": condition_id,
                        "phase_index": phase_index,
                        "initial_phase_rad": initial_phase_rad,
                        "frequency_hz": frequency_hz,
                        "oracle_f0_hz": frequency_hz,
                        "truth_label": "analytic_driver_phase",
                    }
                )

        for start_hz, end_hz in manifest["signals"]["linear_chirp"]["directions_hz"]:
            condition_id = f"f_{float_token(start_hz)}_to_{float_token(end_hz)}"
            for phase_index, initial_phase_rad in enumerate(phases):
                descriptors.append(
                    {
                        "profile_id": profile_id,
                        "family": "linear_chirp",
                        "condition_id": condition_id,
                        "phase_index": phase_index,
                        "initial_phase_rad": initial_phase_rad,
                        "start_frequency_hz": start_hz,
                        "end_frequency_hz": end_hz,
                        "oracle_f0_hz": end_hz,
                        "truth_label": "integrated_driver_phase",
                    }
                )

        for carrier_hz in manifest["signals"]["amplitude_modulated"]["carrier_frequencies_hz"]:
            condition_id = f"f_{float_token(carrier_hz)}"
            for phase_index, initial_phase_rad in enumerate(phases):
                descriptors.append(
                    {
                        "profile_id": profile_id,
                        "family": "amplitude_modulated",
                        "condition_id": condition_id,
                        "phase_index": phase_index,
                        "initial_phase_rad": initial_phase_rad,
                        "carrier_frequency_hz": carrier_hz,
                        "oracle_f0_hz": carrier_hz,
                        "truth_label": "analytic_driver_phase",
                    }
                )

        harmonic = manifest["signals"]["harmonic"]
        for condition in harmonic["conditions"]:
            for phase_index, initial_phase_rad in enumerate(phases):
                descriptors.append(
                    {
                        "profile_id": profile_id,
                        "family": "harmonic",
                        "condition_id": condition["condition_id"],
                        "phase_index": phase_index,
                        "initial_phase_rad": initial_phase_rad,
                        "fundamental_frequency_hz": harmonic["fundamental_frequency_hz"],
                        "harmonics": condition,
                        "oracle_f0_hz": harmonic["fundamental_frequency_hz"],
                        "truth_label": "fundamental_driver_phase",
                    }
                )

        noise = manifest["noise"]
        for noise_type in noise["types"]:
            family = "white_noise" if noise_type == "white" else "pink_noise"
            for carrier_hz in noise["carrier_frequencies_hz"]:
                for snr_db in noise["snr_db"]:
                    condition_id = (
                        f"f_{float_token(carrier_hz)}_snr_{float_token(snr_db)}"
                    )
                    for phase_index, initial_phase_rad in enumerate(phases):
                        descriptors.append(
                            {
                                "profile_id": profile_id,
                                "family": family,
                                "condition_id": condition_id,
                                "phase_index": phase_index,
                                "initial_phase_rad": initial_phase_rad,
                                "carrier_frequency_hz": carrier_hz,
                                "snr_db": snr_db,
                                "seed": noise["paired_seeds"][phase_index],
                                "oracle_f0_hz": carrier_hz,
                                "truth_label": "unpolluted_driver_phase",
                            }
                        )

        for frequency_hz in manifest["signals"]["cutoff_boundary"]["frequencies_hz"]:
            condition_id = f"f_{float_token(frequency_hz)}"
            for phase_index, initial_phase_rad in enumerate(phases):
                descriptors.append(
                    {
                        "profile_id": profile_id,
                        "family": "cutoff_boundary",
                        "condition_id": condition_id,
                        "phase_index": phase_index,
                        "initial_phase_rad": initial_phase_rad,
                        "frequency_hz": frequency_hz,
                        "oracle_f0_hz": frequency_hz,
                        "truth_label": "analytic_driver_phase",
                    }
                )

    for descriptor in descriptors:
        descriptor["case_id"] = (
            f'{descriptor["profile_id"]}__{descriptor["family"]}__'
            f'{descriptor["condition_id"]}__phase_{descriptor["phase_index"]:02d}'
        )
    return descriptors


def generate_case(
    descriptor: dict, profile: dict, synthetic
) -> tuple[np.ndarray, np.ndarray]:
    common = {
        "sampling_rate_hz": profile["sampling_rate_hz"],
        "sample_count": profile["sample_count"],
        "initial_phase_rad": descriptor["initial_phase_rad"],
    }
    family = descriptor["family"]
    if family in {"pure_tone", "cutoff_boundary"}:
        return synthetic.pure_tone(
            **common,
            frequency_hz=descriptor["frequency_hz"],
            amplitude=1.0,
        )
    if family == "linear_chirp":
        return synthetic.linear_chirp(
            **common,
            start_frequency_hz=descriptor["start_frequency_hz"],
            end_frequency_hz=descriptor["end_frequency_hz"],
        )
    if family == "amplitude_modulated":
        settings = load_manifest()["signals"]["amplitude_modulated"]
        return synthetic.amplitude_modulated_tone(
            **common,
            carrier_frequency_hz=descriptor["carrier_frequency_hz"],
            envelope_frequency_hz=settings["envelope_frequency_hz"],
            modulation_depth=settings["modulation_depth"],
            envelope_phase_rad=settings["envelope_phase_rad"],
        )
    if family == "harmonic":
        condition = descriptor["harmonics"]
        return synthetic.harmonic_signal(
            **common,
            fundamental_frequency_hz=descriptor["fundamental_frequency_hz"],
            h2=condition["h2"],
            h3=condition["h3"],
            psi2_rad=condition["psi2_rad"],
            psi3_rad=condition["psi3_rad"],
        )
    if family in {"white_noise", "pink_noise"}:
        clean, truth = synthetic.pure_tone(
            **common,
            frequency_hz=descriptor["carrier_frequency_hz"],
            amplitude=1.0,
        )
        if family == "white_noise":
            raw_noise = synthetic.white_noise(
                sample_count=profile["sample_count"], seed=descriptor["seed"]
            )
        else:
            raw_noise = synthetic.pink_noise_timmer_koenig(
                sample_count=profile["sample_count"], seed=descriptor["seed"]
            )
        noisy, _ = synthetic.add_noise_at_snr(
            clean_signal=clean,
            raw_noise=raw_noise,
            snr_db=descriptor["snr_db"],
        )
        return noisy, truth
    raise AssertionError(f"unhandled family: {family}")


def estimate_endpoint(
    echt_class,
    *,
    profile: dict,
    signal: np.ndarray,
    branch: str,
    oracle_f0_hz: float,
    fixed_f0_hz: float,
) -> complex:
    calibrate = branch != "uncalibrated"
    if branch == "calibrated_oracle_f0":
        f0 = oracle_f0_hz
    elif branch == "calibrated_fixed_f0":
        f0 = fixed_f0_hz
    else:
        f0 = None

    estimator = echt_class(
        l_freq=profile["band_hz"][0],
        h_freq=profile["band_hz"][1],
        sfreq=profile["sampling_rate_hz"],
        n_fft=profile["n_fft"],
        fft_mode=profile["fft_mode"],
        filt_order=profile["filter_order"],
        filter_type=profile["filter_type"],
        calibrate=calibrate,
        f0=f0,
    )
    estimator.fit(signal)
    assert estimator.n_fft == profile["expected_resolved_n_fft"]
    transformed = np.asarray(estimator.transform(signal)).ravel()
    assert transformed.shape == signal.shape
    return complex(transformed[-1])


@lru_cache(maxsize=1)
def run_full_characterization() -> tuple[dict, ...]:
    manifest = load_manifest()
    synthetic = load_synthetic_truth()
    echt_class = load_echt_class()
    profiles = {profile["profile_id"]: profile for profile in manifest["profiles"]}
    rows: list[dict] = []

    for descriptor in expand_case_descriptors(manifest):
        profile = profiles[descriptor["profile_id"]]
        signal, truth_phase = generate_case(descriptor, profile, synthetic)
        for branch in manifest["branches"]:
            endpoint = estimate_endpoint(
                echt_class,
                profile=profile,
                signal=signal,
                branch=branch,
                oracle_f0_hz=descriptor["oracle_f0_hz"],
                fixed_f0_hz=manifest["fixed_calibration_frequency_hz"],
            )
            error = synthetic.wrap_phase(np.angle(endpoint) - truth_phase[-1])
            rows.append(
                {
                    **descriptor,
                    "branch": branch,
                    "endpoint": endpoint,
                    "truth_endpoint_phase_rad": float(truth_phase[-1]),
                    "error_rad": float(error),
                }
            )
    return tuple(rows)


def make_golden_payload() -> dict:
    manifest = load_manifest()
    synthetic = load_synthetic_truth()
    rows = run_full_characterization()
    selected_case_ids = {
        "P0_exact_1500__pure_tone__f_0p75__phase_00",
        "P1_fast_1501__pure_tone__f_1__phase_00",
    }
    complex_endpoints = {
        f'{row["case_id"]}|{row["branch"]}': {
            "real": row["endpoint"].real,
            "imag": row["endpoint"].imag,
        }
        for row in rows
        if row["case_id"] in selected_case_ids
    }

    aggregate_errors: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    pure_complex_errors: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        aggregate_key = (
            row["profile_id"],
            row["family"],
            row["branch"],
        )
        aggregate_errors[aggregate_key].append(row["error_rad"])
        if row["family"] == "pure_tone":
            ideal = np.exp(1j * row["truth_endpoint_phase_rad"])
            pure_complex_errors[
                (row["profile_id"], row["condition_id"], row["branch"])
            ].append(float(abs(row["endpoint"] - ideal) ** 2))

    aggregate_metrics = {
        "|".join(key): synthetic.circular_metrics(np.asarray(errors))
        for key, errors in sorted(aggregate_errors.items())
    }
    pure_tone_complex_mse = {
        "|".join(key): float(np.mean(errors))
        for key, errors in sorted(pure_complex_errors.items())
    }
    return {
        "identity": {
            "schema_version": "1.0",
            "golden_id": "clas-cecht-synthetic-goldens-v1",
            "upstream_commit": manifest["identity"]["upstream_commit"],
            "case_manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
            "canonical_environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "pytest": version("pytest"),
                "jsonschema": version("jsonschema"),
            },
            "complex_endpoint_rtol": 0.0,
            "complex_endpoint_atol": 1e-12,
        },
        "counts": {
            "signal_cases": len(rows) // len(manifest["branches"]),
            "endpoint_evaluations": len(rows),
            "complex_endpoint_goldens": len(complex_endpoints),
            "aggregate_metric_groups": len(aggregate_metrics),
            "pure_tone_complex_mse_groups": len(pure_tone_complex_mse),
        },
        "complex_endpoints": dict(sorted(complex_endpoints.items())),
        "pure_tone_complex_mse": pure_tone_complex_mse,
        "aggregate_metrics": aggregate_metrics,
        "limitations": {
            "regression_only": True,
            "full_trajectories_frozen": False,
            "performance_thresholds_defined": False,
            "oracle_branch_runtime_eligible": False,
        },
    }


def test_manifest_freezes_the_owner_approved_250hz_rebaseline_matrix() -> None:
    assert MANIFEST_PATH.is_file(), "approved Step 3 case manifest is missing"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["identity"] == {
        "schema_version": "1.0",
        "manifest_id": "clas-cecht-synthetic-cases-v1",
        "status": "step_4_250hz_rebaseline_regression_only",
        "upstream_commit": "e079c3115f202a4272cf64143bf33ddddc247778",
    }
    assert manifest["endpoint"] == {
        "index": -1,
        "sample_number": "N-1",
        "time_s": "(N-1)/fs",
        "error": "wrap(angle(z[-1])-phi_true[N-1])",
        "error_sign": "positive_estimate_leads_truth",
    }
    assert manifest["profiles"] == [
        {
            "profile_id": "P0_exact_1500",
            "sampling_rate_hz": 250.0,
            "sample_count": 1500,
            "band_hz": [0.5, 1.25],
            "filter_order": 2,
            "filter_type": "butter",
            "n_fft": 1500,
            "fft_mode": "exact",
            "expected_resolved_n_fft": 1500,
        },
        {
            "profile_id": "P1_fast_1501",
            "sampling_rate_hz": 250.0,
            "sample_count": 1501,
            "band_hz": [0.5, 1.25],
            "filter_order": 2,
            "filter_type": "butter",
            "n_fft": None,
            "fft_mode": "fast",
            "expected_resolved_n_fft": 1512,
        },
    ]
    assert manifest["branches"] == [
        "calibrated_oracle_f0",
        "calibrated_fixed_f0",
        "uncalibrated",
    ]
    assert manifest["fixed_calibration_frequency_hz"] == 0.875
    assert manifest["phase_grid_rad"] == [
        2 * math.pi * index / 16 for index in range(16)
    ]
    assert manifest["noise"]["base_seed"] == 20260806
    assert manifest["noise"]["paired_seeds"] == list(range(20260806, 20260822))
    assert manifest["noise"]["snr_db"] == [20.0, 10.0, 0.0]
    assert manifest["noise"]["snr_definition"] == (
        "10*log10(mean(clean^2)/mean(scaled_noise^2))"
    )
    assert manifest["scope"]["production_parameter_claim"] is False
    assert manifest["scope"]["streaming_adapter"] is False
    assert manifest["scope"]["trigger_or_edf"] is False


def test_100hz_artifacts_are_explicitly_deprecated_historical_records() -> None:
    historical_manifest = json.loads(
        HISTORICAL_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    historical_golden = json.loads(HISTORICAL_GOLDEN_PATH.read_text(encoding="utf-8"))

    assert historical_manifest["identity"] == {
        "schema_version": "1.0",
        "manifest_id": "clas-cecht-synthetic-cases-100hz-v1",
        "status": "deprecated_100hz_historical_only",
        "upstream_commit": "e079c3115f202a4272cf64143bf33ddddc247778",
    }
    assert historical_manifest["historical_record"] == {
        "deprecated": True,
        "runtime_or_evaluation_use_allowed": False,
        "replacement_manifest": "cecht_synthetic_cases.json",
        "replacement_sampling_rate_hz": 250.0,
    }
    assert historical_manifest["scope"]["sampling_rate_hz"] == 100.0
    assert historical_golden["identity"]["golden_id"] == (
        "clas-cecht-synthetic-goldens-100hz-v1"
    )
    assert historical_golden["identity"]["status"] == (
        "deprecated_100hz_historical_only"
    )
    assert historical_golden["historical_record"] == {
        "deprecated": True,
        "runtime_or_evaluation_use_allowed": False,
        "replacement_golden": "cecht_synthetic_goldens.json",
        "replacement_sampling_rate_hz": 250.0,
    }
    assert historical_golden["identity"]["case_manifest_sha256"] == hashlib.sha256(
        HISTORICAL_MANIFEST_PATH.read_bytes()
    ).hexdigest()


def test_wrap_phase_uses_the_negative_pi_closed_interval() -> None:
    synthetic = load_synthetic_truth()

    actual = synthetic.wrap_phase(np.array([-math.pi, math.pi, 3 * math.pi]))
    np.testing.assert_allclose(actual, [-math.pi, -math.pi, -math.pi], atol=0.0)
    assert synthetic.wrap_phase(math.pi / 2) == pytest.approx(math.pi / 2)


def test_circular_metrics_match_a_hand_checked_two_angle_example() -> None:
    synthetic = load_synthetic_truth()

    metrics = synthetic.circular_metrics(np.array([0.0, math.pi / 2]))

    assert metrics == pytest.approx(
        {
            "circular_bias_rad": math.pi / 4,
            "resultant_length": math.sqrt(0.5),
            "circular_sd_rad": math.sqrt(-2.0 * math.log(math.sqrt(0.5))),
            "mean_absolute_error_rad": math.pi / 4,
            "median_absolute_error_rad": math.pi / 4,
            "q95_absolute_error_rad": 0.475 * math.pi,
        }
    )


def test_pure_tone_returns_analytic_signal_and_driver_phase() -> None:
    synthetic = load_synthetic_truth()

    signal, phase = synthetic.pure_tone(
        sampling_rate_hz=4.0,
        sample_count=3,
        frequency_hz=1.0,
        initial_phase_rad=0.0,
        amplitude=2.0,
    )

    np.testing.assert_allclose(phase, [0.0, math.pi / 2, math.pi], atol=1e-15)
    np.testing.assert_allclose(signal, [2.0, 0.0, -2.0], atol=1e-15)


def test_linear_chirp_integrates_frequency_to_the_last_sample() -> None:
    synthetic = load_synthetic_truth()

    signal, phase = synthetic.linear_chirp(
        sampling_rate_hz=4.0,
        sample_count=5,
        start_frequency_hz=0.5,
        end_frequency_hz=1.5,
        initial_phase_rad=0.25,
    )

    assert phase[0] == pytest.approx(0.25)
    assert phase[-1] == pytest.approx(0.25 + 2.0 * math.pi)
    np.testing.assert_allclose(signal, np.cos(phase), atol=1e-15)


def test_am_tone_keeps_the_approved_envelope_strictly_positive() -> None:
    synthetic = load_synthetic_truth()

    signal, phase = synthetic.amplitude_modulated_tone(
        sampling_rate_hz=4.0,
        sample_count=3,
        carrier_frequency_hz=1.0,
        initial_phase_rad=0.0,
        envelope_frequency_hz=1.0,
        modulation_depth=0.5,
        envelope_phase_rad=0.0,
    )

    np.testing.assert_allclose(phase, [0.0, math.pi / 2, math.pi], atol=1e-15)
    np.testing.assert_allclose(signal, [1.5, 0.0, -0.5], atol=1e-15)


def test_harmonic_truth_remains_the_fundamental_driver_phase() -> None:
    synthetic = load_synthetic_truth()

    signal, phase = synthetic.harmonic_signal(
        sampling_rate_hz=4.0,
        sample_count=3,
        fundamental_frequency_hz=1.0,
        initial_phase_rad=0.0,
        h2=0.5,
        h3=0.25,
        psi2_rad=0.0,
        psi3_rad=0.0,
    )

    np.testing.assert_allclose(phase, [0.0, math.pi / 2, math.pi], atol=1e-15)
    np.testing.assert_allclose(signal, [1.75, -0.5, -0.75], atol=1e-15)


def test_white_noise_is_deterministic_per_explicit_seed() -> None:
    synthetic = load_synthetic_truth()

    first = synthetic.white_noise(sample_count=4096, seed=20260806)
    repeated = synthetic.white_noise(sample_count=4096, seed=20260806)
    different = synthetic.white_noise(sample_count=4096, seed=20260807)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert abs(float(np.mean(first))) < 0.05
    assert 0.95 < float(np.sqrt(np.mean(first**2))) < 1.05


def test_pink_noise_is_deterministic_unit_rms_power_law_noise() -> None:
    synthetic = load_synthetic_truth()

    first = synthetic.pink_noise_timmer_koenig(sample_count=8192, seed=20260806)
    repeated = synthetic.pink_noise_timmer_koenig(
        sample_count=8192, seed=20260806
    )
    different = synthetic.pink_noise_timmer_koenig(
        sample_count=8192, seed=20260807
    )

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert float(np.mean(first)) == pytest.approx(0.0, abs=1e-15)
    assert float(np.sqrt(np.mean(first**2))) == pytest.approx(1.0, abs=1e-15)

    frequencies = np.fft.rfftfreq(first.size)[1:]
    power = np.abs(np.fft.rfft(first)[1:]) ** 2
    slope = float(np.polyfit(np.log(frequencies), np.log(power), 1)[0])
    assert -1.2 < slope < -0.8


def test_add_noise_at_snr_uses_exact_window_rms_scaling() -> None:
    synthetic = load_synthetic_truth()
    clean = np.array([1.0, -1.0, 1.0, -1.0])
    raw_noise = np.array([1.0, 1.0, -1.0, -1.0])

    noisy, scaled_noise = synthetic.add_noise_at_snr(
        clean_signal=clean,
        raw_noise=raw_noise,
        snr_db=10.0 * math.log10(4.0),
    )

    np.testing.assert_allclose(scaled_noise, [0.5, 0.5, -0.5, -0.5], atol=1e-15)
    np.testing.assert_allclose(noisy, [1.5, -0.5, 0.5, -1.5], atol=1e-15)
    achieved_snr_db = 10.0 * math.log10(
        np.mean(clean**2) / np.mean(scaled_noise**2)
    )
    assert achieved_snr_db == pytest.approx(10.0 * math.log10(4.0))


@pytest.mark.parametrize("errors", [np.array([]), np.array([0.0, np.nan])])
def test_circular_metrics_reject_empty_or_nonfinite_errors(errors: np.ndarray) -> None:
    synthetic = load_synthetic_truth()

    with pytest.raises(ValueError):
        synthetic.circular_metrics(errors)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sampling_rate_hz": 0.0, "sample_count": 8, "frequency_hz": 1.0},
        {"sampling_rate_hz": 100.0, "sample_count": 1, "frequency_hz": 1.0},
        {"sampling_rate_hz": 100.0, "sample_count": 8, "frequency_hz": 0.0},
        {"sampling_rate_hz": 100.0, "sample_count": 8, "frequency_hz": 1.0, "amplitude": 0.0},
    ],
)
def test_pure_tone_rejects_degenerate_truth_inputs(kwargs: dict) -> None:
    synthetic = load_synthetic_truth()

    with pytest.raises(ValueError):
        synthetic.pure_tone(initial_phase_rad=0.0, **kwargs)


def test_am_tone_rejects_an_envelope_that_can_reach_zero() -> None:
    synthetic = load_synthetic_truth()

    with pytest.raises(ValueError):
        synthetic.amplitude_modulated_tone(
            sampling_rate_hz=100.0,
            sample_count=600,
            carrier_frequency_hz=0.75,
            initial_phase_rad=0.0,
            envelope_frequency_hz=0.1,
            modulation_depth=1.0,
            envelope_phase_rad=0.0,
        )


def test_other_generators_reject_degenerate_time_or_frequency_inputs() -> None:
    synthetic = load_synthetic_truth()

    with pytest.raises(ValueError):
        synthetic.linear_chirp(
            sampling_rate_hz=100.0,
            sample_count=1,
            start_frequency_hz=0.55,
            end_frequency_hz=1.2,
            initial_phase_rad=0.0,
        )
    with pytest.raises(ValueError):
        synthetic.harmonic_signal(
            sampling_rate_hz=0.0,
            sample_count=600,
            fundamental_frequency_hz=0.75,
            initial_phase_rad=0.0,
            h2=0.1,
            h3=0.05,
            psi2_rad=0.0,
            psi3_rad=0.0,
        )
    with pytest.raises(ValueError):
        synthetic.white_noise(sample_count=1, seed=20260806)
    with pytest.raises(ValueError):
        synthetic.pink_noise_timmer_koenig(sample_count=1, seed=20260806)


@pytest.mark.parametrize(
    "clean,noise,snr_db",
    [
        (np.ones(4), np.ones(3), 10.0),
        (np.zeros(4), np.ones(4), 10.0),
        (np.ones(4), np.zeros(4), 10.0),
        (np.ones(4), np.ones(4), np.nan),
    ],
)
def test_snr_scaling_rejects_invalid_windows(
    clean: np.ndarray, noise: np.ndarray, snr_db: float
) -> None:
    synthetic = load_synthetic_truth()

    with pytest.raises(ValueError):
        synthetic.add_noise_at_snr(
            clean_signal=clean,
            raw_noise=noise,
            snr_db=snr_db,
        )


def test_synthetic_truth_has_no_vendor_or_estimator_dependency() -> None:
    tree = ast.parse(SYNTHETIC_PATH.read_text(encoding="utf-8"))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_names.add(node.module)

    assert all("eikeosmers" not in name.lower() for name in imported_names)
    assert all("cecht" not in name.lower() for name in imported_names)
    assert all("phase" not in name.lower() for name in imported_names)


def test_manifest_expands_to_unique_declared_case_grid() -> None:
    manifest = load_manifest()
    descriptors = expand_case_descriptors(manifest)

    assert len(descriptors) == manifest["case_expansion"]["total_signal_cases"]
    assert len({case["case_id"] for case in descriptors}) == len(descriptors)
    assert Counter(case["profile_id"] for case in descriptors) == {
        "P0_exact_1500": 384,
        "P1_fast_1501": 384,
    }
    assert Counter(case["family"] for case in descriptors) == {
        "pure_tone": 128,
        "linear_chirp": 64,
        "amplitude_modulated": 64,
        "harmonic": 64,
        "white_noise": 192,
        "pink_noise": 192,
        "cutoff_boundary": 64,
    }
    noise_cases = [
        case
        for case in descriptors
        if case["profile_id"] == "P0_exact_1500"
        and case["family"] == "white_noise"
        and case["condition_id"] == "f_0p75_snr_20"
    ]
    assert [case["seed"] for case in noise_cases] == manifest["noise"][
        "paired_seeds"
    ]


def test_full_characterization_matches_the_frozen_golden_manifest() -> None:
    assert GOLDEN_PATH.is_file(), "approved Step 3 golden manifest is missing"
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    actual = make_golden_payload()

    assert actual["identity"] == golden["identity"]
    assert actual["counts"] == golden["counts"]
    assert actual["limitations"] == golden["limitations"]
    assert actual["complex_endpoints"].keys() == golden["complex_endpoints"].keys()
    for key, expected in golden["complex_endpoints"].items():
        np.testing.assert_allclose(
            [actual["complex_endpoints"][key]["real"], actual["complex_endpoints"][key]["imag"]],
            [expected["real"], expected["imag"]],
            rtol=0.0,
            atol=1e-12,
        )
    assert actual["pure_tone_complex_mse"].keys() == golden[
        "pure_tone_complex_mse"
    ].keys()
    for key, expected in golden["pure_tone_complex_mse"].items():
        assert actual["pure_tone_complex_mse"][key] == pytest.approx(
            expected, rel=0.0, abs=1e-12
        )
    assert actual["aggregate_metrics"].keys() == golden["aggregate_metrics"].keys()
    for key, expected_metrics in golden["aggregate_metrics"].items():
        assert actual["aggregate_metrics"][key] == pytest.approx(
            expected_metrics, rel=0.0, abs=1e-12
        )


def test_harness_uses_a_fresh_estimator_and_one_fit_per_branch() -> None:
    echt_class = load_echt_class()
    created = []

    class CountingECHT(echt_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fit_calls = 0
            created.append(self)

        def fit(self, X, y=None):
            self.fit_calls += 1
            return super().fit(X, y)

    manifest = load_manifest()
    profile = manifest["profiles"][0]
    signal, _ = load_synthetic_truth().pure_tone(
        sampling_rate_hz=250.0,
        sample_count=1500,
        frequency_hz=0.75,
        initial_phase_rad=0.0,
    )
    for branch in manifest["branches"]:
        estimate_endpoint(
            CountingECHT,
            profile=profile,
            signal=signal,
            branch=branch,
            oracle_f0_hz=0.75,
            fixed_f0_hz=manifest["fixed_calibration_frequency_hz"],
        )

    assert len(created) == 3
    assert [estimator.fit_calls for estimator in created] == [1, 1, 1]
    assert [estimator.f0 for estimator in created] == [0.75, 0.875, None]


def test_matched_pure_tone_oracle_complex_mse_is_not_worse_than_uncalibrated() -> None:
    manifest = load_manifest()
    pure_mse = make_golden_payload()["pure_tone_complex_mse"]

    for profile in manifest["profiles"]:
        for frequency_hz in manifest["signals"]["pure_tone"]["frequencies_hz"]:
            prefix = f'{profile["profile_id"]}|f_{float_token(frequency_hz)}|'
            oracle = pure_mse[prefix + "calibrated_oracle_f0"]
            uncalibrated = pure_mse[prefix + "uncalibrated"]
            assert oracle <= uncalibrated + 1e-15


def test_endpoint_phase_is_scale_invariant_and_polarity_covariant() -> None:
    manifest = load_manifest()
    settings = manifest["properties"]
    profile = next(
        item for item in manifest["profiles"] if item["profile_id"] == settings["profile_id"]
    )
    synthetic = load_synthetic_truth()
    echt_class = load_echt_class()
    base_signal, truth_phase = synthetic.pure_tone(
        sampling_rate_hz=profile["sampling_rate_hz"],
        sample_count=profile["sample_count"],
        frequency_hz=settings["frequency_hz"],
        initial_phase_rad=settings["initial_phase_rad"],
    )

    for branch in manifest["branches"]:
        endpoints = {
            scale: estimate_endpoint(
                echt_class,
                profile=profile,
                signal=scale * base_signal,
                branch=branch,
                oracle_f0_hz=settings["frequency_hz"],
                fixed_f0_hz=manifest["fixed_calibration_frequency_hz"],
            )
            for scale in settings["positive_scales"]
        }
        reference = endpoints[1.0]
        reference_error = synthetic.wrap_phase(np.angle(reference) - truth_phase[-1])
        for scale, endpoint in endpoints.items():
            np.testing.assert_allclose(
                endpoint,
                scale * reference,
                rtol=1e-14,
                atol=1e-14,
            )
            error = synthetic.wrap_phase(np.angle(endpoint) - truth_phase[-1])
            assert error == pytest.approx(reference_error, abs=1e-14)

        inverted = estimate_endpoint(
            echt_class,
            profile=profile,
            signal=-base_signal,
            branch=branch,
            oracle_f0_hz=settings["frequency_hz"],
            fixed_f0_hz=manifest["fixed_calibration_frequency_hz"],
        )
        np.testing.assert_allclose(inverted, -reference, rtol=1e-14, atol=1e-14)
        inverted_error = synthetic.wrap_phase(
            np.angle(inverted) - (truth_phase[-1] + math.pi)
        )
        assert inverted_error == pytest.approx(reference_error, abs=1e-14)
