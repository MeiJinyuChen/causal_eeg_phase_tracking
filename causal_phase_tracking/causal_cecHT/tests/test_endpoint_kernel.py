from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from causal_phase_tracking.causal_cecHT import streaming_core
from causal_phase_tracking.causal_cecHT.streaming_core import (
    CausalCecHTCore,
    CecHTBlockContext,
    CecHTCoreConfig,
    CecHTOutputBuffers,
    CecHTProcessStatus,
    CecHTReasonCode,
    CecHTVerifiedField,
)
from causal_phase_tracking.causal_cecHT.vendor.eikeosmers_cecht.phase import (
    ECHT as PinnedECHT,
)
from causal_phase_tracking.causal_simulation_evaluation import synthetic_truth


ENDPOINT_ATOL = 2e-12
FLOAT32_ENDPOINT_ATOL = 2e-7


def make_config(
    *,
    window_samples: int,
    n_fft: int | None,
    fft_mode: str,
    calibrate: bool,
    input_dtype: np.dtype = np.dtype(np.float64),
) -> CecHTCoreConfig:
    return CecHTCoreConfig(
        sampling_rate_hz=250.0,
        input_dtype=input_dtype,
        numeric_unit_code=1,
        channel_code=1,
        polarity_code=1,
        window_samples=window_samples,
        l_freq=0.5,
        h_freq=1.25,
        filter_order=2,
        filter_type="butter",
        n_fft=n_fft,
        fft_mode=fft_mode,
        calibrate=calibrate,
        f0=0.875 if calibrate else None,
        parameter_version="step4-test-p0p1",
        config_version="step4-test-v1",
        max_block_samples=2048,
    )


def make_fitted_oracle(config: CecHTCoreConfig) -> PinnedECHT:
    oracle = PinnedECHT(
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        sfreq=config.sampling_rate_hz,
        n_fft=config.n_fft,
        fft_mode=config.fft_mode,
        filt_order=config.filter_order,
        filter_type=config.filter_type,
        calibrate=config.calibrate,
        f0=config.f0,
    )
    oracle.fit(np.zeros(config.window_samples, dtype=config.input_dtype))
    return oracle


@pytest.mark.parametrize(
    ("window_samples", "n_fft", "fft_mode", "expected_n_fft"),
    [
        pytest.param(1500, 1500, "exact", 1500, id="P0-exact"),
        pytest.param(1501, None, "fast", 1512, id="P1-fast"),
    ],
)
@pytest.mark.parametrize("calibrate", [False, True], ids=["raw", "calibrated"])
def test_constructor_kernel_matches_pinned_endpoint_for_seeded_windows(
    window_samples: int,
    n_fft: int | None,
    fft_mode: str,
    expected_n_fft: int,
    calibrate: bool,
) -> None:
    config = make_config(
            window_samples=window_samples,
            n_fft=n_fft,
            fft_mode=fft_mode,
            calibrate=calibrate,
    )
    core = CausalCecHTCore(config)
    oracle = make_fitted_oracle(config)
    rng = np.random.default_rng(20260807 + window_samples + int(calibrate))

    assert core.resolved_n_fft == expected_n_fft
    assert core.endpoint_kernel.shape == (window_samples,)
    for _ in range(4):
        window = rng.standard_normal(window_samples)
        expected = np.asarray(oracle.transform(window)).ravel()[-1]
        actual = np.dot(core.endpoint_kernel, window)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ENDPOINT_ATOL)


@pytest.mark.parametrize(
    ("window_samples", "n_fft", "fft_mode"),
    [
        pytest.param(1500, 1500, "exact", id="P0-exact"),
        pytest.param(1501, None, "fast", id="P1-fast"),
    ],
)
@pytest.mark.parametrize("calibrate", [False, True], ids=["raw", "calibrated"])
def test_float32_streaming_endpoint_has_an_explicit_pinned_oracle_tolerance(
    window_samples: int,
    n_fft: int | None,
    fft_mode: str,
    calibrate: bool,
) -> None:
    config = make_config(
        window_samples=window_samples,
        n_fft=n_fft,
        fft_mode=fft_mode,
        calibrate=calibrate,
        input_dtype=np.dtype(np.float32),
    )
    core = CausalCecHTCore(config)
    oracle = make_fitted_oracle(config)
    samples = np.random.default_rng(
        20260808 + window_samples + int(calibrate)
    ).standard_normal(window_samples).astype(np.float32)
    outputs = CecHTOutputBuffers.allocate(window_samples)
    context = CecHTBlockContext(
        stream_epoch=1,
        first_relative_index=0,
        declared_sampling_rate_hz=250.0,
        unit_code=1,
        channel_code=1,
        polarity_code=1,
        verified_fields_mask=CecHTVerifiedField.ALL,
    )

    status = core.process_block_into(samples, context, outputs)

    expected = np.asarray(oracle.transform(samples)).ravel()[-1]
    actual = outputs.amplitude[-1] * np.exp(1j * outputs.phase_rad[-1])
    assert status is CecHTProcessStatus.OK
    assert outputs.ready[-1]
    assert outputs.valid[-1]
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=FLOAT32_ENDPOINT_ATOL,
    )


@pytest.mark.parametrize("calibrate", [False, True], ids=["raw", "calibrated"])
def test_kernel_matches_impulse_tone_scale_and_polarity_cases(
    calibrate: bool,
) -> None:
    config = make_config(
        window_samples=1500,
        n_fft=1500,
        fft_mode="exact",
        calibrate=calibrate,
    )
    core = CausalCecHTCore(config)
    oracle = make_fitted_oracle(config)
    t = np.arange(config.window_samples, dtype=np.float64) / 250.0
    tone = np.cos(2.0 * np.pi * 0.75 * t + np.pi / 8.0)

    windows = []
    for position in (0, config.window_samples // 2, config.window_samples - 1):
        impulse = np.zeros(config.window_samples, dtype=np.float64)
        impulse[position] = 1.0
        windows.append(impulse)
    windows.extend((tone, 0.1 * tone, 10.0 * tone, -tone))

    for window in windows:
        expected = np.asarray(oracle.transform(window)).ravel()[-1]
        actual = np.dot(core.endpoint_kernel, window)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ENDPOINT_ATOL)

    positive = np.dot(core.endpoint_kernel, tone)
    np.testing.assert_allclose(
        np.angle(np.dot(core.endpoint_kernel, 10.0 * tone)),
        np.angle(positive),
        rtol=0.0,
        atol=ENDPOINT_ATOL,
    )
    polarity_shift = (
        np.angle(np.dot(core.endpoint_kernel, -tone)) - np.angle(positive) + math.pi
    ) % (2.0 * math.pi) - math.pi
    assert abs(abs(polarity_shift) - math.pi) <= ENDPOINT_ATOL


def test_endpoint_kernel_is_read_only() -> None:
    core = CausalCecHTCore(
        make_config(
            window_samples=1500,
            n_fft=1500,
            fft_mode="exact",
            calibrate=False,
        )
    )

    expected = core.endpoint_kernel.copy()
    exposed = core.endpoint_kernel
    assert exposed.flags.writeable is False
    with pytest.raises(ValueError):
        exposed[0] = 0.0
    exposed.setflags(write=True)
    exposed[0] = 0.0
    np.testing.assert_array_equal(core.endpoint_kernel, expected)


def test_constructor_rejects_fft_shorter_than_the_declared_window() -> None:
    with pytest.raises(ValueError, match="n_fft.*window_samples"):
        CausalCecHTCore(
            make_config(
                window_samples=1501,
                n_fft=1500,
                fft_mode="exact",
                calibrate=False,
            )
        )


def test_reset_preserves_kernel_without_refitting(monkeypatch: pytest.MonkeyPatch) -> None:
    fit_calls = 0
    original_fit = streaming_core.ECHT.fit

    def counting_fit(self, samples, y=None):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(self, samples, y)

    monkeypatch.setattr(streaming_core.ECHT, "fit", counting_fit)
    core = CausalCecHTCore(
        make_config(
            window_samples=1500,
            n_fft=1500,
            fft_mode="exact",
            calibrate=True,
        )
    )
    kernel_before = core.endpoint_kernel.copy()

    core.reset(CecHTReasonCode.EXPLICIT_RESET)
    core.reset(CecHTReasonCode.EXPLICIT_RESET)

    assert fit_calls == 1
    np.testing.assert_array_equal(core.endpoint_kernel, kernel_before)


@pytest.mark.parametrize(
    ("window_samples", "n_fft", "fft_mode"),
    [
        pytest.param(1500, 1500, "exact", id="P0-exact"),
        pytest.param(1501, None, "fast", id="P1-fast"),
    ],
)
@pytest.mark.parametrize("calibrate", [False, True], ids=["raw", "calibrated"])
def test_kernel_matches_pinned_endpoint_for_every_step3_signal_family(
    window_samples: int,
    n_fft: int | None,
    fft_mode: str,
    calibrate: bool,
) -> None:
    config = make_config(
        window_samples=window_samples,
        n_fft=n_fft,
        fft_mode=fft_mode,
        calibrate=calibrate,
    )
    core = CausalCecHTCore(config)
    oracle = make_fitted_oracle(config)
    common = {
        "sampling_rate_hz": 250.0,
        "sample_count": window_samples,
        "initial_phase_rad": math.pi / 8.0,
    }
    pure, _ = synthetic_truth.pure_tone(frequency_hz=0.75, **common)
    chirp, _ = synthetic_truth.linear_chirp(
        start_frequency_hz=0.55, end_frequency_hz=1.2, **common
    )
    amplitude_modulated, _ = synthetic_truth.amplitude_modulated_tone(
        carrier_frequency_hz=1.0,
        envelope_frequency_hz=0.1,
        modulation_depth=0.5,
        envelope_phase_rad=0.0,
        **common,
    )
    harmonic, _ = synthetic_truth.harmonic_signal(
        fundamental_frequency_hz=0.75,
        h2=0.25,
        h3=0.1,
        psi2_rad=math.pi / 2.0,
        psi3_rad=-math.pi / 3.0,
        **common,
    )
    cutoff_low, _ = synthetic_truth.pure_tone(frequency_hz=0.5, **common)
    cutoff_high, _ = synthetic_truth.pure_tone(frequency_hz=1.25, **common)
    white, _ = synthetic_truth.add_noise_at_snr(
        clean_signal=pure,
        raw_noise=synthetic_truth.white_noise(
            sample_count=window_samples, seed=20260806
        ),
        snr_db=10.0,
    )
    pink, _ = synthetic_truth.add_noise_at_snr(
        clean_signal=pure,
        raw_noise=synthetic_truth.pink_noise_timmer_koenig(
            sample_count=window_samples, seed=20260806
        ),
        snr_db=10.0,
    )

    for window in (
        pure,
        chirp,
        amplitude_modulated,
        harmonic,
        white,
        pink,
        cutoff_low,
        cutoff_high,
    ):
        expected = np.asarray(oracle.transform(window)).ravel()[-1]
        actual = np.dot(core.endpoint_kernel, window)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=ENDPOINT_ATOL)
