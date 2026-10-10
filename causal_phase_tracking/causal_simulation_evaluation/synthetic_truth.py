from __future__ import annotations

import numpy as np


def _validate_sample_count(sample_count: int) -> None:
    if not isinstance(sample_count, (int, np.integer)) or sample_count < 2:
        raise ValueError("sample_count must be an integer >= 2")


def _validate_finite(**values: float) -> None:
    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")


def _time_axis(sampling_rate_hz: float, sample_count: int) -> np.ndarray:
    _validate_sample_count(sample_count)
    _validate_finite(sampling_rate_hz=sampling_rate_hz)
    if sampling_rate_hz <= 0.0:
        raise ValueError("sampling_rate_hz must be positive")
    return np.arange(sample_count, dtype=float) / sampling_rate_hz


def _validate_positive_frequencies(**frequencies_hz: float) -> None:
    _validate_finite(**frequencies_hz)
    if any(value <= 0.0 for value in frequencies_hz.values()):
        raise ValueError("frequencies must be positive")


def wrap_phase(angle: np.ndarray | float) -> np.ndarray | float:
    wrapped = (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi
    if wrapped.ndim == 0:
        return float(wrapped)
    return wrapped


def circular_metrics(errors_rad: np.ndarray) -> dict[str, float]:
    errors = np.asarray(errors_rad, dtype=float).ravel()
    if errors.size == 0 or not np.all(np.isfinite(errors)):
        raise ValueError("errors_rad must contain finite values")
    mean_vector = np.mean(np.exp(1j * errors))
    resultant_length = min(1.0, float(np.abs(mean_vector)))
    absolute_errors = np.abs(errors)
    return {
        "circular_bias_rad": float(np.angle(mean_vector)),
        "resultant_length": resultant_length,
        "circular_sd_rad": float(np.sqrt(-2.0 * np.log(resultant_length))),
        "mean_absolute_error_rad": float(np.mean(absolute_errors)),
        "median_absolute_error_rad": float(np.median(absolute_errors)),
        "q95_absolute_error_rad": float(np.quantile(absolute_errors, 0.95)),
    }


def pure_tone(
    *,
    sampling_rate_hz: float,
    sample_count: int,
    frequency_hz: float,
    initial_phase_rad: float,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    time_s = _time_axis(sampling_rate_hz, sample_count)
    _validate_positive_frequencies(frequency_hz=frequency_hz)
    _validate_finite(initial_phase_rad=initial_phase_rad, amplitude=amplitude)
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")
    phase_rad = initial_phase_rad + 2.0 * np.pi * frequency_hz * time_s
    return amplitude * np.cos(phase_rad), phase_rad


def linear_chirp(
    *,
    sampling_rate_hz: float,
    sample_count: int,
    start_frequency_hz: float,
    end_frequency_hz: float,
    initial_phase_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_s = _time_axis(sampling_rate_hz, sample_count)
    _validate_positive_frequencies(
        start_frequency_hz=start_frequency_hz,
        end_frequency_hz=end_frequency_hz,
    )
    _validate_finite(initial_phase_rad=initial_phase_rad)
    duration_s = (sample_count - 1) / sampling_rate_hz
    slope_hz_per_s = (end_frequency_hz - start_frequency_hz) / duration_s
    phase_rad = initial_phase_rad + 2.0 * np.pi * (
        start_frequency_hz * time_s + 0.5 * slope_hz_per_s * time_s**2
    )
    return np.cos(phase_rad), phase_rad


def amplitude_modulated_tone(
    *,
    sampling_rate_hz: float,
    sample_count: int,
    carrier_frequency_hz: float,
    initial_phase_rad: float,
    envelope_frequency_hz: float,
    modulation_depth: float,
    envelope_phase_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_s = _time_axis(sampling_rate_hz, sample_count)
    _validate_positive_frequencies(
        carrier_frequency_hz=carrier_frequency_hz,
        envelope_frequency_hz=envelope_frequency_hz,
    )
    _validate_finite(
        initial_phase_rad=initial_phase_rad,
        modulation_depth=modulation_depth,
        envelope_phase_rad=envelope_phase_rad,
    )
    if abs(modulation_depth) >= 1.0:
        raise ValueError("abs(modulation_depth) must be < 1 for a positive envelope")
    phase_rad = initial_phase_rad + 2.0 * np.pi * carrier_frequency_hz * time_s
    envelope = 1.0 + modulation_depth * np.cos(
        2.0 * np.pi * envelope_frequency_hz * time_s + envelope_phase_rad
    )
    return envelope * np.cos(phase_rad), phase_rad


def harmonic_signal(
    *,
    sampling_rate_hz: float,
    sample_count: int,
    fundamental_frequency_hz: float,
    initial_phase_rad: float,
    h2: float,
    h3: float,
    psi2_rad: float,
    psi3_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_s = _time_axis(sampling_rate_hz, sample_count)
    _validate_positive_frequencies(
        fundamental_frequency_hz=fundamental_frequency_hz
    )
    _validate_finite(
        initial_phase_rad=initial_phase_rad,
        h2=h2,
        h3=h3,
        psi2_rad=psi2_rad,
        psi3_rad=psi3_rad,
    )
    phase_rad = (
        initial_phase_rad + 2.0 * np.pi * fundamental_frequency_hz * time_s
    )
    signal = (
        np.cos(phase_rad)
        + h2 * np.cos(2.0 * phase_rad + psi2_rad)
        + h3 * np.cos(3.0 * phase_rad + psi3_rad)
    )
    return signal, phase_rad


def white_noise(*, sample_count: int, seed: int) -> np.ndarray:
    _validate_sample_count(sample_count)
    return np.random.default_rng(seed).standard_normal(sample_count)


def pink_noise_timmer_koenig(*, sample_count: int, seed: int) -> np.ndarray:
    _validate_sample_count(sample_count)
    rng = np.random.default_rng(seed)
    frequencies = np.fft.rfftfreq(sample_count)
    spectrum = np.zeros(frequencies.size, dtype=np.complex128)
    real = rng.standard_normal(frequencies.size)
    imag = rng.standard_normal(frequencies.size)
    weights = frequencies[1:] ** -0.5
    spectrum[1:] = (real[1:] + 1j * imag[1:]) * weights
    if sample_count % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    noise = np.fft.irfft(spectrum, n=sample_count)
    noise -= np.mean(noise)
    return noise / np.sqrt(np.mean(noise**2))


def add_noise_at_snr(
    *,
    clean_signal: np.ndarray,
    raw_noise: np.ndarray,
    snr_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    clean = np.asarray(clean_signal, dtype=float)
    noise = np.asarray(raw_noise, dtype=float)
    if clean.ndim != 1 or noise.ndim != 1 or clean.shape != noise.shape:
        raise ValueError("clean_signal and raw_noise must be equal-length 1-D arrays")
    if clean.size == 0 or not np.all(np.isfinite(clean)) or not np.all(np.isfinite(noise)):
        raise ValueError("signal and noise windows must be non-empty and finite")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    clean_rms = np.sqrt(np.mean(clean**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    if clean_rms == 0.0 or noise_rms == 0.0:
        raise ValueError("signal and noise windows must have nonzero RMS")
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    return clean + scaled_noise, scaled_noise
