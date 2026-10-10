from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from offline_src.config import SO_PHASE_BAND


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(signal: np.ndarray, fs: float, band: Tuple[float, float]) -> np.ndarray:
    b, a = butter_bandpass(band[0], band[1], fs, order=4)
    return filtfilt(b, a, signal)


@dataclass
class HilbertPhaseEstimator:
    sfreq: float
    band: Tuple[float, float] = SO_PHASE_BAND

    def transform(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Return filtered signal, phase, amplitude envelope, and instantaneous frequency."""
        filtered = bandpass_filter(signal, self.sfreq, self.band)
        analytic = hilbert(filtered)
        phase = np.angle(analytic)
        phase_unwrapped = np.unwrap(phase)
        amplitude = np.abs(analytic)
        instantaneous_frequency = np.concatenate(
            [
                np.array([0.0]),
                np.diff(phase_unwrapped) / (2 * np.pi) * self.sfreq,
            ]
        )
        return {
            "filtered": filtered,
            "analytic": analytic,
            "phase": phase,
            "amplitude": amplitude,
            "instantaneous_frequency": instantaneous_frequency,
        }
