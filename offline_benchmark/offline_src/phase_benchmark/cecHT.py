from __future__ import annotations

import numpy as np
from scipy.fft import fft, fftshift, ifft, ifftshift, next_fast_len
from scipy.signal import bessel, butter, cheby1, cheby2, ellip, freqz


class ECHT:
    """
    Endpoint Corrected Hilbert Transform with optional calibration.

    This implementation is adapted from:
    https://github.com/eikeosmers/cecHT/blob/main/phase.py

    The upstream cecHT implementation is based on MEEGkit's ECHT class and
    cites Schreglmann et al. (2021). The optional calibration follows Osmers
    and Kolossa (2026).
    """

    def __init__(
        self,
        l_freq: float,
        h_freq: float,
        sfreq: float,
        n_fft: int | None = None,
        fft_mode: str = "fast",
        filt_order: int = 1,
        filter_type: str = "butter",
        calibrate: bool = False,
        f0: float | None = None,
    ):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq
        self.n_fft = n_fft
        self.fft_mode = str(fft_mode).lower()
        if self.fft_mode not in ("fast", "exact"):
            raise ValueError("fft_mode must be 'fast' or 'exact'")
        self.filt_order = filt_order
        self.filter_type = filter_type
        self.calibrate = bool(calibrate)
        self.f0 = f0

        self.h_ = None
        self.coef_ = None
        self.calib_gain_ = None
        self.calib_err_ = None

    @staticmethod
    def _design_bandpass(
        l_freq: float,
        h_freq: float,
        sfreq: float,
        filt_order: int,
        n_fft: int,
        filter_type: str = "butter",
    ) -> tuple[np.ndarray, np.ndarray]:
        h = np.zeros(n_fft, dtype=float)
        h[0] = 1
        if n_fft % 2 == 0:
            h[1 : n_fft // 2] = 2
            h[n_fft // 2] = 1
        else:
            h[1 : (n_fft + 1) // 2] = 2

        wn = [l_freq / (sfreq / 2), h_freq / (sfreq / 2)]
        if filter_type == "butter":
            b, a = butter(filt_order, wn, btype="band")
        elif filter_type == "bessel":
            b, a = bessel(filt_order, wn, btype="band", norm="phase")
        elif filter_type == "cheby1":
            b, a = cheby1(filt_order, 1, wn, btype="band")
        elif filter_type == "cheby2":
            b, a = cheby2(filt_order, 40, wn, btype="band")
        elif filter_type == "ellip":
            b, a = ellip(filt_order, 1, 40, wn, btype="band")
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

        filt_freq = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1 / sfreq))
        _, h_center = freqz(b, a, worN=filt_freq, fs=sfreq)
        return h, h_center

    @staticmethod
    def _calibration(
        f0: float,
        sfreq: float,
        n_samples: int,
        l_freq: float,
        h_freq: float,
        filt_order: int = 1,
        filter_type: str = "butter",
        n_fft: int | None = None,
        fft_mode: str = "fast",
    ) -> dict:
        if n_fft is None:
            n_fft = next_fast_len(n_samples) if fft_mode == "fast" else n_samples

        def dirichlet_n(alpha: np.ndarray) -> np.ndarray:
            alpha = np.asarray(alpha, dtype=float)
            out = np.empty(alpha.shape, dtype=np.complex128)
            small = np.abs(alpha) < 1e-12
            out[small] = n_samples
            alpha_ns = alpha[~small]
            out[~small] = (
                np.exp(1j * alpha_ns * (n_samples - 1) / 2)
                * np.sin(0.5 * n_samples * alpha_ns)
                / np.sin(0.5 * alpha_ns)
            )
            return out

        k = np.arange(n_fft)
        omega_k = 2 * np.pi * k / n_fft
        omega0 = 2 * np.pi * f0 / sfreq
        endpoint = n_samples - 1

        h, h_center = ECHT._design_bandpass(
            l_freq=l_freq,
            h_freq=h_freq,
            sfreq=sfreq,
            filt_order=filt_order,
            n_fft=n_fft,
            filter_type=filter_type,
        )
        g = h * ifftshift(h_center)

        x_plus = 0.5 * dirichlet_n(omega0 - omega_k)
        x_minus = 0.5 * dirichlet_n(-omega0 - omega_k)
        phase = np.exp(1j * omega_k * endpoint)

        p = (g * x_plus * phase).sum() / n_fft
        m = (g * x_minus * phase).sum() / n_fft
        g_plus = p * np.exp(-1j * omega0 * endpoint)
        g_minus = m * np.exp(-1j * omega0 * endpoint)

        denom = np.abs(g_plus) ** 2 + np.abs(g_minus) ** 2
        if denom == 0:
            c_opt = 1 + 0j
            j_opt = 0
        else:
            c_opt = np.conj(g_plus) / denom
            j_opt = np.abs(g_minus) ** 2 / denom

        return {
            "Gplus": g_plus,
            "Gminus": g_minus,
            "C_opt": c_opt,
            "J_opt": j_opt,
        }

    def fit(self, x: np.ndarray, y=None):
        x = np.asarray(x)
        n_samples = x.shape[0]
        if self.n_fft is None:
            self.n_fft = next_fast_len(n_samples) if self.fft_mode == "fast" else n_samples

        self.h_, h_center = self._design_bandpass(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            sfreq=self.sfreq,
            filt_order=self.filt_order,
            n_fft=self.n_fft,
            filter_type=self.filter_type,
        )
        self.coef_ = h_center[:, None]

        self.calib_gain_ = None
        self.calib_err_ = None
        if self.calibrate:
            if self.f0 is None:
                raise ValueError("f0 must be provided when calibrate=True.")
            err = self._calibration(
                f0=self.f0,
                sfreq=self.sfreq,
                n_samples=n_samples,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                filt_order=self.filt_order,
                filter_type=self.filter_type,
                n_fft=self.n_fft,
                fft_mode=self.fft_mode,
            )
            c_opt = err["C_opt"]
            if not (np.isfinite(c_opt.real) and np.isfinite(c_opt.imag)):
                raise RuntimeError("Non-finite calibration gain computed for ECHT.")
            self.calib_gain_ = c_opt
            self.calib_err_ = err
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if not np.isrealobj(x):
            x = np.real(x)
        if x.ndim == 1:
            x = x[:, np.newaxis]

        if self.h_ is None or self.coef_ is None:
            self.fit(x)

        n_samples = x.shape[0]
        xf = fft(x, self.n_fft, axis=0)
        xf = xf * self.h_[:, None]
        xf = fftshift(xf, axes=0)
        xf = xf * self.coef_
        xf = ifftshift(xf, axes=0)
        xf = ifft(xf, axis=0)

        if self.calibrate and self.calib_gain_ is not None:
            xf = xf * self.calib_gain_

        return xf[:n_samples, :]

    def fit_transform(self, x: np.ndarray, y=None) -> np.ndarray:
        return self.fit(x).transform(x)
