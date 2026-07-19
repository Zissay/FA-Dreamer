from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FAAFDMConfig:
    """Parameters for the FA-assisted AFDM channel in the WCL 2026 draft.

    The receiver position is named q in code to match the requested workflow.
    It corresponds to r in the current LaTeX draft.
    """

    n_subcarriers: int = 16
    n_paths: int = 4
    channel_memory: int = 5
    carrier_wavelength: float = 0.1
    bandwidth: float = 10e6
    tx_region: float = 0.5
    rx_region: float = 0.5
    transmit_power: float = 1.0
    noise_power: float = 1e-2
    noise_power_dbm: float | None = -95.0
    channel_gain_scale: float = 2.280350850198276e-6
    doppler_scale: float = 1.0
    c2: float = 1.0 / 97.0
    seed: int = 11


def dbm_to_watt(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def _unitary_dft(n: int) -> np.ndarray:
    row = np.arange(n)[:, None]
    col = np.arange(n)[None, :]
    return np.exp(-1j * 2.0 * np.pi * row * col / n) / np.sqrt(n)


def _sinc(x: np.ndarray) -> np.ndarray:
    return np.sinc(x)


class FAAFDMChannel:
    """Computes H_eff(u, q) and the achievable-rate reward.

    Implemented equations:
    - extra distances and field responses from equations (1)-(3) in the
      user's WCL draft;
    - sampled channel h[n,l;u,q] from equations (6)-(8);
    - H_cpp from equation (10);
    - H_eff = A H_cpp A^H from equation (11);
    - R(u,q) = log2 det(I + Pt/(N sigma^2) H_eff H_eff^H).
    """

    def __init__(self, cfg: FAAFDMConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n = cfg.n_subcarriers
        self.p = cfg.n_paths
        self.lh = cfg.channel_memory
        self.ts = 1.0 / cfg.bandwidth
        if cfg.noise_power_dbm is not None and cfg.noise_power_dbm > -90.0:
            raise ValueError("noise_power_dbm must be <= -90 dBm for the current experiment constraint.")
        self.noise_power = (
            dbm_to_watt(cfg.noise_power_dbm)
            if cfg.noise_power_dbm is not None
            else cfg.noise_power
        )

        self.tx_zeta = self.rng.uniform(-np.pi / 2.0, np.pi / 2.0, self.p)
        self.tx_eta = self.rng.uniform(-np.pi, np.pi, self.p)
        self.rx_zeta = self.rng.uniform(-np.pi / 2.0, np.pi / 2.0, self.p)
        self.rx_eta = self.rng.uniform(-np.pi, np.pi, self.p)

        real = self.rng.normal(0.0, 1.0, self.p)
        imag = self.rng.normal(0.0, 1.0, self.p)
        self.path_gain = cfg.channel_gain_scale * (real + 1j * imag) / np.sqrt(2.0)

        self.delay = self.rng.uniform(0.0, max(1.0, self.lh - 1.0), self.p)
        doppler_index = self.rng.integers(-2, 3, self.p)
        doppler_frac = self.rng.uniform(-0.5, 0.5, self.p)
        doppler = cfg.doppler_scale * (doppler_index + doppler_frac)
        self.digital_doppler = doppler / self.n
        self.alpha_max = int(np.ceil(np.max(np.abs(doppler))))
        self.c1 = (2.0 * self.alpha_max + 1.0) / (2.0 * self.n)

        self.daft = self._build_daft_matrix()

    @property
    def tx_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        half = self.cfg.tx_region / 2.0
        return np.array([-half, -half], dtype=np.float32), np.array([half, half], dtype=np.float32)

    @property
    def rx_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        half = self.cfg.rx_region / 2.0
        return np.array([-half, -half], dtype=np.float32), np.array([half, half], dtype=np.float32)

    def _build_daft_matrix(self) -> np.ndarray:
        idx = np.arange(self.n)
        lambda_c1 = np.diag(np.exp(-1j * 2.0 * np.pi * self.c1 * idx**2))
        lambda_c2 = np.diag(np.exp(-1j * 2.0 * np.pi * self.cfg.c2 * idx**2))
        return lambda_c2 @ _unitary_dft(self.n) @ lambda_c1

    def _rho_tx(self, u: np.ndarray) -> np.ndarray:
        x, y = np.asarray(u, dtype=np.float64)
        return x * np.cos(self.tx_zeta) * np.cos(self.tx_eta) + y * np.cos(self.tx_zeta) * np.sin(self.tx_eta)

    def _rho_rx(self, q: np.ndarray) -> np.ndarray:
        x, y = np.asarray(q, dtype=np.float64)
        return x * np.cos(self.rx_zeta) * np.cos(self.rx_eta) + y * np.cos(self.rx_zeta) * np.sin(self.rx_eta)

    def sampled_channel(self, u: np.ndarray, q: np.ndarray) -> np.ndarray:
        rho_t = self._rho_tx(u)
        rho_r = self._rho_rx(q)
        phase_t = np.exp(1j * 2.0 * np.pi * rho_t / self.cfg.carrier_wavelength)
        phase_r = np.exp(-1j * 2.0 * np.pi * rho_r / self.cfg.carrier_wavelength)

        h = np.zeros((self.n, self.lh), dtype=np.complex128)
        n_idx = np.arange(self.n)[:, None]
        l_idx = np.arange(self.lh)[None, :]
        for path in range(self.p):
            pulse = _sinc(l_idx - self.delay[path])
            doppler = np.exp(1j * 2.0 * np.pi * self.digital_doppler[path] * (n_idx - self.delay[path]))
            omega = self.path_gain[path] / np.sqrt(self.p) * doppler * pulse
            h += phase_r[path] * omega * phase_t[path]
        return h

    def h_cpp(self, u: np.ndarray, q: np.ndarray) -> np.ndarray:
        h = self.sampled_channel(u, q)
        out = np.zeros((self.n, self.n), dtype=np.complex128)
        for n_idx in range(self.n):
            for v_idx in range(self.n):
                delay_tap = (n_idx - v_idx) % self.n
                if delay_tap >= self.lh:
                    continue
                if n_idx >= v_idx:
                    gamma = 1.0
                else:
                    gamma = np.exp(-1j * 2.0 * np.pi * self.c1 * (self.n**2 + 2 * self.n * (v_idx - self.n)))
                out[n_idx, v_idx] = gamma * h[n_idx, delay_tap]
        return out

    def h_eff(self, u: np.ndarray, q: np.ndarray) -> np.ndarray:
        h_cpp = self.h_cpp(u, q)
        return self.daft @ h_cpp @ self.daft.conj().T

    def rate(self, u: np.ndarray, q: np.ndarray) -> float:
        h_eff = self.h_eff(u, q)
        gram = h_eff @ h_eff.conj().T
        scale = self.cfg.transmit_power / (self.n * self.noise_power)
        sign, logdet = np.linalg.slogdet(np.eye(self.n, dtype=np.complex128) + scale * gram)
        if sign.real <= 0:
            return 0.0
        return float(np.real(logdet) / np.log(2.0))


def make_default_channel(seed: int = 11) -> FAAFDMChannel:
    return FAAFDMChannel(FAAFDMConfig(seed=seed))
