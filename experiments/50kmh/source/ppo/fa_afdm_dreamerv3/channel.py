from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


OBJECTIVES = ("afdm_rate", "time_snr_rate", "time_mmse_rate", "time_sinr", "time_sinr_h")


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
    noise_power_dbm: float | None = None
    channel_gain_scale: float = 1.0
    extra_path_loss_db: float = 0.0
    link_distance_m: float = 0.0
    link_distance_min_m: float = 0.0
    link_distance_max_m: float = 0.0
    path_loss_ref_distance_m: float = 100.0
    path_loss_ref_db: float | None = None
    path_loss_exponent: float = 2.5
    doppler_scale: float = 1.0
    fixed_doppler_hz: float | None = None
    angle_dependent_doppler: bool = False
    velocity_direction_deg: float = 0.0
    delay_error_samples: float = 0.0
    tx_angle_error_deg: float = 0.0
    rx_angle_error_deg: float = 0.0
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
        self.tx_zeta = self.tx_zeta + np.deg2rad(cfg.tx_angle_error_deg)
        self.tx_eta = self.tx_eta + np.deg2rad(cfg.tx_angle_error_deg)
        self.rx_zeta = self.rx_zeta + np.deg2rad(cfg.rx_angle_error_deg)
        self.rx_eta = self.rx_eta + np.deg2rad(cfg.rx_angle_error_deg)

        tx_angle_loss = max(0.0, np.cos(np.deg2rad(cfg.tx_angle_error_deg))) ** 2
        rx_angle_loss = max(0.0, np.cos(np.deg2rad(cfg.rx_angle_error_deg))) ** 2
        self._angle_loss = np.sqrt(tx_angle_loss * rx_angle_loss)

        self.delay = self.rng.uniform(0.0, max(1.0, self.lh - 1.0), self.p)
        if cfg.delay_error_samples:
            self.delay = self.delay + cfg.delay_error_samples
        self._sample_path_gains_and_doppler()

        self.daft = self._build_daft_matrix()
        self._torch_rate_cache: dict[str, dict[str, object]] = {}
        self._refresh_rate_cache()

    def _distance_path_loss_db(self, distance: float | np.ndarray) -> float | np.ndarray:
        """Log-distance path loss: PL(d) = PL(d0) + 10 n log10(d/d0).

        The reference path loss at d0 defaults to free space at the carrier
        wavelength: PL0 = 20 log10(4 pi d0 / lambda).  Setting
        ``link_distance_m > 0`` (or ``link_distance_min_m``/``max_m`` for
        per-path distances) activates this model instead of the fixed
        ``channel_gain_scale`` / ``extra_path_loss_db`` knobs.
        """
        cfg = self.cfg
        d0 = cfg.path_loss_ref_distance_m
        pl0 = cfg.path_loss_ref_db
        if pl0 is None:
            pl0 = 20.0 * np.log10(4.0 * np.pi * d0 / cfg.carrier_wavelength)
        return pl0 + 10.0 * cfg.path_loss_exponent * np.log10(np.asarray(distance, dtype=np.float64) / d0)

    def _sample_path_gains_and_doppler(self) -> None:
        """Sample one multipath realization while preserving physical constants."""
        real = self.rng.normal(0.0, 1.0, self.p)
        imag = self.rng.normal(0.0, 1.0, self.p)
        if self.cfg.link_distance_min_m > 0.0 and self.cfg.link_distance_max_m > self.cfg.link_distance_min_m:
            # Per-path link distances drawn from a uniform range; each path
            # suffers its own log-distance path loss.
            self.path_distance = self.rng.uniform(
                self.cfg.link_distance_min_m, self.cfg.link_distance_max_m, self.p
            )
            gain_scale = 10.0 ** (-np.asarray(self._distance_path_loss_db(self.path_distance)) / 20.0)
        elif self.cfg.link_distance_m > 0.0:
            self.path_distance = np.full(self.p, self.cfg.link_distance_m)
            gain_scale = 10.0 ** (-float(self._distance_path_loss_db(self.cfg.link_distance_m)) / 20.0)
        else:
            self.path_distance = np.zeros(self.p)
            extra_loss = 10.0 ** (-self.cfg.extra_path_loss_db / 20.0)
            gain_scale = self.cfg.channel_gain_scale * extra_loss
        self.path_gain = (
            gain_scale
            * self._angle_loss
            * (real + 1j * imag)
            / np.sqrt(2.0)
        )
        if self.cfg.fixed_doppler_hz is None:
            doppler_index = self.rng.integers(-2, 3, self.p)
            doppler_frac = self.rng.uniform(-0.5, 0.5, self.p)
            doppler = self.cfg.doppler_scale * (doppler_index + doppler_frac)
            self.digital_doppler = doppler / self.n
        elif self.cfg.angle_dependent_doppler:
            velocity_direction = np.deg2rad(self.cfg.velocity_direction_deg)
            projection = np.cos(self.rx_zeta) * np.cos(self.rx_eta - velocity_direction)
            physical_doppler_hz = float(self.cfg.fixed_doppler_hz) * projection
            self.digital_doppler = physical_doppler_hz * self.ts
            doppler = self.digital_doppler * self.n
        else:
            # digital_doppler is measured in cycles per sample, so a physical
            # Doppler frequency nu maps to nu * Ts.  A scalar fixed value is
            # intentionally shared by all paths for the fixed-speed baseline.
            self.digital_doppler = np.full(
                self.p,
                float(self.cfg.fixed_doppler_hz) * self.ts,
                dtype=np.float64,
            )
            doppler = self.digital_doppler * self.n
        self.alpha_max = int(np.ceil(np.max(np.abs(doppler))))
        self.c1 = (2.0 * self.alpha_max + 1.0) / (2.0 * self.n)

    def resample_realization(self) -> None:
        """Resample AoA, AoD, complex gains, and angle-dependent Doppler.

        Path delays, distance-dependent gain scale, speed, and all waveform
        parameters remain fixed. This is intended to be called once per episode.
        """
        cfg = self.cfg
        self.tx_zeta = self.rng.uniform(-np.pi / 2.0, np.pi / 2.0, self.p)
        self.tx_eta = self.rng.uniform(-np.pi, np.pi, self.p)
        self.rx_zeta = self.rng.uniform(-np.pi / 2.0, np.pi / 2.0, self.p)
        self.rx_eta = self.rng.uniform(-np.pi, np.pi, self.p)
        self.tx_zeta += np.deg2rad(cfg.tx_angle_error_deg)
        self.tx_eta += np.deg2rad(cfg.tx_angle_error_deg)
        self.rx_zeta += np.deg2rad(cfg.rx_angle_error_deg)
        self.rx_eta += np.deg2rad(cfg.rx_angle_error_deg)
        self._sample_path_gains_and_doppler()
        self.daft = self._build_daft_matrix()
        self._refresh_rate_cache()

    def _refresh_rate_cache(self) -> None:
        """Cache fixed-realization tensors used by every position query.

        A fixed-channel optimization changes only the two antenna positions.
        Delays, Doppler, path gains and all 128x128 index maps are invariant,
        so rebuilding them on every query wastes most of the environment-side
        CPU time without changing the AFDM equation.
        """
        n_time = np.arange(self.n, dtype=np.float64)[None, :]
        delays = np.asarray(self.delay, dtype=np.float64)[:, None]
        l_idx = np.arange(self.lh, dtype=np.float64)[None, :]
        pulse = _sinc(l_idx - delays)[:, None, :]
        doppler = np.exp(
            1j
            * 2.0
            * np.pi
            * np.asarray(self.digital_doppler, dtype=np.float64)[:, None]
            * (n_time - delays)
        )[:, :, None]
        self._omega = (
            np.asarray(self.path_gain, dtype=np.complex128)[:, None, None]
            / np.sqrt(self.p)
            * doppler
            * pulse
        )

        n_idx = np.arange(self.n, dtype=np.int64)[:, None]
        v_idx = np.arange(self.n, dtype=np.int64)[None, :]
        delay_tap = (n_idx - v_idx) % self.n
        valid = delay_tap < self.lh
        row = np.broadcast_to(n_idx, (self.n, self.n))
        col = np.broadcast_to(v_idx, (self.n, self.n))
        gamma = np.ones((self.n, self.n), dtype=np.complex128)
        upper = (row < col) & valid
        gamma[upper] = np.exp(
            -1j
            * 2.0
            * np.pi
            * self.c1
            * (self.n**2 + 2 * self.n * (col[upper] - self.n))
        )
        self._hcpp_valid = valid
        self._hcpp_rows = np.broadcast_to(np.arange(self.n)[:, None], (self.n, self.n))
        self._hcpp_delay_tap = delay_tap
        self._hcpp_gamma = gamma
        self._rate_eye = np.eye(self.n, dtype=np.complex128)
        self._rate_scale = self.cfg.transmit_power / (self.n * self.noise_power)
        self._torch_rate_cache.clear()

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

        # The path/tap tensor is constant for the fixed channel; only the
        # position-dependent field-response phases vary across queries.
        h = np.sum(
            phase_r[:, None, None] * self._omega * phase_t[:, None, None],
            axis=0,
            dtype=np.complex128,
        )
        return h

    def h_cpp(self, u: np.ndarray, q: np.ndarray) -> np.ndarray:
        h = self.sampled_channel(u, q)
        out = np.zeros((self.n, self.n), dtype=np.complex128)
        valid = self._hcpp_valid
        out[valid] = self._hcpp_gamma[valid] * h[
            self._hcpp_rows[valid], self._hcpp_delay_tap[valid]
        ]
        return out

    def h_eff(self, u: np.ndarray, q: np.ndarray) -> np.ndarray:
        h_cpp = self.h_cpp(u, q)
        return self.daft @ h_cpp @ self.daft.conj().T

    def csi_vector(self) -> np.ndarray:
        """Channel state vector (steering directions, gains, delays, Doppler).

        Provides the (known) channel realization as partial CSI to the policy.
        The per-path steering directions cos(zeta)cos(eta), cos(zeta)sin(eta)
        in the FA plane are exposed explicitly, along with normalized path
        gains, phases, delays and Doppler, so the policy can steer the FA
        toward the dominant path instead of searching blindly.
        """
        tx_dir = np.stack(
            [np.cos(self.tx_zeta) * np.cos(self.tx_eta), np.cos(self.tx_zeta) * np.sin(self.tx_eta)],
            axis=1,
        )
        rx_dir = np.stack(
            [np.cos(self.rx_zeta) * np.cos(self.rx_eta), np.cos(self.rx_zeta) * np.sin(self.rx_eta)],
            axis=1,
        )
        gain_mag = np.abs(self.path_gain)
        gain_norm = gain_mag / (np.max(gain_mag) + 1e-12)
        gain_phase = np.angle(self.path_gain)
        return np.concatenate(
            [
                tx_dir.ravel(),
                rx_dir.ravel(),
                gain_norm,
                gain_phase,
                self.delay,
                self.digital_doppler,
            ]
        ).astype(np.float32)

    def rate(self, u: np.ndarray, q: np.ndarray) -> float:
        # DAFT is unitary, hence H_eff H_eff^H is unitarily similar to
        # H_cpp H_cpp^H and has exactly the same log-det achievable rate.
        # Evaluating the invariant form avoids two dense DAFT transforms per
        # queried position while preserving the AFDM-domain rate numerically.
        h_cpp = self.h_cpp(u, q)
        gram = h_cpp @ h_cpp.conj().T
        # I + scale * H H^H is Hermitian positive definite. Cholesky is
        # equivalent to slogdet here and avoids a general LU factorization.
        factor = np.linalg.cholesky(self._rate_eye + self._rate_scale * gram)
        return float(2.0 * np.log(np.real(np.diag(factor))).sum() / np.log(2.0))

    def rate_batch_gpu(
        self,
        u: np.ndarray,
        q: np.ndarray,
        *,
        device: str = "cuda",
    ) -> np.ndarray:
        """Evaluate exact AFDM log-det rates for a batch of physical positions on GPU.

        This is the same fixed-channel Cholesky expression as :meth:`rate`.
        It only changes where a batch is evaluated, not the channel equation.
        """
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on training env.
            raise RuntimeError("rate_batch_gpu requires a CUDA-enabled PyTorch installation") from exc
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("rate_batch_gpu requested CUDA but no CUDA device is available")

        u_np = np.asarray(u, dtype=np.float64).reshape(-1, 2)
        q_np = np.asarray(q, dtype=np.float64).reshape(-1, 2)
        if u_np.shape != q_np.shape:
            raise ValueError(f"u and q must have the same shape, got {u_np.shape} and {q_np.shape}")
        key = str(device)
        cached = self._torch_rate_cache.get(key)
        if cached is None:
            tx_dir = np.stack([
                np.cos(self.tx_zeta) * np.cos(self.tx_eta),
                np.cos(self.tx_zeta) * np.sin(self.tx_eta),
            ], axis=1)
            rx_dir = np.stack([
                np.cos(self.rx_zeta) * np.cos(self.rx_eta),
                np.cos(self.rx_zeta) * np.sin(self.rx_eta),
            ], axis=1)
            cached = {
                "tx_dir": torch.as_tensor(tx_dir, dtype=torch.float64, device=device),
                "rx_dir": torch.as_tensor(rx_dir, dtype=torch.float64, device=device),
                "omega": torch.as_tensor(self._omega, dtype=torch.complex128, device=device),
                "rows": torch.as_tensor(self._hcpp_rows.copy(), dtype=torch.long, device=device),
                # Invalid delay taps are masked after gathering; give them a
                # safe in-range index so CUDA never evaluates an invalid gather.
                "taps": torch.as_tensor(
                    np.where(self._hcpp_valid, self._hcpp_delay_tap, 0).copy(),
                    dtype=torch.long,
                    device=device,
                ),
                "valid": torch.as_tensor(self._hcpp_valid.copy(), dtype=torch.bool, device=device),
                "gamma": torch.as_tensor(self._hcpp_gamma, dtype=torch.complex128, device=device),
                "eye": torch.eye(self.n, dtype=torch.complex128, device=device),
            }
            self._torch_rate_cache[key] = cached

        u_t = torch.as_tensor(u_np, dtype=torch.float64, device=device)
        q_t = torch.as_tensor(q_np, dtype=torch.float64, device=device)
        rho_t = u_t @ cached["tx_dir"].T
        rho_r = q_t @ cached["rx_dir"].T
        phase_t = torch.exp(1j * (2.0 * math.pi / self.cfg.carrier_wavelength) * rho_t)
        phase_r = torch.exp(-1j * (2.0 * math.pi / self.cfg.carrier_wavelength) * rho_r)
        h = torch.einsum("bp,pnl,bp->bnl", phase_r, cached["omega"], phase_t)
        h_cpp = h[:, cached["rows"], cached["taps"]] * cached["gamma"]
        h_cpp = torch.where(cached["valid"].unsqueeze(0), h_cpp, torch.zeros_like(h_cpp))
        gram = h_cpp @ h_cpp.mH
        factor = torch.linalg.cholesky(cached["eye"].unsqueeze(0) + self._rate_scale * gram)
        rates = 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1).real).sum(dim=-1) / math.log(2.0)
        return rates.detach().cpu().numpy().astype(np.float64, copy=False)

    def time_domain_sinr(self, u: np.ndarray, q: np.ndarray) -> float:
        """Time-domain antenna-selection proxy, without applying the DAFT.

        The diagonal of H_cpp is treated as the desired sample-aligned signal
        and its off-diagonal energy as inter-symbol interference.  This is
        deliberately not the time-domain log-det capacity: because the DAFT
        is unitary, that capacity would be identical to :meth:`rate` and would
        not form a meaningful time-domain-only baseline.
        """
        channel = self.h_cpp(u, q)
        power = np.abs(channel) ** 2
        desired = float(np.trace(power).real / self.n)
        interference = float((power.sum().real - np.trace(power).real) / self.n)
        symbol_power = self.cfg.transmit_power / self.n
        sinr = symbol_power * desired / (self.noise_power + symbol_power * interference)
        return float(np.log2(1.0 + max(0.0, sinr)))

    def time_domain_snr_rate(self, u: np.ndarray, q: np.ndarray) -> float:
        """Scalar time-domain rate obtained from the average channel power.

        The sampled response h[n, l; u, q] implements
        f^H(q) Sigma(n, l) g(u).  The power is summed over the delay taps
        and averaged over the N time samples before applying log2(1 + SNR).
        """
        response = self.sampled_channel(u, q)
        average_channel_power = float(np.mean(np.sum(np.abs(response) ** 2, axis=1)))
        symbol_power = self.cfg.transmit_power / self.n
        snr = symbol_power * average_channel_power / self.noise_power
        return float(np.log2(1.0 + max(0.0, snr)))

    def time_domain_sinr_h(self, u: np.ndarray, q: np.ndarray) -> float:
        """Time-domain single-tap SINR computed from the sampled channel h[n, l].

        The sampled impulse response h[n, l; u, q] = f^H(q) Sigma(n, l) g(u)
        is used directly, without constructing the CPP-equivalent matrix H_cpp.
        A single-tap time-domain receiver synchronizes to the strongest delay
        tap (the "main tap") and treats the remaining taps as inter-symbol
        interference.  This is weaker than the AFDM log-det rate because it does
        not coherently combine the multipath energy.
        """
        response = self.sampled_channel(u, q)
        power = np.abs(response) ** 2
        tap_power = power.mean(axis=0)
        main_tap = int(np.argmax(tap_power))
        desired = float(tap_power[main_tap])
        interference = float(tap_power.sum() - tap_power[main_tap])
        symbol_power = self.cfg.transmit_power / self.n
        sinr = symbol_power * desired / (self.noise_power + symbol_power * interference)
        return float(np.log2(1.0 + max(0.0, sinr)))

    def time_domain_mmse_rate(self, u: np.ndarray, q: np.ndarray) -> float:
        """Per-stream achievable rate of a linear MMSE time-domain receiver."""
        channel = self.h_cpp(u, q)
        rho = self.cfg.transmit_power / (self.n * self.noise_power)
        gram = channel.conj().T @ channel
        mse = np.linalg.inv(np.eye(self.n, dtype=np.complex128) + rho * gram)
        diagonal = np.maximum(np.real(np.diag(mse)), np.finfo(float).eps)
        sinr = np.maximum(0.0, 1.0 / diagonal - 1.0)
        return float(np.sum(np.log2(1.0 + sinr)))

    def objective(self, u: np.ndarray, q: np.ndarray, name: str = "afdm_rate") -> float:
        """Return the optimization objective used by the learning environment."""
        if name == "afdm_rate":
            return self.rate(u, q)
        if name == "time_snr_rate":
            return self.time_domain_snr_rate(u, q)
        if name == "time_sinr":
            return self.time_domain_sinr(u, q)
        if name == "time_sinr_h":
            return self.time_domain_sinr_h(u, q)
        if name == "time_mmse_rate":
            return self.time_domain_mmse_rate(u, q)
        raise ValueError(f"Unknown antenna-selection objective: {name!r}")


def make_default_channel(seed: int = 11) -> FAAFDMChannel:
    return FAAFDMChannel(FAAFDMConfig(seed=seed))
