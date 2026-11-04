"""GRPO_CUDA-style Gymnasium environment for the fixed FA-AFDM problem.

The interaction contract intentionally follows ``custom_envs/MISOenv.py``:
continuous bounded position increments, a fixed-horizon episode, channel plus
position observations, dense per-step reward, and projection with a boundary
cost.  Only the physical objective is changed to the true AFDM rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .channel import FAAFDMChannel, FAAFDMConfig
from .gpu_rate_batcher import shared_gpu_rate_batcher


@dataclass(frozen=True)
class MISOStyleAFDMConfig:
    max_steps: int = 50
    action_step: float = 0.025
    # In the single-step static-optimization variant, the action directly
    # parameterises the four physical coordinates rather than an increment.
    # It is deliberately opt-in: the default remains the GRPO_CUDA-style
    # multi-step navigation task.
    direct_position_action: bool = False
    continuing_task: bool = False
    # With a finite horizon, the same position at different t has different
    # remaining control authority; expose t/T to retain the Markov property.
    include_time_feature: bool = False
    # Opt-in to preserve old experiment semantics. A task-defined horizon is
    # terminal; a rollout cut imposed on a continuing task is truncated.
    finite_horizon_terminal: bool = False
    boundary_penalty: float = 10.0
    # The current rate remains dense at every step. This optional weight gives
    # the *same real rate at the final executed state* extra deployment value.
    # It is never a best-so-far statistic or an external optimum label.
    terminal_rate_weight: float = 0.0
    # ``rate`` reproduces the MISO-style dense objective. ``rate_delta`` is
    # potential-difference shaping: over a complete episode its un-discounted
    # sum is exactly final true rate minus the reset true rate. It therefore
    # trains the same terminal-position objective without supplying an oracle
    # target or a best-observed location.
    reward_mode: str = "rate"
    # Optional finite-horizon potential Phi(s,t)=c*(1-t/T)*R(s).
    # With gamma=1 the added episode return is exactly -c*R(reset).
    rate_potential_scale: float = 0.2
    terminal_potential_alpha: float = 1.0 / 1.2
    # Optional low-weight dense auxiliary signal for the terminal objective.
    # It is divided by horizon so its episode contribution stays interpretable.
    dense_rate_coefficient: float = 0.0
    dense_rate_baseline: float = 15.0
    # A smooth, monotone preference for rates near/above a user-specified
    # target.  It never identifies a location; it transforms only measured
    # scalar rate before potential-difference reward shaping.
    rate_target: float | None = None
    target_bonus: float = 0.0
    target_temperature: float = 2.0
    # Formal bounded-policy runs enable this to detect, rather than silently
    # repair, any policy/action-space contract violation.
    strict_action_bounds: bool = False
    # A static AFDM objective can use one public reference state. This is the
    # geometric centre, never a reward-derived or best-observed position.
    random_reset: bool = True
    # Optional externally supplied start pool used only for controlled
    # generalisation diagnostics. Entries are physical [u_x,u_y,q_x,q_y]
    # coordinates and must never be selected using a rate or reference search.
    reset_position_pool: tuple[tuple[float, float, float, float], ...] | None = None
    # When a pool is present, this is the probability of drawing from it;
    # remaining resets use the original uniform physical-region distribution.
    reset_position_pool_probability: float = 1.0
    seed: int = 42
    gpu_rate_batch_size: int = 0
    gpu_rate_device: str = "cuda"
    gpu_rate_backend: str = "local"
    gpu_rate_socket: str = ""


class MISOStyleFAAFDMEnv(gym.Env):
    """50-step AFDM navigation environment with MISOEnv-compatible semantics."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        channel_config: FAAFDMConfig,
        config: MISOStyleAFDMConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or MISOStyleAFDMConfig()
        self.channel = FAAFDMChannel(channel_config)
        self.rng = np.random.default_rng(self.cfg.seed)
        self._inside_step = False
        # Assigned only by the threaded Dreamer wrapper. A fixed slot removes
        # thread-arrival order from the CUDA rate batch while leaving PPO's
        # public environment contract unchanged.
        self.gpu_rate_slot: int | None = None
        self._rate_batcher = (
            shared_gpu_rate_batcher(
                self.channel,
                self.cfg.gpu_rate_batch_size,
                self.cfg.gpu_rate_device,
                self.cfg.gpu_rate_backend,
                self.cfg.gpu_rate_socket,
            )
            if self.cfg.gpu_rate_batch_size > 1
            else None
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        csi_dim = int(self.channel.csi_vector().size)
        tail_dim = 4 + int(self.cfg.include_time_feature)
        self.observation_space = spaces.Box(
            # Gymnasium/SB3 checkers require finite bounds for Box spaces.  The
            # CSI components are numerically many orders below this conservative
            # physical guard; position features retain their exact [-1, 1] box.
            low=np.concatenate([np.full(csi_dim, -1e6), np.full(tail_dim, -1.0)]).astype(np.float32),
            high=np.concatenate([np.full(csi_dim, 1e6), np.full(tail_dim, 1.0)]).astype(np.float32),
            dtype=np.float32,
        )
        self.u = np.zeros(2, dtype=np.float32)
        self.q = np.zeros(2, dtype=np.float32)
        self.t = 0
        self.rate = 0.0
        self.best_rate = -float("inf")
        self.reset_source = "random"
        if self.cfg.reset_position_pool is not None:
            pool = np.asarray(self.cfg.reset_position_pool, dtype=np.float32)
            if pool.ndim != 2 or pool.shape[1] != 4 or len(pool) == 0:
                raise ValueError("reset_position_pool must have shape [count, 4]")
            if not np.isfinite(pool).all() or np.max(np.abs(pool)) > 0.2500001:
                raise ValueError("reset_position_pool must be finite and lie in the 50-cm region")
            if not (0.0 <= self.cfg.reset_position_pool_probability <= 1.0):
                raise ValueError("reset_position_pool_probability must be in [0, 1]")
            self._reset_position_pool = pool
        else:
            self._reset_position_pool = None

    def _sample_position(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return self.rng.uniform(low, high).astype(np.float32)

    @staticmethod
    def _normalise(position: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return (2.0 * (position - low) / (high - low) - 1.0).astype(np.float32)

    def _observation(self) -> np.ndarray:
        tx_low, tx_high = self.channel.tx_bounds
        rx_low, rx_high = self.channel.rx_bounds
        features = [
            self.channel.csi_vector(),
            self._normalise(self.u, tx_low, tx_high),
            self._normalise(self.q, rx_low, rx_high),
        ]
        if self.cfg.include_time_feature:
            features.append(np.array([min(1.0, self.t / max(1, self.cfg.max_steps))], dtype=np.float32))
        return np.concatenate(features).astype(np.float32)

    def _rate(self) -> float:
        if self._rate_batcher is not None and self._inside_step:
            return float(self._rate_batcher.rate(self.u, self.q, self.gpu_rate_slot))
        return float(self.channel.rate(self.u, self.q))

    def _utility(self, rate: float) -> float:
        if self.cfg.rate_target is None or self.cfg.target_bonus == 0.0:
            return float(rate)
        temperature = max(float(self.cfg.target_temperature), 1e-6)
        scaled = (float(rate) - float(self.cfg.rate_target)) / temperature
        return float(rate + self.cfg.target_bonus * np.logaddexp(0.0, scaled))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        tx_low, tx_high = self.channel.tx_bounds
        rx_low, rx_high = self.channel.rx_bounds
        use_pool = self._reset_position_pool is not None and self.rng.random() < self.cfg.reset_position_pool_probability
        if use_pool:
            point = self._reset_position_pool[self.rng.integers(len(self._reset_position_pool))]
            default_u, default_q = point[:2], point[2:]
            self.reset_source = "pool"
        else:
            default_u = self._sample_position(tx_low, tx_high) if self.cfg.random_reset else np.zeros(2, dtype=np.float32)
            default_q = self._sample_position(rx_low, rx_high) if self.cfg.random_reset else np.zeros(2, dtype=np.float32)
            self.reset_source = "random" if self.cfg.random_reset else "centre"
        self.u = np.asarray(options.get("u", default_u), dtype=np.float32)
        self.q = np.asarray(options.get("q", default_q), dtype=np.float32)
        if "u" in options or "q" in options:
            self.reset_source = "provided"
        self.t = 0
        # Reset rate is state construction, not a policy action or training query.
        self.rate = self._rate()
        self.best_rate = self.rate
        return self._observation(), self._info(boundary_distance=0.0)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if self.cfg.strict_action_bounds and not self.action_space.contains(action):
            raise ValueError(f"Policy emitted an out-of-bounds action: {action}")
        action = np.clip(action, -1.0, 1.0)
        previous_rate = self.rate
        tx_low, tx_high = self.channel.tx_bounds
        rx_low, rx_high = self.channel.rx_bounds
        if self.cfg.direct_position_action:
            # [-1, 1]^4 maps exactly to the two 50-cm feasible squares. The
            # policy therefore selects a position in one real interaction;
            # no known good position or search candidate is supplied.
            proposed_u = (tx_low + (action[:2] + 1.0) * 0.5 * (tx_high - tx_low))
            proposed_q = (rx_low + (action[2:] + 1.0) * 0.5 * (rx_high - rx_low))
        else:
            proposed_u = self.u + self.cfg.action_step * action[:2]
            proposed_q = self.q + self.cfg.action_step * action[2:]
        clipped_u = np.clip(proposed_u, tx_low, tx_high).astype(np.float32)
        clipped_q = np.clip(proposed_q, rx_low, rx_high).astype(np.float32)
        boundary_distance = float(
            np.abs(proposed_u - clipped_u).sum() + np.abs(proposed_q - clipped_q).sum()
        )
        self.u, self.q = clipped_u, clipped_q
        self._inside_step = True
        try:
            self.rate = self._rate()
        finally:
            self._inside_step = False
        self.best_rate = max(self.best_rate, self.rate)
        self.t += 1
        horizon_reached = (not self.cfg.continuing_task) and self.t >= self.cfg.max_steps
        terminated = horizon_reached and self.cfg.finite_horizon_terminal
        truncated = horizon_reached and not self.cfg.finite_horizon_terminal
        if self.cfg.reward_mode == "rate":
            # Exact MISOEnv form, with AFDM rate in place of MISO robust sum-rate.
            reward = self.rate / self.cfg.max_steps
        elif self.cfg.reward_mode == "rate_delta":
            # This is based only on the two consecutive true rate queries.
            reward = self._utility(self.rate) - self._utility(previous_rate)
        elif self.cfg.reward_mode == "rate_potential":
            if self.cfg.continuing_task or not self.cfg.finite_horizon_terminal:
                raise ValueError('rate_potential requires a finite terminal horizon')
            remaining = 1.0 - self.t / self.cfg.max_steps
            previous_remaining = 1.0 - (self.t - 1) / self.cfg.max_steps
            reward = self.rate / self.cfg.max_steps + self.cfg.rate_potential_scale * (
                remaining * self.rate - previous_remaining * previous_rate)
        elif self.cfg.reward_mode == "terminal_potential":
            if not self.cfg.finite_horizon_terminal or self.cfg.continuing_task:
                raise ValueError('terminal_potential requires a finite terminal horizon')
            # Phi_T=0. Entire return = scale*(R_T-alpha*R_0)-boundary_cost.
            alpha = self.cfg.terminal_potential_alpha
            potential_now = 0.0 if horizon_reached else alpha * self.rate
            reward = self.cfg.terminal_rate_weight * (potential_now - alpha * previous_rate)
        else:
            raise ValueError(f"Unknown reward_mode: {self.cfg.reward_mode}")
        reward += (
            self.cfg.dense_rate_coefficient
            * (self.rate - self.cfg.dense_rate_baseline)
            / self.cfg.max_steps
        )
        reward -= self.cfg.boundary_penalty * boundary_distance / self.cfg.max_steps
        if horizon_reached:
            reward += self.cfg.terminal_rate_weight * self.rate
        return self._observation(), float(reward), terminated, truncated, self._info(boundary_distance)

    def _info(self, boundary_distance: float) -> dict:
        return {
            "rate": float(self.rate),
            "best_rate": float(self.best_rate),
            "boundary_penalty": float(boundary_distance),
            "u": self.u.astype(float).tolist(),
            "q": self.q.astype(float).tolist(),
            "step": int(self.t),
            "reset_source": self.reset_source,
        }

    def close(self) -> None:
        return None
