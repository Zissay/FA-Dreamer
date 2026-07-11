from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - fallback for old Gym installs.
    try:
        import gym
        from gym import spaces
    except Exception:  # pragma: no cover - keeps the scripts runnable without Gym.
        class _Env:
            metadata: dict = {}

        class _Box:
            def __init__(self, low, high, shape=None, dtype=np.float32) -> None:
                self.low = np.full(shape, low, dtype=dtype) if shape is not None else np.asarray(low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype) if shape is not None else np.asarray(high, dtype=dtype)
                self.shape = self.low.shape
                self.dtype = dtype

            def sample(self) -> np.ndarray:
                return np.random.uniform(self.low, self.high).astype(self.dtype)

        class _Spaces:
            Box = _Box

        class _Gym:
            Env = _Env

        gym = _Gym()
        spaces = _Spaces()

from .channel import FAAFDMChannel, make_default_channel


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 40
    action_step: float = 0.04
    direct_position_action: bool = False
    random_reset: bool = True
    seed: int = 7
    high_rate_threshold: float = 0.0
    high_rate_bonus: float = 0.0
    high_rate_slope: float = 1.0
    rate_baseline: float = 0.0
    reward_scale: float = 1.0
    raw_rate_scale: float = 0.0
    improvement_weight: float = 0.5
    best_improvement_weight: float = 0.0
    movement_penalty: float = 0.01
    action_l2_penalty: float = 0.0
    regression_penalty: float = 0.0
    best_gap_penalty: float = 0.0
    high_rate_move_penalty: float = 0.0
    high_rate_action_l2_penalty: float = 0.0
    action_smooth_penalty: float = 0.0
    high_rate_smooth_penalty: float = 0.0
    boundary_margin: float = 1.0
    boundary_free_dims: int = 0
    boundary_penalty: float = 0.0
    late_rate_weight: float = 0.0
    late_rate_power: float = 1.0
    terminal_rate_weight: float = 0.0
    initial_positions: tuple[tuple[float, ...], ...] | None = None
    observe_best_position: bool = False
    observe_rate_dynamics: bool = False
    query_log_path: str = ""


class FAAFDMEnv(gym.Env):
    """Gym-style environment for FA position optimization.

    State:
        [u_x, u_y, q_x, q_y, last_du_x, last_du_y, last_dq_x, last_dq_y,
         normalized_rate, normalized_step]

    Action:
        Four continuous values in [-1, 1]. By default they are mapped to
        position increments for transmit position u and receive position q.
        If cfg.direct_position_action is enabled, the action directly
        parameterizes the full transmit/receive positions.

    Reward:
        Achievable rate R(u, q), with a small bonus for rate improvement and a
        small movement penalty to discourage jitter.
    """

    metadata = {"render_modes": []}

    def __init__(self, channel: FAAFDMChannel | None = None, cfg: EnvConfig | None = None) -> None:
        super().__init__()
        self.channel = channel or make_default_channel()
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        obs_low = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0]
        obs_high = [1, 1, 1, 1, 1, 1, 1, 1, np.inf, 1]
        if self.cfg.observe_best_position:
            obs_low.extend([-1, -1, -1, -1])
            obs_high.extend([1, 1, 1, 1])
        if self.cfg.raw_rate_scale > 0.0:
            obs_low.append(0)
            obs_high.append(np.inf)
        if self.cfg.observe_rate_dynamics:
            obs_low.extend([-np.inf, 0])
            obs_high.extend([np.inf, np.inf])
        obs_low = np.array(obs_low, dtype=np.float32)
        obs_high = np.array(obs_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.u = np.zeros(2, dtype=np.float32)
        self.q = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.rate = 0.0
        self.prev_rate = 0.0
        self.best_rate = 0.0
        self.best_u = np.zeros(2, dtype=np.float32)
        self.best_q = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.query_count = 0
        if self.cfg.query_log_path:
            path = Path(self.cfg.query_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "query",
                            "episode_step",
                            "rate",
                            "best_rate",
                            "u_x",
                            "u_y",
                            "q_x",
                            "q_y",
                        ],
                    )
                    writer.writeheader()

    def _sample_position(self, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        low, high = bounds
        return self.rng.uniform(low, high).astype(np.float32)

    def _normalize_pos(self, pos: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        low, high = bounds
        return (2.0 * (pos - low) / (high - low) - 1.0).astype(np.float32)

    def _denormalize_pos(self, value: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        low, high = bounds
        value = np.clip(np.asarray(value, dtype=np.float32), -1.0, 1.0)
        return (low + 0.5 * (value + 1.0) * (high - low)).astype(np.float32)

    def _obs(self) -> np.ndarray:
        tx = self._normalize_pos(self.u, self.channel.tx_bounds)
        rx = self._normalize_pos(self.q, self.channel.rx_bounds)
        normalized_rate = np.array([self.rate / max(1.0, self.best_rate)], dtype=np.float32)
        normalized_step = np.array([self.step_count / self.cfg.max_steps], dtype=np.float32)
        parts = [tx, rx, self.last_action, normalized_rate, normalized_step]
        if self.cfg.observe_best_position:
            best_tx = self._normalize_pos(self.best_u, self.channel.tx_bounds)
            best_rx = self._normalize_pos(self.best_q, self.channel.rx_bounds)
            parts.extend([best_tx, best_rx])
        if self.cfg.raw_rate_scale > 0.0:
            parts.append(np.array([self.rate * self.cfg.raw_rate_scale], dtype=np.float32))
        if self.cfg.observe_rate_dynamics:
            denom = max(1.0, self.best_rate)
            parts.append(
                np.array(
                    [
                        (self.rate - self.prev_rate) / denom,
                        max(0.0, self.best_rate - self.rate) / denom,
                    ],
                    dtype=np.float32,
                )
            )
        return np.concatenate(parts).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        if "u" in options and "q" in options:
            self.u = np.asarray(options["u"], dtype=np.float32).copy()
            self.q = np.asarray(options["q"], dtype=np.float32).copy()
        elif self.cfg.initial_positions:
            item = self.cfg.initial_positions[int(self.rng.integers(len(self.cfg.initial_positions)))]
            self.u = np.asarray(item[:2], dtype=np.float32).copy()
            self.q = np.asarray(item[2:], dtype=np.float32).copy()
        elif self.cfg.random_reset:
            self.u = self._sample_position(self.channel.tx_bounds)
            self.q = self._sample_position(self.channel.rx_bounds)
        else:
            self.u = np.zeros(2, dtype=np.float32)
            self.q = np.zeros(2, dtype=np.float32)

        self.last_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.rate = self.channel.rate(self.u, self.q)
        self.prev_rate = self.rate
        self.best_rate = self.rate
        self.best_u = self.u.copy()
        self.best_q = self.q.copy()
        return self._obs(), self._info()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        old_rate = self.rate
        old_best_rate = self.best_rate
        prev_action = self.last_action.copy()

        if self.cfg.direct_position_action:
            self.u = self._denormalize_pos(action[:2], self.channel.tx_bounds)
            self.q = self._denormalize_pos(action[2:], self.channel.rx_bounds)
        else:
            tx_low, tx_high = self.channel.tx_bounds
            rx_low, rx_high = self.channel.rx_bounds
            self.u = np.clip(self.u + self.cfg.action_step * action[:2], tx_low, tx_high).astype(np.float32)
            self.q = np.clip(self.q + self.cfg.action_step * action[2:], rx_low, rx_high).astype(np.float32)

        self.rate = self.channel.rate(self.u, self.q)
        self.prev_rate = old_rate
        if self.rate > self.best_rate:
            self.best_rate = self.rate
            self.best_u = self.u.copy()
            self.best_q = self.q.copy()
        self.last_action = action
        self.step_count += 1
        self.query_count += 1

        improvement = self.rate - old_rate
        best_improvement = max(0.0, self.rate - old_best_rate)
        action_norm = float(np.linalg.norm(action))
        action_l2 = float(np.dot(action, action))
        action_delta_l2 = float(np.dot(action - prev_action, action - prev_action))
        movement_penalty = self.cfg.movement_penalty * action_norm
        action_l2_penalty = self.cfg.action_l2_penalty * action_l2
        action_smooth_penalty = self.cfg.action_smooth_penalty * action_delta_l2
        regression_penalty = self.cfg.regression_penalty * max(0.0, old_rate - self.rate)
        best_gap_penalty = self.cfg.best_gap_penalty * max(0.0, self.best_rate - self.rate)
        pos_norm = np.concatenate(
            [
                self._normalize_pos(self.u, self.channel.tx_bounds),
                self._normalize_pos(self.q, self.channel.rx_bounds),
            ]
        )
        boundary_excess = np.maximum(0.0, np.abs(pos_norm) - self.cfg.boundary_margin)
        if self.cfg.boundary_free_dims > 0:
            boundary_excess = np.sort(boundary_excess)[::-1][self.cfg.boundary_free_dims :]
        boundary_penalty = self.cfg.boundary_penalty * float(np.dot(boundary_excess, boundary_excess))
        rate_objective = self.rate - self.cfg.rate_baseline
        progress = min(1.0, self.step_count / max(1, self.cfg.max_steps))
        late_rate_bonus = (
            self.cfg.late_rate_weight
            * (progress ** self.cfg.late_rate_power)
            * rate_objective
        )
        terminal_rate_bonus = (
            self.cfg.terminal_rate_weight * rate_objective
            if self.step_count >= self.cfg.max_steps
            else 0.0
        )
        high_rate_bonus = 0.0
        if self.cfg.high_rate_bonus > 0.0 and self.cfg.high_rate_threshold > 0.0:
            surplus = max(0.0, self.rate - self.cfg.high_rate_threshold)
            high_rate_bonus = self.cfg.high_rate_bonus * np.tanh(self.cfg.high_rate_slope * surplus)
        high_rate_move_penalty = 0.0
        high_rate_action_l2_penalty = 0.0
        high_rate_smooth_penalty = 0.0
        if self.cfg.high_rate_move_penalty > 0.0 and self.cfg.high_rate_threshold > 0.0:
            surplus = max(0.0, self.rate - self.cfg.high_rate_threshold)
            high_rate_move_penalty = (
                self.cfg.high_rate_move_penalty
                * action_norm
                * surplus
            )
            high_rate_action_l2_penalty = (
                self.cfg.high_rate_action_l2_penalty
                * action_l2
                * surplus
            )
        if self.cfg.high_rate_smooth_penalty > 0.0 and self.cfg.high_rate_threshold > 0.0:
            surplus = max(0.0, self.rate - self.cfg.high_rate_threshold)
            high_rate_smooth_penalty = (
                self.cfg.high_rate_smooth_penalty
                * action_delta_l2
                * surplus
            )
        reward = float(
            (
                rate_objective
                + late_rate_bonus
                + terminal_rate_bonus
                + self.cfg.improvement_weight * improvement
                + self.cfg.best_improvement_weight * best_improvement
                - movement_penalty
                - action_l2_penalty
                - action_smooth_penalty
                - regression_penalty
                - best_gap_penalty
                - boundary_penalty
                - high_rate_move_penalty
                - high_rate_action_l2_penalty
                - high_rate_smooth_penalty
                + high_rate_bonus
            )
            * self.cfg.reward_scale
        )
        if self.cfg.query_log_path:
            with Path(self.cfg.query_log_path).open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "query",
                        "episode_step",
                        "rate",
                        "best_rate",
                        "u_x",
                        "u_y",
                        "q_x",
                        "q_y",
                    ],
                )
                writer.writerow(
                    {
                        "query": int(self.query_count),
                        "episode_step": int(self.step_count),
                        "rate": float(self.rate),
                        "best_rate": float(self.best_rate),
                        "u_x": float(self.u[0]),
                        "u_y": float(self.u[1]),
                        "q_x": float(self.q[0]),
                        "q_y": float(self.q[1]),
                    }
                )
        terminated = False
        truncated = self.step_count >= self.cfg.max_steps
        return self._obs(), reward, terminated, truncated, self._info()

    def _info(self) -> dict[str, float | list[float]]:
        return {
            "rate": float(self.rate),
            "best_rate": float(self.best_rate),
            "u": self.u.astype(float).tolist(),
            "q": self.q.astype(float).tolist(),
            "best_u": self.best_u.astype(float).tolist(),
            "best_q": self.best_q.astype(float).tolist(),
            "step": int(self.step_count),
        }
