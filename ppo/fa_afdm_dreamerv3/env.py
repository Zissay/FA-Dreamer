from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import csv
import threading
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

from .channel import FAAFDMChannel, FAAFDMConfig, make_default_channel
from .gpu_rate_batcher import shared_gpu_rate_batcher


class _SharedQueryLog:
    """One locked CSV writer per run, including threaded vector environments."""

    def __init__(self, path: Path, fields: tuple[str, ...]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists() or path.stat().st_size == 0
        self._handle = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=fields)
        self._lock = threading.Lock()
        self._next_query = 0
        if needs_header:
            self._writer.writeheader()
            self._handle.flush()
        else:
            # A resumed run preserves monotonic query identifiers. New formal
            # runs use a fresh directory and therefore start at one.
            with path.open("r", encoding="utf-8", newline="") as source:
                self._next_query = max(0, sum(1 for _ in source) - 1)

    def write(self, row: dict) -> None:
        with self._lock:
            self._next_query += 1
            row["query"] = self._next_query
            self._writer.writerow(row)

    def flush(self) -> None:
        with self._lock:
            self._handle.flush()


_QUERY_LOG_LOCK = threading.Lock()
_QUERY_LOGS: dict[str, _SharedQueryLog] = {}


def _shared_query_log(path: str, fields: tuple[str, ...]) -> _SharedQueryLog:
    resolved = str(Path(path).resolve())
    with _QUERY_LOG_LOCK:
        logger = _QUERY_LOGS.get(resolved)
        if logger is None:
            logger = _SharedQueryLog(Path(resolved), fields)
            _QUERY_LOGS[resolved] = logger
        return logger


@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 40
    action_step: float = 0.04
    direct_position_action: bool = False
    # With direct_position_action, accept [u_x, u_y, q_x, q_y] directly in
    # metres. No tanh or normalized-position transform is applied.
    direct_position_physical_action: bool = False
    # Direct-position bandit reset: emit a constant state and do not evaluate
    # an irrelevant centre point before the actor chooses its location.
    stateless_direct_position: bool = False
    # When positive, threaded identical direct-position environments submit one
    # action each to a shared CUDA rate batcher.
    gpu_rate_batch_size: int = 0
    gpu_rate_device: str = "cuda"
    gpu_rate_backend: str = "local"
    gpu_rate_socket: str = ""
    # Represent a direct full position as one categorical joint action. With
    # 11 points per coordinate this is exactly the 11^4 AFDM-grid search
    # domain, while retaining a single joint categorical distribution instead
    # of four independent continuous Gaussians.
    discrete_position_action: bool = False
    # When positive, expose a broad raw action interval
    # [-action_tanh_bound, action_tanh_bound] and map it smoothly through
    # tanh before interpreting it as either a normalized position or a
    # normalized position increment. This keeps the physical action bounded
    # without a discontinuous clamp inside the environment.
    action_tanh_bound: float = 0.0
    # If >1, quantize each normalized direct-position action component to an
    # evenly spaced grid. This keeps training and evaluation on the same
    # finite 4-D action set (e.g. 11^4 = 14641 positions).
    action_grid_points: int = 0
    random_reset: bool = True
    seed: int = 7
    high_rate_threshold: float = 0.0
    high_rate_bonus: float = 0.0
    high_rate_slope: float = 1.0
    rate_baseline: float = 0.0
    # Coefficient of the dense instantaneous-rate term.  Keeping this separate
    # from reward_scale permits a 50-query search reward that remains dense
    # while its terminal objective is the best rate encountered.
    rate_weight: float = 1.0
    reward_scale: float = 1.0
    reward_mode: str = "dense"
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
    observe_csi: bool = False
    history_length: int = 0
    novelty_weight: float = 0.0
    query_log_path: str = ""
    query_log_flush_every: int = 128
    objective: str = "afdm_rate"
    randomize_channel_on_reset: bool = False
    # Fixed set of known channels: each entry is a channel seed; the env rotates
    # through them on reset (channel varies only in AoA/AoD/path gains/initial FA
    # position, waveform parameters stay fixed). Empty => use the single channel.
    channel_seeds: tuple[int, ...] = ()


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
    _QUERY_LOG_FIELDS = (
        "query", "episode_step", "rate", "best_rate", "objective", "best_objective",
        "u_x", "u_y", "q_x", "q_y", "best_u_x", "best_u_y", "best_q_x", "best_q_y",
    )

    def __init__(self, channel: FAAFDMChannel | None = None, cfg: EnvConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.channel = channel or make_default_channel()
        self._channel_idx = 0
        if self.cfg.channel_seeds:
            # Build a fixed list of known channels from the base channel's config.
            base_cfg = self.channel.cfg
            self._channels = [
                FAAFDMChannel(FAAFDMConfig(**{**dataclasses.asdict(base_cfg), "seed": s}))
                for s in self.cfg.channel_seeds
            ]
            self.channel = self._channels[0]
        else:
            self._channels = [self.channel]
        self.rng = np.random.default_rng(self.cfg.seed)
        self._rate_batcher = (
            shared_gpu_rate_batcher(
                self.channel,
                self.cfg.gpu_rate_batch_size,
                self.cfg.gpu_rate_device,
                self.cfg.gpu_rate_backend,
                self.cfg.gpu_rate_socket)
            if (self.cfg.gpu_rate_batch_size > 1
                and self.cfg.direct_position_action
                and self.cfg.direct_position_physical_action
                and self.cfg.stateless_direct_position)
            else None
        )

        if self.cfg.discrete_position_action:
            if not self.cfg.direct_position_action or self.cfg.action_grid_points < 2:
                raise ValueError("discrete_position_action requires direct_position_action and action_grid_points >= 2")
            self.action_space = spaces.Discrete(int(self.cfg.action_grid_points) ** 4)
        elif self.cfg.direct_position_action and self.cfg.direct_position_physical_action:
            tx_low, tx_high = self.channel.tx_bounds
            rx_low, rx_high = self.channel.rx_bounds
            self.action_space = spaces.Box(
                low=np.concatenate([tx_low, rx_low]).astype(np.float32),
                high=np.concatenate([tx_high, rx_high]).astype(np.float32),
                dtype=np.float32,
            )
        else:
            action_bound = float(self.cfg.action_tanh_bound)
            if action_bound > 0.0:
                self.action_space = spaces.Box(
                    low=-action_bound, high=action_bound, shape=(4,), dtype=np.float32
                )
            else:
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        obs_low = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0]
        obs_high = [1, 1, 1, 1, 1, 1, 1, 1, np.inf, 1]
        if self.cfg.observe_best_position:
            obs_low.extend([-1, -1, -1, -1])
            obs_high.extend([1, 1, 1, 1])
        if self.cfg.raw_rate_scale > 0.0:
            obs_low.extend([0, 0])
            obs_high.extend([np.inf, np.inf])
        if self.cfg.observe_rate_dynamics:
            obs_low.extend([-np.inf, 0, 0])
            obs_high.extend([np.inf, np.inf, np.inf])
        if self.cfg.observe_csi:
            n_csi = len(self.channel.csi_vector())
            obs_low.extend([-np.inf] * n_csi)
            obs_high.extend([np.inf] * n_csi)
        for _ in range(self.cfg.history_length):
            obs_low.extend([-1, -1, -1, -1, 0])
            obs_high.extend([1, 1, 1, 1, np.inf])
        obs_low = np.array(obs_low, dtype=np.float32)
        obs_high = np.array(obs_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.u = np.zeros(2, dtype=np.float32)
        self.q = np.zeros(2, dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.rate = 0.0
        self.afdm_rate = 0.0
        self.prev_rate = 0.0
        self.best_rate = 0.0
        self.best_u = np.zeros(2, dtype=np.float32)
        self.best_q = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.query_count = 0
        self._history: list[np.ndarray] = []
        self._visited_positions: list[np.ndarray] = []
        self._query_log = None
        self._query_log_pending = 0
        if self.cfg.query_log_path:
            self._query_log = _shared_query_log(self.cfg.query_log_path, self._QUERY_LOG_FIELDS)

    def close(self) -> None:
        if self._query_log is not None:
            # The file is shared by peer threaded environments. Flush but do
            # not close it when one peer closes before the others.
            self._query_log.flush()
        parent_close = getattr(super(), "close", None)
        if parent_close is not None:
            parent_close()

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
            parts.append(
                np.array(
                    [self.rate * self.cfg.raw_rate_scale, self.best_rate * self.cfg.raw_rate_scale],
                    dtype=np.float32,
                )
            )
        if self.cfg.observe_rate_dynamics:
            denom = max(1.0, self.best_rate)
            parts.append(
                np.array(
                    [
                        (self.rate - self.prev_rate) / denom,
                        max(0.0, self.best_rate - self.rate) / denom,
                        max(0.0, self.best_rate - self.prev_best_rate) / denom,
                    ],
                    dtype=np.float32,
                )
            )
        if self.cfg.observe_csi:
            parts.append(self.channel.csi_vector())
        if self.cfg.history_length:
            padding = [np.zeros(5, dtype=np.float32)] * max(0, self.cfg.history_length - len(self._history))
            parts.extend(padding + self._history[-self.cfg.history_length:])
        return np.concatenate(parts).astype(np.float32)

    def _normalized_joint_position(self) -> np.ndarray:
        return np.concatenate([
            self._normalize_pos(self.u, self.channel.tx_bounds),
            self._normalize_pos(self.q, self.channel.rx_bounds),
        ]).astype(np.float32)

    def _afdm_rate(self, u: np.ndarray, q: np.ndarray) -> float:
        if self._rate_batcher is not None:
            return self._rate_batcher.rate(u, q)
        return self.channel.rate(u, q)

    def _record_query(self) -> None:
        pos = self._normalized_joint_position()
        self._visited_positions.append(pos.copy())
        if self.cfg.history_length:
            item = np.concatenate([pos, [self.rate * self.cfg.raw_rate_scale]]).astype(np.float32)
            self._history.append(item)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        if self.cfg.channel_seeds:
            # Rotate through the fixed set of known channels (AoA/AoD/gains vary,
            # waveform parameters fixed). Channel changes on every reset.
            self.channel = self._channels[self._channel_idx % len(self._channels)]
            self._channel_idx += 1
        elif self.cfg.randomize_channel_on_reset:
            self.channel.resample_realization()
        if self.cfg.direct_position_action and self.cfg.stateless_direct_position:
            self.u = np.zeros(2, dtype=np.float32)
            self.q = np.zeros(2, dtype=np.float32)
        elif "u" in options and "q" in options:
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
        # A one-step direct-position bandit has no meaningful pre-action rate.
        # Its sole real query is made after the actor supplies a position.
        if self.cfg.direct_position_action and self.cfg.stateless_direct_position:
            self.rate = 0.0
            self.afdm_rate = 0.0
        # For other environments, ``objective()`` delegates to ``rate()``.
        # Compute it once so the initial observation contains the true state.
        elif self.cfg.objective == "afdm_rate":
            self.rate = self._afdm_rate(self.u, self.q)
            self.afdm_rate = self.rate
        else:
            self.rate = self.channel.objective(self.u, self.q, self.cfg.objective)
            self.afdm_rate = self._afdm_rate(self.u, self.q)
        self.prev_rate = self.rate
        self.best_rate = self.rate
        self.prev_best_rate = self.best_rate
        self.best_u = self.u.copy()
        self.best_q = self.q.copy()
        self.best_afdm_rate = self.afdm_rate
        self._history = []
        self._visited_positions = []
        if not (self.cfg.direct_position_action and self.cfg.stateless_direct_position):
            self._record_query()
        return self._obs(), self._info()

    def step(self, action: np.ndarray):
        if self.cfg.discrete_position_action:
            points = int(self.cfg.action_grid_points)
            index = int(np.clip(np.asarray(action).item(), 0, points**4 - 1))
            digits = np.asarray(np.unravel_index(index, (points,) * 4), dtype=np.int64)
            grid = np.linspace(-1.0, 1.0, points, dtype=np.float32)
            action = grid[digits]
        elif self.cfg.direct_position_action and self.cfg.direct_position_physical_action:
            tx_low, tx_high = self.channel.tx_bounds
            rx_low, rx_high = self.channel.rx_bounds
            low = np.concatenate([tx_low, rx_low])
            high = np.concatenate([tx_high, rx_high])
            # Safety against numerical overshoot; no coordinate transform.
            action = np.clip(np.asarray(action, dtype=np.float32), low, high)
        else:
            raw_action = np.asarray(action, dtype=np.float32)
            action_bound = float(self.cfg.action_tanh_bound)
            if action_bound > 0.0:
                # Keep the policy action in a broad Euclidean space, then map
                # it smoothly to a valid normalized position *or increment*.
                # In the multi-step task this avoids turning all large Gaussian
                # samples into the identical +/- maximum displacement.
                raw_action = np.clip(raw_action, -action_bound, action_bound)
                action = np.tanh(raw_action)
            else:
                action = np.clip(raw_action, -1.0, 1.0)
        if (not self.cfg.discrete_position_action and self.cfg.direct_position_action
                and not self.cfg.direct_position_physical_action
                and self.cfg.action_grid_points > 1):
            points = int(self.cfg.action_grid_points)
            grid = np.linspace(-1.0, 1.0, points, dtype=np.float32)
            action = grid[np.abs(action[..., None] - grid).argmin(axis=-1)]
        old_rate = self.rate
        old_best_rate = self.best_rate
        prev_action = self.last_action.copy()

        if self.cfg.direct_position_action:
            if self.cfg.direct_position_physical_action:
                self.u = np.asarray(action[:2], dtype=np.float32).copy()
                self.q = np.asarray(action[2:], dtype=np.float32).copy()
            else:
                self.u = self._denormalize_pos(action[:2], self.channel.tx_bounds)
                self.q = self._denormalize_pos(action[2:], self.channel.rx_bounds)
        else:
            tx_low, tx_high = self.channel.tx_bounds
            rx_low, rx_high = self.channel.rx_bounds
            self.u = np.clip(self.u + self.cfg.action_step * action[:2], tx_low, tx_high).astype(np.float32)
            self.q = np.clip(self.q + self.cfg.action_step * action[2:], rx_low, rx_high).astype(np.float32)

        if self.cfg.objective == "afdm_rate":
            self.rate = self._afdm_rate(self.u, self.q)
            self.afdm_rate = self.rate
        else:
            self.rate = self.channel.objective(self.u, self.q, self.cfg.objective)
            self.afdm_rate = self._afdm_rate(self.u, self.q)
        self.prev_rate = old_rate
        self.prev_best_rate = old_best_rate
        if self.rate > self.best_rate:
            self.best_rate = self.rate
            self.best_u = self.u.copy()
            self.best_q = self.q.copy()
            self.best_afdm_rate = self.afdm_rate
        self.last_action = action
        self.step_count += 1
        self.query_count += 1

        pos = self._normalized_joint_position()
        novelty = (
            min(1.0, min(float(np.linalg.norm(pos - old)) for old in self._visited_positions))
            if self._visited_positions else 1.0
        )

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
        if self.cfg.reward_mode == "best_so_far":
            # The undiscounted episode return telescopes exactly to
            # reward_scale * (B_T - B_0), where B_t is the best rate found by t.
            reward = float(self.cfg.reward_scale * best_improvement)
        elif self.cfg.reward_mode == "terminal":
            # Sparse terminal reward: only the final deployed position's rate
            # counts.  Encourages the policy to END at a high-rate position,
            # which is where planning (world-model look-ahead) helps most.
            reward = float(
                self.cfg.reward_scale * rate_objective
                if self.step_count >= self.cfg.max_steps
                else 0.0
            )
        elif self.cfg.reward_mode == "dense":
            reward = float(
                (
                    self.cfg.rate_weight * rate_objective
                    + late_rate_bonus
                    + terminal_rate_bonus
                    + self.cfg.improvement_weight * improvement
                    + self.cfg.best_improvement_weight * best_improvement
                    + self.cfg.novelty_weight * novelty
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
        elif self.cfg.reward_mode == "rate_diff":
            # r_t := eta_r * (R_{t+1} - R_base), where R_{t+1} is the rate at the
            # position reached by this action (self.rate after step()).
            reward = float(
                self.cfg.reward_scale * (self.rate - self.cfg.rate_baseline)
            )
        else:
            raise ValueError(f"Unknown reward mode: {self.cfg.reward_mode!r}")
        self._record_query()
        if self._query_log is not None:
            self._query_log.write(
                {
                    "query": 0,
                    "episode_step": int(self.step_count),
                    "rate": float(self.afdm_rate),
                    "best_rate": float(self.best_afdm_rate),
                    "objective": float(self.rate),
                    "best_objective": float(self.best_rate),
                    "u_x": float(self.u[0]),
                    "u_y": float(self.u[1]),
                    "q_x": float(self.q[0]),
                    "q_y": float(self.q[1]),
                    "best_u_x": float(self.best_u[0]),
                    "best_u_y": float(self.best_u[1]),
                    "best_q_x": float(self.best_q[0]),
                    "best_q_y": float(self.best_q[1]),
                }
            )
            self._query_log_pending += 1
            if self._query_log_pending >= max(1, self.cfg.query_log_flush_every):
                self._query_log.flush()
                self._query_log_pending = 0
        terminated = False
        truncated = self.step_count >= self.cfg.max_steps
        return self._obs(), reward, terminated, truncated, self._info()

    def _info(self) -> dict[str, float | list[float]]:
        return {
            "rate": float(self.afdm_rate),
            "best_rate": float(self.best_afdm_rate),
            "objective": float(self.rate),
            "best_objective": float(self.best_rate),
            "objective_name": self.cfg.objective,
            "u": self.u.astype(float).tolist(),
            "q": self.q.astype(float).tolist(),
            "best_u": self.best_u.astype(float).tolist(),
            "best_q": self.best_q.astype(float).tolist(),
            "step": int(self.step_count),
        }
