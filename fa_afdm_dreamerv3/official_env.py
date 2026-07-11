from __future__ import annotations

import numpy as np

try:
    import gym
    from gym import spaces
except Exception as exc:  # pragma: no cover - official DreamerV3 installs gym.
    raise ImportError(
        "Official dreamerv3==1.3.0 depends on gym==0.19.0. "
        "Install it with `pip install dreamerv3==1.3.0` in a Python version "
        "supported by that dependency set."
    ) from exc

from .channel import FAAFDMChannel, FAAFDMConfig
from .env import EnvConfig, FAAFDMEnv
from .position_splits import normalize_position_set


class DreamerV3FAAFDMGymEnv(gym.Env):
    """Old-Gym adapter for the official dreamerv3==1.3.0 FromGym wrapper.

    Official DreamerV3 1.3.0 expects Gym 0.19-style environments:
    - reset() -> obs
    - step(action) -> obs, reward, done, info

    Observations are returned as a dict with key "vector" so DreamerV3 routes
    them through its MLP encoder/decoder rather than image CNN paths.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        seed: int = 7,
        channel_seed: int | None = None,
        env_seed: int | None = None,
        n_subcarriers: int = 16,
        n_paths: int = 4,
        channel_memory: int = 5,
        noise_power_dbm: float | None = None,
        channel_gain_scale: float = 1.0,
        doppler_scale: float = 1.0,
        max_steps: int = 40,
        action_step: float = 0.04,
        direct_position_action: bool = False,
        initial_u: list[float] | None = None,
        initial_q: list[float] | None = None,
        initial_positions: list[dict] | None = None,
        high_rate_threshold: float = 0.0,
        high_rate_bonus: float = 0.0,
        high_rate_slope: float = 1.0,
        rate_baseline: float = 0.0,
        reward_scale: float = 1.0,
        raw_rate_scale: float = 0.0,
        improvement_weight: float = 0.5,
        best_improvement_weight: float = 0.0,
        movement_penalty: float = 0.01,
        action_l2_penalty: float = 0.0,
        regression_penalty: float = 0.0,
        best_gap_penalty: float = 0.0,
        high_rate_move_penalty: float = 0.0,
        high_rate_action_l2_penalty: float = 0.0,
        action_smooth_penalty: float = 0.0,
        high_rate_smooth_penalty: float = 0.0,
        boundary_margin: float = 1.0,
        boundary_free_dims: int = 0,
        boundary_penalty: float = 0.0,
        late_rate_weight: float = 0.0,
        late_rate_power: float = 1.0,
        terminal_rate_weight: float = 0.0,
        observe_best_position: bool = False,
        observe_rate_dynamics: bool = False,
        query_log_path: str = "",
        **_,
    ) -> None:
        super().__init__()
        channel_seed = seed if channel_seed is None else channel_seed
        env_seed = seed if env_seed is None else env_seed
        channel = FAAFDMChannel(
            FAAFDMConfig(
                n_subcarriers=n_subcarriers,
                n_paths=n_paths,
                channel_memory=channel_memory,
                noise_power_dbm=noise_power_dbm,
                channel_gain_scale=channel_gain_scale,
                doppler_scale=doppler_scale,
                seed=channel_seed,
            )
        )
        self._env = FAAFDMEnv(
            channel=channel,
            cfg=EnvConfig(
                max_steps=max_steps,
                action_step=action_step,
                direct_position_action=direct_position_action,
                random_reset=initial_positions is None and (initial_u is None or initial_q is None),
                seed=env_seed,
                high_rate_threshold=high_rate_threshold,
                high_rate_bonus=high_rate_bonus,
                high_rate_slope=high_rate_slope,
                rate_baseline=rate_baseline,
                reward_scale=reward_scale,
                raw_rate_scale=raw_rate_scale,
                improvement_weight=improvement_weight,
                best_improvement_weight=best_improvement_weight,
                movement_penalty=movement_penalty,
                action_l2_penalty=action_l2_penalty,
                regression_penalty=regression_penalty,
                best_gap_penalty=best_gap_penalty,
                high_rate_move_penalty=high_rate_move_penalty,
                high_rate_action_l2_penalty=high_rate_action_l2_penalty,
                action_smooth_penalty=action_smooth_penalty,
                high_rate_smooth_penalty=high_rate_smooth_penalty,
                boundary_margin=boundary_margin,
                boundary_free_dims=boundary_free_dims,
                boundary_penalty=boundary_penalty,
                late_rate_weight=late_rate_weight,
                late_rate_power=late_rate_power,
                terminal_rate_weight=terminal_rate_weight,
                observe_best_position=observe_best_position,
                observe_rate_dynamics=observe_rate_dynamics,
                query_log_path=query_log_path,
                initial_positions=None if initial_positions is None else normalize_position_set(initial_positions),
            ),
        )
        self._initial_u = None if initial_u is None else np.asarray(initial_u, dtype=np.float32)
        self._initial_q = None if initial_q is None else np.asarray(initial_q, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "vector": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self._env.observation_space.shape,
                    dtype=np.float32,
                )
            }
        )

    def reset(self):
        if self._initial_u is not None and self._initial_q is not None:
            obs, _ = self._env.reset(options={"u": self._initial_u, "q": self._initial_q})
        else:
            obs, _ = self._env.reset()
        return {"vector": obs.astype(np.float32)}

    def set_initial_position(self, u: list[float] | np.ndarray, q: list[float] | np.ndarray) -> None:
        self._initial_u = np.asarray(u, dtype=np.float32)
        self._initial_q = np.asarray(q, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        info = dict(info)
        info["is_terminal"] = bool(terminated)
        info["discount"] = 0.0 if terminated else 1.0
        return {"vector": obs.astype(np.float32)}, float(reward), done, info

    def render(self, mode="rgb_array"):
        return None

    def close(self):
        pass


def register_env(
    *,
    seed: int = 7,
    channel_seed: int | None = None,
    env_seed: int | None = None,
    n_subcarriers: int = 16,
    n_paths: int = 4,
    channel_memory: int = 5,
    noise_power_dbm: float | None = None,
    channel_gain_scale: float = 1.0,
    doppler_scale: float = 1.0,
    max_steps: int = 40,
    action_step: float = 0.04,
    direct_position_action: bool = False,
    initial_u: list[float] | None = None,
    initial_q: list[float] | None = None,
    initial_positions: list[dict] | None = None,
    high_rate_threshold: float = 0.0,
    high_rate_bonus: float = 0.0,
    high_rate_slope: float = 1.0,
    rate_baseline: float = 0.0,
    reward_scale: float = 1.0,
    raw_rate_scale: float = 0.0,
    improvement_weight: float = 0.5,
    best_improvement_weight: float = 0.0,
    movement_penalty: float = 0.01,
    action_l2_penalty: float = 0.0,
    regression_penalty: float = 0.0,
    best_gap_penalty: float = 0.0,
    high_rate_move_penalty: float = 0.0,
    high_rate_action_l2_penalty: float = 0.0,
    action_smooth_penalty: float = 0.0,
    high_rate_smooth_penalty: float = 0.0,
    boundary_margin: float = 1.0,
    boundary_free_dims: int = 0,
    boundary_penalty: float = 0.0,
    late_rate_weight: float = 0.0,
    late_rate_power: float = 1.0,
    terminal_rate_weight: float = 0.0,
    observe_best_position: bool = False,
    observe_rate_dynamics: bool = False,
    query_log_path: str = "",
) -> None:
    """Register the environment ID used by train_official_dreamerv3.py."""

    env_id = "FAAFDM-v0"
    registry = getattr(gym.envs.registration, "registry", {})
    registered = env_id in registry.env_specs if hasattr(registry, "env_specs") else env_id in registry
    if registered:
        if hasattr(registry, "env_specs"):
            registry.env_specs.pop(env_id, None)
        else:
            registry.pop(env_id, None)
    gym.envs.registration.register(
        id=env_id,
        entry_point="fa_afdm_dreamerv3.official_env:DreamerV3FAAFDMGymEnv",
        kwargs={
            "seed": seed,
            "channel_seed": channel_seed,
            "env_seed": env_seed,
            "n_subcarriers": n_subcarriers,
            "n_paths": n_paths,
            "channel_memory": channel_memory,
            "noise_power_dbm": noise_power_dbm,
            "channel_gain_scale": channel_gain_scale,
            "doppler_scale": doppler_scale,
            "max_steps": max_steps,
            "action_step": action_step,
            "direct_position_action": direct_position_action,
            "initial_u": initial_u,
            "initial_q": initial_q,
            "initial_positions": initial_positions,
            "high_rate_threshold": high_rate_threshold,
            "high_rate_bonus": high_rate_bonus,
            "high_rate_slope": high_rate_slope,
            "rate_baseline": rate_baseline,
            "reward_scale": reward_scale,
            "raw_rate_scale": raw_rate_scale,
            "improvement_weight": improvement_weight,
            "best_improvement_weight": best_improvement_weight,
            "movement_penalty": movement_penalty,
            "action_l2_penalty": action_l2_penalty,
            "regression_penalty": regression_penalty,
            "best_gap_penalty": best_gap_penalty,
            "high_rate_move_penalty": high_rate_move_penalty,
            "high_rate_action_l2_penalty": high_rate_action_l2_penalty,
            "action_smooth_penalty": action_smooth_penalty,
            "high_rate_smooth_penalty": high_rate_smooth_penalty,
            "boundary_margin": boundary_margin,
            "boundary_free_dims": boundary_free_dims,
            "boundary_penalty": boundary_penalty,
            "late_rate_weight": late_rate_weight,
            "late_rate_power": late_rate_power,
            "terminal_rate_weight": terminal_rate_weight,
            "observe_best_position": observe_best_position,
            "observe_rate_dynamics": observe_rate_dynamics,
            "query_log_path": query_log_path,
        },
    )
