"""Old-Gym bridge to the successful PPO task, without changing its physics.

The phase encoding is the NumPy equivalent of PPO's ChannelPhaseFeatures.
It consumes only CSI, current position and time, never a measured rate.
"""
from dataclasses import asdict, fields
from pathlib import Path
import csv
import threading

import gym
import numpy as np

from .channel import FAAFDMConfig
from .miso_style_afdm_env import MISOStyleAFDMConfig, MISOStyleFAAFDMEnv


def matched_configs(baseline, seed, amount=1, socket=""):
    channel = FAAFDMConfig(**{f.name: baseline[f.name] for f in fields(FAAFDMConfig)
                             if f.name in baseline and f.name != "seed"},
                          seed=baseline["channel_seed"])
    env = MISOStyleAFDMConfig(
        max_steps=baseline["episode_steps"], action_step=baseline["action_step"],
        direct_position_action=baseline["direct_position_action"],
        include_time_feature=True, finite_horizon_terminal=True,
        boundary_penalty=baseline["boundary_penalty"],
        terminal_rate_weight=baseline["terminal_rate_weight"],
        reward_mode=baseline["reward_mode"], strict_action_bounds=True,
        random_reset=False, seed=seed,
        gpu_rate_batch_size=amount if socket else 0,
        gpu_rate_backend="rpc" if socket else "local", gpu_rate_socket=socket)
    # Both matched tasks reset at the geometric centre. Incremental actions
    # are limited by action_step; direct actions map every policy output to a
    # valid absolute location in the 50-cm square and have no step limit.
    assert baseline["fixed_center_start"]
    assert not baseline["continuing_task"] and not baseline["train_start_pool"]
    assert baseline["dense_rate_coefficient"] == 0 and baseline["target_bonus"] == 0
    return channel, env


def phase_encode(obs, cfg):
    obs = np.asarray(obs, np.float32)
    p = cfg.n_paths
    pos = obs[-5:-1] * np.asarray([cfg.tx_region / 2] * 2 + [cfg.rx_region / 2] * 2, np.float32)
    tx, rx = obs[:2*p].reshape(p, 2), obs[2*p:4*p].reshape(p, 2)
    phase = (2 * np.pi / cfg.carrier_wavelength) * (
        (tx * pos[:2]).sum(-1) - (rx * pos[2:]).sum(-1)) + obs[5*p:6*p]
    i, j = np.triu_indices(p, 1)
    delta = phase[i] - phase[j]
    return np.concatenate([obs, np.sin(delta), np.cos(delta)]).astype(np.float32)


class TrainingLog:
    def __init__(self, directory, amount, budget):
        self.amount, self.budget, self.count, self.next_index = amount, budget, 0, 0
        self.lock = threading.Lock()
        self.source = "uniform_prefill"
        self.qfile = (Path(directory) / "training_queries.csv").open("x", newline="", buffering=1)
        self.efile = (Path(directory) / "episode_terminal_rates.csv").open("x", newline="", buffering=1)
        self.query = csv.writer(self.qfile)
        self.episode = csv.writer(self.efile)
        self.query.writerow(["steps", "env_index", "episode_step", "rate", "reward", "episode_terminal",
                             "u_x", "u_y", "q_x", "q_y", "a0", "a1", "a2", "a3", "source"])
        self.episode.writerow(["steps", "env_index", "terminal_rate", "episode_return", "initial_rate"])

    def allocate(self):
        with self.lock:
            index = self.next_index
            self.next_index += 1
            assert self.next_index <= self.amount
            return index

    def append(self, index, local_count, action, reward, info, done, total, initial):
        with self.lock:
            self.count += 1
            assert self.count <= self.budget, "Real action budget exceeded"
            step = (local_count - 1) * self.amount + index + 1
            pos = [*info["u"], *info["q"]]
            assert np.isfinite([*pos, info["rate"], reward]).all()
            assert np.max(np.abs(pos)) <= .25000001
            self.query.writerow([step, index, info["step"], info["rate"], reward, int(done),
                                 *pos, *action, self.source])
            if done:
                self.episode.writerow([local_count * self.amount, index, info["rate"], total, initial])

    def close(self):
        self.qfile.close()
        self.efile.close()


class MatchedDreamerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, channel, config, log=None, potential_model=False):
        self.core = MISOStyleFAAFDMEnv(channel, config)
        self.log = log
        self.index = log.allocate() if log else 0
        if log is not None:
            self.core.gpu_rate_slot = self.index
        self.local_count = 0
        self.potential_model = potential_model
        self.action_space = gym.spaces.Box(-1., 1., (4,), np.float32)
        dim = self.core.observation_space.shape[0] + channel.n_paths * (channel.n_paths - 1)
        self.observation_space = gym.spaces.Dict({"vector": gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32)})
        if potential_model:
            # Explicit encoder/decoder mlp_keys='vector' excludes this target.
            # Do not use log_*: official replay intentionally drops those keys.
            self.observation_space.spaces['rate_target']=gym.spaces.Box(-np.inf,np.inf,(),np.float32)

    def observation(self,obs):
        result={"vector":phase_encode(obs,self.core.channel.cfg)}
        if self.potential_model:
            result['rate_target']=np.float32(self.core.rate)
        return result

    def reset(self):
        obs, info = self.core.reset()
        self.total = 0.
        self.initial = info["rate"]
        return self.observation(obs)

    def step(self, action):
        obs, reward, terminal, truncated, info = self.core.step(action)
        assert not truncated
        self.local_count += 1
        self.total += reward
        if self.log:
            self.log.append(self.index, self.local_count, action, reward, info, terminal, self.total, self.initial)
        info["is_terminal"] = terminal
        return self.observation(obs), reward, terminal, info

    def close(self):
        self.core.close()
