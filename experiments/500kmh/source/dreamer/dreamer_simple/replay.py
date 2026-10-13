"""Uniform/elite replay mixing using only rates observed in this run.

The elite branch samples high-rate *sequences*, not actions or positions. The
task actor still optimizes imagined returns through the learned world model.
"""

import heapq

import numpy as np
from dreamerv3 import embodied


class ObservedRateMix:
    def __init__(self, elite_fraction=0.02, elite_sample_fraction=0.5, seed=0,
                 score_mode="sequence_max", episode_steps=None):
        if not 0 < elite_fraction <= 1 or not 0 <= elite_sample_fraction <= 1:
            raise ValueError("Invalid elite replay fractions")
        if score_mode not in ("sequence_max", "complete_terminal"):
            raise ValueError(f"Invalid replay score mode: {score_mode}")
        if score_mode == "complete_terminal" and not (episode_steps and episode_steps > 0):
            raise ValueError("Complete-terminal replay needs episode_steps")
        self.elite_fraction = float(elite_fraction)
        self.elite_sample_fraction = float(elite_sample_fraction)
        self.score_mode = score_mode
        self.episode_steps = episode_steps
        self.rng = np.random.default_rng(seed)
        self.keys = []
        self.indices = {}
        self.scores = {}
        self.elite = []
        self.dirty = 0
        self.elite_draws = 0
        self.uniform_draws = 0

    def __setitem__(self, key, steps):
        rates = np.asarray([np.asarray(step["rate_target"]).item() for step in steps])
        if not np.isfinite(rates).all():
            raise ValueError("Replay sequence has a nonfinite observed rate")
        self.indices[key] = len(self.keys)
        self.keys.append(key)
        if self.score_mode == "complete_terminal":
            first = np.asarray([np.asarray(step["is_first"]).item() for step in steps], bool)
            terminal = np.asarray(
                [np.asarray(step["is_terminal"]).item() for step in steps], bool)
            complete = []
            for end in np.flatnonzero(terminal):
                starts = np.flatnonzero(first[:end + 1])
                if len(starts) and end - starts[-1] == self.episode_steps:
                    complete.append(float(rates[end]))
            self.scores[key] = max(complete, default=-np.inf)
        else:
            self.scores[key] = float(rates.max())
        self.dirty += 1

    def __delitem__(self, key):
        index = self.indices.pop(key)
        last = self.keys.pop()
        if index < len(self.keys):
            self.keys[index] = last
            self.indices[last] = index
        del self.scores[key]
        if key in self.elite:
            self.elite.remove(key)
            self.dirty = 256
        self.dirty += 1

    def __call__(self):
        if self.rng.random() < self.elite_sample_fraction:
            if self.dirty >= 256 or not self.elite:
                candidates = (self.keys if self.score_mode == "sequence_max" else
                              [key for key in self.keys if np.isfinite(self.scores[key])])
                count = max(1, round(len(candidates) * self.elite_fraction))
                self.elite = heapq.nlargest(
                    count, candidates, key=self.scores.__getitem__)
                self.dirty = 0
            if self.elite:
                self.elite_draws += 1
                return self.elite[self.rng.integers(len(self.elite))]
        self.uniform_draws += 1
        return self.keys[self.rng.integers(len(self.keys))]


def make_observed_rate_replay(config, directory, elite_fraction=0.02,
                              elite_sample_fraction=0.5,
                              score_mode="sequence_max"):
    return embodied.replay.Generic(
        length=config.batch_length,
        capacity=config.replay_size,
        remover=embodied.replay.selectors.Fifo(),
        sampler=ObservedRateMix(
            elite_fraction, elite_sample_fraction,
            score_mode=score_mode, episode_steps=config.afdm.episode_steps),
        limiter=embodied.replay.limiters.MinSize(1),
        directory=directory,
        online=config.replay_online,
        chunks=1024,
    )
