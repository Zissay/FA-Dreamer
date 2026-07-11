from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    blocks: list[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        blocks.extend([nn.Linear(last, hidden), nn.SiLU()])
        last = hidden
    blocks.append(nn.Linear(last, out_dim))
    return nn.Sequential(*blocks)


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000) -> None:
        self.capacity = capacity
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.next_obs: list[np.ndarray] = []
        self.continues: list[float] = []

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        if len(self.obs) >= self.capacity:
            self.obs.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_obs.pop(0)
            self.continues.pop(0)
        self.obs.append(obs.astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.rewards.append(float(reward))
        self.next_obs.append(next_obs.astype(np.float32))
        self.continues.append(0.0 if done else 1.0)

    def __len__(self) -> int:
        return len(self.obs)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, len(self.obs), size=batch_size)
        return {
            "obs": torch.tensor(np.stack([self.obs[i] for i in idx]), dtype=torch.float32, device=DEVICE),
            "actions": torch.tensor(np.stack([self.actions[i] for i in idx]), dtype=torch.float32, device=DEVICE),
            "rewards": torch.tensor([self.rewards[i] for i in idx], dtype=torch.float32, device=DEVICE).unsqueeze(-1),
            "next_obs": torch.tensor(np.stack([self.next_obs[i] for i in idx]), dtype=torch.float32, device=DEVICE),
            "continues": torch.tensor([self.continues[i] for i in idx], dtype=torch.float32, device=DEVICE).unsqueeze(-1),
        }


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 16, deter_dim: int = 64) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.deter_dim = deter_dim

        self.sequence = nn.GRUCell(latent_dim + action_dim, deter_dim)
        self.prior = mlp(deter_dim, 128, 2 * latent_dim)
        self.posterior = mlp(deter_dim + obs_dim, 128, 2 * latent_dim)
        self.decoder = mlp(deter_dim + latent_dim, 128, obs_dim)
        self.reward_head = mlp(deter_dim + latent_dim, 128, 1)
        self.continue_head = mlp(deter_dim + latent_dim, 128, 1)

    @property
    def feature_dim(self) -> int:
        return self.deter_dim + self.latent_dim

    def init_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.deter_dim, device=DEVICE)
        z = torch.zeros(batch_size, self.latent_dim, device=DEVICE)
        return h, z

    def _dist(self, stats: torch.Tensor) -> Normal:
        mean, raw_std = stats.chunk(2, dim=-1)
        std = F.softplus(raw_std) + 0.1
        return Normal(mean, std)

    def observe(self, obs: torch.Tensor, prev_action: torch.Tensor, prev_state=None):
        if prev_state is None:
            prev_state = self.init_state(obs.shape[0])
        h_prev, z_prev = prev_state
        h = self.sequence(torch.cat([z_prev, prev_action], dim=-1), h_prev)
        prior = self._dist(self.prior(h))
        posterior = self._dist(self.posterior(torch.cat([h, obs], dim=-1)))
        z = posterior.rsample() if self.training else posterior.mean
        return h, z, prior, posterior

    def imagine_step(self, state: tuple[torch.Tensor, torch.Tensor], action: torch.Tensor):
        h, z = state
        next_h = self.sequence(torch.cat([z, action], dim=-1), h)
        prior = self._dist(self.prior(next_h))
        next_z = prior.rsample()
        feat = self.features(next_h, next_z)
        pred_obs = self.decoder(feat)
        pred_reward = self.reward_head(feat)
        pred_continue = torch.sigmoid(self.continue_head(feat))
        return (next_h, next_z), pred_obs, pred_reward, pred_continue

    def features(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z], dim=-1)


class Actor(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = mlp(feature_dim, 128, action_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(features))


class Critic(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = mlp(feature_dim, 128, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass
class AgentConfig:
    seed: int = 7
    warmup_steps: int = 800
    train_steps: int = 1200
    batch_size: int = 64
    imagination_horizon: int = 8
    gamma: float = 0.98
    world_lr: float = 1e-3
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    exploration_std: float = 0.25


class DreamerStyleAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: AgentConfig) -> None:
        self.cfg = cfg
        self.world = WorldModel(obs_dim, action_dim).to(DEVICE)
        self.actor = Actor(self.world.feature_dim, action_dim).to(DEVICE)
        self.critic = Critic(self.world.feature_dim).to(DEVICE)
        self.world_opt = torch.optim.Adam(self.world.parameters(), lr=cfg.world_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def act(self, obs: np.ndarray, explore: bool = False) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        zeros = torch.zeros(1, self.world.action_dim, device=DEVICE)
        self.world.eval()
        self.actor.eval()
        with torch.no_grad():
            h, z, _, _ = self.world.observe(obs_t, zeros, None)
            action = self.actor(self.world.features(h, z)).squeeze(0).cpu().numpy()
        if explore:
            action += np.random.normal(0.0, self.cfg.exploration_std, size=action.shape).astype(np.float32)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def train_world(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.world.train()
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        continues = batch["continues"]
        zero_action = torch.zeros_like(actions)

        h, z, prior, posterior = self.world.observe(obs, zero_action, None)
        _, pred_next_obs, pred_reward, pred_continue = self.world.imagine_step((h, z), actions)

        recon_loss = F.mse_loss(pred_next_obs, next_obs)
        reward_loss = F.mse_loss(pred_reward, rewards)
        continue_loss = F.binary_cross_entropy(pred_continue.clamp(1e-4, 1.0 - 1e-4), continues)
        kl = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1).mean()
        kl_loss = torch.maximum(kl, torch.tensor(1.0, device=DEVICE))
        loss = recon_loss + reward_loss + 0.2 * continue_loss + 0.01 * kl_loss

        self.world_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world.parameters(), 100.0)
        self.world_opt.step()
        return {
            "world": float(loss.detach().cpu()),
            "recon": float(recon_loss.detach().cpu()),
            "reward": float(reward_loss.detach().cpu()),
        }

    def train_actor_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        zero_action = torch.zeros_like(actions)
        self.world.eval()

        with torch.no_grad():
            h, z, _, _ = self.world.observe(obs, zero_action, None)

        state = (h.detach(), z.detach())
        rewards: list[torch.Tensor] = []
        feats: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(self.cfg.imagination_horizon):
                action = self.actor(self.world.features(*state))
                state, _, reward, _ = self.world.imagine_step(state, action)
                rewards.append(reward.detach())
                feats.append(self.world.features(*state).detach())

        values = [self.critic(feat) for feat in feats]
        bootstrap = self.critic(self.world.features(*state)).detach()
        returns: list[torch.Tensor] = []
        ret = bootstrap
        for reward in reversed(rewards):
            ret = reward + self.cfg.gamma * ret
            returns.append(ret)
        returns.reverse()

        critic_loss = torch.stack([F.mse_loss(value, target.detach()) for value, target in zip(values, returns)]).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_opt.step()

        state = (h.detach(), z.detach())
        actor_loss = torch.zeros((), device=DEVICE)
        action_penalty = torch.zeros((), device=DEVICE)
        discount = 1.0
        for _ in range(self.cfg.imagination_horizon):
            feat = self.world.features(*state)
            action = self.actor(feat)
            state, _, reward, _ = self.world.imagine_step(state, action)
            actor_loss = actor_loss - discount * reward.mean()
            action_penalty = action_penalty + action.pow(2).mean()
            discount *= self.cfg.gamma
        actor_loss = actor_loss / self.cfg.imagination_horizon + 0.001 * action_penalty

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()

        return {"actor": float(actor_loss.detach().cpu()), "critic": float(critic_loss.detach().cpu())}

    def save(self, path: str) -> None:
        torch.save(
            {
                "world": self.world.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=DEVICE)
        self.world.load_state_dict(checkpoint["world"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
