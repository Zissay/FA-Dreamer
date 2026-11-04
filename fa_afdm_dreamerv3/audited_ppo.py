"""Unmodified SB3 PPO updates with durable per-rollout diagnostics."""
import json
import numbers
from pathlib import Path
import torch
from stable_baselines3 import PPO

class AuditedPPO(PPO):
    def train(self):
        before = self.policy.action_net.weight.detach().clone()
        critic_only = self.num_timesteps <= getattr(self, 'critic_adaptation_until', 0)
        frozen = []
        if critic_only:
            if any(p.requires_grad for p in self.policy.features_extractor.parameters()):
                raise ValueError('Critic-only adaptation requires a parameter-free shared feature extractor')
            for name, p in self.policy.named_parameters():
                if name.startswith(('action_net.', 'mlp_extractor.policy_net.')):
                    frozen.append((p, p.requires_grad, p.detach().clone()))
                    p.grad = None
                    p.requires_grad_(False)
        try:
            super().train()
            if critic_only:
                assert all(torch.equal(p.detach(), old) for p, _, old in frozen), 'Actor changed during critic adaptation'
        finally:
            for p, required, _ in frozen:
                p.requires_grad_(required)
        # Diagnostic pass on already collected states. No environment queries,
        # parameter writes, optimizer calls or additional training.
        sample = next(self.rollout_buffer.get(batch_size=min(256, self.batch_size)))
        with torch.no_grad():
            dist = self.policy.get_distribution(sample.observations)
            d = dist.distribution
            std = d.stddev
            if hasattr(d, 'concentration1'):
                std = 2 * std  # unit Beta -> [-1, 1] commands
            values, logp, _ = self.policy.evaluate_actions(sample.observations, sample.actions)
            ratio = (logp-sample.old_log_prob).exp()
        # SB3's Gaussian log_std is an unused placeholder for BetaPolicy.
        # Report the actual bounded distribution width rather than exp(0)=1.
        self.logger.record('train/std',float(std.mean()))
        record = {k:float(v) for k,v in self.logger.name_to_value.items()
                  if k.startswith('train/') and isinstance(v,numbers.Real)}
        record.update(steps=int(self.num_timesteps),
                      critic_only_adaptation=critic_only,
                      action_std_mean=float(std.mean()),
                      head_weight_update_norm=float((self.policy.action_net.weight-before).norm()),
                      ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                      value_std=float(values.std()),
                      advantage_std=float(sample.advantages.std()))
        path = getattr(self, 'diagnostics_path', None)
        if path:
            with Path(path).open('a') as f:
                f.write(json.dumps(record)+'\n')
        frequency = getattr(self, 'checkpoint_freq', 0)
        if frequency and self.num_timesteps % frequency == 0:
            folder=Path(self.checkpoint_dir)
            folder.mkdir(parents=True,exist_ok=True)
            self.save(folder / f'ppo_{self.num_timesteps}')
            normalizer=self.get_vec_normalize_env()
            if normalizer is not None:
                normalizer.save(str(folder / f'vecnormalize_{self.num_timesteps}.pkl'))
