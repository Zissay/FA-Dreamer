"""A directly bounded physical-coordinate policy for Stable-Baselines PPO.

Unlike a Gaussian followed by environment-side clipping, this distribution has
support exactly on the antenna region.  Therefore the action executed by the
channel, its log probability, and the PPO likelihood ratio all refer to the
same physical coordinate.
"""

from __future__ import annotations

import math

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Beta, Categorical
from stable_baselines3.common.distributions import DiagGaussianDistribution, sum_independent_dims
from stable_baselines3.common.policies import ActorCriticPolicy


class TruncatedDiagGaussianDistribution(DiagGaussianDistribution):
    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        super().__init__(int(np.asarray(low).size))
        self._low_np = np.asarray(low, dtype=np.float32).reshape(-1)
        self._high_np = np.asarray(high, dtype=np.float32).reshape(-1)
        if self._low_np.shape != self._high_np.shape or np.any(self._low_np >= self._high_np):
            raise ValueError("Invalid physical action bounds")

    def _bounds(self, value: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return value.new_tensor(self._low_np), value.new_tensor(self._high_np)

    def _standard_limits(self) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        assert self.distribution is not None
        low, high = self._bounds(self.distribution.mean)
        std = self.distribution.scale
        return (low - self.distribution.mean) / std, (high - self.distribution.mean) / std, std

    def _normalizer(self) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        alpha, beta, std = self._standard_limits()
        z = (th.special.ndtr(beta) - th.special.ndtr(alpha)).clamp_min(1e-12)
        return alpha, beta, z

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert self.distribution is not None
        _, _, z = self._normalizer()
        return sum_independent_dims(self.distribution.log_prob(actions) - th.log(z))

    def entropy(self):
        # Entropy of a univariate truncated standard normal, summed over dims.
        alpha, beta, z = self._normalizer()
        normalizer = math.sqrt(2.0 * math.pi)
        phi_alpha = th.exp(-0.5 * alpha.square()) / normalizer
        phi_beta = th.exp(-0.5 * beta.square()) / normalizer
        _, _, std = self._standard_limits()
        entropy = th.log(std * math.sqrt(2.0 * math.pi * math.e) * z)
        entropy = entropy + (alpha * phi_alpha - beta * phi_beta) / (2.0 * z)
        return sum_independent_dims(entropy)

    def sample(self) -> th.Tensor:
        assert self.distribution is not None
        alpha, _, z = self._normalizer()
        u = (th.special.ndtr(alpha) + th.rand_like(self.distribution.mean) * z).clamp(1e-7, 1.0 - 1e-7)
        return self.distribution.mean + self.distribution.scale * self.distribution.icdf(u)

    def mode(self) -> th.Tensor:
        assert self.distribution is not None
        low, high = self._bounds(self.distribution.mean)
        return th.maximum(th.minimum(self.distribution.mean, high), low)


class TruncatedGaussianPolicy(ActorCriticPolicy):
    """SB3 MLP actor with a true truncated Gaussian in physical coordinates."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_dist = TruncatedDiagGaussianDistribution(
            np.asarray(self.action_space.low, dtype=np.float32),
            np.asarray(self.action_space.high, dtype=np.float32),
        )


class BetaDistribution(DiagGaussianDistribution):
    """Independent unimodal Beta laws, rescaled directly to Box coordinates."""

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        super().__init__(int(np.asarray(low).size))
        self._low_np = np.asarray(low, dtype=np.float32).reshape(-1)
        self._high_np = np.asarray(high, dtype=np.float32).reshape(-1)

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        # Two positive concentration parameters for every physical coordinate.
        net = nn.Linear(latent_dim, 2 * self.action_dim)
        ignored_log_std = nn.Parameter(th.zeros(self.action_dim), requires_grad=False)
        return net, ignored_log_std

    def proba_distribution(self, params: th.Tensor, _unused: th.Tensor):
        alpha_raw, beta_raw = th.chunk(params, 2, dim=-1)
        # alpha,beta > 1 makes the deterministic mode unique and interior.
        alpha = th.nn.functional.softplus(alpha_raw) + 1.1
        beta = th.nn.functional.softplus(beta_raw) + 1.1
        self.distribution = Beta(alpha, beta)
        return self

    def _bounds(self, value: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        low = value.new_tensor(self._low_np)
        high = value.new_tensor(self._high_np)
        return low, high, high - low

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert self.distribution is not None
        low, _, width = self._bounds(actions)
        unit = ((actions - low) / width).clamp(1e-7, 1.0 - 1e-7)
        return sum_independent_dims(self.distribution.log_prob(unit) - th.log(width))

    def entropy(self):
        assert self.distribution is not None
        _, _, width = self._bounds(self.distribution.concentration1)
        return sum_independent_dims(self.distribution.entropy() + th.log(width))

    def sample(self) -> th.Tensor:
        assert self.distribution is not None
        low, _, width = self._bounds(self.distribution.concentration1)
        return low + width * self.distribution.rsample()

    def mode(self) -> th.Tensor:
        assert self.distribution is not None
        low, _, width = self._bounds(self.distribution.concentration1)
        alpha, beta = self.distribution.concentration1, self.distribution.concentration0
        # SB3's deterministic Gaussian action is its mean.  Use the analogous
        # Beta mean rather than the mathematical mode: near concentration one,
        # the latter moves disproportionately toward a boundary and is not the
        # centre of the behaviour distribution used to collect PPO rollouts.
        return low + width * (alpha / (alpha + beta))


class BetaPolicy(ActorCriticPolicy):
    """SB3 actor whose action distribution has support exactly on the Box."""

    def __init__(self, *args, initial_concentration: float | None = None, **kwargs) -> None:
        lr_schedule = kwargs.get("lr_schedule", args[2] if len(args) > 2 else None)
        if lr_schedule is None:
            raise ValueError("BetaPolicy requires an SB3 learning-rate schedule")
        super().__init__(*args, **kwargs)
        self.action_dist = BetaDistribution(
            np.asarray(self.action_space.low, dtype=np.float32),
            np.asarray(self.action_space.high, dtype=np.float32),
        )
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            self.mlp_extractor.latent_dim_pi, self.log_std_init)
        if initial_concentration is not None:
            if initial_concentration <= 1.1:
                raise ValueError("initial_concentration must exceed 1.1")
            # Bounded analogue of Gaussian log_std_init: centred exploration
            # with an explicit and reproducible initial scale.
            nn.init.zeros_(self.action_net.weight)
            raw = float(np.log(np.expm1(initial_concentration - 1.1)))
            self.action_net.bias.data.fill_(raw)
        if self.ortho_init:
            self.action_net.apply(lambda module: self.init_weights(module, gain=0.01))
        # ``super`` built an optimizer for its temporary Gaussian output head.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


class MeanPrecisionBetaDistribution(BetaDistribution):
    """The same independent Beta family with separate centre/precision heads."""
    def proba_distribution(self, params: th.Tensor, _unused: th.Tensor):
        mean_logit, log_precision = th.chunk(params,2,dim=-1)
        centre = th.sigmoid(mean_logit)
        precision = th.exp(log_precision.clamp(-5.0,12.0))
        self.distribution = Beta(1.1+centre*precision,1.1+(1.0-centre)*precision)
        return self


class MeanPrecisionBetaPolicy(BetaPolicy):
    """Bounded PPO actor; both centre and concentration learn from PPO loss."""
    def __init__(self,*args,initial_concentration=2.0,**kwargs):
        super().__init__(*args,initial_concentration=None,**kwargs)
        if initial_concentration <= 1.1:
            raise ValueError('initial_concentration must exceed 1.1')
        self.action_dist=MeanPrecisionBetaDistribution(
            np.asarray(self.action_space.low),np.asarray(self.action_space.high))
        nn.init.zeros_(self.action_net.weight)
        with th.no_grad():
            self.action_net.bias[:self.action_dist.action_dim].zero_()
            self.action_net.bias[self.action_dist.action_dim:].fill_(math.log(2*(initial_concentration-1.1)))


class MixtureBetaDistribution(DiagGaussianDistribution):
    """A PPO-compatible mixture of independent, exactly bounded Beta laws.

    This is a policy distribution, not a candidate buffer: every action is
    sampled from the mixture and its PPO likelihood is the exact mixture
    density.  Multiple components let an on-policy actor retain several
    plausible regions of a multimodal continuous objective while rewards
    decide which component eventually receives probability mass.
    """

    def __init__(self, low: np.ndarray, high: np.ndarray, components: int) -> None:
        super().__init__(int(np.asarray(low).size))
        if components < 2:
            raise ValueError("MixtureBetaDistribution needs at least two components")
        self._low_np = np.asarray(low, dtype=np.float32).reshape(-1)
        self._high_np = np.asarray(high, dtype=np.float32).reshape(-1)
        self.components = int(components)
        self.mixture: Categorical | None = None
        self.component_distribution: Beta | None = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        # Mixture logits plus alpha/beta parameters for each component/action.
        net = nn.Linear(latent_dim, self.components * (1 + 2 * self.action_dim))
        ignored_log_std = nn.Parameter(th.zeros(self.action_dim), requires_grad=False)
        return net, ignored_log_std

    def proba_distribution(self, params: th.Tensor, _unused: th.Tensor):
        batch = params.shape[0]
        shaped = params.view(batch, self.components, 1 + 2 * self.action_dim)
        logits = shaped[..., 0]
        alpha_raw = shaped[..., 1:1 + self.action_dim]
        beta_raw = shaped[..., 1 + self.action_dim:]
        alpha = th.nn.functional.softplus(alpha_raw) + 1.1
        beta = th.nn.functional.softplus(beta_raw) + 1.1
        self.mixture = Categorical(logits=logits)
        self.component_distribution = Beta(alpha, beta)
        return self

    def _bounds(self, value: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        low = value.new_tensor(self._low_np)
        high = value.new_tensor(self._high_np)
        return low, high, high - low

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert self.mixture is not None and self.component_distribution is not None
        low, _, width = self._bounds(actions)
        unit = ((actions - low) / width).clamp(1e-7, 1.0 - 1e-7)
        component_log_prob = self.component_distribution.log_prob(unit.unsqueeze(1)).sum(dim=-1)
        component_log_prob = component_log_prob - th.log(width).sum()
        return th.logsumexp(self.mixture.logits + component_log_prob, dim=-1)

    def entropy(self):
        assert self.mixture is not None and self.component_distribution is not None
        # A differentiable component-entropy surrogate.  PPO's exact action
        # likelihood remains the mixture log-density above.
        _, _, width = self._bounds(self.component_distribution.concentration1)
        component_entropy = (self.component_distribution.entropy() + th.log(width)).sum(dim=-1)
        return self.mixture.entropy() + (self.mixture.probs * component_entropy).sum(dim=-1)

    def sample(self) -> th.Tensor:
        assert self.mixture is not None and self.component_distribution is not None
        index = self.mixture.sample()
        unit = self.component_distribution.rsample()
        chosen = unit.gather(
            1, index[:, None, None].expand(-1, 1, self.action_dim)
        ).squeeze(1)
        low, _, width = self._bounds(chosen)
        return low + width * chosen

    def mode(self) -> th.Tensor:
        assert self.mixture is not None and self.component_distribution is not None
        # Deployment chooses the most probable component and its Beta mean.
        index = self.mixture.probs.argmax(dim=-1)
        alpha = self.component_distribution.concentration1
        beta = self.component_distribution.concentration0
        unit_means = alpha / (alpha + beta)
        chosen = unit_means.gather(
            1, index[:, None, None].expand(-1, 1, self.action_dim)
        ).squeeze(1)
        low, _, width = self._bounds(chosen)
        return low + width * chosen


class MixtureBetaPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with a multimodal bounded action distribution."""

    def __init__(self, *args, mixture_components: int = 4, **kwargs) -> None:
        lr_schedule = kwargs.get("lr_schedule", args[2] if len(args) > 2 else None)
        if lr_schedule is None:
            raise ValueError("MixtureBetaPolicy requires an SB3 learning-rate schedule")
        self.mixture_components = int(mixture_components)
        super().__init__(*args, **kwargs)
        self.action_dist = MixtureBetaDistribution(
            np.asarray(self.action_space.low, dtype=np.float32),
            np.asarray(self.action_space.high, dtype=np.float32),
            self.mixture_components,
        )
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            self.mlp_extractor.latent_dim_pi, self.log_std_init)
        # Domain-only, low-discrepancy initial component centres.  They cover
        # the bounded action box but encode neither AFDM rates nor a location.
        nn.init.zeros_(self.action_net.weight)
        with th.no_grad():
            values = []
            concentration = 8.0
            for component in range(self.mixture_components):
                values.append(0.0)  # equal initial mixture logits
                for dim in range(self.action_space.shape[0]):
                    fraction = ((component + 1) * (2 * dim + 1) * 0.61803398875) % 1.0
                    unit_mean = 0.18 + 0.64 * fraction
                    alpha = concentration * unit_mean
                    values.append(float(np.log(np.expm1(alpha - 1.1))))
                for dim in range(self.action_space.shape[0]):
                    fraction = ((component + 1) * (2 * dim + 1) * 0.61803398875) % 1.0
                    unit_mean = 0.18 + 0.64 * fraction
                    beta = concentration * (1.0 - unit_mean)
                    values.append(float(np.log(np.expm1(beta - 1.1))))
            self.action_net.bias.copy_(th.tensor(values, dtype=self.action_net.bias.dtype))
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
