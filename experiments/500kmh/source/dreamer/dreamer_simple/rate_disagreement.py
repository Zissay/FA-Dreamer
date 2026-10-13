"""Physical-rate ensemble uncertainty for the learned exploration actor.

Predictors learn only from this run's measured replay rates. Imagined reward
uses phase features alone, never measured labels, oracle calls or candidates.
"""
import jax
import jax.numpy as jnp
from dreamerv3 import jaxutils, nets, ninjax as nj


class PhaseRateDisagreement(nj.Module):
    def __init__(self, config):
        self.nets = [nets.MLP(
            (), layers=config.disag_head.layers, units=config.disag_head.units,
            act='silu', norm='none', dist='mse', outscale=0.1,
            name=f'rate_disag{i}') for i in range(config.disag_models)]
        self.opt = jaxutils.Optimizer(name='disag_opt', **config.expl_opt)

    def predictions(self, features):
        phase = features['vector'][..., 165:]
        assert phase.shape[-1] == 380
        return jnp.stack([10.0 * net(phase).mean() for net in self.nets])

    def uncertainty(self, features):
        variance = self.predictions(features).var(0)
        # Finite derivative when all members agree. Output units are bps/Hz.
        return jnp.sqrt(variance + 1e-8) - 1e-4

    def __call__(self, traj):
        # Reward for arriving at the next imagined physical state, aligned
        # with the extrinsic reward's [time+1] state and native critic score.
        return self.uncertainty(traj)[1:]

    def loss(self, data):
        pred = self.predictions({'vector': jax.lax.stop_gradient(data['vector'])})
        truth = jax.lax.stop_gradient(data['rate_target'].reshape(data['is_first'].shape))
        valid = (~data['is_first'].astype(bool)) & jnp.isfinite(truth)
        truth = jnp.where(valid, truth, 0.0)
        # Independent bootstrap masks diversify members without rating or
        # selecting actions. Reset-only labels are excluded from supervision.
        mask = jax.random.bernoulli(nj.rng(), 0.8, pred.shape) & valid[None]
        residual = jnp.where(mask, (pred - truth[None]) / 10.0, 0.0)
        axes = tuple(range(1, pred.ndim))
        loss = ((residual ** 2).sum(axes) / jnp.maximum(1, mask.sum(axes))).sum()
        count = jnp.maximum(1, valid.sum())
        mse = jnp.where(valid, (pred.mean(0) - truth) ** 2, 0.0).sum() / count
        return loss, dict(rate_ensemble_rmse=jnp.sqrt(mse),
                          rate_ensemble_observed_count=valid.sum(),
                          rate_ensemble_std_mean=jnp.where(valid, pred.std(0), 0).sum()/count)

    def train(self, data):
        metrics, stats = self.opt(self.nets, self.loss, data, has_aux=True)
        return {**metrics, **stats}
