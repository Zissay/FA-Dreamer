"""Advantage-weighted self-imitation of this run's completed episodes.

This is an optional hybrid actor auxiliary, not an unbiased off-policy policy
gradient or part of original DreamerV3. It never reads an oracle action.
Replay action[t] was chosen from observation[t] and produces observation[t+1].
"""

import jax
import jax.numpy as jnp


def complete_episode_weights(rate, first, terminal, episode_steps):
    """Return one positive-advantage complete route per replay sequence."""
    rate = jnp.asarray(rate).reshape(first.shape)
    first = jnp.asarray(first, bool)
    terminal = jnp.asarray(terminal, bool)
    batch, length = first.shape
    time = jnp.broadcast_to(jnp.arange(length)[None, :], (batch, length))
    starts = jax.lax.associative_scan(
        jnp.maximum, jnp.where(first, time, -length), axis=1)
    complete = terminal & (time - starts == episode_steps)
    end = jnp.argmax(jnp.where(complete, rate, -jnp.inf), axis=1)
    valid = complete.any(axis=1)
    selected_start = jnp.take_along_axis(starts, end[:, None], axis=1)[:, 0]
    selected_rate = jnp.take_along_axis(rate, end[:, None], axis=1)[:, 0]
    terminal_count = jnp.maximum(1, terminal.sum())
    baseline = jnp.where(terminal, rate, 0.).sum() / terminal_count
    variance = jnp.where(terminal, (rate - baseline) ** 2, 0.).sum() / terminal_count
    scale = jnp.maximum(1., jnp.sqrt(variance))
    weight = jnp.where(valid, jnp.clip((selected_rate - baseline) / scale, 0., 3.), 0.)
    # Index t is the pre-action state. The terminal at end was reached by
    # action[end - 1], so no action at the terminal row is reinforced.
    action_mask = valid[:, None] & (time >= selected_start[:, None]) & (time < end[:, None])
    return jax.lax.stop_gradient(weight), action_mask, baseline


def replay_actor_auxiliary(dist, action, rate, first, terminal, episode_steps):
    weight, mask, baseline = complete_episode_weights(
        rate, first, terminal, episode_steps)
    logprob = dist.log_prob(jax.lax.stop_gradient(action))
    weighted = jnp.where(mask, weight[:, None] * logprob, 0.)
    denom = jnp.maximum(1., mask.sum())
    loss = -weighted.sum() / denom
    metrics = {
        'replay_imitation_raw_loss': loss,
        'replay_imitation_baseline': baseline,
        'replay_imitation_complete_count': (mask.any(axis=1)).sum(),
        'replay_imitation_weighted_count': (weight > 0).sum(),
        'replay_imitation_action_count': mask.sum(),
    }
    return loss, metrics
