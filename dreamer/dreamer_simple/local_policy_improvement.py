"""Auxiliary one-step actor improvement from states already visited online.

Only the selection mask reads measured rate. The action is differentiated
through known kinematics and the learned rate head, never the true objective.
"""

import jax
import jax.numpy as jnp

from .kinematics import advance


def high_state_model_gain(actor_dist, rate_model, context, config):
    count = int(config.afdm.local_rate_start_count)
    vector = jax.lax.stop_gradient(context["vector"]).reshape(
        (-1, context["vector"].shape[-1]))
    deter = jax.lax.stop_gradient(context["deter"]).reshape(
        (-1, context["deter"].shape[-1]))
    measured = jax.lax.stop_gradient(context["rate_target"]).reshape(-1)
    nonterminal = ~context["is_terminal"].reshape(-1).astype(bool)
    count = min(count, vector.shape[0])
    scores = jnp.where(nonterminal, measured, -jnp.inf)
    _, indices = jax.lax.top_k(scores, count)
    selected = vector[indices]
    mask = nonterminal[indices] & (
        measured[indices] >= config.afdm.local_rate_threshold)
    # 'deter' is required for the MLP's leading-dimension convention, but
    # task_actor_vector_only makes its actual input solely the vector.
    action = actor_dist({"vector": selected, "deter": deter[indices]}).mean()
    next_vector = advance(
        selected, action, config.afdm.wavelength,
        config.afdm.action_step_m, config.afdm.position_scale_m,
        config.afdm.episode_steps, config.afdm.direct_position_action)
    old_rate = jax.lax.stop_gradient(rate_model(selected))
    new_rate = rate_model(next_vector)
    gain = new_rate - old_rate
    divisor = jnp.maximum(1, mask.sum())
    mean_gain = jnp.where(mask, gain, 0.).sum() / divisor
    loss = -config.afdm.local_rate_gain_scale * mean_gain
    metrics = {
        "local_rate_high_start_count": mask.sum(),
        "local_rate_predicted_gain": mean_gain,
        "local_rate_aux_loss": loss,
    }
    return loss, metrics
