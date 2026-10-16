"""Known position/observation dynamics only. Never evaluates the rate function."""

import jax.numpy as jnp
from dreamerv3 import jaxutils
from .actor_features import with_posterior_probabilities


def advance(vector, action, wavelength=0.0107068735,
            action_step_m=0.001, position_scale_m=0.25,
            episode_steps=50, direct_position_action=False):
    p = 20
    raw = vector[..., :165]
    if direct_position_action:
        pos = jnp.clip(action, -1.0, 1.0)
    else:
        pos = jnp.clip(
            raw[..., 160:164] + (action_step_m / position_scale_m) * action,
            -1.0, 1.0)
    time = jnp.minimum(1.0, raw[..., 164:165] + 1.0 / episode_steps)
    raw = jnp.concatenate([raw[..., :160], pos, time], axis=-1)
    physical = position_scale_m * pos
    tx = raw[..., :40].reshape(raw.shape[:-1] + (p, 2))
    rx = raw[..., 40:80].reshape(raw.shape[:-1] + (p, 2))
    phase = (2 * jnp.pi / wavelength) * (
        (tx * physical[..., None, :2]).sum(-1) - (rx * physical[..., None, 2:]).sum(-1)
    )
    phase += raw[..., 100:120]
    i, j = jnp.triu_indices(p, 1)
    delta = phase[..., i] - phase[..., j]
    return jnp.concatenate([raw, jnp.sin(delta), jnp.cos(delta)], axis=-1)


def rollout(wm, policy, start, horizon):
    keys = tuple(wm.rssm.initial(1).keys())
    first = {k: start[k] for k in keys}
    first["vector"] = start["vector"]
    if wm.config.afdm.actor_posterior_mean_input:
        first = with_posterior_probabilities(first)
    first["action"] = policy(first)

    def step(prev, _):
        obs = advance(
            prev["vector"], prev["action"], wm.config.afdm.wavelength,
            wm.config.afdm.action_step_m, wm.config.afdm.position_scale_m,
            wm.config.afdm.episode_steps,
            wm.config.afdm.direct_position_action)
        latent = {k: prev[k] for k in keys}
        post, _ = wm.rssm.obs_step(
            latent,
            prev["action"],
            wm.encoder({"vector": obs}),
            jnp.zeros(obs.shape[:-1], jnp.float32),
        )
        next_state = {**post, "vector": obs}
        if wm.config.afdm.actor_posterior_mean_input:
            next_state = with_posterior_probabilities(next_state)
        return {**next_state, "action": policy(next_state)}

    traj = jaxutils.scan(step, jnp.arange(horizon), first, wm.config.imag_unroll)
    return {k: jnp.concatenate([first[k][None], v], 0) for k, v in traj.items()}
