"""Predeclared optimization schedules; no reward thresholds or oracle inputs."""

import jax.numpy as jnp


def linear_transition(step, begin, end, initial, final):
    fraction = jnp.clip((step - begin) / max(1, end - begin), 0.0, 1.0)
    value = initial + fraction * (final - initial)
    return jnp.where(step >= end, final, jnp.where(step <= begin, initial, value))


def actor_factor(step, config):
    factor = linear_transition(
        step, config.lr_decay_start, config.lr_decay_end, 1.0, config.lr_final_fraction
    )
    # Optional second, clock-only consolidation schedule. It cannot inspect
    # rewards, checkpoints, high-rate positions, or the current actor.
    if getattr(config, "late_lr_decay_start", None) is not None:
        factor *= linear_transition(
            step,
            config.late_lr_decay_start,
            config.late_lr_decay_end,
            1.0,
            config.late_lr_final_fraction,
        )
    if getattr(config, "lr_compensate_warmup", False):
        warmup = max(1, int(config.lr_warmup_steps))
        peak = jnp.minimum(1.0, config.lr_decay_start / warmup)
        current = jnp.maximum(jnp.minimum(1.0, step / warmup), 1e-6)
        # The native optimizer already multiplies by current. Remove its
        # continuing increase after decay begins, so effective LR cannot
        # rebound while the actor is meant to consolidate a discovered route.
        factor *= jnp.minimum(1.0, peak / current)
    return factor


def entropy_coefficient(step, config):
    if hasattr(config, "entropy_mid"):
        first = linear_transition(
            step,
            config.entropy_decay_start,
            config.entropy_mid_step,
            config.entropy_initial,
            config.entropy_mid,
        )
        second = linear_transition(
            step,
            config.entropy_final_start,
            config.entropy_decay_end,
            config.entropy_mid,
            config.entropy_final,
        )
        return jnp.where(
            step <= config.entropy_mid_step,
            first,
            jnp.where(step < config.entropy_final_start, config.entropy_mid, second),
        )
    return linear_transition(
        step,
        config.entropy_decay_start,
        config.entropy_decay_end,
        config.entropy_initial,
        config.entropy_final,
    )


def replay_imitation_multiplier(step, config):
    """Clock-only scaling of actor imitation from its own completed replay."""
    final = float(config.replay_imitation_late_final_multiplier)
    if final == 1.0:
        return jnp.asarray(1.0)
    return linear_transition(
        step,
        int(config.replay_imitation_late_start_update),
        int(config.replay_imitation_late_end_update),
        1.0,
        final,
    )
