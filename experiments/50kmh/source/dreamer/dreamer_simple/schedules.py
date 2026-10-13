"""Predeclared optimization schedules; no reward thresholds or oracle inputs."""

import jax.numpy as jnp


def linear_transition(step, begin, end, initial, final):
    fraction = jnp.clip((step - begin) / max(1, end - begin), 0.0, 1.0)
    value = initial + fraction * (final - initial)
    return jnp.where(step >= end, final, jnp.where(step <= begin, initial, value))


def actor_factor(step, config):
    return linear_transition(
        step, config.lr_decay_start, config.lr_decay_end, 1.0, config.lr_final_fraction
    )


def entropy_coefficient(step, config):
    return linear_transition(
        step,
        config.entropy_decay_start,
        config.entropy_decay_end,
        config.entropy_initial,
        config.entropy_final,
    )
