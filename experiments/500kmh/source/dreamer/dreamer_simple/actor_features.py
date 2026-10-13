"""Actor-only posterior features; world model and critic keep sampled states."""

import jax


def with_posterior_probabilities(state):
    """Expose categorical posterior probabilities without modifying RSSM state."""
    return {**state, "posterior_probabilities": jax.nn.softmax(state["logit"], -1)}
