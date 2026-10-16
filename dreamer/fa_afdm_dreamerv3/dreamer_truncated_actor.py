"""Patch the bundled DreamerV3 distribution with a true bounded actor.

DreamerV3's config lists ``trunc_normal`` but version 1.3.0 does not build it.
This runtime-only patch adds a TFP truncated normal on [-1, 1]^d. The Gym
wrapper then linearly exposes that support as the native physical Box; no
environment-side action clipping is used for policy samples.
"""

from __future__ import annotations

import numpy as np


def install_truncated_normal_actor(stable_sampling: bool = False) -> None:
    from dreamerv3 import nets

    if getattr(nets.Dist, "_fa_afdm_trunc_normal_installed", False):
        if stable_sampling and not getattr(nets.Dist, "_fa_afdm_stable_trunc_sampling", False):
            raise RuntimeError("Install stable sampling before the legacy actor patch")
        return
    original_inner = nets.Dist.inner

    class StableTruncatedNormal(nets.tfd.TruncatedNormal):
        """Avoid erfinv(+/-1) and its infinite derivative in float32 sampling.

        Same inverse-CDF distribution, except the sub-float32 tail endpoint.
        The legacy JAX sampler only clips AFTER erfinv: its forward sample is
        finite but reverse-mode can then multiply zero by infinity -> NaN.
        """
        def _sample_n(self, n, seed=None):
            from tensorflow_probability.substrates.jax.internal import samplers
            loc, scale = self.loc, self.scale
            shape = (n,) + tuple(self.batch_shape)
            root2 = nets.jnp.sqrt(nets.jnp.float32(2))
            lower, upper = (self.low-loc)/scale, (self.high-loc)/scale
            a, b = nets.jax.lax.erf(lower/root2), nets.jax.lax.erf(upper/root2)
            uniform = nets.jax.random.uniform(samplers.sanitize_seed(seed), shape, dtype=nets.jnp.float32)
            u = a + (b-a)*uniform
            open_low = nets.jnp.nextafter(nets.jnp.float32(-1), nets.jnp.float32(0))
            open_high = nets.jnp.nextafter(nets.jnp.float32(1), nets.jnp.float32(0))
            u = nets.jnp.clip(u, open_low, open_high)
            z = root2 * nets.jax.lax.erf_inv(u)
            sample = loc + scale*z
            return nets.jnp.clip(sample, self.low, self.high)

    def inner(self, inputs):
        if self._dist != "trunc_normal":
            return original_inner(self, inputs)
        kw = {"outscale": self._outscale, "outnorm": self._outnorm}
        shape = self._shape
        out = self.get("out", nets.Linear, int(np.prod(shape)), **kw)(inputs)
        out = out.reshape(inputs.shape[:-1] + shape).astype(nets.f32)
        std = self.get("std", nets.Linear, int(np.prod(shape)), **kw)(inputs)
        std = std.reshape(inputs.shape[:-1] + shape).astype(nets.f32)
        lo, hi = self._minstd, self._maxstd
        std = (hi - lo) * nets.jax.nn.sigmoid(std + 2.0) + lo
        distribution = StableTruncatedNormal if stable_sampling else nets.tfd.TruncatedNormal
        base = distribution(
            low=-1.0,
            high=1.0,
            loc=nets.jnp.tanh(out),
            scale=std,
        )
        dist = nets.tfd.Independent(base, len(shape))
        # Dreamer's entropy normalization uses these two attributes.
        dist.minent = np.prod(shape) * nets.tfd.Normal(0.0, lo).entropy()
        dist.maxent = np.prod(shape) * nets.tfd.Normal(0.0, hi).entropy()
        return dist

    nets.Dist.inner = inner
    nets.Dist._fa_afdm_trunc_normal_installed = True
    nets.Dist._fa_afdm_stable_trunc_sampling = stable_sampling
    nets.Dist._fa_afdm_stable_distribution = StableTruncatedNormal
