"""Explicit subclasses: state-rate world model, known clock, replay-value critic.

No runtime replacement of WorldModel/actor/critic methods. The sole package
compatibility shim adds its missing numerically-safe truncated distribution.
"""

import math
import jax
import jax.numpy as jnp
from dreamerv3 import agent as native, behaviors, jaxagent, jaxutils, nets, ninjax as nj
from .actor_features import with_posterior_probabilities

sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)


def actor_rollout_inputs(state, physical_feedback=False):
    """Optionally retain physical feedback for the vector-only full-return actor.

    Latent features and any training targets retain their stop-gradient. Only
    the measured-vector path through known kinematics can carry feedback.
    """
    inputs = sg(state)
    if physical_feedback:
        inputs = {**inputs, "vector": state["vector"]}
    return inputs


def rescale_actor_std(raw_std, minimum, maximum, cap):
    """Compress the learned standard-deviation range without a dead gradient."""
    fraction = (raw_std - minimum) / (maximum - minimum)
    scaled = minimum + (cap - minimum) * fraction
    return jnp.where(cap >= maximum, raw_std, scaled)


def truncated_normal_kl(policy, reference):
    """Exact per-state KL for independent, equally bounded truncated normals.

    Unlike evaluating log probabilities at stop-gradient actions, this has a
    nonzero gradient toward the snapshot policy when the policies differ.
    """
    current, old = policy.distribution, reference.distribution
    mu, sigma = current.loc, current.scale
    old_mu, old_sigma = sg(old.loc), sg(old.scale)
    root2pi = math.sqrt(2.0 * math.pi)

    def moments(loc, scale):
        lower = (current.low - loc) / scale
        upper = (current.high - loc) / scale
        normalizer = jax.scipy.special.ndtr(upper) - jax.scipy.special.ndtr(lower)
        normalizer = jnp.maximum(normalizer, 1e-7)
        lower_density = jnp.exp(-0.5 * lower**2) / root2pi
        upper_density = jnp.exp(-0.5 * upper**2) / root2pi
        shift = (lower_density - upper_density) / normalizer
        mean = loc + scale * shift
        variance = scale**2 * (1 + (lower * lower_density - upper * upper_density)
                               / normalizer - shift**2)
        return normalizer, mean, jnp.maximum(variance, 0.0)

    normalizer, mean, variance = moments(mu, sigma)
    old_normalizer, _, _ = moments(old_mu, old_sigma)
    kl_per_axis = (
        jnp.log(old_sigma / sigma)
        + jnp.log(old_normalizer / normalizer)
        + (variance + (mean - old_mu)**2) / (2 * old_sigma**2)
        - (variance + (mean - mu)**2) / (2 * sigma**2)
    )
    return kl_per_axis.sum(axis=-1)


def valid_imagination_indices(key, terminal, amount):
    """Uniform nonterminal subset with static shape and an empty-safe fallback."""
    terminal = terminal.astype(bool)
    count = terminal.shape[0]
    amount = min(int(amount), count)
    # Top-k iid uniform ranks are a uniform sample without replacement. Invalid
    # entries are selected only if fewer than `amount` valid starts exist; the
    # actor objective masks such entries completely.
    scores = jax.random.uniform(key, (count,), dtype=jnp.float32)
    scores = jnp.where(~terminal, scores, -jnp.inf)
    return jax.lax.top_k(scores, amount)[1]


def initial_imagination_indices(key, terminal, first, amount):
    """Select true episode starts before other valid replay states.

    Selection only uses reset and terminal flags. It never reads rewards,
    rates, actions, or positions. The valid-state fallback keeps the operation
    static and safe for small diagnostic batches.
    """
    terminal = terminal.astype(bool)
    first = first.astype(bool)
    count = terminal.shape[0]
    amount = min(int(amount), count)
    valid = ~terminal
    preferred = valid & first
    scores = jax.random.uniform(key, (count,), dtype=jnp.float32)
    scores += jnp.where(preferred, 2.0, jnp.where(valid, 0.0, -jnp.inf))
    return jax.lax.top_k(scores, amount)[1]


def repeated_initial_imagination_indices(key, terminal, first, amount):
    """Sample true episode starts with replacement, with a valid-state fallback.

    The sampler uses reset/terminal flags only. It never observes reward, rate,
    action, position, or any reference solution.
    """
    terminal = terminal.astype(bool)
    first = first.astype(bool)
    valid = ~terminal
    preferred = valid & first
    fallback = jnp.where(valid.any(), valid, jnp.ones_like(valid, dtype=bool))
    candidates = jnp.where(preferred.any(), preferred, fallback)
    logits = jnp.where(candidates, 0.0, -jnp.inf)
    return jax.random.categorical(
        key, logits, shape=(int(amount),)
    ).astype(jnp.int32)


def repeated_clock_zero_indices(key, terminal, vector, time_index, amount):
    """Draw actual physical episode starts, not replay-slice starts.

    Replay may set `is_first` at a sampled sequence boundary even when the
    physical clock is nonzero. The observed clock is part of the state and
    cannot reveal reward or an optimal action.
    """
    valid = ~terminal.astype(bool)
    at_start = jnp.abs(vector[..., time_index]) < 1e-6
    preferred = valid & at_start
    fallback = jnp.where(valid.any(), valid, jnp.ones_like(valid, dtype=bool))
    candidates = jnp.where(preferred.any(), preferred, fallback)
    logits = jnp.where(candidates, 0.0, -jnp.inf)
    return jax.random.categorical(
        key, logits, shape=(int(amount),)
    ).astype(jnp.int32)


def mixed_clock_zero_rate_indices(key, terminal, vector, rates, time_index,
                                  amount, initial_fraction, temperature):
    """Mix physical resets with high-rate states observed in this replay batch."""
    if not 0.0 <= float(initial_fraction) <= 1.0:
        raise ValueError('initial_fraction must be in [0, 1]')
    initial_count = int(round(int(amount) * float(initial_fraction)))
    other_count = int(amount) - initial_count
    initial_key, other_key = jax.random.split(key)
    initial = repeated_clock_zero_indices(
        initial_key, terminal, vector, time_index, initial_count)
    clock = vector[..., time_index]
    mid_episode = (clock > 1e-6) & (clock < 1 - 1e-6)
    valid = ~terminal.astype(bool)
    eligible = valid & mid_episode
    candidates = jnp.where(eligible.any(), eligible, valid)
    candidates = jnp.where(candidates.any(), candidates, jnp.ones_like(valid))
    values = jnp.where(candidates, rates, 0.0)
    count = jnp.maximum(1.0, candidates.sum())
    mean = values.sum() / count
    variance = jnp.where(candidates, (rates - mean) ** 2, 0.0).sum() / count
    standardized = (rates - mean) / jnp.maximum(jnp.sqrt(variance), 1e-4)
    logits = jnp.where(
        candidates,
        jnp.clip(standardized / max(float(temperature), 1e-4), -10.0, 10.0),
        -jnp.inf,
    )
    other = jax.random.categorical(
        other_key, logits, shape=(other_count,)).astype(jnp.int32)
    return jnp.concatenate([initial, other], axis=0)


def replay_return_to_go(reward, first, terminal):
    """Observed future return from each state, excluding its arrival reward."""
    carry = jnp.zeros(reward.shape[0], reward.dtype)
    result = []
    for time_index in reversed(range(reward.shape[1])):
        result.append(carry)
        carry = reward[:, time_index] + (1 - terminal[:, time_index]) * carry
        carry = jnp.where(first[:, time_index], 0.0, carry)
    return jnp.stack(result[::-1], axis=1)


def reward_prioritized_indices(key, terminal, returns, amount,
                               priority_fraction=0.5, temperature=1.0,
                               priority_valid=None):
    """Mix uniform valid starts with reward-prioritized replay starts."""
    terminal = terminal.reshape(-1).astype(bool)
    returns = returns.reshape(-1)
    amount = min(int(amount), terminal.shape[0])
    priority_count = int(round(amount * float(priority_fraction)))
    uniform_count = amount - priority_count
    key_uniform, key_priority = jax.random.split(key)
    uniform = valid_imagination_indices(key_uniform, terminal, uniform_count)
    valid = ~terminal
    if priority_valid is not None:
        eligible = valid & priority_valid.reshape(-1).astype(bool)
        valid = jnp.where(eligible.any(), eligible, valid)
        returns = jnp.where(eligible.any(), returns, jnp.zeros_like(returns))
    valid_values = jnp.where(valid, returns, 0.0)
    count = jnp.maximum(1.0, valid.sum())
    mean = valid_values.sum() / count
    variance = jnp.where(valid, (returns - mean) ** 2, 0.0).sum() / count
    standardized = (returns - mean) / jnp.maximum(jnp.sqrt(variance), 1e-4)
    logits = jnp.where(
        valid,
        jnp.clip(standardized / max(float(temperature), 1e-4), -10.0, 10.0),
        -jnp.inf,
    )
    priority = jax.random.categorical(
        key_priority, logits, shape=(priority_count,)
    ).astype(jnp.int32)
    return jnp.concatenate([uniform, priority], axis=0)


def prioritized_with_initial_indices(key, terminal, first, scores, amount,
                                     priority_fraction, temperature,
                                     initial_fraction, priority_valid=None,
                                     physical_vector=None, time_index=None):
    """Reserve actor-imagination starts for actual episode starts.

    The reserved states use only reset/terminal flags; the remaining states
    retain the existing uniform/high-observed-rate mix. No action label or
    reference optimum is supplied. Sampling starts with replacement because
    replay batches contain far fewer first frames than other states.
    """
    initial_fraction = float(initial_fraction)
    if not 0.0 <= initial_fraction <= 1.0:
        raise ValueError("initial_fraction must be in [0, 1]")
    initial_count = int(round(int(amount) * initial_fraction))
    if initial_count == 0:
        return reward_prioritized_indices(
            key, terminal, scores, amount, priority_fraction,
            temperature, priority_valid)
    key_initial, key_other = jax.random.split(key)
    if physical_vector is None:
        initial = repeated_initial_imagination_indices(
            key_initial, terminal.reshape(-1), first.reshape(-1), initial_count)
    else:
        initial = repeated_clock_zero_indices(
            key_initial, terminal.reshape(-1), physical_vector.reshape(
                (-1, physical_vector.shape[-1])), time_index, initial_count)
    remaining = reward_prioritized_indices(
        key_other, terminal, scores, int(amount) - initial_count,
        priority_fraction, temperature, priority_valid)
    return jnp.concatenate([initial, remaining], axis=0)


def completed_terminal_scores(rates, first, last, terminal):
    """Propagate only observed episode terminal rates within a replay slice.

    An unfinished suffix or a time-limit truncation is not an observed terminal
    outcome. Those states remain available to uniform sampling, but cannot be
    ranked as successful terminal trajectories.
    """
    rates = rates.reshape(first.shape)
    score = jnp.zeros_like(rates[:, 0])
    known = jnp.zeros_like(first[:, 0], dtype=bool)
    scores, masks = [], []
    for t in reversed(range(rates.shape[1])):
        stop = last[:, t].astype(bool) | terminal[:, t].astype(bool)
        score = jnp.where(stop, 0., score)
        known = jnp.where(stop, False, known)
        score = jnp.where(terminal[:, t], rates[:, t], score)
        known = known | terminal[:, t].astype(bool)
        scores.append(score)
        masks.append(known & ~terminal[:, t].astype(bool))
        score = jnp.where(first[:, t], 0., score)
        known = known & ~first[:, t].astype(bool)
    return jnp.stack(scores[::-1], 1), jnp.stack(masks[::-1], 1)


def potential_reward(rate, cont, terminal_weight):
    return (rate[1:] - rate[:-1]
            + terminal_weight * (1 - cont[1:]) * rate[1:])


def masked_percentile(values, mask, percentile):
    """Static-shape percentile excluding masked entries; finite when empty."""
    values = values.reshape(-1)
    mask = jnp.broadcast_to(mask, values.shape).reshape(-1).astype(bool)
    ordered = jnp.sort(jnp.where(mask, values, jnp.inf))
    count = mask.sum().astype(jnp.int32)
    position = (jnp.maximum(count, 1) - 1) * (percentile / 100.0)
    lower = jnp.floor(position).astype(jnp.int32)
    upper = jnp.ceil(position).astype(jnp.int32)
    fraction = position - lower
    result = ordered[lower] * (1 - fraction) + ordered[upper] * fraction
    return jnp.where(count > 0, result, jnp.asarray(0., values.dtype))


def masked_tensorstats(values, mask, prefix):
    """Finite scalar diagnostics containing no masked samples."""
    mask = jnp.broadcast_to(mask, values.shape).astype(bool)
    count = mask.sum().astype(jnp.float32)
    denom = jnp.maximum(1., count)
    safe = jnp.where(mask, values, 0.)
    mean = safe.sum() / denom
    variance = jnp.where(mask, (values - mean) ** 2, 0.).sum() / denom
    minimum = jnp.where(
        count > 0, jnp.where(mask, values, jnp.inf).min(),
        jnp.asarray(0., values.dtype))
    maximum = jnp.where(
        count > 0, jnp.where(mask, values, -jnp.inf).max(),
        jnp.asarray(0., values.dtype))
    magnitude = jnp.where(
        count > 0, jnp.where(mask, jnp.abs(values), 0.).max(),
        jnp.asarray(0., values.dtype))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": jnp.sqrt(jnp.maximum(0., variance)),
        f"{prefix}_mag": magnitude,
        f"{prefix}_min": minimum,
        f"{prefix}_max": maximum,
        f"{prefix}_count": count,
    }


def full_return_actor_objective(reward, cont, entropy, offset, invscale, coefficient):
    """Full remaining return, preserving the terminal transition reward."""
    # cont[t] says whether state t can execute a real transition. Thus cont[:-1]
    # includes the final valid action while excluding all post-terminal padding.
    active = cont[:-1].astype(bool)
    # Multiplication is unsafe here because 0 * NaN is NaN.
    safe_reward = jnp.where(active, reward, 0.)
    returns = safe_reward.sum(0)
    valid_start = cont[0].astype(bool)
    normalized = (returns - sg(offset)) / sg(invscale)
    entropy_sum = jnp.where(active, entropy, 0.).sum(0)
    entropy_count = active.sum(0)
    entropy_mean = entropy_sum / jnp.maximum(1, entropy_count)
    per_start = -normalized - coefficient * entropy_mean
    loss = jnp.where(valid_start, per_start, 0.).sum()
    loss /= jnp.maximum(1, valid_start.sum())
    return loss, returns, normalized, active, valid_start


def critic_bootstrap_actor_objective(reward, cont, value, step, offset, invscale):
    """Auxiliary actor objective: model prefix reward plus critic continuation.

    The critic predicts the *remaining* shaped return at the selected imagined
    state. Only starts still active at that state contribute; terminal padding
    never supplies a value target. The critic's parameters are updated by its
    own replay/imagined losses, while this objective differentiates through
    the imagined state to the actor.
    """
    valid = cont[step].astype(bool)
    prefix = jnp.where(cont[:step].astype(bool), reward[:step], 0.).sum(0)
    estimate = prefix + jnp.where(valid, value, 0.)
    normalized = (estimate - sg(offset)) / sg(invscale)
    loss = -jnp.where(valid, normalized, 0.).sum() / jnp.maximum(1, valid.sum())
    return loss, estimate, valid


def replay_returns(reward, first, last, terminal, value, discount=1.0, lam=0.95):
    carry = value[:, -1]
    targets = []
    for t in reversed(range(reward.shape[1] - 1)):
        disc = discount * (1 - terminal[:, t + 1])
        mix = lam * (1 - last[:, t + 1])
        carry = reward[:, t + 1] + disc * ((1 - mix) * value[:, t + 1] + mix * carry)
        carry = jnp.where(first[:, t + 1], value[:, t], carry)
        targets.append(carry)
    return jnp.stack(targets[::-1], 1), (1 - first[:, 1:]) * (1 - terminal[:, :-1])


class ObservedPhaseRateHead(nj.Module):
    """Learn a static fixed-channel rate from existing phase observations.

    No oracle rate computation, measured target, time or actor action enters
    the forward call. RSSM still drives the policy and critic. This optional
    head bypasses the stochastic latent bottleneck only for rate prediction.
    """
    def __init__(self, units):
        self.net = nets.MLP((), layers=2, units=units, act='silu', norm='none',
                            dist='mse', outscale=0.1, name='mlp')

    def __call__(self, features):
        phase = features['vector'][..., 165:]
        assert phase.shape[-1] == 380
        rate = 10.0 * self.net(phase).mean()
        return jaxutils.MSEDist(rate, 0, 'sum')


class WorldModel(native.WorldModel):
    def __init__(self, obs_space, act_space, config):
        super().__init__(obs_space, act_space, config)
        self.observed_rate_head = bool(config.afdm.observed_rate_head)
        if self.observed_rate_head:
            assert config.afdm.known_kinematics
            assert tuple(obs_space['vector'].shape) == (545,)
            self.heads['reward'] = ObservedPhaseRateHead(
                config.reward_head.units, name='observed_rate')

    def loss(self, data, state):
        target = dict(data, reward=data["rate_target"].reshape(data["reward"].shape))
        if not self.observed_rate_head:
            return super().loss(target, state)
        # Same native RSSM/decoder/continuation losses, with the observed vector
        # supplied only to the optional phase-rate head. Native features alone
        # do not carry the vector. Rate targets never enter encoder inputs.
        embed = self.encoder(target)
        previous, prev_action = state
        prev_actions = jnp.concatenate([prev_action[:, None], target['action'][:, :-1]], 1)
        post, prior = self.rssm.observe(embed, prev_actions, target['is_first'], previous)
        feats = {**post, 'embed': embed, 'vector': target['vector']}
        dists = {}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            dists.update(out if isinstance(out, dict) else {name: out})
        losses = {
            'dyn': self.rssm.dyn_loss(post, prior, **self.config.dyn_loss),
            'rep': self.rssm.rep_loss(post, prior, **self.config.rep_loss),
        }
        for key, dist in dists.items():
            loss = -dist.log_prob(target[key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2]
            # This is MSE in rate/10, matching the fixed-data feasibility test.
            losses[key] = loss / 100.0 if key == 'reward' else loss
        model_loss = sum(loss * self.scales[key] for key, loss in losses.items())
        outs = {'embed': embed, 'post': post, 'prior': prior}
        outs.update({f'{key}_loss': value for key, value in losses.items()})
        state = ({k: v[:, -1] for k, v in post.items()}, target['action'][:, -1])
        metrics = self._metrics(target, dists, post, prior, losses, model_loss)
        return model_loss.mean(), (state, outs, metrics)

    def imagine(self, policy, start, horizon):
        if self.config.afdm.known_kinematics:
            from .kinematics import rollout

            traj = rollout(self, policy, start, horizon)
        else:
            traj = super().imagine(policy, start, horizon)
        episode_steps = int(self.config.afdm.episode_steps)
        clock = jnp.rint(start["vector"][..., self.config.afdm.time_index] * episode_steps).astype(
            jnp.int32
        )
        cont = ((clock[None] + jnp.arange(horizon + 1)[:, None]) < episode_steps).astype(
            jnp.float32
        )
        cont *= 1 - start["is_terminal"][None]
        traj["cont"] = cont
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * cont, 0) / discount
        return traj

    def _metrics(self, data, dists, post, prior, losses, model_loss):
        out = super()._metrics(data, dists, post, prior, losses, model_loss)
        truth = sg(data["reward"])
        mask = 1 - data["is_first"]
        avg = lambda x: (x * mask).sum() / jnp.maximum(1.0, mask.sum())
        var = avg((truth - avg(truth)) ** 2)
        if self.observed_rate_head:
            mse = avg((sg(dists['reward'].mean()) - truth) ** 2)
            out['observed_rate_rmse'] = jnp.sqrt(mse)
            out['observed_rate_r2'] = 1 - mse / jnp.maximum(var, 1e-8)
            return out
        for key, pred in [
            ("posterior", dists["reward"].mean()),
            ("prior", self.heads["reward"](prior).mean()),
        ]:
            mse = avg((sg(pred) - truth) ** 2)
            out[f"{key}_reward_rmse"] = jnp.sqrt(mse)
            out[f"{key}_reward_r2"] = 1 - mse / jnp.maximum(var, 1e-8)
        return out


class ActorCritic(native.ImagActorCritic):
    def __init__(self, critics, scales, act_space, config, reward_fn=None,
                 rate_model=None, uncertainty_fn=None):
        super().__init__(critics, scales, act_space, config)
        self.reward_fn = reward_fn or critics["extr"].rewfn
        self.rate_model = rate_model
        self.uncertainty_fn = uncertainty_fn
        self.physical_feedback_gradient = bool(getattr(
            config.afdm, 'actor_physical_feedback_gradient', False))
        if self.physical_feedback_gradient and not (
                config.afdm.full_return_actor and config.afdm.known_kinematics
                and config.afdm.task_actor_vector_only):
            raise ValueError('Physical feedback gradient requires known kinematics '
                             'and a vector-only full-return actor')
        self.replay_imitation_scale = float(config.afdm.replay_imitation_scale)
        if self.replay_imitation_scale and not (
                config.afdm.full_return_actor and config.afdm.task_actor_vector_only):
            raise ValueError('Replay self-imitation requires the full-return vector-only actor')
        if config.afdm.local_rate_gain_scale:
            if not (config.afdm.known_kinematics and config.afdm.observed_rate_head
                    and config.afdm.task_actor_vector_only and rate_model is not None):
                raise ValueError("Local model gain requires known kinematics, "
                                 "a learned observed-rate head and vector-only actor")
        self.actor_kl_scale = float(config.afdm.actor_kl_scale)
        self.reference_actor = None
        self.reference_updater = None
        if self.actor_kl_scale:
            disc = act_space.discrete
            self.reference_actor = nets.MLP(
                name="actor_reference", dims="deter", shape=act_space.shape,
                **config.actor,
                dist=(config.actor_dist_disc if disc else config.actor_dist_cont),
            )
            # A hard, periodic policy snapshot is the Dreamer analogue of
            # PPO's fixed old policy during a short optimization epoch.  It
            # never reads rewards, positions, or reference solutions.
            self.reference_updater = jaxutils.SlowUpdater(
                self.actor, self.reference_actor, 1.0,
                int(config.afdm.actor_reference_period),
            )
        if config.afdm.schedule.enabled:
            import optax
            from .schedules import actor_factor

            # Preserve native Adam, gradient clipping and learning-rate warmup;
            # multiply the resulting parameter updates by a predeclared factor.
            self.opt.opt = optax.chain(
                self.opt.opt,
                optax.scale_by_schedule(
                    lambda step: actor_factor(step, config.afdm.schedule)
                ),
            )

    def _actor_std_cap(self):
        final = float(self.config.afdm.actor_std_cap_final)
        if final <= 0:
            return None
        from .schedules import linear_transition

        return linear_transition(
            self.opt.step.read(),
            int(self.config.afdm.actor_std_cap_start),
            int(self.config.afdm.actor_std_cap_end),
            float(self.config.actor.maxstd),
            final,
        )

    def _actor_dist(self, state, actor=None):
        """Actor distribution with an optional late upper-tail variance cap."""
        dist = (self.actor if actor is None else actor)(state)
        cap = self._actor_std_cap()
        if cap is None:
            return dist
        base = dist.distribution
        consistent = self.config.afdm.consistent_actor_distribution
        scale = (rescale_actor_std(
            base.scale, self.config.actor.minstd, self.config.actor.maxstd, cap)
            if consistent else jnp.minimum(base.scale, cap))
        bounded = type(base)(
            low=base.low,
            high=base.high,
            loc=base.loc,
            scale=scale,
        )
        capped = nets.tfd.Independent(bounded, len(self.act_space.shape))
        capped.minent = dist.minent
        capped.maxent = (math.prod(self.act_space.shape)
                         * nets.tfd.Normal(0.0, cap).entropy()
                         if consistent else dist.maxent)
        return capped

    def _loss_dist(self, state, actor=None):
        if self.config.afdm.consistent_actor_distribution:
            return self._actor_dist(state, actor)
        return (self.actor if actor is None else actor)(state)

    def _consistent_loss(self, traj):
        """Native Dreamer objective evaluated under the executed distribution."""
        metrics, advs = {}, []
        total = sum(self.scales[key] for key in self.critics)
        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self._actor_dist)
            offset, invscale = self.retnorms[key](ret)
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            advs.append((normed_ret - normed_base) * self.scales[key] / total)
            metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
            metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
            metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
            metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
        adv = jnp.stack(advs).sum(0)
        policy = self._actor_dist(sg(traj))
        logpi = policy.log_prob(sg(traj['action']))[:-1]
        loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj['weight'])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics

    def policy(self, state, carry):
        return {"action": self._actor_dist(state)}, carry

    def _masked_return_norm(self, returns, valid):
        """Update return normalization from valid nonterminal starts only."""
        norm = self.retnorms["extr"]
        if norm.impl == "off":
            return jnp.asarray(0., jnp.float32), jnp.asarray(1., jnp.float32)
        assert norm.impl == "perc_ema", (
            "full_return_actor requires the configured perc_ema retnorm", norm.impl)
        valid = valid.astype(bool)
        has_valid = valid.any()
        low = masked_percentile(returns, valid, norm.perclo)
        high = masked_percentile(returns, valid, norm.perchi)
        old_low, old_high = norm.low.read(), norm.high.read()
        norm.low.write(jnp.where(
            has_valid, norm.decay * old_low + (1 - norm.decay) * low, old_low))
        norm.high.write(jnp.where(
            has_valid, norm.decay * old_high + (1 - norm.decay) * high, old_high))
        return norm.stats()

    def loss(self, traj):
        if self.config.afdm.critic_bootstrap_actor:
            return self._critic_bootstrap_loss(traj)
        if self.config.afdm.full_return_actor:
            assert self.config.imag_horizon >= 50 and self.config.return_lambda == 1.
            # The main objective uses the complete imagined return. An optional
            # small critic continuation term is tested independently below.
            rew = self.reward_fn(traj)
            bonus_metrics = {}
            if getattr(self, 'uncertainty_fn', None) is not None:
                step = self.opt.step.read()
                begin = int(self.config.afdm.task_intrinsic_decay_start)
                end = int(self.config.afdm.task_intrinsic_decay_end)
                fraction = jnp.clip((step - begin) / max(1, end - begin), 0., 1.)
                coefficient = self.config.afdm.task_intrinsic_scale * (1. - fraction)
                uncertainty = self.uncertainty_fn(traj)
                active_bonus = traj['cont'][:-1].astype(bool)
                count = jnp.maximum(1., active_bonus.sum())
                raw = jnp.where(active_bonus, uncertainty, 0.)
                rms = sg(jnp.sqrt((raw ** 2).sum() / count))
                normalized_bonus = raw / jnp.maximum(rms, 0.01)
                rew = rew + coefficient * normalized_bonus
                bonus_metrics = dict(
                    full_return_intrinsic_coefficient=coefficient,
                    full_return_intrinsic_raw_mean=raw.sum() / count,
                    full_return_intrinsic_rms=rms,
                    full_return_intrinsic_applied_mean=(
                        coefficient * normalized_bonus).sum() / count)
            dist = self._loss_dist(sg(traj))
            ent = dist.entropy()[:-1]
            active = traj["cont"][:-1].astype(bool)
            provisional_returns = jnp.where(active, rew, 0.).sum(0)
            valid_start = traj["cont"][0].astype(bool)
            offset, invscale = self._masked_return_norm(
                sg(provisional_returns), valid_start)
            coefficient = self.config.actent
            if self.config.afdm.schedule.enabled:
                from .schedules import entropy_coefficient, actor_factor
                step = self.opt.step.read()
                coefficient = entropy_coefficient(step, self.config.afdm.schedule)
            loss, returns, normalized, active, valid_start = full_return_actor_objective(
                rew, traj["cont"], ent, offset, invscale, coefficient)
            loss *= self.config.loss_scales.actor
            critic_scale = float(getattr(self.config.afdm, 'actor_critic_loss_scale', 0.))
            critic_metrics = {}
            if critic_scale:
                bootstrap_step = int(self.config.afdm.actor_critic_bootstrap_step)
                if not 0 < bootstrap_step < self.config.imag_horizon:
                    raise ValueError('actor_critic_bootstrap_step must be inside imagination')
                features = {key: traj[key][bootstrap_step]
                            for key in self.config.critic.inputs}
                continuation = self.critics['extr'].net(features).mean()
                critic_loss, critic_estimate, critic_valid = (
                    critic_bootstrap_actor_objective(
                        rew, traj['cont'], continuation, bootstrap_step,
                        offset, invscale))
                step = self.opt.step.read()
                begin = int(self.config.afdm.actor_critic_start_update)
                end = int(self.config.afdm.actor_critic_ramp_end_update)
                ramp = jnp.clip((step - begin) / max(1, end - begin), 0., 1.)
                active_scale = critic_scale * ramp
                loss += self.config.loss_scales.actor * active_scale * critic_loss
                critic_metrics = dict(
                    actor_critic_aux_loss=critic_loss,
                    actor_critic_active_scale=active_scale,
                    actor_critic_valid_start_count=critic_valid.sum(),
                    actor_critic_estimated_return_mean=(
                        jnp.where(critic_valid, critic_estimate, 0.).sum()
                        / jnp.maximum(1, critic_valid.sum())),
                    actor_critic_continuation_mean=(
                        jnp.where(critic_valid, continuation, 0.).sum()
                        / jnp.maximum(1, critic_valid.sum())))
            logpi = dist.log_prob(sg(traj['action']))[:-1]
            randomness = (ent - dist.minent) / (dist.maxent - dist.minent)
            if len(randomness.shape) > 2:
                randomness = randomness.mean(range(2, len(randomness.shape)))
            metrics = dict(bonus_metrics, **critic_metrics)
            metrics.update(masked_tensorstats(
                traj['action'][:-1], active[..., None], "action"))
            metrics.update(masked_tensorstats(
                randomness, active, "policy_randomness"))
            metrics.update(masked_tensorstats(ent, active, "policy_entropy"))
            metrics.update(masked_tensorstats(logpi, active, "policy_logprob"))
            metrics.update(masked_tensorstats(
                normalized, valid_start, "adv"))
            metrics.update(masked_tensorstats(rew, active, "extr_reward"))
            return_stats = masked_tensorstats(returns, valid_start, "extr_return_raw")
            metrics.update(return_stats)
            metrics.update(masked_tensorstats(
                normalized, valid_start, "extr_return_normed"))
            metrics.update(
                full_return_start_mean=return_stats["extr_return_raw_mean"],
                full_return_valid_start_count=valid_start.sum(),
                full_return_active_transition_count=active.sum(),
                full_return_no_critic_objective=jnp.asarray(
                    0. if critic_scale else 1.))
            if self.reference_actor is not None:
                reference = self._actor_dist(sg(traj), self.reference_actor)
                sample_kl = truncated_normal_kl(dist, reference)[:-1]
                denom = jnp.maximum(1.0, active.sum())
                kl_mean = jnp.where(active, sample_kl, 0.0).sum() / denom
                kl_scale = self.actor_kl_scale * (
                    self.opt.step.read() >= self.config.afdm.actor_kl_start_update)
                loss += kl_scale * kl_mean
                metrics.update(masked_tensorstats(
                    sample_kl, active, "actor_reference_kl"))
                metrics["scheduled_actor_kl_scale"] = kl_scale
            cap = self._actor_std_cap()
            if cap is not None:
                metrics["scheduled_actor_std_cap"] = cap
            if self.config.afdm.schedule.enabled:
                factor = actor_factor(step, self.config.afdm.schedule)
                warmup = jnp.minimum(1., step / max(1, self.config.actor_opt.warmup))
                metrics.update(scheduled_entropy_coefficient=coefficient,
                               scheduled_actor_lr_factor=factor,
                               scheduled_actor_learning_rate=self.config.actor_opt.lr*warmup*factor)
            return loss, metrics
        loss, metrics = (self._consistent_loss(traj)
                         if self.config.afdm.consistent_actor_distribution
                         else super().loss(traj))
        if self.uncertainty_fn is not None:
            step = self.opt.step.read()
            begin = int(self.config.afdm.task_intrinsic_decay_start)
            end = int(self.config.afdm.task_intrinsic_decay_end)
            fraction = jnp.clip((step - begin) / max(1, end - begin), 0.0, 1.0)
            coefficient = self.config.afdm.task_intrinsic_scale * (1.0 - fraction)
            uncertainty = self.uncertainty_fn(traj)
            active = traj['cont'][1:].astype(bool)
            weight = sg(traj['weight'][1:])
            denom = jnp.maximum(1.0, active.sum())
            raw_mean = jnp.where(active, uncertainty, 0.0).sum() / denom
            # Ensemble disagreement is often only ~0.01 bps/Hz after fitting.
            # Normalize by its detached RMS as Dreamer's intrinsic critic does,
            # so the coefficient has a meaningful scale without using labels.
            rms = sg(jnp.sqrt(jnp.where(
                active, uncertainty**2, 0.0).sum() / denom))
            bonus = (jnp.where(active, uncertainty * weight, 0.0).sum()
                     / denom / jnp.maximum(rms, 0.01))
            loss -= coefficient * bonus
            metrics.update(task_intrinsic_coefficient=coefficient,
                           task_intrinsic_bonus_mean=bonus,
                           task_intrinsic_raw_mean=raw_mean,
                           task_intrinsic_rms=rms)
        cap = self._actor_std_cap()
        if cap is not None:
            metrics["scheduled_actor_std_cap"] = cap
        if self.reference_actor is not None:
            policy = self._loss_dist(sg(traj))
            reference = self._loss_dist(sg(traj), self.reference_actor)
            sample_kl = truncated_normal_kl(policy, reference)[:-1]
            active = traj["cont"][:-1].astype(bool)
            denom = jnp.maximum(1.0, active.sum())
            kl_mean = jnp.where(active, sample_kl, 0.0).sum() / denom
            kl_scale = self.actor_kl_scale * (
                self.opt.step.read() >= self.config.afdm.actor_kl_start_update)
            loss += kl_scale * kl_mean
            metrics.update(masked_tensorstats(
                sample_kl, active, "actor_reference_kl"))
            metrics["scheduled_actor_kl_scale"] = kl_scale
        if self.config.afdm.schedule.enabled:
            from .schedules import entropy_coefficient, actor_factor

            step = self.opt.step.read()
            coefficient = entropy_coefficient(step, self.config.afdm.schedule)
            ent = self._loss_dist(sg(traj)).entropy()[:-1]
            correction = (
                (coefficient - self.config.actent) * ent * sg(traj["weight"])[:-1]
            )
            loss -= self.config.loss_scales.actor * correction.mean()
            metrics["scheduled_entropy_coefficient"] = coefficient
            metrics["scheduled_actor_lr_factor"] = actor_factor(
                step, self.config.afdm.schedule
            )
            warmup = jnp.minimum(1.0, step / max(1, self.config.actor_opt.warmup))
            metrics["scheduled_actor_learning_rate"] = (
                self.config.actor_opt.lr * warmup * metrics["scheduled_actor_lr_factor"]
            )
        return loss, metrics

    def _critic_bootstrap_loss(self, traj):
        """One actor objective: imagined prefix reward plus critic tail value."""
        if self.config.afdm.full_return_actor:
            raise ValueError('Choose either full return or critic bootstrap, not both')
        if (self.config.afdm.task_intrinsic_scale or self.replay_imitation_scale
                or self.config.afdm.actor_critic_loss_scale or self.actor_kl_scale
                or self.config.afdm.local_rate_gain_scale):
            raise ValueError('Bootstrap pilot has one actor objective, no auxiliaries')
        reward = self.reward_fn(traj)
        cont = traj['cont']
        active = cont[:-1].astype(bool)
        valid = cont[0].astype(bool)
        prefix = jnp.where(active, reward, 0.).sum(0)
        final_features = {key: traj[key][-1] for key in self.config.critic.inputs}
        value = self.critics['extr'].net(final_features).mean()
        continuation = jnp.where(cont[-1].astype(bool), value, 0.)
        estimate = prefix + continuation
        offset, invscale = self._masked_return_norm(sg(estimate), valid)
        normalized = (estimate - sg(offset)) / sg(invscale)
        dist = self._loss_dist(sg(traj))
        entropy = dist.entropy()[:-1]
        entropy_mean = (jnp.where(active, entropy, 0.).sum(0)
                        / jnp.maximum(1, active.sum(0)))
        coefficient = self.config.actent
        step = self.opt.step.read()
        if self.config.afdm.schedule.enabled:
            from .schedules import entropy_coefficient, actor_factor
            coefficient = entropy_coefficient(step, self.config.afdm.schedule)
        per_start = -normalized - coefficient * entropy_mean
        loss = (jnp.where(valid, per_start, 0.).sum()
                / jnp.maximum(1, valid.sum())) * self.config.loss_scales.actor
        metrics = dict(
            bootstrap_actor_objective=loss,
            bootstrap_actor_prefix_return_mean=(
                jnp.where(valid, prefix, 0.).sum() / jnp.maximum(1, valid.sum())),
            bootstrap_actor_critic_tail_mean=(
                jnp.where(valid, continuation, 0.).sum() / jnp.maximum(1, valid.sum())),
            bootstrap_actor_valid_starts=valid.sum(),
            bootstrap_actor_terminal_at_horizon_fraction=(
                1. - cont[-1].astype(jnp.float32)).mean(),
            bootstrap_actor_return_mean=(
                jnp.where(valid, estimate, 0.).sum() / jnp.maximum(1, valid.sum())),
        )
        cap = self._actor_std_cap()
        if cap is not None:
            metrics['scheduled_actor_std_cap'] = cap
        if self.config.afdm.schedule.enabled:
            factor = actor_factor(step, self.config.afdm.schedule)
            warmup = jnp.minimum(1., step / max(1, self.config.actor_opt.warmup))
            metrics.update(
                scheduled_entropy_coefficient=coefficient,
                scheduled_actor_lr_factor=factor,
                scheduled_actor_learning_rate=(self.config.actor_opt.lr
                                               * warmup * factor))
        return loss, metrics

    def train(self, imagine, start, context):
        if self.config.afdm.actor_posterior_mean_input:
            start = with_posterior_probabilities(start)
        critic_source = start
        if self.reference_actor is not None:
            # Materialize both lazy MLPs before synchronizing the snapshot.
            self.actor(sg(start))
            self.reference_actor(sg(start))
            self.reference_updater()
        stride = int(self.config.afdm.imagination_start_stride)
        requested = int(self.config.afdm.imagination_start_count)
        selection_metrics = {}
        if self.config.afdm.imagination_start_mode in (
                'reward_prioritized', 'terminal_rate_prioritized',
                'completed_terminal_prioritized',
                'observed_state_rate_prioritized'):
            assert requested > 0
            returns = replay_return_to_go(
                context['reward'], context['is_first'], context['is_terminal'])
            # Potential rewards telescope as
            #   return_to_go + current_rate = 1.2 * terminal_rate.
            # Broadcasting this trajectory score to all preceding states makes
            # a rare successful route useful as a whole. It only uses rewards
            # and measured rates already present in replay; no position target,
            # exhaustive-search result, or oracle action enters the actor.
            scores = returns
            eligible = None
            if self.config.afdm.imagination_start_mode == 'terminal_rate_prioritized':
                rates = context['rate_target'].reshape(context['reward'].shape)
                scores = returns + rates
            elif self.config.afdm.imagination_start_mode == 'observed_state_rate_prioritized':
                # Prefer high-rate states actually visited in this run, even
                # when the episode subsequently ended poorly. This changes
                # imagination starts only; the actor still learns from the
                # model-based reward objective, without action labels.
                scores = context['rate_target'].reshape(context['reward'].shape)
            elif self.config.afdm.imagination_start_mode == 'completed_terminal_prioritized':
                scores, eligible = completed_terminal_scores(
                    context['rate_target'], context['is_first'],
                    context['is_last'], context['is_terminal'])
                selection_metrics['priority_completed_fraction'] = eligible.mean()
            physical_start = (
                self.config.afdm.imagination_start_mode == 'completed_terminal_prioritized'
                and self.config.afdm.physical_start_for_completed_priority)
            indices = prioritized_with_initial_indices(
                nj.rng(), start['is_terminal'], context['is_first'],
                scores, requested,
                self.config.afdm.priority_fraction,
                self.config.afdm.priority_temperature,
                self.config.afdm.initial_start_fraction,
                priority_valid=eligible,
                physical_vector=(context['vector'] if physical_start else None),
                time_index=(self.config.afdm.time_index if physical_start else None),
            )
            start = jax.tree_util.tree_map(lambda value: value[indices], start)
        elif self.config.afdm.imagination_start_mode == 'mixed_initial':
            assert requested > 0
            indices = repeated_initial_imagination_indices(
                nj.rng(), start['is_terminal'], start['is_first'], requested)
            extra = jax.tree_util.tree_map(lambda value: value[indices], start)
            start = jax.tree_util.tree_map(
                lambda value, addition: jnp.concatenate([value, addition], axis=0),
                start,
                extra,
            )
        elif requested > 0 or stride > 1:
            # Selection uses reset/terminal flags only, never rate or reward.
            count = start['is_terminal'].shape[0]
            amount = requested if requested > 0 else (count + stride - 1) // stride
            if self.config.afdm.imagination_start_mode == 'initial':
                indices = initial_imagination_indices(
                    nj.rng(), start['is_terminal'], start['is_first'], amount)
            elif self.config.afdm.imagination_start_mode == 'initial_repeated':
                indices = repeated_initial_imagination_indices(
                    nj.rng(), start['is_terminal'], start['is_first'], amount)
            elif self.config.afdm.imagination_start_mode == 'clock_zero_repeated':
                indices = repeated_clock_zero_indices(
                    nj.rng(), start['is_terminal'], start['vector'],
                    self.config.afdm.time_index, amount)
            elif self.config.afdm.imagination_start_mode == 'clock_zero_rate_mixed':
                rates = context['rate_target'].reshape(-1)
                indices = mixed_clock_zero_rate_indices(
                    nj.rng(), start['is_terminal'], start['vector'], rates,
                    self.config.afdm.time_index, amount,
                    self.config.afdm.initial_start_fraction,
                    self.config.afdm.priority_temperature)
                initial_count = int(round(
                    amount * float(self.config.afdm.initial_start_fraction)))
                selection_metrics['imagination_priority_start_rate_mean'] = (
                    rates[indices[initial_count:]].mean())
            else:
                assert self.config.afdm.imagination_start_mode == 'uniform'
                indices = valid_imagination_indices(
                    nj.rng(), start['is_terminal'], amount)
            start = jax.tree_util.tree_map(lambda value: value[indices], start)

        def actor_loss(start):
            policy = lambda s: self._actor_dist(actor_rollout_inputs(
                s, self.physical_feedback_gradient)).sample(seed=nj.rng())
            traj = imagine(policy, start, self.config.imag_horizon)
            loss, metrics = self.loss(traj)
            if self.config.afdm.local_rate_gain_scale:
                from .local_policy_improvement import high_state_model_gain
                local_loss, local_metrics = high_state_model_gain(
                    self._actor_dist, self.rate_model, context, self.config)
                loss += local_loss
                metrics.update(local_metrics)
            if self.replay_imitation_scale:
                from .replay_imitation import replay_actor_auxiliary
                # The MLP uses deter for its leading-dimension convention,
                # although actor.inputs is vector-only in this configuration.
                replay_dist = self._actor_dist(sg(context))
                aux_loss, aux_metrics = replay_actor_auxiliary(
                    replay_dist, context['action'], context['rate_target'],
                    context['is_first'], context['is_terminal'],
                    int(self.config.afdm.episode_steps))
                step = self.opt.step.read()
                from .schedules import replay_imitation_multiplier
                late_multiplier = replay_imitation_multiplier(step, self.config.afdm)
                active_scale = self.replay_imitation_scale * late_multiplier * (
                    step >= self.config.afdm.replay_imitation_start_update)
                loss += active_scale * aux_loss
                metrics.update(aux_metrics)
                metrics['replay_imitation_active_scale'] = active_scale
            metrics['imagination_start_count'] = jnp.asarray(start['is_terminal'].shape[0])
            metrics['imagination_initial_start_fraction'] = start['is_first'].mean()
            metrics['imagination_physical_start_fraction'] = (
                jnp.abs(start['vector'][..., self.config.afdm.time_index]) < 1e-6
            ).mean()
            metrics['imagination_transition_count'] = jnp.asarray(
                start['is_terminal'].shape[0] * self.config.imag_horizon)
            metrics['actor_physical_feedback_gradient'] = jnp.asarray(
                float(self.physical_feedback_gradient))
            return loss, (traj, metrics)

        mets, (traj, metrics) = self.opt(self.actor, actor_loss, start, has_aux=True)
        metrics.update(selection_metrics)
        metrics.update(mets)
        critic = self.critics["extr"]
        target = sg(critic.score(traj)[1])

        # Fit V on current-policy continuations from real replay states near
        # the intended bootstrap clock. The existing actor rollout prioritizes
        # reset and high-current-rate states; it can miss low-rate bridge states
        # that still lead to a high terminal rate. This adds no true-rate query
        # and never uses historical best positions or action labels.
        mc_scale = float(getattr(self.config.afdm, 'critic_mc_distill_scale', 0.))
        mc_features = mc_target = None
        mc_metrics = {}
        if mc_scale:
            count = int(self.config.afdm.critic_mc_distill_count)
            repeats = int(self.config.afdm.critic_mc_distill_repeats)
            clock_step = int(self.config.afdm.critic_mc_distill_step)
            if not (0 < count <= critic_source['is_terminal'].shape[0]
                    and repeats > 0 and 0 < clock_step < self.config.afdm.episode_steps):
                raise ValueError('Invalid critic MC distillation count/repeats/step')
            clock = critic_source['vector'][..., self.config.afdm.time_index]
            distance = jnp.abs(clock - clock_step / self.config.afdm.episode_steps)
            jitter = 1e-4 * jax.random.uniform(nj.rng(), distance.shape)
            score = jnp.where(~critic_source['is_terminal'].astype(bool),
                              -distance + jitter, -jnp.inf)
            indices = jax.lax.top_k(score, count)[1]
            mc_start = jax.tree_util.tree_map(lambda value: value[indices], critic_source)
            mc_repeated = jax.tree_util.tree_map(
                lambda value: jnp.repeat(value, repeats, axis=0), mc_start)
            mc_policy = lambda s: self._actor_dist(sg(s)).sample(seed=nj.rng())
            mc_traj = imagine(mc_policy, mc_repeated, self.config.imag_horizon)
            mc_reward = self.reward_fn(mc_traj)
            mc_active = mc_traj['cont'][:-1].astype(bool)
            mc_returns = jnp.where(mc_active, mc_reward, 0.).sum(0)
            mc_target = sg(mc_returns.reshape(count, repeats).mean(1))
            mc_features = {key: sg(mc_start[key])
                           for key in self.config.critic.inputs}
            mc_metrics = dict(
                critic_mc_distill_start_clock_mean=clock[indices].mean(),
                critic_mc_distill_terminal_fraction=(
                    1. - mc_traj['cont'][-1]).mean(),
                critic_mc_distill_target_mean=mc_target.mean(),
            )

        def joint_loss(traj, target, context, mc_features, mc_target):
            loss, m = critic.loss(traj, target)
            inputs = (("deter", "stoch", "vector")
                      if self.config.afdm.critic_observed_vector
                      else ("deter", "stoch"))
            features = {k: sg(context[k]) for k in inputs}
            value = sg(critic.slow(features).mean())
            returns, mask = replay_returns(
                context["reward"],
                context["is_first"],
                context["is_last"],
                context["is_terminal"],
                value,
                1 - 1 / self.config.horizon,
                self.config.afdm.replay_return_lambda,
            )
            dist = critic.net({k: v[:, :-1] for k, v in features.items()})
            denom = jnp.maximum(1.0, mask.sum())
            replay_loss = (-dist.log_prob(sg(returns)) * mask).sum() / denom
            if mc_target is not None:
                mc_prediction = critic.net(mc_features).mean()
                mc_loss = jnp.mean((mc_prediction - sg(mc_target)) ** 2)
                loss += mc_scale * mc_loss
                m['critic_mc_distill_loss'] = mc_loss
                m['critic_mc_distill_rmse'] = jnp.sqrt(mc_loss)
            pred = dist.mean()
            m.update(
                replay_value_loss=replay_loss,
                replay_value_mean=(pred * mask).sum() / denom,
                replay_target_mean=(returns * mask).sum() / denom,
                replay_value_rmse=jnp.sqrt(
                    ((pred - returns) ** 2 * mask).sum() / denom
                ),
            )
            terminal = mask * context["is_terminal"][:, 1:]
            m["real_terminal_value_rmse"] = jnp.sqrt(
                ((pred - context["reward"][:, 1:]) ** 2 * terminal).sum()
                / jnp.maximum(1.0, terminal.sum())
            )
            return loss + self.config.afdm.replay_value_scale * replay_loss, m

        mets, cm = critic.opt(
            critic.net, joint_loss, traj, target, context,
            mc_features, mc_target, has_aux=True
        )
        cm.update(mets)
        critic.updater()
        metrics.update({f"extr_critic_{k}": v for k, v in cm.items()})
        metrics.update(mc_metrics)
        return traj, metrics


class Behavior(nj.Module):
    def __init__(self, wm, act_space, config):
        if config.afdm.task_actor_vector_only:
            if not config.afdm.actor_observed_vector:
                raise ValueError('Vector-only task actor requires observed vector policy path')
            # Task actor only: exploration, RSSM and critics retain their inputs.
            # Both online and imagined states contain the same physical vector.
            config = config.update({'actor.inputs': ['vector']})
        def reward(traj):
            rate = wm.heads["reward"](traj).mean()
            return potential_reward(rate, traj['cont'], config.afdm.terminal_weight)

        critic = native.VFunction(reward, config, name="critic")
        self.uncertainty = None
        if config.afdm.task_intrinsic_scale:
            from .rate_disagreement import PhaseRateDisagreement
            self.uncertainty = PhaseRateDisagreement(config, name="task_uncertainty")
        self.ac = ActorCritic(
            {"extr": critic}, {"extr": 1.0}, act_space, config,
            reward_fn=reward,
            rate_model=lambda vector: wm.heads["reward"]({"vector": vector}).mean(),
            uncertainty_fn=self.uncertainty,
            name="ac")

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        extra = {}
        if self.uncertainty is not None:
            extra = {f"task_uncertainty_{key}": value
                     for key, value in self.uncertainty.train(data).items()}
        traj, metrics = self.ac.train(imagine, start, data)
        return traj, {**metrics, **extra}

    def report(self, data):
        return {}


class TaskAlignedExplore(behaviors.Explore):
    """Dreamer exploration actor with task reward and selectable uncertainty."""

    def __init__(self, wm, act_space, config):
        self.config = config
        signal = config.afdm.exploration_signal
        if signal not in ('latent_disagreement', 'phase_rate_disagreement'):
            raise ValueError(f'Unknown exploration signal: {signal}')
        self.rewards = {}
        critics = {}
        for key, scale in config.expl_rewards.items():
            if not scale:
                continue
            if key == 'extr':
                def rewfn(traj):
                    return potential_reward(
                        wm.heads['reward'](traj).mean(), traj['cont'],
                        config.afdm.terminal_weight)
            elif key == 'disag' and signal == 'phase_rate_disagreement':
                from .rate_disagreement import PhaseRateDisagreement
                rewfn = PhaseRateDisagreement(config, name='disag_reward')
                self.rewards[key] = rewfn
            else:
                rewfn = self.REWARDS[key](
                    wm, act_space, config, name=key + '_reward')
                self.rewards[key] = rewfn
            critics[key] = native.VFunction(rewfn, config, name=key)
        scales = {k: v for k, v in config.expl_rewards.items() if v}
        self.ac = native.ImagActorCritic(
            critics, scales, act_space, config, name='ac')


@jaxagent.Wrapper
class Agent(native.Agent.inner):
    def __init__(self, obs_space, act_space, step, config):
        self.config, self.obs_space, self.act_space = (
            config,
            obs_space,
            act_space["action"],
        )
        self.step = step
        self.wm = WorldModel(obs_space, act_space, config, name="wm")
        self.task_behavior = Behavior(
            self.wm, self.act_space, config, name="task_behavior"
        )
        if config.expl_behavior == 'None':
            self.expl_behavior = self.task_behavior
        elif config.expl_behavior == 'TaskAlignedExplore':
            self.expl_behavior = TaskAlignedExplore(
                self.wm, self.act_space, self.config, name='expl_behavior')
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(
                self.wm, self.act_space, self.config, name='expl_behavior')

    def policy(self, obs, state, mode='train'):
        if not self.config.afdm.actor_observed_vector:
            return super().policy(obs, state, mode)
        obs = self.preprocess(obs)
        (prev_latent, prev_action), task_state, expl_state = state
        embed = self.wm.encoder(obs)
        latent, _ = self.wm.rssm.obs_step(
            prev_latent, prev_action, embed, obs['is_first'])
        policy_state = {**latent, 'vector': obs['vector']}
        if self.config.afdm.actor_posterior_mean_input:
            policy_state = with_posterior_probabilities(policy_state)
        # Both actors see the same measured observation, never rate_target.
        self.expl_behavior.policy(policy_state, expl_state)
        task_outs, task_state = self.task_behavior.policy(policy_state, task_state)
        expl_outs, expl_state = self.expl_behavior.policy(policy_state, expl_state)
        if mode == 'eval':
            outs = task_outs
            outs['action'] = outs['action'].sample(seed=nj.rng())
            outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
        elif mode == 'explore':
            outs = expl_outs
            outs['log_entropy'] = outs['action'].entropy()
            outs['action'] = outs['action'].sample(seed=nj.rng())
        elif mode == 'train':
            outs = task_outs
            outs['log_entropy'] = outs['action'].entropy()
            outs['action'] = outs['action'].sample(seed=nj.rng())
        else:
            raise ValueError(mode)
        state = ((latent, outs['action']), task_state, expl_state)
        return outs, state
