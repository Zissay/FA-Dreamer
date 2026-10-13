"""Explicit subclasses: state-rate world model, known clock, replay-value critic.

No runtime replacement of WorldModel/actor/critic methods. The sole package
compatibility shim adds its missing numerically-safe truncated distribution.
"""

import jax
import jax.numpy as jnp
from dreamerv3 import agent as native, jaxagent, jaxutils, ninjax as nj

sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)


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


class WorldModel(native.WorldModel):
    def loss(self, data, state):
        target = dict(data, reward=data["rate_target"].reshape(data["reward"].shape))
        return super().loss(target, state)

    def imagine(self, policy, start, horizon):
        if self.config.afdm.known_kinematics:
            from .kinematics import rollout

            traj = rollout(self, policy, start, horizon)
        else:
            traj = super().imagine(policy, start, horizon)
        clock = jnp.rint(start["vector"][..., self.config.afdm.time_index] * 50).astype(
            jnp.int32
        )
        cont = ((clock[None] + jnp.arange(horizon + 1)[:, None]) < 50).astype(
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
        for key, pred in [
            ("posterior", dists["reward"].mean()),
            ("prior", self.heads["reward"](prior).mean()),
        ]:
            mse = avg((sg(pred) - truth) ** 2)
            out[f"{key}_reward_rmse"] = jnp.sqrt(mse)
            out[f"{key}_reward_r2"] = 1 - mse / jnp.maximum(var, 1e-8)
        return out


class ActorCritic(native.ImagActorCritic):
    def __init__(self, critics, scales, act_space, config, reward_fn=None):
        super().__init__(critics, scales, act_space, config)
        self.reward_fn = reward_fn or critics["extr"].rewfn
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
        if self.config.afdm.full_return_actor:
            assert self.config.imag_horizon >= 50 and self.config.return_lambda == 1.
            # The actor consumes only learned world-model rewards. No critic
            # value, bootstrap, baseline, or critic-derived statistic is read.
            rew = self.reward_fn(traj)
            dist = self.actor(sg(traj))
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
            logpi = dist.log_prob(sg(traj['action']))[:-1]
            randomness = (ent - dist.minent) / (dist.maxent - dist.minent)
            if len(randomness.shape) > 2:
                randomness = randomness.mean(range(2, len(randomness.shape)))
            metrics = {}
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
                full_return_no_critic_objective=jnp.asarray(1.))
            if self.config.afdm.schedule.enabled:
                factor = actor_factor(step, self.config.afdm.schedule)
                warmup = jnp.minimum(1., step / max(1, self.config.actor_opt.warmup))
                metrics.update(scheduled_entropy_coefficient=coefficient,
                               scheduled_actor_lr_factor=factor,
                               scheduled_actor_learning_rate=self.config.actor_opt.lr*warmup*factor)
            return loss, metrics
        loss, metrics = super().loss(traj)
        if self.config.afdm.schedule.enabled:
            from .schedules import entropy_coefficient, actor_factor

            step = self.opt.step.read()
            coefficient = entropy_coefficient(step, self.config.afdm.schedule)
            ent = self.actor(sg(traj)).entropy()[:-1]
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

    def train(self, imagine, start, context):
        stride = int(self.config.afdm.imagination_start_stride)
        requested = int(self.config.afdm.imagination_start_count)
        if requested > 0 or stride > 1:
            # Selection uses reset/terminal flags only, never rate or reward.
            count = start['is_terminal'].shape[0]
            amount = requested if requested > 0 else (count + stride - 1) // stride
            if self.config.afdm.imagination_start_mode == 'initial':
                indices = initial_imagination_indices(
                    nj.rng(), start['is_terminal'], start['is_first'], amount)
            else:
                assert self.config.afdm.imagination_start_mode == 'uniform'
                indices = valid_imagination_indices(
                    nj.rng(), start['is_terminal'], amount)
            start = jax.tree_util.tree_map(lambda value: value[indices], start)

        def actor_loss(start):
            policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
            traj = imagine(policy, start, self.config.imag_horizon)
            loss, metrics = self.loss(traj)
            metrics['imagination_start_count'] = jnp.asarray(start['is_terminal'].shape[0])
            metrics['imagination_initial_start_fraction'] = start['is_first'].mean()
            metrics['imagination_transition_count'] = jnp.asarray(
                start['is_terminal'].shape[0] * self.config.imag_horizon)
            return loss, (traj, metrics)

        mets, (traj, metrics) = self.opt(self.actor, actor_loss, start, has_aux=True)
        metrics.update(mets)
        critic = self.critics["extr"]
        target = sg(critic.score(traj)[1])

        def joint_loss(traj, target, context):
            loss, m = critic.loss(traj, target)
            features = {k: sg(context[k]) for k in ("deter", "stoch")}
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
            critic.net, joint_loss, traj, target, context, has_aux=True
        )
        cm.update(mets)
        critic.updater()
        metrics.update({f"extr_critic_{k}": v for k, v in cm.items()})
        return traj, metrics


class Behavior(nj.Module):
    def __init__(self, wm, act_space, config):
        def reward(traj):
            rate = wm.heads["reward"](traj).mean()
            return (
                rate[1:]
                - rate[:-1]
                + config.afdm.terminal_weight * (1 - traj["cont"][1:]) * rate[1:]
            )

        critic = native.VFunction(reward, config, name="critic")
        self.ac = ActorCritic(
            {"extr": critic}, {"extr": 1.0}, act_space, config,
            reward_fn=reward, name="ac")

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        return self.ac.train(imagine, start, data)

    def report(self, data):
        return {}


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
        self.expl_behavior = self.task_behavior
