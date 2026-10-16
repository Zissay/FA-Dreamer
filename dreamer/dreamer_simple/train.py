"""One direct training entry. Real environment actions are the budget unit."""

import argparse, json, pickle, shutil, subprocess, sys
from dataclasses import asdict
from pathlib import Path

sys.modules["tensorflow"] = None
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from dreamerv3 import train as native
from dreamerv3.embodied.envs.from_gym import FromGym
from fa_afdm_dreamerv3.dreamer_matched_env import (
    MatchedDreamerEnv,
    TrainingLog,
    matched_configs,
)
from fa_afdm_dreamerv3.dreamer_truncated_actor import install_truncated_normal_actor
from dreamer_simple.agent import Agent
from dreamer_simple.replay import make_observed_rate_replay

e = native.embodied


class SeededRandomAgent:
    """Deterministic replay prefill for bounded continuous action spaces."""

    def __init__(self, act_space, seed):
        self.act_space = act_space
        self.random = np.random.RandomState(seed)

    def policy(self, obs, state=None, mode="train"):
        batch_size = len(next(iter(obs.values())))
        actions = {}
        for key, space in self.act_space.items():
            if key == "reset":
                continue
            if not np.issubdtype(space.dtype, np.floating):
                raise TypeError(f"Unsupported deterministic prefill space: {key} {space}")
            shape = (batch_size,) + tuple(space.shape)
            actions[key] = self.random.uniform(
                space.low, space.high, shape
            ).astype(space.dtype)
        return actions, state


class SeededGoalPrefillAgent:
    """Reward-blind, episode-coherent coverage of the reachable position box.

    Each synchronized episode receives stratified random terminal coordinates.
    Actions follow a straight feasible path to that target. The sampler never
    observes rewards, rates, checkpoints, or exhaustive-search references.
    """

    def __init__(self, act_space, seed, episode_steps=50, action_step=0.001,
                 position_scale=0.25):
        self.act_space = act_space
        self.random = np.random.RandomState(seed)
        self.episode_steps = int(episode_steps)
        self.action_step = float(action_step)
        self.position_scale = float(position_scale)
        self.reachable = self.episode_steps * self.action_step
        self.targets = None

    def _sample_targets(self, batch_size):
        # Latin-hypercube marginals avoid the severe clustering of iid goals
        # while remaining independent of the objective function.
        unit = np.empty((batch_size, 4), np.float32)
        for dim in range(4):
            order = self.random.permutation(batch_size)
            unit[:, dim] = (order + self.random.uniform(size=batch_size)) / batch_size
        return ((2.0 * unit - 1.0) * self.reachable).astype(np.float32)

    def policy(self, obs, state=None, mode="train"):
        vector = np.asarray(obs["vector"], np.float32)
        batch_size = len(vector)
        if self.targets is None or len(self.targets) != batch_size:
            self.targets = self._sample_targets(batch_size)
        first = np.asarray(
            obs.get("is_first", vector[:, 164] <= 1e-6), dtype=bool
        ).reshape(batch_size)
        if first.any():
            fresh = self._sample_targets(batch_size)
            self.targets[first] = fresh[first]
        position = self.position_scale * vector[:, 160:164]
        elapsed = np.clip(np.rint(vector[:, 164] * self.episode_steps), 0,
                          self.episode_steps - 1).astype(np.int32)
        remaining = np.maximum(1, self.episode_steps - elapsed)[:, None]
        action = (self.targets - position) / (remaining * self.action_step)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        return {"action": action}, state


class SeededCorrelatedExplorationPolicy:
    """Add reward-blind episode-persistent exploration to a Dreamer policy."""

    def __init__(self, policy, seed, count, initial, final, begin, end,
                 training_log=None):
        self.policy = policy
        self.random = np.random.RandomState(seed)
        self.count = count
        self.initial = float(initial)
        self.final = float(final)
        self.begin = int(begin)
        self.end = int(end)
        self.bias = None
        self.training_log = training_log

    def _scale(self):
        step = int(self.count())
        fraction = np.clip(
            (step - self.begin) / max(1, self.end - self.begin), 0.0, 1.0
        )
        return self.initial + fraction * (self.final - self.initial)

    def __call__(self, obs, state=None):
        actions, state = self.policy(obs, state)
        action = np.asarray(actions["action"])
        batch_size = len(action)
        if self.bias is None or len(self.bias) != batch_size:
            self.bias = self.random.uniform(-1.0, 1.0, action.shape).astype(np.float32)
        first = np.asarray(obs.get("is_first", np.zeros(batch_size)), dtype=bool).reshape(batch_size)
        if first.any():
            fresh = self.random.uniform(-1.0, 1.0, action.shape).astype(np.float32)
            self.bias[first] = fresh[first]
        scale = self._scale()
        modified = dict(actions)
        modified["action"] = np.clip(
            action + scale * self.bias, -1.0, 1.0
        ).astype(action.dtype)
        # The RSSM must consume the action that the environment actually saw.
        # Keeping the unperturbed proposal in its recurrent carry silently
        # corrupts the next observation update during correlated exploration.
        (latent, _), task_state, expl_state = state
        state = ((latent, modified["action"]), task_state, expl_state)
        if self.training_log is not None:
            self.training_log.source = (
                "actor_policy_correlated_exploration" if scale > 0
                else "actor_policy"
            )
        return modified, state


def model_config(hp, seed, platform="cuda", episode_steps=50,
                 action_step_m=0.001, position_scale_m=0.25,
                 direct_position_action=False):
    units = hp["units"]
    cfg = e.Config(Agent.configs["defaults"]).update(Agent.configs["small"])
    cfg = cfg.update(
        {
            "seed": seed,
            "jax.platform": platform,
            "jax.precision": "float32",
            "jax.prealloc": False,
            "batch_size": hp["batch_size"],
            "batch_length": hp["batch_length"],
            "replay_size": hp["replay_size"],
            # Multiple loader threads race on the replay sampler RNG and make
            # identical seeds consume different batches. Zero workers keeps
            # sampling synchronous and reproducible for controlled experiments.
            "data_loaders": hp.get("data_loaders", 2),
            "rssm.deter": units,
            "rssm.units": units,
            "encoder.mlp_units": units,
            "decoder.mlp_units": units,
            "encoder.mlp_keys": "vector",
            "encoder.cnn_keys": "$^",
            "decoder.mlp_keys": "vector",
            "decoder.cnn_keys": "$^",
            "reward_head.units": units,
            "cont_head.units": units,
            "actor.units": units,
            # Initialization only, never an online override of learned actions.
            # Default preserves all archived runs. Zero initializes output
            # projections without zeroing the hidden representation or noise.
            "actor.outscale": hp.get("actor_outscale", 1.0),
            "actor.inputs": (["deter", "posterior_probabilities", "vector"]
                             if hp.get("actor_posterior_mean_input", False)
                             else (["deter", "stoch", "vector"]
                                   if hp.get("actor_observed_vector", False)
                                   else ["deter", "stoch"])),
            "critic.units": units,
            "critic.inputs": (["deter", "stoch", "vector"]
                              if hp.get("critic_observed_vector", False)
                              else ["deter", "stoch"]),
            "reward_head.dist": "mse",
            "actor_dist_cont": "trunc_normal",
            "actor_grad_cont": "backprop",
            "actor.minstd": hp["minstd"],
            "actor.maxstd": hp["maxstd"],
            "actor_opt.lr": hp["actor_lr"],
            "model_opt.lr": hp["model_lr"],
            "critic_opt.lr": hp["critic_lr"],
            "critic_opt.clip": hp.get("critic_clip", 100),
            "slow_critic_fraction": hp.get("slow_critic_fraction", 0.02),
            "actor_opt.warmup": hp["actor_warmup"],
            "actor_opt.clip": hp["actor_clip"],
            "model_opt.clip": hp["model_clip"],
            "actent": hp["entropy"],
            "expl_behavior": hp.get("expl_behavior", "None"),
            "expl_rewards": hp.get("expl_rewards", {"extr": 1.0, "disag": 0.1}),
            "disag_models": hp.get("disag_models", 4),
            "disag_head.units": hp.get("disag_units", units),
            "disag_head.layers": hp.get("disag_layers", 2),
            "expl_opt.lr": hp.get("expl_lr", 0.0001),
            "imag_horizon": hp["imag_horizon"],
            "horizon": 1000000000,
            "loss_scales.reward": hp["reward_loss_scale"],
            "loss_scales.vector": hp["vector_loss_scale"],
            "return_lambda": hp["return_lambda"],
            "wrapper.length": int(episode_steps),
        }
    )
    # Config construction adds task-owned keys; no legacy CLI forwarding.
    return e.Config(
        {
            **dict(cfg),
            "afdm": {
                "time_index": 164,
                "terminal_weight": 0.2,
                "replay_value_scale": hp["replay_value_scale"],
                "critic_mc_distill_scale": hp.get("critic_mc_distill_scale", 0.0),
                "critic_mc_distill_count": hp.get("critic_mc_distill_count", 16),
                "critic_mc_distill_repeats": hp.get("critic_mc_distill_repeats", 4),
                "critic_mc_distill_step": hp.get("critic_mc_distill_step", 20),
                "replay_imitation_scale": hp.get("replay_imitation_scale", 0.0),
                "replay_imitation_start_update": hp.get("replay_imitation_start_update", 0),
                "replay_imitation_late_start_update": hp.get(
                    "replay_imitation_late_start_update", 0),
                "replay_imitation_late_end_update": hp.get(
                    "replay_imitation_late_end_update", 0),
                "replay_imitation_late_final_multiplier": hp.get(
                    "replay_imitation_late_final_multiplier", 1.0),
                "replay_return_lambda": hp.get("replay_return_lambda", hp["return_lambda"]),
                "imagination_start_stride": hp.get("imagination_start_stride", 1),
                "imagination_start_count": hp.get("imagination_start_count", 0),
                "imagination_start_mode": hp.get("imagination_start_mode", "uniform"),
                "priority_fraction": hp.get("priority_fraction", 0.5),
                "priority_temperature": hp.get("priority_temperature", 1.0),
                "initial_start_fraction": hp.get("initial_start_fraction", 0.0),
                "physical_start_for_completed_priority": hp.get(
                    "physical_start_for_completed_priority", False),
                "consistent_actor_distribution": hp.get("consistent_actor_distribution", False),
                "full_return_actor": hp.get("full_return_actor", False),
                "critic_bootstrap_actor": hp.get("critic_bootstrap_actor", False),
                "actor_critic_loss_scale": hp.get("actor_critic_loss_scale", 0.0),
                "actor_critic_bootstrap_step": hp.get("actor_critic_bootstrap_step", 20),
                "actor_critic_start_update": hp.get("actor_critic_start_update", 2000),
                "actor_critic_ramp_end_update": hp.get("actor_critic_ramp_end_update", 4000),
                "local_rate_gain_scale": hp.get("local_rate_gain_scale", 0.0),
                "local_rate_threshold": hp.get("local_rate_threshold", 16.0),
                "local_rate_start_count": hp.get("local_rate_start_count", 64),
                "known_kinematics": hp.get("known_kinematics", False),
                "observed_rate_head": hp.get("observed_rate_head", False),
                "actor_observed_vector": hp.get("actor_observed_vector", False),
                "task_actor_vector_only": hp.get("task_actor_vector_only", False),
                "actor_physical_feedback_gradient": hp.get("actor_physical_feedback_gradient", False),
                "actor_posterior_mean_input": hp.get("actor_posterior_mean_input", False),
                "critic_observed_vector": hp.get("critic_observed_vector", False),
                "exploration_signal": hp.get("exploration_signal", "latent_disagreement"),
                "actor_kl_scale": hp.get("actor_kl_scale", 0.0),
                "actor_kl_start_update": hp.get("actor_kl_start_update", 0),
                "actor_reference_period": hp.get("actor_reference_period", 16),
                "task_intrinsic_scale": hp.get("task_intrinsic_scale", 0.0),
                "task_intrinsic_decay_start": hp.get("task_intrinsic_decay_start", 0),
                "task_intrinsic_decay_end": hp.get("task_intrinsic_decay_end", 1),
                "actor_std_cap_start": hp.get("actor_std_cap_start", 0),
                "actor_std_cap_end": hp.get("actor_std_cap_end", 1),
                "actor_std_cap_final": hp.get("actor_std_cap_final", 0.0),
                "wavelength": 0.0107068735,
                "action_step_m": float(action_step_m),
                "position_scale_m": float(position_scale_m),
                "episode_steps": int(episode_steps),
                "direct_position_action": bool(direct_position_action),
                "schedule": hp.get("schedule", {"enabled": False}),
            },
        }
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--queries", type=int, default=100000)
    p.add_argument("--seed", type=int, default=3101)
    p.add_argument("--platform", choices=("cuda", "cpu"), default="cuda",
                   help="JAX platform; CUDA is used for full training, CPU is useful for smoke checks")
    p.add_argument("--num-envs", type=int, choices=(1, 20), default=20,
                   help="Parallel real environments; one env records a terminal every50 queries")
    p.add_argument("--config", type=Path, default=ROOT / "dreamer_simple/config.json")
    p.add_argument("--task-config", type=Path, default=ROOT / "dreamer_simple/task.json",
                   help="Fixed-channel task specification; defaults to the 50-step protocol")
    p.add_argument(
        "--imag-horizon",
        type=int,
        help="Single controlled override for horizon experiments",
    )
    p.add_argument("--gpu-rate-socket", required=True)
    p.add_argument("--snapshot-every", type=int, default=25000)
    p.add_argument("--from-checkpoint", type=Path,
                   help="Continue agent and replay state from a completed compatible run.")
    args = p.parse_args()
    hp = json.loads(args.config.read_text())
    if args.imag_horizon is not None:
        hp["imag_horizon"] = args.imag_horizon
    assert args.queries % 1000 == 0 and args.queries >= 3000
    assert args.snapshot_every > 0 and args.snapshot_every % 1000 == 0
    baseline_path = args.task_config
    baseline = json.loads(baseline_path.read_text())
    channel, task = matched_configs(baseline, args.seed, args.num_envs, args.gpu_rate_socket)
    assert (
        channel.n_paths == 20 and abs(channel.carrier_wavelength - 0.0107068735) < 1e-12
    )
    assert channel.tx_region == channel.rx_region == 0.5
    assert task.max_steps > 0
    assert task.direct_position_action or task.action_step > 0
    # Both configured reward forms below are potential-based formulations of
    # the terminal-rate objective for the fixed-start finite-horizon task.
    # Do not accept the dense ``rate`` mode here: it optimizes path-average
    # rate instead of the declared final-position objective.
    assert task.reward_mode in ("rate_delta", "terminal_potential")
    assert task.terminal_rate_weight > 0
    if not task.direct_position_action:
        # Incremental-action protocol: its action step defines the reachable
        # box. Direct-position tasks have no such constraint.
        assert (
            task.action_step * task.max_steps
            <= min(channel.tx_region, channel.rx_region) / 2 + 1e-12
        )
    if args.from_checkpoint is not None:
        assert args.from_checkpoint.is_file()
        with args.from_checkpoint.open("rb") as f:
            resume_data = pickle.load(f)
    else:
        resume_data = None
    args.run_dir.mkdir(parents=True, exist_ok=True)
    log = TrainingLog(
        args.run_dir, args.num_envs, args.queries
    )  # Exclusive files; no accidental resume.
    cfg = model_config(
        hp, args.seed, args.platform, task.max_steps,
        task.action_step, channel.tx_region / 2,
        task.direct_position_action)
    cfg.save(e.Path(args.run_dir / "config.yaml"))
    meta = dict(
        seed=args.seed,
        num_envs=args.num_envs,
        terminal_record_interval=args.num_envs * task.max_steps,
        real_action_budget=args.queries,
        internal_frames=args.queries * (task.max_steps + 1) // task.max_steps,
        baseline=str(baseline_path),
        channel=asdict(channel),
        environment=asdict(task),
        from_checkpoint=(str(args.from_checkpoint) if args.from_checkpoint else None),
        algorithm="RSSM Dreamer, state-rate model, exact horizon and replay-value critic",
        known_kinematics=hp.get("known_kinematics", False),
        rate_model_input=("observed_pairwise_phase_only" if hp.get("observed_rate_head", False)
                          else "rssm_latent"),
        rate_loss_units=("rate_divided_by_10" if hp.get("observed_rate_head", False)
                         else "rate"),
        hyperparameters=hp,
        prefill_rng_seed=args.seed + 1000003,
        jax_platform=args.platform,
        imagined_termination="known finite task clock",
    )
    (args.run_dir / "experiment.json").write_text(json.dumps(meta, indent=2))
    install_truncated_normal_actor(stable_sampling=True)

    def env_ctor():
        return native.wrap_env(
            FromGym(MatchedDreamerEnv(channel, task, log, True)), cfg
        )

    env = e.BatchEnv([e.Parallel(env_ctor, "thread") for _ in range(args.num_envs)], parallel=True)
    counter = e.Counter()
    logger = e.Logger(
        counter, [e.logger.JSONLOutput(e.Path(args.run_dir), "metrics.jsonl")]
    )
    if args.from_checkpoint is not None:
        source_replay = args.from_checkpoint.parent / "replay"
        assert source_replay.is_dir()
        shutil.copytree(source_replay, args.run_dir / "replay")
    if hp.get("observed_rate_replay_elite_fraction", 0):
        replay = make_observed_rate_replay(
            cfg, e.Path(args.run_dir / "replay"),
            hp["observed_rate_replay_elite_fraction"],
            hp.get("observed_rate_replay_sample_fraction", 0.5),
            hp.get("observed_rate_replay_score_mode", "sequence_max"),
        )
    else:
        replay = native.make_replay(cfg, e.Path(args.run_dir / "replay"))
    model = Agent(env.obs_space, env.act_space, counter, cfg)
    if resume_data is not None:
        counter.load(resume_data["step"])
        model.load(resume_data["agent"])
        replay.load(resume_data["replay"])
    resume_start_frames = int(counter)
    driver = e.Driver(env)
    driver.on_step(lambda tran, worker: counter.increment())
    driver.on_step(replay.add)
    prefill_mode = hp.get("prefill_mode", "uniform")
    if prefill_mode == "uniform":
        random = SeededRandomAgent(env.act_space, args.seed + 1000003)
        log.source = "uniform_prefill"
    elif prefill_mode == "stratified_goal":
        random = SeededGoalPrefillAgent(
            env.act_space, args.seed + 1000003,
            episode_steps=task.max_steps, action_step=task.action_step,
            position_scale=channel.tx_region / 2,
        )
        log.source = "stratified_goal_prefill"
    else:
        raise ValueError(f"Unsupported prefill_mode: {prefill_mode}")
    stats = e.Metrics()
    state = None
    next_log = int(counter) + 2040
    next_save = int(counter) + 10200
    next_plot = args.snapshot_every

    def save():
        # Explicit atomic synchronous save avoids the old package's save bug.
        data = {"step": counter.save(), "agent": model.save(), "replay": replay.save()}
        tmp = args.run_dir / "checkpoint.tmp"
        with tmp.open("wb") as f:
            pickle.dump(data, f)
        tmp.replace(args.run_dir / "checkpoint.pkl")

    def plot():
        if hp.get("retain_milestone_checkpoints", False):
            save()
            saved = args.run_dir / "milestone_checkpoints"
            saved.mkdir(exist_ok=True)
            shutil.copy2(
                args.run_dir / "checkpoint.pkl",
                saved / f"checkpoint_{log.count:06d}.pkl",
            )

    try:
        explicit_prefill = 0 if resume_data is not None else int(hp.get("prefill_real_actions", 0))
        if explicit_prefill:
            assert explicit_prefill % (args.num_envs * task.max_steps) == 0
            assert explicit_prefill >= hp["batch_size"] * hp["batch_length"]
            # Driver holds one queued prefill action batch when collection
            # stops. It is executed after the update callback is attached.
            collection_target = explicit_prefill - args.num_envs
            while log.count < collection_target:
                driver(random.policy, steps=min(100, collection_target - log.count))
            assert log.count == collection_target
        elif resume_data is not None:
            pass
        elif args.num_envs == 1:
            # Match the existing20-env run's2260 true prefill actions, not its
            # replay-length threshold (sequence overlap differs with env count).
            # Driver executes its queued action before proposing the next one.
            while log.count < 2259 or len(replay) < hp["batch_size"] * hp["batch_length"]:
                driver(random.policy, steps=1)
            assert log.count == 2259, "Replay needs a separately declared prefill budget"
        else:
            # Preserve the old20-env behavior exactly for existing launchers.
            while len(replay) < hp["batch_size"] * hp["batch_length"]:
                driver(random.policy, steps=100)
        dataset = iter(model.dataset(replay.dataset))
        ratio = e.when.Ratio(
            hp["train_ratio"] / (hp["batch_size"] * hp["batch_length"])
        )

        def update(tran, worker):
            nonlocal state
            for _ in range(ratio(counter)):
                _, state, metrics = model.train(next(dataset), state)
                bad = [
                    k
                    for k, v in metrics.items()
                    if (k.endswith("_opt_loss") or k.endswith("_opt_grad_norm"))
                    and not np.isfinite(v).all()
                ]
                if bad:
                    raise FloatingPointError(str(bad))
                stats.add(metrics, prefix="train")
            model.sync()

        driver.on_step(update)
        explore_real_actions = int(hp.get("explore_real_actions", 0))
        if explore_real_actions:
            assert hp.get("expl_behavior", "None") != "None"
            assert explore_real_actions % args.num_envs == 0

        def base_policy(obs, state=None):
            if log.count < explore_real_actions:
                log.source = "disagreement_exploration_policy"
                return model.policy(obs, state, mode="explore")
            log.source = "actor_policy"
            return model.policy(obs, state, mode="train")
        correlated_initial = float(hp.get("correlated_exploration_initial", 0.0))
        if correlated_initial:
            policy = SeededCorrelatedExplorationPolicy(
                base_policy,
                args.seed + 2000003,
                lambda: log.count,
                correlated_initial,
                float(hp.get("correlated_exploration_final", 0.0)),
                int(hp.get("correlated_exploration_begin", 0)),
                int(hp["correlated_exploration_end"]),
                training_log=log,
            )
            actor_source = "actor_policy_correlated_exploration"
        else:
            policy = base_policy
            actor_source = "actor_policy"
        # The first action batch was proposed by the prefill policy.
        driver(policy, steps=args.num_envs)
        if explicit_prefill:
            assert log.count == explicit_prefill
        # `driver` has already queued the first action from `base_policy` and
        # `base_policy` set its exact source. Do not overwrite that label before
        # the queued batch is executed on the next driver call.
        if correlated_initial:
            log.source = actor_source
        target_frames = resume_start_frames + args.queries * (task.max_steps + 1) // task.max_steps
        while int(counter) < target_frames:
            driver(policy, steps=min(100, target_frames - int(counter)))
            if int(counter) >= next_log:
                logger.add(stats.result())
                logger.write(fps=True)
                print(f"TRAIN real_actions={log.count}", flush=True)
                if hp.get("observed_rate_replay_elite_fraction", 0):
                    sampler = replay.sampler
                    print(
                        "REPLAY "
                        f"elite_draws={sampler.elite_draws} "
                        f"uniform_draws={sampler.uniform_draws} "
                        f"elite_size={len(sampler.elite)} "
                        f"elite_min_rate={min(map(sampler.scores.__getitem__, sampler.elite)):.6f}",
                        flush=True,
                    )
                next_log += 2040
            if int(counter) >= next_save:
                save()
                next_save += 10200
            # Milestones are checked at exact real-action counts; reset frames
            # do not create samples. Driver batches can cross the boundary.
            if log.count >= next_plot:
                plot()
                next_plot += args.snapshot_every
        assert log.count == args.queries
        save()
        logger.add(stats.result())
        logger.write()
        (args.run_dir / "training_complete.json").write_text(
            json.dumps({"real_actions": log.count, "status": "complete"})
        )
    except Exception as ex:
        (args.run_dir / "training_failure.json").write_text(
            json.dumps({"real_actions": log.count, "error": repr(ex)})
        )
        raise
    finally:
        log.close()
        env.close()


if __name__ == "__main__":
    main()
