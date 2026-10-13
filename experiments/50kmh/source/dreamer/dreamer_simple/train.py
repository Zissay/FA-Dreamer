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

e = native.embodied


def model_config(hp, seed, platform="cuda"):
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
            "data_loaders": 2,
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
            "critic.units": units,
            "reward_head.dist": "mse",
            "actor_dist_cont": "trunc_normal",
            "actor_grad_cont": "backprop",
            "actor.minstd": hp["minstd"],
            "actor.maxstd": hp["maxstd"],
            "actor_opt.lr": hp["actor_lr"],
            "model_opt.lr": hp["model_lr"],
            "critic_opt.lr": hp["critic_lr"],
            "actor_opt.warmup": hp["actor_warmup"],
            "actor_opt.clip": hp["actor_clip"],
            "model_opt.clip": hp["model_clip"],
            "actent": hp["entropy"],
            "imag_horizon": hp["imag_horizon"],
            "horizon": 1000000000,
            "loss_scales.reward": hp["reward_loss_scale"],
            "loss_scales.vector": hp["vector_loss_scale"],
            "return_lambda": hp["return_lambda"],
            "wrapper.length": 50,
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
                "replay_return_lambda": hp.get("replay_return_lambda", hp["return_lambda"]),
                "imagination_start_stride": hp.get("imagination_start_stride", 1),
                "imagination_start_count": hp.get("imagination_start_count", 0),
                "full_return_actor": hp.get("full_return_actor", False),
                "known_kinematics": hp.get("known_kinematics", False),
                "wavelength": 0.0107068735,
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
    p.add_argument(
        "--imag-horizon",
        type=int,
        help="Single controlled override for horizon experiments",
    )
    p.add_argument("--gpu-rate-socket", required=True)
    p.add_argument("--snapshot-every", type=int, default=25000)
    args = p.parse_args()
    hp = json.loads(args.config.read_text())
    if args.imag_horizon is not None:
        hp["imag_horizon"] = args.imag_horizon
    assert args.queries % 1000 == 0 and args.queries >= 3000
    assert args.snapshot_every > 0 and args.snapshot_every % 1000 == 0
    baseline_path = ROOT / "dreamer_simple/task.json"
    baseline = json.loads(baseline_path.read_text())
    channel, task = matched_configs(baseline, args.seed, args.num_envs, args.gpu_rate_socket)
    assert (
        channel.n_paths == 20 and abs(channel.carrier_wavelength - 0.0107068735) < 1e-12
    )
    assert channel.tx_region == channel.rx_region == 0.5
    assert (
        task.max_steps == 50
        and task.action_step == 0.001
        and task.terminal_rate_weight == 0.2
    )
    assert (
        task.action_step * task.max_steps
        < min(channel.tx_region, channel.rx_region) / 2
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    log = TrainingLog(
        args.run_dir, args.num_envs, args.queries
    )  # Exclusive files; no accidental resume.
    cfg = model_config(hp, args.seed, args.platform)
    cfg.save(e.Path(args.run_dir / "config.yaml"))
    meta = dict(
        seed=args.seed,
        num_envs=args.num_envs,
        terminal_record_interval=args.num_envs * task.max_steps,
        real_action_budget=args.queries,
        internal_frames=args.queries * 51 // 50,
        baseline=str(baseline_path),
        channel=asdict(channel),
        environment=asdict(task),
        from_checkpoint=None,
        algorithm="RSSM Dreamer, state-rate model, exact horizon and replay-value critic",
        known_kinematics=hp.get("known_kinematics", False),
        hyperparameters=hp,
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
    replay = native.make_replay(cfg, e.Path(args.run_dir / "replay"))
    model = Agent(env.obs_space, env.act_space, counter, cfg)
    driver = e.Driver(env)
    driver.on_step(lambda tran, worker: counter.increment())
    driver.on_step(replay.add)
    random = e.RandomAgent(env.act_space)
    stats = e.Metrics()
    state = None
    next_log, next_save, next_plot = 2040, 10200, args.snapshot_every

    def save():
        # Explicit atomic synchronous save avoids the old package's save bug.
        data = {"step": counter.save(), "agent": model.save(), "replay": replay.save()}
        tmp = args.run_dir / "checkpoint.tmp"
        with tmp.open("wb") as f:
            pickle.dump(data, f)
        tmp.replace(args.run_dir / "checkpoint.pkl")

    def plot():
        destination = ROOT / "output/pdf" / args.run_dir.name / f"step_{log.count:06d}"
        if hp.get("retain_milestone_checkpoints", False):
            save()
            saved = args.run_dir / "milestone_checkpoints"
            saved.mkdir(exist_ok=True)
            shutil.copy2(
                args.run_dir / "checkpoint.pkl",
                saved / f"checkpoint_{log.count:06d}.pkl",
            )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/plot_dreamer_intermediate_snapshot.py"),
                "--run-dir",
                str(args.run_dir),
                "--output-dir",
                str(destination),
            ],
            check=True,
        )

    try:
        if args.num_envs == 1:
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
        policy = lambda *a: model.policy(*a, mode="train")
        # The first action batch was proposed by the prefill policy.
        driver(policy, steps=args.num_envs)
        log.source = "actor_policy"
        target_frames = args.queries * 51 // 50
        while int(counter) < target_frames:
            driver(policy, steps=min(100, target_frames - int(counter)))
            if int(counter) >= next_log:
                logger.add(stats.result())
                logger.write(fps=True)
                print(f"TRAIN real_actions={log.count}", flush=True)
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
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/report_dreamer_fixed_center_matched.py"),
            "--run-dir",
            str(args.run_dir),
            "--output-dir",
            str(ROOT / "output/pdf" / args.run_dir.name),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
