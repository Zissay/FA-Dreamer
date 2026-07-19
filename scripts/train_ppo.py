from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from fa_afdm_dreamerv3.channel import FAAFDMChannel, FAAFDMConfig
from fa_afdm_dreamerv3.env import EnvConfig, FAAFDMEnv
from fa_afdm_dreamerv3.position_splits import load_position_set, normalize_position_set


def _make_env(args, seed_offset: int = 0, positions: list[dict] | None = None) -> FAAFDMEnv:
    channel_seed = args.seed if args.channel_seed is None else args.channel_seed
    env_seed = args.seed if args.env_seed is None else args.env_seed
    channel = FAAFDMChannel(
        FAAFDMConfig(
            n_subcarriers=args.n_subcarriers,
            n_paths=args.n_paths,
            channel_memory=args.channel_memory,
            noise_power_dbm=args.noise_power_dbm,
            channel_gain_scale=args.channel_gain_scale,
            seed=channel_seed,
        )
    )
    return FAAFDMEnv(
        channel=channel,
        cfg=EnvConfig(
            max_steps=args.episode_steps,
            action_step=args.action_step,
            direct_position_action=args.direct_position_action,
            random_reset=args.random_reset,
            seed=env_seed + seed_offset,
            high_rate_threshold=args.high_rate_threshold,
            high_rate_bonus=args.high_rate_bonus,
            high_rate_slope=args.high_rate_slope,
            rate_baseline=args.rate_baseline,
            reward_scale=args.reward_scale,
            raw_rate_scale=args.raw_rate_scale,
            improvement_weight=args.improvement_weight,
            best_improvement_weight=args.best_improvement_weight,
            movement_penalty=args.movement_penalty,
            action_l2_penalty=args.action_l2_penalty,
            regression_penalty=args.regression_penalty,
            best_gap_penalty=args.best_gap_penalty,
            high_rate_move_penalty=args.high_rate_move_penalty,
            high_rate_action_l2_penalty=args.high_rate_action_l2_penalty,
            action_smooth_penalty=args.action_smooth_penalty,
            high_rate_smooth_penalty=args.high_rate_smooth_penalty,
            boundary_margin=args.boundary_margin,
            boundary_free_dims=args.boundary_free_dims,
            boundary_penalty=args.boundary_penalty,
            late_rate_weight=args.late_rate_weight,
            late_rate_power=args.late_rate_power,
            terminal_rate_weight=args.terminal_rate_weight,
            initial_positions=None if positions is None else normalize_position_set(positions),
        ),
    )


def _linear_schedule(initial_value: float):
    def schedule(progress_remaining: float) -> float:
        return float(progress_remaining) * initial_value

    return schedule


def _evaluate_model(model, args, episodes: int) -> dict:
    eval_positions = load_position_set(args.position_split_file, "eval") if args.position_split_file else None
    env = _make_env(args, seed_offset=10_000, positions=eval_positions)
    best_rates = []
    final_rates = []
    steps_to_best = []
    num_episodes = len(eval_positions) if eval_positions is not None else episodes
    for episode in range(num_episodes):
        if eval_positions is not None:
            pos = eval_positions[episode]
            obs, info = env.reset(options={"u": pos["u"], "q": pos["q"]})
        else:
            obs, info = env.reset(seed=args.eval_seed + episode)
        done = False
        best_rate = float(info["rate"])
        best_step = 0
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            step += 1
            done = bool(terminated or truncated)
            rate = float(info["rate"])
            if rate > best_rate:
                best_rate = rate
                best_step = step
        best_rates.append(best_rate)
        final_rates.append(float(info["rate"]))
        steps_to_best.append(best_step)
    env.close()
    return {
        "mean_best_rate": float(np.mean(best_rates)),
        "std_best_rate": float(np.std(best_rates)),
        "mean_final_rate": float(np.mean(final_rates)),
        "mean_steps_to_best": float(np.mean(steps_to_best)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO baseline for FA-AFDM position optimization.")
    parser.add_argument("--logdir", default="runs/ppo")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channel-seed", type=int, default=None)
    parser.add_argument("--env-seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=7000)
    parser.add_argument("--total-timesteps", type=int, default=20000)
    parser.add_argument(
        "--from-model",
        default="",
        help="Optional PPO .zip checkpoint to continue training from.",
    )
    parser.add_argument(
        "--reset-num-timesteps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether SB3 should reset the internal timestep counter when continuing training.",
    )
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--episode-steps", type=int, default=40)
    parser.add_argument(
        "--random-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize the initial FA position at reset. Disable for fixed-channel direct-position bandit optimization.",
    )
    parser.add_argument("--action-step", type=float, default=0.04)
    parser.add_argument(
        "--direct-position-action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interpret actions as full normalized antenna positions instead of position increments.",
    )
    parser.add_argument("--n-subcarriers", type=int, default=16)
    parser.add_argument("--n-paths", type=int, default=4)
    parser.add_argument("--channel-memory", type=int, default=5)
    parser.add_argument("--noise-power-dbm", type=float, default=-95.0)
    parser.add_argument("--channel-gain-scale", type=float, default=2.280350850198276e-6)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=None,
        help="Initial Gaussian policy log standard deviation. Lower values reduce PPO action sampling noise.",
    )
    parser.add_argument(
        "--log-std-final",
        type=float,
        default=None,
        help="If set, linearly anneal the Gaussian policy log standard deviation to this value during training.",
    )
    parser.add_argument(
        "--log-std-anneal-steps",
        type=int,
        default=0,
        help="Number of environment steps used for --log-std-final annealing. Defaults to total timesteps.",
    )
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--raw-rate-scale", type=float, default=0.0)
    parser.add_argument("--improvement-weight", type=float, default=0.5)
    parser.add_argument("--best-improvement-weight", type=float, default=0.0)
    parser.add_argument("--rate-baseline", type=float, default=0.0)
    parser.add_argument("--movement-penalty", type=float, default=0.01)
    parser.add_argument("--action-l2-penalty", type=float, default=0.0)
    parser.add_argument("--regression-penalty", type=float, default=0.0)
    parser.add_argument("--best-gap-penalty", type=float, default=0.0)
    parser.add_argument("--high-rate-move-penalty", type=float, default=0.0)
    parser.add_argument("--high-rate-action-l2-penalty", type=float, default=0.0)
    parser.add_argument("--action-smooth-penalty", type=float, default=0.0)
    parser.add_argument("--high-rate-smooth-penalty", type=float, default=0.0)
    parser.add_argument("--boundary-margin", type=float, default=1.0)
    parser.add_argument("--boundary-free-dims", type=int, default=0)
    parser.add_argument("--boundary-penalty", type=float, default=0.0)
    parser.add_argument("--late-rate-weight", type=float, default=0.0)
    parser.add_argument("--late-rate-power", type=float, default=1.0)
    parser.add_argument("--terminal-rate-weight", type=float, default=0.0)
    parser.add_argument("--high-rate-threshold", type=float, default=0.0)
    parser.add_argument("--high-rate-bonus", type=float, default=0.0)
    parser.add_argument("--high-rate-slope", type=float, default=1.0)
    parser.add_argument(
        "--position-split-file",
        default="",
        help="JSON file with train/eval initial-position splits. Training samples train; evaluation iterates eval.",
    )
    parser.add_argument(
        "--query-log",
        default="",
        help="Optional CSV path for logging every real training environment query.",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
    except Exception as exc:
        raise SystemExit(
            "PPO baseline requires stable-baselines3. Install it with:\n"
            "  pip install stable-baselines3\n"
            f"Original error: {exc}"
        ) from exc

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    metrics_path = logdir / "metrics.jsonl"
    eval_checkpoint_dir = logdir / "eval_checkpoints"
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    query_log_path = Path(args.query_log) if args.query_log else None
    if query_log_path is not None:
        query_log_path.parent.mkdir(parents=True, exist_ok=True)
        with query_log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "query",
                    "timesteps",
                    "rate",
                    "best_rate",
                    "u_x",
                    "u_y",
                    "q_x",
                    "q_y",
                ],
            )
            writer.writeheader()

    class EvalJsonlCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.started = time.perf_counter()
            self.best_final_rate = -float("inf")
            self.best_rate = -float("inf")

        def _on_step(self) -> bool:
            if query_log_path is not None:
                infos = self.locals.get("infos", [])
                with query_log_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "query",
                            "timesteps",
                            "rate",
                            "best_rate",
                            "u_x",
                            "u_y",
                            "q_x",
                            "q_y",
                        ],
                    )
                    for info in infos:
                        if "rate" not in info:
                            continue
                        u = info.get("u", [np.nan, np.nan])
                        q = info.get("q", [np.nan, np.nan])
                        writer.writerow(
                            {
                                "query": int(self.num_timesteps),
                                "timesteps": int(self.num_timesteps),
                                "rate": float(info["rate"]),
                                "best_rate": float(info.get("best_rate", info["rate"])),
                                "u_x": float(u[0]),
                                "u_y": float(u[1]),
                                "q_x": float(q[0]),
                                "q_y": float(q[1]),
                            }
                        )
            if self.num_timesteps % args.eval_freq != 0:
                return True
            stats = _evaluate_model(self.model, args, args.eval_episodes)
            row = {
                "timesteps": int(self.num_timesteps),
                "wall_time_sec": float(time.perf_counter() - self.started),
                **stats,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            self.model.save(eval_checkpoint_dir / f"ppo_step_{row['timesteps']:06d}.zip")
            if row["mean_final_rate"] > self.best_final_rate:
                self.best_final_rate = row["mean_final_rate"]
                self.model.save(logdir / "ppo_best_final_rate.zip")
            if row["mean_best_rate"] > self.best_rate:
                self.best_rate = row["mean_best_rate"]
                self.model.save(logdir / "ppo_best_best_rate.zip")
            print(
                f"ppo_steps={row['timesteps']} "
                f"mean_best_rate={row['mean_best_rate']:.6f} "
                f"mean_final_rate={row['mean_final_rate']:.6f} "
                f"mean_steps_to_best={row['mean_steps_to_best']:.2f}"
            )
            return True

    class LogStdAnnealCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.start_value: float | None = None

        def _on_training_start(self) -> None:
            if args.log_std_final is None or not hasattr(self.model.policy, "log_std"):
                return
            self.start_value = float(self.model.policy.log_std.detach().mean().cpu().item())

        def _on_step(self) -> bool:
            if args.log_std_final is None or self.start_value is None:
                return True
            if not hasattr(self.model.policy, "log_std"):
                return True
            horizon = args.log_std_anneal_steps or args.total_timesteps
            frac = min(1.0, self.num_timesteps / max(1, horizon))
            value = (1.0 - frac) * self.start_value + frac * float(args.log_std_final)
            self.model.policy.log_std.data.fill_(value)
            return True

    class CombinedCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.eval_callback = EvalJsonlCallback()
            self.std_callback = LogStdAnnealCallback()

        def _init_callback(self) -> None:
            self.eval_callback.init_callback(self.model)
            self.std_callback.init_callback(self.model)

        def _on_training_start(self) -> None:
            self.eval_callback.on_training_start(self.locals, self.globals)
            self.std_callback.on_training_start(self.locals, self.globals)

        def _on_step(self) -> bool:
            return bool(
                self.std_callback.on_step()
                and self.eval_callback.on_step()
            )

    train_positions = load_position_set(args.position_split_file, "train") if args.position_split_file else None
    env = Monitor(_make_env(args, positions=train_positions))
    learning_rate = (
        _linear_schedule(args.learning_rate)
        if args.lr_schedule == "linear"
        else args.learning_rate
    )
    if args.from_model:
        model = PPO.load(args.from_model, env=env, device=args.device)
        model.verbose = args.verbose
        # SB3 restores optimizer hyperparameters from the checkpoint. For staged
        # ablations we want the continuation phase to honor the command line.
        model.learning_rate = learning_rate
        model.lr_schedule = (
            learning_rate
            if callable(learning_rate)
            else (lambda _progress_remaining: float(learning_rate))
        )
        model.ent_coef = args.ent_coef
        model.clip_range = (
            _linear_schedule(args.clip_range)
            if callable(model.clip_range)
            else (lambda _progress_remaining: float(args.clip_range))
        )
        model.gamma = args.gamma
        model.gae_lambda = args.gae_lambda
        model.max_grad_norm = args.max_grad_norm
        model.n_epochs = args.n_epochs
    else:
        policy_kwargs = {}
        if args.log_std_init is not None:
            policy_kwargs["log_std_init"] = float(args.log_std_init)
        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
        )

    start = time.perf_counter()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CombinedCallback(),
        progress_bar=False,
        reset_num_timesteps=args.reset_num_timesteps,
    )
    elapsed = time.perf_counter() - start
    model_path = logdir / "ppo_fa_afdm.zip"
    model.save(model_path)
    summary = {
        "method": "ppo",
        "total_timesteps": int(args.total_timesteps),
        "wall_time_sec": float(elapsed),
        "model_path": str(model_path),
        "from_model": args.from_model,
        "reset_num_timesteps": bool(args.reset_num_timesteps),
        "best_final_rate_model_path": str(logdir / "ppo_best_final_rate.zip"),
        "best_best_rate_model_path": str(logdir / "ppo_best_best_rate.zip"),
        "metrics_path": str(metrics_path),
        "learning_rate": float(args.learning_rate),
        "lr_schedule": args.lr_schedule,
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "n_epochs": int(args.n_epochs),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "clip_range": float(args.clip_range),
        "ent_coef": float(args.ent_coef),
        "log_std_init": None if args.log_std_init is None else float(args.log_std_init),
        "log_std_final": None if args.log_std_final is None else float(args.log_std_final),
        "log_std_anneal_steps": int(args.log_std_anneal_steps),
        "reward_scale": float(args.reward_scale),
        "random_reset": bool(args.random_reset),
        "raw_rate_scale": float(args.raw_rate_scale),
        "improvement_weight": float(args.improvement_weight),
        "best_improvement_weight": float(args.best_improvement_weight),
        "rate_baseline": float(args.rate_baseline),
        "movement_penalty": float(args.movement_penalty),
        "action_l2_penalty": float(args.action_l2_penalty),
        "regression_penalty": float(args.regression_penalty),
        "best_gap_penalty": float(args.best_gap_penalty),
        "high_rate_move_penalty": float(args.high_rate_move_penalty),
        "high_rate_action_l2_penalty": float(args.high_rate_action_l2_penalty),
        "action_smooth_penalty": float(args.action_smooth_penalty),
        "high_rate_smooth_penalty": float(args.high_rate_smooth_penalty),
        "boundary_margin": float(args.boundary_margin),
        "boundary_free_dims": int(args.boundary_free_dims),
        "boundary_penalty": float(args.boundary_penalty),
        "late_rate_weight": float(args.late_rate_weight),
        "late_rate_power": float(args.late_rate_power),
        "terminal_rate_weight": float(args.terminal_rate_weight),
        "high_rate_threshold": float(args.high_rate_threshold),
        "high_rate_bonus": float(args.high_rate_bonus),
    }
    (logdir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved PPO model: {model_path}")
    print(f"Training wall time: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
