from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from fa_afdm_dreamerv3.official_env import register_env
from fa_afdm_dreamerv3.position_splits import load_position_set


def _check_jax_backend(platform: str, allow_cpu_fallback: bool) -> str:
    if platform == "auto":
        try:
            import jax

            devices = jax.devices()
            if any(device.platform == "gpu" for device in devices):
                return "cuda"
            return "cpu"
        except Exception:
            return "cpu"

    if platform in {"cuda", "gpu"}:
        try:
            import jax

            devices = jax.devices("gpu")
        except Exception as exc:
            if allow_cpu_fallback:
                print(f"CUDA JAX backend is unavailable; falling back to CPU. Original error: {exc}")
                return "cpu"
            raise SystemExit(
                "CUDA JAX backend is unavailable. Official DreamerV3 uses JAX, so GPU training needs "
                "a CUDA-enabled JAX/JAXLIB installation, usually easiest under Linux or WSL2.\n"
                "Install a compatible CUDA JAX build, or rerun with `--jax-platform cpu`."
            ) from exc
        if not devices:
            if allow_cpu_fallback:
                print("No JAX GPU devices found; falling back to CPU.")
                return "cpu"
            raise SystemExit("No JAX GPU devices found. Rerun with `--jax-platform cpu` or install CUDA JAX.")
        print("Using JAX GPU devices:", devices)
        return "cuda"

    return platform


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train FA-AFDM antenna-position optimization with official dreamerv3==1.3.0."
    )
    parser.add_argument("--logdir", default="runs/official_fa_afdm")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--from-checkpoint", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channel-seed", type=int, default=None)
    parser.add_argument("--env-seed", type=int, default=None)
    parser.add_argument("--configs", default="debug", help="Use debug for CPU smoke tests; use small for longer runs.")
    parser.add_argument("--train-ratio", type=float, default=32.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-length", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--n-subcarriers", type=int, default=16)
    parser.add_argument("--n-paths", type=int, default=4)
    parser.add_argument("--channel-memory", type=int, default=5)
    parser.add_argument("--noise-power-dbm", type=float, default=None)
    parser.add_argument("--channel-gain-scale", type=float, default=1.0)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument(
        "--initial-u",
        type=float,
        nargs=2,
        default=None,
        help="Optional fixed initial transmit FA position. Providing both --initial-u and --initial-q disables random reset.",
    )
    parser.add_argument(
        "--initial-q",
        type=float,
        nargs=2,
        default=None,
        help="Optional fixed initial receive FA position. Providing both --initial-u and --initial-q disables random reset.",
    )
    parser.add_argument("--action-step", type=float, default=0.04)
    parser.add_argument(
        "--direct-position-action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interpret actions as full normalized antenna positions instead of position increments.",
    )
    parser.add_argument("--high-rate-threshold", type=float, default=0.0)
    parser.add_argument("--high-rate-bonus", type=float, default=0.0)
    parser.add_argument("--high-rate-slope", type=float, default=1.0)
    parser.add_argument("--rate-baseline", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--raw-rate-scale", type=float, default=0.0)
    parser.add_argument("--improvement-weight", type=float, default=0.5)
    parser.add_argument("--best-improvement-weight", type=float, default=0.0)
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
    parser.add_argument(
        "--observe-best-position",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append the best measured position in the current episode to the observation vector.",
    )
    parser.add_argument(
        "--observe-rate-dynamics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append normalized one-step rate change and best-current rate gap to the observation vector.",
    )
    parser.add_argument(
        "--position-split-file",
        default="",
        help="JSON file with train/eval initial-position splits. Dreamer training samples the train split.",
    )
    parser.add_argument(
        "--query-log",
        default="",
        help="Optional CSV path for logging every real training environment query.",
    )
    parser.add_argument(
        "--jax-platform",
        default="cuda",
        choices=["auto", "cpu", "cuda", "gpu"],
        help="JAX backend for official DreamerV3. Use cuda/gpu for GPU training.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Fall back to CPU if --jax-platform cuda is requested but unavailable.",
    )
    args, extra = parser.parse_known_args()

    # Gym 0.26 still references np.bool8, which NumPy 2.x removed.
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_

    train_positions = load_position_set(args.position_split_file, "train") if args.position_split_file else None
    register_env(
        seed=args.seed,
        channel_seed=args.channel_seed,
        env_seed=args.env_seed,
        n_subcarriers=args.n_subcarriers,
        n_paths=args.n_paths,
        channel_memory=args.channel_memory,
        noise_power_dbm=args.noise_power_dbm,
        channel_gain_scale=args.channel_gain_scale,
        max_steps=args.episode_steps,
        action_step=args.action_step,
        direct_position_action=args.direct_position_action,
        initial_u=args.initial_u,
        initial_q=args.initial_q,
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
        observe_best_position=args.observe_best_position,
        observe_rate_dynamics=args.observe_rate_dynamics,
        initial_positions=train_positions,
        query_log_path=args.query_log,
    )
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    # Official DreamerV3 uses its own flag parser. We call it programmatically
    # after registering the Gym environment so task=gym_FAAFDM-v0 can be made.
    dreamer_argv = [
        "--configs",
        args.configs,
        "--task",
        "gym_FAAFDM-v0",
        "--run.logdir",
        args.logdir,
        "--run.steps",
        str(args.steps),
        "--run.from_checkpoint",
        args.from_checkpoint,
        "--run.train_ratio",
        str(args.train_ratio),
        "--run.train_fill",
        str(max(args.batch_size * args.batch_length, 100)),
        "--run.log_every",
        str(args.log_every),
        "--run.save_every",
        str(args.save_every),
        "--run.eval_every",
        str(args.eval_every),
        "--batch_size",
        str(args.batch_size),
        "--batch_length",
        str(args.batch_length),
        "--envs.amount",
        "1",
        "--envs.parallel",
        "none",
        "--envs.restart",
        "False",
        "--wrapper.length",
        str(args.episode_steps),
        "--jax.platform",
        "cpu",
        "--jax.prealloc",
        "False",
        "--encoder.mlp_keys",
        "vector",
        "--encoder.cnn_keys",
        "$^",
        "--decoder.mlp_keys",
        "vector",
        "--decoder.cnn_keys",
        "$^",
    ]
    dreamer_argv.extend(extra)

    # DreamerV3 1.3.0's custom Path helper only splits on '/', so on Windows
    # package-relative reads of configs.yaml can fall back to the CWD during
    # package import. Copy the package config before importing dreamerv3.
    spec = importlib.util.find_spec("dreamerv3")
    if spec and spec.submodule_search_locations:
        package_config = Path(next(iter(spec.submodule_search_locations))) / "configs.yaml"
        cwd_config = Path("configs.yaml")
        if package_config.exists() and not cwd_config.exists():
            shutil.copyfile(package_config, cwd_config)

    try:
        from dreamerv3 import train as official_train
    except Exception as exc:
        raise SystemExit(
            "Could not import official dreamerv3. Install it first, preferably "
            "in Python 3.10 or 3.11:\n\n"
            "  pip install dreamerv3==1.3.0\n\n"
            f"Original error: {exc}"
        ) from exc

    jax_platform = _check_jax_backend(args.jax_platform, args.allow_cpu_fallback)
    sys.argv = [sys.argv[0], *dreamer_argv]
    for index, value in enumerate(sys.argv):
        if value == "--jax.platform":
            sys.argv[index + 1] = jax_platform
            break
    official_train.main()


if __name__ == "__main__":
    main()
