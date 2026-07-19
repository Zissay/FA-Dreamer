from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from fa_afdm_dreamerv3.channel import FAAFDMChannel, FAAFDMConfig
from fa_afdm_dreamerv3.official_env import DreamerV3FAAFDMGymEnv, register_env


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
                "CUDA JAX backend is unavailable. Official DreamerV3 uses JAX, so GPU evaluation needs "
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


def _prepare_windows_compat() -> None:
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_
    spec = importlib.util.find_spec("dreamerv3")
    if spec and spec.submodule_search_locations:
        package_config = Path(next(iter(spec.submodule_search_locations))) / "configs.yaml"
        cwd_config = Path("configs.yaml")
        if package_config.exists() and not cwd_config.exists():
            shutil.copyfile(package_config, cwd_config)


def _build_config(argv: list[str]):
    from dreamerv3 import agent as agt
    import embodied

    parsed, other = embodied.Flags(configs=["defaults"]).parse_known(argv)
    config = embodied.Config(agt.Agent.configs["defaults"])
    for name in parsed.configs:
        config = config.update(agt.Agent.configs[name])
    config = embodied.Flags(config).parse(other)
    return config


def _make_agent_and_env(args):
    _prepare_windows_compat()
    jax_platform = _check_jax_backend(args.jax_platform, args.allow_cpu_fallback)
    register_env(
        seed=args.seed,
        n_subcarriers=args.n_subcarriers,
        n_paths=args.n_paths,
        channel_memory=args.channel_memory,
        noise_power_dbm=args.noise_power_dbm,
        channel_gain_scale=args.channel_gain_scale,
        doppler_scale=getattr(args, "doppler_scale", 1.0),
        max_steps=args.episode_steps,
        action_step=args.action_step,
        direct_position_action=getattr(args, "direct_position_action", False),
        initial_u=args.initial_u,
        initial_q=args.initial_q,
        rate_baseline=getattr(args, "rate_baseline", 0.0),
        reward_scale=getattr(args, "reward_scale", 1.0),
        raw_rate_scale=getattr(args, "raw_rate_scale", 0.0),
        improvement_weight=getattr(args, "improvement_weight", 0.5),
        action_l2_penalty=getattr(args, "action_l2_penalty", 0.0),
        high_rate_action_l2_penalty=getattr(args, "high_rate_action_l2_penalty", 0.0),
        boundary_margin=getattr(args, "boundary_margin", 1.0),
        boundary_free_dims=getattr(args, "boundary_free_dims", 0),
        boundary_penalty=getattr(args, "boundary_penalty", 0.0),
        late_rate_weight=getattr(args, "late_rate_weight", 0.0),
        late_rate_power=getattr(args, "late_rate_power", 1.0),
        terminal_rate_weight=getattr(args, "terminal_rate_weight", 0.0),
        observe_best_position=getattr(args, "observe_best_position", False),
        observe_rate_dynamics=getattr(args, "observe_rate_dynamics", False),
    )

    from dreamerv3 import agent as agt
    from dreamerv3 import train as official_train
    import embodied
    from embodied.envs.from_gym import FromGym

    config_argv = [
        "--configs",
        args.configs,
        "--task",
        "gym_FAAFDM-v0",
        "--run.logdir",
        args.logdir,
        "--run.steps",
        str(args.steps),
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
        jax_platform,
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
    config_argv.extend(getattr(args, "extra_args", []) or [])
    config = _build_config(config_argv)

    raw_env = DreamerV3FAAFDMGymEnv(
        seed=args.seed,
        n_subcarriers=args.n_subcarriers,
        n_paths=args.n_paths,
        channel_memory=args.channel_memory,
        noise_power_dbm=args.noise_power_dbm,
        channel_gain_scale=args.channel_gain_scale,
        doppler_scale=getattr(args, "doppler_scale", 1.0),
        max_steps=args.episode_steps,
        action_step=args.action_step,
        direct_position_action=getattr(args, "direct_position_action", False),
        initial_u=args.initial_u,
        initial_q=args.initial_q,
        rate_baseline=getattr(args, "rate_baseline", 0.0),
        reward_scale=getattr(args, "reward_scale", 1.0),
        raw_rate_scale=getattr(args, "raw_rate_scale", 0.0),
        improvement_weight=getattr(args, "improvement_weight", 0.5),
        action_l2_penalty=getattr(args, "action_l2_penalty", 0.0),
        high_rate_action_l2_penalty=getattr(args, "high_rate_action_l2_penalty", 0.0),
        boundary_margin=getattr(args, "boundary_margin", 1.0),
        boundary_free_dims=getattr(args, "boundary_free_dims", 0),
        boundary_penalty=getattr(args, "boundary_penalty", 0.0),
        late_rate_weight=getattr(args, "late_rate_weight", 0.0),
        late_rate_power=getattr(args, "late_rate_power", 1.0),
        terminal_rate_weight=getattr(args, "terminal_rate_weight", 0.0),
        observe_best_position=getattr(args, "observe_best_position", False),
        observe_rate_dynamics=getattr(args, "observe_rate_dynamics", False),
    )
    env = FromGym(raw_env)
    env = official_train.wrap_env(env, config)
    batch_env = embodied.BatchEnv([env], parallel=False)

    step = embodied.Counter()
    agent = agt.Agent(batch_env.obs_space, batch_env.act_space, step, config)
    checkpoint = embodied.Checkpoint(log=True)
    checkpoint.agent = agent
    checkpoint.load(args.checkpoint, keys=["agent"])
    return agent, batch_env, raw_env


def _policy_rollout(args) -> dict:
    agent, env, raw_env = _make_agent_and_env(args)
    state = None
    trajectories = []

    def run_episode(episode: int, record: bool) -> list[dict]:
        nonlocal state
        action = {
            "reset": np.array([True]),
            "action": np.zeros((1, 4), dtype=np.float32),
        }
        obs = env.step(action)
        state = None
        trajectory = []
        for step in range(args.episode_steps):
            policy_action, state = agent.policy(obs, state, mode="eval")
            policy_action = {k: v for k, v in policy_action.items() if k in env.act_space}
            policy_action.setdefault("reset", np.array([False]))
            obs = env.step(policy_action)
            if record:
                info = raw_env._env._info()
                trajectory.append(
                    {
                        "episode": episode + 1,
                        "step": step + 1,
                        "rate": float(info["rate"]),
                        "best_rate": float(info["best_rate"]),
                        "u": [float(x) for x in info["u"]],
                        "q": [float(x) for x in info["q"]],
                        "action": np.asarray(policy_action["action"][0], dtype=float).tolist(),
                    }
                )
            if bool(obs["is_last"][0]):
                break
        return trajectory

    for episode in range(args.warmup_episodes):
        run_episode(episode, record=False)

    start = time.perf_counter()
    for episode in range(args.eval_episodes):
        trajectory = run_episode(episode, record=True)
        trajectories.append(trajectory)
    elapsed = time.perf_counter() - start
    env.close()
    flat = [item for trajectory in trajectories for item in trajectory]
    return {
        "final": trajectories[-1][-1],
        "best": max(flat, key=lambda item: item["rate"]),
        "trajectories": trajectories,
        "_wall_time_sec": float(elapsed),
    }


def _make_channel(args) -> FAAFDMChannel:
    return FAAFDMChannel(
        FAAFDMConfig(
            n_subcarriers=args.n_subcarriers,
            n_paths=args.n_paths,
            channel_memory=args.channel_memory,
            noise_power_dbm=args.noise_power_dbm,
            channel_gain_scale=args.channel_gain_scale,
            doppler_scale=getattr(args, "doppler_scale", 1.0),
            seed=args.seed,
        )
    )


def _random_search(args) -> dict:
    channel = _make_channel(args)
    rng = np.random.default_rng(args.search_seed)
    tx_low, tx_high = channel.tx_bounds
    rx_low, rx_high = channel.rx_bounds
    best_rate = channel.rate(np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    best_u = np.zeros(2, dtype=np.float32)
    best_q = np.zeros(2, dtype=np.float32)
    for _ in range(args.random_samples):
        u = rng.uniform(tx_low, tx_high).astype(np.float32)
        q = rng.uniform(rx_low, rx_high).astype(np.float32)
        rate = channel.rate(u, q)
        if rate > best_rate:
            best_rate = rate
            best_u = u
            best_q = q
    return {
        "rate": float(best_rate),
        "u": [float(x) for x in best_u],
        "q": [float(x) for x in best_q],
        "samples": int(args.random_samples),
    }


def _local_refine(args, starts: list[dict]) -> dict:
    channel = FAAFDMChannel(
        FAAFDMConfig(
            n_subcarriers=args.n_subcarriers,
            n_paths=args.n_paths,
            channel_memory=args.channel_memory,
            noise_power_dbm=args.noise_power_dbm,
            channel_gain_scale=args.channel_gain_scale,
            doppler_scale=getattr(args, "doppler_scale", 1.0),
            seed=args.seed,
        )
    )
    rng = np.random.default_rng(args.search_seed)
    tx_low, tx_high = channel.tx_bounds
    rx_low, rx_high = channel.rx_bounds

    candidates = []
    for start in starts:
        candidates.append(
            {
                "rate": channel.rate(np.array(start["u"], dtype=np.float32), np.array(start["q"], dtype=np.float32)),
                "u": np.array(start["u"], dtype=np.float32),
                "q": np.array(start["q"], dtype=np.float32),
                "source": start.get("source", "start"),
            }
        )
    best = max(candidates, key=lambda item: item["rate"])

    radii = np.geomspace(args.refine_radius, args.refine_radius / 20.0, args.refine_rounds)
    per_round = max(1, args.refine_samples // args.refine_rounds)
    for radius in radii:
        anchors = sorted(candidates, key=lambda item: item["rate"], reverse=True)[: args.refine_anchors]
        for anchor in anchors:
            for _ in range(per_round):
                u = np.clip(anchor["u"] + rng.normal(0.0, radius, size=2), tx_low, tx_high).astype(np.float32)
                q = np.clip(anchor["q"] + rng.normal(0.0, radius, size=2), rx_low, rx_high).astype(np.float32)
                rate = channel.rate(u, q)
                item = {"rate": float(rate), "u": u, "q": q, "source": "local_refine"}
                candidates.append(item)
                if rate > best["rate"]:
                    best = item

    # Also test small deterministic coordinate moves around the best candidate.
    steps = np.linspace(-args.coordinate_radius, args.coordinate_radius, args.coordinate_points)
    anchor_u = best["u"].copy()
    anchor_q = best["q"].copy()
    for dim in range(4):
        for delta in steps:
            x = np.concatenate([anchor_u, anchor_q])
            x[dim] += delta
            u = np.clip(x[:2], tx_low, tx_high).astype(np.float32)
            q = np.clip(x[2:], rx_low, rx_high).astype(np.float32)
            rate = channel.rate(u, q)
            if rate > best["rate"]:
                best = {"rate": float(rate), "u": u, "q": q, "source": "coordinate_refine"}

    return {
        "rate": float(best["rate"]),
        "u": [float(x) for x in best["u"]],
        "q": [float(x) for x in best["q"]],
        "source": best["source"],
        "samples": int(args.refine_samples),
    }


def _coarse_grid(args) -> dict:
    channel = _make_channel(args)
    tx_low, tx_high = channel.tx_bounds
    rx_low, rx_high = channel.rx_bounds
    values = np.linspace(-0.5, 0.5, args.grid_points)
    best_rate = -np.inf
    best_u = np.zeros(2, dtype=np.float32)
    best_q = np.zeros(2, dtype=np.float32)
    for ux in values:
        for uy in values:
            for qx in values:
                for qy in values:
                    u = np.array([ux * tx_high[0] * 2.0, uy * tx_high[1] * 2.0], dtype=np.float32)
                    q = np.array([qx * rx_high[0] * 2.0, qy * rx_high[1] * 2.0], dtype=np.float32)
                    rate = channel.rate(u, q)
                    if rate > best_rate:
                        best_rate = rate
                        best_u = u
                        best_q = q
    return {
        "rate": float(best_rate),
        "u": [float(x) for x in best_u],
        "q": [float(x) for x in best_q],
        "points_per_dim": int(args.grid_points),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate official DreamerV3 checkpoint on FA-AFDM optimization.")
    parser.add_argument("--checkpoint", default="runs/official_train_test/checkpoint.pkl")
    parser.add_argument("--logdir", default="runs/official_eval")
    parser.add_argument("--configs", default="debug")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-subcarriers", type=int, default=16)
    parser.add_argument("--n-paths", type=int, default=4)
    parser.add_argument("--channel-memory", type=int, default=5)
    parser.add_argument("--noise-power-dbm", type=float, default=-95.0)
    parser.add_argument("--channel-gain-scale", type=float, default=2.280350850198276e-6)
    parser.add_argument("--doppler-scale", type=float, default=1.0)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument(
        "--warmup-episodes",
        type=int,
        default=0,
        help="Run unmeasured policy episodes before timing, useful to exclude JAX compilation from deployment time.",
    )
    parser.add_argument("--episode-steps", type=int, default=40)
    parser.add_argument("--action-step", type=float, default=0.04)
    parser.add_argument(
        "--direct-position-action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interpret actions as full normalized antenna positions instead of position increments.",
    )
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--raw-rate-scale", type=float, default=0.0)
    parser.add_argument("--improvement-weight", type=float, default=0.5)
    parser.add_argument("--boundary-margin", type=float, default=1.0)
    parser.add_argument("--boundary-free-dims", type=int, default=0)
    parser.add_argument("--boundary-penalty", type=float, default=0.0)
    parser.add_argument("--late-rate-weight", type=float, default=0.0)
    parser.add_argument("--late-rate-power", type=float, default=1.0)
    parser.add_argument("--terminal-rate-weight", type=float, default=0.0)
    parser.add_argument("--observe-best-position", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--observe-rate-dynamics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--initial-u", type=float, nargs=2, default=None)
    parser.add_argument("--initial-q", type=float, nargs=2, default=None)
    parser.add_argument(
        "--jax-platform",
        default="cuda",
        choices=["auto", "cpu", "cuda", "gpu"],
        help="JAX backend for official DreamerV3 evaluation.",
    )
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--compare-baselines", action="store_true")
    parser.add_argument("--random-samples", type=int, default=20000)
    parser.add_argument("--grid-points", type=int, default=9)
    parser.add_argument("--refine-samples", type=int, default=10000)
    parser.add_argument("--refine-radius", type=float, default=0.08)
    parser.add_argument("--refine-rounds", type=int, default=4)
    parser.add_argument("--refine-anchors", type=int, default=3)
    parser.add_argument("--coordinate-radius", type=float, default=0.04)
    parser.add_argument("--coordinate-points", type=int, default=17)
    parser.add_argument("--search-seed", type=int, default=123)
    parser.add_argument("--output", default="runs/official_eval/evaluation_dreamerv3_200eps.json")
    args = parser.parse_args()

    policy = _policy_rollout(args)
    elapsed = policy.pop("_wall_time_sec")
    random_search = _random_search(args) if args.compare_baselines else None
    coarse_grid = _coarse_grid(args) if args.compare_baselines else None
    deploy_best = {**policy["best"], "source": "dreamerv3_rollout_best"}

    result = {
        "method": "official_dreamerv3",
        "wall_time_sec": float(elapsed),
        "num_rate_evaluations": int(sum(len(trajectory) for trajectory in policy["trajectories"])),
        "policy": policy,
        "deploy_best": {
            "rate": float(deploy_best["rate"]),
            "u": [float(x) for x in deploy_best["u"]],
            "q": [float(x) for x in deploy_best["q"]],
            "source": deploy_best["source"],
        },
    }
    if args.compare_baselines:
        result["random_search"] = random_search
        result["coarse_grid"] = coarse_grid
        result["gap_to_random_best_final"] = float(random_search["rate"] - result["policy"]["final"]["rate"])
        result["gap_to_random_best_rollout_best"] = float(random_search["rate"] - result["policy"]["best"]["rate"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    final = result["policy"]["final"]
    best = result["policy"]["best"]
    deploy = result["deploy_best"]
    print("\nOfficial DreamerV3 policy evaluation")
    print(f"final_rate = {final['rate']:.6f}")
    print(f"best_rate_in_rollout = {best['rate']:.6f} at episode {best['episode']} step {best['step']}")
    print(f"u = [{final['u'][0]:.6f}, {final['u'][1]:.6f}]")
    print(f"q = [{final['q'][0]:.6f}, {final['q'][1]:.6f}]")
    print(f"best_u = [{best['u'][0]:.6f}, {best['u'][1]:.6f}]")
    print(f"best_q = [{best['q'][0]:.6f}, {best['q'][1]:.6f}]")
    if args.compare_baselines:
        random_best = result["random_search"]
        grid_best = result["coarse_grid"]
        print("\nRandom-search baseline, comparison only")
        print(f"best_rate = {random_best['rate']:.6f} from {random_best['samples']} samples")
        print(f"u = [{random_best['u'][0]:.6f}, {random_best['u'][1]:.6f}]")
        print(f"q = [{random_best['q'][0]:.6f}, {random_best['q'][1]:.6f}]")
        print("\nCoarse-grid baseline, comparison only")
        print(f"best_rate = {grid_best['rate']:.6f} with {grid_best['points_per_dim']} points per dimension")
        print(f"u = [{grid_best['u'][0]:.6f}, {grid_best['u'][1]:.6f}]")
        print(f"q = [{grid_best['q'][0]:.6f}, {grid_best['q'][1]:.6f}]")
        print(f"\ngap_to_random_best_final = {result['gap_to_random_best_final']:.6f}")
        print(f"gap_to_random_best_rollout_best = {result['gap_to_random_best_rollout_best']:.6f}")
    print("\nDreamerV3-only deployment recommendation")
    print(f"deploy_rate = {deploy['rate']:.6f} source={deploy['source']}")
    print(f"deploy_u = [{deploy['u'][0]:.6f}, {deploy['u'][1]:.6f}]")
    print(f"deploy_q = [{deploy['q'][0]:.6f}, {deploy['q'][1]:.6f}]")
    print(f"saved = {output}")
    print(f"wall_time_sec = {elapsed:.3f}")


if __name__ == "__main__":
    main()
