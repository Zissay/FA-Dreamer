from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from fa_afdm_dreamerv3.channel import FAAFDMChannel, FAAFDMConfig
from fa_afdm_dreamerv3.env import EnvConfig, FAAFDMEnv


def _make_env(args, seed_offset: int = 0) -> FAAFDMEnv:
    channel = FAAFDMChannel(
        FAAFDMConfig(
            n_subcarriers=args.n_subcarriers,
            n_paths=args.n_paths,
            channel_memory=args.channel_memory,
            seed=args.seed,
        )
    )
    return FAAFDMEnv(
        channel=channel,
        cfg=EnvConfig(max_steps=args.episode_steps, action_step=args.action_step, seed=args.seed + seed_offset),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO baseline on FA-AFDM.")
    parser.add_argument("--model", default="runs/ppo/ppo_fa_afdm.zip")
    parser.add_argument("--output", default="runs/ppo/evaluation.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-seed", type=int, default=7000)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--episode-steps", type=int, default=40)
    parser.add_argument("--action-step", type=float, default=0.04)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--n-subcarriers", type=int, default=16)
    parser.add_argument("--n-paths", type=int, default=4)
    parser.add_argument("--channel-memory", type=int, default=5)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise SystemExit(
            "PPO evaluation requires stable-baselines3. Install it with:\n"
            "  pip install stable-baselines3\n"
            f"Original error: {exc}"
        ) from exc

    model = PPO.load(args.model, device=args.device)
    env = _make_env(args, seed_offset=20_000)
    trajectories = []
    start = time.perf_counter()

    for episode in range(args.eval_episodes):
        obs, info = env.reset(seed=args.eval_seed + episode)
        trajectory = []
        for step in range(args.episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            trajectory.append(
                {
                    "episode": episode + 1,
                    "step": step + 1,
                    "rate": float(info["rate"]),
                    "best_rate": float(info["best_rate"]),
                    "u": [float(x) for x in info["u"]],
                    "q": [float(x) for x in info["q"]],
                    "action": np.asarray(action, dtype=float).tolist(),
                }
            )
            if terminated or truncated:
                break
        trajectories.append(trajectory)

    elapsed = time.perf_counter() - start
    flat = [item for trajectory in trajectories for item in trajectory]
    best = max(flat, key=lambda item: item["rate"])
    final = trajectories[-1][-1]
    result = {
        "method": "ppo",
        "wall_time_sec": float(elapsed),
        "num_rate_evaluations": int(len(flat)),
        "policy": {
            "final": final,
            "best": best,
            "trajectories": trajectories,
        },
        "deploy_best": {
            "rate": float(best["rate"]),
            "u": [float(x) for x in best["u"]],
            "q": [float(x) for x in best["q"]],
            "source": "ppo_rollout_best",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nPPO policy evaluation")
    print(f"best_rate_in_rollout = {best['rate']:.6f} at episode {best['episode']} step {best['step']}")
    print(f"deploy_u = [{best['u'][0]:.6f}, {best['u'][1]:.6f}]")
    print(f"deploy_q = [{best['q'][0]:.6f}, {best['q'][1]:.6f}]")
    print(f"wall_time_sec = {elapsed:.3f}")
    print(f"saved = {output}")


if __name__ == "__main__":
    main()
