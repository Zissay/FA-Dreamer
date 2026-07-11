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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4D grid search for FA-AFDM antenna positions.")
    parser.add_argument("--output-dir", default="runs/grid")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--grid-points", type=int, default=9)
    parser.add_argument("--n-subcarriers", type=int, default=16)
    parser.add_argument("--n-paths", type=int, default=4)
    parser.add_argument("--channel-memory", type=int, default=5)
    parser.add_argument("--noise-power-dbm", type=float, default=None)
    parser.add_argument("--channel-gain-scale", type=float, default=1.0)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    channel = FAAFDMChannel(
        FAAFDMConfig(
            n_subcarriers=args.n_subcarriers,
            n_paths=args.n_paths,
            channel_memory=args.channel_memory,
            noise_power_dbm=args.noise_power_dbm,
            channel_gain_scale=args.channel_gain_scale,
            seed=args.seed,
        )
    )
    tx_low, tx_high = channel.tx_bounds
    rx_low, rx_high = channel.rx_bounds
    values = np.linspace(-0.5, 0.5, args.grid_points)

    best_rate = -np.inf
    best_u = np.zeros(2, dtype=np.float32)
    best_q = np.zeros(2, dtype=np.float32)
    curve_path = outdir / "grid_search_curve.csv"
    start = time.perf_counter()
    eval_index = 0

    with curve_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "evaluation",
                "elapsed_time_sec",
                "rate",
                "best_so_far_rate",
                "u_x",
                "u_y",
                "q_x",
                "q_y",
                "best_u_x",
                "best_u_y",
                "best_q_x",
                "best_q_y",
            ],
        )
        writer.writeheader()
        for ux in values:
            for uy in values:
                for qx in values:
                    for qy in values:
                        eval_index += 1
                        u = np.array([ux * tx_high[0] * 2.0, uy * tx_high[1] * 2.0], dtype=np.float32)
                        q = np.array([qx * rx_high[0] * 2.0, qy * rx_high[1] * 2.0], dtype=np.float32)
                        rate = channel.rate(u, q)
                        if rate > best_rate:
                            best_rate = rate
                            best_u = u
                            best_q = q
                        writer.writerow(
                            {
                                "evaluation": eval_index,
                                "elapsed_time_sec": time.perf_counter() - start,
                                "rate": rate,
                                "best_so_far_rate": best_rate,
                                "u_x": float(u[0]),
                                "u_y": float(u[1]),
                                "q_x": float(q[0]),
                                "q_y": float(q[1]),
                                "best_u_x": float(best_u[0]),
                                "best_u_y": float(best_u[1]),
                                "best_q_x": float(best_q[0]),
                                "best_q_y": float(best_q[1]),
                            }
                        )

    elapsed = time.perf_counter() - start
    result = {
        "method": "grid_search",
        "grid_points": int(args.grid_points),
        "num_rate_evaluations": int(eval_index),
        "wall_time_sec": float(elapsed),
        "deploy_best": {
            "rate": float(best_rate),
            "u": [float(x) for x in best_u],
            "q": [float(x) for x in best_q],
            "source": "grid_search",
        },
        "curve_path": str(curve_path),
    }
    result_path = outdir / "grid_search_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nGrid search result")
    print(f"best_rate = {best_rate:.6f}")
    print(f"best_u = [{best_u[0]:.6f}, {best_u[1]:.6f}]")
    print(f"best_q = [{best_q[0]:.6f}, {best_q[1]:.6f}]")
    print(f"evaluations = {eval_index}")
    print(f"wall_time_sec = {elapsed:.3f}")
    print(f"saved = {result_path}")


if __name__ == "__main__":
    main()
