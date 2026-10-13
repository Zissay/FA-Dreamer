"""Entry point for preserved 50/500 km/h experiments; legacy scripts remain available."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="Check archived terminal records, source hashes and references")
    verify.add_argument("--physics", action="store_true", help="Recompute 100 random positions and each stored best position on CPU")
    train = sub.add_parser("train", help="Start a fresh preserved training run")
    train.add_argument("--speed", type=int, choices=(50, 500), required=True)
    train.add_argument("--algorithm", choices=("dreamer", "ppo"), required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--python", help="Trainer interpreter; defaults to the current interpreter")
    train.add_argument("--gpu-python", help="GPU rate-server interpreter; defaults to trainer interpreter")
    train.add_argument("--paper-prefix", action="store_true", help="Stop the original 500k 50 km/h PPO protocol at the plotted 50k prefix")
    train.add_argument("--dry-run", action="store_true", help="Print commands without training or creating a run")
    plot = sub.add_parser("plot", help="Reconstruct one training panel from preserved or new online logs")
    plot.add_argument("--speed", type=int, choices=(50, 500), required=True)
    plot.add_argument("--dreamer-run", type=Path)
    plot.add_argument("--ppo-run", type=Path)
    plot.add_argument("--output-dir", type=Path, default=ROOT / "figures/reproduced")
    plot.add_argument("--font-dir", type=Path)
    all_plots = sub.add_parser("plot-all", help="Reconstruct the three independent paper panels")
    all_plots.add_argument("--output-dir", type=Path, default=ROOT / "figures/reproduced")
    all_plots.add_argument("--font-dir", type=Path)
    for speed in (50, 500):
        for algorithm in ("dreamer", "ppo"):
            all_plots.add_argument(f"--{algorithm}-{speed}-run", type=Path)
    baseline = sub.add_parser("baseline", help="Recompute an independent grid/random reference")
    baseline.add_argument("--speed", type=int, choices=(50, 500), required=True)
    baseline.add_argument("--method", choices=("random", "daft", "time"), required=True)
    baseline.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    baseline.add_argument("--grid", type=int, help="Override the default 81/21 grid; output is explicitly labelled")
    baseline.add_argument("--batch", type=int, default=128)
    baseline.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        from reproduction_tools.verify import verify_all
        print(json.dumps(verify_all(physics=args.physics), indent=2))
    elif args.command == "train":
        from reproduction_tools.training import launch
        launch(args.speed, args.algorithm, args.run_dir, args.python, args.gpu_python, args.paper_prefix, args.dry_run)
    elif args.command == "plot":
        from reproduction_tools.plotting import plot_training
        plot_training(args.speed, args.output_dir, args.dreamer_run, args.ppo_run, args.font_dir)
    elif args.command == "plot-all":
        from reproduction_tools.plotting import plot_training, plot_bar
        runs = [getattr(args, f"{a}_{s}_run") for s in (50, 500) for a in ("dreamer", "ppo")]
        if any(r is not None for r in runs) and not all(r is not None for r in runs):
            parser.error("Provide all four new run directories, or omit all to use recorded results")
        results = {s: plot_training(s, args.output_dir, getattr(args, f"dreamer_{s}_run"),
                                   getattr(args, f"ppo_{s}_run"), args.font_dir) for s in (50, 500)}
        plot_bar(results, args.output_dir, args.font_dir)
    elif args.command == "baseline":
        from reproduction_tools.baselines import search
        result = search(args.speed, args.method, args.output, args.device, args.batch, args.grid)
        print(json.dumps({k: result[k] for k in ("method", "speed_kmh", "status")}, indent=2))


if __name__ == "__main__":
    main()
