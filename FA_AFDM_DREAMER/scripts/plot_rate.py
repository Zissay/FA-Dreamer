from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fa_afdm_dreamerv3.channel import FAAFDMChannel, FAAFDMConfig


FIG_DIR = ROOT / "figures"


def load_case_curves(path: Path, method: str, max_step: int = 50) -> pd.DataFrame:
    data = pd.read_csv(path)
    rows = []
    for case, group in data.groupby("case"):
        group = group.sort_values("step")
        values = dict(zip(group["step"].astype(int), group["best_rate_so_far"].astype(float)))
        last = values[min(values)]
        for step in range(max_step + 1):
            if step in values:
                last = values[step]
            rows.append({"method": method, "case": int(case), "step": step, "best_rate": float(last)})
    return pd.DataFrame(rows)


def random_search_case_curves(
    n_cases: int,
    repeats: int = 100,
    max_step: int = 50,
    seed: int = 20260704,
) -> pd.DataFrame:
    channel = FAAFDMChannel(
        FAAFDMConfig(
            n_subcarriers=16,
            n_paths=4,
            channel_memory=5,
            noise_power_dbm=-95.0,
            channel_gain_scale=2.2839976470784646e-6,
            seed=7,
        )
    )
    rows = []
    for case in range(n_cases):
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + case * repeats + repeat)
            best = -np.inf
            random_case = case * repeats + repeat
            for step in range(max_step + 1):
                u = rng.uniform(channel.tx_bounds[0], channel.tx_bounds[1]).astype(np.float32)
                q = rng.uniform(channel.rx_bounds[0], channel.rx_bounds[1]).astype(np.float32)
                best = max(best, float(channel.rate(u, q)))
                rows.append(
                    {
                        "method": "random",
                        "case": random_case,
                        "base_case": case,
                        "repeat": repeat,
                        "step": step,
                        "best_rate": best,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    source_dir = ROOT / "runs/direct_position_rate20_four_model_midexplore_with_random_fixed"
    out = FIG_DIR
    out.mkdir(parents=True, exist_ok=True)

    mean_curves = pd.read_csv(source_dir / "four_model_midexplore_mean_curves.csv")
    mean_curves = mean_curves[mean_curves["method"] != "random_fixed"].copy()

    case_curves_path = source_dir / "four_model_midexplore_case_curves.csv"
    case_curves = pd.read_csv(case_curves_path)
    extra_sources = {}
    extra_case_curves = []
    for method, path in extra_sources.items():
        if path.exists():
            extra_case_curves.append(load_case_curves(path, method))
        else:
            print(f"missing optional 200k curve: {path}")
    if extra_case_curves:
        added_cases = pd.concat(extra_case_curves, ignore_index=True)
        added_mean = (
            added_cases.groupby(["method", "step"], as_index=False)
            .agg(best_mean=("best_rate", "mean"), best_std=("best_rate", "std"))
            .fillna({"best_std": 0.0})
        )
        case_curves = pd.concat([case_curves, added_cases], ignore_index=True)
        mean_curves = pd.concat([mean_curves, added_mean], ignore_index=True)
    n_cases = int(case_curves["case"].nunique())
    random_repeats = 100
    random_cases = random_search_case_curves(n_cases=n_cases, repeats=random_repeats)
    random_mean = (
        random_cases.groupby(["method", "step"], as_index=False)
        .agg(best_mean=("best_rate", "mean"), best_std=("best_rate", "std"))
        .fillna({"best_std": 0.0})
    )
    all_mean = pd.concat([mean_curves, random_mean], ignore_index=True)

    grid_path = ROOT / "runs/grid_rate20_noise_m95_gain2p284e6/grid_search_result.json"
    grid_best = float(json.loads(grid_path.read_text(encoding="utf-8"))["deploy_best"]["rate"])

    labels = {
        "dreamer100k": "Proposed Alg. 1 (100k)",
        "dreamer20k": "Proposed Alg. 1 (20k)",
        "ppo100k": "PPO (100k)",
        "ppo20k": "PPO (20k)",
        "random": "Random Search",
    }
    styles = {
        "dreamer100k": ("#2563eb", "-", 3.0),
        "ppo100k": ("#2563eb", "--", 3.0),
        "dreamer20k": ("#dc2626", "-", 3.0),
        "ppo20k": ("#dc2626", "--", 3.0),
        "random": ("#6b7280", "-.", 3.0),
    }
    order = ["dreamer100k", "dreamer20k", "ppo100k", "ppo20k", "random"]
    order = [method for method in order if method in set(all_mean["method"])]

    summary = (
        all_mean.sort_values("step")
        .groupby("method", as_index=False)
        .tail(1)[["method", "best_mean"]]
        .rename(columns={"best_mean": "final_mean_best_rate"})
    )
    summary["label"] = summary["method"].map(labels)
    summary["ratio_to_grid"] = summary["final_mean_best_rate"] / grid_best
    summary = summary[["method", "label", "final_mean_best_rate", "ratio_to_grid"]]

    random_cases.to_csv(out / "four_model_midexplore_random_case_curves.csv", index=False)
    all_mean.to_csv(out / "four_model_midexplore_mean_curves.csv", index=False)
    summary.to_csv(out / "four_model_midexplore_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13.5,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.linewidth": 1.0,
            "legend.frameon": True,
        }
    )
    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    for method in order:
        part = all_mean[all_mean["method"] == method].sort_values("step")
        color, linestyle, width = styles[method]
        ax.plot(
            part["step"],
            part["best_mean"],
            color=color,
            linestyle=linestyle,
            linewidth=width,
            label=labels[method],
        )
    ax.plot(
        np.arange(0, 51, 10),
        np.full(6, grid_best),
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        marker="^",
        markersize=5.2,
        markerfacecolor="white",
        markeredgecolor="#111827",
        markeredgewidth=1.0,
        label="Exhaustive Search",
        zorder=5,
    )
    ax.set_xlabel("Evaluation Steps")
    ax.set_ylabel("Achievable Rate (bps/Hz)")
    ax.set_xlim(0, 50)
    ax.set_ylim(18.50, 20.25)
    ax.set_yticks(np.arange(18.50, 20.25 + 0.001, 0.25))
    ax.grid(True, alpha=0.28)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        framealpha=0.9,
        borderpad=0.55,
        labelspacing=0.38,
        handlelength=2.4,
        handletextpad=0.7,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"rate.{suffix}")
    plt.close(fig)

    random_curve = all_mean[all_mean["method"] == "random"].sort_values("step")
    monotonic = bool((random_curve["best_mean"].diff().fillna(0) >= -1e-12).all())
    print(summary.to_string(index=False))
    print(f"random_repeats={random_repeats}")
    print(f"random_trajectories={n_cases * random_repeats}")
    print(f"random_monotonic={monotonic}")
    print(f"output_dir={out}")


if __name__ == "__main__":
    main()










