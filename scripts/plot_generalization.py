from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "figures"
MAX_STEP = 20

SOURCES = [
    (
        "distance",
        "Delay Error",
        "#2563eb",
        ROOT / "runs/generalization_200k_distance_doppler_paths/distance/d080/dreamer/hitting_time_trajectories_dreamerv3_gpu.csv",
        ROOT / "runs/generalization_200k_distance_doppler_paths/distance/d080/ppo/hitting_time_trajectories_ppo.csv",
    ),
    (
        "doppler",
        "Doppler Error",
        "#dc2626",
        ROOT / "runs/generalization_200k_distance_doppler_paths/doppler/f050/dreamer/hitting_time_trajectories_dreamerv3_gpu.csv",
        ROOT / "runs/generalization_200k_distance_doppler_paths/doppler/f050/ppo/hitting_time_trajectories_ppo.csv",
    ),
    (
        "paths",
        "Path Error",
        "#111111",
        ROOT / "runs/generalization_200k_hard_ood/paths/p08/dreamer/hitting_time_trajectories_dreamerv3_gpu.csv",
        ROOT / "runs/generalization_200k_hard_ood/paths/p08/ppo/hitting_time_trajectories_ppo.csv",
    ),
]


def load_curve(path: Path, factor: str, label: str, method: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    rows = []
    for case, group in data.groupby("case"):
        group = group.sort_values("step")
        values = dict(zip(group["step"].astype(int), group["best_rate_so_far"].astype(float)))
        last = values[min(values)]
        for step in range(MAX_STEP + 1):
            if step in values:
                last = values[step]
            rows.append(
                {
                    "factor": factor,
                    "label": label,
                    "method": method,
                    "case": int(case),
                    "step": step,
                    "best_rate": float(last),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for factor, label, _color, dreamer_path, ppo_path in SOURCES:
        frames.append(load_curve(dreamer_path, factor, label, "dreamer"))
        frames.append(load_curve(ppo_path, factor, label, "ppo"))
    curves = pd.concat(frames, ignore_index=True)
    mean = (
        curves.groupby(["factor", "label", "method", "step"], as_index=False)
        .agg(best_mean=("best_rate", "mean"), best_std=("best_rate", "std"))
        .fillna({"best_std": 0.0})
    )
    curves.to_csv(OUT / "selected_bestsofar_20step_case_curves.csv", index=False)
    mean.to_csv(OUT / "selected_bestsofar_20step_mean_curves.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13.0,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.linewidth": 1.0,
            "legend.frameon": True,
        }
    )
    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    summary_rows = []
    legend_handles = {}
    for factor, label, color, _dreamer_path, _ppo_path in SOURCES:
        for method, linestyle, method_label in [("dreamer", "-", "Proposed Alg. 1"), ("ppo", "--", "PPO")]:
            group = mean[(mean["factor"] == factor) & (mean["method"] == method)].sort_values("step")
            (line,) = ax.plot(
                group["step"],
                group["best_mean"],
                color=color,
                linestyle=linestyle,
                linewidth=3.0,
                label=f"{method_label}, {label}",
            )
            legend_handles[(label, method)] = line
            final = float(group[group["step"] == MAX_STEP]["best_mean"].iloc[0])
            summary_rows.append({"factor": factor, "label": label, "method": method, "best_mean_step20": final})
    ax.set_xlim(0, MAX_STEP)
    ax.set_xlabel("Evaluation Steps")
    ax.set_ylabel("Achievable Rate (bps/Hz)")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.label.set_size(13)
    ax.tick_params(axis="y", pad=6)
    ax.grid(True, alpha=0.28)
    legend_order = []
    legend_labels = []
    # Matplotlib fills multi-column legends column-wise. Put all proposed
    # algorithm entries first so the left column corresponds to Proposed Alg. 1.
    for method in ["dreamer", "ppo"]:
        for _factor, label, _color, _dreamer_path, _ppo_path in SOURCES:
            legend_order.append(legend_handles[(label, method)])
            legend_labels.append(legend_handles[(label, method)].get_label())
    ax.legend(
        legend_order,
        legend_labels,
        frameon=True,
        loc="center",
        bbox_to_anchor=(0.62, 0.35),
        ncol=2,
        borderpad=0.45,
        labelspacing=0.32,
        handlelength=1.5,
        handletextpad=0.45,
        columnspacing=0.45,
    )
    ax.set_position(
        [
            65.913295 / 576.0,
            (432.0 - 18.458182 - 365.592727) / 432.0,
            489.542159 / 576.0,
            365.592727 / 432.0,
        ]
    )
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"generalization.{suffix}")
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    pivot = summary.pivot(index=["factor", "label"], columns="method", values="best_mean_step20").reset_index()
    pivot["dreamer_minus_ppo"] = pivot["dreamer"] - pivot["ppo"]
    pivot.to_csv(OUT / "selected_bestsofar_20step_summary.csv", index=False)
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
