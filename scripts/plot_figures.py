"""Draw the paper's training-curve figure from two finished runs.

  python scripts/plot_figures.py --dreamer runs/dreamer --ppo runs/ppo --output figures

Writes training_curve.pdf. Curves: mean of the 20 online terminal rates in each
1K-interaction batch, then a causal trailing mean of up to five batches. Final rate:
mean of the last 100 online terminal rates (last 5K interactions).
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
PAGE = (570.41, 406.5)                # points
FRAME = (62.3, 38.5, 490.6, 365.4)    # left, bottom, width, height (points)
LINE = {"proposed": "#036EAD", "ppo": "#DD656D", "daft": "#F3962F", "time": "#61A7D0", "random": "#8798C9"}


def font(size, bold=False):
    """Times New Roman (regular/bold) when installed, otherwise a serif fallback."""
    for folder in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/msttcorefonts")):
        path = folder / ("timesbd.ttf" if bold else "times.ttf")
        if path.is_file():
            return FontProperties(fname=str(path), size=size)
    return FontProperties(family="serif", size=size, weight="bold" if bold else "normal")


def terminal_rates(run):
    batches = collections.defaultdict(list)
    with (Path(run) / "episode_terminal_rates.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            batches[int(row["steps"])].append(float(row["terminal_rate"]))
    steps = sorted(batches)
    assert steps == list(range(1000, 50001, 1000)) and all(len(batches[s]) == 20 for s in steps), run
    means = np.array([np.mean(batches[s]) for s in steps])
    shown = np.array([means[max(0, i - 4):i + 1].mean() for i in range(len(means))])
    final = float(np.mean([v for s in steps if s > 45000 for v in batches[s]]))
    return np.array(steps), shown, final


def references():
    folder = ROOT / "baselines"
    daft = json.loads((folder / "daft_grid81.json").read_text())["best_rate_bps_per_hz"]
    time = json.loads((folder / "time_grid21.json").read_text())["selected_afdm_rate_bps_per_hz"]
    with (folder / "random_positions_100.csv").open(newline="") as stream:
        random = float(np.mean([float(r["true_afdm_rate_bps_per_hz"]) for r in csv.DictReader(stream)]))
    return daft, time, random


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dreamer", type=Path, required=True)
    p.add_argument("--ppo", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "figures")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "pdf.fonttype": 42})
    dreamer, ppo = terminal_rates(a.dreamer), terminal_rates(a.ppo)
    daft, time, random = references()

    fig = plt.figure(figsize=(PAGE[0] / 72, PAGE[1] / 72))
    ax = fig.add_axes((FRAME[0] / PAGE[0], FRAME[1] / PAGE[1], FRAME[2] / PAGE[0], FRAME[3] / PAGE[1]))
    ax.plot(*dreamer[:2], color=LINE["proposed"], linewidth=2.5, label="Proposed")
    ax.plot(*ppo[:2], color=LINE["ppo"], linestyle="--", linewidth=2.5, label="PPO")
    marks = np.linspace(0, 50000, 11)
    ax.plot(marks, np.full_like(marks, daft), color=LINE["daft"], linestyle="--", linewidth=2.5, marker="^",
            markersize=5.5, markerfacecolor="white", markeredgecolor=LINE["daft"],
            label="DAFT-Domain Exhaustive Search")
    ax.axhline(time, color=LINE["time"], linestyle="-.", linewidth=2.5, label="Time-Domain Exhaustive Search")
    ax.axhline(random, color=LINE["random"], linestyle=":", linewidth=2.5, label="Random Position")
    ax.set_xlim(0, 50000)
    ax.set_ylim(4.5, 20.2)
    ax.set_xlabel("Environment Interaction Steps", fontproperties=font(20.6, True), labelpad=1)
    ax.set_ylabel("Rate (bps/Hz)", fontproperties=font(26.8, True), labelpad=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0.0" if v == 0 else f"{v / 1000:.1f}K"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", top=True, right=True, width=1.4, labelsize=13, pad=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("black")
    ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.08), framealpha=0.95, prop=font(17),
              handlelength=1.7, handletextpad=0.6)
    fig.savefig(a.output / "training_curve.pdf")
    print(f"Final rates (bps/Hz): Proposed {dreamer[2]:.3f}, PPO {ppo[2]:.3f}, "
          f"DAFT-domain search {daft:.3f}, time-domain search {time:.3f}, random {random:.3f}")


if __name__ == "__main__":
    main()
