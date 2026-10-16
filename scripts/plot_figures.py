"""Draw the paper figures from four finished runs.

  python scripts/plot_figures.py --dreamer-50 runs/dreamer_50 --ppo-50 runs/ppo_50 \
      --dreamer-500 runs/dreamer_500 --ppo-500 runs/ppo_500 --output figures

Writes training_curve_50.pdf, training_curve_500.pdf and rate_bar_50kmh_vs_500kmh.pdf.
Curves: mean of the 20 online terminal rates in each 1K-interaction batch, then a causal
trailing mean of up to five batches. Final rate: mean of the last 100 terminal rates (last 5K).
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
PAGE = (570.41, 406.5)                      # points
CURVE_FRAME = (62.3, 38.5, 490.6, 365.4)    # left, bottom, width, height (points)
BAR_FRAME = (62.45, 37.3, 490.35, 365.1)
LINE = {"proposed": "#036EAD", "ppo": "#DD656D", "daft": "#F3962F", "time": "#61A7D0", "random": "#8798C9"}
BAR = ["#3C5488", "#E64B35", "#00A087", "#4DBBD5"]


def font(size, bold=False):
    """Times New Roman (regular/bold) when installed, otherwise a serif fallback."""
    for folder in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/msttcorefonts")):
        path = folder / ("timesbd.ttf" if bold else "times.ttf")
        if path.is_file():
            return FontProperties(fname=str(path), size=size)
    return FontProperties(family="serif", size=size, weight="bold" if bold else "normal")


def figure(frame):
    fig = plt.figure(figsize=(PAGE[0] / 72, PAGE[1] / 72))
    ax = fig.add_axes((frame[0] / PAGE[0], frame[1] / PAGE[1], frame[2] / PAGE[0], frame[3] / PAGE[1]))
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("black")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylabel("Rate (bps/Hz)", fontproperties=font(26.8, True), labelpad=2)
    return fig, ax


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


def references(speed):
    folder = ROOT / "baselines" / f"{speed}kmh"
    daft = json.loads((folder / "daft_grid81.json").read_text())["best_rate_bps_per_hz"]
    time = json.loads((folder / "time_grid21.json").read_text())["selected_afdm_rate_bps_per_hz"]
    with (folder / "random_positions_100.csv").open(newline="") as stream:
        random = float(np.mean([float(r["true_afdm_rate_bps_per_hz"]) for r in csv.DictReader(stream)]))
    return daft, time, random


def training_curve(path, dreamer, ppo, refs):
    fig, ax = figure(CURVE_FRAME)
    daft, time, random = refs
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
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0.0" if v == 0 else f"{v / 1000:.1f}K"))
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", top=True, right=True, width=1.4, labelsize=13, pad=5)
    ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.08), framealpha=0.95, prop=font(17),
              handlelength=1.7, handletextpad=0.6)
    fig.savefig(path)
    plt.close(fig)


def rate_bar(path, values50, values500):
    fig, ax = figure(BAR_FRAME)
    labels = ["Proposed", "PPO", "DAFT-Domain\nExhaustive Search", "Time-Domain\nExhaustive Search"]
    x = np.arange(2)
    width, spacing = 41.75 / 260.0, 43.5 / 260.0
    for i, (label, color) in enumerate(zip(labels, BAR)):
        bars = ax.bar(x + (i - 1.5) * spacing, [values50[i], values500[i]], width=width, color=color,
                      edgecolor="none", label=label, zorder=3)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontproperties=font(11))
    ax.set_xticks(x)
    ax.set_xticklabels(["50", "500"])
    ax.set_xlim(-0.4419, 1.4441)
    ax.set_ylim(0, 26)
    ax.set_yticks(np.arange(0, 25, 2))
    ax.set_xlabel("Receiver Speed (km/h)", fontproperties=font(20.6, True), labelpad=1)
    ax.grid(True, axis="y", color="#DADFE5", linewidth=0.75, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=13, pad=5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False, prop=font(12.5),
              handlelength=1.6, handletextpad=0.5, columnspacing=1.0, borderpad=0.2)
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("dreamer-50", "ppo-50", "dreamer-500", "ppo-500"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "figures")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "pdf.fonttype": 42})
    finals = {}
    for speed in (50, 500):
        dreamer = terminal_rates(getattr(a, f"dreamer_{speed}"))
        ppo = terminal_rates(getattr(a, f"ppo_{speed}"))
        refs = references(speed)
        training_curve(a.output / f"training_curve_{speed}.pdf", dreamer, ppo, refs)
        finals[speed] = [dreamer[2], ppo[2], refs[0], refs[1]]
        print(f"{speed} km/h final rates: Proposed {dreamer[2]:.3f}, PPO {ppo[2]:.3f}, "
              f"DAFT {refs[0]:.3f}, time-domain {refs[1]:.3f}, random {refs[2]:.3f}")
    rate_bar(a.output / "rate_bar_50kmh_vs_500kmh.pdf", finals[50], finals[500])


if __name__ == "__main__":
    main()
