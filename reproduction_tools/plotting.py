"""Three independent panels with identical frame geometry and a shared crop."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

from .records import load_results

SIZE_PT = (571.16, 422.2)
FRAME_PT = (63.2, 50.4, 490.6, 365.4)
CROP_PT = (0.75, 11.95, 571.16, 418.45)
COLORS = ["#036EAD", "#DD656D", "#F3962F", "#61A7D0", "#8798C9"]
BAR_COLORS = ["#3C5488", "#E64B35", "#00A087", "#4DBBD5"]


def font(size, bold=False, font_dir=None):
    if font_dir:
        path = Path(font_dir) / ("timesbd.ttf" if bold else "times.ttf")
        if not path.is_file():
            raise FileNotFoundError(path)
        return FontProperties(fname=str(path), size=size)
    for directory in (Path("C:/Windows/Fonts"), Path("/mnt/c/Windows/Fonts")):
        path = directory / ("timesbd.ttf" if bold else "times.ttf")
        if path.is_file():
            return FontProperties(fname=str(path), size=size)
    names = {item.name for item in font_manager.fontManager.ttflist}
    family = next((n for n in ("Times New Roman", "Liberation Serif", "DejaVu Serif") if n in names), "serif")
    return FontProperties(family=family, size=size, weight="bold" if bold else "normal")


def canvas(xlabel, font_dir=None):
    plt.rcParams.update({"font.family": "serif", "font.serif": [font(13, font_dir=font_dir).get_name()],
                         "pdf.fonttype": 42, "path.simplify": False})
    fig = plt.figure(figsize=(SIZE_PT[0] / 72, SIZE_PT[1] / 72))
    x, y, w, h = FRAME_PT
    ax = fig.add_axes((x / SIZE_PT[0], y / SIZE_PT[1], w / SIZE_PT[0], h / SIZE_PT[1]))
    ax.set_xlabel(xlabel, fontproperties=font(20.6, True, font_dir), labelpad=1)
    ax.set_ylabel("Rate (bps/Hz)", fontproperties=font(26.8, True, font_dir), labelpad=2)
    # Fix the physical label position across Agg/PDF renderers and all panels.
    ax.yaxis.set_label_coords(15.8 / SIZE_PT[0], (y + h / 2) / SIZE_PT[1], transform=fig.transFigure)
    ax.yaxis.label.set_verticalalignment("center")
    ax.tick_params(direction="in", top=True, right=True, width=1.4, labelsize=13, pad=5)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_axisbelow(True)
    ax.grid(alpha=.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("black")
        spine.set_capstyle("butt")
    return fig, ax


def export(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)
    reader = PdfReader(path.with_suffix(".pdf"))
    page = reader.pages[0]
    for attr in ("mediabox", "cropbox", "trimbox"):
        setattr(page, attr, RectangleObject(CROP_PT))
    writer = PdfWriter()
    writer.add_page(page)
    writer.write(path.with_suffix(".pdf"))
    image = Image.open(path.with_suffix(".png"))
    sx, sy = image.width / SIZE_PT[0], image.height / SIZE_PT[1]
    left, bottom, right, top = CROP_PT
    image.crop((round(left * sx), round((SIZE_PT[1] - top) * sy),
                round(right * sx), round((SIZE_PT[1] - bottom) * sy))).save(path.with_suffix(".png"))


def save_records(result, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for algorithm in ("dreamer", "ppo"):
        curve = result[algorithm]
        with (out / f"{algorithm}_display_points.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["environment_interaction_step", "batch_mean", "displayed_trailing_5_batch_mean"])
            writer.writerows(zip(curve["steps"], curve["batch_mean"], curve["displayed"]))
        with (out / f"{algorithm}_terminal_points.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["environment_interaction_step", "true_terminal_rate"])
            writer.writerows(zip(curve["terminal_steps"], curve["terminal_rates"]))
    audit = {"interaction_limit_per_method": 50000, "raw_terminal_points_per_method": 1000,
             "display_points_per_method": 50, "parallel_episodes_per_batch": 20,
             "causal_trailing_batches": 5, "frozen_evaluation_included": False,
             "best_so_far_included": False, "frame_size_pt": list(FRAME_PT[2:]),
             "frame_linewidth_pt": 1.4, "crop_rectangle_pt": list(CROP_PT),
             "last_5k_mean": {a: result[a]["last_5k_mean"] for a in ("dreamer", "ppo")},
             "source_csv_sha256": {a: result[a]["file_sha256"] for a in ("dreamer", "ppo")},
             "offline_references": {a: result[a] for a in ("daft", "time", "random")}}
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def plot_training(speed, output, dreamer_run=None, ppo_run=None, font_dir=None):
    result = load_results(speed, dreamer_run, ppo_run)
    fig, ax = canvas("Environment Interaction Steps", font_dir)
    for key, color, style, label in [("dreamer", COLORS[0], "-", "Proposed"),
                                     ("ppo", COLORS[1], "--", "PPO")]:
        ax.plot(result[key]["steps"], result[key]["displayed"], color=color, linestyle=style, linewidth=2.5, label=label)
    steps = np.linspace(0, 50000, 11)
    ax.plot(steps, np.full_like(steps, result["daft"]), color=COLORS[2], linestyle="--", linewidth=2.5,
            marker="^", markersize=5.5, markerfacecolor="white", label="DAFT-Domain Exhaustive Search")
    ax.axhline(result["time"], color=COLORS[3], linestyle="-.", linewidth=2.5, label="Time-Domain Exhaustive Search")
    ax.axhline(result["random"], color=COLORS[4], linestyle=":", linewidth=2.5, label="Random Position")
    ax.set(xlim=(0, 50000), ylim=(4.5, 20.2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0.0" if value == 0 else f"{value / 1000:.1f}K"))
    placed = False
    for size in (19, 18, 17, 16, 15, 14):
        for y in (.105, .095, .085, .075, .065):
            ax.legend(loc="lower right", bbox_to_anchor=(1.01, y), framealpha=.95,
                      prop=font(size, font_dir=font_dir))
            fig.canvas.draw()
            box = ax.get_legend().get_window_extent(fig.canvas.get_renderer())
            if not any(line.get_path().transformed(ax.transData).intersects_bbox(box, filled=False) for line in ax.lines):
                placed = True
                break
        if placed:
            break
    if not placed:
        raise ValueError("No unobstructed legend position found; adjust analysis layout, not training data")
    output = Path(output)
    save_records(result, output / f"{speed}kmh")
    export(fig, output / f"training_curve_{speed}")
    return result


def plot_bar(results, output, font_dir=None):
    fig, ax = canvas("Receiver Speed (km/h)", font_dir)
    x = np.arange(2)
    labels = ["Dreamer", "PPO", "DAFT-Domain\nExhaustive Search", "Time-Domain\nExhaustive Search"]
    keys = ["dreamer", "ppo", "daft", "time"]
    values = []
    for i, (key, color, label) in enumerate(zip(keys, BAR_COLORS, labels)):
        ys = [results[s][key]["last_5k_mean"] if key in ("dreamer", "ppo") else results[s][key] for s in (50, 500)]
        values.append(ys)
        bars = ax.bar(x + (i - 1.5) * 362 / 2166, ys, width=350 / 2166, color=color,
                      edgecolor="none", linewidth=0, label=label)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontproperties=font(11, font_dir=font_dir))
    ax.set_xticks(x, ["50", "500"])
    ax.set(xlim=(-.4425, 1.4448), ylim=(0, 26))
    ax.set_yticks(np.arange(0, 25, 2))
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", color="#DADFE5", linewidth=.75, alpha=1)
    ax.tick_params(length=0)
    ax.legend(loc="upper center", ncol=4, frameon=True, framealpha=1, facecolor="white", edgecolor="white",
              prop=font(14, font_dir=font_dir), borderpad=.2, labelspacing=.3, columnspacing=.7, handlelength=1.2, handletextpad=.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = ax.get_legend().get_window_extent(renderer)
    if any(box.overlaps(text.get_window_extent(renderer)) for text in ax.texts):
        raise ValueError("Bar legend overlaps a value label")
    export(fig, Path(output) / "rate_bar_50kmh_vs_500kmh")
    with (Path(output) / "bar_values.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "50_km_h", "500_km_h"])
        writer.writerows((label.replace("\n", " "), *ys) for label, ys in zip(labels, values))
