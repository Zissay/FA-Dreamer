from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs/dreamer_direct_position_rate20_20000_worldmodel_dense_log10_channel_mag_rel"
OUT_DIR = ROOT / "figures"
KEY = "train/channel_mag_rel_error_mean"


def read_metrics(path: Path) -> pd.DataFrame:
    rows = []
    max_step_seen = -1.0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            if "step" not in item or KEY not in item:
                continue
            step = float(item["step"])
            if step <= max_step_seen:
                continue
            max_step_seen = step
            rows.append(
                {
                    "step": step,
                    "channel_magnitude_relative_error_percent": float(item[KEY]) * 100.0,
                }
            )
    data = pd.DataFrame(rows)
    if data.empty:
        return data
    return data.groupby("step", as_index=False).mean().sort_values("step")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = read_metrics(RUN_DIR / "metrics.jsonl")
    if data.empty:
        raise SystemExit(f"No {KEY} records found.")
    data.to_csv(
        OUT_DIR / "channel_magnitude_relative_error_percent_source.csv", index=False
    )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        data["step"],
        data["channel_magnitude_relative_error_percent"],
        color="#2563eb",
        linewidth=3.0,
    )
    ax.set_xlim(0, 200000)
    ax.xaxis.set_major_locator(MultipleLocator(20000))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "0" if value == 0 else f"{value / 1000:.0f}k")
    )
    ax.set_xlabel("Training steps")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.set_ylabel("Channel magnitude relative error")
    ax.grid(True, which="major", color="#94a3b8", alpha=0.24, linewidth=0.8)
    ax.grid(True, which="minor", color="#94a3b8", alpha=0.10, linewidth=0.5)
    fig.tight_layout()

    base = OUT_DIR / "channel"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(base.with_suffix(f".{suffix}"), dpi=600)
    plt.close(fig)

    series = data["channel_magnitude_relative_error_percent"]
    summary = {
        "records": len(data),
        "first_step": data["step"].iloc[0],
        "first_channel_magnitude_relative_error_percent": series.iloc[0],
        "last_step": data["step"].iloc[-1],
        "last_channel_magnitude_relative_error_percent": series.iloc[-1],
        "min_channel_magnitude_relative_error_percent": series.min(),
    }
    pd.DataFrame([summary]).to_csv(
        OUT_DIR / "channel_magnitude_relative_error_percent_summary.csv", index=False
    )
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"saved={base}.png/.pdf/.svg")


if __name__ == "__main__":
    main()


