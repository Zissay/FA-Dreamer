from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .channel import FAAFDMChannel


def make_position_set(channel: FAAFDMChannel, *, count: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    tx_low, tx_high = channel.tx_bounds
    rx_low, rx_high = channel.rx_bounds
    positions = []
    for case in range(count):
        positions.append(
            {
                "case": case,
                "u": rng.uniform(tx_low, tx_high).astype(float).tolist(),
                "q": rng.uniform(rx_low, rx_high).astype(float).tolist(),
            }
        )
    return positions


def load_position_set(path: str | Path, split: str = "eval") -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if "positions" in data:
        return data["positions"]
    if split in data:
        return data[split]
    raise ValueError(f"Could not find split '{split}' in {path}")


def normalize_position_set(positions: list[dict]) -> tuple[tuple[float, ...], ...]:
    normalized = []
    for item in positions:
        u = tuple(float(x) for x in item["u"])
        q = tuple(float(x) for x in item["q"])
        normalized.append((*u, *q))
    return tuple(normalized)
