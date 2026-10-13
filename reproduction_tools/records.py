"""Read genuine terminal actions and apply the declared causal display rule."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def case_dir(speed):
    if int(speed) not in (50, 500):
        raise ValueError("The preserved cases are 50 and 500 km/h")
    return ROOT / "experiments" / f"{int(speed)}kmh"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve(path, budget=50000):
    """Accept an archived terminal-only CSV or a new complete interaction log."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if int(row["steps"]) <= budget and int(row["episode_terminal"]) == 1]
    rows.sort(key=lambda row: int(row["steps"]))
    steps = np.array([int(row["steps"]) for row in rows])
    rates = np.array([float(row["rate"]) for row in rows])
    if len(rows) != budget // 50 or len(np.unique(steps)) != len(steps):
        raise ValueError(f"Expected {budget // 50} distinct terminal records: {path}")
    if not np.isfinite(rates).all():
        raise ValueError("Non-finite terminal rate")
    coordinates = np.asarray([[float(row[k]) for k in ("u_x", "u_y", "q_x", "q_y")] for row in rows])
    if np.max(np.abs(coordinates)) > 0.0500001:
        raise ValueError("Terminal position outside the 50-step reachable region")
    ends = ((steps - 1) // 1000 + 1) * 1000
    batches = np.arange(1000, budget + 1, 1000)
    if any(np.count_nonzero(ends == end) != 20 for end in batches):
        raise ValueError("Each synchronized batch must contain 20 terminal episodes")
    raw = np.array([rates[ends == end].mean() for end in batches])
    displayed = np.array([raw[max(0, i - 4):i + 1].mean() for i in range(len(raw))])
    return {"steps": batches, "batch_mean": raw, "displayed": displayed,
            "terminal_steps": steps, "terminal_rates": rates,
            "last_5k_mean": float(rates[steps > budget - 5000].mean()),
            "file_sha256": sha256(path)}


def reference_rates(speed):
    folder = case_dir(speed)
    protocol = read_json(folder / "configs/protocol.json")
    daft = read_json(folder / "baselines/daft_grid81.json")
    time = read_json(folder / "baselines/time_grid21.json")
    for result, count, spacing in [(daft, 81, .00125), (time, 21, .005)]:
        if result["channel"] != protocol["channel"]:
            raise ValueError(f"Reference channel differs from the {speed} km/h experiment")
        if result["grid_points_per_coordinate"] != count or result["query_count"] != count ** 4:
            raise ValueError("Reference grid accounting mismatch")
        if abs(result["grid_spacing_m"] - spacing) > 1e-12:
            raise ValueError("Reference spatial spacing mismatch")
        if result["reachable_bounds_m"] != [-.05, .05] or result["training_data_used"]:
            raise ValueError("Reference is not an independent reachable-grid calculation")
    with (folder / "baselines/random_positions_100.csv").open(newline="") as stream:
        random = [float(row["true_afdm_rate_bps_per_hz"]) for row in csv.DictReader(stream)]
    if len(random) != 100:
        raise ValueError("Expected 100 independent random positions")
    return {"daft": daft["best_rate_bps_per_hz"],
            "time": time["selected_afdm_rate_bps_per_hz"], "random": float(np.mean(random))}


def load_results(speed, dreamer_run=None, ppo_run=None):
    folder = case_dir(speed)
    if ppo_run and (Path(ppo_run) / "training/config.json").is_file():
        ppo_run = Path(ppo_run) / "training"
    dpath = (Path(dreamer_run) / "training_queries.csv") if dreamer_run else folder / "recorded/dreamer/terminal_interactions.csv"
    ppath = (Path(ppo_run) / "training_interaction_rates.csv") if ppo_run else folder / "recorded/ppo/terminal_interactions.csv"
    if (dreamer_run is None) != (ppo_run is None):
        raise ValueError("Provide both new run directories, or use both recorded curves")
    dm_seed = read_json(Path(dreamer_run) / "experiment.json")["seed"] if dreamer_run else read_json(folder / "recorded/dreamer/experiment.json")["seed"]
    pm_seed = read_json(Path(ppo_run) / "config.json")["seed"] if ppo_run else read_json(folder / "recorded/ppo/config.json")["seed"]
    if dm_seed != pm_seed:
        raise ValueError("Dreamer/PPO training seeds must match for this comparison")
    if dreamer_run:
        from dataclasses import asdict
        from .physics import FAAFDMConfig
        expected = read_json(folder / "configs/protocol.json")
        dm = read_json(Path(dreamer_run) / "experiment.json")
        pm = read_json(Path(ppo_run) / "config.json")
        fields = asdict(FAAFDMConfig())
        candidate = {k: pm.get(k, value) for k, value in fields.items()}
        candidate["seed"] = pm["channel_seed"]
        if dm["channel"] != expected["channel"] or candidate != expected["channel"]:
            raise ValueError("New-run channel does not match the selected speed")
        if dm["num_envs"] != 20 or pm["n_envs"] != 20:
            raise ValueError("The display protocol requires 20 parallel environments")
        env = dm["environment"]
        if env["max_steps"] != pm["episode_steps"] or env["max_steps"] != 50:
            raise ValueError("Episode horizon mismatch")
        for key in ["action_step", "reward_mode", "terminal_rate_weight", "boundary_penalty",
                    "finite_horizon_terminal", "direct_position_action", "continuing_task"]:
            if env[key] != pm[key]:
                raise ValueError(f"New-run environment mismatch: {key}")
        if env["random_reset"] or not pm["fixed_center_start"] or pm["train_start_pool"] is not None:
            raise ValueError("New runs must use the same fixed-center reset rule")
    return {"dreamer": curve(dpath), "ppo": curve(ppath), **reference_rates(speed)}
