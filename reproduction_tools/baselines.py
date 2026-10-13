"""Independent random/grid references; no reference data is used by training."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .physics import FAAFDMChannel, FAAFDMConfig
from .records import case_dir, read_json


def channel_for(speed):
    cfg = read_json(case_dir(speed) / "configs/protocol.json")["channel"]
    if not math.isclose(cfg["fixed_doppler_hz"], (speed / 3.6) / cfg["carrier_wavelength"], rel_tol=1e-12):
        raise ValueError("Speed and Doppler configuration disagree")
    return FAAFDMChannel(FAAFDMConfig(**cfg))


def validate_reference_positions(speed):
    channel = channel_for(speed)
    folder = case_dir(speed)
    daft = read_json(folder / "baselines/daft_grid81.json")
    time = read_json(folder / "baselines/time_grid21.json")
    p = np.array(daft["best_position_m"])
    assert abs(channel.rate(p[:2], p[2:]) - daft["best_rate_bps_per_hz"]) < 1e-8
    p = np.array(time["best_position_m"])
    assert abs(channel.time_domain_sinr_h(p[:2], p[2:]) - time["best_time_domain_value"]) < 1e-8
    assert abs(channel.rate(p[:2], p[2:]) - time["selected_afdm_rate_bps_per_hz"]) < 1e-8
    positions = np.random.default_rng(20260926).uniform(-.05, .05, (100, 4))
    values = np.asarray([channel.rate(p[:2], p[2:]) for p in positions])
    with (folder / "baselines/random_positions_100.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    preserved_positions = np.asarray([[float(row[k]) for k in ("u_x_m", "u_y_m", "q_x_m", "q_y_m")] for row in rows])
    preserved_values = np.array([float(row["true_afdm_rate_bps_per_hz"]) for row in rows])
    np.testing.assert_allclose(positions, preserved_positions, rtol=0, atol=1e-14)
    np.testing.assert_allclose(values, preserved_values, rtol=0, atol=1e-8)
    return {"speed_kmh": speed, "random_mean": float(values.mean()), "reference_position_check": "PASS"}


def search(speed, method, output, device="cuda", batch=128, grid=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    channel = channel_for(speed)
    if method == "random":
        positions = np.random.default_rng(20260926).uniform(-.05, .05, (100, 4))
        values = np.array([channel.rate(p[:2], p[2:]) for p in positions])
        result = {"method": "independent_uniform_random_positions", "random_seed": 20260926,
                  "sample_count": 100, "mean_afdm_rate_bps_per_hz": float(values.mean()),
                  "positions_m": positions.tolist(), "afdm_rates": values.tolist()}
    else:
        grid = grid or (81 if method == "daft" else 21)
        if grid < 2 or batch < 1:
            raise ValueError("grid >= 2 and batch >= 1 are required")
        axis = np.linspace(-.05, .05, grid, dtype=np.float64)
        total = grid ** 4
        best, best_position = -np.inf, None
        objective = channel.rate if method == "daft" else channel.time_domain_sinr_h
        gpu_objective = channel.rate_batch_gpu if method == "daft" else channel.time_domain_sinr_h_batch_gpu
        progress = max(batch, total // 20)
        next_progress = progress
        for start in range(0, total, batch):
            indices = np.arange(start, min(start + batch, total), dtype=np.int64)
            positions = axis[np.stack(np.unravel_index(indices, (grid,) * 4), axis=-1)]
            if device == "cuda":
                values = gpu_objective(positions[:, :2], positions[:, 2:], device="cuda")
            else:
                values = np.array([objective(p[:2], p[2:]) for p in positions])
            if not np.isfinite(values).all():
                raise ValueError("Non-finite search objective")
            index = int(np.argmax(values))
            if values[index] > best:
                best, best_position = float(values[index]), positions[index].copy()
            if start + len(indices) >= next_progress or start + len(indices) == total:
                print(f"queries={start + len(indices)}/{total} best={best:.9f}", flush=True)
                next_progress += progress
        checked = objective(best_position[:2], best_position[2:])
        assert abs(checked - best) < 1e-8
        result = {"method": "exact_exhaustive_cartesian_grid" if method == "daft" else "exact_time_domain_exhaustive_cartesian_grid",
                  "grid_points_per_coordinate": grid, "grid_spacing_m": float(axis[1] - axis[0]),
                  "query_count": total, "best_position_m": best_position.tolist(),
                  "cpu_gpu_best_difference": abs(checked - best),
                  "continuous_global_optimum_proven": False}
        if method == "daft":
            result["best_rate_bps_per_hz"] = checked
        else:
            result.update(selection_objective="time_domain_sinr_h", evaluation_metric="afdm_rate",
                          best_time_domain_value=checked,
                          selected_afdm_rate_bps_per_hz=channel.rate(best_position[:2], best_position[2:]))
    result.update(channel=asdict(channel.cfg), speed_kmh=speed, reachable_bounds_m=[-.05, .05],
                  training_data_used=False, status="complete")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
