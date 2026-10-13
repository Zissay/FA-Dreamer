"""Audit complete new online logs, captured source bytes and physical transitions."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from .baselines import channel_for
from .records import case_dir, read_json, sha256


def learner_dir(run):
    run = Path(run)
    return run / "training" if (run / "training/config.json").exists() else run


def rows_from(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def audit_run(speed, algorithm, run, physical_samples=128):
    if speed != 50:
        raise ValueError("This complete-log audit applies to the fresh 50 km/h reconstruction")
    run = Path(run)
    data_dir = learner_dir(run)
    launch = read_json(run / "reproduction_launch.json")
    assert launch["pretraining_capture"] and launch["training_interactions"] == 50000
    assert read_json(run / "reproduction_exit.json")["returncode"] == 0
    source = case_dir(speed) / "source" / algorithm
    repo = case_dir(speed).parents[1]
    # Past runs are authenticated against their captured launch bytes. Legitimate
    # later source recovery must not invalidate or relabel an earlier attempt.
    current_source_matches = all((repo / rel).exists() and sha256(repo / rel) == digest
                                 for rel, digest in launch["source_sha256"].items())
    assert sha256(run / "reproduction_launcher.py") == launch["launcher_sha256"]
    assert sha256(repo / "vendor/dreamerv3-installed.tar.gz") == launch["vendor_sha256"]
    if (run / "source_snapshot_manifest.json").is_file():
        snapshot = read_json(run / "source_snapshot_manifest.json")
        archive = run / "verified_launch_source.tar.gz"
    else:
        snapshot = {"archive_sha256": launch["source_archive_sha256"],
                    "launch_json_sha256": sha256(run / "reproduction_launch.json")}
        archive = run / "pretraining_source.tar.gz"
    assert sha256(archive) == snapshot["archive_sha256"]
    assert sha256(run / "reproduction_launch.json") == snapshot["launch_json_sha256"]
    with tarfile.open(archive) as stored:
        for rel, digest in launch["source_sha256"].items():
            assert hashlib.sha256(stored.extractfile(rel).read()).hexdigest() == digest
        assert hashlib.sha256(stored.extractfile("reproduction_tools/training.py").read()).hexdigest() == launch["launcher_sha256"]
        assert hashlib.sha256(stored.extractfile("vendor/dreamerv3-installed.tar.gz").read()).hexdigest() == launch["vendor_sha256"]
    seed = read_json(run / "reproduction_configs/protocol.json")[f"{algorithm}_seed"]
    metadata = read_json(data_dir / ("experiment.json" if algorithm == "dreamer" else "config.json"))
    assert metadata["seed"] == seed
    if algorithm == "dreamer":
        assert metadata["from_checkpoint"] is None
        assert read_json(data_dir / "training_complete.json")["real_actions"] == 50000
        filename = "training_queries.csv"
    else:
        assert metadata["resume_run"] is None and metadata["timesteps"] == 50000
        filename = "training_interaction_rates.csv"
    log_path = data_dir / filename
    if not log_path.exists():
        log_path = log_path.with_suffix(log_path.suffix + ".gz")
    rows = sorted(rows_from(log_path), key=lambda row: int(row["steps"]))
    assert len(rows) == 50000
    assert [int(row["steps"]) for row in rows] == list(range(1, 50001))
    channel = channel_for(speed)
    initial_rate = channel.rate(np.zeros(2), np.zeros(2))
    by_env = {i: [] for i in range(20)}
    for row in rows:
        by_env[int(row["env_index"])].append(row)
    terminals, max_transition_error, max_reward_error = [], 0., 0.
    for index, sequence in by_env.items():
        assert len(sequence) == 2500
        position, previous_rate = np.zeros(4), initial_rate
        for local_step, row in enumerate(sequence):
            step = local_step % 50 + 1
            assert int(row["episode_step"]) == step
            terminal = int(row["episode_terminal"])
            assert terminal == int(step == 50)
            action = np.array([float(row[f"a{i}"]) for i in range(4)])
            after = np.array([float(row[k]) for k in ("u_x", "u_y", "q_x", "q_y")])
            rate = float(row["rate"])
            assert np.isfinite([*action, *after, rate, float(row["reward"])]).all()
            assert np.abs(action).max() <= 1.000001
            assert np.abs(after).max() <= step * .001 + 1e-7
            error = float(np.abs(after - (position + .001 * action)).max())
            max_transition_error = max(max_transition_error, error)
            assert error < 3e-8
            reward_error = abs(float(row["reward"]) - (rate - previous_rate + terminal * .2 * rate))
            max_reward_error = max(max_reward_error, reward_error)
            assert reward_error < 1e-8
            if terminal:
                terminals.append(row)
                position, previous_rate = np.zeros(4), initial_rate
            else:
                position, previous_rate = after, rate
    assert len(terminals) == 1000
    indices = set(np.linspace(0, len(rows) - 1, physical_samples, dtype=int).tolist())
    indices.update(i for i, row in enumerate(rows) if int(row["episode_terminal"]))
    rate_errors = []
    for i in sorted(indices):
        row = rows[i]
        p = np.array([float(row[k]) for k in ("u_x", "u_y", "q_x", "q_y")])
        rate_errors.append(abs(channel.rate(p[:2], p[2:]) - float(row["rate"])))
    assert max(rate_errors) < 1e-8
    report = {"algorithm": algorithm, "seed": seed, "channel_seed": channel.cfg.seed,
              "real_actions": len(rows), "terminal_records": len(terminals), "parallel_envs": 20,
              "last_5k_mean": float(np.mean([float(r["rate"]) for r in terminals if int(r["steps"]) > 45000])),
              "source_counts": dict(Counter(row["source"] for row in rows)),
              "max_transition_error_m": max_transition_error, "max_reward_error": max_reward_error,
              "independent_cpu_rate_checks": len(rate_errors), "max_cpu_gpu_rate_error": max(rate_errors),
              "raw_log_sha256": hashlib.sha256(gzip.open(log_path, "rb").read()).hexdigest() if str(log_path).endswith(".gz") else sha256(log_path),
              "launch_sha256": sha256(run / "reproduction_launch.json"),
              "verified_source_archive_sha256": snapshot["archive_sha256"],
              "fresh_initialization": True, "source_and_config_match_pretraining_hashes": True,
              "checkpoint_restore_used": False, "status": "PASS"}
    if algorithm == "dreamer":
        assert metadata["prefill_rng_seed"] == seed + 1000003
        prefills = [r for r in rows if r["source"] == "uniform_prefill"]
        # Stored env indices may be allocated by parallel constructors in any order.
        # Each prefill action batch must contain exactly the 20 seeded commands.
        rng = np.random.RandomState(metadata["prefill_rng_seed"])
        for start in range(0, len(prefills), 20):
            if start and start % 1000 == 0:
                # The driver samples a command at terminal observations, then
                # masks it out for the reset-only frame. That draw is not an action.
                rng.uniform(-1., 1., (20, 4))
            batch = prefills[start:start+20]
            actual = np.array([[float(r[f"a{i}"]) for i in range(4)] for r in batch])
            expected = rng.uniform(-1., 1., (20, 4)).astype(np.float32)
            sort = lambda a: a[np.lexsort(a.T[::-1])]
            np.testing.assert_allclose(sort(actual), sort(expected), rtol=0, atol=1e-7)
        metrics = [json.loads(line) for line in (data_dir / "metrics.jsonl").read_text().splitlines()]
        for name in ("train/actor_opt_grad_norm", "train/model_opt_grad_norm", "train/extr_critic_critic_opt_grad_norm"):
            values = [float(r[name]) for r in metrics if name in r]
            assert values and np.isfinite(values).all() and max(values) > 0
        report.update(prefill_rng_seed=metadata["prefill_rng_seed"], seeded_prefill_verified=True,
                      replay_sampler_seed=metadata.get("replay_sampler_seed"), data_loaders=metadata["hyperparameters"]["data_loaders"],
                      nonzero_finite_actor_model_critic_gradients=True)
    else:
        updates = [json.loads(line) for line in (data_dir / "ppo_updates.jsonl").read_text().splitlines()]
        assert updates and int(updates[-1]["steps"]) == 50000
        assert all(np.isfinite(float(r["head_weight_update_norm"])) for r in updates)
        assert any(float(r["head_weight_update_norm"]) > 0 for r in updates)
        report.update(ppo_update_records=len(updates), actual_actor_weight_updates_verified=True)
    report["current_repository_matches_launch"] = current_source_matches
    return report
