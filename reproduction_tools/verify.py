"""Validate preserved experiment parameters, source bytes and measured results."""
from __future__ import annotations

import ast
import math

from .records import ROOT, case_dir, load_results, read_json, sha256


def verify_all(physics=False):
    summary = {}
    for speed in (50, 500):
        folder = case_dir(speed)
        protocol = read_json(folder / "configs/protocol.json")
        hp = read_json(folder / "configs/dreamer.json")
        original = read_json(folder / "recorded/dreamer/experiment.json")
        ppo = read_json(folder / "configs/ppo.json")
        assert hp == original["hyperparameters"]
        assert protocol["channel"] == original["channel"]
        assert protocol["channel"]["seed"] == ppo["channel_seed"] == 7
        assert ppo["fixed_doppler_hz"] == protocol["channel"]["fixed_doppler_hz"]
        for key in ("n_subcarriers", "n_paths", "carrier_wavelength", "transmit_power", "noise_power_dbm"):
            assert ppo[key] == protocol["channel"][key]
        assert ppo["episode_steps"] == protocol["environment"]["max_steps"] == 50
        assert ppo["action_step"] == protocol["environment"]["action_step"] == .001
        assert ppo["n_envs"] == original["num_envs"] == 20
        count = 0
        recovered = {}
        for algorithm in ("dreamer", "ppo"):
            manifest = read_json(folder / "source" / f"{algorithm}_source_manifest.json")
            for rel, info in manifest["files"].items():
                path = folder / "source" / algorithm / rel
                assert sha256(path) == info["sha256"], f"Preserved source changed: {path}"
                ast.parse(path.read_text(encoding="utf-8-sig"))
                count += 1
            recovered[algorithm] = manifest["dependencies_restored_from_workspace"]
        results = load_results(speed)
        expected = read_json(folder / "expected_results.json")
        actual = {"dreamer": results["dreamer"]["last_5k_mean"], "ppo": results["ppo"]["last_5k_mean"],
                  **{key: results[key] for key in ("daft", "time", "random")}}
        for key, value in expected.items():
            assert math.isclose(actual[key], value, abs_tol=1e-10), (speed, key, actual[key], value)
        summary[str(speed)] = {"source_file_hash_checks": count, "results": actual,
                               "source_provenance": protocol["source_provenance"],
                               "recovered_dependencies": recovered, "status": "PASS"}
        if physics:
            from .baselines import validate_reference_positions
            summary[str(speed)]["physics"] = validate_reference_positions(speed)
    manifest_path = ROOT / "experiments/MANIFEST.sha256.json"
    if manifest_path.exists():
        for rel, digest in read_json(manifest_path).items():
            assert sha256(ROOT / rel) == digest, f"Published artifact changed: {rel}"
    return summary
