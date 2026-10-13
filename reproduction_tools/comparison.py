"""Check reproduction of the historical reported result independently of integrity."""
from pathlib import Path
from .records import case_dir, curve, read_json, sha256


def compare_historical_50(dreamer_run):
    run = Path(dreamer_run)
    metadata = read_json(run / "experiment.json")
    status = read_json(case_dir(50) / "reproduction_status.json")
    target = status["historical"]["dreamer"]
    actual = curve(run / "training_queries.csv")["last_5k_mean"]
    complete = read_json(run / "training_complete.json")["real_actions"] == 50000
    fresh = metadata["from_checkpoint"] is None
    same_seed = metadata["seed"] == target["seed"] == status["required_common_seed"]
    same_loaders = metadata["hyperparameters"]["data_loaders"] == target["data_loaders"]
    original = read_json(case_dir(50) / "historical_target/dreamer/experiment.json")
    historical_hp = {**original["hyperparameters"], "data_loaders": target["data_loaders"]}
    same_parameters = metadata["hyperparameters"] == historical_hp and metadata["channel"] == original["channel"]
    launch = read_json(run / "reproduction_launch.json")
    recovery = read_json(case_dir(50) / "historical_target/source/recovery_manifest.json")
    same_source = all(launch["source_sha256"].get("experiments/50kmh/source/dreamer/" + rel) == digest
                      for rel, digest in recovery["recovered_sha256"].items())
    # The paper reported two decimal places. Declare that accuracy before seeing
    # the new run; never relax it after observing performance.
    reported_value_matches = round(actual, 2) == round(target["last_5k_mean"], 2)
    return {"status": "REPRODUCED_REPORTED_FINAL_MEAN" if all((complete, fresh, same_seed,
                same_loaders, same_parameters, same_source, reported_value_matches)) else "NOT_REPRODUCED",
            "metric": status["metric"], "historical_mean": target["last_5k_mean"],
            "new_mean": actual, "difference": actual - target["last_5k_mean"],
            "criterion": "Same final-5k terminal metric, rounds to18.84 at2 decimals; no checkpoint initialization",
            "complete_50k": complete, "fresh_initialization": fresh,
            "same_dreamer_seed": same_seed, "same_data_loaders": same_loaders,
            "same_historical_learning_parameters_and_channel": same_parameters,
            "recovered_train_and_agent_source_match": same_source,
            "training_log_sha256": sha256(run / "training_queries.csv"),
            "trajectory_reproduced": False,
            "ppo_common_seed_comparison_reproduced": False}
