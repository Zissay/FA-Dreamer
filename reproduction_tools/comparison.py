"""Check reproduction of the historical reported result independently of integrity."""
from pathlib import Path
import numpy as np
from .records import case_dir, curve, read_json, sha256


def compare_historical_50(dreamer_run):
    run = Path(dreamer_run)
    metadata = read_json(run / "experiment.json")
    status = read_json(case_dir(50) / "reproduction_status.json")
    target = status["historical"]["dreamer"]
    current_curve = curve(run / "training_queries.csv")
    historical_curve = curve(case_dir(50) / "historical_target/dreamer/terminal_interactions.csv")
    actual = current_curve["last_5k_mean"]
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
    # User accepts similar performance and a two-stage rise; exact decimals and
    # pointwise equality are not required. Set a conservative0.2bps/Hz margin
    # before seeing the fully recovered original-runtime run's final outcome.
    reported_value_matches = round(actual, 2) == round(target["last_5k_mean"], 2)
    performance_close = abs(actual - target["last_5k_mean"]) <= 0.2
    plateau_mask = (current_curve["steps"] > 20000) & (current_curve["steps"] <= 35000)
    new_plateau = float(np.mean(current_curve["batch_mean"][plateau_mask]))
    old_plateau = float(np.mean(historical_curve["batch_mean"][plateau_mask]))
    old_late_gain = target["last_5k_mean"] - old_plateau
    new_late_gain = actual - new_plateau
    first_gain = new_plateau - float(np.mean(current_curve["batch_mean"][:5]))
    two_stage_rise = first_gain > 1 and new_late_gain >= 0.5 * old_late_gain
    flags = launch["environment"].get("XLA_FLAGS", "")
    runtime_match = ("--xla_gpu_deterministic_ops" not in flags
                     and "--xla_gpu_enable_triton_gemm=false" in flags
                     and "--xla_gpu_cuda_data_dir=" in flags
                     and launch["environment"].get("PYTHONHASHSEED") is None)
    return {"status": "REPRODUCED_APPROXIMATE_DREAMER_PERFORMANCE_AND_TREND" if all((complete, fresh, same_seed,
                same_loaders, same_parameters, same_source, runtime_match, performance_close, two_stage_rise)) else "NOT_REPRODUCED",
            "metric": status["metric"], "historical_mean": target["last_5k_mean"],
            "new_mean": actual, "difference": actual - target["last_5k_mean"],
            "relative_final_mean_difference_percent": 100 * (actual / target["last_5k_mean"] - 1),
            "within_0_1_bps_per_hz": abs(actual - target["last_5k_mean"]) <= 0.1,
            "same_final_reported_mean": reported_value_matches,
            "curve_mean_absolute_difference": float(np.mean(np.abs(current_curve["displayed"] - historical_curve["displayed"]))),
            "curve_max_absolute_difference": float(np.max(np.abs(current_curve["displayed"] - historical_curve["displayed"]))),
            "criterion": "Original learning parameters/seed and XLA profile; final mean within0.2bps/Hz of18.83755; initial rise and later gain at least half the historical second rise. No checkpoint initialization.",
            "final_performance_tolerance_bps_per_hz": 0.2,
            "final_performance_close": performance_close,
            "historical_mid_training_plateau": old_plateau,
            "new_mid_training_plateau": new_plateau,
            "historical_late_gain": old_late_gain,
            "new_late_gain": new_late_gain,
            "two_stage_rise": two_stage_rise,
            "original_xla_profile_matches": runtime_match,
            "complete_50k": complete, "fresh_initialization": fresh,
            "same_dreamer_seed": same_seed, "same_data_loaders": same_loaders,
            "same_historical_learning_parameters_and_channel": same_parameters,
            "recovered_train_and_agent_source_match": same_source,
            "training_log_sha256": sha256(run / "training_queries.csv"),
            "trajectory_reproduced": False,
            "ppo_common_seed_comparison_reproduced": False}
