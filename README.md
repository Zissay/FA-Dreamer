# FA-AFDM Dreamer Experiments

This repository contains the minimal code path used to train, evaluate, and plot the three figures in the paper.

## Structure

```text
fa_afdm_dreamerv3/     Core FA-AFDM channel and environment code.
scripts/              Training, evaluation, exhaustive search, and plotting scripts.
runs/                 Generated checkpoints, metrics, CSV, and JSON files.
figures/              Generated paper figures.
configs.yaml          DreamerV3 configuration file.
requirements.txt      Python dependencies.
```

## Environment

Use Python 3.10 or 3.11 for the DreamerV3 route.

```bash
pip install -r requirements.txt
```

Run all commands from the repository root.

## Training Scripts

### Proposed Alg. 1

```bash
python scripts/train_dreamer.py --logdir runs/dreamer_200k --steps 200000 --configs small --device cuda
```

### PPO

```bash
python scripts/train_ppo.py --logdir runs/ppo_200k --total-timesteps 200000 --device cuda
```

### Exhaustive Search

```bash
python scripts/grid_search.py --output-dir runs/grid_rate20_noise_m95_gain2p284e6 --grid-points 9 --noise-power-dbm -95 --channel-gain-scale 2.2839976470784646e-6
```

## Evaluation Scripts

Evaluate trained checkpoints to produce deployment trajectories.

```bash
python scripts/eval_dreamer.py --checkpoint runs/dreamer_200k/checkpoint.pkl --logdir runs/eval_dreamer --output runs/eval_dreamer/evaluation.json
python scripts/eval_ppo.py --model runs/ppo_200k/ppo_fa_afdm.zip --output runs/eval_ppo/evaluation.json
```

The final plotting scripts expect CSV/JSON files under `runs/`. The key required paths are documented in each plot script near the top of the file.

## Plotting Scripts

### Fig. 1: Achievable Rate Comparison

```bash
python scripts/plot_rate.py
```

Output:

```text
figures/rate.pdf
figures/rate.png
figures/rate.svg
```

Required inputs:

```text
runs/direct_position_rate20_four_model_midexplore_with_random_fixed/four_model_midexplore_mean_curves.csv
runs/direct_position_rate20_four_model_midexplore_with_random_fixed/four_model_midexplore_case_curves.csv
runs/grid_rate20_noise_m95_gain2p284e6/grid_search_result.json
```

### Fig. 2: Channel Magnitude Relative Error

```bash
python scripts/plot_channel.py
```

Output:

```text
figures/channel.pdf
figures/channel.png
figures/channel.svg
```

Required input:

```text
runs/dreamer_direct_position_rate20_20000_worldmodel_dense_log10_channel_mag_rel/metrics.jsonl
```

### Fig. 3: Generalization Comparison

```bash
python scripts/plot_generalization.py
```

Output:

```text
figures/generalization.pdf
figures/generalization.png
figures/generalization.svg
```

Required inputs:

```text
runs/generalization_200k_distance_doppler_paths/distance/d080/dreamer/hitting_time_trajectories_dreamerv3_gpu.csv
runs/generalization_200k_distance_doppler_paths/distance/d080/ppo/hitting_time_trajectories_ppo.csv
runs/generalization_200k_distance_doppler_paths/doppler/f050/dreamer/hitting_time_trajectories_dreamerv3_gpu.csv
runs/generalization_200k_distance_doppler_paths/doppler/f050/ppo/hitting_time_trajectories_ppo.csv
runs/generalization_200k_hard_ood/paths/p08/dreamer/hitting_time_trajectories_dreamerv3_gpu.csv
runs/generalization_200k_hard_ood/paths/p08/ppo/hitting_time_trajectories_ppo.csv
```

## Notes

- Generated figures and experiment outputs are ignored by git.
- The scripts keep the final paper figure style: Times New Roman, 4:3 aspect ratio, and PDF/SVG/PNG export.
- `Proposed Alg. 1` denotes the DreamerV3-based method used in the manuscript.
