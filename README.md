# FA-Dreamer: 50 km/h and 500 km/h experiments

Preserved configurations, training sources and online terminal records for the three panels of Fig. 4: training at 50 km/h, training at 500 km/h, and final-rate comparison. Each mobility case has its own channel configuration and independently trained policies.

**[中文复现说明](docs/REPRODUCTION_CN.md)** · [50 km/h](experiments/50kmh/README.md) · [500 km/h](experiments/500kmh/README.md)

## Rebuild the paper figures

Run from the repository root. Rebuilding the archived figures requires no training or GPU.

```bash
python -m pip install -r requirements/analysis.txt
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py verify --physics
python scripts/reproduce.py plot-all --output-dir figures/reproduced
```

On Windows PowerShell, set `$env:OPENBLAS_NUM_THREADS='1'` before running `verify`.
The three independent PDF/PNG panels share the same axis-frame dimensions, spine widths, typography and crop. Times New Roman is selected from Windows fonts when available; use `--font-dir` with `times.ttf` and `timesbd.ttf` to obtain the same font elsewhere. Original bar colors and the label **Random Position** are preserved. CSV and plot-audit files are also exported.

### Expected rates (bps/Hz)

| Speed | Dreamer | PPO | DAFT grid | Time-selected positions, AFDM rate | Random positions |
|---|---:|---:|---:|---:|---:|
| 50 km/h | 18.837550 | 15.441477 | 19.416280 | 13.257986 | 5.919826 |
| 500 km/h | 18.301658 | 15.277230 | 19.418465 | 13.252067 | 5.920893 |

Dreamer/PPO rates average terminal episodes during the final 5k of the first 50k real environment interactions. Training curves use 1k-interaction batches and a trailing, causal five-batch mean. Random references average 100 independent joint positions on the corresponding speed's channel. See [aggregation rules](docs/TRAINING_CURVE_RULES.md).

## Preserved experiments

| Setting | 50 km/h | 500 km/h |
|---|---:|---:|
| Dreamer seed | 3501 | 5215 |
| PPO seed | 3004 | 5215 |
| Channel seed | 7 | 7 |
| Parallel environments / episode actions | 20 / 50 | 20 / 50 |
| Maximum coordinate displacement per action | 1 mm | 1 mm |
| Dreamer imagination horizon | 10 | 50 |
| Dreamer batch size / train ratio | 16 / 256 | 8 / 128 |
| Dreamer original budget | 50k | 50k |
| PPO original budget | **500k** | 50k |
| Budget shown in Fig. 4 | first 50k | first 50k |

All original learning rates, entropy schedules, replay settings, network sizes, task constraints and physical channel parameters are in `experiments/<speed>/configs/`. Recorded Dreamer `resolved_config.yaml` files retain the full model configurations. Actual channel settings are `N=128`, `P=20`, transmit power 1.5 W and noise −95 dBm.

### Train from scratch

Training uses Linux/WSL, Python 3.12 and an NVIDIA CUDA GPU. Create separate Dreamer and PPO environments using `requirements/dreamer.txt` and `requirements/ppo.txt`; the original 500 km/h dependency inventories are in `requirements/locks/`. The preserved installed Dreamer package is included in `vendor/` and selected by the launcher.

```bash
.venv-dreamer/bin/python scripts/reproduce.py train --speed 50 --algorithm dreamer --run-dir runs/50kmh/dreamer_seed3501
.venv-dreamer/bin/python scripts/reproduce.py train --speed 500 --algorithm dreamer --run-dir runs/500kmh/dreamer_seed5215
.venv-ppo/bin/python scripts/reproduce.py train --speed 50 --algorithm ppo --paper-prefix --gpu-python .venv-dreamer/bin/python --run-dir runs/50kmh/ppo_seed3004
.venv-ppo/bin/python scripts/reproduce.py train --speed 500 --algorithm ppo --gpu-python .venv-dreamer/bin/python --run-dir runs/500kmh/ppo_seed5215
```

For 50 km/h PPO, `--paper-prefix` stops after the plotted 50k interactions while preserving all other parameters, including the absolute 20k entropy schedule. Omit it to run the original 500k protocol. Add `--dry-run` to inspect a command. Every run requires a fresh directory and records configurations, source hashes, dependency inventories and launch settings before training. See the Chinese guide for CUDA setup, baseline search and plotting new training logs.

## Provenance and reproducibility

Archived records and figure values are directly verifiable. The **500 km/h** trainers come from their respective pretraining source snapshots; archive hashes and original Git heads are recorded in the case directory. The **50 km/h** sources come from the 2026-09-11 historical package, with two missing PPO dependencies recovered from the retained workspace. Its original launch snapshot and dependency lock are unavailable, so this is a runnable reconstruction candidate rather than a guarantee of the original training trajectory. Source manifests identify every recovered file.

New trajectories can vary with JAX/CUDA, hardware and asynchronous data loading. Plot new runs from their own recorded data. No pretrained weights are distributed.

```text
experiments/50kmh/       Configs, selected historical sources, terminal records and references
experiments/500kmh/      Configs, pretraining sources, terminal records and references
reproduction_tools/     Shared launch, record verification, baseline and figure tools
requirements/           Analysis/training dependencies and original launch inventories
vendor/                 Preserved installed Dreamer source and its license
scripts/reproduce.py    Entry point for verify/train/plot/plot-all/baseline
tests/                  Record causality, protocol and trainer-config checks
```

The originally uploaded root modules and scripts remain available; their earlier instructions are retained in [legacy workflow](docs/LEGACY_WORKFLOW.md). Use the experiment-specific entry point above for these two Fig. 4 cases.
