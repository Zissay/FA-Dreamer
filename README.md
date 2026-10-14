# FA-Dreamer: 50 km/h and 500 km/h experiments

> **最近两轮完整从零控制：** 原学习参数和seed3501、随机模型/空回放，各完整50k，固定回放线程交错的末5k为 **15.262785 / 13.931290 bps/Hz**，均未达到原18.837550及二次抬升。完整来源、物理核验和两份各12175次实际采样记录通过。原交错顺序未知，运行控制明确声明。见[完整报告](docs/loader_cycle_pair_audit_20261014.json)。所有此前失败与原30k诊断续训分别保留。

> **复现状态：未成功。** 原 50 km/h 目标为 Dreamer 18.837550、PPO 15.441477 bps/Hz。当前 `recorded/` 中的 15.242103 / 13.739768 是未达到原结果的真实尝试，不能替换论文结果。原 PPO seed 为3004；后续两种方法统一3501，旧数据不会改标 seed。见 `experiments/50kmh/reproduction_status.json`。

Preserved configurations, training sources and online terminal records for the three panels of Fig. 4: training at 50 km/h, training at 500 km/h, and final-rate comparison. Each mobility case has its own channel configuration and independently trained policies.

**[中文复现说明](docs/REPRODUCTION_CN.md)** · [原50km/h源码恢复进展](docs/HISTORICAL_50KMH_RECOVERY_CN.md) · [50 km/h](experiments/50kmh/README.md) · [500 km/h](experiments/500kmh/README.md)

**[统一30k＋20k恢复训练入口](docs/STAGED_RECOVERY_50KMH_CN.md)**：`scripts/recover_staged_50.py resume`统一输入校验、20k恢复训练、物理核验及两阶段汇总；`plan`检查输入，`assemble`汇总已完成阶段。该流程依赖原30k检查点，保留恢复标识。

**[从零整合30k＋本轮恢复20k](docs/INTEGRATED_FRESH_50KMH_CN.md)**：`scripts/train_integrated_50.py`先从空模型训练30k，再以本轮模型/优化器/回放接续20k。原参数及seed3501固定。完整50k已完成，末段均值**13.071403bps/Hz**，未复现原二次抬升/18.84；来源及物理核验通过。30k的采样器/carry重建边界明确记录，见[报告](docs/integrated_fresh_audit_20261014.json)。

**[持续取证与调度控制](docs/EARLY_STATE_INVESTIGATION_CN.md)**：所有已完成控制均未复现。直接保存的原、新前116帧逐字段一致；全部10个里程碑更新计数一致。FP32控制15.289609，creator启动顺序控制13.085993，固定交错两轮15.262785/13.931290，初始领先32控制15.287511，失败记录均保留。各515项可变交错/配对FP32短诊断均未识别原状态或满足登记的完整训练触发门槛。后续启动时序、四轮局部交错细化、精度配对和八组采样预热诊断已完成，均未达到登记的早期筛选门槛。单独登记一次推测初始顺序的完整从零50k验证，明确旧门槛未通过，最终约18.84及二次抬升验收不变。该顺序尚未识别为原历史状态；诊断临时模型不供完整训练。

`scripts/verify_fresh_trace_50.py`为通过性能门槛的候选提供从零计算复核：只读取实际采样索引，以新模型、新生成观测训练；不读取候选权重或旧观测。候选核验方案必须在启动前登记。

## Rebuild the paper figures

Run from the repository root. Rebuilding recorded attempts requires no training or GPU; these50km/h attempt figures are not accepted paper replacements.

```bash
python -m pip install -r requirements/analysis.txt
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py verify --physics
python scripts/reproduce.py plot-all --allow-unreproduced --output-dir figures/failed_attempt
```

On Windows PowerShell, set `$env:OPENBLAS_NUM_THREADS='1'` before running `verify`.
The three independent PDF/PNG panels share the same axis-frame dimensions, spine widths, typography and crop. Times New Roman is selected from Windows fonts when available; use `--font-dir` with `times.ttf` and `timesbd.ttf` to obtain the same font elsewhere. Original bar colors and the label **Random Position** are preserved. CSV and plot-audit files are also exported.

### Recorded attempt rates (bps/Hz), not reproduced historical targets

| Speed | Dreamer | PPO | DAFT grid | Time-selected positions, AFDM rate | Random positions |
|---|---:|---:|---:|---:|---:|
| 50 km/h | 15.242103 | 13.739768 | 19.416280 | 13.257986 | 5.919826 |
| 500 km/h | 18.301658 | 15.277230 | 19.418465 | 13.252067 | 5.920893 |

Dreamer/PPO rates average terminal episodes during the final 5k of the first 50k real environment interactions. Training curves use 1k-interaction batches and a trailing, causal five-batch mean. Random references average 100 independent joint positions on the corresponding speed's channel. See [aggregation rules](docs/TRAINING_CURVE_RULES.md).

## Preserved experiments

| Setting | 50 km/h | 500 km/h |
|---|---:|---:|
| Dreamer seed | 3501 | 5215 |
| PPO seed | 3501 | 5215 |
| Channel seed | 7 | 7 |
| Parallel environments / episode actions | 20 / 50 | 20 / 50 |
| Maximum coordinate displacement per action | 1 mm | 1 mm |
| Dreamer imagination horizon | 10 | 50 |
| Dreamer batch size / train ratio | 16 / 256 | 8 / 128 |
| Dreamer training budget | 50k | 50k |
| PPO training budget | 50k | 50k |
| Budget shown in Fig. 4 | 50k | 50k |

The learning rates, entropy schedules, replay settings, network sizes, task constraints and physical channel parameters are in `experiments/<speed>/configs/`. Recorded Dreamer `resolved_config.yaml` files retain the full model configurations. Actual channel settings are `N=128`, `P=20`, transmit power 1.5 W and noise −95 dBm.

### Train from scratch

Training uses Linux/WSL, Python 3.12 and an NVIDIA CUDA GPU. Create separate Dreamer and PPO environments using `requirements/dreamer.txt` and `requirements/ppo.txt`; the original 500 km/h dependency inventories are in `requirements/locks/`. The preserved installed Dreamer package is included in `vendor/` and selected by the launcher.

```bash
.venv-dreamer/bin/python scripts/reproduce.py train --speed 50 --algorithm dreamer --run-dir runs/50kmh/dreamer_seed3501
.venv-dreamer/bin/python scripts/reproduce.py train --speed 500 --algorithm dreamer --run-dir runs/500kmh/dreamer_seed5215
.venv-ppo/bin/python scripts/reproduce.py train --speed 50 --algorithm ppo --gpu-python .venv-dreamer/bin/python --run-dir runs/50kmh/ppo_seed3501
.venv-ppo/bin/python scripts/reproduce.py train --speed 500 --algorithm ppo --gpu-python .venv-dreamer/bin/python --run-dir runs/500kmh/ppo_seed5215
```

All four training commands stop after 50k real interactions. PPO retains the recorded absolute 20k entropy schedule. Add `--dry-run` to inspect a command. Every run requires a fresh directory and records configurations, source hashes, dependency inventories and launch settings before training. See the Chinese guide for CUDA setup, baseline search and plotting new training logs.

## Provenance and reproducibility

The **50 km/h recorded attempt** was trained from scratch with **seed3501 for both methods**. It failed to reproduce the historical result. Later recovered-source trials and a checkpoint continuation are archived separately under `reproduction_attempts/`. Original train.py/agent.py were reconstructed from the archived package and timestamped prelaunch edits. Source recovery, restored XLA settings and remaining runtime differences are documented in [recovery](docs/HISTORICAL_50KMH_RECOVERY_CN.md). The historical results are preserved in Git history and a separately hashed local archive. They are not mixed into the fresh comparison.

The new case publishes complete compressed 50k-action logs, terminal records, optimizer diagnostics, launch records, source copies verified against pretraining hashes and an independent physical audit. The failed attempt used seeded uniform prefill and data_loaders0; active configs now restore historical data_loaders2 and recovered original train/agent sources; its model sizes, H10, batch settings, training ratio and learning schedules are retained. PPO uses the same training seed and records executed actions and raw physical rewards. The **500 km/h** trainers and results retain their original pretraining provenance.

New trajectories can vary with JAX/CUDA, hardware and asynchronous data loading. Plot new runs from their own recorded data. No pretrained weights are distributed.

Local reconstruction changes and original/current source hashes are recorded in the source manifests.

```text
experiments/50kmh/       Configs, selected historical sources, terminal records and references
experiments/500kmh/      Configs, pretraining sources, terminal records and references
reproduction_tools/     Shared launch, record verification, baseline and figure tools
requirements/           Analysis/training dependencies and original launch inventories
vendor/                 Preserved installed Dreamer source and its license
scripts/reproduce.py    Entry point for verify/audit/train/plot/plot-all/baseline
tests/                  Record causality, protocol and trainer-config checks
```

The repository contains the code and records needed for these two Fig. 4 experiments. Archived metadata under `recorded/` retains the original data provenance; executable experiment settings are in `configs/`.
