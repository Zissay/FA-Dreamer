# 50 km/h: historical recovery and authentic reproduction attempts

> **最新从零验证：** 原参数、seed3501、空模型/空回放，两轮均完整执行50k；最后5k均值为 **13.623687 / 14.351138 bps/Hz**，未达到原18.837550。全部动作/源码/物理核验通过。原30k续训仅作诊断，不计作从零复现成功。见[完整报告](../../docs/fresh_repeat_audit_20261014.json)。
> **后续独立从零验证：** 完整50k，末5k **13.115381 bps/Hz**，二次抬升门槛未达到。来源、初始化、逐批轨迹及1125项CPU速率核验通过。此实验使用明示的估计启动顺序，不代表恢复了唯一原历史状态。[结果与限制](../../docs/refined_owner_prefix_full_audit_20261014.json)。
> **原生异步重复组：** 三轮均为seed3501完整50k，未覆盖线程顺序。末5k **15.602728 / 15.764732 / 13.183422**；全部审计通过，但都未复现。见[组审计](../../docs/native_async_repeat_audit_20261014.json)。

> **复现状态：未成功。** 原 50 km/h 目标为 Dreamer 18.837550、PPO 15.441477 bps/Hz。当前 `recorded/` 中的 15.242103 / 13.739768 是未达到原结果的真实尝试，不能替换论文结果。原 PPO seed 为3004；后续两种方法统一3501，旧数据不会改标 seed。见 `experiments/50kmh/reproduction_status.json`。

Paper panel (a), and the 50 km/h group of panel (c).

* Dreamer: seed 3501, 50,000 genuine interactions, H=10, 20 environments.
* PPO: seed 3501, 50,000 genuine interactions, with the recorded learning
  parameters and absolute 20k entropy schedule.
* Channel seed 7, Doppler 1297.193703548369 Hz, 1 mm action increments,
  50-action center-start episodes. Complete configs are under `configs/`.
* Source: a reconstruction based on the 2026-09-11 historical package;
  train.py/agent.py were recovered using timestamped prelaunch edits. Fresh trials and checkpoint continuation are explicitly separated.
  Dreamer uses seeded uniform prefill (1003504) and restored historical data_loaders=2 in the current recovered version.
  The retained model sizes, H10, batch16, train ratio256 and schedules are explicit.
  See `docs/SOURCE_SEARCH_50KMH.json` and the source manifests for all changes.

## Entry points (run from the repository root)

```bash
python scripts/reproduce.py plot --speed 50 --allow-unreproduced
python scripts/reproduce.py train --speed 50 --algorithm dreamer --run-dir runs/50kmh/dreamer_seed3501 --dry-run
python scripts/reproduce.py train --speed 50 --algorithm ppo --run-dir runs/50kmh/ppo_seed3501 --dry-run
```

Remove `--dry-run` only after installing the separate training environments
described in the root README. Both learners use the paper's 50k budget.

Failed-attempt final-window means (bps/Hz): Dreamer 15.24210271381839;
PPO 13.739767931943193. See `expected_results.json` for all independent references.
Terminal CSVs retain 1,000 genuine terminal actions per learner,
with their actual interaction indices and positions. `recorded/provenance.json`
records the complete new interaction-file hashes and terminal extraction rule.

Complete compressed logs, optimizer diagnostics and verified source snapshots
are under `recorded/`. Their physical/action/source audit is published alongside
them. This attempt failed to reproduce the historical comparison;
the original artifacts remain available in Git history and an external backup.

```bash
python scripts/reproduce.py audit --dreamer-run experiments/50kmh/recorded/dreamer --ppo-run experiments/50kmh/recorded/ppo
```
