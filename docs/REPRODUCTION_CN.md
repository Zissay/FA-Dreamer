# 50 km/h 与 500 km/h 论文结果复现说明

当前图 4 的三个子图分别是 50 km/h 训练曲线、500 km/h 训练曲线，以及两种速度的最终速率柱状图。
每种速度分别训练，并非将同一个 50 km/h 模型直接用于 500 km/h 泛化测试。

## 对应实验

| 参数 | 50 km/h | 500 km/h |
|---|---:|---:|
| Dreamer 训练 seed | 3501 | 5215 |
| PPO 训练 seed | 3004 | 5215 |
| 信道 seed | 7 | 7 |
| 并行环境 | 20 | 20 |
| 回合真实动作数 | 50 | 50 |
| 每步每个坐标最大位移 | 1 mm | 1 mm |
| 绘图交互预算 | 前 50k | 前 50k |
| Dreamer imagination horizon | 10 | 50 |
| Dreamer batch size | 16 | 8 |
| Dreamer train ratio | 256 | 128 |
| Dreamer replay lambda | 0.95 | 1 |
| Dreamer data loaders | 2（历史值） | 0 |
| Dreamer prefill | 历史随机 prefill | 10k stratified-goal prefill |
| Dreamer/PPO 训练预算 | 50k | 50k |

完整学习率、熵系数、调度、网络配置、动作分布与 replay 参数在各速度目录的 `configs/` 中。
`recorded/dreamer/resolved_config.yaml` 是原运行实际保存的完整模型配置。

## 一键检查和重绘

从仓库根目录执行：

```bash
python -m pip install -r requirements/analysis.txt
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py verify --physics
python scripts/reproduce.py plot-all --output-dir figures/reproduced
```

生成三个独立、统一裁边的 PDF/PNG。CSV 和 audit.json 一并输出。
横轴是真实环境交互次数，每 1k 交互含 20 个终点 episode；图中每点是最近 5 批的因果平均。
柱状图中的 Dreamer/PPO 数值是前 50k 交互最后 5k 的平均终点速率。
Random Position 是在各自速度信道下，对 100 组独立随机位置计算 AFDM 速率的平均值。
没有使用未来数据、累积最大值、评估曲线或穷搜结果替换训练记录。

## 从零训练

训练使用 Linux/WSL、Python 3.12 和 NVIDIA CUDA GPU。示例中的 Python 路径按本机环境填写：

```bash
python3.12 -m venv .venv-dreamer
.venv-dreamer/bin/python -m pip install -r requirements/dreamer.txt
python3.12 -m venv .venv-ppo
.venv-ppo/bin/python -m pip install -r requirements/ppo.txt

.venv-dreamer/bin/python scripts/reproduce.py train --speed 50 --algorithm dreamer --run-dir runs/50kmh/dreamer_seed3501
.venv-dreamer/bin/python scripts/reproduce.py train --speed 500 --algorithm dreamer --run-dir runs/500kmh/dreamer_seed5215

.venv-ppo/bin/python scripts/reproduce.py train --speed 50 --algorithm ppo --gpu-python .venv-dreamer/bin/python --run-dir runs/50kmh/ppo_seed3004
.venv-ppo/bin/python scripts/reproduce.py train --speed 500 --algorithm ppo --gpu-python .venv-dreamer/bin/python --run-dir runs/500kmh/ppo_seed5215
```

两种速度的 Dreamer 与 PPO 均固定训练 50k 真实环境交互。
PPO 保留原种子和其他训练参数，包括按真实 timestep 计数的 20k 熵调度。
`configs/` 用于当前实验运行，`recorded/` 中的原始元数据用于核验已发布曲线的数据来源。
先加 `--dry-run` 可以检查命令，不会启动训练。

输出目录必须是新目录；启动时保存源码哈希、配置、依赖列表、运行命令和环境信息。
训练器仍保存自己的模型 checkpoint 与 replay，新的大文件在 `runs/` 中，不纳入 Git。
新训练的图使用实际新日志重画：

```bash
python scripts/reproduce.py plot --speed 50 --dreamer-run runs/50kmh/dreamer_seed3501 --ppo-run runs/50kmh/ppo_seed3004 --output-dir figures/new_run

python scripts/reproduce.py plot-all --dreamer-50-run runs/50kmh/dreamer_seed3501 --ppo-50-run runs/50kmh/ppo_seed3004 --dreamer-500-run runs/500kmh/dreamer_seed5215 --ppo-500-run runs/500kmh/ppo_seed5215 --output-dir figures/new_all
```

## 信道与离线基线

两组图实际采用 `N=128`、`P=20`、`P_t=1.5 W`、噪声 `-95 dBm`、波长 `0.0107068735 m`。
50/500 km/h 的固定 Doppler 分别为约 1297.19/12971.94 Hz。完整参数以各 `protocol.json` 为准。
这些参数分别保存在各速度的实验配置中。

搜索坐标范围均为 `[-0.05,0.05] m` 的四维可达区域。DAFT 域：81 点/坐标、1.25 mm、`81^4` 次查询。
时域：21 点/坐标、5 mm、`21^4` 次查询。
时域选位目标是 strongest-delay-tap SINR，其余抽头视为 ISI；图中报告这些位置的 AFDM 速率。
离散网格最优不等于已证明的连续空间全局最优。

```bash
python scripts/reproduce.py baseline --speed 50 --method random --device cpu --output runs/50kmh/random.json
python scripts/reproduce.py baseline --speed 500 --method time --device cuda --output runs/500kmh/time_grid21.json
python scripts/reproduce.py baseline --speed 500 --method daft --device cuda --output runs/500kmh/daft_grid81.json
```

81^4 穷搜需要较长时间。`verify --physics` 只复算存储的最佳位置和 100 个随机位置，
检查配置与速率是否一致，不会重新遍历整个网格。

## 可复现范围

* 原始终点记录、曲线聚合规则和最终数值可直接核验和重绘。
* 500 km/h 的 Dreamer 与 PPO 源码取自各自训练前快照，完整依赖记录保存在 `requirements/locks/`。
* 50 km/h 的超参数和终点记录完整，源码采用 2026-09-11 历史包；缺少原训练启动快照和原依赖锁。
  两个 PPO 依赖模块从保留的工作区恢复，文件名与哈希写入 `ppo_source_manifest.json`。
  这条路径是可运行的历史复现候选，不能宣称原始训练曲线必然逐点或逐位重现。
  历史源码尚未显式命名 `afdm.imagination_start_mode`；原配置为 `uniform`、start count 为 0，
  历史实现直接使用全部 replay starts。这一配置字段差异由检查脚本明确列出。
* JAX/CUDA、异步数据加载和硬件可能影响新的训练轨迹。新训练应报告实际结果，不能强制对齐旧曲线。

两个速度的所有源码、配置、基线和终点数据都有 SHA-256 清单。
附带的绘图辅助脚本已删除不用的历史 PPO 默认路径，比较曲线必须显式指定运行目录。
这些编辑在源码清单中同时保留原哈希和当前哈希，训练逻辑保持原实现。

## 检查训练配置

普通检查不启动训练。以下四条配置检查需要使用对应的训练环境：

```bash
python -m unittest discover -s tests -v
.venv-dreamer/bin/python tests/check_training_config.py --speed 50 --algorithm dreamer
.venv-dreamer/bin/python tests/check_training_config.py --speed 500 --algorithm dreamer
.venv-ppo/bin/python tests/check_training_config.py --speed 50 --algorithm ppo
.venv-ppo/bin/python tests/check_training_config.py --speed 500 --algorithm ppo
```

Dreamer 检查会把实际训练器生成的完整配置与原运行保存的 YAML 对照；50 km/h 的一个历史隐式字段单独报告。
PPO 检查会把启动参数解析结果与原 JSON 对照。已执行的验证范围见 `docs/validation_20261013.json`。

GPU 速率服务可以单独检查：

```bash
.venv-dreamer/bin/python tests/check_gpu_rate_service.py --speed 500 --algorithm dreamer
.venv-ppo/bin/python tests/check_gpu_rate_service.py --speed 50 --algorithm ppo --gpu-python .venv-dreamer/bin/python
```

这个检查启动临时服务，将该训练源码的 20 组位置的 CUDA 速率与其 CPU 计算比较，不进行学习。
