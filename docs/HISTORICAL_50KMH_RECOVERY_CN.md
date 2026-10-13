# 原50km/h实验源码恢复与结果验收

## 当前目标

恢复原实验并复现先前报告的性能。原 Dreamer 最后5k交互的终点均值为18.837550331174892 bps/Hz，原 PPO 为15.441476879351 bps/Hz。
首次重建运行得到15.242103 /13.739768，未达到目标，不能替换论文图4或相关文字。
完整训练、动作核验、物理核验通过与“原结果复现成功”分别报告。

## 找到的源代码证据

当天13:34生成的便携包早于19:31的原实验启动，单独使用该包不能代表原启动源码。
原训练线程保留了以下启动前修改；时刻为北京时间：

| 时间（2026-09-11） | 修改 |
|---|---|
| 17:05:31 | 增加 imagination_start_mode，默认uniform；增加相应agent分支及诊断输出 |
| 18:42:52 | 用私有RandomState固定预填充，seed=训练seed+1000003 |
| 19:21:32 | data_loaders可配置，默认保留2 |
| 19:31:43 | 启动原scheduled H10、20环境、seed3501、50k实验 |

这些修改已按原补丁逐段应用到当天代码包，保存于
`experiments/50kmh/historical_target/source/`；当前Dreamer入口与agent采用该恢复版本。
补丁上下文、事件ID、时刻、基包哈希及恢复后源码哈希见该目录的`recovery_manifest.json`。
由于原启动时没有保存源码哈希，此恢复仍须用完整训练验证性能，不能仅凭代码恢复宣布成功。

恢复版全部2260个预填充动作，按每批匹配后的四维位置及速率与原运行完全一致，最大差为0。
见`prefill_historical_match_50kmh.json`；这只确认预填充轨迹，不代替后续训练结果检查。

PPO缺失的`truncated_gaussian_policy.py`与`audited_ppo.py`也已通过9月8日代码包及原启动前修改核对，
与当前使用版本一致（比较时仅统一换行）。见`historical_target/ppo/dependency_recovery.json`。
原PPO模型内的Python、PyTorch、SB3、NumPy、Gymnasium版本与本次CPU训练环境一致。

## 原始数据核验

原两种方法前50k中的各1000个终点均用相同50km/h信道独立复算。
最大误差：Dreamer约9.95e-14，PPO约6.28e-11 bps/Hz。
数值及原日志哈希见`historical_50kmh_physical_check.json`。
这证明终点位置和物理速率一致，不代替从零训练复现，也不证明完整训练流程的全部来源。

## 同seed要求

当前可执行配置固定Dreamer与PPO均为3501；500km/h均为5215。
原50km/h记录的Dreamer为3501、PPO为3004，历史元数据保持真实seed。
旧PPO记录仅是待核对的历史性能目标，不能冒充统一3501后的结果。
完整同seed对比需要单独验证，不能将历史两条曲线直接拼接为新的公平比较。

## 尚存在的非确定性

原data_loaders=2会由两个Prefetch线程共享Uniform回放采样器。
随机数分配给哪个批次位置、采样时可见哪些新经验，取决于线程时序。
原Replay.save()只保存经验文件，没有保存采样器RNG状态或每次训练批次的采样顺序。
因此相同seed与配置并不保证逐点相同的轨迹。
恢复版与原运行首个训练窗口的更新计数、调度系数和想象起点数量一致，但损失已有差别，
见`first_window_comparison_50kmh.json`。异步回放是已确认的非确定性机制，尚不能据此断言它解释了全部性能差距。

## 验收方式

恢复版从空模型开始，原设置data_loaders=2，固定50k真实交互，保留全过程和启动前源码快照。
最终指标按最后5k真实训练交互中的终点取均值，不采用最佳checkpoint、最高曲线点或训练后评估。

```bash
python scripts/reproduce.py compare-historical --dreamer-run runs/50kmh/dreamer_seed3501_recovered_source_20261013
```

用户接受原参数与seed一致、二次抬升趋势和最终约18.84的性能相近，不要求逐点或两位小数完全一致。
比较工具采用±0.2bps/Hz的明确范围，并要求初期抬升、后期增益至少达到原二次抬升的一半；检查源码、原学习参数、信道、seed及XLA启动配置。
该检查用于从空模型开始的完整50k训练，检查点续训与同seed PPO比较分别报告。
失败尝试全部保留，不能反复抽取线程时序或挑选高分运行宣称复现。

## 恢复版完整运行结果

恢复版已从零完成50k交互，Dreamer最后5k终点均值为 **15.26364726 bps/Hz**。
与原18.83755033相差 **-3.57390307 bps/Hz**，性能复现状态为 `NOT_REPRODUCED`。
源码、配置、完整动作和物理速率核验通过；该核验不代替性能复现。统一seed3501的PPO实跑均值为13.73976793，旧15.44147688对应seed3004。
本次记录单独发布于`experiments/50kmh/reproduction_attempts/recovered_source_20261013/`，不替换论文图4。

## 用户提供目录与原版本的对应关系

`E:/fa_afdm_dreamerv3/data/50km-h/dreamer/`中的`plot_audit.json`指向原seed3501训练目录。
`+random/dreamer_parallel_batch_means.csv`的50个原始批次均值及五批因果均值与原训练终点记录一致，最大差约1.42e−14。
原训练目录为`runs/user/t1_dreamer_h10_scheduled_env20_seed3501_50k`，包含全程日志、回放经验及5k到50k的模型检查点。
原曲线确实有二次抬升，不是累计最优值，也不是训练后评估曲线。见`provided_50kmh_figure_trace.json`。

## 源码之外的运行差异

原9月11日启动未使用`--xla_gpu_deterministic_ops`；它在9月12日才加入。
15.263647的恢复源码试验使用了后加的参数，因此不能代表原XLA配置的完整检验。
已恢复原XLA参数、移除后来强制的PYTHONHASHSEED、保持原学习参数与seed3501，重新从零完成50k：最终 **13.12229165 bps/Hz**，仍未复现。
该试验全部50k动作、1000终点及1125次CPU独立速率复算通过，记录单独保存在`reproduction_attempts/original_runtime_20261013/`。

历史GPU为RTX4060/驱动591.86，当前同型号GPU驱动为610.88。
Windows设备安装日志记录2026-10-09 21:36安装610.88，晚于原实验；见`gpu_runtime_difference_50kmh.json`。
这是已确认的运行环境差异，尚未验证它是否造成性能差距。
当前两个异步回放线程仍无法恢复当年的批次采样时序。
原训练约18k已偶尔采到≥17的终点，两次恢复源码的从零试验均未采到，早期探索经历已经分化。

## 原检查点的独立实测

使用原策略代码，在50km/h、相同50步任务上，对原检查点各做100个独立采样episode，不更新参数：

| 原训练检查点 | 终点平均速率（bps/Hz） |
|---|---:|
| 30k | 10.847807 |
| 35k | 13.531384 |
| 40k | 17.785754 |
| 50k | 18.909014 |

评估前后全部模型变量哈希一致。实测支持原模型确实学到了较高性能，也保留了随检查点提升的现象。
这些是单独的诊断动作，不计入训练预算、不替换原在线训练均值18.837550。
代码：`scripts/check_historical_50_checkpoints.py`；完整数据：`historical_checkpoint_policy_check_20261013.json`。

## 保持参数与seed的30k→50k续训验证

恢复原30k模型、actor/critic/world model及优化器状态（各7080次更新），保持H10、batch16、ratio256、data_loaders2、学习率/熵调度、seed3501和原信道。
回放仅使用原30k之前的数据；新增20k全部由实际策略产生，未使用原30k之后的动作、位置或速率。
可重建JAX RNG调用数为8497；回放采样器从seed0重启。
原采样器队列/历史顺序及最后一次更新前的policy carry未保存，所以这是重建续训，不是逐状态精确恢复。

结果再次出现平台后的第二次抬升，约39k首次出现≥17的终点，最后5k均值 **18.67251692**，与原值差 **−0.16503341 bps/Hz**。
全部20k动作的位置变化、奖励及400终点CPU物理复算通过。见`second_rise_restore_audit_20261013.json`。
初次启动因优化器字段路径遗漏`/value`在执行任何实际动作前失败；修正后仅一次完整续训，结果完整保留。
它证明恢复代码能继续原学习过程并达到接近原性能；从零训练及同seed PPO对比仍未复现。不可把它标作新的独立50k训练。

### 在本机运行（WSL，仓库根目录）

检查点及44MB前缀回放文件保留在本机，GitHub不分发这些预训练权重/回放；输入文件哈希会在训练前检查。
必须使用新的输出目录，以下目录名是示例。

```bash
/home/user1/fa_afdm_dreamer_gpu/bin/python scripts/diagnose_second_rise_50.py \
  --parent-run /mnt/e/fa_afdm_dreamerv3/runs/user/t1_dreamer_h10_scheduled_env20_seed3501_50k \
  --prefix-file /mnt/e/fa_afdm_dreamerv3/output/analysis/second_rise_intervention/prefix_replay.npz \
  --run-dir runs/50kmh/restore30k_repeat

python scripts/audit_second_rise_50.py \
  --run-dir experiments/50kmh/reproduction_attempts/restore30k_20261013 \
  --plot-output figures/diagnostic_restore30k/second_rise.png
```

第二条命令可直接核验仓库中的压缩新训练日志和启动前源码快照，不需要原模型或GPU。
异步线程和当前运行环境使再次执行不保证逐点一致；本次续训值是实际测得值，没有作为训练终止条件。
诊断图本机位置：`figures/diagnostic_restore30k/second_rise.png`，橙色只展示30k之后的新续训段；原论文图保持独立。

## 2026-10-14：登记两轮从零验证

原启动记录要求全新目录，原metadata的`from_checkpoint=null`，原实验确为从零50k。用户明确要求同样从空模型、空回放开始；续训仅作机制诊断。
完整resolved config与原配置逐字段相同，63个当前库源码/配置文件与保留副本逐字节相同，见`full_runtime_config_check_20261014.json`。文件时间均早于原实验，但mtime不是原启动哈希证明。
已在结果产生前登记两轮原参数/seed3501的独立从零试验：`fresh_repeat_protocol_20261014.json`。每轮50k，顺序执行；无论分数高低均完成并报告，禁止只保留最高分。
原双线程回放时序未记录，当前GPU驱动也不同；本轮用于观察相同配方的运行波动，不改变学习率、熵调度或seed，不加载任何原模型/经验。
