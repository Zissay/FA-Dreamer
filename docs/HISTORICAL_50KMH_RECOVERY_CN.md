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

按论文两位小数检查是否仍为18.84，并同时核对恢复源码、原学习参数、信道及seed。
该检查只判定已报告的Dreamer最终均值；完整轨迹与同seed PPO比较仍需分别报告。
失败尝试全部保留，不能反复抽取线程时序或挑选高分运行宣称复现。

## 恢复版完整运行结果

恢复版已从零完成50k交互，Dreamer最后5k终点均值为 **15.26364726 bps/Hz**。
与原18.83755033相差 **-3.57390307 bps/Hz**，按论文两位小数检查状态为 `NOT_REPRODUCED`。
源码、配置、完整动作和物理速率核验通过；该核验不代替性能复现。统一seed3501的PPO实跑均值为13.73976793，旧15.44147688对应seed3004。
本次记录单独发布于`experiments/50kmh/reproduction_attempts/recovered_source_20261013/`，不替换论文图4。
