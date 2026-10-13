# 50 km/h：统一两阶段恢复训练入口

入口：`scripts/recover_staged_50.py`。

第一阶段复用原seed3501实验已有的前30k记录、30k模型及优化器检查点。第二阶段从该状态恢复，再产生20k真实动作。学习参数、信道、seed、学习率及熵调度保持原设定。

已完成的两阶段流程末段均值为**18.6725169195bps/Hz**。这属于恢复训练；原检查点依赖、重建回放和阶段边界都保留在元数据中。独立从零50k复现的状态仍为`NOT_REPRODUCED`。

整合验证见[`staged_recovery_result_20261014.json`](staged_recovery_result_20261014.json)。已逐字段核对50000条动作记录，除累计步数偏移及新增阶段标识外，原动作、位置、奖励和速率全部保留。公开的阶段终点和曲线在`experiments/50kmh/diagnostics/staged_recovery_20261014/`。

## 一个命令恢复、核验并汇总

在WSL的仓库根目录执行，输出目录必须是全新的。

```bash
/home/user1/fa_afdm_dreamer_gpu/bin/python scripts/recover_staged_50.py resume \
  --parent-run /mnt/e/fa_afdm_dreamerv3/runs/user/t1_dreamer_h10_scheduled_env20_seed3501_50k \
  --prefix-file /mnt/e/fa_afdm_dreamerv3/output/analysis/second_rise_intervention/prefix_replay.npz \
  --run-dir runs/50kmh/staged_restore50k_new \
  --plot
```

执行顺序：

1. 校验原实验元数据、日志、30k检查点和前缀回放的哈希，核对seed3501及原学习参数。
2. 调用已验证的恢复训练器，恢复world model、actor、critic及三个优化器，按原累计计数继续调度，完整执行20k。
3. 独立复算新增400个终点的信道速率，校验全部20k动作、位置、奖励和启动前源码快照。
4. 在新运行目录的`integrated/`中汇总两阶段日志、曲线和来源清单。

将`resume`换成`plan`并去掉`--plot`，只检查输入并显示运行计划，不执行训练。
这个入口调用原恢复训练器；网络结构和学习更新代码没有修改。复用的第一阶段不会被重新训练。

## 汇总已完成的两阶段记录

不重新训练，也不需要GPU：

```bash
python scripts/recover_staged_50.py assemble \
  --parent-run /mnt/e/fa_afdm_dreamerv3/runs/user/t1_dreamer_h10_scheduled_env20_seed3501_50k \
  --continuation-run experiments/50kmh/reproduction_attempts/restore30k_20261013 \
  --output-dir figures/staged_restore50k_export \
  --plot
```

汇总输出：

| 文件 | 内容 |
|---|---|
| `original_first30k.csv.gz` | 原阶段前30000条真实动作 |
| `lineage_interactions.csv.gz` | 50000条分阶段记录，保留阶段内步数和环境编号 |
| `lineage_terminal_interactions.csv` | 600个原终点＋400个新增终点 |
| `lineage_curve.csv` | 每1k终点均值及因果五批滑动均值 |
| `lineage_curve.png` | 可选对照图，在30k处标出恢复边界 |
| `lineage_manifest.json` | 参数、seed、输入/输出哈希、独立物理核验及恢复范围 |

原30k之后的历史动作、位置和速率不会进入恢复训练或汇总曲线；原完整日志的哈希仅用于确认输入来源。两阶段环境编号是各阶段局部编号，不能直接当作同一个worker的连续轨迹。

## 保留的恢复限制

原双线程采样顺序和最后一次更新前的policy carry没有保存。采样器从seed0重新开始，training carry重新初始化，policy carry按已保存模型和可恢复的随机调用数重建。这是近似状态恢复，再次运行仍可能产生不同轨迹。不能仅因合并了入口和日志，就认定独立从零实验已经复现。

检查点和44MB前缀回放保留本机；GitHub不分发预训练权重。已完成阶段的来源、核验和终点数据均保留，论文原图未替换。
