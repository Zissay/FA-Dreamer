# 从零整合两个训练过程：30k＋本轮状态恢复20k

入口：`scripts/train_integrated_50.py`。这是用户要求的学习过程整合，前30k在本轮从空模型、空回放实际训练，后20k接续本轮状态。整个流程不接受原实验检查点、历史回放或目标曲线作为训练输入。

WSL仓库根目录，一个命令执行完整50k：

```bash
/home/user1/fa_afdm_dreamer_gpu/bin/python scripts/train_integrated_50.py \
  --run-dir runs/50kmh/dreamer_seed3501_integrated_fresh_new
```

加`--dry-run`只显示计划。完整运行要求新的输出目录，seed3501及原学习参数由保留配置固定。

## 两段如何接起来

1. 第一阶段调用恢复出的原训练核心，执行30000真实动作，记录本轮实际观测的前30个episode。保存自己的模型和三个优化器（各7080次更新）、30620回调计数及回放前缀。
2. 第二阶段加载该目录中的本轮检查点，验证模型变量逐项摘要和优化器计数未变。用自己的前30k回放，按已验证的后段恢复方法继续20000真实动作。
3. 保持H10、batch16、length64、ratio256、data_loaders2、原学习率/熵调度、信道和seed3501。两阶段累计训练计数延续，调度不回到起点。
4. 完成后生成全部50000条真实动作和模型来源记录。原始阶段日志单独保留；累计日志把两阶段日志编号映射回真实driver worker，保证逐环境轨迹可核验。

## 需要如实记录的区别

原实验没有主动30k恢复边界。这次整合会在30k重启回放采样器和双线程，清空training carry，并用本轮已保存模型重建policy carry及随机调用位置。它是一项明确声明的两阶段从零实验，不能隐藏该协议变化。

该版本是否达到二次抬升和约18.84，要以完整运行结果判断。原参数和相同seed不保证相同异步轨迹。一个完整试验已预先登记在`integrated_fresh_protocol_20261014.json`，无论结果如何均保留。

既有`recover_staged_50.py`用于恢复历史30k；本入口的第一阶段会重新训练，后段的权重和观测都来自本轮。

## 已完成验证：当前整合版本未复现原性能

方案在启动前提交并推送（`4933de6`），单轮完整执行50000真实动作，没有选取最佳片段或加载历史权重。前30k末5k均值12.759983；全程末5k为**13.071402858bps/Hz**。中段20k–35k平台12.784379，后期净增仅0.287023，未出现原二次抬升，状态为`NOT_REPRODUCED`。

两阶段模型变量摘要在恢复前后完全一致，三个优化器从7080延续到12175。逐字段核对50000动作，30000条自己的回放动作与阶段日志对齐通过；600＋400共1000终点、1125次独立CPU速率复算通过，最大误差9.41e−14。没有历史模型或历史观测作为训练输入。

这说明已把两段实际学习过程接起来，并完成从随机初始化到50k的验证，但尚未达到用户要求的原性能。原30k检查点恢复到18.6725是另一个依赖历史状态的结果，不能替代本轮从零训练结果。

完整来源及核验：[integrated_fresh_audit_20261014.json](integrated_fresh_audit_20261014.json)。公开记录在`experiments/50kmh/reproduction_attempts/dreamer_seed3501_integrated_fresh_20261014/`，权重及完整前缀回放保留本机。

本机完整核验、归档及对照图重建：

```bash
python scripts/archive_integrated_50.py \
  --run-dir runs/50kmh/dreamer_seed3501_integrated_fresh_20261014 \
  --output docs/integrated_fresh_audit_20261014.json \
  --plot-output figures/integrated_fresh_20261014/comparison.png
```

已公开曲线无需GPU或模型即可重画：

```bash
python scripts/archive_integrated_50.py \
  --recorded-report docs/integrated_fresh_audit_20261014.json \
  --plot-output figures/integrated_fresh_20261014/comparison.png
```
