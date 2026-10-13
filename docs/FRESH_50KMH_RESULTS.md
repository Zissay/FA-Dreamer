# 50 km/h 实验更新

无法确认原 50 km/h 运行的完整启动源码，查找证据见 `SOURCE_SEARCH_50KMH.json`。
当前发布结果来自本次使用仓库代码从零执行的两次完整训练，Dreamer 与 PPO 均使用 seed **3501**、
信道 seed **7** 和 **50k** 真实交互。未加载原 checkpoint、未挑选高点 checkpoint，也未搜索更高分的种子。

| 最后 5k 真实交互的平均终点速率 | bps/Hz |
|---|---:|
| Dreamer | 15.24210271381839 |
| PPO | 13.739767931943193 |
| Dreamer 相对 PPO 的差值 | 1.502334781875197 |
| DAFT 81点网格参考 | 19.41627979996462 |
| Dreamer 与 DAFT 参考的差距 | 4.174177086146230 |

原 50 km/h 图中的 **18.84 / 15.44** 已由新运行的 **15.24 / 13.74** 替换。
论文中对应的约 **3.40** 提升及约 **0.58** DAFT 差距，应相应改为约 **1.50** 和 **4.17**。
500 km/h 的已有结果保持 **18.30 / 15.28**，本次未重新训练该组。

## 核验结果

- 每种方法恰好 50,000 个连续真实动作、20 个环境、1,000 个终点。
- 全部动作/位置符合 1 mm 步幅与固定中心起点；全部记录的原始物理奖励符合相邻速率差及终点奖励。
- 每种方法独立复算 1,125 个位置，其中包括全部 1,000 个终点；CPU/GPU 最大速率差小于 1.2e-13 bps/Hz。
- Dreamer 的全部 2,260 个预填充动作与固定 RNG seed 1003504 对应；模型、actor、critic 梯度有限且非零。
- PPO 保留 50 条真实优化器更新记录，实际 actor 权重更新已核验。
- 源码与配置副本逐字节符合启动前保存的哈希；真实依赖、完整 gzip 日志和核验报告已发布。

新训练包含两项明确的复现控制：Dreamer 固定随机预填充，并使用同步 `data_loaders=0`。
主要学习参数保留历史 H10、batch16、train ratio256 及原学习率/熵调度；PPO 改用与 Dreamer 相同的种子。
这些结果是一组匹配种子的实跑对比，不能把它们描述为多个种子的统计结论。

## 图 4 注释参考

```latex
\caption{Training performance comparison at receiver speeds of
(a) 50 km/h and (b) 500 km/h, and (c) average terminal rates over the
final 5k training interactions. Each learning method uses 50k real
environment interactions, with a common training seed within each
speed setting (3501 at 50 km/h and 5215 at 500 km/h).}
```

数据核验入口与使用说明见 `REPRODUCTION_CN.md`。图形重建使用原始训练终点记录的因果平均，
柱状图直接采用最后 5k 的平均值；PPO 的训练后确定性评估不参与这些图表数值。
