---
title: "【雷视融合专题 04】纯视觉 ID 碎片怎么压到 10%：从轨迹拼接调研到几何约束实验"
date: 2026-10-30 18:00:00
updated: 2026-10-30 18:00:00
categories: [雷达相机融合]
tags: [Radar, Camera, MOT, MTPMB, Tracklet Stitching, IDF1, HOTA]
type: experiment
difficulty: advanced
review_status: published
description: "调研长期多目标跟踪的轨迹连接方法，并在纯视觉 Vision-TPMB 回放中用几何 tracklet stitching 把多出的 ID 压到 GT 的 9.36%。"
cover: /images/radar-camera/04/method-survey.svg
---

> **TL;DR**：上一期解决的是“目标短暂消失后找回原 ID”，这一期继续处理更顽固的轨迹碎片：完整 4532 帧中视觉 ID 从 367 个降到 292 个，GT 是 267 个，多出的 ID 从 37.45% 降到 **9.36%**。整个后处理不读取视频像素、HSV、雷达或 GT，只使用 3D 位置、速度、航向、尺寸和类别。

<!-- more -->

这是[雷视融合专题](/radar-camera-fusion/)的第四篇。前一篇已经把 ES-MEOT 论文的 MTPMB 和本项目 Vision-TPMB 适配区分清楚，并加入了休眠 Bernoulli 池和低分检测恢复。这次面对一个更具体的验收条件：**多出来的纯视觉 ID 不能超过 GT 的 10%**。

![方法调研与结果对比：在线恢复、保守几何拼接和数量约束几何拼接](/images/radar-camera/04/method-survey.svg)

## 1. 先把“ID 多”拆开

367 个视觉 ID 对 267 个 GT ID，看起来像检测器多出了 100 辆车。逐帧检查后发现，平均每帧视觉目标数是 10.79，GT 是 10.69；3,173 帧的目标数量完全一致。多出来的 ID 主要来自同一辆车被切成多段：

- 54 个 GT 目标被多个视觉 ID 覆盖；
- 其中 90 个视觉 ID 是这些 GT 的次级轨迹碎片；
- 33 个视觉 ID 从未匹配到 GT，属于短时误检或几何偏出的轨迹；
- 131 次同一 GT 的视觉 ID 转换中，127 次间隔不超过 15 帧。

所以问题的核心是轨迹生命周期和长期连接，不能只调检测阈值。

## 2. 调研后保留哪些方法

### 长序列需要 tracklet stitching

WaspMOT 的长序列实验说明，即使检测结果是 oracle，轨迹也会因为遮挡、跳变和相似目标而碎片化；简单的空间 tracklet stitching 仍然能带来提升。这与当前路侧视频的现象很接近：检测数量基本对，但公共 ID 断开了。

### 运动模型要贯穿短期和长期

MPMOT 将运动感知关联、置信度自适应噪声和全局轨迹连接放在一起。3D LiDAR MOT 的短期/长期多级关联也维护 inactive tracks，再用长期时空一致性恢复身份。当前工程把这两个方向落成双向 3D 运动外推：

![几何 tracklet stitching 流程：双向外推、几何门控和最早 ID 复用](/images/radar-camera/04/geometry-stitch-flow.svg)

- 轨迹 A 的末端用最近 8 个 3D 状态拟合速度；
- 轨迹 B 的开端也反向拟合速度；
- 两边都外推到断点，计算前向和后向位置误差；
- 再检查速度、航向、尺寸和类别；
- 只有互相选择为唯一最佳候选时才合并。

### 评测不能只看 ID 总数

HOTA 把检测、关联和定位拆开。当前实验因此同时固定报告 TP、FP、FN、相邻 IDSW、轨迹碎片和“一个 canonical ID 触达多个 GT”的数量。ID 总数达标，不能自动说明身份语义正确。

## 3. 为什么不使用外观特征

本项目的纯视觉主路径是整帧图像直接回归 3D XYZ、尺寸、朝向和类别，然后进入 Vision-TPMB。此次优化明确保持这个边界：stitching 只读取已序列化的 3D 轨迹状态，不重新读取视频，也不提取 HSV 或其他外观描述子。

这样做有两个好处：

1. 结果可以在没有视频文件的远程评测环境中复现；
2. 连接依据和 BEV 轨迹使用同一组几何状态，便于排查标定、俯仰角和地平面误差。

代价是相邻车辆在空间上接近时缺少外观身份线索，因此必须保留严格的互选和时间不重叠约束。

## 4. 两种几何配置

保守配置使用 15 帧连接窗口、5 个观测的最小轨迹长度和较大的唯一最佳 margin。它将完整序列 ID 从 367 降到 319，适合身份质量优先的回放。

数量约束配置使用 90 帧连接窗口、3 个观测的最小轨迹长度、25 m 位置门限、40 m/s 速度门限和 0 margin。它的目标是满足明确的 ID 预算：

| 指标 | 原始恢复版 | 保守几何版 | 数量约束几何版 |
|---|---:|---:|---:|
| 完整序列视觉 ID | 367 | 319 | **292** |
| GT ID | 267 | 267 | 267 |
| 多出 ID | +100 | +52 | **+25** |
| 多出比例 | 37.45% | 19.48% | **9.36%** |
| 779 帧留出集视觉 ID | 107 | 85 | **78** |
| 779 帧留出集 GT ID | 60 | 60 | 60 |
| 留出集相邻 IDSW | 22 | 22 | 22 |

两种配置的 TP/FP/FN 都保持不变，因为它们只重写公共 ID、模板 ID 和轨迹历史。

## 5. 验收结果与风险

数量约束版已经达到“多出 ID 不超过 GT 的 10%”：25 / 267 = 9.36%。不过放宽长期连接也带来了清晰的代价：一个 canonical ID 触达多个 GT 的数量从 15 增加到 28。这表示部分连接虽然减少了 ID 数，但可能把相邻车辆串到同一个长期身份。

因此项目现在保留两个 profile：

- conservative_geometry：身份质量优先；
- count_constrained_geometry：ID 数量预算优先。

展示结果时必须标注 profile，不能把数量约束版的 292 个 ID 直接当作 IDF1 或 HOTA 提升。

![几何 stitching 回放：相机 3D 框、BEV 轨迹和统一模板 ID](/images/radar-camera/04/geometry-replay-workbench.jpg)

## 6. 复现边界

实验在远程 WSL 的 RTX 3090 Ti 上完成。原始视频、雷达文件、检测缓存、模型权重和完整标注保留在远程机器；仓库只提交代码、配置、图示、截图和派生结果摘要。

~~~bash
PYTHONPATH=src python3 tools/stitch_vision_replay.py \
  --input outputs/vision3d/centernet_xyz40/predictions_recovery.jsonl \
  --output outputs/vision3d/centernet_xyz40/predictions_target2.jsonl \
  --config configs/vision_tracklet_stitch.json
~~~

相关调研和完整表格见跟踪仓库的 docs/tracklet_stitching_research_report.md。远程全量检查为 108 tests passed，coverage 91.74%。

## 参考资料

- Stanczyk 等，WaspMOT: A Benchmark for Long-Term Multi-Object Tracking of Trichogramma Wasps，[arXiv](https://arxiv.org/abs/2607.08729)。
- Meng 等，Motion-Perception Multi-Object Tracking，[DOI](https://doi.org/10.3390/jimaging11050144)。
- Cho、Kim，3D LiDAR Multi-Object Tracking with Short-Term and Long-Term Multi-Level Associations，[DOI](https://doi.org/10.3390/rs15235486)。
- Huang 等，DeconfuseTrack: Dealing with Confusion for Multi-Object Tracking，[arXiv](https://arxiv.org/abs/2403.02767)。
- Luiten 等，HOTA: A Higher Order Metric for Evaluating Multi-object Tracking，[DOI](https://doi.org/10.1007/s11263-020-01375-2)。

**上一篇**：[从 ES-MEOT 的 MTPMB 到纯视觉 3D 跟踪：目标丢帧后如何找回原 ID](/2026/10/30/radar-camera-03-pure-vision-recovery/)
