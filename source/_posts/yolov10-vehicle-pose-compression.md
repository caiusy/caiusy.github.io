---
title: 从 YOLOv10 到车辆关键点部署：一个 4M 参数小模型的压缩实战
date: 2026-08-01 18:30:00
updated: 2026-08-01 18:30:00
description: "从 CarFusion 数据、14 关键点双分支训练，到 FastHead、结构化剪枝、知识蒸馏与 P5-Zero 三门禁验收：复盘一个约 4M 参数车辆姿态模型为什么难压缩，以及最终如何取得 14.83% 的真实 FP16 加速。"
categories:
  - 计算机视觉
  - 工程实践
tags:
  - YOLOv10
  - Vehicle-Pose
  - 车辆关键点
  - CarFusion
  - 模型压缩
  - 知识蒸馏
  - PyTorch
type: tech
difficulty: intermediate
review_status: published
cover: /images/yolov10-vehicle-pose-compression/01-infographic-project-overview.webp
banner: /images/yolov10-vehicle-pose-compression/01-infographic-project-overview.webp
---

> 一个只有约 400 万参数的 YOLOv10n，还能不能在不使用 TensorRT 的前提下继续加速？答案不是“按比例剪掉通道”这么简单。本项目从 CarFusion 数据治理、14 关键点训练与验证出发，连续验证 FastHead、Backbone/Neck 改造、结构化剪枝、稀疏训练和知识蒸馏，最终找到一个同时通过速度、精度和显存门禁的 P5-Zero 方案。

这篇文章不只给出成功数字，也保留那些更有价值的失败证据：为什么参数少了 25%，延迟却只降 6%；为什么训练可以救精度，却救不了一个已经跑得更慢的执行图；以及为什么把数据移进 WSL ext4，收益可能远大于一次复杂的网络改造。

<!-- more -->

![YOLOv10 Vehicle Pose 项目核心成果：P5-Zero 延迟下降 14.83%，参数下降 15.14%，并通过精度门禁](/images/yolov10-vehicle-pose-compression/01-infographic-project-overview.webp)

## 1. 问题不是“给 YOLO 加一组关键点”

任务表面上很直接：检测一辆车，同时预测 14 个关键点。但要让它成为可训练、可验证、可部署的工程系统，至少需要闭环四条链路：

1. **数据链路**：授权包校验、双层解压、COCO Keypoints 合并、YOLO Pose 转换、固定切分、质检与预览。
2. **模型链路**：YOLOv10 P3/P4/P5 特征、one-to-many / one-to-one 双分支、14×3 关键点塔和 NMS-free 推理。
3. **评测链路**：Box/Pose mAP、PCK@0.05/0.10、NME、可见性和尺度分组，以及固定图像的视觉验收。
4. **部署链路**：同进程 paired fused-FP16 延迟、独立进程 CUDA 显存、制品哈希和可复现配置。

![从 CarFusion 授权数据到 YOLOv10 Vehicle Pose 部署验收的完整技术链路](/images/yolov10-vehicle-pose-compression/02-flowchart-end-to-end-pipeline.webp)

模型复用 YOLOv10 的 Backbone/Neck，在检测头之外增加两套纯卷积关键点塔。训练阶段保留 top-k=10 的 one-to-many 分配和 top-k=1 的 one-to-one 分配；推理阶段启用 `one2one_only=True`，跳过不会参与最终输出的 one-to-many 塔，并用同一组 anchor top-k 索引同步 gather bbox 与 14×3 keypoints。

这套设计的关键不是多一个 Head，而是让检测和关键点在双分支训练、NMS-free 解码和验证配对中始终保持一致。

## 2. 先建立一条可信的 baseline

项目使用 CarFusion 授权数据。转换后，完整验证集包含 3,606 张图像和 11,569 个实例。200 epoch baseline 的主要结果如下：

| 指标 | 结果 |
|---|---:|
| Box mAP50 / mAP50-95 | 0.818 / 约 0.665–0.668 |
| Pose mAP50 / mAP50-95 | 0.765 / 约 0.515 |
| PCK@0.05 / PCK@0.10 | 0.6686 / 0.8525 |
| NME | 0.05960 |

这里故意保留了 baseline mAP 的小范围差异：标准精度验证与最终 fused-FP16 部署复验不是完全相同的记录口径。项目没有从较有利的一组数字反推门槛，而是提前固定发布线：

- Box mAP50-95 不低于 `0.64761`；
- Pose mAP50-95 不低于 `0.49455`；
- paired fused-FP16 latency 至少下降 10%；
- candidate 独立进程推理峰值显存严格低于 baseline。

门槛一旦建立，后续候选就不能通过更换基线、混用 FP32/FP16 或引用不同时间的独立测速来“优化结论”。

## 3. 为什么常见压缩思路接连失败

第一轮实验很像一堂“小模型压缩反直觉”课程。

### FastHead：参数少了，GPU 没快多少

FastHead 把姿态塔缩到 64×1，并把 Box tower depth 降为 1：参数下降 24.64%，但 fused FP16 延迟只下降 6.39%；Pose mAP50-95 从 0.51455 降到 0.37569。

模块剖析显示，FastHead 中 Backbone + Neck 仍占约 71.64% 的时间。Head 参数大幅减少，并不意味着端到端时间会同比下降。对 batch=1 的 nano 模型，kernel launch、访存、通道对齐和算子利用率都可能比 FLOPs 更重要。

### Joint RepC3：一次改太多，特征语义断裂

同时替换 Backbone、Neck 和 Head 的 Joint RepC3 候选，在训练前静态门禁上看起来足够快，但累计 30 epoch 后 Box/Pose 只有 0.48820/0.23798。同形权重迁移、输出 KD 和特征 KD 都不足以恢复被大幅扰动的关键点空间表示。

### 通道剪枝：GFLOPs 降了，延迟甚至更慢

常规结构化剪枝和 BN-scale 稀疏训练产生了多个参数更少、GFLOPs 更低的候选，但真实 paired 延迟大多持平或变慢。最激进的 active-head 剪枝把融合计算量从 13.6 GFLOPs 降到 5.4 GFLOPs，真实延迟仍没有改善，精度则几乎归零。

![FastHead、Joint RepC3、通道剪枝与 P5-Zero 的速度精度决策对比](/images/yolov10-vehicle-pose-compression/03-comparison-experiment-map.webp)

这些失败共同说明：**训练可以尝试恢复精度，但不能把一个已经实测更慢的执行图训练得更快。** 所以后续流程改为先做训练前真实延迟门禁，未达 10% 的候选不再消耗 GPU 预算。

## 4. P5-Zero：只动低分辨率路径

最终成功的 P5-Zero 没有继续缩 Head，也没有破坏 P3/P4 高分辨率定位路径。它以已恢复的 LitePSA + C2f-P4D1 模型为直接父模型，只清空两个 P5 stage 的内部 bottleneck：

- layer 8：backbone P5 的 C2f；
- layer 22：neck P5 的 C2fCIB。

Box、Class、Pose 三个预测塔完整保留。权重迁移器复制 605 个同形张量，并对两处输出投影做 C2f 语义折叠，做到 missing/unexpected key 均为 0。

恢复过程分为两阶段：

1. **全模型同源 KD**：从结构相近的 LitePSA + C2f-P4D1 教师快速恢复。第 2 轮达到 Box/Pose 0.66245/0.49369，Pose 只差门槛 0.00086。
2. **Pose-only 精修**：冻结其余参数，只训练 48 个姿态塔张量。第 1 轮达到 0.66254/0.49592，跨过门槛后立即停止。

![P5-Zero 从原始 baseline、同源教师、折叠迁移到两阶段知识蒸馏的结构与训练路线](/images/yolov10-vehicle-pose-compression/04-framework-p5zero-recovery.webp)

最终结果：

| 指标 | 原始 baseline | P5-Zero | 变化 |
|---|---:|---:|---:|
| paired fused-FP16 latency | 5.94537 ms | 5.06345 ms | **-14.83371%** |
| Box mAP50-95 | 约 0.665 | **0.66254** | 通过 |
| Pose mAP50-95 | 约 0.515 | **0.49592** | 通过 |
| fused 参数量 | 3,992,210 | 3,387,794 | **-15.13989%** |
| 模型驻留 CUDA memory | 57,721,856 B | 54,093,312 B | **-6.28626%** |
| 推理峰值 CUDA memory | 72,237,568 B | 68,609,024 B | **-5.02307%** |

需要诚实说明：Pose 只比发布门槛高 0.00137。它是“通过项目门禁”，不是“精度无损”。换 GPU、PyTorch/CUDA、数据版本或验证实现，都应重新执行三门禁。

## 5. 一个容易被忽略的 Pareto 点：只裁部署分支

YOLOv10 训练时需要 one-to-many 和 one-to-one 双分支，但 `one2one_only=True` 推理并不会执行前者。因此项目还实现了 deployment-only checkpoint，物理删除 one-to-many 的 Box/Class/Pose towers。

它的 dense output 与裁剪前逐元素相同，完整验证精度保持一致；参数和 checkpoint 约下降 27%，推理峰值显存下降 8.99%。但 paired 延迟只改善约 0.19%。

这是一种很有用的工程区分：

- 要**真实加速**，选 P5-Zero；
- 要**缩小模型和显存、保持精度**，选 deployment-only 分支裁剪；
- 要**最低风险和最高精度**，保留原始 baseline。

## 6. 最快的一次“优化”其实不在网络里

训练与验证早期，数据位于 Windows E 盘，通过 WSL 的 `/mnt` 路径读取。把数据迁到 WSL ext4 后，端到端延迟从 60.10 ms 降到 16.63 ms，改善 72.3%，达到 3.61×。

这个数字不应该和模型前向的 5–6 ms 混为一谈，但它提醒我们：在进入复杂的结构搜索前，先排除文件系统、DataLoader、多进程、字体下载、日志平台和环境差异。系统瓶颈可能比网络瓶颈更大，也更便宜。

## 7. 可复用的压缩决策框架

后续类似项目可以复用以下顺序：

```text
提出一个最小结构改动
  → 权重迁移与 finite forward 测试
  → paired fused-FP16 训练前速度门禁
      ├─ 未达 10%：停止
      └─ 达标：短程同源恢复
          → 精度趋势仍可能达线？
              ├─ 否：停止
              └─ 是：选择性精修
                  → 完整精度 + 独立显存 + paired latency 复验
```

其中最重要的三条纪律是：

1. 不用参数量或 GFLOPs 代替真实延迟。
2. 不混用不同 checkpoint、不同精度模式或不同时间的独立测速。
3. 不用“再多训一点”逃避已经缺少可证伪前提的失败路线。

## 8. 数据与复现边界

项目仍有几个明确限制：

- CarFusion 中关键点索引 13 在 62,434 个转换实例里 visibility 全为 0。实现保持 14×3 兼容，但数据只能支持 13 个有效点的准确性结论。
- 左右关键点语义映射未人工确认，水平翻转保持关闭。
- 没有权威车辆 keypoint sigma，uniform sigma OKS 只能作为实验性辅助指标。
- 原始数据、baseline、同源教师和 P5-Zero 权重不进入 Git，精确复现必须取得受控制品并核对 SHA256。
- P5-Zero 只在当前 YOLOv10n checkpoint 上验证；`s/l` 虽已共用通用剪枝代码，但必须分别重建 baseline 与三门禁。

项目代码与复现入口：[Caius-Lu/yolov10-vehicle-pose](https://github.com/Caius-Lu/yolov10-vehicle-pose)。

## 结语

这个项目最后留下的，不只是一个快 14.83% 的模型，而是一套更可靠的实验观：**先建立不可移动的门槛，再让真实硬件和完整验证淘汰直觉。**

对于已经很小的网络，好的压缩不一定来自更激进的剪刀，而可能来自更克制的结构改动、更同源的恢复路径，以及更严格的停止规则。
