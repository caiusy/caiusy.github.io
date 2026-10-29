---
title: "【雷视融合专题 03】从 ES-MEOT 的 MTPMB 到纯视觉 3D 跟踪：目标丢帧后如何找回原 ID"
date: 2026-10-30 10:00:00
updated: 2026-10-30 10:00:00
categories: [雷达相机融合]
tags: [Radar, Camera, MOT, MTPMB, 3D Tracking, ID Recovery]
type: experiment
difficulty: advanced
review_status: published
description: "把 ES-MEOT 的 MTPMB 多目标跟踪思想落到纯视觉直接 XYZ 检测上：用休眠轨迹池、低分检测二次关联和自适应协方差，减少目标丢失与 ID 切换。"
cover: /images/radar-camera/03/pure-vision-recovery-flow.svg
---

> **TL;DR**：纯视觉跟踪的难点通常不在“当前帧有没有框”，而在“目标消失几帧后回来，系统还能不能认出它”。这次我保留 ES-MEOT 论文中 MTPMB 的多假设生命周期，把雷达扩展目标似然换成相机端的目标级 3D 观测，并加入休眠轨迹池和低分检测恢复。779 帧固定留出集上，F1 从 93.645% 提升到 94.153%，误检减少 103，漏检减少 8，相邻帧 ID 切换保持 43 次。它是同序列标注验证，不是独立录像或官方 IDF1/HOTA 指标。

<!-- more -->

这是[雷视融合专题](/radar-camera-fusion/)的第三篇。第一篇梳理了[路侧雷视公开数据](/2026/10/27/radar-camera-01-datasets/)，第二篇解释了[一辆车为什么会变成多个雷达点](/2026/10/27/radar-camera-02-point-vs-eot/)。这一篇换一个方向：如果暂时不把雷达测量放进跟踪器，只让相机图像输出 3D 目标，能不能把原论文的多目标跟踪骨架用起来？

答案是可以借结构，但不能假装输入相同。ES-MEOT 原论文的方法名是 **MTPMB**，它面向雷达反射点、相机关键点和扩展目标似然；下面的实现是**纯视觉点观测适配**，不是原论文融合流程的完整复现。

> **术语说明**：上游 ES-MEOT 仓库的命名并不完全统一：README 的目录项使用 `MTPMB Tracker`，后文又出现 `TPMBM Tracker`，代码文件则叫 `TPMB_RCfusion.py`，函数和参数使用 `TPMB_RCfusion_*` 与 `TPMB-RCfusion`。因此本文用 **MTPMB** 指论文方法，用 **TPMB_RCfusion** 指上游实现命名，用 **Vision-TPMB** 指本项目自己的纯视觉适配。三个名字指代层次不同，不能直接互换。

## 1. 先看问题：为什么目标会“丢”

单帧检测器偶尔漏掉一辆车，或者远处车辆被遮挡几帧，本身很常见。真正麻烦的是后续：

- 漏检后，预测状态继续外推，误差逐渐累积；
- 目标回来时，位置门限已经对不上；
- 原来的 Bernoulli 被裁剪或回收，公共 ID 消失；
- 新检测被当成新目标，系统给了一个不同的 ID；
- 拥挤场景里，两个相邻车辆还可能交换身份。

这和“把检测框画得更准”不是同一个问题。它需要状态生命周期、关联和恢复策略一起工作。

![MTPMB 风格纯视觉跟踪与休眠轨迹恢复的数据流](/images/radar-camera/03/pure-vision-recovery-flow.svg)

这张图里有一个容易被忽略的分支：低分检测不进入出生分支，只进入已有轨迹的恢复关联。这个约束很关键。否则，把检测阈值从 0.25 降到 0.10，虽然可能找回被遮挡车辆，也会制造大量幽灵轨迹。

## 2. 原论文哪些东西值得保留

ES-MEOT 的核心不是“某个网络输出一个框”，而是多目标随机有限集跟踪的几个环节：

| 论文/方法概念 | 本项目的位置 | 本次是否保留 |
|---|---|---|
| PPP 出生强度 | `VisionTPMBTracker._birth` | 保留 |
| Bernoulli 存在概率 | `finish` 中的存在概率更新 | 保留 |
| Murty top-K 全局关联 | `k_best_assignments` | 保留，K=5 |
| 多假设矩匹配 | `moment_match` | 保留 |
| 杂波、漏检和回收 | `update` / `finish` | 保留并增加休眠池 |
| 雷达反射点扩展目标似然 | 原论文 MTPMB | 纯视觉路径不适用 |
| 相机关键点 / MDN 似然 | 原论文 MTPMB | 不作为本路径的输入 |

纯视觉版本的状态仍然是一个 12 维高斯：

```text
x_t = [p_x, p_y, p_z, v, omega, y_dot, delta_theta^T,
       log l, log w, log h]^T
```

位置、速度和地平面 CTRV 负责运动；SO(3) 切空间负责完整朝向；三个对数尺寸负责车辆长宽高。相机图像经过 CenterNet 直接回归到米制 XYZ，随后交给 Vision-TPMB 适配器，而不是先用像素中心和深度反投影再拼一个伪 3D 框。

## 3. 原来的关联为什么不够

最开始的 Vision-TPMB 主要用 XYZ 的 Mahalanobis 距离做门限，姿态、尺寸和 2D 中心只是在 XYZ 门限内做一个有界重排。这个设计很保守，结果是：

- 误检被压下去了；
- 但漏检后的目标不容易重新接上；
- 相邻车辆位置很近时，单靠 XYZ 不足以判断身份。

我们也试过把姿态、尺寸和 UV 放进联合关联。消融结果很诚实：误检只从 787 降到 783，F1 几乎没有变化。原因是检测器的 XYZ 和 UV 头共享图像特征，单靠 UV 并不能提供真正独立的身份证据；在遮挡和边缘截断时，尺寸和姿态也会退化。

所以最后的重点不是继续给 XYZ 加权重，而是补上**轨迹生命周期**。

## 4. 三个改动：休眠、低分二次关联、自适应外推

### 4.1 休眠轨迹池

已确认的轨迹在连续漏检达到阈值后，不再直接进入回收质量，而是进入一个有寿命的休眠池：

- 休眠期间不输出目标；
- 但公共 ID 被保留；
- 位置状态继续按 CTRV 预测；
- 超过 1 秒仍没有恢复，就过期，ID 不复用。

这相当于把“轨迹消失”和“身份永久结束”分开。目标只是暂时看不见，不等于它已经从世界里消失。

### 4.2 低分检测只做恢复，不做出生

检测被分成两层：

- `score >= 0.25`：可以进入正常 Vision-TPMB 出生与关联；
- `0.10 <= score < 0.25`：只能关联到已确认的活动轨迹或休眠轨迹；
- 低于 `0.10`：忽略。

低分检测没有私有出生列，也没有 PPP 质量。因此它即使匹配失败，也不会创建新 ID、消耗出生质量或污染全局假设。这个设计借鉴了 ByteTrack “把低分框留给未匹配轨迹”的思想，但在这里进一步约束为“只能恢复已有身份”。

### 4.3 漏帧后放大不确定性

轨迹漏检越多，位置预测越不可靠。实现中让过程噪声随漏检次数有界增长：

```text
s_miss = min(s_max, 1 + rho * n_miss)
```

它不会无限放大，也不会在每一帧额外叠加一个跳变协方差。目标重新出现时，观测仍然占主导，避免旧状态把新位置拖回去。

## 5. 一个可手算的恢复例子

下面的数字是合成例子，只用来说明门限，不代表某辆车的真实标注。

假设一辆车在 20 m 处，休眠轨迹预测到当前帧为 `(0.2, 0, 20.0)`。第 3 帧来了一个低分检测 `(0.8, 0, 20.4)`：

- 中心距离约 `0.72 m`，小于 2 m；
- 类别都是 `car`；
- 朝向残差约 8°，小于 30°；
- 尺寸对数残差通过门限；
- 低分检测的 score 为 0.14，只给位置更新。

如果同一位置还有另一辆同类车，两个候选的代价差小于 1 nat，系统会拒绝这次恢复。宁可暂时不接，也不把 A 车的 ID 强行贴到 B 车上。恢复需要连续两次通过严格门限，或者代价差达到立即恢复阈值。

这个“拒绝歧义”的规则比单纯追求召回更重要。ID 跟踪里，错误的“找回来”往往比短暂的“没找回”更难排查。

## 6. 实验结果

固定 CenterNet 检测、固定 5 m XYZ 一对一匹配、固定 96 帧选择集和 779 帧留出集。参数只在选择集上选择，留出集只评估一次。

| 方法 | Precision | Recall | F1 | FP | FN | 相邻帧 ID 切换 |
|---|---:|---:|---:|---:|---:|---:|
| Legacy CTRV | 82.62% | 96.93% | 89.20% | 2090 | 315 | 153 |
| Vision-TPMB K=5 | 92.51% | 94.81% | 93.645% | 787 | 532 | 43 |
| confirm-2 K=5 | 93.19% | 94.58% | 93.879% | 708 | 556 | 40 |
| **recovery K=5** | **93.43%** | **94.89%** | **94.153%** | **684** | **524** | **43** |

和 Vision-TPMB K=5 基线相比，恢复版本多找回 8 个真目标，少产生 103 个误检，F1 提升 0.508 个百分点。相邻帧 ID 切换没有下降，但也没有增加。表中的 Vision-TPMB 是本文适配实现的名字，不是论文方法 MTPMB 的替代写法。

另外记录了“轨迹间隙”诊断：在 779 帧中检测到 335 次轨迹间隙，其中 61 次恢复到同一个 ID，32 次恢复了不同 ID，242 次没有匹配上，93 次产生了新的匹配片段。这个指标比只看相邻帧切换更接近“丢帧后找回”的问题，但它仍然不是官方 IDF1/HOTA。

![新版纯视觉跟踪回放：相机、BEV、轨迹和目标 ID 同步显示](/images/radar-camera/03/replay-workbench.jpg)

上图的重点是两种视图共用同一组 3D 状态：左侧相机框、右侧 BEV 轨迹和模板 ID 不是各自独立生成的。页面上仍把原论文结果、纯视觉结果、雷达点云和原始检测分层显示，避免把不同来源的结果混成一个“看起来很好”的合成画面。

## 7. 这次没有解决的问题

这次最明显的提升来自生命周期和低分二次关联，不是外观重识别。当前没有可靠的车辆 Re-ID 特征，因此不能用“看起来像同一辆车”来直接恢复。轨迹间隙里仍有 242 次没有匹配上，说明以下问题还在：

- 长时间遮挡后，CTRV 外推误差超过位置门限；
- 远处目标只有几个像素，3D 尺寸和朝向不稳定；
- 相邻车辆在同一车道排队时，几何约束仍可能歧义；
- 休眠池的 1 秒寿命对高速驶离车辆可能太短；
- 当前指标是同一录像序列上的时间留出，不是跨路口、跨天气的独立测试。

所以“尽量一个都不丢”是一个目标，不是这次实验已经完成的结论。更稳妥的说法是：在不增加相邻帧 ID 切换的前提下，召回和误检同时得到小幅改善。

## 8. 复现边界

实验在远程 WSL 的 RTX 3090 Ti 上运行。原始视频、雷达文件、检测缓存和完整标注留在远程机器；仓库只保留代码、配置、精选截图和派生指标。

```bash
PYTHONPATH=src python3 tools/infer_direct3d.py \
  --checkpoint outputs/vision3d/centernet_xyz40/best.pt \
  --video data/interim/official/output.avi \
  --output outputs/vision3d/centernet_xyz40/predictions_recovery.jsonl \
  --frames 4532 --score .25 --tracker tpmb \
  --tracker-config configs/vision_tpmb_recovery.json \
  --ground configs/esmeot_camera_ground.json
```

评测命令和完整配置见仓库的 `experiments/2026-10-30-vision-recovery/`。如果后续要做真正的“零丢失”验证，下一步应加入跨录像评测、IDF1/HOTA、不同距离分桶和真实遮挡标注，而不是继续在同一段视频上调门限。

## 参考资料

- Deng 等，*3D Multiple Extended Object Tracking by Fusing Roadside Radar and Camera Sensors*，[DOI](https://doi.org/10.1109/JSEN.2024.3493952)。
- Zhang 等，*ByteTrack: Multi-Object Tracking by Associating Every Detection Box*，[ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf)。
- Cao 等，*Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking*，[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.pdf)。

**上一篇**：[一辆车为什么会有多个雷达点？从点目标跟踪到 EOT](/2026/10/27/radar-camera-02-point-vs-eot/)
