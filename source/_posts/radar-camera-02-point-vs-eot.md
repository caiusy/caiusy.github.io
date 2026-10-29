---
title: "【雷视融合专题 02】一辆车为什么会有多个雷达点？从点目标跟踪到 EOT"
date: 2026-10-27 09:10:00
updated: 2026-10-27 09:10:00
categories: [雷达相机融合]
tags: [Radar, EOT, MOT, Extended Object Tracking]
type: deep-dive
difficulty: intermediate
review_status: published
description: "用排队车辆的反射点解释点目标、扩展目标、单目标 EOT 与多目标 MEOT 的结构差异。"
cover: /images/radar-camera/02/point-vs-extended.svg
mathjax: true
---

> **TL;DR**：点目标跟踪假设“一帧一个目标最多给一个测量”；EOT 允许一辆车产生多个空间分散的反射点，并联合估计运动与形状。EOT 解决的不是简单“多加几个点”，而是测量属于哪辆车、反射点为何偏离几何中心、形状如何随视角可见的问题。

<!-- more -->

上一篇先区分了[雷达数据层级](/2026/10/27/radar-camera-01-datasets/)。现在回答最基础的问题：为什么一个目标会变成一簇点？

## 1. 点目标与扩展目标

![点目标与扩展车辆的测量差异](/images/radar-camera/02/point-vs-extended.svg)

点目标常用状态是：

{% raw %}
$$
\mathbf{x}_{\text{point}}=[p_x,p_y,v_x,v_y]^T
$$
{% endraw %}

EOT 的教学状态可以写成：

{% raw %}
$$
\mathbf{x}_{\text{EOT}}=[p_x,p_y,v_x,v_y,\theta,l,w]^T
$$
{% endraw %}

这里的 $p_x,p_y$ 是参考位置，$v_x,v_y$ 是速度，$\theta$ 是朝向，$l,w$ 是长宽。**这只是教学简化式，不是某篇论文的完整状态定义。** 三维方法还会增加高度、姿态、角速度以及形状节点。

毫米波回波常来自保险杠、车轮、车角、金属边缘等强散射部位。镜面反射、遮挡和多径会让点迹分布不均；测量均值因此不一定是车辆几何中心。

## 2. 两辆排队车为什么难分

![两辆排队车辆的反射点混合与误分区](/images/radar-camera/02/queue-mixing.svg)

设前车生成点集 $Z_A$，后车生成 $Z_B$，雷达只看到合并集合：

{% raw %}
$$
Z_k=Z_{A,k}\cup Z_{B,k}\cup C_k
$$
{% endraw %}

$C_k$ 是杂波。近距离排队时，两簇在空间上可能连接，DBSCAN 会把它们合成一簇；如果只看最近邻，前车尾部点也可能被分给后车。

### 一个可手算的混合测量例子（演示数据）

假设目标 A、B 的预测中心分别为 10 m、13 m，三个雷达点在 10.4、11.6、12.7 m。用一维高斯教学模型，标准差取 1 m：

{% raw %}
$$
p(z\mid A)\propto e^{-(z-10)^2/2},\qquad p(z\mid B)\propto e^{-(z-13)^2/2}
$$
{% endraw %}

点 11.6 m 到 A、B 的距离分别 1.6 m 和 1.4 m，单帧硬分配几乎没有把握；轨迹速度、视觉位置与形状约束才有价值。

## 3. EOT 到底多估计了什么

![运动中心、几何中心与雷达反射中心的偏差](/images/radar-camera/02/center-bias.svg)

典型 EOT 包含三部分：

1. **运动状态**：位置、速度、朝向、角速度。
2. **扩展状态**：椭圆、随机矩阵、星凸轮廓或弹性骨架。
3. **测量生成模型**：一个目标一帧能产生多少测量，这些测量在形状上如何分布。

单目标 EOT 已知当前主要跟一辆车，重点是形状与运动；MEOT 还必须管理目标出生/消失、点簇分区、轨迹身份和数据关联。后者不是在 EOT 外面简单套一个 ID。

## 4. 指标别混用

| 指标 | 回答的问题 | 方向 | 单位 |
|---|---|---:|---|
| 中心误差 | 估计中心离 GT 多远 | ↓ | m |
| 速度 RMSE | 速度估计误差 | ↓ | m/s |
| 投影 IoU | 形状投影重叠多少 | ↑ | 0–1 |
| MOTA / IDF1 | 多目标检测和身份整体如何 | ↑ | 比例 |
| IDSW | 身份切换多少次 | ↓ | 次 |

IoU 高不保证 ID 连续；MOTA 高也不意味着几何形状准确。第二阶段文章会分别解释这些指标的实验上下文。

## 5. 厂商 Track 还需要 EOT 吗

如果厂商已经稳定输出位置、速度、朝向、尺寸与 ID，业务目标只是“与视觉轨迹融合”，通常不必把目标级 Track 拆回点云再做完整 EOT。更实用的是：

- 保留 Radar Track ID 作为一阶连续性约束；
- 对位置定义做校准：雷达参考点、视觉框底点、车辆几何中心并不相同；
- 用速度、朝向、类别和时间连续性做跨模态关联；
- 在拥堵、分裂/合并与长遮挡时允许 ID 可信度下降。

若需要车辆轮廓、姿态或点级可见性，且能拿到原始点云，EOT 才重新成为核心。

## 6. 论文结论与工程建议的边界

“一目标多测量”与“反射点偏离几何中心”是 EOT 的建模事实；“保留厂商 ID 作为一阶约束”是本文面向目标级接口的工程建议，不是 ES-MEOT 或 ES-EOT 的原论文步骤。

## 参考资料

- Granström 等，*Extended Object Tracking: Introduction, Overview and Applications*，综述入口：[DOI](https://doi.org/10.1007/s11263-016-0932-6)。
- Deng 等，*3D Extended Object Tracking by Fusing Roadside Sparse Radar Point Clouds and Pixel Keypoints*：[arXiv:2404.17903](https://arxiv.org/abs/2404.17903)。

**下一篇**：[从 ES-MEOT 的 MTPMB 到纯视觉 3D 跟踪：目标丢帧后如何找回原 ID](/2026/10/30/radar-camera-03-pure-vision-recovery/)
