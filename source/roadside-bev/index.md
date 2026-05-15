---
title: Roadside BEV Perception
date: 2026-05-16
updated: 2026-05-16
type: topic
layout: page
description: 路侧 BEV 感知专题 — 相机几何、图像到 BEV、BEV 感知方法、雷视融合、有效检测区域评估、工程系统实践。
---

## Roadside BEV Perception

围绕**路侧场景**的 BEV 感知专题。从相机几何基础到工程系统实践，按从下到上的顺序组织。

---

## 1. 相机几何基础

理解一切 BEV 方法之前的数学语言。

**核心概念**
- 相机内参（focal length, principal point, skew）
- 相机外参（rotation, translation）
- 世界坐标系 / 相机坐标系 / 像素坐标系
- 投影矩阵 P = K [R | t]
- 消失点（vanishing point）
- 单应性矩阵（homography）

**相关文章**
- [相机几何完整推导](/categories/计算机视觉/) (按现有文章映射)
- TODO: 单应性矩阵的 8 个自由度
- TODO: 路侧相机如何选高度和俯仰角

---

## 2. 图像到 BEV

如何把一张前视图像“拍平”成俯视图。

**核心概念**
- 地面假设（flat ground assumption）
- Homography（图像平面 ↔ 地面平面）
- IPM (Inverse Perspective Mapping)
- 相机高度、pitch、yaw 对 BEV 的影响
- 图像点 → 地面点的投影公式

**相关文章**
- TODO: IPM 的代码实现与误差边界
- TODO: 路侧相机俯仰角变化对 BEV 的影响

---

## 3. BEV 感知方法

主流方法的脉络。

**核心方法**
- **LSS** (Lift-Splat-Shoot) — 显式深度预测 + voxel pooling
- **BEVDet** — LSS 框架下的检测扩展
- **Fast-BEV** — 加速 BEV 推理
- **PETR** — query-based BEV
- **DETR3D** — 3D query 直投到 2D 图像
- **GaussianLSS** — 高斯深度分布替代 categorical bin

**相关文章**
- TODO: LSS 论文精读
- TODO: BEVDet vs LSS 对比
- TODO: GaussianLSS 论文精读

---

## 4. 雷达-相机融合

详细专题见 [Radar-Camera Fusion](/radar-camera-fusion/)。

**本节关注 BEV 视角的融合**
- 时间同步（硬件触发 / PTP / 软同步）
- 坐标变换（雷达坐标系 → BEV 网格）
- 雷达点投影到图像
- 图像目标 ↔ 雷达目标关联
- Early Fusion vs Late Fusion
- BEV 空间内的 feature 级融合

**相关文章**
- 见 [Radar-Camera Fusion](/radar-camera-fusion/)

---

## 5. 有效检测区域评估

路侧感知的特殊问题：哪个 BEV 区域结果可信？

**核心问题**
- 雷达点迹密度 → 哪里能稳定检测
- BEV 网格统计 → 给定网格的命中率
- 有效区域多边形（valid polygon）
- 最远检测距离 → 角度 vs 距离的权衡
- 盲区分析（雷达盲区 / 相机盲区 / 重叠盲区）

**相关文章**
- TODO: 有效检测区域的多边形生成方法
- TODO: 雷达点云密度热力图

---

## 6. 工程系统实践

把研究方法变成可用工具。

**关键模块**
- 标定工具（PyQt + OpenCV）
- 可视化工具（matplotlib / Qt 自绘）
- JSON 数据格式约定（雷达包 / 标定结果 / 评测结果）
- 同步数据加载（按时间戳对齐）
- 评测流程（指标定义、回归测试）

**相关文章**
- TODO: 雷达-相机标定工具的状态机
- TODO: JSON 格式设计与 schema 演进
- TODO: 标定回归测试

---

## 推荐阅读顺序

如果你是第一次接触路侧 BEV，建议按这个顺序：

1. 第 1 节 → 把相机几何这门“数学语言”打牢
2. 第 2 节 → 理解图像到 BEV 的最简单情形（IPM）
3. 第 3 节 → 看主流方法是怎么把 IPM 替换成端到端学习的
4. 第 5 节 → 评测先行，知道怎么衡量好坏再做改进
5. 第 4 节 → 在已有 BEV 基础上加雷达
6. 第 6 节 → 工具链上手

---

## 状态

这是一个**长期更新**的专题。每写一篇相关文章，都会回到这里挂钩。
