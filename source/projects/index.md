---
title: Projects
date: 2026-05-16
updated: 2026-05-16
type: projects
layout: page
description: Caius Lu 的代表项目 — 雷达-相机标定融合可视化系统、路侧 BEV 感知专题、GaussianLSS 复现、多模态大模型学习路线。
---

## Projects

代表性工程与研究项目。每个项目都对应一组博客文章和长期更新的笔记。

> 部分链接尚未公开，状态保持为 TODO。

---

## 1. Radar-Camera Calibration & Fusion Visualization System

**一句话**：雷达-相机同步加载、点对匹配、单应性矩阵求解、标定结果导出与可视化验证的工程系统。

**背景**
路侧场景下雷达与相机各自采集独立数据流，缺少统一坐标系下的快速验证工具。常见痛点包括：标定结果难以一眼看出错没错、雷达点投影到图像上的偏差无法量化、雷达-相机时间不同步导致融合失效。

**解决的问题**
- 雷达 / 相机同步数据加载与回放
- 雷达点 ↔ 图像点对匹配
- 单应性矩阵求解（地面平面假设）
- 标定结果 JSON 序列化导出
- 投影、回投、误差可视化验证

**技术栈**
PyQt5 / OpenCV / NumPy / JSON / Matplotlib

**系统架构**
```
data loader  →  point matcher  →  homography solver  →  exporter
                                          ↓
                              visualization & error analysis
```

**输入输出**
- 输入：同步雷达 JSON、对应图像、相机内参
- 输出：单应性矩阵 H、外参 R/t、投影误差可视化

**当前状态**：内部使用中，工程文章持续更新

**相关文章**
- TODO: 写《雷达-相机标定的工程化思考》
- TODO: 写《PyQt 标定工具的状态机设计》

**链接**
- GitHub：TODO
- Demo：TODO

---

## 2. Roadside BEV Perception Notes

**一句话**：围绕路侧多相机、BEV 表征、地面单应性、车辆轨迹、感知评测建立的专题知识库。

**为什么做这个**
路侧 BEV 感知和车端 BEV 在外参约束、相机高度、视角、ROI 划分上差异很大。车端方法（LSS / BEVDet / PETR）不能直接迁移，需要按路侧场景重新组织一套自己的认知体系。

**涉及技术**
相机几何 / Homography / IPM / LSS / BEVDet / GaussianLSS / 多相机 BEV 融合 / 评测指标

**推荐阅读顺序**
1. 相机几何基础（内参、外参、单应性矩阵）
2. 图像到 BEV 的几何映射（IPM、地面假设）
3. 车端 BEV 主流方法（LSS / BEVDet / PETR）
4. 路侧场景下的特殊问题（高视角、远距离、稀疏目标）
5. BEV 空间下的雷视融合
6. 有效检测区域评估

**入口**：[Roadside BEV](/roadside-bev/)

---

## 3. GaussianLSS Reproduction & Experiments

**一句话**：围绕 GaussianLSS 论文的复现、实验、loss 配置、IoU 曲线分析与工程改造。

**论文背景**
GaussianLSS 在 LSS 框架基础上引入高斯深度分布，使深度预测更稳健。本项目记录复现过程、超参选择、训练曲线与改造方向。

**复现目标**
- 在标准基准上复现论文报告的 IoU 指标
- 拆解每个模块的贡献（消融）
- 探索路侧场景下的迁移可能

**实验内容**
- depth bin 配置实验
- loss 权重消融
- BEV 网格分辨率对比

**当前进展**：实验记录持续更新，[Experiment Logs](/categories/实验记录/)

**相关文章**
- TODO: 写《GaussianLSS 论文精读》
- TODO: 写《GaussianLSS 复现踩坑记》

**链接**
- GitHub：TODO
- Paper：TODO

---

## 4. Multimodal LLM Reading Map

**一句话**：从 Transformer、BERT、GPT 到 CLIP、LLaVA、MoE、LoRA、QLoRA 的大模型学习路线图。

**这是什么**
不是一份阅读清单，而是一张**有顺序、有依赖、有复习问题**的学习地图。每个节点都标注：核心问题、前置知识、读完能理解什么、推荐文章、复习问题。

**适合读者**
- 正在系统学习多模态大模型的工程师
- 想从 Transformer 出发把基础打牢的研究新人
- 需要一套可复习、可串联的知识结构的人

**入口**：[LLM Reading Map](/llm-reading-map/)

---

## 项目状态总览

| 项目 | 类型 | 状态 | 公开度 |
|---|---|---|---|
| Radar-Camera Calibration Tool | 工程 | 内部使用中 | TODO |
| Roadside BEV Notes | 知识库 | 持续更新 | 公开 |
| GaussianLSS Reproduction | 复现 | 实验中 | TODO |
| Multimodal LLM Reading Map | 学习路线 | 持续更新 | 公开 |
