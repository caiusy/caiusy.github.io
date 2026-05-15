---
title: Learning Notes
date: 2026-05-16
updated: 2026-05-16
type: topic
layout: page
description: 长期复习用的笔记总入口 — 概念笔记、论文笔记、工程笔记、实验记录、复习索引。
---

## Learning Notes

这里不是正式文章区，而是我的**长期学习笔记区**。笔记用于复习、查漏补缺、串联知识，不追求每篇都像正式博客一样完整，但每篇都要有清晰结构。

---

## 1. Concept Notes / 概念笔记

记录基础概念。每篇结构：一句话结论 → 直觉 → 数学 → 工程用法 → 易错点 → 复习问题。

**典型主题**
- 数学：线性代数、概率、矩阵分解
- 算法：损失函数、优化器、正则化
- 深度学习：Attention、BatchNorm、Dropout
- 相机几何：Homography、Vanishing Point、相机内外参
- 雷达基础：FMCW、点迹、跟踪

**入口**：[/categories/学习笔记/概念笔记/](/categories/学习笔记/概念笔记/)

**新建**：`hexo new learning-note "概念名称"`

---

## 2. Paper Notes / 论文笔记

记录每一篇精读过的论文。每篇结构：一句话总结 → 解决什么问题 → 方法总览 → 核心模块 → 公式 → 实验 → 我的理解 → 可迁移到工程的点 → 复习问题。

**典型论文**
- BEV：LSS、BEVDet、BEVFormer、PETR、GaussianLSS
- 多模态：Transformer、CLIP、BLIP、LLaVA
- 微调：LoRA、QLoRA
- 经典基础：ResNet、ViT、DETR

**入口**：[/categories/学习笔记/论文笔记/](/categories/学习笔记/论文笔记/)

**新建**：`hexo new paper-note "论文标题"`

---

## 3. Engineering Notes / 工程笔记

记录踩过的工程坑和环境问题。每篇结构：现象 → 环境 → 错误日志 → 原因 → 解决 → 验证 → 避坑总结。

**典型主题**
- 环境：Docker / CUDA / PyTorch / 多版本共存
- 框架：mmcv / detectron2 / torch.distributed
- 工具：Git / VSCode / Jupyter / tmux
- 调试：显存、性能、segfault

**入口**：[/categories/学习笔记/工程笔记/](/categories/学习笔记/工程笔记/)

**新建**：`hexo new engineering-note "问题描述"`

---

## 4. Experiment Notes / 实验记录

记录每次跑实验的目标、配置、结果。每篇结构：目标 → 配置 → 数据 → loss → 指标 → 曲线分析 → 结论 → 下一步。

**典型实验**
- GaussianLSS 复现实验
- LoRA rank 消融
- BEV 网格分辨率对比

**入口**：[/categories/实验记录/](/categories/实验记录/)

**新建**：`hexo new experiment-log "实验编号-描述"`

---

## 5. Review Index / 复习索引

按方向、状态、难度、笔记类型组织所有笔记，方便快速复习。

**入口**：[Review Index](/review/)

---

## 笔记 front-matter 字段约定

```yaml
type: note               # note / experiment / project / deep-dive / archive
note_type: concept       # concept / paper / engineering / algorithm
difficulty: beginner     # beginner / intermediate / advanced
review_status: new       # new / reviewing / mastered / need-review / archived
prerequisites: [...]     # 前置知识引用
related: [...]           # 相关笔记的链接
next_review: 2026-06-01  # 下次复习日期
```

每篇笔记正文都至少包含：

- 一句话结论
- 核心概念
- 易错点
- 复习问题
- 我的理解
- 相关笔记
