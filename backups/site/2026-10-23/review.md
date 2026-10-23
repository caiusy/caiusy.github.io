---
title: Review Index
date: 2026-05-16
updated: 2026-05-16
type: topic
layout: page
description: 复习索引 — 按方向、状态、难度、笔记类型快速找到要复习的内容。
---

## Review Index

按多个维度索引博客中的知识点，帮助我快速复习。

---

## 1. 按方向复习

### BEV / 相机几何

- 相机内参 — TODO
- 相机外参 — TODO
- 单应性矩阵 — TODO
- Vanishing Point — TODO
- IPM (Inverse Perspective Mapping) — TODO
- LSS — TODO
- BEVDet — TODO
- GaussianLSS — TODO

📁 全部入口：[/categories/BEV感知/](/categories/BEV感知/)

### 雷达-相机融合

- 雷达坐标系 — TODO
- 图像坐标系 — TODO
- 雷达到图像投影 — TODO
- 点框匹配 — TODO
- BEV 空间融合 — TODO
- 有效检测区域 — TODO

📁 全部入口：[/categories/雷达相机融合/](/categories/雷达相机融合/)

### 多模态大模型

- Transformer — TODO
- Attention 机制 — TODO
- BERT — TODO
- GPT — TODO
- CLIP — [LLaVA 终极指南](/) (现有文章) · TODO 重写
- LLaVA — TODO
- LoRA — TODO
- QLoRA — TODO
- MoE — TODO

📁 全部入口：[/categories/多模态大模型/](/categories/多模态大模型/)

### 工程实践

- Docker — TODO
- CUDA — TODO
- PyTorch — TODO
- Git — TODO
- PyQt — TODO
- OpenCV — TODO
- 实验管理 — TODO

📁 全部入口：[/categories/工程实践/](/categories/工程实践/)

### 算法基础

- 动态规划 — 见 [/categories/算法基础/](/categories/算法基础/)
- 图论 / BFS / DFS — TODO
- 链表 — TODO
- 双指针 — TODO
- 单调栈 — TODO

---

## 2. 按复习状态复习

| 状态 | 含义 | 链接 |
|---|---|---|
| `new` | 新笔记，尚未复习 | [/tags/new/](/tags/new/) |
| `reviewing` | 正在复习 | [/tags/reviewing/](/tags/reviewing/) |
| `mastered` | 已经掌握 | [/tags/mastered/](/tags/mastered/) |
| `need-review` | 需要重新复习 | [/tags/need-review/](/tags/need-review/) |
| `archived` | 归档 | [/tags/archived/](/tags/archived/) |

> **使用说明**：在每篇 `note` 类型文章的 front-matter 加上 `tags: [reviewing]`（或对应状态），就会自动出现在上面对应链接里。Stellar 主题不支持 front-matter 字段直接做筛选 facet，所以暂时用 tag 实现。

---

## 3. 按难度复习

| 难度 | 含义 |
|---|---|
| `beginner` | 入门，第一次接触此话题就能读 |
| `intermediate` | 有基础后再读会更顺 |
| `advanced` | 需要相关研究背景 |

入口：[/tags/beginner/](/tags/beginner/) · [/tags/intermediate/](/tags/intermediate/) · [/tags/advanced/](/tags/advanced/)

---

## 4. 按笔记类型复习

| 类型 | 用途 |
|---|---|
| `concept` | 概念笔记 |
| `paper` | 论文精读 |
| `engineering` | 工程踩坑 |
| `experiment` | 实验记录 |
| `algorithm` | 算法模板 |
| `project` | 项目文档 |

对应分类入口：

- 概念：[/categories/学习笔记/概念笔记/](/categories/学习笔记/概念笔记/)
- 论文：[/categories/学习笔记/论文笔记/](/categories/学习笔记/论文笔记/)
- 工程：[/categories/学习笔记/工程笔记/](/categories/学习笔记/工程笔记/)
- 实验：[/categories/实验记录/](/categories/实验记录/)
- 算法：[/categories/算法基础/](/categories/算法基础/)
- 项目：[/projects/](/projects/)

---

## 5. 快速复习清单（每周）

> 每周日花 30 分钟，过一遍这五道题：

1. 这周新写的笔记里，最重要的一条结论是什么？
2. 上周标记 `need-review` 的内容，现在能凭印象回答吗？
3. 这周遇到的某个工程坑，做成了一篇 engineering note 吗？
4. 上次读的论文，能用 3 分钟讲清楚核心方法吗？
5. 这周的实验，能用一张图说明结论吗？

---

> **维护原则**：本页随博客增长持续更新。每篇新增 `type: note` 的文章都应在第 1 节挂一个链接。
