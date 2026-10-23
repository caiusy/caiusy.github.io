---
title: 复习索引
date: 2026-05-16
updated: 2026-10-23
type: topic
layout: page
menu_id: review
description: 从相机几何、大模型到算法基础，沿着已发布的文章，用具体问题检验理解。
---

复习时先尝试回答问题，再打开文章核对公式、图示和代码。下面的入口都指向已经发布的内容。

## 视觉与空间几何

| 文章 | 读完后试着回答 |
|---|---|
| [相机几何完整指南](/2026/01/19/camera-geometry-complete-guide/) | 内参和外参各自完成哪一步坐标变换？ |
| [从图像到俯视图](/2026/01/19/image-to-bird-eye-view/) | 单应性为什么需要平面假设？ |
| [BEV 与路侧感知专题](/roadside-bev/) | 透视变换、BEV 表征和三维感知的边界在哪里？ |
| [雷达与相机融合专题](/radar-camera-fusion/) | 坐标对齐和时间同步分别会产生什么误差？ |

## 大模型：从表示到高效推理

| 文章 | 读完后试着回答 |
|---|---|
| [Transformer 深入解析](/2026/02/02/Transformer-DeepDive/) | Q、K、V 的形状如何随多头划分变化？ |
| [LLaVA 终极指南](/2026/03/04/LLaVA-Ultimate-Guide/) | 视觉特征如何进入语言模型的 token 序列？ |
| [结合 MiniMind 理解 LoRA 与 MoE](/2026/08/30/结合-MiniMind-源码彻底搞懂-LoRA-与-MoE/) | LoRA 和 MoE 分别节省了什么，增加了什么？ |
| [从 KV 缓存到 MLA](/2026/09/13/kv-cache-rope-mla-explained/) | 为什么历史 Q 不需要缓存？MLA 缓存里究竟是什么？ |
| [DeepSeek DSA 与 Lightning Indexer](/2026/10/23/deepseek-dsa-lightning-indexer/) | 当前 Gate 为什么决定历史位置排名？Top-K 后读哪套缓存？ |

**动手验证**：[打开 DSA 交互实验室](/dsa-lab/)，改变一个 Gate，预测 Top-K 的变化，再检查结果。

需要连续学习时，从[大模型阅读路线](/llm-reading-map/)进入。

## 算法：从状态到实现

| 文章 | 读完后试着回答 |
|---|---|
| [动态规划：从递归到状态转移](/2026/07/13/dynamic-programming-from-recursion-to-state-transition/) | 状态包含哪些信息？为什么这些信息足够？ |
| [图论与搜索](/2026/02/23/graph-theory-and-search/) | BFS 和 DFS 在状态访问顺序上有什么差异？ |
| [单调栈指南](/2026/02/19/monotonic-stack-guide/) | 栈里保持什么单调性？弹出时确定了哪个答案？ |
| [链表指针完整指南](/2026/02/22/linkedlist-pointer-complete-guide/) | 修改 next 之前，需要保留哪些后续节点？ |

## 一次 30 分钟的复习

1. **5 分钟回忆**：选一个问题，不看原文写出自己的解释。
2. **10 分钟核对**：对照文章里的例子，找出遗漏的前提和推导。
3. **10 分钟动手**：复算一个数值、改一个参数，或写最小实现。
4. **5 分钟记录**：写下一条仍然不确定的问题，下次从这里开始。

更多内容：[学习笔记](/notes/) · [工程实践](/engineering/) · [完整归档](/archives/)
