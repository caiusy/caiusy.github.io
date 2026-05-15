---
title: Engineering Practice
date: 2026-05-16
updated: 2026-05-16
type: topic
layout: page
description: 工程实践专题 — 标定工具、PyQt 开发、OpenCV 可视化、Docker/CUDA/PyTorch 环境、Git 工作流、实验管理与复现、远程开发。
---

## Engineering Practice

研究方法落地成可用工具的所有"脏活"。这里不写论文，只写工程。

---

## 1. Radar-Camera Calibration Tool

- 状态机驱动的标定 UI
- 点对采集与可视化
- 单应性矩阵求解（DLT + RANSAC）
- 标定结果 JSON 导出与加载

**相关文章**：TODO

---

## 2. PyQt / Qt 工具开发

- 信号槽 vs 命令模式
- 自绘控件（QGraphicsView / QPainter）
- 多线程数据加载（QThread vs QtConcurrent）
- 状态机模型

**相关文章**：TODO

---

## 3. OpenCV 可视化

- 投影、回投、误差热力图
- BEV 双视图同步
- 视频回放 + 时间轴拖动

**相关文章**：TODO

---

## 4. Docker / CUDA / PyTorch 环境

- 多版本 CUDA 共存
- PyTorch + CUDA 兼容矩阵
- Dockerfile 模板（训练 / 推理）
- 显存监控与调优

**相关文章**：TODO

---

## 5. Git / GitHub 工作流

- 分支模型（feature / refactor / hotfix）
- commit message 规范
- rebase vs merge
- GitHub Actions 自动部署

**相关文章**：TODO

---

## 6. 实验管理与复现

- TensorBoard / wandb 选型
- config-driven 训练框架
- 实验编号与 git commit 绑定
- 实验日志模板

**相关文章**：见 [Experiment Logs](/categories/实验记录/)

---

## 7. 远程开发

- VSCode Remote SSH
- tmux + 远程长任务
- rsync / sshfs 数据同步
- Jupyter on remote

**相关文章**：TODO

---

每个模块的相关文章会随着博客更新逐步挂上来。
