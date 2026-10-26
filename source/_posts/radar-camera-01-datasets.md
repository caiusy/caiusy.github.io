---
title: "【雷视融合专题 01】路侧雷视公开数据集：从 RAD、点云到稳定 Radar Track"
date: 2026-10-27 09:00:00
updated: 2026-10-27 09:00:00
categories:
  - 雷达相机融合
tags: [雷视融合, Dataset, Radar, Camera, MOT]
type: deep-dive
difficulty: intermediate
review_status: published
description: "先辨清 RAD、雷达点云、目标级 Track 与跨模态真值，再按任务选择 8 个路侧雷视数据集。"
cover: /images/radar-camera/01/data-layers.svg
mathjax: true
---

> **TL;DR**：数据集“同时有雷达和相机”不等于能训练雷视多目标跟踪。先问雷达交付的是原始 RAD、点云、检测还是稳定 Track，再问是否同步、是否标定、ID 是传感器内 ID 还是跨模态真值。本文只报告官方页面可核实字段；没有确认的地方直接写“未核实”。

<!-- more -->

这是[雷视融合专题](/radar-camera-fusion/)的第一篇。后续文章依次讨论 EOT、ES-MEOT、ES-EOT 与在线标定。

## 1. 先把“雷达数据”拆成五层

![雷达从原始回波到跨模态真值的数据层级](/images/radar-camera/01/data-layers.svg)

| 层级 | 常见字段 | 能做什么 | 不能默认得到什么 |
|---|---|---|---|
| ADC / Radar Cube | 天线、chirp、sample | 自研信号处理 | 目标位置、ID |
| RAD / RA 热力图 | range、azimuth、Doppler、能量 | 雷达检测网络 | 稳定目标 ID |
| 点云 / detection | x、y、z、Doppler、RCS | 聚类、EOT、点级融合 | 轨迹连续性 |
| 厂商 Track | objectId、位置、速度、朝向、尺寸 | 轨迹级融合 | 跨相机同一 ID |
| 跨模态 GT | 统一 ID、框、轨迹、传感器对应 | 训练和严格评测 | 数据许可自动放宽 |

最容易混淆的是最后两层：`radar objectId=17` 通常只表示雷达跟踪器内部的一条轨迹，**不等于**相机标注里的 `track_id=17`。IPC 2025 的官方页面甚至明确说明 `ObjectId` 在不同传感器和不同会话之间不一致。

### 一个可手算的同步例子（教学简化式）

相机帧时间为 10.040 s，最近两帧雷达 Track 在 10.000 s 与 10.100 s，横向位置分别为 4.0 m 与 4.6 m。线性插值为：

{% raw %}
$$
x(10.040)=4.0+\frac{10.040-10.000}{10.100-10.000}(4.6-4.0)=4.24\ \text{m}
$$
{% endraw %}

这只是时间对齐，不是跨模态关联。即使时间戳一致，也仍要解决坐标映射、检测延迟与 ID 对应。

## 2. 八个候选数据集怎么选

![同步视频、点云与 Track 的时间轴区别](/images/radar-camera/01/timeline.svg)

| 数据集 | 场景 | 雷达形式 | 相机/同步 | 标定与 GT | 大小/许可（官方页） | 适合任务 |
|---|---|---|---|---|---|---|
| ES_MEOT | 路侧 | `radarPcl.npy` + `radarTrack.npy` | `output.avi` | `annotation.npy`、config；跨模态 ID 语义需读代码确认 | 视频 2.35 GB、点云 389 MB、Track 59.3 MB、标注 5.96 MB；CC BY 4.0 | ES-MEOT 数据结构、点云/Track 对照 |
| FusionCalib v3 | 路侧 | 4D 雷达 Track | 单目视频 | LiDAR、底点、像素轨迹、3D 框、标定结果 | 官方条目列出字段；CC BY 4.0 | 单节点标定、底点映射 |
| V2X-Radar-I | 路侧子集 | 4D 雷达点云 | 多视角图像 | calib、3D label | 全集约 97.6 GB；CC BY-NC-ND 4.0 | 3D 检测、路侧多模态感知 |
| ATCLL | 智慧灯杆/路口 | 雷达检测 | 视频检测 | CPM；跨模态统一真值未核实 | 640.2 MB；许可未在当前页面明确显示 | 协同感知消息与长时交通统计 |
| IPC 2025 | 路侧路口 | 厂商目标 Track CSV | 相机 bbox CSV | 坐标各自独立；部分会话有 GPS | 93 + 19 + 26.2 MB；许可链接需单独阅读 | 厂商 Track 融合、时间对齐 |
| Urban mmWave-Camera | 路侧城市道路 | JSON 点云 | 8393 帧图像 | XML/YOLO 检测标注；轨迹 ID 未核实 | 4.7 GB；许可未核实 | 检测/分类，不宜直接宣称 MOT GT |
| VIDETEC-2 | 路口基础设施 | RD map + 点检测 | 8 路相机 | 共享坐标/时间；测试 VRU 有 RTK，雷达无人工标签 | 48.0 GB；许可未核实 | VRU、多传感器同步、RD/点级研究 |
| DRaT | 高速汇入段 | 商用雷达目标 Track | 无人机 4K 真值 | 8 Hz 一一时间对齐；雷达/无人机 ID 需关联 | 90.9 MB；版权声明，许可文件随数据 | 厂商 Track 对无人机轨迹评测 |

> 表内“未核实”不是缺点标签，而是发布时的证据边界。数据页后来更新时也应重新核查。

## 3. 任务兼容矩阵

![任务与数据层级的兼容矩阵](/images/radar-camera/01/task-matrix.svg)

| 任务 | 最低必要数据 | 优先选择 | 主要缺口 |
|---|---|---|---|
| 原始雷达检测网络 | ADC/RAD + 标签 | VIDETEC-2 | 标签映射要自建 |
| 点级 EOT/MEOT | 雷达点云 + 同步关键点 | ES_MEOT、V2X-Radar-I | 是否提供连续身份真值 |
| 厂商 Track 与视觉融合 | Radar Track + 图像检测 + 时间戳 | IPC、FusionCalib、DRaT | 跨传感器 ID 通常不存在 |
| 在线标定 | 同步目标对 + 坐标/标定参照 | FusionCalib、VIDETEC-2 | 目标配对质量 |
| 轨迹评测 | 连续 GT ID + 时间同步 | DRaT（雷达评测） | 不等于雷视融合 GT |

## 4. 下载前的四步核查

1. 先读许可：尤其区分 CC BY、非商用与禁止演绎。
2. 只下载一个小文件验证 schema，不把“页面列出”写成“我已跑通”。
3. 检查时间单位、时区、帧率和丢帧；`timestamp` 同名不保证同一时钟。
4. 抽三帧画图：图像、雷达坐标与 GT 同时可视化，确认轴向、单位和 ID 生命周期。

## 5. 只有厂商 Radar Track 时怎么选

如果输入已经是 `id, x, y, vx, vy, heading, length, width`，优先选择 IPC、FusionCalib、DRaT 这类目标级数据验证接口与轨迹行为。ES-MEOT/ES-EOT 的点迹分区和反射点几何无法原封不动套用；可以借鉴状态约束，但不应假装输入等价。

## 来源与校核

- [ES_MEOT 官方数据页](https://data.mendeley.com/datasets/k2kggnrtxg/1)：文件名、大小、CC BY 4.0，2026-10-27 核查。
- [FusionCalib 官方数据页](https://data.mendeley.com/datasets/xr38s95sm9/3)：v3 字段与许可。
- [V2X-Radar 官方仓库](https://github.com/yanglei18/V2X-Radar)与[数据页](https://huggingface.co/datasets/yanglei18/V2X-Radar)：目录、规模、许可。
- [ATCLL](https://zenodo.org/records/15651340)、[IPC 2025](https://data.nlr.gov/submissions/314)、[Urban mmWave-Camera](https://zenodo.org/records/8301276)、[VIDETEC-2](https://zenodo.org/records/17799385)、[DRaT](https://zenodo.org/records/20171110)：均为官方托管页。

**下一篇**：[一辆车为什么会有多个雷达点？从点目标跟踪到 EOT](/2026/10/27/radar-camera-02-point-vs-eot/)
