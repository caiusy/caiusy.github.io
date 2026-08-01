---
title: 一个 4M 参数的 YOLOv10，把我对模型压缩的直觉打碎了
date: 2026-08-01 18:30:00
updated: 2026-08-01 21:10:00
description: >-
  我在 YOLOv10 车辆关键点项目里连续尝试 FastHead、RepC3、通道剪枝、稀疏训练和知识蒸馏。参数少了不一定更快，训练也救不了错误的执行图。最后真正通过速度、精度和显存门禁的，是只动低分辨率 P5 路径的 P5-Zero。
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

一开始，我觉得这个项目不会太难。

模型只有 400 万参数左右。把 Head 缩一点，把 Backbone 里看起来重复的层剪掉，再用知识蒸馏补一补精度，怎么也该快个 10% 吧。

结果完全不是这样。

我前后试了 FastHead、RepC3、结构化通道剪枝、BN 稀疏训练和多组宽度方案。参数最多砍掉一半，真实延迟却可能原地踏步，甚至更慢。直到最后，我才找到一个同时过速度、精度和显存门槛的方案。

它叫 P5-Zero。

<!-- more -->

![YOLOv10 Vehicle Pose 项目核心成果：P5-Zero 延迟下降 14.83%，参数下降 15.14%，并通过精度门禁](/images/yolov10-vehicle-pose-compression/01-infographic-project-overview.webp)

## 先说结果

在 RTX 3090 Ti、640 输入、batch 1、Conv/BN fuse、FP16 的同进程配对测试里，P5-Zero 把延迟从 5.94537 ms 降到 5.06345 ms，提升 14.83%。

Box mAP50-95 是 0.66254，Pose mAP50-95 是 0.49592。参数下降 15.14%，推理峰值显存下降 5.02%。

这组数字不是完美答案。Pose 只比项目门槛高 0.00137，余量很小。但它确实是整个实验里唯一同时通过三项门禁的结构方案。

这篇文章真正想讲的，也不是 P5-Zero 有多神奇，而是我在这个过程中改掉了三个很危险的直觉。

参数更少，不代表 GPU 一定更快。

训练更久，不代表精度一定能回来。

GFLOPs 更低，更不能代替真实部署测速。

## 这个项目到底在做什么

任务是用 YOLOv10 做单类别车辆检测，同时预测 14 个车辆关键点。

模型训练时保留 one-to-many 和 one-to-one 两条分支。推理时只走 one-to-one 的 NMS-free 路径，再用同一组 anchor 索引把 bbox 和关键点一起取出来。

听起来只是多了一个 Pose Head，真正做起来却是一整条链路。

CarFusion 授权包要做哈希校验和双层解压，COCO Keypoints 要转换成 14×3 的 YOLO Pose 标签，训练集和验证集要按路口分组，最后还要同时看 Box mAP、Pose mAP、PCK、NME、可见性分组和固定图片的可视化结果。

![从 CarFusion 授权数据到 YOLOv10 Vehicle Pose 部署验收的完整技术链路](/images/yolov10-vehicle-pose-compression/02-flowchart-end-to-end-pipeline.webp)

下面这张不是示意图，而是 baseline 在 CarFusion 原始街景上的真实输出。红框是车辆检测结果，彩色点是模型预测的车辆关键点。这张图来自修复 bbox 坐标格式之后的 100 图视觉验收。

![YOLOv10 Vehicle Pose baseline 在 CarFusion 街景上的真实车辆检测与关键点预测](/images/yolov10-vehicle-pose-compression/05-real-baseline-street.webp)

*真实实验图：`reports/visualization/review_boxfix/4_28982.jpg`*

200 epoch baseline 跑完后，Box mAP50-95 约为 0.665，Pose mAP50-95 约为 0.515。这个 baseline 后来既是所有候选的比较对象，也是蒸馏和权重迁移的起点。

完整训练曲线也保留了下来。前几十个 epoch 提升最快，后半程逐步进入平台。这里能看到 one-to-many 和 one-to-one 两组 loss，以及 Box 和 Pose 的验证指标变化。

![YOLOv10 Vehicle Pose baseline 的 200 epoch 真实训练与验证曲线](/images/yolov10-vehicle-pose-compression/07-real-training-curves.webp)

*真实实验图：`reports/training/final_assets/results.png`*

我提前定了三条线：延迟至少下降 10%，Box 不低于 0.64761，Pose 不低于 0.49455，同时推理峰值显存还要下降。

当时还没意识到，这几条线会帮我省下很多无效训练。

验证集并不只有近距离大车。下面的 batch 里同时包含远处小目标、遮挡车辆和不同拍摄距离。它也是我后来不愿意继续压缩 P3、P4 的直观原因：关键点定位很依赖高分辨率细节。

![CarFusion 验证 batch 中远近车辆的真实检测与 14 关键点预测结果](/images/yolov10-vehicle-pose-compression/06-real-val-batch.webp)

*真实实验图：`reports/visualization/review/val_batch0_pred.jpg`*

## 第一次打脸：Head 砍了四分之一，速度只快了 6%

我最先动的是 Head。

原因很朴素。Pose tower 和 Box tower 参数不少，而且每张图都要执行。把姿态塔缩到 64×1，再把 Box tower 深度降到 1，看起来是最直接的做法。

FastHead 的参数下降了 24.64%。看到这个数字时，我原本很乐观。

真实结果是延迟只下降 6.39%，Pose mAP50-95 则从 0.51455 掉到 0.37569。

后来做模块耗时剖析才发现，改完之后 Backbone 和 Neck 仍占 71.64% 的时间。Head 少掉很多参数，并不等于端到端时间会按比例下降。这个模型太小了，batch 又只有 1，kernel 启动、访存、通道对齐和 GPU 利用率都在影响结果。

更麻烦的是，Pose Head 比普通分类头敏感得多。它承担的是空间位置、可见性和关键点之间的几何关系。结构一旦砍得太狠，靠输出蒸馏很难补回来。

## 第二次打脸：计算量降了，模型反而没变快

接下来我把注意力转向 Backbone 和 Neck。

Joint RepC3 在训练前看起来很漂亮。参数下降 17.70%，静态测速也有大约 16% 的收益。可训练到 30 epoch，Box 和 Pose 只有 0.48820 和 0.23798。

问题不是 NaN，也不是显存不够。是我一次改了太多地方，Backbone、Neck 和 Head 的特征语义同时变了。同形权重迁移、输出蒸馏和特征蒸馏都救不回来。

后面的结构化通道剪枝更反直觉。

有一组候选把融合计算量从 13.6 GFLOPs 降到 5.4 GFLOPs，真实延迟仍然没有改善。另一些候选参数少了三四成，却比 baseline 更慢一点。

![FastHead、Joint RepC3、通道剪枝与 P5-Zero 的速度精度决策对比](/images/yolov10-vehicle-pose-compression/03-comparison-experiment-map.webp)

从这里开始，我改了实验顺序。

以前是先训练，再看能不能加速。后来变成先做同进程 paired FP16 测速。训练前连 10% 都达不到的结构，直接停。

因为训练可以修精度，但不会改变执行图。一个已经实测更慢的模型，训练 50 epoch 以后仍然是那个更慢的模型。

## 转折点：别再碰高分辨率路径

几轮失败以后，一个规律越来越明显。

车辆关键点很依赖 P3 和 P4。它们保留了更多高分辨率细节，直接关系到车灯、车轮、挡风玻璃边缘这些位置能不能找准。

更激进的 P4-both0 + P5-Zero 其实跑得更快，但蒸馏到第 3 轮就进入平台，Pose 只有 0.41628。继续压 P4 的方向基本可以排除了。

于是我把改动收缩到 P5。

P5-Zero 只清空 backbone P5 和 neck P5 两个低分辨率 stage 的内部 bottleneck。P3、P4 以及 Box、Class、Pose 三个任务塔全部保留。

权重也不能简单截断。迁移器直接复制了 605 个同形张量，再对 layer 8 和 layer 22 的输出投影做语义折叠。最终 missing key 和 unexpected key 都是 0。

## 精度差 0.00086 时，我没有继续全模型硬训

P5-Zero 的恢复分两段。

第一段是全模型同源蒸馏。教师不是原始 baseline，而是结构更接近的 LitePSA + C2f-P4D1 恢复模型。

第 2 个 epoch，Box 已经到 0.66245，Pose 到 0.49369。Box 过线了，Pose 还差 0.00086。

第 3 和第 4 个 epoch 反而开始回落。

如果按照以前的思路，我可能会继续加 epoch，期待它自己涨回来。这次我停了全模型训练，只解冻 Pose tower 的 48 个张量，其余参数和 BN 全部固定。

Pose-only 精修第 1 轮，结果来到 0.66254 和 0.49592。过线后立即停止。

![P5-Zero 从原始 baseline、同源教师、折叠迁移到两阶段知识蒸馏的结构与训练路线](/images/yolov10-vehicle-pose-compression/04-framework-p5zero-recovery.webp)

这一步给我的感受很深。最后的恢复不一定需要更大范围的训练，可能只需要找到真正拖后腿的那一小部分参数。

## 还有一个意外：最快的优化不在模型里

整个项目里，收益最大的一次优化其实和网络结构无关。

早期数据放在 Windows E 盘，WSL 通过挂载路径读取。迁到 WSL ext4 后，端到端延迟从 60.10 ms 降到 16.63 ms，提升 3.61 倍。

它和模型前向的 5 至 6 ms 不是同一个口径，不能混在一张性能表里。但这个结果很有提醒意义。

如果文件系统、DataLoader、字体下载或日志平台正在拖后腿，花几天改网络可能还不如先花半小时检查环境。

## 如果再做一次，我会这样安排

先把 baseline、验证集和测速口径固定住。没有固定比较对象，后面的百分比都不可靠。

每次只改一个小地方。先确认权重能迁移、forward 正常，再跑真实硬件速度门禁。

速度不过线就停止。速度过线后只做短程恢复，用趋势决定是否继续，而不是用更多 epoch 掩盖方向错误。

最后再跑完整精度、独立进程显存和配对延迟。三项都过，才把它叫作可交付方案。

## 最后说几个不够漂亮的地方

CarFusion 里第 14 个关键点在 62,434 个转换实例中 visibility 全是 0。所以模型虽然保持 14×3 输出兼容，数据真正支持的只有 13 个有效点。

P5-Zero 的 Pose 余量也只有 0.00137。换 GPU、PyTorch、CUDA 或数据版本，都应该重新验收，不能直接照搬结论。

另外，这个结果只在 YOLOv10n 上验证过。仓库虽然已经把通用剪枝入口扩展到 n、s、l，但 s 和 l 仍然要各自建立 baseline 和门禁。

项目代码和完整复现说明放在 [Caius-Lu/yolov10-vehicle-pose](https://github.com/Caius-Lu/yolov10-vehicle-pose)。

这次实验让我真正记住了一件事。

小模型压缩最容易骗人的，就是那些看起来特别漂亮的参数量和 GFLOPs。最终能不能交付，还是要让真实硬件和完整验证说话。
