---
title: MobileNetsV2
categories: 深度学习
date: 2019-08-19 00:00:00
tags: 深度学习
  - Deep Learning
  - 论文阅读
---

论文地址:[MobileNetV2: Inverted Residuals and Linear Bottlenecks ](https://arxiv.org/pdf/1801.04381.pdf)  
非官方代码:[pytorch](https://github.com/tonylins/pytorch-mobilenet-v2)

## 介绍

这篇文章是谷歌在2019提出来的文章在MobileNets 基础上做的改进。

## 深度可分离卷积示例

  * 首先在Xception 中被广泛使用
  * 好处： 理论上可以成倍的减少卷积层的时间复杂度和空间复杂度![](/images/20190819_MobileNetsV2_MobileNets.jpg)

## 文章内容

### 与MobileNets 的对比

  * 相同点
    * 都采用 Depth-wise (DW) 卷积搭配 Point-wise (PW) 卷积的方式来提特征
  * 不同点
    * Linear Bottleneck
    * V2 在 DW 卷积之前新加了一个 PW 卷积。
      * DW卷积由于本身的计算特性不能改变通道数的能力。若通道数很少的话，DW在提取地低纬特征，效果可能并不会好。
      * 在每个DW之前，增加了PW用于升维，这样DW可以更好的提取特征
    * V2 去掉了第二个 PW 的激活函数
      * 激活函数在高维空间能够有效的增加非线性，而在低维空间时则会破坏特征
      * 第二个 PW 的主要功能就是降维

* * *

### 与ResNet的对比

  * 相同点
    * MobileNet V2 借鉴 ResNet，都采用了 1x1->3x3->1x1的模式
    * MobileNet V2 借鉴 ResNet，同样使用 Shortcut 将输出与输入相加
  * 不同点
    * Inverted Residual Block
    * ResNet 使用 标准卷积 提特征，MobileNet 始终使用 DW卷积 提特征
    * ResNet 先降维 (0.25倍)、卷积、再升维，而 MobileNet V2 则是 先升维 (6倍)、卷积、再降维。直观的形象上来看，ResNet 的微结构是沙漏形，而 MobileNet V2 则是纺锤形，刚好相反。因此论文作者将 MobileNet V2 的结构称为 Inverted Residual Block。使用DW卷积而作的适配，特征提取能够在高维进行

* * *

## 结论

  * MobileNets 与MobileNets V2在模型结构上的对比
  * MobileNetsV2 的卷积层数比V1要多，但是时间复杂度，以及空间复杂度，以及在cpu上的推理时间要远远优于MobileNets

参考文章：<https://zhuanlan.zhihu.com/p/33075914>
