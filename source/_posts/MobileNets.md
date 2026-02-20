---
title: MobileNets
date: 2019-08-19 00:00:00
categories:
  - 深度学习
tags:
  - PyTorch
  - CNN
---

论文地址:[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications ](https://arxiv.org/abs/1704.04861)  
非官方代码:[pytorch/models](https://github.com/marvis/pytorch-mobilenet)  
  
## 前言

这篇文章是谷歌在2017针对手机等嵌入式设备提出的一种轻量级深层网络，这篇论文主要的贡献点在于提出了一种深度可分离卷积。

  * 主要解决的问题是注重优化延迟，同时也兼顾了模型的大小，不像有些模型虽然参数量比较小，但是速度也是慢的可以。
  * MobileNets使用了大量的3 × 3的卷积核，极大地减少了计算量（1/8到1/9之间），同时准确率下降的很少，相比其他的方法确有优势。

## 深度可分离卷积示例

![](/2019/08/19/MobileNets/images/20190819_MobileNets_MobileNets.jpg)

## 模型结构和训练

MobileNets结构建立在上述深度可分解卷积中（只有第一层是标准卷积）。该网络允许我们探索网络拓扑，找到一个适合的良好网络。其具体架构在表1说明。除了最后的全连接层，所有层后面跟了batchnorm和ReLU，最终输入到softmax进行分类。图3对比了标准卷积和分解卷积的结构，二者都附带了BN和ReLU层。按照作者的计算方法，MobileNets总共28层（1 + 2 × 13 + 1 = 28）
