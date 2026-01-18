---
title: adaptive_pose
typora-copy-images-to: adaptive_pose
date: 2023-01-22 21:15:24
tags: [pose, deep learning]
---
>[AdaptivePose++](https://arxiv.org/pdf/2210.04014.pdf) : [github](https://github.com/buptxyb666/AdaptivePose)

![img](../images/framework.jpg)

#### 问题

1. z['ap'] 虽然输出了，但是没有参与最终的计算， 这个ap更像是网络自己学的一个中间级的过程，这个ap没有监督 能学到像论文的示意么。
2. Resample2D的作用是啥，在flownet2中搜到了，但还是不太清楚， 是将不同位置的特征进行融合么。 self.gradient_mul是为了控制ap回传的梯度范围么，这个是经验值吗。
3. 我之前在centernet中加入oks 替换掉了原来的RegWeightedL1Loss_coco()， 你这是额外加了oks的loss，这没有重复学这个回归信息么， 如果为了加快收敛可不可以先RegWeightedL1Loss_coco()，再oks。我一直很好奇如果是自己的数据集，oks中的sigma一般怎么估计呀。 还有oks 的应该是以绝对位置作为计算吧， 我看代码里好像是相对中心点的偏移。

#### 回答

1. z['ap'] 只是用于可视化，没有显示的监督，adaptivepose使用中心特征预测ap偏移，再取出ap位置的特征第二跳偏移，整个两跳path是梯度可回传的，所以相当于隐式监督的。
2. Resample2D就是warp操作，通过双线性插值取ap位置特征。self.gradient_mul这块意思跟降低该层的学习率一个意思。
3. 我这边实验效果 oks+L1 > oks > L1。先RegWeightedL1Loss_coco()，再oks这个操作你可以自己试试。自己的数据集如果是人体关键点你直接按着coco取对应位置的sigma就可以了，sigma跟数据集无关。off_to_pose中将中心坐标加到偏移上。首先认为标注过程符合高斯分布，sigma 跟 scale这俩参数乘积，就是高斯分布的方差，直觉上理解就是对偏差的容忍度，比如同样偏移五个像素，可能对于eye的预测误差就是不可容忍的，对于hip的预测误差是可容忍的，对于large scale是可容忍的，对small scale是不可容忍的。coco上提供的标注，也是脸部关键点的sigma最小，其他的大一些，你可以按着这个思路来估算下你所估计的点的sigma。

#### 代码解析

![image-20230123160115331](../images/adaptive-pose/image-20230123160115331.png)

![image-20230123160148686](../images/adaptive-pose/image-20230123160148686.png)

![image-20230123160203161](../images/adaptive-pose/image-20230123160203161.png)

![image-20230123160217277](../images/adaptive-pose/image-20230123160217277.png)

![image-20230123160227108](../images/adaptive-pose/image-20230123160227108.png)
