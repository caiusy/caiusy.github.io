---
title: SqueezeNet
categories: 深度学习
date: 2019-08-20 00:00:00
tags: 深度学习
  - Deep Learning
  - 论文阅读
---

论文地址: [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360.pdf)  
非官方代码: [pytorch](https://github.com/arvention/SqueezeNet-PyTorch)

## 介绍

这篇文章是DeepScal，加州大学伯克利分校，以及斯坦福大学在ICLR 2017发表的一篇文章。文章的主要目的是为了压缩模型，提高运行速度。这篇文章主要提出了SqueezeNet: 使用少量参数保持精度。

## 结构设计策略

这篇文章的首要目标是在保持准确率 的同时，有几个参数的CNN架构。这篇文章在设计CNN架构的时候采取了三个主要策略。这篇文章的主要模块 是Fire模块。

  * 用1x1的卷积核代替3x3的卷积核，从而减少参数量。1x1 卷积的参数比3x3的卷积核少了 9X.
  * 减少3x3 卷积输入通道的数量。假设有一个卷积层, 它完全由3x3 卷积组成。此层中参数的总数量为：(输入通道数) _(过滤器数)_ (3 * 3)。要在squeeze层中将输入的通道数减少。
  * 在网络中减少下采样(maxpooling)实现, 以便卷积层具有较大的特征图。

## Fire Module

Fire Module是将原来一层conv层变成两层：squeeze层+expand层，各自带上Relu激活层。在squeeze层里面全是1x1的卷积kernel，数量记为S11；在expand层里面有1x1和3x3的卷积kernel，expand层之后将1x1和3x3的卷积output feature maps在channel维度cat。  
![](/images/20190820_squeezenet_fire.png)

* * *

自己手推的一张图，字比较丑，也没时间重现写一下。  
![](/images/20190820_squeezenet_squeezenet.jpg)

### fire moudle的pytorch代码

很奇怪的是论文中用的是3个1x1，以及expand用的是4个1x1的卷积核和4个 3x3的卷积核，但是pytroch版本的代码并没有体现出来。  

    
    

```python
    class fire(nn.Module):
        def __init__(self, inplanes,squeeze_planes, expand_planes):
            super(fire,self).__init__()
            self.conv1 = nn.Conv2d(inplanes,squeeze_planes, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm2d(squeeze_planes)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
            self.bn2 = nn.BatchNorm2d(expand_planes)
            self.conv3 = nn.Conv2d(squeeze_planes,expand_planes,kernel_size=3, stride=1,padding=1)
            self.bn3 = nn.BatchNorm2d(expand_planes)
            self.relu2 = nn.ReLU(inplace=True)
            # using MSR initialization
            for m in self.modules():
                if isinstance(m,nn.Conv2d):
                    n = m.kernel_size[0]*m.kernel_size[1]*m.in_channels
                    m.weight.data.normal_(0,math.sqrt(2./n))
        def forward(self,x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            out1 = self.conv2(x)
            out1 = self.bn2(out1)
            out2 = self.conv3(x)
            out2 = self.bn3(out2)
            out = torch.cat([out1,out2],1)
            out = self.relu2(out)
            return out
```

  
## SqueezeNet的具体网络结构

![](/images/20190820_squeezenet_struct.png)

## 实验结果

imagenet数据上比较了alexnet，可以看到准确率差不多的情况下，squeezeNet模型参数数量显著降低了（下表倒数第三行），参数减少50X；如果再加上deep compression技术，压缩比可以达到461X！还是不错的结果。  
![](/images/20190820_squeezenet_params.png)

参考文章：<https://blog.csdn.net/xbinworld/article/details/50897870>
