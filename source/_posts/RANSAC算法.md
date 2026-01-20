---
title: RANSAC算法
categories: 算法
typora-copy-images-to: ./RANSAC算法
date: 2023-01-22 16:55:53
tags: 算法
---
### RANSAC算法

___

**RANSAC**(**RA**ndom **SA**mple **C**onsensus,随机采样一致)算法

___

**RANSAC**(**RA**ndom **SA**mple **C**onsensus,随机采样一致)算法是从一组含有“外点”(outliers)的数据中正确估计数学模型参数的迭代算法。“外点”一般指的的数据中的噪声，比如说匹配中的误匹配和估计曲线中的离群点。所以，RANSAC也是一种“外点”检测算法。RANSAC算法是一种不确定算法，它只能在一种概率下产生结果，并且这个概率会随着迭代次数的增加而加大（之后会解释为什么这个算法是这样的）。RANSAC算最早是由Fischler和Bolles在SRI上提出用来解决LDP(Location Determination Proble)问题的。

对于RANSAC算法来说一个**基本的假设**就是数据是由“内点”和“外点”组成的。“内点”就是组成模型参数的数据，“外点”就是不适合模型的数据。同时RANSAC假设：在给定一组含有少部分“内点”的数据，存在一个程序可以估计出符合“内点”的模型。

#### 算法基本思想和流程

RANSAC是通过反复选择数据集去估计出模型，一直迭代到估计出认为比较好的模型。
具体的实现步骤可以分为以下几步：

1. 选择出可以估计出模型的最小数据集；(对于直线拟合来说就是两个点，对于计算Homography矩阵就是4个点)
2. 使用这个数据集来计算出数据模型；
3. 将所有数据带入这个模型，计算出“内点”的数目；(累加在一定误差范围内的适合当前迭代推出模型的数据)
4. 比较当前模型和之前推出的最好的模型的“内点“的数量，记录最大“内点”数的模型参数和“内点”数；
5. 重复1-4步，直到迭代结束或者当前模型已经足够好了(“内点数目大于一定数量”)。

#### 迭代次数推导

假设“内点”在数据中的占比为 ![[公式]](https://www.zhihu.com/equation?tex=t)

![[公式]](https://www.zhihu.com/equation?tex=t%3D%5Cfrac%7Bn_%7Bi+n+l+i+e+r+s%7D%7D%7Bn_%7Bi+n+l+i+e+r+s%7D%2Bn_%7Bo+u+t+l+i+e+r+s%7D%7D+%5C%5C)

那么我们每次计算模型使用 ![[公式]](https://www.zhihu.com/equation?tex=N) 个点的情况下，选取的点至少有一个外点的情况就是

![[公式]](https://www.zhihu.com/equation?tex=+1+-+t%5EN+%5C%5C)

也就是说，在迭代 ![[公式]](https://www.zhihu.com/equation?tex=k) 次的情况下， ![[公式]](https://www.zhihu.com/equation?tex=%281-t_n%29%5Ek) 就是 ![[公式]](https://www.zhihu.com/equation?tex=k) 次迭代计算模型都至少采样到一个“外点”去计算模型的概率。那么能采样到正确的 ![[公式]](https://www.zhihu.com/equation?tex=N) 个点去计算出正确模型的概率就是

![[公式]](https://www.zhihu.com/equation?tex=P%3D1-%5Cleft%281-t%5E%7Bn%7D%5Cright%29%5E%7Bk%7D+%5C%5C)

通过上式，可以求得

![[公式]](https://www.zhihu.com/equation?tex=k%3D%5Cfrac%7B%5Clog+%281-P%29%7D%7B%5Clog+%5Cleft%281-t%5E%7Bn%7D%5Cright%29%7D++%5C%5C)

内点”的概率 ![[公式]](https://www.zhihu.com/equation?tex=t) 通常是一个先验值。然后 ![[公式]](https://www.zhihu.com/equation?tex=P) 是我们希望RANSAC得到正确模型的概率。如果事先不知道 ![[公式]](https://www.zhihu.com/equation?tex=t) 的值，可以使用自适应迭代次数的方法。也就是一开始设定一个无穷大的迭代次数，然后每次更新模型参数估计的时候，用当前的“内点”比值当成 ![[公式]](https://www.zhihu.com/equation?tex=t) 来估算出迭代次数。

##### 用Python实现直线拟合

~~~~python
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# 数据量。
SIZE = 50
# 产生数据。np.linspace 返回一个一维数组，SIZE指定数组长度。
# 数组最小值是0，最大值是10。所有元素间隔相等。
X = np.linspace(0, 10, SIZE)
Y = 3 * X + 10

fig = plt.figure()
# 画图区域分成1行1列。选择第一块区域。
ax1 = fig.add_subplot(1,1, 1)
# 标题
ax1.set_title("RANSAC")


# 让散点图的数据更加随机并且添加一些噪声。
random_x = []
random_y = []
# 添加直线随机噪声
for i in range(SIZE):
    random_x.append(X[i] + random.uniform(-0.5, 0.5)) 
    random_y.append(Y[i] + random.uniform(-0.5, 0.5)) 
# 添加随机噪声
for i in range(SIZE):
    random_x.append(random.uniform(0,10))
    random_y.append(random.uniform(10,40))
RANDOM_X = np.array(random_x) # 散点图的横轴。
RANDOM_Y = np.array(random_y) # 散点图的纵轴。

# 画散点图。
ax1.scatter(RANDOM_X, RANDOM_Y)
# 横轴名称。
ax1.set_xlabel("x")
# 纵轴名称。
ax1.set_ylabel("y")

# 使用RANSAC算法估算模型
# 迭代最大次数，每次得到更好的估计会优化iters的数值
iters = 100000
# 数据和模型之间可接受的差值
sigma = 0.25
# 最好模型的参数估计和内点数目
best_a = 0
best_b = 0
pretotal = 0
# 希望的得到正确模型的概率
P = 0.99
for i in range(iters):
    # 随机在数据中红选出两个点去求解模型
    sample_index = random.sample(range(SIZE * 2),2)
    x_1 = RANDOM_X[sample_index[0]]
    x_2 = RANDOM_X[sample_index[1]]
    y_1 = RANDOM_Y[sample_index[0]]
    y_2 = RANDOM_Y[sample_index[1]]

    # y = ax + b 求解出a，b
    a = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - a * x_1

    # 算出内点数目
    total_inlier = 0
    for index in range(SIZE * 2):
        y_estimate = a * RANDOM_X[index] + b
        if abs(y_estimate - RANDOM_Y[index]) < sigma:
            total_inlier = total_inlier + 1

    # 判断当前的模型是否比之前估算的模型好
    if total_inlier > pretotal:
        iters = math.log(1 - P) / math.log(1 - pow(total_inlier / (SIZE * 2), 2))
        pretotal = total_inlier
        best_a = a
        best_b = b

    # 判断是否当前模型已经符合超过一半的点
    if total_inlier > SIZE:
        break

# 用我们得到的最佳估计画图
Y = best_a * RANDOM_X + best_b

# 直线图
ax1.plot(RANDOM_X, Y)
text = "best_a = " + str(best_a) + "\nbest_b = " + str(best_b)
plt.text(5,10, text,
         fontdict={'size': 8, 'color': 'r'})
plt.show()
~~~~

![image-20220718002303153](./images/image-20220718002303153.png)


