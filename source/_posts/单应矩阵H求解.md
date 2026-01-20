---
title: 单应矩阵H求解
categories: 技术
typora-copy-images-to: ./单应矩阵H求解
date: 2023-01-22 16:55:53
tags: 技术
---
单应矩阵H求解
---


所需求解的单应矩阵：
$$
H_{3 \times 3}=\left[\begin{array}{lll}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{array}\right]
$$
单应变换关系：
$$
\mathrm{s}\left[\begin{array}{c}
x^{\prime} \\
y^{\prime} \\
1
\end{array}\right]=H\left[\begin{array}{l}
x \\
y \\
1
\end{array}\right]=\left[\begin{array}{lll}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{array}\right]\left[\begin{array}{l}
x \\
y \\
1
\end{array}\right]
$$
为减少自由度， 令h<sub>33</sub> =1, s为尺度因子。
$$
h_{31} x+h_{32} y+h_{33}=h_{31} x+h_{32} y+1
$$

$$
\begin{aligned}
x_{i}^{\prime} &=\frac{h_{11} x_{i}+h_{12} y_{i}+h_{13}}{h_{31} x_{i}+h_{32} y_{i}+h_{33}} \\
y_{i}^{\prime} &=\frac{h_{21} x_{i}+h_{22} y_{i}+h_{23}}{h_{31} x_{i}+h_{32} y_{i}+h_{33}}
\end{aligned}
$$

在图像上取在真实世界构成矩形的顺时针的四个角点。

(x1,y1),(x2,y2),(x3,y3),(x4,y4)以及构成直角的三个点

(x5,y5),(x6,y6),(x7,y7), 其中（x6,y6)为直角点

预设期望变换后的矩形，第一个焦点的位置（x1',y1')(x2',y2') 设计方程组：

令（x1，y1) (x2,y2) 变换到预设位置（x1',y1')(x2',y2') 
$$
\begin{aligned}
x_{1}^{\prime} &=\frac{h_{11} x_{1}+h_{12} y_{1}+h_{13}}{h_{31} x_{1}+h_{32} y_{1}+h_{33}} \\
x_{2}^{\prime} &=\frac{h_{11} x_{2}+h_{12} y_{2}+h_{13}}{h_{31} x_{2}+h_{32} y_{2}+h_{33}} \\
y_{1}^{\prime} &=\frac{h_{21} x_{1}+h_{22} y_{1}+h_{23}}{h_{31} x_{1}+h_{32} y_{1}+h_{33}} \\
y_{2}^{\prime} &=\frac{h_{21} x_{2}+h_{22} y_{2}+h_{23}}{h_{31} x_{2}+h_{32} y_{2}+h_{33}}
\end{aligned}
$$
令（x1',y1')和(x4',y4')构成的直线垂直与（x1',y1')(x2',y2') 构成的直线
$$
\frac{y_{2}^{\prime}-y_{1}^{\prime}}{x_{2}^{\prime}-x_{1}^{\prime}} \cdot \frac{y_{4}^{\prime}-y_{1}^{\prime}}{x_{4}^{\prime}-x_{1}^{\prime}}=-1
$$
令x'4到x1'的距离等于x3' 到x2'的距离， y4'到y1'的距离等于y3'到y2‘的距离，使得(x4',y4')和(x3',y3')在（x1',y1')和(x2',y2') 构成的直线的同一侧， 且结合上一个约束， 使得(x2',y2') 和(x3',y3')直线平行与（x1',y1')和(x4',y4') 构成的直线，即(x2',y2') 和(x3',y3') 垂直与

（x1',y1')和(x2',y2') 构成的直线
$$
\begin{aligned}
&x_{4}^{\prime}-x_{3}^{\prime}=x_{1}^{\prime}-x_{2}^{\prime} \\
&y_{4}^{\prime}-y_{3}^{\prime}=y_{1}^{\prime}-y_{2}^{\prime}
\end{aligned}
$$
令(x5',y5') 和(x6',y6') 垂直与（x6',y6')和(x7',y7') 构成的直线
$$
\frac{y_{5}^{\prime}-y_{6}^{\prime}}{x_{5}^{\prime}-x_{6}^{\prime}} \cdot \frac{y_{7}^{\prime}-y_{6}^{\prime}}{x_{7}^{\prime}-x_{6}^{\prime}}=-1
$$
联立上述8个方程和h33=1可求出单应矩阵H

此外为了简化求解， 可使得预设期望的变换后的矩形第一个角点（x1',y1')，第二个角点的位置（x2',y2')的y1和y2 或者x1和x2 相等， 只需要x3=x2 ,x4=x1 即可以保证垂直关系， 计算将大大简化

通过多选几组点， 结合RANSAC等算法可进行全局优化。

![image-20220717235918026](./images/image-20220717235918026.png)