---
title: 相机几何完全指南：从世界坐标到图像坐标的完整推导
date: 2026-01-19 22:30:00
tags: 
  - 计算机视觉
  - 相机标定
  - 多视几何
  - 矩阵分解
categories: 计算机视觉
mathjax: true
---

本文详细推导从3D世界坐标系到2D图像坐标系的完整数学过程，包括相机内外参数、旋转矩阵、单应矩阵的推导与分解，并提供完整的Python可视化代码。

<!-- more -->



## 📖 目录

1. [坐标系统概述](#坐标系统概述)
2. [从世界坐标系到相机坐标系（外参）](#外参推导)
3. [从相机坐标系到图像坐标系（内参）](#内参推导)
4. [旋转矩阵详解](#旋转矩阵)
5. [单应矩阵推导](#单应矩阵)
6. [相机矩阵分解](#矩阵分解)
7. [Python完整实现](#Python实现)

---

## 一、坐标系统概述 {#坐标系统概述}

在计算机视觉中，从3D世界到2D图像需要经过**四个坐标系统**的转换：

### 1.1 四个坐标系统


#### **1. 世界坐标系 (World Coordinate System)**
- 符号：$(X_w, Y_w, Z_w)$
- 描述：真实世界中的3D坐标系统
- 单位：通常为米(m)或毫米(mm)
- 原点：任意选定的参考点

#### **2. 相机坐标系 (Camera Coordinate System)**
- 符号：$(X_c, Y_c, Z_c)$
- 描述：以相机光心为原点的3D坐标系
- 单位：米(m)或毫米(mm)
- 原点：相机光心
- 特点：$Z_c$ 轴为光轴方向

#### **3. 图像坐标系 (Image Coordinate System)**
- 符号：$(x, y)$
- 描述：成像平面上的物理坐标
- 单位：毫米(mm)
- 原点：图像中心（主点）

#### **4. 像素坐标系 (Pixel Coordinate System)**
- 符号：$(u, v)$
- 描述：数字图像的离散像素坐标
- 单位：像素(pixel)
- 原点：图像左上角

### 1.2 完整的投影公式

从世界坐标到像素坐标的完整变换：

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K \cdot [R|t] \cdot \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

其中：
- $K$：内参矩阵 (3×3)
- $[R|t]$：外参矩阵 (3×4)
- $\sim$：表示齐次坐标意义下的相等（差一个尺度因子）

---

## 二、从世界坐标系到相机坐标系（外参矩阵）{#外参推导}

### 2.1 刚体变换

世界坐标系到相机坐标系的转换是一个**刚体变换**（Rigid Body Transformation），包含旋转和平移：

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = R \begin{bmatrix} X_w \\ Y_w \\ Z_w \end{bmatrix} + t
$$

其中：
- $R \in \mathbb{R}^{3 \times 3}$：旋转矩阵（Rotation Matrix）
- $t \in \mathbb{R}^{3 \times 1}$：平移向量（Translation Vector）

### 2.2 齐次坐标表示

使用齐次坐标可以将旋转和平移统一表示：

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
R & t \\
0^T & 1
\end{bmatrix}
\begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

在实际应用中，我们通常使用 $3 \times 4$ 的外参矩阵：

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = 
[R|t] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

其中：

$$
[R|t] = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z
\end{bmatrix}
$$

### 2.3 外参的物理意义

- **旋转矩阵 $R$**：描述相机坐标系相对于世界坐标系的方向
- **平移向量 $t$**：描述相机光心在世界坐标系中的位置
- **自由度**：6个（3个旋转 + 3个平移）

### 2.4 外参的逆变换

从相机坐标系回到世界坐标系：

$$
\begin{bmatrix} X_w \\ Y_w \\ Z_w \end{bmatrix} = R^T \left( \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} - t \right) = R^T \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} - R^T t
$$

注意：
- $R^T = R^{-1}$（旋转矩阵的转置等于其逆）
- 相机在世界坐标系中的位置为 $C = -R^T t$

---

## 三、从相机坐标系到图像坐标系（内参矩阵）{#内参推导}

### 3.1 针孔相机模型

针孔相机模型是最基本的相机模型：

```
光心 O ────────────────→ 成像平面
          ↗         ↗
      3D点 P    投影点 p
      
相似三角形: x/f = Xc/Zc, y/f = Yc/Zc
因此: x = f·(Xc/Zc), y = f·(Yc/Zc)
```

#### 透视投影公式

根据相似三角形原理：

$$
\frac{x}{f} = \frac{X_c}{Z_c}, \quad \frac{y}{f} = \frac{Y_c}{Z_c}
$$

其中 $f$ 是焦距（focal length），单位为毫米。

因此：

$$
x = f \frac{X_c}{Z_c}, \quad y = f \frac{Y_c}{Z_c}
$$

### 3.2 从图像坐标到像素坐标

图像坐标 $(x, y)$ 是物理坐标（毫米），需要转换为像素坐标 $(u, v)$：

$$
\begin{cases}
u = \alpha x + c_x \\
v = \beta y + c_y
\end{cases}
$$

其中：
- $\alpha = \frac{1}{dx}$：x方向的像素密度（像素/毫米）
- $\beta = \frac{1}{dy}$：y方向的像素密度（像素/毫米）
- $(c_x, c_y)$：主点坐标（图像中心在像素坐标系中的位置）

### 3.3 内参矩阵推导

将上述两步合并：

$$
\begin{aligned}
u &= \alpha \cdot f \frac{X_c}{Z_c} + c_x = f_x \frac{X_c}{Z_c} + c_x \\
v &= \beta \cdot f \frac{Y_c}{Z_c} + c_y = f_y \frac{Y_c}{Z_c} + c_y
\end{aligned}
$$

其中：
- $f_x = \alpha \cdot f$：x方向焦距（像素单位）
- $f_y = \beta \cdot f$：y方向焦距（像素单位）

使用齐次坐标表示：

$$
Z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = 
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
$$

### 3.4 完整的内参矩阵

考虑像素倾斜（skew）的一般形式：

$$
K = \begin{bmatrix}
f_x & s & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

其中：
- $f_x, f_y$：焦距（像素单位）
- $c_x, c_y$：主点坐标（像素）
- $s$：倾斜系数（通常为0）

### 3.5 内参矩阵的性质

- **维度**：$3 \times 3$
- **自由度**：5个（现代相机中$s=0$，则为4个）
- **特点**：上三角矩阵
- **物理意义**：描述相机的内部几何特性

---

## 四、旋转矩阵详解 {#旋转矩阵}

### 4.1 旋转矩阵的定义与性质

旋转矩阵 $R \in SO(3)$ 是一个特殊正交矩阵，满足：

1. **正交性**：$R^T R = R R^T = I$
2. **行列式**：$\det(R) = 1$
3. **保持长度**：$\|Rv\| = \|v\|$
4. **保持角度**：$(Rv_1) \cdot (Rv_2) = v_1 \cdot v_2$

### 4.2 基本旋转矩阵


![基本旋转矩阵可视化](camera-geometry-complete-guide/rotation_matrices_visualization.png)
#### 绕X轴旋转（Roll）

$$
R_x(\alpha) = \begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\alpha & -\sin\alpha \\
0 & \sin\alpha & \cos\alpha
\end{bmatrix}
$$

#### 绕Y轴旋转（Pitch）

$$
R_y(\beta) = \begin{bmatrix}
\cos\beta & 0 & \sin\beta \\
0 & 1 & 0 \\
-\sin\beta & 0 & \cos\beta
\end{bmatrix}
$$

#### 绕Z轴旋转（Yaw）

$$
R_z(\gamma) = \begin{bmatrix}
\cos\gamma & -\sin\gamma & 0 \\
\sin\gamma & \cos\gamma & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

### 4.3 欧拉角表示

任意旋转可以分解为三个基本旋转的组合（有多种顺序）：

**ZYX欧拉角（常用）**：

$$
R = R_z(\gamma) R_y(\beta) R_x(\alpha)
$$

展开为：

$$
R = \begin{bmatrix}
\cos\gamma\cos\beta & \cos\gamma\sin\beta\sin\alpha - \sin\gamma\cos\alpha & \cos\gamma\sin\beta\cos\alpha + \sin\gamma\sin\alpha \\
\sin\gamma\cos\beta & \sin\gamma\sin\beta\sin\alpha + \cos\gamma\cos\alpha & \sin\gamma\sin\beta\cos\alpha - \cos\gamma\sin\alpha \\
-\sin\beta & \cos\beta\sin\alpha & \cos\beta\cos\alpha
\end{bmatrix}
$$

⚠️ **万向锁问题**：当 $\beta = \pm 90°$ 时，会出现万向锁（Gimbal Lock）。

### 4.4 轴角表示（Axis-Angle）

用旋转轴 $\mathbf{n} = (n_x, n_y, n_z)^T$（单位向量）和旋转角 $\theta$ 表示旋转。

**罗德里格斯公式（Rodrigues' Formula）**：

$$
R = I + \sin\theta [\mathbf{n}]_\times + (1-\cos\theta)[\mathbf{n}]_\times^2
$$

其中 $[\mathbf{n}]_\times$ 是反对称矩阵：

$$
[\mathbf{n}]_\times = \begin{bmatrix}
0 & -n_z & n_y \\
n_z & 0 & -n_x \\
-n_y & n_x & 0
\end{bmatrix}
$$

### 4.5 四元数表示（Quaternion）

四元数 $q = q_0 + q_1i + q_2j + q_3k$ 可以避免万向锁，其中 $q_0^2 + q_1^2 + q_2^2 + q_3^2 = 1$。

**四元数到旋转矩阵**：

$$
R = \begin{bmatrix}
1-2(q_2^2+q_3^2) & 2(q_1q_2-q_0q_3) & 2(q_1q_3+q_0q_2) \\
2(q_1q_2+q_0q_3) & 1-2(q_1^2+q_3^2) & 2(q_2q_3-q_0q_1) \\
2(q_1q_3-q_0q_2) & 2(q_2q_3+q_0q_1) & 1-2(q_1^2+q_2^2)
\end{bmatrix}
$$

### 4.6 旋转矩阵的列向量含义 ⭐

旋转矩阵 $R$ 的**列向量**具有重要的几何意义：

$$
R = \begin{bmatrix} | & | & | \\ \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{r}_3 \\ | & | & | \end{bmatrix}
$$

**核心理解**：

> **$R$ 的第 $i$ 列 $\mathbf{r}_i$ 表示世界坐标系的第 $i$ 个基向量在相机坐标系下的表示。**

具体来说：

- **第1列 $\mathbf{r}_1$**：世界坐标系的 X 轴方向在相机坐标系中的表示
- **第2列 $\mathbf{r}_2$**：世界坐标系的 Y 轴方向在相机坐标系中的表示
- **第3列 $\mathbf{r}_3$**：世界坐标系的 Z 轴方向在相机坐标系中的表示

**推导**：

世界坐标系的基向量为：
$$
\mathbf{e}_1 = \begin{bmatrix}1\\0\\0\end{bmatrix}, \quad 
\mathbf{e}_2 = \begin{bmatrix}0\\1\\0\end{bmatrix}, \quad 
\mathbf{e}_3 = \begin{bmatrix}0\\0\\1\end{bmatrix}
$$

在相机坐标系中：
$$
R\mathbf{e}_1 = \mathbf{r}_1, \quad R\mathbf{e}_2 = \mathbf{r}_2, \quad R\mathbf{e}_3 = \mathbf{r}_3
$$

**示例**：

假设：
$$
R = \begin{bmatrix}
0.866 & -0.500 & 0 \\
0.500 & 0.866 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

这是绕Z轴旋转30°的旋转矩阵。

- $\mathbf{r}_1 = [0.866, 0.500, 0]^T$：世界X轴在相机系中指向 $(0.866, 0.500, 0)$
- $\mathbf{r}_2 = [-0.500, 0.866, 0]^T$：世界Y轴在相机系中指向 $(-0.500, 0.866, 0)$
- $\mathbf{r}_3 = [0, 0, 1]^T$：世界Z轴在相机系中仍指向 $(0, 0, 1)$

### 4.7 旋转矩阵的行向量含义

相反地，$R^T$ 的列（即 $R$ 的行）表示**相机坐标系的基向量在世界坐标系下的表示**：

$$
R^T = \begin{bmatrix} 
- & \mathbf{r}_1^T & - \\ 
- & \mathbf{r}_2^T & - \\ 
- & \mathbf{r}_3^T & - 
\end{bmatrix}
$$

由于 $R^T = R^{-1}$，我们有：
- $\mathbf{r}_1^T$：相机X轴在世界坐标系中的方向
- $\mathbf{r}_2^T$：相机Y轴在世界坐标系中的方向
- $\mathbf{r}_3^T$：相机Z轴（光轴）在世界坐标系中的方向

---

## 五、单应矩阵推导 {#单应矩阵}

![单应矩阵变换示意图](camera-geometry-complete-guide/homography_transformation.png)

### 5.1 单应矩阵的定义

单应矩阵（Homography Matrix）$H$ 描述两个平面之间的投影变换关系：

$$
\mathbf{p}' \sim H \mathbf{p}
$$

即：

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim 
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

### 5.2 单应矩阵的推导

#### 场景1：平面物体的投影

假设世界坐标系中的平面满足 $Z_w = 0$，则：

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = 
R \begin{bmatrix} X_w \\ Y_w \\ 0 \end{bmatrix} + t = 
\begin{bmatrix} r_1 & r_2 & r_3 \end{bmatrix}
\begin{bmatrix} X_w \\ Y_w \\ 0 \end{bmatrix} + t
$$

$$
= \begin{bmatrix} r_1 & r_2 \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \end{bmatrix} + t
= [r_1 \; r_2 \; t] \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix}
$$

因此像素坐标为：

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim 
K [r_1 \; r_2 \; t] \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix}
$$

定义单应矩阵：

$$
H = K [r_1 \; r_2 \; t] = K \begin{bmatrix} | & | & | \\ r_1 & r_2 & t \\ | & | & | \end{bmatrix}
$$

### 5.3 单应矩阵的性质

- **维度**：$3 \times 3$
- **自由度**：8（因为尺度不定性，9个元素减去1个尺度）
- **可逆**：$H^{-1}$ 描述逆向映射
- **非线性**：由于齐次坐标的尺度不定性

### 5.4 单应矩阵的求解

给定 $n$ 对对应点 $(x_i, y_i) \leftrightarrow (x_i', y_i')$，每对点提供2个约束方程：

$$
\begin{aligned}
x_i' &= \frac{h_{11}x_i + h_{12}y_i + h_{13}}{h_{31}x_i + h_{32}y_i + h_{33}} \\
y_i' &= \frac{h_{21}x_i + h_{22}y_i + h_{23}}{h_{31}x_i + h_{32}y_i + h_{33}}
\end{aligned}
$$

交叉相乘后得到线性方程：

$$
\begin{bmatrix}
-x_i & -y_i & -1 & 0 & 0 & 0 & x_i'x_i & x_i'y_i & x_i' \\
0 & 0 & 0 & -x_i & -y_i & -1 & y_i'x_i & y_i'y_i & y_i'
\end{bmatrix}
\begin{bmatrix} h_{11} \\ h_{12} \\ h_{13} \\ h_{21} \\ h_{22} \\ h_{23} \\ h_{31} \\ h_{32} \\ h_{33} \end{bmatrix} = 0
$$

**最少需要4对点**（8个方程）来求解8个未知数。

**SVD求解**：

构建矩阵 $A$（$2n \times 9$），求解 $A\mathbf{h} = 0$：

$$
A = \begin{bmatrix}
-x_1 & -y_1 & -1 & 0 & 0 & 0 & x_1'x_1 & x_1'y_1 & x_1' \\
0 & 0 & 0 & -x_1 & -y_1 & -1 & y_1'x_1 & y_1'y_1 & y_1' \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
-x_n & -y_n & -1 & 0 & 0 & 0 & x_n'x_n & x_n'y_n & x_n' \\
0 & 0 & 0 & -x_n & -y_n & -1 & y_n'x_n & y_n'y_n & y_n'
\end{bmatrix}
$$

对 $A$ 进行SVD分解：$A = U\Sigma V^T$，解为 $V$ 的最后一列。

### 5.5 从单应矩阵恢复R和t

给定 $H = K[r_1 \; r_2 \; t]$，可以恢复旋转和平移：

1. **计算**：$[r_1 \; r_2 \; t] = K^{-1}H$

2. **归一化**：由于尺度不定性，需要归一化：
   $$
   \lambda = \frac{1}{\|K^{-1}H_{:,1}\|} = \frac{1}{\|K^{-1}H_{:,2}\|}
   $$

3. **提取**：
   $$
   \begin{aligned}
   r_1 &= \lambda K^{-1} H_{:,1} \\
   r_2 &= \lambda K^{-1} H_{:,2} \\
   t &= \lambda K^{-1} H_{:,3}
   \end{aligned}
   $$

4. **计算第三列**：
   $$
   r_3 = r_1 \times r_2
   $$

5. **确保正交性**：由于噪声，$[r_1, r_2, r_3]$ 可能不完全正交，需要SVD校正：
   $$
   [r_1, r_2, r_3] = U V^T
   $$
   其中 $U\Sigma V^T$ 是 $[r_1, r_2, r_3]$ 的SVD分解。

---

## 六、相机矩阵分解 {#矩阵分解}

### 6.1 投影矩阵

完整的投影矩阵为：

$$
P = K[R|t] = \begin{bmatrix}
p_{11} & p_{12} & p_{13} & p_{14} \\
p_{21} & p_{22} & p_{23} & p_{24} \\
p_{31} & p_{32} & p_{33} & p_{34}
\end{bmatrix}
$$

**目标**：从 $P$ 分解出 $K$、$R$、$t$。

### 6.2 RQ分解方法

投影矩阵的前3列可以写为：

$$
M = P_{:,1:3} = KR
$$

其中 $K$ 是上三角矩阵，$R$ 是正交矩阵。

**RQ分解步骤**：

1. 将矩阵翻转
2. 进行QR分解
3. 将结果翻转回来

**Python实现**：

```python
import numpy as np
from scipy.linalg import rq

# RQ分解
K, R = rq(M)

# 确保K的对角元素为正
T = np.diag(np.sign(np.diag(K)))
K = K @ T
R = T @ R

# 确保det(R) = 1
if np.linalg.det(R) < 0:
    R = -R
    K = -K

# 归一化K
K = K / K[2, 2]
```

### 6.3 提取平移向量

$$
t = K^{-1} P_{:,3}
$$

### 6.4 SVD分解验证旋转矩阵

为了确保 $R$ 是有效的旋转矩阵（正交且行列式为1）：

```python
U, S, Vt = np.linalg.svd(R)
R_corrected = U @ Vt

if np.linalg.det(R_corrected) < 0:
    U[:, -1] *= -1
    R_corrected = U @ Vt
```

### 6.5 完整的分解流程

```python
def decompose_projection_matrix(P):
    """
    分解投影矩阵 P = K[R|t]
    
    参数:
        P: 3x4 投影矩阵
    
    返回:
        K: 3x3 内参矩阵
        R: 3x3 旋转矩阵
        t: 3x1 平移向量
        camera_center: 3x1 相机中心在世界坐标系中的位置
    """
    # 分离前3列
    M = P[:, :3]
    p4 = P[:, 3]
    
    # RQ分解
    K, R = rq(M)
    
    # 确保K的对角元素为正
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R
    
    # 确保det(R) = 1
    if np.linalg.det(R) < 0:
        R = -R
        K = -K
    
    # 归一化K
    K = K / K[2, 2]
    
    # 提取平移向量
    t = np.linalg.inv(K) @ p4
    
    # 计算相机中心
    camera_center = -R.T @ t
    
    return K, R, t, camera_center
```

---

## 七、Python完整实现与可视化 {#Python实现}

### 7.1 坐标系可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

class Arrow3D(FancyArrowPatch):
    """3D箭头类"""
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def draw_coordinate_frame(ax, origin, rotation, scale=1.0, labels=['X', 'Y', 'Z'], 
                         colors=['r', 'g', 'b'], linewidth=2):
    """绘制坐标系"""
    axes = rotation @ np.eye(3) * scale
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        arrow = Arrow3D(origin[0], origin[1], origin[2],
                       axes[0, i], axes[1, i], axes[2, i],
                       mutation_scale=20, lw=linewidth, 
                       arrowstyle='-|>', color=color)
        ax.add_artist(arrow)
        end_point = origin + axes[:, i]
        ax.text(end_point[0], end_point[1], end_point[2], 
               label, fontsize=12, weight='bold')

def draw_camera(ax, position, rotation, scale=0.5, color='blue'):
    """绘制相机模型"""
    # 相机锥体的顶点（在相机坐标系下）
    camera_points = np.array([
        [0, 0, 0],           # 光心
        [-1, -1, 2],         # 图像平面四个角
        [1, -1, 2],
        [1, 1, 2],
        [-1, 1, 2]
    ]) * scale
    
    # 转换到世界坐标系
    world_points = (rotation @ camera_points.T).T + position
    
    # 绘制相机锥体
    for i in range(1, 5):
        ax.plot([world_points[0, 0], world_points[i, 0]],
               [world_points[0, 1], world_points[i, 1]],
               [world_points[0, 2], world_points[i, 2]], 
               color='black', linewidth=1)
    
    # 图像平面的四条边
    for i in range(1, 5):
        next_i = i + 1 if i < 4 else 1
        ax.plot([world_points[i, 0], world_points[next_i, 0]],
               [world_points[i, 1], world_points[next_i, 1]],
               [world_points[i, 2], world_points[next_i, 2]], 
               color=color, linewidth=2)
    
    return world_points

def visualize_coordinate_systems():
    """可视化四个坐标系统"""
    fig = plt.figure(figsize=(18, 12))
    
    # 世界坐标系
    world_origin = np.array([0, 0, 0])
    world_rotation = np.eye(3)
    
    # 相机位置和姿态
    camera_position = np.array([3, 2, 4])
    theta_y = -np.pi/6  # 绕Y轴旋转
    theta_x = -np.pi/9  # 绕X轴旋转
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    camera_rotation = Ry @ Rx
    
    # 绘制3D场景
    ax = fig.add_subplot(221, projection='3d')
    ax.set_title('世界坐标系 → 相机坐标系', fontsize=14, weight='bold')
    
    # 绘制世界坐标系
    draw_coordinate_frame(ax, world_origin, world_rotation, scale=2.0, 
                         labels=['Xw', 'Yw', 'Zw'], 
                         colors=['red', 'green', 'blue'], linewidth=3)
    
    # 绘制相机坐标系
    draw_coordinate_frame(ax, camera_position, camera_rotation, scale=1.5,
                         labels=['Xc', 'Yc', 'Zc'], 
                         colors=['darkred', 'darkgreen', 'darkblue'], linewidth=2)
    
    # 绘制相机模型
    draw_camera(ax, camera_position, camera_rotation, scale=0.6, color='cyan')
    
    # 绘制一个3D点
    point_world = np.array([1.5, 1.5, 1.0])
    ax.scatter(*point_world, c='purple', s=200, marker='o', 
              edgecolors='black', linewidths=2)
    ax.text(point_world[0]+0.2, point_world[1]+0.2, point_world[2]+0.2, 
           'P(Xw,Yw,Zw)', fontsize=11, weight='bold', color='purple')
    
    # 投影线
    ax.plot([point_world[0], camera_position[0]],
           [point_world[1], camera_position[1]],
           [point_world[2], camera_position[2]], 
           'purple', linestyle='--', linewidth=2, alpha=0.6)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_xlim([-1, 5])
    ax.set_ylim([-1, 5])
    ax.set_zlim([-1, 5])
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('coordinate_systems_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

# 运行可视化
visualize_coordinate_systems()
```

保存为 `visualize_coordinates.py`

### 7.2 完整的相机几何计算示例

```python
# camera_geometry_demo.py
import numpy as np
from scipy.linalg import rq

class CameraGeometry:
    """相机几何计算类"""
    
    def __init__(self, fx, fy, cx, cy, width, height):
        """
        初始化相机内参
        
        参数:
            fx, fy: 焦距（像素）
            cx, cy: 主点（像素）
            width, height: 图像尺寸
        """
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        self.width = width
        self.height = height
    
    @staticmethod
    def rotation_matrix_from_euler(roll, pitch, yaw, order='xyz'):
        """从欧拉角创建旋转矩阵"""
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        if order == 'xyz':
            return Rz @ Ry @ Rx
        elif order == 'zyx':
            return Rx @ Ry @ Rz
        else:
            raise ValueError("Order must be 'xyz' or 'zyx'")
    
    @staticmethod
    def rotation_matrix_from_axis_angle(axis, theta):
        """从轴角创建旋转矩阵（罗德里格斯公式）"""
        axis = axis / np.linalg.norm(axis)  # 归一化
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R
    
    def project_point(self, point_world, R, t):
        """
        将世界坐标点投影到图像
        
        参数:
            point_world: 3D点（世界坐标系）
            R: 旋转矩阵
            t: 平移向量
        
        返回:
            pixel: 像素坐标 (u, v)
        """
        # 世界坐标 -> 相机坐标
        point_camera = R @ point_world + t
        
        # 相机坐标 -> 图像坐标（透视投影）
        if point_camera[2] <= 0:
            raise ValueError("Point is behind the camera")
        
        # 投影到像素坐标
        point_homo = self.K @ point_camera
        pixel = point_homo[:2] / point_homo[2]
        
        return pixel
    
    def compute_projection_matrix(self, R, t):
        """计算投影矩阵 P = K[R|t]"""
        return self.K @ np.hstack([R, t.reshape(-1, 1)])
    
    def decompose_projection_matrix(self, P):
        """分解投影矩阵"""
        M = P[:, :3]
        p4 = P[:, 3]
        
        # RQ分解
        K, R = rq(M)
        
        # 确保K的对角元素为正
        T = np.diag(np.sign(np.diag(K)))
        K = K @ T
        R = T @ R
        
        # 确保det(R) = 1
        if np.linalg.det(R) < 0:
            R = -R
            K = -K
        
        # 归一化K
        K = K / K[2, 2]
        
        # 提取平移向量
        t = np.linalg.inv(K) @ p4
        
        # 相机中心
        camera_center = -R.T @ t
        
        return K, R, t, camera_center
    
    def compute_homography(self, R, t, n, d):
        """
        计算平面的单应矩阵
        
        参数:
            R, t: 外参
            n: 平面法向量（世界坐标系）
            d: 平面到原点的距离
        
        返回:
            H: 3x3 单应矩阵
        """
        H = self.K @ (R - (t @ n.T) / d) @ np.linalg.inv(self.K)
        return H / H[2, 2]  # 归一化

# 示例：完整的投影流程
if __name__ == "__main__":
    # 创建相机
    camera = CameraGeometry(
        fx=800, fy=800,  # 焦距
        cx=320, cy=240,  # 主点
        width=640, height=480  # 图像尺寸
    )
    
    print("=" * 60)
    print("相机内参矩阵 K:")
    print(camera.K)
    print()
    
    # 创建外参
    roll, pitch, yaw = np.deg2rad([10, 20, 30])
    R = camera.rotation_matrix_from_euler(roll, pitch, yaw, order='zyx')
    t = np.array([1.0, 2.0, 5.0])
    
    print("旋转矩阵 R:")
    print(R)
    print(f"\ndet(R) = {np.linalg.det(R):.6f} (应该为1)")
    print(f"R^T @ R =")
    print(R.T @ R)
    print("(应该为单位矩阵)")
    print()
    
    print("平移向量 t:")
    print(t)
    print()
    
    # 旋转矩阵列向量的含义
    print("=" * 60)
    print("旋转矩阵列向量的几何意义:")
    print("第1列（世界X轴在相机系中的方向）:", R[:, 0])
    print("第2列（世界Y轴在相机系中的方向）:", R[:, 1])
    print("第3列（世界Z轴在相机系中的方向）:", R[:, 2])
    print()
    
    # 相机在世界坐标系中的位置
    camera_center = -R.T @ t
    print(f"相机中心在世界坐标系中的位置: {camera_center}")
    print()
    
    # 投影一个3D点
    point_3d = np.array([2.0, 3.0, 4.0])
    print("=" * 60)
    print(f"3D点（世界坐标）: {point_3d}")
    
    try:
        pixel = camera.project_point(point_3d, R, t)
        print(f"投影到图像（像素坐标）: ({pixel[0]:.2f}, {pixel[1]:.2f})")
    except ValueError as e:
        print(f"投影失败: {e}")
    print()
    
    # 计算并分解投影矩阵
    P = camera.compute_projection_matrix(R, t)
    print("=" * 60)
    print("投影矩阵 P = K[R|t]:")
    print(P)
    print()
    
    # 分解投影矩阵
    K_recovered, R_recovered, t_recovered, center_recovered = \
        camera.decompose_projection_matrix(P)
    
    print("分解后的内参矩阵 K:")
    print(K_recovered)
    print("\n分解后的旋转矩阵 R:")
    print(R_recovered)
    print("\n分解后的平移向量 t:")
    print(t_recovered)
    print("\n分解后的相机中心:")
    print(center_recovered)
    print()
    
    # 验证
    print("=" * 60)
    print("验证分解结果:")
    print(f"K误差: {np.linalg.norm(camera.K - K_recovered):.2e}")
    print(f"R误差: {np.linalg.norm(R - R_recovered):.2e}")
    print(f"t误差: {np.linalg.norm(t - t_recovered):.2e}")
    print()
    
    # 单应矩阵示例
    print("=" * 60)
    print("单应矩阵示例（地平面 Z=0）:")
    n = np.array([0, 0, 1])  # 平面法向量
    d = 0  # 平面距离原点
    
    # 对于Z=0平面，单应矩阵为 H = K[r1 r2 t]
    H = camera.K @ np.column_stack([R[:, 0], R[:, 1], t])
    H = H / H[2, 2]  # 归一化
    
    print("单应矩阵 H:")
    print(H)
    print()
    
    # 测试单应变换
    point_2d_world = np.array([2.0, 3.0, 1.0])  # 世界平面上的点（齐次坐标）
    point_2d_image = H @ point_2d_world
    point_2d_image = point_2d_image / point_2d_image[2]
    
    print(f"平面上的点（世界坐标）: ({point_2d_world[0]}, {point_2d_world[1]})")
    print(f"通过单应矩阵投影到图像: ({point_2d_image[0]:.2f}, {point_2d_image[1]:.2f})")
    
    # 验证：通过完整投影
    point_3d_on_plane = np.array([point_2d_world[0], point_2d_world[1], 0.0])
    pixel_verify = camera.project_point(point_3d_on_plane, R, t)
    print(f"通过完整投影验证: ({pixel_verify[0]:.2f}, {pixel_verify[1]:.2f})")
    print(f"误差: {np.linalg.norm(point_2d_image[:2] - pixel_verify):.2e}")
```

保存为 `camera_geometry_demo.py`

---

## 八、实际应用场景

### 8.1 相机标定

通过拍摄标定板（如棋盘格）的多张图像，可以估计相机内参和畸变参数。

### 8.2 3D重建

利用多视图几何和外参矩阵，可以从多张图像重建3D场景。

### 8.3 增强现实（AR）

通过相机位姿估计，可以将虚拟物体精确地叠加到真实场景中。

### 8.4 视觉SLAM

同时定位与地图构建（SLAM）需要实时估计相机的位姿（外参）。

### 8.5 图像拼接

利用单应矩阵可以将多张图像拼接成全景图。

---

## 九、总结

本文详细推导了从3D世界坐标到2D图像坐标的完整数学过程：

1. **外参矩阵 $[R|t]$**：描述相机在世界中的位置和方向（6自由度）
2. **内参矩阵 $K$**：描述相机的内部几何参数（4-5自由度）
3. **旋转矩阵 $R$**：可用欧拉角、轴角、四元数表示，其列向量表示世界坐标系的基向量在相机系中的表示
4. **单应矩阵 $H$**：描述平面到图像的投影变换（8自由度）
5. **矩阵分解**：可以从投影矩阵恢复出内外参数

完整的投影公式为：

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = 
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z
\end{bmatrix}
\begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

配套的Python代码提供了完整的实现和可视化，可以直接用于实际项目。

---

## 参考资料

1. Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision*. Cambridge University Press.
2. Szeliski, R. (2010). *Computer Vision: Algorithms and Applications*. Springer.
3. OpenCV Documentation: Camera Calibration and 3D Reconstruction

---

## 附录：运行代码

将上述Python代码保存为对应的文件名，然后运行：

```bash
# 安装依赖
pip install numpy matplotlib scipy

# 运行可视化
python visualize_coordinates.py

# 运行完整示例
python camera_geometry_demo.py
```

---

**关键词**：相机几何、坐标变换、旋转矩阵、内参矩阵、外参矩阵、单应矩阵、投影矩阵分解、计算机视觉

**博客标签**：#计算机视觉 #相机标定 #多视几何 #矩阵分解 #Python
