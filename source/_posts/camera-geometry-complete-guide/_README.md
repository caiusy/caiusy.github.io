# 相机几何完全指南 - 代码说明

本目录包含完整的相机几何推导、可视化代码和博客文章。

## 📁 文件结构

```
camera-geometry-complete-guide/
├── README.md                          # 本文件
├── requirements.txt                   # Python依赖
├── visualize_coordinates.py           # 坐标系统可视化
├── camera_geometry_demo.py            # 完整的相机几何演示
├── rotation_visualization.py          # 旋转矩阵可视化
├── homography_visualization.py        # 单应矩阵可视化
└── images/                            # 生成的图像（自动创建）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install numpy matplotlib scipy
```

### 2. 运行可视化

#### 坐标系统可视化
展示从世界坐标系到像素坐标系的完整转换流程：

```bash
python visualize_coordinates.py
```

生成图像：`coordinate_systems_complete.png`

#### 相机几何完整演示
运行所有相机几何计算的演示（投影、分解、旋转等）：

```bash
python camera_geometry_demo.py
```

输出详细的计算过程和验证结果。

#### 旋转矩阵可视化
展示旋转矩阵的各种表示方法和几何意义：

```bash
python rotation_visualization.py
```

生成图像：`rotation_matrix_visualization.png`

#### 单应矩阵可视化
展示平面到平面的投影变换：

```bash
python homography_visualization.py
```

生成图像：`homography_visualization.png`

### 3. 一键运行所有可视化

```bash
# Linux/Mac
./run_all.sh

# Windows
run_all.bat
```

## 📊 可视化说明

### 1. 坐标系统可视化（visualize_coordinates.py）

包含6个子图：

1. **世界坐标系 → 相机坐标系**：展示3D空间中的刚体变换
2. **针孔相机模型**：透视投影的几何原理
3. **图像坐标系 → 像素坐标系**：物理坐标到离散像素的转换
4. **完整的坐标转换流程**：流程图
5. **内参矩阵K**：参数说明
6. **外参矩阵[R|t]**：参数说明

### 2. 旋转矩阵可视化（rotation_visualization.py）

包含6个子图：

1. **旋转矩阵列向量的几何意义**：R的列表示世界坐标系基向量在相机系中的方向
2. **绕X轴旋转（Roll）**：Rx旋转矩阵
3. **绕Y轴旋转（Pitch）**：Ry旋转矩阵
4. **绕Z轴旋转（Yaw）**：Rz旋转矩阵
5. **欧拉角组合旋转**：ZYX顺序的组合旋转
6. **轴角表示**：Rodrigues公式

### 3. 单应矩阵可视化（homography_visualization.py）

包含6个子图：

1. **基本单应变换**：矩形到任意四边形
2. **网格变换效果**：透视变换对网格的影响
3. **单应矩阵的构成**：数学公式和物理意义
4. **仿射变换 vs 透视变换**：两种变换的对比
5. **DLT算法**：直接线性变换求解方法
6. **应用示例**：文档矫正

## 🎓 核心功能演示（camera_geometry_demo.py）

### CameraGeometry类的主要方法

#### 1. 旋转矩阵相关

```python
from camera_geometry_demo import CameraGeometry
import numpy as np

# 从欧拉角创建旋转矩阵
roll, pitch, yaw = np.deg2rad([10, 20, 30])
R = CameraGeometry.rotation_matrix_from_euler(roll, pitch, yaw, order='zyx')

# 从轴角创建旋转矩阵
axis = np.array([1, 1, 1])
theta = np.deg2rad(45)
R = CameraGeometry.rotation_matrix_from_axis_angle(axis, theta)

# 验证旋转矩阵
CameraGeometry.verify_rotation_matrix(R)

# 提取欧拉角
roll, pitch, yaw = CameraGeometry.euler_from_rotation_matrix(R)

# 提取轴角
axis, theta = CameraGeometry.axis_angle_from_rotation_matrix(R)
```

#### 2. 投影相关

```python
# 创建相机
camera = CameraGeometry(fx=800, fy=800, cx=320, cy=240, width=640, height=480)

# 定义外参
R = CameraGeometry.rotation_matrix_from_euler(0.1, 0.2, 0.3)
t = np.array([1.0, 2.0, 5.0])

# 投影单个3D点
point_3d = np.array([2.0, 3.0, 4.0])
pixel, point_camera = camera.project_point(point_3d, R, t)
print(f"像素坐标: {pixel}")

# 批量投影
points_3d = np.array([[1,1,2], [2,1,3], [1,2,3]])
pixels, points_camera = camera.project_points(points_3d, R, t)

# 反投影
depth = 3.0
pixel = (400, 300)
point_3d = camera.backproject_pixel(pixel, depth)
```

#### 3. 投影矩阵分解

```python
# 计算投影矩阵
P = camera.compute_projection_matrix(R, t)

# 分解投影矩阵
K_recovered, R_recovered, t_recovered, camera_center = camera.decompose_projection_matrix(P)

# 验证
print(f"K误差: {np.linalg.norm(camera.K - K_recovered)}")
print(f"R误差: {np.linalg.norm(R - R_recovered)}")
```

#### 4. 单应矩阵

```python
# 计算平面的单应矩阵（Z=0平面）
n = np.array([0, 0, 1])  # 法向量
d = 0  # 距离
H = camera.compute_homography(R, t, n, d)

# DLT估计单应矩阵
src_points = np.array([[0,0], [10,0], [10,10], [0,10]], dtype=float)
dst_points = np.array([[1,0], [11,1], [10,11], [0,10]], dtype=float)
H_estimated = CameraGeometry.estimate_homography_dlt(src_points, dst_points)

# 从单应矩阵恢复R和t（平面Z=0）
R_recovered, t_recovered = camera.decompose_homography(H)
```

#### 5. Look-At相机

```python
# 创建"look at"相机位姿
camera_position = np.array([5, 5, 10])
target_position = np.array([0, 0, 0])
R, t = camera.look_at(camera_position, target_position)
```

## 📐 数学公式参考

### 完整投影公式

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K[R|t] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

### 内参矩阵 K

$$
K = \begin{bmatrix}
f_x & s & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

### 外参矩阵 [R|t]

$$
[R|t] = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z
\end{bmatrix}
$$

### Rodrigues公式

$$
R = I + \sin\theta [n]_\times + (1-\cos\theta)[n]_\times^2
$$

### 单应矩阵（平面Z=0）

$$
H = K[r_1 \; r_2 \; t]
$$

## 🔍 常见问题

### Q1: 为什么旋转矩阵必须满足 R^T R = I？

**答**：旋转矩阵是正交矩阵，它保持向量的长度和角度不变。正交性保证了这一点。

### Q2: 旋转矩阵的列向量表示什么？

**答**：R的第i列表示世界坐标系的第i个基向量在相机坐标系中的表示。这是理解旋转矩阵最直观的方式。

### Q3: 内参矩阵中的 fx 和 fy 为什么可能不同？

**答**：
- fx = f / dx，fy = f / dy
- dx 和 dy 是像素的物理尺寸（mm/pixel）
- 如果像素不是正方形，fx ≠ fy

### Q4: 相机中心和平移向量有什么关系？

**答**：
- 平移向量 t：将世界坐标转换到相机坐标的平移
- 相机中心 C = -R^T t：相机在世界坐标系中的位置

### Q5: 为什么单应矩阵只有8个自由度？

**答**：单应矩阵是3×3矩阵（9个元素），但它在齐次坐标下具有尺度不定性，因此实际自由度为9-1=8。

### Q6: 什么时候可以使用单应矩阵？

**答**：当场景是平面时，或者两个视图之间是纯旋转时，可以用单应矩阵描述点的对应关系。

## 🛠️ 自定义和扩展

### 修改相机参数

```python
# 创建不同的相机
camera_wide = CameraGeometry(fx=400, fy=400, cx=320, cy=240)  # 广角
camera_tele = CameraGeometry(fx=1600, fy=1600, cx=320, cy=240)  # 长焦
```

### 添加畸变模型

当前实现是针孔相机模型，不包含畸变。要添加畸变，需要在投影后应用畸变模型：

```python
def apply_distortion(pixel, k1, k2, p1, p2):
    """径向和切向畸变"""
    x, y = pixel
    r2 = x**2 + y**2
    x_distorted = x * (1 + k1*r2 + k2*r2**2) + 2*p1*x*y + p2*(r2 + 2*x**2)
    y_distorted = y * (1 + k1*r2 + k2*r2**2) + p1*(r2 + 2*y**2) + 2*p2*x*y
    return np.array([x_distorted, y_distorted])
```

## 📚 参考资料

1. **Multiple View Geometry in Computer Vision** - Hartley & Zisserman
2. **Computer Vision: Algorithms and Applications** - Richard Szeliski
3. **An Invitation to 3D Vision** - Ma, Soatto, Kosecka, Sastry
4. **OpenCV Documentation**: Camera Calibration and 3D Reconstruction

## 💡 提示

1. **运行前**：确保安装了所有依赖
2. **生成图像**：所有可视化脚本会自动保存PNG图像
3. **修改参数**：可以直接编辑脚本中的参数来观察不同效果
4. **性能**：可视化生成可能需要几秒钟，请耐心等待

## 🐛 故障排除

### 问题1：import错误
```
ModuleNotFoundError: No module named 'numpy'
```
**解决**：运行 `pip install -r requirements.txt`

### 问题2：中文显示乱码
**解决**：检查系统是否安装了中文字体，或在代码中修改字体设置

### 问题3：SVD不收敛
```
LinAlgError: SVD did not converge
```
**解决**：检查输入数据是否有效，特别是单应矩阵估计时的对应点

## 📧 联系方式

如有问题或建议，请通过博客评论或GitHub Issue反馈。

---

**Happy Coding! 🎉**
