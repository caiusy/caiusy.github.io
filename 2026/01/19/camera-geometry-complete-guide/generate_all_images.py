#!/usr/bin/env python3
"""
生成相机几何博客所需的所有示意图
"""
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("开始生成相机几何示意图...")
print("="*60)

# ==================== 图1: 针孔相机模型 ====================
print("\n[1/6] 生成针孔相机模型...")
fig, ax = plt.subplots(figsize=(14, 8))

# 光轴
ax.arrow(0, 0, 5, 0, head_width=0.15, head_length=0.2, fc='blue', ec='blue', linewidth=2.5)
ax.text(5.4, 0, 'Optical Axis (Zc)', fontsize=13, color='blue', fontweight='bold')

# 针孔（相机中心）
ax.plot(0, 0, 'ko', markersize=16)
ax.text(0.15, -0.35, 'Pinhole\n(Camera Center)', fontsize=12, ha='left', fontweight='bold')

# 成像平面
focal_length = 3
ax.plot([focal_length, focal_length], [-2, 2], 'r-', linewidth=4)
ax.text(focal_length, 2.4, 'Image Plane', fontsize=13, color='red', fontweight='bold', ha='center')

# 3D空间中的点
point_z = 6
point_y = 1.5
ax.plot(point_z, point_y, 'go', markersize=14)
ax.text(point_z+0.3, point_y+0.25, 'Point P(Xc, Yc, Zc)', fontsize=12, color='green', fontweight='bold')

# 投影线
image_y = focal_length * point_y / point_z
ax.plot([0, point_z], [0, point_y], 'g--', linewidth=2, alpha=0.7)
ax.plot(focal_length, image_y, 'ro', markersize=12)
ax.text(focal_length+0.35, image_y, "p'(x, y)", fontsize=12, color='red', fontweight='bold')

# 标注距离
ax.plot([0, focal_length], [-0.05, -0.05], 'b-', linewidth=2.5)
ax.text(focal_length/2, -0.3, f'f = {focal_length} mm', fontsize=12, ha='center', 
        color='blue', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# 标注Yc
ax.plot([point_z, point_z], [0, point_y], 'g:', linewidth=2)
ax.text(point_z+0.15, point_y/2, f'Yc = {point_y}', fontsize=11, color='green', 
        rotation=90, va='center')

# 标注y
ax.plot([focal_length, focal_length], [0, image_y], 'r:', linewidth=2)
ax.text(focal_length-0.3, image_y/2, f'y = {image_y:.2f}', fontsize=11, color='red', 
        rotation=90, va='center')

ax.set_xlim([-0.5, 7.5])
ax.set_ylim([-2.5, 2.8])
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlabel('Z axis (mm)', fontsize=13, fontweight='bold')
ax.set_ylabel('Y axis (mm)', fontsize=13, fontweight='bold')
ax.set_title('Pinhole Camera Model - Perspective Projection\n' + 
             r'$\frac{y}{f} = \frac{Y_c}{Z_c}  \Rightarrow  y = f \cdot \frac{Y_c}{Z_c}$', 
             fontsize=15, fontweight='bold', pad=20)

plt.savefig('pinhole_camera_model.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("✓ 已生成: pinhole_camera_model.png")

# ==================== 图2: 四个坐标系统 ====================
print("\n[2/6] 生成四个坐标系统...")
fig = plt.figure(figsize=(16, 12))

# 世界坐标系
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.quiver(0, 0, 0, 1.2, 0, 0, color='red', arrow_length_ratio=0.15, linewidth=3.5, label='Xw')
ax1.quiver(0, 0, 0, 0, 1.2, 0, color='green', arrow_length_ratio=0.15, linewidth=3.5, label='Yw')
ax1.quiver(0, 0, 0, 0, 0, 1.2, color='blue', arrow_length_ratio=0.15, linewidth=3.5, label='Zw')
ax1.set_xlabel('X', fontsize=12, fontweight='bold')
ax1.set_ylabel('Y', fontsize=12, fontweight='bold')
ax1.set_zlabel('Z', fontsize=12, fontweight='bold')
ax1.set_title('World Coordinate System\n(Xw, Yw, Zw)', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='upper right')
ax1.set_xlim([-0.5, 1.5])
ax1.set_ylim([-0.5, 1.5])
ax1.set_zlim([-0.5, 1.5])

# 相机坐标系
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.quiver(0, 0, 0, 1, 0, 0, color='red', arrow_length_ratio=0.15, linewidth=3.5, label='Xc')
ax2.quiver(0, 0, 0, 0, 1, 0, color='green', arrow_length_ratio=0.15, linewidth=3.5, label='Yc')
ax2.quiver(0, 0, 0, 0, 0, 1.5, color='blue', arrow_length_ratio=0.1, linewidth=3.5, label='Zc (Optical)')
# 绘制简化相机
camera_size = 0.3
vertices = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
                      [-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2]]) * camera_size
for i in range(4):
    ax2.plot3D(*zip(vertices[i], vertices[(i+1)%4]), 'gray', alpha=0.6, linewidth=1.5)
    ax2.plot3D(*zip(vertices[i+4], vertices[(i+1)%4+4]), 'gray', alpha=0.6, linewidth=1.5)
    ax2.plot3D(*zip(vertices[i], vertices[i+4]), 'gray', alpha=0.6, linewidth=1.5)
ax2.set_xlabel('X', fontsize=12, fontweight='bold')
ax2.set_ylabel('Y', fontsize=12, fontweight='bold')
ax2.set_zlabel('Z', fontsize=12, fontweight='bold')
ax2.set_title('Camera Coordinate System\n(Xc, Yc, Zc)', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=11, loc='upper right')
ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
ax2.set_zlim([-0.5, 2])

# 图像坐标系
ax3 = fig.add_subplot(2, 2, 3)
ax3.arrow(0, 0, 1.2, 0, head_width=0.08, head_length=0.12, fc='red', ec='red', linewidth=2.5)
ax3.arrow(0, 0, 0, -1.2, head_width=0.08, head_length=0.12, fc='green', ec='green', linewidth=2.5)
ax3.text(1.35, 0, 'x (mm)', fontsize=13, color='red', fontweight='bold', va='center')
ax3.text(0, -1.35, 'y (mm)', fontsize=13, color='green', fontweight='bold', ha='center')
ax3.plot(0, 0, 'ko', markersize=12)
ax3.text(0.08, 0.08, 'O (Principal Point)', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.4, linestyle='--')
ax3.set_xlim([-0.6, 1.6])
ax3.set_ylim([-1.6, 0.6])
ax3.set_aspect('equal')
ax3.set_title('Image Coordinate System\n(x, y) - Physical Units (mm)', 
              fontsize=14, fontweight='bold', pad=15)

# 像素坐标系
ax4 = fig.add_subplot(2, 2, 4)
ax4.arrow(0, 0, 120, 0, head_width=6, head_length=12, fc='red', ec='red', linewidth=2.5)
ax4.arrow(0, 0, 0, 120, head_width=6, head_length=12, fc='green', ec='green', linewidth=2.5)
ax4.text(135, 0, 'u (pixels)', fontsize=13, color='red', fontweight='bold', va='center')
ax4.text(0, 135, 'v (pixels)', fontsize=13, color='green', fontweight='bold', ha='center')
# 像素网格
for i in range(0, 121, 20):
    ax4.axhline(i, color='gray', alpha=0.25, linewidth=0.8)
    ax4.axvline(i, color='gray', alpha=0.25, linewidth=0.8)
ax4.plot(0, 0, 'ro', markersize=10)
ax4.text(6, 6, 'Origin (0,0)\nTop-left', fontsize=10, fontweight='bold')
ax4.set_xlim([-10, 145])
ax4.set_ylim([145, -10])
ax4.set_aspect('equal')
ax4.set_title('Pixel Coordinate System\n(u, v) - Discrete Pixels', 
              fontsize=14, fontweight='bold', pad=15)

plt.suptitle('Four Coordinate Systems in Camera Geometry', 
             fontsize=18, fontweight='bold', y=0.98)
plt.savefig('coordinate_systems_complete.png', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("✓ 已生成: coordinate_systems_complete.png")

# ==================== 图3: 旋转矩阵 ====================
print("\n[3/6] 生成旋转矩阵可视化...")
fig = plt.figure(figsize=(18, 6))

def plot_rotation(ax, angle, axis_name, rotation_matrix):
    """绘制单个旋转"""
    # 原始坐标系（浅色）
    ax.quiver(0, 0, 0, 1, 0, 0, color='red', alpha=0.25, arrow_length_ratio=0.15, linewidth=2)
    ax.quiver(0, 0, 0, 0, 1, 0, color='green', alpha=0.25, arrow_length_ratio=0.15, linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, 1, color='blue', alpha=0.25, arrow_length_ratio=0.15, linewidth=2)
    
    # 旋转后的坐标系（深色）
    x_new = rotation_matrix @ np.array([1, 0, 0])
    y_new = rotation_matrix @ np.array([0, 1, 0])
    z_new = rotation_matrix @ np.array([0, 0, 1])
    
    ax.quiver(0, 0, 0, x_new[0], x_new[1], x_new[2], color='darkred', 
              arrow_length_ratio=0.15, linewidth=3.5, label="X' (rotated)")
    ax.quiver(0, 0, 0, y_new[0], y_new[1], y_new[2], color='darkgreen', 
              arrow_length_ratio=0.15, linewidth=3.5, label="Y' (rotated)")
    ax.quiver(0, 0, 0, z_new[0], z_new[1], z_new[2], color='darkblue', 
              arrow_length_ratio=0.15, linewidth=3.5, label="Z' (rotated)")
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title(f'Rotation around {axis_name}-axis\nθ = {angle}°', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])

# 绕X轴旋转
angle_x = 45
rad = np.radians(angle_x)
Rx = np.array([[1, 0, 0],
               [0, np.cos(rad), -np.sin(rad)],
               [0, np.sin(rad), np.cos(rad)]])
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
plot_rotation(ax1, angle_x, 'X', Rx)

# 绕Y轴旋转
angle_y = 45
rad = np.radians(angle_y)
Ry = np.array([[np.cos(rad), 0, np.sin(rad)],
               [0, 1, 0],
               [-np.sin(rad), 0, np.cos(rad)]])
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
plot_rotation(ax2, angle_y, 'Y', Ry)

# 绕Z轴旋转
angle_z = 45
rad = np.radians(angle_z)
Rz = np.array([[np.cos(rad), -np.sin(rad), 0],
               [np.sin(rad), np.cos(rad), 0],
               [0, 0, 1]])
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
plot_rotation(ax3, angle_z, 'Z', Rz)

plt.suptitle('Basic Rotation Matrices - Euler Angles', fontsize=18, fontweight='bold', y=0.98)
plt.savefig('rotation_matrices_visualization.png', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("✓ 已生成: rotation_matrices_visualization.png")

# ==================== 图4: 完整投影流程 ====================
print("\n[4/6] 生成完整投影流程...")
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# 世界坐标系
ax.quiver(0, 0, 0, 1.5, 0, 0, color='red', alpha=0.5, arrow_length_ratio=0.1, linewidth=2.5)
ax.quiver(0, 0, 0, 0, 1.5, 0, color='green', alpha=0.5, arrow_length_ratio=0.1, linewidth=2.5)
ax.quiver(0, 0, 0, 0, 0, 1.5, color='blue', alpha=0.5, arrow_length_ratio=0.1, linewidth=2.5)
ax.text(1.7, 0, 0, 'Xw', fontsize=12, color='red', fontweight='bold')
ax.text(0, 1.7, 0, 'Yw', fontsize=12, color='green', fontweight='bold')
ax.text(0, 0, 1.7, 'Zw', fontsize=12, color='blue', fontweight='bold')

# 相机位置
camera_pos = np.array([3, 3, 2])
ax.scatter(*camera_pos, color='black', s=250, marker='^', edgecolors='red', linewidths=2)
ax.text(camera_pos[0]+0.25, camera_pos[1], camera_pos[2]+0.3, 
        'Camera', fontsize=12, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 3D点
point_3d = np.array([1.5, 1.5, 0.5])
ax.scatter(*point_3d, color='orange', s=200, marker='o', edgecolors='black', linewidths=2)
ax.text(point_3d[0]+0.15, point_3d[1], point_3d[2]+0.25, 
        'P(Xw, Yw, Zw)', fontsize=12, color='orange', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 投影线
ax.plot([camera_pos[0], point_3d[0]], 
        [camera_pos[1], point_3d[1]], 
        [camera_pos[2], point_3d[2]], 
        'k--', linewidth=2.5, alpha=0.7, label='Projection Ray')

# 成像平面标注
camera_direction = np.array([-1, -1, 0])
camera_direction = camera_direction / np.linalg.norm(camera_direction)
image_plane_center = camera_pos + 0.6 * camera_direction
ax.text(image_plane_center[0], image_plane_center[1], image_plane_center[2]-0.4,
        'Image\nPlane', fontsize=11, ha='center', color='purple', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

ax.set_xlabel('X (world)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (world)', fontsize=12, fontweight='bold')
ax.set_zlabel('Z (world)', fontsize=12, fontweight='bold')
ax.set_title('Complete Projection Pipeline: World → Camera → Image\n' + 
             r'$P_{pixel} = K \cdot [R|t] \cdot P_{world}$',
             fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.set_xlim([-0.5, 4])
ax.set_ylim([-0.5, 4])
ax.set_zlim([0, 3])

plt.savefig('projection_pipeline.png', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("✓ 已生成: projection_pipeline.png")

# ==================== 图5: 单应矩阵变换 ====================
print("\n[5/6] 生成单应矩阵变换...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 原始平面
ax1 = axes[0]
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
ax1.plot(square[:, 0], square[:, 1], 'b-', linewidth=4)
ax1.fill(square[:, 0], square[:, 1], alpha=0.3, color='blue')
ax1.plot(square[:-1, 0], square[:-1, 1], 'bo', markersize=12)
ax1.set_xlim([-0.5, 1.5])
ax1.set_ylim([-0.5, 1.5])
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.set_title('Original Plane (Source)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('x', fontsize=13, fontweight='bold')
ax1.set_ylabel('y', fontsize=13, fontweight='bold')
# 点标签
for i, pt in enumerate(square[:-1]):
    ax1.text(pt[0]-0.1, pt[1]+0.1, f'P{i+1}', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 变换后的平面
ax2 = axes[1]
H = np.array([[0.9, 0.3, 0.1],
              [0.1, 0.8, 0.2],
              [0.05, 0.1, 1.0]])
transformed_square = []
for point in square[:-1]:
    p_homog = np.array([point[0], point[1], 1])
    p_transformed = H @ p_homog
    p_transformed = p_transformed / p_transformed[2]
    transformed_square.append([p_transformed[0], p_transformed[1]])
transformed_square.append(transformed_square[0])
transformed_square = np.array(transformed_square)

ax2.plot(transformed_square[:, 0], transformed_square[:, 1], 'r-', linewidth=4)
ax2.fill(transformed_square[:, 0], transformed_square[:, 1], alpha=0.3, color='red')
ax2.plot(transformed_square[:-1, 0], transformed_square[:-1, 1], 'ro', markersize=12)
ax2.set_xlim([-0.5, 1.5])
ax2.set_ylim([-0.5, 1.5])
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.4, linestyle='--')
ax2.set_title('Transformed Plane (Target)\nby Homography H', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel("x'", fontsize=13, fontweight='bold')
ax2.set_ylabel("y'", fontsize=13, fontweight='bold')
# 点标签
for i, pt in enumerate(transformed_square[:-1]):
    ax2.text(pt[0]+0.06, pt[1]+0.06, f"P'{i+1}", fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle(r"Homography Transformation: $p' \sim H \cdot p$", 
             fontsize=16, fontweight='bold')
plt.savefig('homography_transformation.png', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("✓ 已生成: homography_transformation.png")

# ==================== 图6: 外参矩阵 ====================
print("\n[6/6] 生成外参矩阵示意图...")
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 世界坐标系
origin_world = np.array([0, 0, 0])
ax.quiver(*origin_world, 1.8, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=3.5, alpha=0.7)
ax.quiver(*origin_world, 0, 1.8, 0, color='green', arrow_length_ratio=0.1, linewidth=3.5, alpha=0.7)
ax.quiver(*origin_world, 0, 0, 1.8, color='blue', arrow_length_ratio=0.1, linewidth=3.5, alpha=0.7)
ax.text(2.0, 0, 0, 'Xw', fontsize=14, color='red', fontweight='bold')
ax.text(0, 2.0, 0, 'Yw', fontsize=14, color='green', fontweight='bold')
ax.text(0, 0, 2.0, 'Zw', fontsize=14, color='blue', fontweight='bold')
ax.scatter(*origin_world, color='red', s=150, marker='o', edgecolors='black', linewidths=2)
ax.text(origin_world[0]-0.4, origin_world[1], origin_world[2]-0.4, 
        'World\nOrigin', fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 相机坐标系
t = np.array([3, 2, 1])  # 平移向量
angle = 45
rad = np.radians(angle)
R = np.array([[np.cos(rad), -np.sin(rad), 0],
              [np.sin(rad), np.cos(rad), 0],
              [0, 0, 1]])

camera_axes = R @ np.eye(3) * 1.5
ax.quiver(*t, *camera_axes[:, 0], color='darkred', arrow_length_ratio=0.1, linewidth=3.5)
ax.quiver(*t, *camera_axes[:, 1], color='darkgreen', arrow_length_ratio=0.1, linewidth=3.5)
ax.quiver(*t, *camera_axes[:, 2], color='darkblue', arrow_length_ratio=0.1, linewidth=3.5)
ax.text(t[0]+camera_axes[0,0]+0.3, t[1]+camera_axes[1,0], t[2]+camera_axes[2,0], 
        'Xc', fontsize=14, color='darkred', fontweight='bold')
ax.text(t[0]+camera_axes[0,1], t[1]+camera_axes[1,1]+0.3, t[2]+camera_axes[2,1], 
        'Yc', fontsize=14, color='darkgreen', fontweight='bold')
ax.text(t[0]+camera_axes[0,2], t[1]+camera_axes[1,2], t[2]+camera_axes[2,2]+0.4, 
        'Zc', fontsize=14, color='darkblue', fontweight='bold')
ax.scatter(*t, color='darkblue', s=150, marker='^', edgecolors='black', linewidths=2)
ax.text(t[0]+0.3, t[1], t[2]+0.4, 'Camera\nCenter', fontsize=11, ha='left', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# 平移向量
ax.plot([origin_world[0], t[0]], [origin_world[1], t[1]], [origin_world[2], t[2]], 
        'k--', linewidth=3, alpha=0.7)
ax.text(t[0]/2, t[1]/2-0.4, t[2]/2, r'$\mathbf{t}$' + '\n(translation)', fontsize=12, 
        ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# 旋转标注
ax.text(0.5, 0.5, 2.8, r'$\mathbf{R}$' + ' (rotation)\n45° around Z-axis', fontsize=12, 
        ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

ax.set_xlabel('X', fontsize=13, fontweight='bold')
ax.set_ylabel('Y', fontsize=13, fontweight='bold')
ax.set_zlabel('Z', fontsize=13, fontweight='bold')
ax.set_title(r'Extrinsic Parameters: $[\mathbf{R}|\mathbf{t}]$' + '\nRotation R and Translation t', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlim([-1, 5])
ax.set_ylim([-1, 4.5])
ax.set_zlim([0, 3.5])

plt.savefig('extrinsic_parameters.png', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("✓ 已生成: extrinsic_parameters.png")

print("\n" + "="*60)
print("🎉 所有图片生成完成！")
print("="*60)
print("\n生成的图片列表:")
print("  1. pinhole_camera_model.png         - 针孔相机模型")
print("  2. coordinate_systems_complete.png  - 四个坐标系统")
print("  3. rotation_matrices_visualization.png - 旋转矩阵")
print("  4. projection_pipeline.png          - 完整投影流程")
print("  5. homography_transformation.png    - 单应矩阵变换")
print("  6. extrinsic_parameters.png         - 外参矩阵")
print("="*60)
