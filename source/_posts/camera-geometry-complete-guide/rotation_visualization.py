"""
旋转矩阵可视化
展示旋转矩阵的列向量含义、欧拉角、轴角等
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Arrow3D(FancyArrowPatch):
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


def draw_axis(ax, origin, direction, length, color, label, linewidth=2):
    """绘制单个坐标轴"""
    arrow = Arrow3D(origin[0], origin[1], origin[2],
                   direction[0]*length, direction[1]*length, direction[2]*length,
                   mutation_scale=20, lw=linewidth, 
                   arrowstyle='-|>', color=color)
    ax.add_artist(arrow)
    
    end = origin + direction * length * 1.15
    ax.text(end[0], end[1], end[2], label, 
           fontsize=12, weight='bold', color=color)


def draw_frame(ax, origin, R, scale, labels, colors, linewidth=2):
    """绘制坐标系"""
    for i, (label, color) in enumerate(zip(labels, colors)):
        direction = R[:, i]
        draw_axis(ax, origin, direction, scale, color, label, linewidth)


def visualize_rotation_matrix():
    """可视化旋转矩阵的含义"""
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('旋转矩阵详解与可视化', fontsize=16, weight='bold', y=0.98)
    
    # ==================== 图1: 旋转矩阵列向量含义 ====================
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.set_title('① 旋转矩阵列向量的几何意义\nR的列 = 世界坐标系基向量在相机系中的表示', 
                  fontsize=11, weight='bold', pad=15)
    
    # 世界坐标系（蓝色）
    origin = np.array([0, 0, 0])
    world_frame = np.eye(3)
    
    draw_frame(ax1, origin, world_frame, 1.5, 
              ['Xw', 'Yw', 'Zw'], 
              ['red', 'green', 'blue'], linewidth=3)
    
    # 旋转矩阵（绕Z轴30度）
    theta = np.deg2rad(30)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # 相机坐标系（深色）
    camera_origin = np.array([0, 0, 0])
    draw_frame(ax1, camera_origin, R, 1.5, 
              ['Xc', 'Yc', 'Zc'], 
              ['darkred', 'darkgreen', 'darkblue'], linewidth=2.5)
    
    # 标注列向量
    r1 = R[:, 0]
    r2 = R[:, 1]
    r3 = R[:, 2]
    
    # 绘制从原点到列向量端点的虚线
    ax1.plot([0, r1[0]], [0, r1[1]], [0, r1[2]], 
            'r--', linewidth=2, alpha=0.5, label='r1 (Xw在相机系中)')
    ax1.plot([0, r2[0]], [0, r2[1]], [0, r2[2]], 
            'g--', linewidth=2, alpha=0.5, label='r2 (Yw在相机系中)')
    ax1.plot([0, r3[0]], [0, r3[1]], [0, r3[2]], 
            'b--', linewidth=2, alpha=0.5, label='r3 (Zw在相机系中)')
    
    # 添加列向量的数值标注
    text_str = f'R的列向量:\nr1=[{r1[0]:.3f}, {r1[1]:.3f}, {r1[2]:.3f}]ᵀ\n'
    text_str += f'r2=[{r2[0]:.3f}, {r2[1]:.3f}, {r2[2]:.3f}]ᵀ\n'
    text_str += f'r3=[{r3[0]:.3f}, {r3[1]:.3f}, {r3[2]:.3f}]ᵀ'
    
    ax1.text2D(0.02, 0.98, text_str, transform=ax1.transAxes,
              fontsize=8, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])
    ax1.set_xlabel('X', fontsize=10, weight='bold')
    ax1.set_ylabel('Y', fontsize=10, weight='bold')
    ax1.set_zlabel('Z', fontsize=10, weight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.view_init(elev=20, azim=45)
    ax1.grid(True, alpha=0.3)
    
    # ==================== 图2: 绕X轴旋转 ====================
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.set_title('② 绕X轴旋转（Roll）', fontsize=11, weight='bold', pad=15)
    
    # 原始坐标系
    draw_frame(ax2, origin, np.eye(3), 1.2, 
              ['X', 'Y', 'Z'], 
              ['red', 'green', 'blue'], linewidth=2)
    
    # 旋转后的坐标系
    angle = np.deg2rad(45)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    
    draw_frame(ax2, origin, Rx, 1.2, 
              ["X'", "Y'", "Z'"], 
              ['darkred', 'darkgreen', 'darkblue'], linewidth=2.5)
    
    # 绘制旋转轴
    ax2.plot([0, 1.5], [0, 0], [0, 0], 'k--', linewidth=3, alpha=0.7, label='旋转轴')
    
    # 绘制旋转弧
    theta_arc = np.linspace(0, angle, 50)
    r_arc = 0.5
    y_arc = r_arc * np.cos(theta_arc)
    z_arc = r_arc * np.sin(theta_arc)
    x_arc = np.ones_like(theta_arc) * 0.3
    ax2.plot(x_arc, y_arc, z_arc, 'orange', linewidth=3, label=f'旋转{np.rad2deg(angle):.0f}°')
    
    matrix_text = f'Rx({np.rad2deg(angle):.0f}°) =\n'
    matrix_text += f'[1    0       0     ]\n'
    matrix_text += f'[0  cos𝜃  -sin𝜃]\n'
    matrix_text += f'[0  sin𝜃   cos𝜃]'
    
    ax2.text2D(0.02, 0.98, matrix_text, transform=ax2.transAxes,
              fontsize=9, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.set_zlabel('Z', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.view_init(elev=20, azim=60)
    ax2.grid(True, alpha=0.3)
    
    # ==================== 图3: 绕Y轴旋转 ====================
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.set_title('③ 绕Y轴旋转（Pitch）', fontsize=11, weight='bold', pad=15)
    
    draw_frame(ax3, origin, np.eye(3), 1.2, 
              ['X', 'Y', 'Z'], 
              ['red', 'green', 'blue'], linewidth=2)
    
    angle = np.deg2rad(45)
    Ry = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    
    draw_frame(ax3, origin, Ry, 1.2, 
              ["X'", "Y'", "Z'"], 
              ['darkred', 'darkgreen', 'darkblue'], linewidth=2.5)
    
    ax3.plot([0, 0], [0, 1.5], [0, 0], 'k--', linewidth=3, alpha=0.7, label='旋转轴')
    
    theta_arc = np.linspace(0, angle, 50)
    r_arc = 0.5
    x_arc = r_arc * np.cos(theta_arc)
    z_arc = r_arc * np.sin(theta_arc)
    y_arc = np.ones_like(theta_arc) * 0.3
    ax3.plot(x_arc, y_arc, z_arc, 'orange', linewidth=3, label=f'旋转{np.rad2deg(angle):.0f}°')
    
    matrix_text = f'Ry({np.rad2deg(angle):.0f}°) =\n'
    matrix_text += f'[ cos𝜃  0  sin𝜃]\n'
    matrix_text += f'[   0    1    0   ]\n'
    matrix_text += f'[-sin𝜃  0  cos𝜃]'
    
    ax3.text2D(0.02, 0.98, matrix_text, transform=ax3.transAxes,
              fontsize=9, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax3.set_xlim([-1.5, 1.5])
    ax3.set_ylim([-1.5, 1.5])
    ax3.set_zlim([-1.5, 1.5])
    ax3.set_xlabel('X', fontsize=10)
    ax3.set_ylabel('Y', fontsize=10)
    ax3.set_zlabel('Z', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.view_init(elev=20, azim=45)
    ax3.grid(True, alpha=0.3)
    
    # ==================== 图4: 绕Z轴旋转 ====================
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.set_title('④ 绕Z轴旋转（Yaw）', fontsize=11, weight='bold', pad=15)
    
    draw_frame(ax4, origin, np.eye(3), 1.2, 
              ['X', 'Y', 'Z'], 
              ['red', 'green', 'blue'], linewidth=2)
    
    angle = np.deg2rad(45)
    Rz = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    draw_frame(ax4, origin, Rz, 1.2, 
              ["X'", "Y'", "Z'"], 
              ['darkred', 'darkgreen', 'darkblue'], linewidth=2.5)
    
    ax4.plot([0, 0], [0, 0], [0, 1.5], 'k--', linewidth=3, alpha=0.7, label='旋转轴')
    
    theta_arc = np.linspace(0, angle, 50)
    r_arc = 0.5
    x_arc = r_arc * np.cos(theta_arc)
    y_arc = r_arc * np.sin(theta_arc)
    z_arc = np.ones_like(theta_arc) * 0.3
    ax4.plot(x_arc, y_arc, z_arc, 'orange', linewidth=3, label=f'旋转{np.rad2deg(angle):.0f}°')
    
    matrix_text = f'Rz({np.rad2deg(angle):.0f}°) =\n'
    matrix_text += f'[cos𝜃  -sin𝜃  0]\n'
    matrix_text += f'[sin𝜃   cos𝜃  0]\n'
    matrix_text += f'[  0       0     1]'
    
    ax4.text2D(0.02, 0.98, matrix_text, transform=ax4.transAxes,
              fontsize=9, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax4.set_xlim([-1.5, 1.5])
    ax4.set_ylim([-1.5, 1.5])
    ax4.set_zlim([-1.5, 1.5])
    ax4.set_xlabel('X', fontsize=10)
    ax4.set_ylabel('Y', fontsize=10)
    ax4.set_zlabel('Z', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.view_init(elev=20, azim=45)
    ax4.grid(True, alpha=0.3)
    
    # ==================== 图5: 欧拉角组合 ====================
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax5.set_title('⑤ 欧拉角组合旋转 (ZYX)', fontsize=11, weight='bold', pad=15)
    
    # 原始坐标系
    draw_frame(ax5, origin, np.eye(3), 1.0, 
              ['X', 'Y', 'Z'], 
              ['red', 'green', 'blue'], linewidth=2)
    
    # 依次应用旋转
    roll = np.deg2rad(20)
    pitch = np.deg2rad(30)
    yaw = np.deg2rad(40)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    # 显示中间步骤
    R1 = Rx
    R2 = Ry @ Rx
    R3 = Rz @ Ry @ Rx
    
    # 只显示最终结果
    draw_frame(ax5, origin, R3, 1.0, 
              ["X''", "Y''", "Z''"], 
              ['darkred', 'darkgreen', 'darkblue'], linewidth=2.5)
    
    euler_text = f'欧拉角 (ZYX顺序):\n'
    euler_text += f'Roll  (X) = {np.rad2deg(roll):.1f}°\n'
    euler_text += f'Pitch (Y) = {np.rad2deg(pitch):.1f}°\n'
    euler_text += f'Yaw   (Z) = {np.rad2deg(yaw):.1f}°\n\n'
    euler_text += f'R = Rz·Ry·Rx'
    
    ax5.text2D(0.02, 0.98, euler_text, transform=ax5.transAxes,
              fontsize=9, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax5.set_xlim([-1.5, 1.5])
    ax5.set_ylim([-1.5, 1.5])
    ax5.set_zlim([-1.5, 1.5])
    ax5.set_xlabel('X', fontsize=10)
    ax5.set_ylabel('Y', fontsize=10)
    ax5.set_zlabel('Z', fontsize=10)
    ax5.view_init(elev=20, azim=45)
    ax5.grid(True, alpha=0.3)
    
    # ==================== 图6: 轴角表示 ====================
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    ax6.set_title('⑥ 轴角表示（Rodrigues公式）', fontsize=11, weight='bold', pad=15)
    
    draw_frame(ax6, origin, np.eye(3), 1.0, 
              ['X', 'Y', 'Z'], 
              ['red', 'green', 'blue'], linewidth=2)
    
    # 任意旋转轴和角度
    axis = np.array([1, 1, 1])
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(60)
    
    # 罗德里格斯公式
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R_rodrigues = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    draw_frame(ax6, origin, R_rodrigues, 1.0, 
              ["X'", "Y'", "Z'"], 
              ['darkred', 'darkgreen', 'darkblue'], linewidth=2.5)
    
    # 绘制旋转轴
    ax6.plot([-axis[0]*1.5, axis[0]*1.5], 
            [-axis[1]*1.5, axis[1]*1.5], 
            [-axis[2]*1.5, axis[2]*1.5], 
            'k--', linewidth=3, alpha=0.7, label='旋转轴')
    
    axis_text = f'旋转轴:\n'
    axis_text += f'n = [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]ᵀ\n'
    axis_text += f'𝜃 = {np.rad2deg(theta):.1f}°\n\n'
    axis_text += f'Rodrigues公式:\n'
    axis_text += f'R = I + sin𝜃[n]× +\n'
    axis_text += f'    (1-cos𝜃)[n]×²'
    
    ax6.text2D(0.02, 0.98, axis_text, transform=ax6.transAxes,
              fontsize=8, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    ax6.set_xlim([-1.5, 1.5])
    ax6.set_ylim([-1.5, 1.5])
    ax6.set_zlim([-1.5, 1.5])
    ax6.set_xlabel('X', fontsize=10)
    ax6.set_ylabel('Y', fontsize=10)
    ax6.set_zlabel('Z', fontsize=10)
    ax6.legend(fontsize=8)
    ax6.view_init(elev=20, azim=45)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rotation_matrix_visualization.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ 旋转矩阵可视化已保存: rotation_matrix_visualization.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("生成旋转矩阵可视化...")
    print("=" * 60)
    visualize_rotation_matrix()
    print("=" * 60)
    print("完成！")
