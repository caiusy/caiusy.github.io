import matplotlib
matplotlib.use("Agg")
"""
相机几何可视化 - 坐标系统转换
展示从世界坐标系到像素坐标系的完整转换过程
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
                         colors=['r', 'g', 'b'], linewidth=2, label_offset=1.1):
    """
    绘制坐标系
    
    参数:
        ax: 3D axes对象
        origin: 原点位置
        rotation: 旋转矩阵
        scale: 坐标轴长度
        labels: 坐标轴标签
        colors: 坐标轴颜色
        linewidth: 线宽
        label_offset: 标签偏移量
    """
    axes = rotation @ np.eye(3) * scale
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        arrow = Arrow3D(origin[0], origin[1], origin[2],
                       axes[0, i], axes[1, i], axes[2, i],
                       mutation_scale=20, lw=linewidth, 
                       arrowstyle='-|>', color=color)
        ax.add_artist(arrow)
        end_point = origin + axes[:, i] * label_offset
        ax.text(end_point[0], end_point[1], end_point[2], 
               label, fontsize=11, weight='bold', color=color)


def draw_camera(ax, position, rotation, scale=0.5, color='blue'):
    """
    绘制相机模型（锥体）
    
    参数:
        ax: 3D axes对象
        position: 相机位置
        rotation: 相机旋转矩阵
        scale: 相机大小
        color: 图像平面颜色
    """
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
    
    # 绘制从光心到四个角的线
    for i in range(1, 5):
        ax.plot([world_points[0, 0], world_points[i, 0]],
               [world_points[0, 1], world_points[i, 1]],
               [world_points[0, 2], world_points[i, 2]], 
               color='black', linewidth=1, alpha=0.6)
    
    # 绘制图像平面的四条边
    for i in range(1, 5):
        next_i = i + 1 if i < 4 else 1
        ax.plot([world_points[i, 0], world_points[next_i, 0]],
               [world_points[i, 1], world_points[next_i, 1]],
               [world_points[i, 2], world_points[next_i, 2]], 
               color=color, linewidth=2.5)
    
    # 填充图像平面
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts = [world_points[1:5]]
    poly = Poly3DCollection(verts, alpha=0.3, facecolor=color, edgecolor=color)
    ax.add_collection3d(poly)
    
    # 标记光心
    ax.scatter(*position, c='red', s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
    
    return world_points


def visualize_all_coordinate_systems():
    """可视化完整的坐标系统转换过程"""
    
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('相机几何：从世界坐标到像素坐标的完整转换', 
                 fontsize=16, weight='bold', y=0.98)
    
    # ==================== 图1: 世界坐标系 → 相机坐标系 ====================
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.set_title('① 世界坐标系 → 相机坐标系\n外参变换: [R|t]', 
                  fontsize=12, weight='bold', pad=15)
    
    # 世界坐标系
    world_origin = np.array([0, 0, 0])
    world_rotation = np.eye(3)
    
    # 相机位置和姿态
    camera_position = np.array([3, 2, 4])
    theta_y = -np.pi/6
    theta_x = -np.pi/9
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    camera_rotation = Ry @ Rx
    
    # 绘制世界坐标系
    draw_coordinate_frame(ax1, world_origin, world_rotation, scale=2.0, 
                         labels=['Xw', 'Yw', 'Zw'], 
                         colors=['red', 'green', 'blue'], linewidth=3)
    
    # 绘制相机坐标系
    draw_coordinate_frame(ax1, camera_position, camera_rotation, scale=1.5,
                         labels=['Xc', 'Yc', 'Zc'], 
                         colors=['darkred', 'darkgreen', 'darkblue'], linewidth=2)
    
    # 绘制相机模型
    draw_camera(ax1, camera_position, camera_rotation, scale=0.6, color='cyan')
    
    # 绘制一个3D点
    point_world = np.array([1.5, 1.5, 1.0])
    ax1.scatter(*point_world, c='purple', s=200, marker='o', 
              edgecolors='black', linewidths=2, zorder=10)
    ax1.text(point_world[0]+0.2, point_world[1]+0.2, point_world[2]+0.3, 
           'P(Xw,Yw,Zw)', fontsize=10, weight='bold', color='purple',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 投影线
    ax1.plot([point_world[0], camera_position[0]],
           [point_world[1], camera_position[1]],
           [point_world[2], camera_position[2]], 
           'purple', linestyle='--', linewidth=2, alpha=0.5)
    
    ax1.set_xlabel('X', fontsize=10, weight='bold')
    ax1.set_ylabel('Y', fontsize=10, weight='bold')
    ax1.set_zlabel('Z', fontsize=10, weight='bold')
    ax1.set_xlim([-1, 5])
    ax1.set_ylim([-1, 5])
    ax1.set_zlim([-1, 5])
    ax1.view_init(elev=20, azim=45)
    ax1.grid(True, alpha=0.3)
    
    # ==================== 图2: 针孔相机模型（侧视图）====================
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title('② 针孔相机模型（透视投影）\n相机坐标系 → 图像坐标系', 
                  fontsize=12, weight='bold', pad=15)
    
    # 绘制光轴
    ax2.arrow(0, 0, 5, 0, head_width=0.15, head_length=0.2, 
             fc='black', ec='black', linewidth=2)
    ax2.text(5.3, 0, 'Zc (光轴)', fontsize=11, weight='bold', va='center')
    
    # 光心
    ax2.scatter(0, 0, s=250, c='red', marker='o', edgecolors='black', 
               linewidths=2, zorder=5)
    ax2.text(-0.3, -0.4, 'O\n(光心)', fontsize=10, weight='bold', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # 3D点
    point_z = 4
    point_y = 2
    ax2.scatter(point_z, point_y, s=180, c='purple', marker='o', 
               edgecolors='black', linewidths=2, zorder=5)
    ax2.text(point_z+0.3, point_y+0.3, 'P(Xc,Yc,Zc)', fontsize=10, 
            weight='bold', color='purple',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    
    # 焦距和成像平面
    focal_length = 1.5
    ax2.plot([focal_length, focal_length], [-3, 3], 'b-', linewidth=4, 
            label='成像平面', alpha=0.8)
    ax2.text(focal_length, -3.5, f'焦距 f={focal_length}', fontsize=10, 
            weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 投影点
    image_y = focal_length * point_y / point_z
    ax2.scatter(focal_length, image_y, s=180, c='orange', marker='s', 
               edgecolors='black', linewidths=2, zorder=5)
    ax2.text(focal_length+0.4, image_y, "p'(x,y)", fontsize=10, 
            weight='bold', color='orange',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 投影线
    ax2.plot([0, point_z], [0, point_y], 'purple', linestyle='--', 
            linewidth=2, alpha=0.6, label='投影光线')
    ax2.plot([focal_length, point_z], [image_y, point_y], 'gray', 
            linestyle=':', linewidth=1.5)
    
    # 标注Zc
    ax2.plot([point_z, point_z], [0, 0.15], 'k-', linewidth=1.5)
    ax2.text(point_z, -0.5, f'Zc={point_z}', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
    
    # 标注Yc
    ax2.plot([0, 0.15], [point_y, point_y], 'k-', linewidth=1.5)
    ax2.text(-0.4, point_y, f'Yc={point_y}', fontsize=9, ha='right', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
    
    # 标注y（图像坐标）
    ax2.annotate('', xy=(focal_length-0.15, 0), xytext=(focal_length-0.15, image_y),
                arrowprops=dict(arrowstyle='<->', color='orange', lw=2.5))
    ax2.text(focal_length-0.45, image_y/2, f'y={image_y:.2f}mm', fontsize=9, 
            ha='right', color='orange', weight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    # 相似三角形公式
    formula_text = r'$\frac{y}{f} = \frac{Y_c}{Z_c}$'
    ax2.text(2.5, -2.5, formula_text, fontsize=13, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax2.set_xlim([-0.8, 5.5])
    ax2.set_ylim([-4, 4])
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.set_xlabel('Z 轴 (深度方向)', fontsize=10, weight='bold')
    ax2.set_ylabel('Y 轴', fontsize=10, weight='bold')
    
    # ==================== 图3: 图像坐标系 → 像素坐标系 ====================
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title('③ 图像坐标系 → 像素坐标系\n内参矩阵 K', 
                  fontsize=12, weight='bold', pad=15)
    
    # 图像平面尺寸（毫米）
    image_width_mm = 10
    image_height_mm = 8
    
    # 像素分辨率
    image_width_px = 640
    image_height_px = 480
    
    # 绘制图像平面（图像坐标系，原点在中心）
    ax3.add_patch(mpatches.Rectangle((-image_width_mm/2, -image_height_mm/2), 
                                     image_width_mm, image_height_mm,
                                     fill=False, edgecolor='blue', linewidth=3))
    
    # 图像坐标系原点（中心）
    ax3.scatter(0, 0, s=200, c='blue', marker='o', edgecolors='black', 
               linewidths=2, zorder=5)
    ax3.text(0.4, -0.6, 'O (图像中心)', fontsize=10, weight='bold', color='blue',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 图像坐标系坐标轴
    ax3.arrow(0, 0, 4, 0, head_width=0.3, head_length=0.3, 
             fc='blue', ec='blue', linewidth=2.5, alpha=0.8)
    ax3.text(4.5, 0, 'x (mm)', fontsize=11, weight='bold', color='blue')
    ax3.arrow(0, 0, 0, 3, head_width=0.3, head_length=0.3, 
             fc='blue', ec='blue', linewidth=2.5, alpha=0.8)
    ax3.text(0, 3.5, 'y (mm)', fontsize=11, weight='bold', color='blue')
    
    # 像素坐标系原点（左上角）
    pixel_origin = (-image_width_mm/2, image_height_mm/2)
    ax3.scatter(*pixel_origin, s=200, c='red', marker='s', 
               edgecolors='black', linewidths=2, zorder=5)
    ax3.text(pixel_origin[0]-0.6, pixel_origin[1]+0.5, "O' (0,0)", 
            fontsize=10, weight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # 像素坐标系坐标轴
    ax3.arrow(pixel_origin[0], pixel_origin[1], 4, 0, head_width=0.3, 
             head_length=0.3, fc='red', ec='red', linewidth=2.5, 
             linestyle='--', alpha=0.8)
    ax3.text(pixel_origin[0]+4.5, pixel_origin[1], 'u (px)', fontsize=11, 
            weight='bold', color='red')
    ax3.arrow(pixel_origin[0], pixel_origin[1], 0, -3, head_width=0.3, 
             head_length=0.3, fc='red', ec='red', linewidth=2.5, 
             linestyle='--', alpha=0.8)
    ax3.text(pixel_origin[0], pixel_origin[1]-3.5, 'v (px)', fontsize=11, 
            weight='bold', color='red')
    
    # 一个示例点
    point_image = np.array([2, 1.5])
    ax3.scatter(*point_image, s=180, c='green', marker='o', 
               edgecolors='black', linewidths=2, zorder=5)
    ax3.text(point_image[0]+0.4, point_image[1]+0.4, 'p(x,y)', fontsize=10, 
            weight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 标注转换
    ax3.annotate('', xy=point_image, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2, linestyle=':'))
    ax3.annotate('', xy=point_image, xytext=pixel_origin,
                arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle=':'))
    
    # 添加像素网格
    grid_step = 2
    for i in range(-5, 6, grid_step):
        ax3.axvline(x=i, color='gray', alpha=0.15, linewidth=0.5)
        ax3.axhline(y=i, color='gray', alpha=0.15, linewidth=0.5)
    
    ax3.set_xlim([-7, 7])
    ax3.set_ylim([-6, 6])
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('X (mm)', fontsize=10, weight='bold')
    ax3.set_ylabel('Y (mm)', fontsize=10, weight='bold')
    
    # 添加转换公式
    dx = image_width_mm / image_width_px
    dy = image_height_mm / image_height_px
    formula_text = f'u = x/dx + cx\nv = y/dy + cy\n'
    formula_text += f'dx={dx:.4f}mm/px\ndy={dy:.4f}mm/px'
    ax3.text(-6.5, -5, formula_text, fontsize=9, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ==================== 图4: 完整投影流程图 ====================
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    ax4.set_title('④ 完整的坐标转换流程', fontsize=12, weight='bold', pad=15)
    
    boxes = [
        {'name': '世界坐标系\n(Xw, Yw, Zw, 1)', 'pos': (0.05, 0.78), 
         'size': (0.35, 0.15), 'color': '#FFE6E6'},
        {'name': '相机坐标系\n(Xc, Yc, Zc)', 'pos': (0.05, 0.52), 
         'size': (0.35, 0.15), 'color': '#E6F3FF'},
        {'name': '归一化平面\n(Xc/Zc, Yc/Zc, 1)', 'pos': (0.05, 0.26), 
         'size': (0.35, 0.15), 'color': '#E6FFE6'},
        {'name': '像素坐标系\n(u, v, 1)', 'pos': (0.05, 0.0), 
         'size': (0.35, 0.15), 'color': '#FFF9E6'},
    ]
    
    for box in boxes:
        fancy_box = FancyBboxPatch(box['pos'], box['size'][0], box['size'][1],
                                   boxstyle="round,pad=0.015", 
                                   edgecolor='black', facecolor=box['color'],
                                   linewidth=2.5)
        ax4.add_patch(fancy_box)
        ax4.text(box['pos'][0] + box['size'][0]/2, 
                box['pos'][1] + box['size'][1]/2, 
                box['name'], ha='center', va='center', 
                fontsize=10, weight='bold')
    
    # 箭头和公式
    arrows_formulas = [
        {'from': (0.225, 0.78), 'to': (0.225, 0.67), 
         'label': r'$\begin{bmatrix}Xc\\Yc\\Zc\end{bmatrix} = R\begin{bmatrix}Xw\\Yw\\Zw\end{bmatrix} + t$',
         'pos': (0.5, 0.725)},
        {'from': (0.225, 0.52), 'to': (0.225, 0.41), 
         'label': r'透视投影: $\div Z_c$',
         'pos': (0.5, 0.465)},
        {'from': (0.225, 0.26), 'to': (0.225, 0.15), 
         'label': r'$\begin{bmatrix}u\\v\\1\end{bmatrix} = K\begin{bmatrix}Xc/Zc\\Yc/Zc\\1\end{bmatrix}$',
         'pos': (0.5, 0.205)},
    ]
    
    for item in arrows_formulas:
        arr = FancyArrowPatch(item['from'], item['to'],
                             arrowstyle='->', mutation_scale=25, 
                             linewidth=3, color='darkblue')
        ax4.add_patch(arr)
        ax4.text(item['pos'][0], item['pos'][1], item['label'],
                ha='left', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 总公式
    ax4.text(0.5, 0.95, '完整投影公式:', fontsize=11, weight='bold', ha='center')
    formula = r'$s\begin{bmatrix}u\\v\\1\end{bmatrix} = K[R|t]\begin{bmatrix}X_w\\Y_w\\Z_w\\1\end{bmatrix}$'
    ax4.text(0.5, 0.88, formula, fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    # ==================== 图5: 内参矩阵K ====================
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    ax5.set_title('⑤ 内参矩阵 K（Camera Intrinsics）', 
                  fontsize=12, weight='bold', pad=15)
    
    K_text = r'''
内参矩阵 K (3×3):

$K = \begin{bmatrix}
f_x & s & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}$

参数含义:
• fx, fy: 焦距(像素单位)
  - fx = f / dx
  - fy = f / dy
  
• cx, cy: 主点坐标(像素)
  - 图像中心在像素坐标系中的位置
  - 通常 ≈ (width/2, height/2)
  
• s: 倾斜系数
  - 理想情况下 s = 0
  - 表示像素坐标轴的非正交性

自由度: 4-5个
特点: 上三角矩阵
'''
    
    ax5.text(0.1, 0.95, K_text, fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    
    # ==================== 图6: 外参矩阵 [R|t] ====================
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    ax6.set_title('⑥ 外参矩阵 [R|t]（Camera Extrinsics）', 
                  fontsize=12, weight='bold', pad=15)
    
    RT_text = r'''
外参矩阵 [R|t] (3×4):

$[R|t] = \begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z
\end{bmatrix}$

参数含义:
• R (3×3): 旋转矩阵
  - 描述相机坐标系相对世界坐标系的方向
  - 性质: R^T·R = I, det(R) = 1
  - 第i列 = 世界坐标系第i个基向量在相机系中的表示
  
• t (3×1): 平移向量
  - 描述相机光心在世界坐标系中的位置
  - 相机中心: C = -R^T·t

自由度: 6个 (3个旋转 + 3个平移)
'''
    
    ax6.text(0.1, 0.95, RT_text, fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    
    #plt.tight_layout()
    plt.savefig('coordinate_systems_complete.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ 图像已保存: coordinate_systems_complete.png")
    #plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("开始生成相机几何可视化图像...")
    print("=" * 60)
    visualize_all_coordinate_systems()
    print("=" * 60)
    print("完成！")
