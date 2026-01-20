"""
坐标系统可视化
展示世界坐标系、相机坐标系、图像坐标系和像素坐标系之间的关系
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Arrow3D(FancyArrowPatch):
    """3D箭头类，用于绘制坐标轴"""
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

def draw_coordinate_system(ax, origin, rotation_matrix, scale=1.0, label_prefix='', 
                          colors=['r', 'g', 'b'], linewidth=2):
    """
    绘制3D坐标系
    
    参数:
        ax: 3D axes对象
        origin: 原点位置 [x, y, z]
        rotation_matrix: 3x3旋转矩阵
        scale: 坐标轴长度
        label_prefix: 标签前缀
        colors: 三个轴的颜色
        linewidth: 线宽
    """
    axes = rotation_matrix @ np.eye(3) * scale
    axis_labels = ['X', 'Y', 'Z']
    
    for i, (color, axis_label) in enumerate(zip(colors, axis_labels)):
        arrow = Arrow3D(origin[0], origin[1], origin[2],
                       axes[0, i], axes[1, i], axes[2, i],
                       mutation_scale=20, lw=linewidth, 
                       arrowstyle='-|>', color=color)
        ax.add_artist(arrow)
        
        # 添加轴标签
        end_point = origin + axes[:, i]
        ax.text(end_point[0], end_point[1], end_point[2], 
               f'{label_prefix}{axis_label}', 
               fontsize=11, weight='bold', color=color)

def draw_camera(ax, position, rotation, scale=0.5):
    """
    绘制相机模型（锥体）
    
    参数:
        ax: 3D axes对象
        position: 相机位置
        rotation: 相机旋转矩阵
        scale: 相机大小
    
    返回:
        world_points: 相机顶点在世界坐标系中的位置
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
               'k-', linewidth=1, alpha=0.6)
    
    # 绘制图像平面的四条边
    for i in range(1, 5):
        next_i = i + 1 if i < 4 else 1
        ax.plot([world_points[i, 0], world_points[next_i, 0]],
               [world_points[i, 1], world_points[next_i, 1]],
               [world_points[i, 2], world_points[next_i, 2]], 
               'b-', linewidth=2.5)
    
    # 在图像平面上添加半透明填充
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    image_plane = [world_points[1:5]]
    poly = Poly3DCollection(image_plane, alpha=0.2, facecolor='cyan', edgecolor='none')
    ax.add_collection3d(poly)
    
    return world_points

def visualize_all_coordinate_systems():
    """完整的四个坐标系统可视化"""
    
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('相机几何：四个坐标系统详解', fontsize=16, weight='bold', y=0.98)
    
    # ==================== 图1: 世界坐标系 → 相机坐标系 ====================
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.set_title('(a) 世界坐标系 → 相机坐标系', fontsize=13, weight='bold', pad=15)
    
    # 世界坐标系
    world_origin = np.array([0, 0, 0])
    world_rotation = np.eye(3)
    
    # 相机位置和姿态
    camera_position = np.array([3, 2, 4])
    theta_y = -np.pi/6  # 绕Y轴-30度
    theta_x = -np.pi/9  # 绕X轴-20度
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    camera_rotation = Ry @ Rx
    
    # 绘制世界坐标系
    draw_coordinate_system(ax1, world_origin, world_rotation, scale=2.0, 
                          label_prefix='W_', colors=['red', 'green', 'blue'], linewidth=3)
    
    # 绘制相机坐标系
    draw_coordinate_system(ax1, camera_position, camera_rotation, scale=1.5,
                          label_prefix='C_', colors=['darkred', 'darkgreen', 'darkblue'], 
                          linewidth=2)
    
    # 绘制相机模型
    camera_points = draw_camera(ax1, camera_position, camera_rotation, scale=0.6)
    
    # 绘制一个3D点
    point_world = np.array([1.5, 1.5, 1.0])
    ax1.scatter(*point_world, c='purple', s=200, marker='o', 
               edgecolors='black', linewidths=2, zorder=10)
    ax1.text(point_world[0]+0.2, point_world[1]+0.2, point_world[2]+0.3, 
            'P(Xw,Yw,Zw)', fontsize=10, weight='bold', color='purple')
    
    # 从点到相机光心的投影线
    ax1.plot([point_world[0], camera_position[0]],
            [point_world[1], camera_position[1]],
            [point_world[2], camera_position[2]], 
            'purple', linestyle='--', linewidth=2, alpha=0.7, label='投影射线')
    
    ax1.set_xlabel('X', fontsize=11, weight='bold')
    ax1.set_ylabel('Y', fontsize=11, weight='bold')
    ax1.set_zlabel('Z', fontsize=11, weight='bold')
    ax1.set_xlim([-0.5, 5])
    ax1.set_ylim([-0.5, 5])
    ax1.set_zlim([-0.5, 5])
    ax1.view_init(elev=20, azim=45)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    
    # ==================== 图2: 针孔相机模型（侧视图） ====================
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title('(b) 针孔相机模型（侧视图）', fontsize=13, weight='bold', pad=15)
    
    # 绘制光轴
    ax2.arrow(0, 0, 5.5, 0, head_width=0.15, head_length=0.2, 
             fc='black', ec='black', linewidth=2.5)
    ax2.text(5.8, 0, 'Zc (光轴)', fontsize=11, weight='bold', va='center')
    
    # 光心
    ax2.scatter(0, 0, s=250, c='red', marker='o', edgecolors='black', 
               linewidths=2.5, zorder=5, label='光心O')
    ax2.text(-0.35, -0.35, 'O', fontsize=12, weight='bold', ha='right')
    
    # 3D点
    point_z = 4.5
    point_y = 2.2
    ax2.scatter(point_z, point_y, s=180, c='purple', marker='o', 
               edgecolors='black', linewidths=2, zorder=5)
    ax2.text(point_z+0.25, point_y+0.25, 'P(Xc,Yc,Zc)', fontsize=11, 
            weight='bold', color='purple')
    
    # 焦距和成像平面
    focal_length = 1.8
    ax2.plot([focal_length, focal_length], [-3.5, 3.5], 'b-', linewidth=4, 
            label='成像平面', zorder=3)
    ax2.text(focal_length, -4.0, f'焦距 f={focal_length}', fontsize=10, 
            weight='bold', ha='center', bbox=dict(boxstyle='round', 
            facecolor='lightblue', alpha=0.7))
    
    # 投影点
    image_y = focal_length * point_y / point_z
    ax2.scatter(focal_length, image_y, s=180, c='orange', marker='s', 
               edgecolors='black', linewidths=2, zorder=5)
    ax2.text(focal_length+0.35, image_y, "p'(x,y)", fontsize=11, 
            weight='bold', color='orange')
    
    # 投影线
    ax2.plot([0, point_z], [0, point_y], 'purple', linestyle='--', 
            linewidth=2.5, alpha=0.7, label='投影射线')
    ax2.plot([focal_length, point_z], [image_y, point_y], 'gray', 
            linestyle=':', linewidth=1.5)
    
    # 标注Zc
    ax2.plot([point_z, point_z], [0, 0.15], 'k-', linewidth=1.5)
    ax2.text(point_z, -0.5, f'Zc={point_z}', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 标注Yc
    ax2.plot([0, 0.15], [point_y, point_y], 'k-', linewidth=1.5)
    ax2.text(-0.35, point_y, f'Yc={point_y}', fontsize=10, ha='right', 
            va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 标注y（成像高度）
    ax2.annotate('', xy=(focal_length-0.15, 0), xytext=(focal_length-0.15, image_y),
                arrowprops=dict(arrowstyle='<->', color='orange', lw=2.5))
    ax2.text(focal_length-0.45, image_y/2, f'y={image_y:.2f}', fontsize=10, 
            ha='right', color='orange', weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # 相似三角形公式
    formula_text = r'$\frac{y}{f} = \frac{Y_c}{Z_c}$'
    ax2.text(0.5, 3.2, formula_text, fontsize=13, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax2.set_xlim([-0.8, 6.5])
    ax2.set_ylim([-4.5, 4])
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_xlabel('Z 轴（深度方向）', fontsize=11, weight='bold')
    ax2.set_ylabel('Y 轴', fontsize=11, weight='bold')
    
    # ==================== 图3: 图像坐标系 → 像素坐标系 ====================
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title('(c) 图像坐标系 → 像素坐标系', fontsize=13, weight='bold', pad=15)
    
    # 图像平面尺寸（毫米）
    image_width_mm = 10
    image_height_mm = 8
    
    # 像素分辨率
    image_width_px = 640
    image_height_px = 480
    
    # 绘制图像平面（图像坐标系，原点在中心）
    rect = mpatches.Rectangle((-image_width_mm/2, -image_height_mm/2), 
                              image_width_mm, image_height_mm,
                              fill=False, edgecolor='blue', linewidth=3.5)
    ax3.add_patch(rect)
    
    # 图像坐标系原点（中心）
    ax3.scatter(0, 0, s=250, c='blue', marker='o', edgecolors='black', 
               linewidths=2.5, zorder=5, label='图像坐标原点')
    ax3.text(0.4, -0.6, 'O (图像中心)', fontsize=10, weight='bold', color='blue')
    
    # 图像坐标系坐标轴
    ax3.arrow(0, 0, 4.5, 0, head_width=0.35, head_length=0.35, 
             fc='blue', ec='blue', linewidth=2.5, alpha=0.8)
    ax3.text(5.0, 0, 'x (mm)', fontsize=12, weight='bold', color='blue')
    ax3.arrow(0, 0, 0, 3.5, head_width=0.35, head_length=0.35, 
             fc='blue', ec='blue', linewidth=2.5, alpha=0.8)
    ax3.text(0, 4.0, 'y (mm)', fontsize=12, weight='bold', color='blue')
    
    # 像素坐标系原点（左上角）
    pixel_origin = (-image_width_mm/2, image_height_mm/2)
    ax3.scatter(*pixel_origin, s=250, c='red', marker='s', edgecolors='black', 
               linewidths=2.5, zorder=5, label='像素坐标原点')
    ax3.text(pixel_origin[0]-0.6, pixel_origin[1]+0.6, "O' (0,0)", fontsize=10, 
            weight='bold', color='red')
    
    # 像素坐标系坐标轴
    ax3.arrow(pixel_origin[0], pixel_origin[1], 4.5, 0, head_width=0.35, 
             head_length=0.35, fc='red', ec='red', linewidth=2.5, 
             linestyle='--', alpha=0.8)
    ax3.text(pixel_origin[0]+5.0, pixel_origin[1], 'u (px)', fontsize=12, 
            weight='bold', color='red')
    ax3.arrow(pixel_origin[0], pixel_origin[1], 0, -3.5, head_width=0.35, 
             head_length=0.35, fc='red', ec='red', linewidth=2.5, 
             linestyle='--', alpha=0.8)
    ax3.text(pixel_origin[0], pixel_origin[1]-4.0, 'v (px)', fontsize=12, 
            weight='bold', color='red')
    
    # 示例点
    point_image = np.array([2.5, 1.8])  # 图像坐标
    ax3.scatter(*point_image, s=200, c='green', marker='o', edgecolors='black', 
               linewidths=2, zorder=5)
    ax3.text(point_image[0]+0.3, point_image[1]+0.4, 'p(x,y)', fontsize=10, 
            weight='bold', color='green')
    
    # 标注从两个原点到点的向量
    ax3.annotate('', xy=point_image, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2, linestyle=':'))
    ax3.annotate('', xy=point_image, xytext=pixel_origin,
                arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle=':'))
    
    # 像素网格
    grid_step = 2
    for i in range(int(image_width_mm/grid_step)+1):
        x = -image_width_mm/2 + i * grid_step
        ax3.plot([x, x], [-image_height_mm/2, image_height_mm/2], 
                'gray', alpha=0.25, linewidth=0.5)
    for i in range(int(image_height_mm/grid_step)+1):
        y = -image_height_mm/2 + i * grid_step
        ax3.plot([-image_width_mm/2, image_width_mm/2], [y, y], 
                'gray', alpha=0.25, linewidth=0.5)
    
    # 转换公式
    dx = image_width_mm / image_width_px
    dy = image_height_mm / image_height_px
    cx = image_width_px / 2
    cy = image_height_px / 2
    
    textstr = '转换关系:\n'
    textstr += r'$u = \frac{x}{d_x} + c_x$' + '\n'
    textstr += r'$v = -\frac{y}{d_y} + c_y$' + '\n\n'
    textstr += f'像素尺寸:\n'
    textstr += f'dx={dx:.4f} mm\n'
    textstr += f'dy={dy:.4f} mm\n'
    textstr += f'主点: ({cx}, {cy}) px'
    
    ax3.text(-6.8, -5.5, textstr, fontsize=8.5, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            family='monospace')
    
    ax3.set_xlim([-7.5, 7.5])
    ax3.set_ylim([-6.5, 6])
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.set_xlabel('X 方向', fontsize=11, weight='bold')
    ax3.set_ylabel('Y 方向', fontsize=11, weight='bold')
    
    # ==================== 图4: 旋转矩阵列向量的几何意义 ====================
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.set_title('(d) 旋转矩阵R的列向量含义', fontsize=13, weight='bold', pad=15)
    
    # 绘制世界坐标系
    draw_coordinate_system(ax4, world_origin, world_rotation, scale=1.5, 
                          label_prefix='W_', colors=['red', 'green', 'blue'], 
                          linewidth=2.5)
    
    # 绘制相机坐标系
    camera_pos_simple = np.array([2, 1.5, 2])
    theta = np.pi/4  # 45度
    R_simple = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
    
    draw_coordinate_system(ax4, camera_pos_simple, R_simple, scale=1.5,
                          label_prefix='C_', colors=['darkred', 'darkgreen', 'darkblue'], 
                          linewidth=2.5)
    
    # 标注：世界X轴在相机系中的表示
    r1 = R_simple[:, 0] * 1.5
    ax4.plot([camera_pos_simple[0], camera_pos_simple[0] + r1[0]],
            [camera_pos_simple[1], camera_pos_simple[1] + r1[1]],
            [camera_pos_simple[2], camera_pos_simple[2] + r1[2]],
            'orange', linewidth=3, linestyle='--', label='R的第1列')
    
    # 添加文本说明
    text_pos = camera_pos_simple + r1 * 0.5
    ax4.text(text_pos[0], text_pos[1], text_pos[2]+0.3, 
            'r₁=R[:,0]\n世界X在相机系', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax4.set_xlabel('X', fontsize=11, weight='bold')
    ax4.set_ylabel('Y', fontsize=11, weight='bold')
    ax4.set_zlabel('Z', fontsize=11, weight='bold')
    ax4.set_xlim([-1, 4])
    ax4.set_ylim([-1, 4])
    ax4.set_zlim([-1, 4])
    ax4.view_init(elev=25, azim=45)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9, loc='upper left')
    
    # ==================== 图5: 完整投影流程图 ====================
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    ax5.set_title('(e) 完整的坐标转换流程', fontsize=13, weight='bold', pad=15)
    
    # 绘制流程框
    boxes = [
        {'name': '世界坐标系\n(Xw, Yw, Zw, 1)', 'pos': (0.15, 0.75), 'color': '#FFE6E6'},
        {'name': '相机坐标系\n(Xc, Yc, Zc)', 'pos': (0.15, 0.5), 'color': '#E6F3FF'},
        {'name': '图像坐标系\n(x, y)', 'pos': (0.15, 0.25), 'color': '#E6FFE6'},
        {'name': '像素坐标系\n(u, v)', 'pos': (0.15, 0.0), 'color': '#FFF9E6'},
    ]
    
    for box in boxes:
        fancy_box = FancyBboxPatch((box['pos'][0], box['pos'][1]), 0.35, 0.15,
                                   boxstyle="round,pad=0.015", 
                                   edgecolor='black', facecolor=box['color'],
                                   linewidth=2.5, transform=ax5.transAxes)
        ax5.add_patch(fancy_box)
        ax5.text(box['pos'][0]+0.175, box['pos'][1]+0.075, box['name'],
                ha='center', va='center', fontsize=10, weight='bold',
                transform=ax5.transAxes)
    
    # 绘制箭头和标签
    arrows = [
        {'from': (0.325, 0.75), 'to': (0.325, 0.58), 'label': '外参: [R|t]\n刚体变换'},
        {'from': (0.325, 0.5), 'to': (0.325, 0.33), 'label': '透视投影\n÷Zc'},
        {'from': (0.325, 0.25), 'to': (0.325, 0.08), 'label': '内参: K\n缩放+平移'},
    ]
    
    for arrow in arrows:
        arr = FancyArrowPatch(arrow['from'], arrow['to'],
                             arrowstyle='->', mutation_scale=35, 
                             linewidth=3.5, color='darkblue',
                             transform=ax5.transAxes)
        ax5.add_patch(arr)
        mid_x = arrow['from'][0] + 0.25
        mid_y = (arrow['from'][1] + arrow['to'][1]) / 2
        ax5.text(mid_x, mid_y, arrow['label'],
                ha='left', va='center', fontsize=9.5, weight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                transform=ax5.transAxes)
    
    # 完整公式
    ax5.text(0.5, 0.95, '完整投影公式:', fontsize=11, weight='bold', 
            ha='center', transform=ax5.transAxes)
    formula = r'$s\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \cdot [R|t] \cdot \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$'
    ax5.text(0.5, 0.87, formula, fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            transform=ax5.transAxes)
    
    # ==================== 图6: 投影矩阵分解 ====================
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    ax6.set_title('(f) 投影矩阵 P = K[R|t] 的分解', fontsize=13, weight='bold', pad=15)
    
    # 矩阵公式
    matrix_text = r'''
$P = K[R|t] = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_x \\
r_{21} & r_{22} & r_{23} & t_y \\
r_{31} & r_{32} & r_{33} & t_z
\end{bmatrix}$
'''
    
    ax6.text(0.5, 0.75, matrix_text, fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            transform=ax6.transAxes)
    
    # 参数说明
    params_text = '''
内参矩阵 K (3×3):
  • fx, fy: 焦距(像素单位)
  • cx, cy: 主点坐标
  • 自由度: 4 (通常倾斜s=0)

外参矩阵 [R|t] (3×4):
  • R: 旋转矩阵 (3×3)
     - 正交矩阵: R^T R = I
     - det(R) = 1
  • t: 平移向量 (3×1)
  • 自由度: 6 (3旋转+3平移)

分解方法:
  1. RQ分解: M = P[:,:3] = KR
  2. 提取t: t = K^(-1) P[:,3]
  3. 相机中心: C = -R^T t
'''
    
    ax6.text(0.05, 0.45, params_text, fontsize=8.5, 
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7),
            transform=ax6.transAxes)
    
    #plt.tight_layout()
    plt.savefig('coordinate_systems_complete.png', dpi=300, bbox_inches='tight')
    print("✓ 图像已保存: coordinate_systems_complete.png")
    #plt.show()

if __name__ == "__main__":
    print("正在生成坐标系统可视化...")
    visualize_all_coordinate_systems()
    print("完成！")
