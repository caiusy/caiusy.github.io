import matplotlib
matplotlib.use("Agg")
"""
单应矩阵可视化
展示平面到平面的投影变换
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def estimate_homography_dlt(src_points, dst_points):
    """使用DLT算法估计单应矩阵"""
    n = src_points.shape[0]
    A = []
    for i in range(n):
        x, y = src_points[i]
        x_prime, y_prime = dst_points[i]
        A.append([-x, -y, -1, 0, 0, 0, x_prime*x, x_prime*y, x_prime])
        A.append([0, 0, 0, -x, -y, -1, y_prime*x, y_prime*y, y_prime])
    
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_homography(H, points):
    """应用单应变换"""
    points_homo = np.column_stack([points, np.ones(len(points))])
    transformed_homo = (H @ points_homo.T).T
    transformed = transformed_homo[:, :2] / transformed_homo[:, 2:3]
    return transformed


def visualize_homography():
    """可视化单应矩阵变换"""
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('单应矩阵（Homography）可视化', fontsize=16, weight='bold', y=0.98)
    
    # ==================== 图1: 基本单应变换 ====================
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title('① 基本单应变换：矩形 → 四边形', fontsize=12, weight='bold', pad=15)
    
    # 源矩形
    src_rect = np.array([
        [0, 0],
        [4, 0],
        [4, 3],
        [0, 3]
    ], dtype=float)
    
    # 目标四边形（透视变换后）
    dst_quad = np.array([
        [1, 0.5],
        [6, 1],
        [5.5, 4],
        [0.5, 3.5]
    ], dtype=float)
    
    # 估计单应矩阵
    H = estimate_homography_dlt(src_rect, dst_quad)
    
    # 绘制源矩形
    rect_patch = Polygon(src_rect, fill=True, facecolor='lightblue', 
                        edgecolor='blue', linewidth=3, alpha=0.5, label='源图像')
    ax1.add_patch(rect_patch)
    
    # 绘制对应点
    for i, (src, dst) in enumerate(zip(src_rect, dst_quad)):
        ax1.plot(*src, 'bo', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        ax1.text(src[0]-0.3, src[1]-0.3, f'p{i+1}', fontsize=10, weight='bold', color='blue')
    
    # 绘制目标四边形
    quad_patch = Polygon(dst_quad, fill=True, facecolor='lightcoral', 
                        edgecolor='red', linewidth=3, alpha=0.5, label='目标图像')
    ax1.add_patch(quad_patch)
    
    for i, (src, dst) in enumerate(zip(src_rect, dst_quad)):
        ax1.plot(*dst, 'rs', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        ax1.text(dst[0]+0.2, dst[1]+0.2, f"p{i+1}'", fontsize=10, weight='bold', color='red')
    
    # 绘制对应关系
    for src, dst in zip(src_rect, dst_quad):
        ax1.annotate('', xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='--', alpha=0.6))
    
    # 显示单应矩阵
    H_text = 'H = \n'
    for i in range(3):
        H_text += f'[{H[i,0]:6.3f} {H[i,1]:6.3f} {H[i,2]:6.3f}]\n'
    
    ax1.text(0.02, 0.98, H_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlim([-1, 7])
    ax1.set_ylim([-1, 5])
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlabel('X', fontsize=10, weight='bold')
    ax1.set_ylabel('Y', fontsize=10, weight='bold')
    
    # ==================== 图2: 网格变换 ====================
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title('② 网格变换效果', fontsize=12, weight='bold', pad=15)
    
    # 创建网格
    x_grid = np.linspace(0, 4, 9)
    y_grid = np.linspace(0, 3, 7)
    
    # 绘制原始网格
    for x in x_grid:
        line_points = np.column_stack([np.ones(len(y_grid))*x, y_grid])
        ax2.plot(line_points[:, 0], line_points[:, 1], 'b-', linewidth=1, alpha=0.3)
    
    for y in y_grid:
        line_points = np.column_stack([x_grid, np.ones(len(x_grid))*y])
        ax2.plot(line_points[:, 0], line_points[:, 1], 'b-', linewidth=1, alpha=0.3)
    
    # 绘制变换后的网格
    for x in x_grid:
        line_points = np.column_stack([np.ones(len(y_grid))*x, y_grid])
        transformed = apply_homography(H, line_points)
        ax2.plot(transformed[:, 0], transformed[:, 1], 'r-', linewidth=2, alpha=0.7)
    
    for y in y_grid:
        line_points = np.column_stack([x_grid, np.ones(len(x_grid))*y])
        transformed = apply_homography(H, line_points)
        ax2.plot(transformed[:, 0], transformed[:, 1], 'r-', linewidth=2, alpha=0.7)
    
    # 绘制边界
    rect_patch = Polygon(src_rect, fill=False, edgecolor='blue', linewidth=3, label='原始')
    ax2.add_patch(rect_patch)
    quad_patch = Polygon(dst_quad, fill=False, edgecolor='red', linewidth=3, label='变换后')
    ax2.add_patch(quad_patch)
    
    ax2.set_xlim([-1, 7])
    ax2.set_ylim([-1, 5])
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlabel('X', fontsize=10, weight='bold')
    ax2.set_ylabel('Y', fontsize=10, weight='bold')
    
    # ==================== 图3: 单应矩阵的分解 ====================
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.axis('off')
    ax3.set_title('③ 单应矩阵的构成', fontsize=12, weight='bold', pad=15)
    
    decomp_text = '''
单应矩阵 H (3×3):

H = [h11  h12  h13]
    [h21  h22  h23]
    [h31  h32  h33]

自由度: 8 (9个元素 - 1个尺度)

物理意义:
━━━━━━━━━━━━━━━━━━━━━━
对于平面 Z=0:

H = K[r1  r2  t]

其中:
• K: 内参矩阵
• r1, r2: 旋转矩阵的前两列
• t: 平移向量

━━━━━━━━━━━━━━━━━━━━━━
一般形式:

H = K(R - t·nᵀ/d)K⁻¹

其中:
• R, t: 外参
• n: 平面法向量
• d: 平面距离

━━━━━━━━━━━━━━━━━━━━━━
应用场景:
✓ 图像拼接/全景图
✓ 文档矫正
✓ 增强现实
✓ 相机标定
✓ 平面目标跟踪
'''
    
    ax3.text(0.1, 0.95, decomp_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # ==================== 图4: 不同类型的变换 ====================
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title('④ 仿射变换 vs 透视变换', fontsize=12, weight='bold', pad=15)
    
    # 原始正方形
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    
    # 仿射变换（保持平行线平行）
    affine_dst = np.array([[0.5, 0.2], [3, 0.5], [2.5, 2.5], [0, 2]], dtype=float)
    H_affine = estimate_homography_dlt(square, affine_dst)
    
    # 透视变换（不保持平行线平行）
    perspective_dst = np.array([[0.5, 0], [3.5, 0.5], [3, 3], [0, 2.5]], dtype=float)
    H_perspective = estimate_homography_dlt(square, perspective_dst)
    
    # 绘制原始正方形
    square_patch = Polygon(square, fill=True, facecolor='lightgreen', 
                          edgecolor='green', linewidth=2, alpha=0.4, label='原始')
    ax4.add_patch(square_patch)
    
    # 绘制仿射变换结果
    affine_patch = Polygon(affine_dst, fill=False, edgecolor='blue', 
                          linewidth=3, linestyle='--', label='仿射变换')
    ax4.add_patch(affine_patch)
    
    # 绘制透视变换结果
    persp_patch = Polygon(perspective_dst, fill=False, edgecolor='red', 
                         linewidth=3, linestyle='-.', label='透视变换')
    ax4.add_patch(persp_patch)
    
    # 添加说明
    ax4.text(0.5, 2.8, '仿射变换:\n保持平行线平行', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax4.text(2, 3.3, '透视变换:\n平行线不再平行', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    ax4.set_xlim([-0.5, 4])
    ax4.set_ylim([-0.5, 3.5])
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_xlabel('X', fontsize=10, weight='bold')
    ax4.set_ylabel('Y', fontsize=10, weight='bold')
    
    # ==================== 图5: DLT算法求解 ====================
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    ax5.set_title('⑤ DLT算法求解单应矩阵', fontsize=12, weight='bold', pad=15)
    
    dlt_text = '''
直接线性变换 (DLT) 算法:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

给定: N对对应点 (xᵢ,yᵢ) ↔ (xᵢ',yᵢ')

单应变换:
x' = (h11·x + h12·y + h13) / (h31·x + h32·y + h33)
y' = (h21·x + h22·y + h23) / (h31·x + h32·y + h33)

交叉相乘得到线性方程:
[-x  -y  -1   0   0   0  x'x  x'y  x'] [h11]
[ 0   0   0  -x  -y  -1  y'x  y'y  y'] [h12]
                                        [h13]
                                        [h21]
                                        [h22]   = 0
                                        [h23]
                                        [h31]
                                        [h32]
                                        [h33]

构建矩阵 A (2N×9):
• 每对点贡献2行
• 至少需要4对点 (8个方程)

SVD求解:
1. A = U·Σ·Vᵀ
2. h = Vᵀ的最后一列
3. reshape为3×3矩阵
4. 归一化: H = H/H[2,2]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
优点: 简单、快速、线性
缺点: 对噪声敏感，需要RANSAC
'''
    
    ax5.text(0.05, 0.95, dlt_text, transform=ax5.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ==================== 图6: 实际应用示例 ====================
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title('⑥ 应用: 文档矫正', fontsize=12, weight='bold', pad=15)
    
    # 倾斜的文档（梯形）
    tilted_doc = np.array([
        [1, 0.5],
        [5, 1],
        [4.5, 3.5],
        [0.5, 3]
    ], dtype=float)
    
    # 矫正后的矩形
    rect_doc = np.array([
        [0, 0],
        [4, 0],
        [4, 3],
        [0, 3]
    ], dtype=float)
    
    # 绘制倾斜文档
    tilted_patch = Polygon(tilted_doc, fill=True, facecolor='lightyellow', 
                          edgecolor='orange', linewidth=3, alpha=0.6)
    ax6.add_patch(tilted_patch)
    
    # 添加文字模拟
    for i in range(6):
        y_pos = 0.7 + i * 0.4
        src_start = np.array([1.5, y_pos])
        src_end = np.array([4.5, y_pos + 0.1])
        ax6.plot([src_start[0], src_end[0]], [src_start[1], src_end[1]], 
                'gray', linewidth=2, alpha=0.5)
    
    ax6.text(3, 2, '倾斜的\n文档', fontsize=11, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    # 添加箭头
    ax6.annotate('', xy=(6.5, 1.75), xytext=(5.5, 1.75),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    ax6.text(6, 2.2, '单应\n变换', fontsize=10, ha='center', weight='bold',
            color='green')
    
    # 绘制矫正后的文档
    rect_patch = Polygon(rect_doc + np.array([7, 0]), fill=True, 
                        facecolor='white', edgecolor='blue', linewidth=3, alpha=0.9)
    ax6.add_patch(rect_patch)
    
    # 添加矫正后的文字
    for i in range(6):
        y_pos = 0.5 + i * 0.4
        ax6.plot([7.5, 10.5], [y_pos, y_pos], 'gray', linewidth=2, alpha=0.5)
    
    ax6.text(8.5, 1.5, '矫正后的\n文档', fontsize=11, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax6.set_xlim([0, 11.5])
    ax6.set_ylim([-0.5, 4])
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.2)
    ax6.set_xlabel('X', fontsize=10, weight='bold')
    ax6.set_ylabel('Y', fontsize=10, weight='bold')
    
    #plt.tight_layout()
    plt.savefig('homography_visualization.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✓ 单应矩阵可视化已保存: homography_visualization.png")
    #plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("生成单应矩阵可视化...")
    print("=" * 60)
    visualize_homography()
    print("=" * 60)
    print("完成！")
