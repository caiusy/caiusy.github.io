import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
print("生成简化图片...")

# 图1: 针孔相机模型
fig, ax = plt.subplots(figsize=(12, 6))
f = 3
ax.arrow(0, 0, 5, 0, head_width=0.1, head_length=0.2, fc='blue', ec='blue', lw=2)
ax.plot(0, 0, 'ko', ms=12)
ax.plot([f, f], [-2, 2], 'r-', lw=3)
ax.plot(6, 1.5, 'go', ms=10)
ax.plot([0, 6], [0, 1.5], 'g--', lw=1.5, alpha=0.6)
ax.plot(f, f*1.5/6, 'ro', ms=8)
ax.text(f/2, -0.3, f'f={f}', fontsize=11)
ax.text(f, 2.3, 'Image Plane', fontsize=12)
ax.set_xlim([-0.5, 7])
ax.set_ylim([-2.5, 2.5])
ax.grid(True, alpha=0.3)
ax.set_title('Pinhole Camera Model', fontsize=14, fontweight='bold')
plt.savefig('pinhole_camera_model.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ pinhole_camera_model.png")

# 图2: 单应矩阵
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
square = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
ax1.plot(square[:,0], square[:,1], 'b-', lw=3)
ax1.fill(square[:,0], square[:,1], alpha=0.2, color='blue')
ax1.set_title('Source Plane', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
H = np.array([[0.9,0.3,0.1],[0.1,0.8,0.2],[0.05,0.1,1.0]])
trans = []
for p in square[:-1]:
    ph = H @ np.array([p[0], p[1], 1])
    trans.append([ph[0]/ph[2], ph[1]/ph[2]])
trans.append(trans[0])
trans = np.array(trans)
ax2.plot(trans[:,0], trans[:,1], 'r-', lw=3)
ax2.fill(trans[:,0], trans[:,1], alpha=0.2, color='red')
ax2.set_title('Homography Transform', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
plt.savefig('homography_transformation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ homography_transformation.png")

print("完成！")
