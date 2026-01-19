"""
相机几何完整演示
包含所有核心功能：投影、分解、旋转矩阵、单应矩阵等
"""

import numpy as np
from scipy.linalg import rq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CameraGeometry:
    """相机几何计算类"""
    
    def __init__(self, fx, fy, cx, cy, width=None, height=None):
        """
        初始化相机内参
        
        参数:
            fx, fy: 焦距（像素单位）
            cx, cy: 主点坐标（像素）
            width, height: 图像尺寸（可选）
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=float)
    
    def __repr__(self):
        return f"CameraGeometry(fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy})"
    
    # ==================== 旋转矩阵相关 ====================
    
    @staticmethod
    def rotation_matrix_from_euler(roll, pitch, yaw, order='zyx'):
        """
        从欧拉角创建旋转矩阵
        
        参数:
            roll: 绕X轴旋转角度（弧度）
            pitch: 绕Y轴旋转角度（弧度）
            yaw: 绕Z轴旋转角度（弧度）
            order: 旋转顺序，'xyz'或'zyx'
            
        返回:
            R: 3x3旋转矩阵
        """
        # 基本旋转矩阵
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
    def euler_from_rotation_matrix(R, order='zyx'):
        """
        从旋转矩阵提取欧拉角
        
        参数:
            R: 3x3旋转矩阵
            order: 旋转顺序
            
        返回:
            (roll, pitch, yaw): 欧拉角（弧度）
        """
        if order == 'zyx':
            # ZYX顺序
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            
            singular = sy < 1e-6
            
            if not singular:
                roll = np.arctan2(R[2, 1], R[2, 2])
                pitch = np.arctan2(-R[2, 0], sy)
                yaw = np.arctan2(R[1, 0], R[0, 0])
            else:
                roll = np.arctan2(-R[1, 2], R[1, 1])
                pitch = np.arctan2(-R[2, 0], sy)
                yaw = 0
                
            return roll, pitch, yaw
        else:
            raise NotImplementedError(f"Order {order} not implemented")
    
    @staticmethod
    def rotation_matrix_from_axis_angle(axis, theta):
        """
        从轴角表示创建旋转矩阵（罗德里格斯公式）
        
        参数:
            axis: 旋转轴（3D向量）
            theta: 旋转角度（弧度）
            
        返回:
            R: 3x3旋转矩阵
        """
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)  # 归一化
        
        # 反对称矩阵
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # 罗德里格斯公式
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R
    
    @staticmethod
    def axis_angle_from_rotation_matrix(R):
        """
        从旋转矩阵提取轴角表示
        
        参数:
            R: 3x3旋转矩阵
            
        返回:
            axis: 旋转轴
            theta: 旋转角度
        """
        theta = np.arccos((np.trace(R) - 1) / 2)
        
        if np.abs(theta) < 1e-10:
            # 无旋转
            return np.array([1, 0, 0]), 0.0
        
        if np.abs(theta - np.pi) < 1e-10:
            # 180度旋转，特殊处理
            # 找到最大对角元素
            i = np.argmax(np.diag(R))
            axis = np.zeros(3)
            axis[i] = np.sqrt((R[i, i] + 1) / 2)
            for j in range(3):
                if j != i:
                    axis[j] = R[i, j] / (2 * axis[i])
        else:
            # 一般情况
            axis = np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ]) / (2 * np.sin(theta))
        
        return axis, theta
    
    @staticmethod
    def verify_rotation_matrix(R, verbose=True):
        """
        验证旋转矩阵的有效性
        
        参数:
            R: 待验证的矩阵
            verbose: 是否打印详细信息
            
        返回:
            is_valid: 是否为有效的旋转矩阵
        """
        # 检查维度
        if R.shape != (3, 3):
            if verbose:
                print(f"✗ 维度错误: {R.shape}, 应该是 (3, 3)")
            return False
        
        # 检查正交性: R^T @ R = I
        RTR = R.T @ R
        orthogonality_error = np.linalg.norm(RTR - np.eye(3))
        
        # 检查行列式: det(R) = 1
        det = np.linalg.det(R)
        det_error = np.abs(det - 1.0)
        
        is_orthogonal = orthogonality_error < 1e-6
        is_det_one = det_error < 1e-6
        
        is_valid = is_orthogonal and is_det_one
        
        if verbose:
            print(f"旋转矩阵验证:")
            print(f"  正交性误差: {orthogonality_error:.2e} {'✓' if is_orthogonal else '✗'}")
            print(f"  行列式误差: {det_error:.2e} {'✓' if is_det_one else '✗'}")
            print(f"  结果: {'有效 ✓' if is_valid else '无效 ✗'}")
        
        return is_valid
    
    # ==================== 投影相关 ====================
    
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
        point_world = np.array(point_world, dtype=float)
        t = np.array(t, dtype=float).reshape(3)
        
        # 世界坐标 -> 相机坐标
        point_camera = R @ point_world + t
        
        # 检查点是否在相机前方
        if point_camera[2] <= 0:
            raise ValueError(f"点在相机后方: Zc = {point_camera[2]:.3f}")
        
        # 投影到像素坐标
        point_homo = self.K @ point_camera
        pixel = point_homo[:2] / point_homo[2]
        
        return pixel, point_camera
    
    def project_points(self, points_world, R, t):
        """
        批量投影3D点
        
        参数:
            points_world: Nx3数组，N个3D点
            R: 旋转矩阵
            t: 平移向量
            
        返回:
            pixels: Nx2数组，像素坐标
            points_camera: Nx3数组，相机坐标
        """
        points_world = np.array(points_world)
        t = np.array(t).reshape(3, 1)
        
        # 转换到相机坐标系
        points_camera = (R @ points_world.T + t).T
        
        # 投影
        points_homo = (self.K @ points_camera.T).T
        pixels = points_homo[:, :2] / points_homo[:, 2:3]
        
        return pixels, points_camera
    
    def backproject_pixel(self, pixel, depth):
        """
        将像素坐标和深度反投影到3D点（相机坐标系）
        
        参数:
            pixel: (u, v) 像素坐标
            depth: 深度值 Zc
            
        返回:
            point_camera: 3D点（相机坐标系）
        """
        u, v = pixel
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        
        return np.array([x, y, z])
    
    # ==================== 投影矩阵相关 ====================
    
    def compute_projection_matrix(self, R, t):
        """
        计算投影矩阵 P = K[R|t]
        
        参数:
            R: 旋转矩阵
            t: 平移向量
            
        返回:
            P: 3x4投影矩阵
        """
        t = np.array(t).reshape(3, 1)
        return self.K @ np.hstack([R, t])
    
    def decompose_projection_matrix(self, P):
        """
        分解投影矩阵 P = K[R|t]
        
        参数:
            P: 3x4投影矩阵
        
        返回:
            K: 内参矩阵
            R: 旋转矩阵
            t: 平移向量
            camera_center: 相机中心（世界坐标系）
        """
        # 分离前3列和第4列
        M = P[:, :3]
        p4 = P[:, 3]
        
        # RQ分解
        K, R = rq(M)
        
        # 确保K的对角元素为正
        T = np.diag(np.sign(np.diag(K)))
        if T[0, 0] == 0: T[0, 0] = 1
        if T[1, 1] == 0: T[1, 1] = 1
        if T[2, 2] == 0: T[2, 2] = 1
        
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
    
    # ==================== 单应矩阵相关 ====================
    
    def compute_homography(self, R, t, n, d):
        """
        计算平面的单应矩阵
        
        参数:
            R, t: 外参
            n: 平面法向量（世界坐标系，单位向量）
            d: 平面到原点的距离
        
        返回:
            H: 3x3单应矩阵
        """
        n = np.array(n).reshape(3, 1)
        t = np.array(t).reshape(3, 1)
        
        H = R - (t @ n.T) / d
        H = self.K @ H @ np.linalg.inv(self.K)
        
        # 归一化
        return H / H[2, 2]
    
    def decompose_homography(self, H):
        """
        从单应矩阵分解R和t（假设已知K）
        
        参数:
            H: 3x3单应矩阵
            
        返回:
            solutions: 可能的(R, t, n)解（最多4组）
        """
        # H = K[r1 r2 t]对于Z=0平面
        # 这里简化为Z=0平面的情况
        
        H_normalized = H / np.linalg.norm(np.linalg.inv(self.K) @ H[:, 0])
        
        # 提取
        inv_K = np.linalg.inv(self.K)
        H_prime = inv_K @ H_normalized
        
        r1 = H_prime[:, 0]
        r2 = H_prime[:, 1]
        t = H_prime[:, 2]
        
        # 归一化
        r1 = r1 / np.linalg.norm(r1)
        r2 = r2 / np.linalg.norm(r2)
        
        # 计算r3
        r3 = np.cross(r1, r2)
        
        # 构建R并修正为标准旋转矩阵
        R_approx = np.column_stack([r1, r2, r3])
        
        # SVD修正
        U, S, Vt = np.linalg.svd(R_approx)
        R = U @ Vt
        
        if np.linalg.det(R) < 0:
            R = -R
        
        return R, t
    
    @staticmethod
    def estimate_homography_dlt(src_points, dst_points):
        """
        使用DLT算法估计单应矩阵
        
        参数:
            src_points: Nx2数组，源图像点
            dst_points: Nx2数组，目标图像点
            
        返回:
            H: 3x3单应矩阵
        """
        n = src_points.shape[0]
        if n < 4:
            raise ValueError("至少需要4对点")
        
        # 构建A矩阵
        A = []
        for i in range(n):
            x, y = src_points[i]
            x_prime, y_prime = dst_points[i]
            
            A.append([-x, -y, -1, 0, 0, 0, x_prime*x, x_prime*y, x_prime])
            A.append([0, 0, 0, -x, -y, -1, y_prime*x, y_prime*y, y_prime])
        
        A = np.array(A)
        
        # SVD求解
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        
        # 归一化
        return H / H[2, 2]
    
    # ==================== 相机位姿相关 ====================
    
    def get_camera_center(self, R, t):
        """
        计算相机中心在世界坐标系中的位置
        
        参数:
            R: 旋转矩阵
            t: 平移向量
            
        返回:
            C: 相机中心（世界坐标系）
        """
        t = np.array(t).reshape(3)
        return -R.T @ t
    
    def look_at(self, camera_position, target_position, up_vector=np.array([0, 0, 1])):
        """
        构建"look at"相机位姿
        
        参数:
            camera_position: 相机位置（世界坐标系）
            target_position: 目标位置（世界坐标系）
            up_vector: 上方向向量
            
        返回:
            R: 旋转矩阵
            t: 平移向量
        """
        camera_position = np.array(camera_position, dtype=float)
        target_position = np.array(target_position, dtype=float)
        up_vector = np.array(up_vector, dtype=float)
        
        # 相机朝向（Z轴）
        z_axis = target_position - camera_position
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # X轴（右方向）
        x_axis = np.cross(up_vector, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y轴（下方向，因为图像坐标系Y向下）
        y_axis = np.cross(z_axis, x_axis)
        
        # 构建旋转矩阵（相机坐标系的基向量）
        R = np.column_stack([x_axis, y_axis, z_axis])
        
        # 平移向量
        t = -R.T @ camera_position
        
        return R.T, t


def demo_camera_geometry():
    """演示相机几何的所有功能"""
    
    print("=" * 80)
    print("相机几何完整演示")
    print("=" * 80)
    print()
    
    # 创建相机
    camera = CameraGeometry(
        fx=800, fy=800,  # 焦距
        cx=320, cy=240,  # 主点
        width=640, height=480  # 图像尺寸
    )
    
    print("【1】相机内参矩阵 K:")
    print(camera.K)
    print()
    
    # ==================== 旋转矩阵演示 ====================
    print("=" * 80)
    print("【2】旋转矩阵演示")
    print("=" * 80)
    
    # 从欧拉角创建
    roll, pitch, yaw = np.deg2rad([10, 20, 30])
    print(f"\n欧拉角（度）: roll={np.rad2deg(roll):.1f}, pitch={np.rad2deg(pitch):.1f}, yaw={np.rad2deg(yaw):.1f}")
    
    R = camera.rotation_matrix_from_euler(roll, pitch, yaw, order='zyx')
    print("\n旋转矩阵 R:")
    print(R)
    
    # 验证
    camera.verify_rotation_matrix(R)
    
    # 提取欧拉角
    roll_recovered, pitch_recovered, yaw_recovered = camera.euler_from_rotation_matrix(R)
    print(f"\n恢复的欧拉角（度）: roll={np.rad2deg(roll_recovered):.1f}, "
          f"pitch={np.rad2deg(pitch_recovered):.1f}, yaw={np.rad2deg(yaw_recovered):.1f}")
    
    # 轴角表示
    print("\n" + "-" * 80)
    axis, theta = camera.axis_angle_from_rotation_matrix(R)
    print(f"轴角表示:")
    print(f"  旋转轴: {axis}")
    print(f"  旋转角: {np.rad2deg(theta):.2f}°")
    
    # 从轴角恢复旋转矩阵
    R_from_axis = camera.rotation_matrix_from_axis_angle(axis, theta)
    print(f"\n从轴角恢复的旋转矩阵误差: {np.linalg.norm(R - R_from_axis):.2e}")
    
    # 旋转矩阵列向量含义
    print("\n" + "-" * 80)
    print("旋转矩阵列向量的几何意义:")
    print(f"  第1列（世界X轴在相机系中）: {R[:, 0]}")
    print(f"  第2列（世界Y轴在相机系中）: {R[:, 1]}")
    print(f"  第3列（世界Z轴在相机系中）: {R[:, 2]}")
    
    # ==================== 投影演示 ====================
    print("\n" + "=" * 80)
    print("【3】投影演示")
    print("=" * 80)
    
    t = np.array([1.0, 2.0, 5.0])
    print(f"\n平移向量 t: {t}")
    
    camera_center = camera.get_camera_center(R, t)
    print(f"相机中心（世界坐标系）: {camera_center}")
    
    # 投影单个点
    point_3d = np.array([2.0, 3.0, 4.0])
    print(f"\n3D点（世界坐标）: {point_3d}")
    
    try:
        pixel, point_cam = camera.project_point(point_3d, R, t)
        print(f"相机坐标: {point_cam}")
        print(f"像素坐标: ({pixel[0]:.2f}, {pixel[1]:.2f})")
    except ValueError as e:
        print(f"投影失败: {e}")
    
    # 批量投影
    print("\n批量投影演示:")
    points_3d = np.array([
        [1, 1, 2],
        [2, 1, 3],
        [1, 2, 3],
        [2, 2, 4]
    ])
    pixels, points_cam = camera.project_points(points_3d, R, t)
    
    for i, (p3d, px) in enumerate(zip(points_3d, pixels)):
        print(f"  点{i+1}: {p3d} -> ({px[0]:.1f}, {px[1]:.1f})")
    
    # 反投影
    print("\n反投影演示:")
    depth = 3.0
    pixel_test = (400, 300)
    point_backproj = camera.backproject_pixel(pixel_test, depth)
    print(f"  像素 {pixel_test}, 深度 {depth} -> 相机坐标 {point_backproj}")
    
    # ==================== 投影矩阵分解 ====================
    print("\n" + "=" * 80)
    print("【4】投影矩阵分解演示")
    print("=" * 80)
    
    P = camera.compute_projection_matrix(R, t)
    print("\n投影矩阵 P = K[R|t]:")
    print(P)
    
    # 分解
    K_rec, R_rec, t_rec, C_rec = camera.decompose_projection_matrix(P)
    
    print("\n分解结果:")
    print("内参矩阵 K:")
    print(K_rec)
    print("\n旋转矩阵 R:")
    print(R_rec)
    print(f"\n平移向量 t: {t_rec}")
    print(f"相机中心 C: {C_rec}")
    
    # 验证
    print("\n验证分解:")
    print(f"  K误差: {np.linalg.norm(camera.K - K_rec):.2e}")
    print(f"  R误差: {np.linalg.norm(R - R_rec):.2e}")
    print(f"  t误差: {np.linalg.norm(t - t_rec):.2e}")
    
    # ==================== 单应矩阵演示 ====================
    print("\n" + "=" * 80)
    print("【5】单应矩阵演示")
    print("=" * 80)
    
    # 地平面 Z=0
    n = np.array([0, 0, 1])  # 法向量
    d = 0  # 距离
    
    print(f"\n平面法向量: {n}")
    print(f"平面距离原点: {d}")
    
    # 对于Z=0平面，单应矩阵简化为 H = K[r1 r2 t]
    H = camera.K @ np.column_stack([R[:, 0], R[:, 1], t])
    H = H / H[2, 2]
    
    print("\n单应矩阵 H:")
    print(H)
    
    # 测试单应变换
    print("\n单应变换测试:")
    point_on_plane = np.array([2.0, 3.0])  # 平面上的点
    point_homo = np.array([point_on_plane[0], point_on_plane[1], 1.0])
    
    pixel_homo = H @ point_homo
    pixel_from_H = pixel_homo[:2] / pixel_homo[2]
    
    # 通过完整投影验证
    point_3d_plane = np.array([point_on_plane[0], point_on_plane[1], 0.0])
    pixel_from_proj, _ = camera.project_point(point_3d_plane, R, t)
    
    print(f"  平面点: ({point_on_plane[0]}, {point_on_plane[1]})")
    print(f"  通过H投影: ({pixel_from_H[0]:.2f}, {pixel_from_H[1]:.2f})")
    print(f"  通过完整投影: ({pixel_from_proj[0]:.2f}, {pixel_from_proj[1]:.2f})")
    print(f"  误差: {np.linalg.norm(pixel_from_H - pixel_from_proj):.2e}")
    
    # DLT估计单应矩阵
    print("\n" + "-" * 80)
    print("DLT单应矩阵估计:")
    
    # 生成对应点
    src_pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    dst_pts_homo = (H @ np.column_stack([src_pts, np.ones(4)]).T).T
    dst_pts = dst_pts_homo[:, :2] / dst_pts_homo[:, 2:3]
    
    H_estimated = camera.estimate_homography_dlt(src_pts, dst_pts)
    
    print("估计的单应矩阵:")
    print(H_estimated)
    print(f"\n与真实H的误差: {np.linalg.norm(H / H[2,2] - H_estimated / H_estimated[2,2]):.2e}")
    
    # ==================== Look At 演示 ====================
    print("\n" + "=" * 80)
    print("【6】Look At 相机位姿")
    print("=" * 80)
    
    cam_pos = np.array([5, 5, 10])
    target = np.array([0, 0, 0])
    
    R_lookat, t_lookat = camera.look_at(cam_pos, target)
    
    print(f"\n相机位置: {cam_pos}")
    print(f"目标位置: {target}")
    print(f"\nLook-at旋转矩阵:")
    print(R_lookat)
    print(f"\nLook-at平移向量: {t_lookat}")
    
    camera.verify_rotation_matrix(R_lookat)
    
    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    demo_camera_geometry()
