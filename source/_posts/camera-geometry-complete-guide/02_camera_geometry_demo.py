"""
相机几何完整演示
包含所有核心功能的实现和数值示例
"""

import numpy as np
from scipy.linalg import rq
import matplotlib.pyplot as plt

class CameraGeometry:
    """相机几何计算类"""
    
    def __init__(self, fx, fy, cx, cy, width=None, height=None, skew=0):
        """
        初始化相机内参
        
        参数:
            fx, fy: 焦距（像素单位）
            cx, cy: 主点坐标（像素）
            width, height: 图像尺寸（可选）
            skew: 倾斜系数（通常为0）
        """
        self.K = np.array([
            [fx, skew, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        self.width = width
        self.height = height
        
    def get_intrinsics(self):
        """获取内参矩阵"""
        return self.K.copy()
    
    # =============== 旋转矩阵生成 ===============
    
    @staticmethod
    def rotation_matrix_from_euler(roll, pitch, yaw, order='zyx'):
        """
        从欧拉角生成旋转矩阵
        
        参数:
            roll: 绕X轴旋转角度（弧度）
            pitch: 绕Y轴旋转角度（弧度）
            yaw: 绕Z轴旋转角度（弧度）
            order: 旋转顺序 ('xyz', 'zyx', 'zxy'等)
        
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
        
        # 根据顺序组合
        rotation_map = {'x': Rx, 'y': Ry, 'z': Rz}
        R = np.eye(3)
        for axis in order:
            R = rotation_map[axis] @ R
        
        return R
    
    @staticmethod
    def rotation_matrix_from_axis_angle(axis, theta):
        """
        从轴角表示生成旋转矩阵（罗德里格斯公式）
        
        参数:
            axis: 旋转轴（3D向量，会被自动归一化）
            theta: 旋转角度（弧度）
        
        返回:
            R: 3x3旋转矩阵
        """
        axis = np.array(axis, dtype=np.float64)
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
    def rotation_matrix_from_quaternion(q):
        """
        从四元数生成旋转矩阵
        
        参数:
            q: 四元数 [q0, q1, q2, q3] 或 [qw, qx, qy, qz]
               要求: q0^2 + q1^2 + q2^2 + q3^2 = 1
        
        返回:
            R: 3x3旋转矩阵
        """
        q = np.array(q, dtype=np.float64)
        q = q / np.linalg.norm(q)  # 归一化
        
        q0, q1, q2, q3 = q
        
        R = np.array([
            [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
        ])
        
        return R
    
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
            # 提取欧拉角（ZYX顺序）
            pitch = np.arcsin(-R[2, 0])
            
            if np.abs(np.cos(pitch)) > 1e-6:
                roll = np.arctan2(R[2, 1], R[2, 2])
                yaw = np.arctan2(R[1, 0], R[0, 0])
            else:
                # 万向锁情况
                roll = 0
                yaw = np.arctan2(-R[0, 1], R[1, 1])
            
            return roll, pitch, yaw
        else:
            raise NotImplementedError(f"Order '{order}' not implemented")
    
    # =============== 投影相关 ===============
    
    def project_point(self, point_world, R, t):
        """
        将世界坐标点投影到图像
        
        参数:
            point_world: 3D点（世界坐标系） [X_w, Y_w, Z_w]
            R: 3x3旋转矩阵
            t: 3x1平移向量
        
        返回:
            pixel: 像素坐标 [u, v]
            point_camera: 相机坐标系中的点
        """
        point_world = np.array(point_world, dtype=np.float64)
        t = np.array(t, dtype=np.float64)
        
        # 世界坐标 -> 相机坐标
        point_camera = R @ point_world + t
        
        # 检查点是否在相机前方
        if point_camera[2] <= 0:
            raise ValueError(f"Point is behind camera (Z={point_camera[2]:.3f})")
        
        # 相机坐标 -> 像素坐标
        point_homo = self.K @ point_camera
        pixel = point_homo[:2] / point_homo[2]
        
        return pixel, point_camera
    
    def project_points(self, points_world, R, t):
        """
        批量投影3D点
        
        参数:
            points_world: Nx3数组
            R, t: 外参
        
        返回:
            pixels: Nx2数组
        """
        points_world = np.array(points_world, dtype=np.float64)
        if points_world.ndim == 1:
            points_world = points_world.reshape(1, -1)
        
        # 转换到相机坐标系
        points_camera = (R @ points_world.T).T + t
        
        # 透视投影
        points_homo = (self.K @ points_camera.T).T
        pixels = points_homo[:, :2] / points_homo[:, 2:3]
        
        return pixels
    
    def compute_projection_matrix(self, R, t):
        """
        计算投影矩阵 P = K[R|t]
        
        参数:
            R: 3x3旋转矩阵
            t: 3x1平移向量
        
        返回:
            P: 3x4投影矩阵
        """
        t = np.array(t, dtype=np.float64).reshape(-1, 1)
        return self.K @ np.hstack([R, t])
    
    # =============== 矩阵分解 ===============
    
    @staticmethod
    def decompose_projection_matrix(P):
        """
        分解投影矩阵 P = K[R|t]
        
        参数:
            P: 3x4投影矩阵
        
        返回:
            K: 3x3内参矩阵
            R: 3x3旋转矩阵
            t: 3x1平移向量
            camera_center: 3x1相机中心在世界坐标系中的位置
        """
        P = np.array(P, dtype=np.float64)
        
        # 分离M和p4
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
        
        # 归一化K（使K[2,2]=1）
        K = K / K[2, 2]
        
        # 提取平移向量
        t = np.linalg.inv(K) @ p4
        
        # 计算相机中心
        camera_center = -R.T @ t
        
        return K, R, t, camera_center
    
    # =============== 单应矩阵 ===============
    
    def compute_homography_for_plane(self, R, t, plane_normal, plane_distance):
        """
        计算平面的单应矩阵
        
        参数:
            R, t: 外参
            plane_normal: 平面法向量（世界坐标系）
            plane_distance: 平面到原点的距离
        
        返回:
            H: 3x3单应矩阵
        """
        n = np.array(plane_normal, dtype=np.float64).reshape(-1, 1)
        t = np.array(t, dtype=np.float64).reshape(-1, 1)
        
        # H = K(R - t*n^T/d)K^(-1)
        H = self.K @ (R - (t @ n.T) / plane_distance) @ np.linalg.inv(self.K)
        
        # 归一化
        return H / H[2, 2]
    
    @staticmethod
    def decompose_homography(H, K):
        """
        从单应矩阵恢复旋转和平移（针对平面Z=0）
        
        参数:
            H: 3x3单应矩阵
            K: 3x3内参矩阵
        
        返回:
            R: 3x3旋转矩阵
            t: 3x1平移向量
        """
        # H = K[r1 r2 t]，其中r1, r2是R的前两列
        temp = np.linalg.inv(K) @ H
        
        # 归一化
        scale = 1.0 / np.linalg.norm(temp[:, 0])
        temp = temp * scale
        
        r1 = temp[:, 0]
        r2 = temp[:, 1]
        t = temp[:, 2]
        
        # 计算第三列
        r3 = np.cross(r1, r2)
        
        # 组合成旋转矩阵
        R_temp = np.column_stack([r1, r2, r3])
        
        # 确保正交性（通过SVD）
        U, S, Vt = np.linalg.svd(R_temp)
        R = U @ Vt
        
        # 确保det(R) = 1
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        
        return R, t
    
    # =============== 工具函数 ===============
    
    @staticmethod
    def verify_rotation_matrix(R, tol=1e-6):
        """
        验证旋转矩阵的有效性
        
        参数:
            R: 待验证的矩阵
            tol: 容差
        
        返回:
            is_valid: 布尔值
            error_msg: 错误信息
        """
        # 检查正交性
        I = R.T @ R
        orthogonality_error = np.linalg.norm(I - np.eye(3))
        
        # 检查行列式
        det = np.linalg.det(R)
        det_error = abs(det - 1.0)
        
        if orthogonality_error > tol:
            return False, f"Not orthogonal (error={orthogonality_error:.2e})"
        
        if det_error > tol:
            return False, f"Determinant != 1 (det={det:.6f})"
        
        return True, "Valid rotation matrix"
    
    def is_point_in_front(self, point_world, R, t):
        """检查点是否在相机前方"""
        point_camera = R @ np.array(point_world) + np.array(t)
        return point_camera[2] > 0

# =============== 演示和测试 ===============

def demo_basic_projection():
    """基本投影演示"""
    print("=" * 70)
    print("演示1: 基本投影流程")
    print("=" * 70)
    
    # 创建相机
    camera = CameraGeometry(
        fx=800, fy=800,
        cx=320, cy=240,
        width=640, height=480
    )
    
    print("\n【相机内参矩阵 K】")
    print(camera.K)
    
    # 创建外参
    roll, pitch, yaw = np.deg2rad([10, 20, 30])
    R = camera.rotation_matrix_from_euler(roll, pitch, yaw, order='zyx')
    t = np.array([1.0, 2.0, 5.0])
    
    print("\n【旋转矩阵 R】")
    print(R)
    print(f"det(R) = {np.linalg.det(R):.10f}")
    print(f"正交性检查 ||R^T R - I|| = {np.linalg.norm(R.T @ R - np.eye(3)):.2e}")
    
    print("\n【平移向量 t】")
    print(t)
    
    # 相机位置
    camera_center = -R.T @ t
    print(f"\n【相机中心（世界坐标系）】")
    print(f"C = -R^T t = {camera_center}")
    
    # 投影3D点
    point_3d = np.array([2.0, 3.0, 4.0])
    print(f"\n【3D点（世界坐标）】")
    print(f"P_w = {point_3d}")
    
    pixel, point_camera = camera.project_point(point_3d, R, t)
    print(f"\n【相机坐标】")
    print(f"P_c = {point_camera}")
    print(f"\n【像素坐标】")
    print(f"p = ({pixel[0]:.2f}, {pixel[1]:.2f})")
    
    # 验证
    print(f"\n【验证】点在相机前方: {point_camera[2] > 0}")
    print(f"深度 Z_c = {point_camera[2]:.3f} m")

def demo_rotation_representations():
    """旋转表示方法演示"""
    print("\n" + "=" * 70)
    print("演示2: 旋转矩阵的多种表示方法")
    print("=" * 70)
    
    camera = CameraGeometry(fx=800, fy=800, cx=320, cy=240)
    
    # 方法1: 欧拉角
    print("\n【方法1: 欧拉角】")
    roll, pitch, yaw = np.deg2rad([15, 30, 45])
    print(f"欧拉角 (度): Roll={np.rad2deg(roll):.1f}, Pitch={np.rad2deg(pitch):.1f}, Yaw={np.rad2deg(yaw):.1f}")
    R1 = camera.rotation_matrix_from_euler(roll, pitch, yaw, order='zyx')
    print("旋转矩阵:")
    print(R1)
    
    # 方法2: 轴角
    print("\n【方法2: 轴角表示】")
    axis = np.array([1, 1, 0])  # 绕(1,1,0)轴
    theta = np.deg2rad(60)
    print(f"旋转轴: {axis / np.linalg.norm(axis)}")
    print(f"旋转角: {np.rad2deg(theta):.1f}度")
    R2 = camera.rotation_matrix_from_axis_angle(axis, theta)
    print("旋转矩阵:")
    print(R2)
    
    # 方法3: 四元数
    print("\n【方法3: 四元数表示】")
    q = [np.cos(theta/2), *(axis/np.linalg.norm(axis) * np.sin(theta/2))]
    print(f"四元数: q = [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
    R3 = camera.rotation_matrix_from_quaternion(q)
    print("旋转矩阵:")
    print(R3)
    
    # 验证
    print("\n【验证轴角和四元数的等价性】")
    print(f"R2 和 R3 的差异: {np.linalg.norm(R2 - R3):.2e}")

def demo_rotation_column_meaning():
    """旋转矩阵列向量含义演示"""
    print("\n" + "=" * 70)
    print("演示3: 旋转矩阵列向量的几何意义")
    print("=" * 70)
    
    camera = CameraGeometry(fx=800, fy=800, cx=320, cy=240)
    
    # 创建一个简单的旋转（绕Z轴45度）
    theta = np.deg2rad(45)
    R = camera.rotation_matrix_from_euler(0, 0, theta, order='zyx')
    
    print(f"\n绕Z轴旋转45度的旋转矩阵:")
    print(R)
    
    print("\n【列向量的物理意义】")
    print(f"第1列 r1 = {R[:, 0]}")
    print(f"  → 世界坐标系的X轴在相机坐标系中的方向")
    print(f"  → 长度: {np.linalg.norm(R[:, 0]):.6f} (应该为1)")
    
    print(f"\n第2列 r2 = {R[:, 1]}")
    print(f"  → 世界坐标系的Y轴在相机坐标系中的方向")
    print(f"  → 长度: {np.linalg.norm(R[:, 1]):.6f}")
    
    print(f"\n第3列 r3 = {R[:, 2]}")
    print(f"  → 世界坐标系的Z轴在相机坐标系中的方向")
    print(f"  → 长度: {np.linalg.norm(R[:, 2]):.6f}")
    
    print("\n【列向量的正交性】")
    print(f"r1 · r2 = {np.dot(R[:, 0], R[:, 1]):.2e} (应该为0)")
    print(f"r1 · r3 = {np.dot(R[:, 0], R[:, 2]):.2e}")
    print(f"r2 · r3 = {np.dot(R[:, 1], R[:, 2]):.2e}")
    
    print("\n【行向量的物理意义】")
    print(f"R^T 的列（即R的行）表示相机坐标系的基向量在世界坐标系中的方向")
    print(f"第1行 = {R[0, :]} → 相机X轴在世界系中的方向")
    print(f"第2行 = {R[1, :]} → 相机Y轴在世界系中的方向")
    print(f"第3行 = {R[2, :]} → 相机Z轴（光轴）在世界系中的方向")

def demo_projection_matrix_decomposition():
    """投影矩阵分解演示"""
    print("\n" + "=" * 70)
    print("演示4: 投影矩阵分解")
    print("=" * 70)
    
    # 创建原始相机参数
    camera = CameraGeometry(fx=850, fy=860, cx=325, cy=245)
    
    roll, pitch, yaw = np.deg2rad([12, 25, 35])
    R_original = camera.rotation_matrix_from_euler(roll, pitch, yaw, order='zyx')
    t_original = np.array([1.5, 2.5, 6.0])
    
    print("\n【原始参数】")
    print("内参矩阵 K:")
    print(camera.K)
    print("\n旋转矩阵 R:")
    print(R_original)
    print("\n平移向量 t:")
    print(t_original)
    
    # 计算投影矩阵
    P = camera.compute_projection_matrix(R_original, t_original)
    print("\n【投影矩阵 P = K[R|t]】")
    print(P)
    
    # 分解投影矩阵
    K_recovered, R_recovered, t_recovered, center_recovered = \
        camera.decompose_projection_matrix(P)
    
    print("\n【分解后的参数】")
    print("内参矩阵 K (恢复):")
    print(K_recovered)
    print("\n旋转矩阵 R (恢复):")
    print(R_recovered)
    print("\n平移向量 t (恢复):")
    print(t_recovered)
    print("\n相机中心 C (恢复):")
    print(center_recovered)
    
    # 验证
    print("\n【误差分析】")
    print(f"K 误差: {np.linalg.norm(camera.K - K_recovered):.2e}")
    print(f"R 误差: {np.linalg.norm(R_original - R_recovered):.2e}")
    print(f"t 误差: {np.linalg.norm(t_original - t_recovered):.2e}")
    
    # 相机中心验证
    center_original = -R_original.T @ t_original
    print(f"相机中心误差: {np.linalg.norm(center_original - center_recovered):.2e}")

def demo_homography():
    """单应矩阵演示"""
    print("\n" + "=" * 70)
    print("演示5: 单应矩阵（平面投影）")
    print("=" * 70)
    
    camera = CameraGeometry(fx=800, fy=800, cx=320, cy=240)
    
    # 外参
    R = camera.rotation_matrix_from_euler(0, np.deg2rad(30), 0, order='xyz')
    t = np.array([0, 2, 5])
    
    print("\n【场景设置】")
    print("地平面: Z = 0")
    print(f"相机高度: {-(-R.T @ t)[2]:.2f} m")
    
    # 计算单应矩阵（对于Z=0平面）
    # H = K[r1 r2 t]
    H_simple = camera.K @ np.column_stack([R[:, 0], R[:, 1], t])
    H_simple = H_simple / H_simple[2, 2]
    
    print("\n【单应矩阵 H】")
    print(H_simple)
    
    # 测试点
    points_2d = np.array([
        [1, 0],
        [0, 1],
        [2, 3],
        [-1, 2]
    ])
    
    print("\n【平面上的点投影】")
    for i, (x, y) in enumerate(points_2d):
        # 方法1: 通过单应矩阵
        p_homo = H_simple @ np.array([x, y, 1])
        pixel_h = p_homo[:2] / p_homo[2]
        
        # 方法2: 通过完整3D投影
        point_3d = np.array([x, y, 0])
        pixel_full, _ = camera.project_point(point_3d, R, t)
        
        print(f"点 ({x}, {y}, 0):")
        print(f"  单应变换: ({pixel_h[0]:.2f}, {pixel_h[1]:.2f})")
        print(f"  完整投影: ({pixel_full[0]:.2f}, {pixel_full[1]:.2f})")
        print(f"  误差: {np.linalg.norm(pixel_h - pixel_full):.2e}")
    
    # 从单应矩阵恢复R和t
    print("\n【从单应矩阵恢复外参】")
    R_recovered, t_recovered = camera.decompose_homography(H_simple, camera.K)
    
    print("恢复的旋转矩阵:")
    print(R_recovered)
    print(f"\n旋转矩阵误差: {np.linalg.norm(R - R_recovered):.2e}")
    
    print("\n恢复的平移向量:")
    print(t_recovered)
    print(f"平移向量误差: {np.linalg.norm(t - t_recovered):.2e}")

def demo_batch_projection():
    """批量投影演示"""
    print("\n" + "=" * 70)
    print("演示6: 批量点投影")
    print("=" * 70)
    
    camera = CameraGeometry(fx=800, fy=800, cx=320, cy=240)
    
    R = camera.rotation_matrix_from_euler(0, 0, np.deg2rad(15), order='xyz')
    t = np.array([0, 0, 10])
    
    # 生成3D点云（立方体）
    cube_points = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [3, 5]:
                cube_points.append([x, y, z])
    
    cube_points = np.array(cube_points)
    
    print(f"\n投影 {len(cube_points)} 个3D点...")
    pixels = camera.project_points(cube_points, R, t)
    
    print("\n【投影结果】")
    for i, (point_3d, pixel) in enumerate(zip(cube_points, pixels)):
        print(f"点{i+1}: {point_3d} → ({pixel[0]:.1f}, {pixel[1]:.1f})")

# =============== 主程序 ===============

if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  相机几何完整演示 - Camera Geometry Complete Demo".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    demo_basic_projection()
    demo_rotation_representations()
    demo_rotation_column_meaning()
    demo_projection_matrix_decomposition()
    demo_homography()
    demo_batch_projection()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
