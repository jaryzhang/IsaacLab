import numpy as np

# ===== 工具函数 =====
def rotation_matrix_from_euler(rx, ry, rz):
    """
    根据欧拉角 (rx, ry, rz，单位: 弧度) 生成旋转矩阵
    按顺序: ZYX (航向, 俯仰, 横滚)
    """
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]])

    Ry = np.array([[cy, 0, sy],
                   [ 0, 1,  0],
                   [-sy,0, cy]])

    Rx = np.array([[1, 0,  0],
                   [0, cx, -sx],
                   [0, sx,  cx]])

    return Rz @ Ry @ Rx


def world_to_pixel(Pw, C, R, fov_x, fov_y, W, H):
    """
    Pw: 世界坐标点 [X, Y, Z]
    C: 摄像机位置 [Cx, Cy, Cz]
    R: 摄像机旋转矩阵 (世界->相机)
    fov_x, fov_y: 视场角 (弧度)
    W, H: 图像分辨率 (像素)
    """
    # 1. 计算焦距 fx, fy
    fx = W / (2 * np.tan(fov_x / 2))
    fy = H / (2 * np.tan(fov_y / 2))
    cx, cy = W / 2, H / 2

    # 2. 世界坐标 -> 相机坐标
    Pw = np.array(Pw)
    C = np.array(C)
    Pc = R @ (Pw - C)

    # 如果点在相机后面 (Z<=0)，投影无效
    if Pc[2] <= 0:
        return None

    # 3. 相机坐标 -> 像素坐标
    u = fx * (Pc[0] / Pc[2]) + cx
    v = fy * (Pc[1] / Pc[2]) + cy

    return (u, v)


# ===== 示例使用 =====
if __name__ == "__main__":
    # 摄像机参数
    C = [0, 0.19, 0]                 # 摄像机在世界坐标的位置
    R = rotation_matrix_from_euler(3.14, -1.53, 1.57)  # 摄像机朝向 (无旋转，Z 轴朝前)

    fov_x = np.radians(115.6)  # 水平视场角 90°
    fov_y = np.radians(94.7)  # 垂直视场角 60°
    W, H = 200, 150         # 图像分辨率

    # 物体坐标 (在世界坐标系下)
    Pw = [0.28, 0.04, 0]  # 位于摄像机前方 5 米

    pixel = world_to_pixel(Pw, C, R, fov_x, fov_y, W, H)
    print("像素坐标:", pixel)