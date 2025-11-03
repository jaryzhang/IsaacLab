import numpy as np
import open3d as o3d
import cv2

def depth_to_pointcloud(depth_image, fx, fy, cx, cy, rgb_image=None, output_path="pointcloud.ply"):
    """
    将深度图转换为点云（可选带颜色）
    
    参数:
        depth_image : np.ndarray
            深度图（H, W），单位为米。
        fx, fy, cx, cy : float
            相机内参。
        rgb_image : np.ndarray, optional
            彩色图（H, W, 3），与深度图对齐。
        output_path : str
            点云保存路径。
    """
    assert len(depth_image.shape) == 2, "深度图必须是单通道 (H, W)"
    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 深度图中无效值置0（避免NaN）
    depth = np.nan_to_num(depth_image, nan=0.0)
    depth = (depth.max() - depth)

    mask = depth > 0  # 有效深度
    
    # 反投影到3D空间
    Z = depth[mask]
    X = (u[mask] - cx) * Z / fx
    Y = (v[mask] - cy) * Z / fy
    points = np.stack((X, -Y, Z), axis=-1)
    
    # 构建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 如果有彩色图，提取对应颜色
    if rgb_image is not None:
        assert rgb_image.shape[:2] == depth_image.shape, "RGB 与深度图尺寸不匹配"
        rgb = rgb_image.astype(np.float32) / 255.0
        colors = rgb[v[mask], u[mask]]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存点云
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"✅ 点云已保存到: {output_path}, 共 {len(points)} 个点")
    
    return pcd

if __name__ == "__main__":
    # 读取深度图（米为单位）
    depth = cv2.imread("1.png", cv2.IMREAD_UNCHANGED).astype(np.float32)

    # 相机内参
    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5

    # 调用函数
    pcd = depth_to_pointcloud(depth, fx, fy, cx, cy, output_path="colored_pointcloud.ply")
    pcd_down = pcd.voxel_down_sample(voxel_size=5)
    # 可视化
    o3d.visualization.draw_geometries([pcd_down])