import numpy as np

# 加载 .npy 文件
data = np.load("obs_logs/obs_step_0.npy")  # 替换为你的文件路径

# 打印数据基本信息
print("Shape:", data.shape)      # 数组形状
print("Data type:", data.dtype)  # 数据类型
print("First few values:\n", data)  # 打印数据（若数组过大，只会显示部分）