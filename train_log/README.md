
## 文件命名规则
所有日志文件按**时间步**命名，格式为：  
`{type}_step_{step_number}.npy`  
例如：  
- `act_step_0.npy`（第0步的动作）


---

## 数据格式说明

### 1. 动作（action）数据
- **路径**: `act_logs/`
- **格式**: `.npy` (NumPy 二进制文件)
- **内容**: 每步执行的动作向量（维度取决于具体环境）。

### 2. 观测（observation）数据
- **路径**: `obs_logs/`
- **格式**: `.npy`
- **内容**: 
  - 形状为 `(128, 128, 3)` 的三通道 RGB 像素数组(归一化后)。

### 3. 奖励（reward）数据
- **路径**: `rew_logs/`
- **格式**: `.npy`
- **内容**: 每步的奖励值。

---

## 可视化数据

### 1. 还原图像（IMAGES）
- **路径**: `IMAGES/`
- **格式**: `.png`
- **生成方式**:  
  由 `obs_logs/` 中的 `.npy` 文件转换而来，直接保存为图像格式。  
  命名与观测文件一致。

### 2. 训练视频（videos）
- **路径**: `rl-video-step-0.mp4`
- **格式**: `.mp4` 
- **内容**:  
  Isaac Sim 训练时录制的视频（全局视角）。

---

## 如何查看数据
### 1. 查看 `.npy` 文件
```python
import numpy as np
data = np.load("obs_logs/obs_step_0.npy")  # 示例：加载观测数据
print("Shape:", data.shape)