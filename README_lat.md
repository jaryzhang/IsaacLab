# 项目说明

## 1. 替换 rsl_rl 库

将 `rsl_rl.zip` 解压，替换当前 Isaac Lab 项目所调用的 Python 库中的 `rsl_rl` 文件夹。

## 2. 替换 lift 文件夹

将此文件夹中的 `lift` 文件夹，替换 Isaac Lab 项目内以下路径下的 `lift` 文件夹：

IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift


## 3. 替换其他 .py 文件

将本文件夹中的其他 `.py` 文件，根据下图所示的路径，分别替换 Isaac Lab 项目中的对应文件。

> ⚠️ 请参考下图路径对应替换文件。
> 
> ![alt text](image.png)

## 4. 启动训练任务

训练任务启动脚本命令如下：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Lift-Cube-CoarseArm-v0 --enable_cameras --distributed --headless
```
> ⚠️如果是用pip安裝的Isaac sim需要执行：
```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Lift-Cube-CoarseArm-v0 --enable_cameras --distributed --headless
```

