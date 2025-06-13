import matplotlib.pyplot as plt

# 初始化空列表
dis_values = []
lift_values = []
rew1 = []
dis_values2 = []
lift_values2 = []
# 读取txt文件
with open('output_651.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('action_rate_l2 :'):
            value = float(line.split(':')[1])
            dis_values.append(value)
        elif line.startswith('joint_vel_l2 :'):
            value = float(line.split(':')[1])
            lift_values.append(value)

with open('output_652.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('action_rate_l2 :'):
            value = float(line.split(':')[1])
            dis_values2.append(value)
        elif line.startswith('joint_vel_l2 :'):
            value = float(line.split(':')[1])
            lift_values2.append(value)

# 绘图
plt.figure(figsize=(10,6))
# plt.plot(range(len(dis_values)), dis_values, label='dis', marker='o',markersize=0.1)
# plt.plot(range(len(rew1)), rew1, label='rew', marker='o',markersize=0.1)
# plt.plot(range(len(lift_values)), lift_values, label='lift', marker='x',markersize=0.1)
plt.scatter([i for i in range(len(lift_values2))], lift_values2, color='green', s=1, marker='^',label='before')
plt.scatter([i for i in range(len(lift_values))], lift_values, color='red', s=1, marker='^',label='after')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Joint Vel curves')
plt.ylim(0, 100)
plt.legend()
plt.grid(True)
plt.show()
