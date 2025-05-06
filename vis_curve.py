import matplotlib.pyplot as plt

# 初始化空列表
dis_values = []
lift_values = []
rew1 = []

# 读取txt文件
with open('output_formres2.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('dis:'):
            value = float(line.split(':')[1])
            dis_values.append(value)
        elif line.startswith('lift:'):
            value = float(line.split(':')[1])
            lift_values.append(value)

with open('output_formres4.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('rew1: '):
            value = float(line.split(':')[1])
            rew1.append(value)

# 绘图
plt.figure(figsize=(10,6))
# plt.plot(range(len(dis_values)), dis_values, label='dis', marker='o',markersize=0.1)
plt.plot(range(len(rew1)), rew1, label='rew', marker='o',markersize=0.1)
# plt.plot(range(len(lift_values)), lift_values, label='lift', marker='x',markersize=0.1)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Dis and Lift curves')
plt.legend()
plt.grid(True)
plt.show()
