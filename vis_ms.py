import numpy as np
import matplotlib.pyplot as plt

m1 = []
s1 = []
m2 = []
s2 = []
m3 = []
s3 = []
m4 = []
s4 = []
m5 = []
s5 = []
m6 = []
s6 = []

# 读取txt文件
with open('output_651.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('joint_vel_l2 :'):
            value = float(line.split(':')[1])
            m1.append(value)
        elif line.startswith('action_rate_l2 :'):
            value = float(line.split(':')[1])
            s1.append(value)
        elif line.startswith('obs2 m:'):
            value = float(line.split(':')[1])
            m2.append(value)
        elif line.startswith('obs2 s:'):
            value = float(line.split(':')[1])
            s2.append(value)
        elif line.startswith('obs3 m:'):
            value = float(line.split(':')[1])
            m3.append(value)
        elif line.startswith('obs3 s:'):
            value = float(line.split(':')[1])
            s3.append(value)
        elif line.startswith('obs4 m:'):
            value = float(line.split(':')[1])
            m4.append(value)
        elif line.startswith('obs4 s:'):
            value = float(line.split(':')[1])
            s4.append(value)
        elif line.startswith('obs5 m:'):
            value = float(line.split(':')[1])
            m5.append(value)
        elif line.startswith('obs5 s:'):
            value = float(line.split(':')[1])
            s5.append(value)
        elif line.startswith('obs6 m:'):
            value = float(line.split(':')[1])
            m6.append(value)
        elif line.startswith('obs6 s:'):
            value = float(line.split(':')[1])
            s6.append(value)

with open('output_652.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('joint_vel_l2 :'):
            value = float(line.split(':')[1])
            m2.append(value)
        elif line.startswith('action_rate_l2 :'):
            value = float(line.split(':')[1])
            s2.append(value)
# 示例数据：3组实验数据
groups1 = np.array([i for i in range(len(m1))])
groups2 = np.array([i for i in range(len(m2))])
groups3 = np.array([i for i in range(len(s1))])
groups4 = np.array([i for i in range(len(s2))])
groups5 = np.array([i for i in range(len(m5))])
groups6 = np.array([i for i in range(len(m6))])
# means = [5.5, 7.2, 4.8]
# std_devs = [0.8, 1.1, 0.6]

# groups2 = np.array([i for i in range(len(m1))])+2000
# groups3 = np.array([i for i in range(len(m1))])+4000
# groups4 = np.array([i for i in range(len(m1))])+6000
# groups5 = np.array([i for i in range(len(m1))])+8000
# groups6 = np.array([i for i in range(len(m1))])+10000
colors = [('darkred', 'lightcoral'), ('darkorange', 'gold'), 
          ('darkblue', 'deepskyblue'), ('darkgreen', 'lightgreen'), 
          ('indigo', 'plum'),('saddlebrown', 'burlywood')]
# 绘制点线图 (Mean ± SD)
plt.figure(figsize=(10, 8))

# plt.errorbar(groups1, m1, yerr=s1, fmt='o-', capsize=0.05, capthick=0.001, 
#              ecolor=colors[0][1], color=colors[0][0], label=f'joint pos = {np.mean(m1)}',markersize=0.1)
# plt.errorbar(groups2, m2, yerr=s2, fmt='o-', capsize=0.05, capthick=0.001, 
#              ecolor=colors[1][1], color=colors[1][0], label=f'joint vel = {np.mean(m2)}',markersize=0.1)
# plt.errorbar(groups3, m3, yerr=s3, fmt='o-', capsize=0.05, capthick=0.001, 
#              ecolor=colors[2][1], color=colors[2][0], label=f'obj pos = {np.mean(m3)}',markersize=0.1)
# plt.errorbar(groups4, m4, yerr=s4, fmt='o-', capsize=0.05, capthick=0.001, 
#              ecolor=colors[3][1], color=colors[3][0], label=f'target pos = {np.mean(m4)}',markersize=0.1)
# plt.errorbar(groups5, m5, yerr=s5, fmt='o-', capsize=0.05, capthick=0.001, 
#              ecolor=colors[4][1], color=colors[4][0], label=f'action = {np.mean(m5)}',markersize=0.1)
# plt.errorbar(groups6, m6, yerr=s6, fmt='o-', capsize=0.05, capthick=0.001, 
#              ecolor=colors[5][1], color=colors[5][0], label=f'image feature = {np.mean(m6)}',markersize=0.1)

plt.errorbar(groups1, m1, fmt='o-', capsize=0.05, capthick=0.001, 
             ecolor=colors[0][1], color=colors[0][0], label=f'joint pos = {np.mean(m1)}',markersize=0.1)
plt.errorbar(groups2, m2, fmt='o-', capsize=0.05, capthick=0.001, 
             ecolor=colors[1][1], color=colors[1][0], label=f'joint vel = {np.mean(m2)}',markersize=0.1)
plt.errorbar(groups3, m3, fmt='o-', capsize=0.05, capthick=0.001, 
             ecolor=colors[2][1], color=colors[2][0], label=f'obj pos = {np.mean(m3)}',markersize=0.1)
plt.errorbar(groups4, m4, fmt='o-', capsize=0.05, capthick=0.001, 
             ecolor=colors[3][1], color=colors[3][0], label=f'target pos = {np.mean(m4)}',markersize=0.1)
plt.errorbar(groups5, m5, fmt='o-', capsize=0.05, capthick=0.001, 
             ecolor=colors[4][1], color=colors[4][0], label=f'action = {np.mean(m5)}',markersize=0.1)
plt.errorbar(groups6, m6, fmt='o-', capsize=0.05, capthick=0.001, 
             ecolor=colors[5][1], color=colors[5][0], label=f'image feature = {np.mean(m6)}',markersize=0.1)

# plt.scatter(groups1,m1, s=0.1, c=colors[0][0], alpha=0.7)
# plt.scatter(groups2,m2, s=0.1, c=colors[1][0], alpha=0.7)
# plt.scatter(groups3,m3, s=0.1, c=colors[2][0], alpha=0.7)
# plt.scatter(groups4,m4, s=0.1, c=colors[3][0], alpha=0.7)
# plt.scatter(groups5,m5, s=0.1, c=colors[4][0], alpha=0.7)
# plt.scatter(groups6,m6, s=0.1, c=colors[5][0], alpha=0.7)
plt.title("Observation part Mean Plot")
plt.xlabel("index")
plt.ylabel("Values")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()