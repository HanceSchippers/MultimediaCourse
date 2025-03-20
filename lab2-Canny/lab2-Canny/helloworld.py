import numpy as np
import matplotlib.pyplot as plt

# 参数设置
k_min, k_max = 0.0, 2.0  # k 的范围
num_k = 1000             # k 的采样点数
transient = 1000         # 用于消除初始瞬态的迭代次数
plot_iterations = 300    # 在图上绘制的后续迭代次数
x0 = 0.1                 # 初始值

# 创建 k 的序列
k_values = np.linspace(k_min, k_max, num_k)

# 准备绘图数据存储
k_plot = []
x_plot = []

for k in k_values:
    x = x0
    # 消除初态影响的迭代
    for _ in range(transient):
        x = 1 - k * x**2
    # 开始记录稳定后的值
    for _ in range(plot_iterations):
        x = 1 - k * x**2
        k_plot.append(k)
        x_plot.append(x)

# 绘图
plt.figure(figsize=(10,6))
plt.scatter(k_plot, x_plot, s=0.1, color='black')
plt.title("Bifurcation diagram for X_{n+1} = 1 - kX_n^2")
plt.xlabel('k')
plt.ylabel('x')
plt.xlim(k_min, k_max)
plt.ylim(-1, 2)  # 根据需要调整
plt.show()
