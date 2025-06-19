import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import inspect
from itertools import product, combinations

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)

pictures_dir = os.path.join(this_directory, "pictures")
if not os.path.exists(pictures_dir):
    os.makedirs(pictures_dir)
DSE_plot_path = os.path.join(pictures_dir, "DSE_Split.png")



fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 生成散点数据
num_points = 500  # 每个小立方体的散点数量
gray_points = int(num_points * 0.9)  # 灰色散点数量
color_points = num_points - gray_points  # 彩色散点数量

# 定义小立方体的范围
cubes = [
    (0, 0, 0), (0, 0, 0.5), (0, 0.5, 0), (0, 0.5, 0.5),
    (0.5, 0, 0), (0.5, 0, 0.5), (0.5, 0.5, 0), (0.5, 0.5, 0.5)
]

for (x_offset, y_offset, z_offset) in cubes:
    # 生成浅灰色散点
    x_gray = np.random.rand(gray_points) * 0.5 + x_offset
    y_gray = np.random.rand(gray_points) * 0.5 + y_offset
    z_gray = np.random.rand(gray_points) * 0.5 + z_offset

    # 生成彩色散点
    x_color = np.random.rand(color_points) * 0.5 + x_offset
    y_color = np.random.rand(color_points) * 0.5 + y_offset
    z_color = np.random.rand(color_points) * 0.5 + z_offset
    colors = np.random.rand(color_points, 3)  # 随机颜色

    # 绘制浅灰色散点
    ax.scatter(x_gray, y_gray, z_gray, color='lightgray', s=1, alpha=0.5)
    
    # 绘制彩色散点
    ax.scatter(x_color, y_color, z_color, color=colors, s=1, alpha=0.7)

# 绘制立方体的边界
r = [0, 0.5, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s - e)) == r[1]:  # 确保是边界线
        ax.plot3D(*zip(s, e), color="k")

# 设置坐标轴标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 设置标题
ax.set_title('3D Scatter Plot of Subdivided Cube with Colored Points')

# 显示图形
plt.show()
plt.savefig(DSE_plot_path, dpi=300, bbox_inches='tight')
