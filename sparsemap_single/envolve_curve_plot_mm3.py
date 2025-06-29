import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.arange(1, 31)  # 横轴：进化代数，范围是1到30
y_1 = [
    8.958310e+10,
    8.545415e+10,
    6.570399e+10,
    5.647282e+10,
    5.623002e+10,
    5.121097e+10,
    4.998651e+10,
    4.676474e+10,
    4.612247e+10,
    4.548019e+10,
    4.419565e+10,
    4.217306e+10,
    4.209807e+10,
    4.066388e+10,
    3.930476e+10,
    3.889234e+10,
    3.418600e+10,
    3.171278e+10,
    2.983954e+10,
    2.802981e+10,
    2.707513e+10,
    2.660402e+10,
    2.651796e+10,
    2.649501e+10,
    2.647206e+10,
    2.646059e+10,
    2.651796e+10,
    2.649501e+10,
    2.647206e+10,
    2.646059e+10
] # 本文工作


y_2= [
    9.383501e+10,
    8.140737e+10,
    7.972239e+10,
    7.361275e+10,
    7.340888e+10,
    7.320501e+10,
    7.300115e+10,
    7.287420e+10,
    7.287420e+10,
    7.287420e+10,
    7.246646e+10,
    7.246646e+10,
    7.226260e+10,
    7.165100e+10,
    7.063168e+10,
    6.940848e+10,
    6.859302e+10,
    6.618529e+10,
    6.536983e+10,
    6.494277e+10,
    6.451572e+10,
    6.268093e+10,
    6.084614e+10,
    5.880749e+10,
    5.595337e+10,
    5.383369e+10,
    5.383369e+10,
    5.383369e+10,
    5.383369e+10,
    5.383369e+10
]                                                                                    # 常规初始化 + 高低敏感度进化
y_3 = [
    9.383501e+10,
    8.340737e+10,
    7.652239e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
    7.561275e+10,
]                                                                         #常规ES
# 创建折线图
plt.figure(figsize=(6, 6))
plt.plot(x, y_1, marker='.', linestyle='-', color='g', label='This work')
plt.plot(x, y_2, marker='.', linestyle='-', color='b', label='Naive ES + HLenvolve')
plt.plot(x, y_3, marker='.', linestyle='-', color='c', label='Naive ES')
plt.title('Population Average EDP MM3')
plt.xlabel('Evolution Generation')
plt.ylabel('Population Average EDP (cycles * pJ)')

# 设置横轴刻度标签
plt.xticks([0, 10, 20, 30], fontsize=10, rotation=0)

# 添加图例
plt.legend()

# 保存图像
plt.savefig('mm3_population_average_EDP.png', dpi=300, bbox_inches='tight')  # 保存为PNG文件

# 如果不需要显示图像，可以注释掉这行
# plt.show()

