import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# 输入灰度图路径
grayscale_path = '/mnt/F/zt_global_elemapping02/map/zt_20250506_201531_height_190_distancex_2264_distancey_2264.png'  # 示例路径，需替换
output_dir = '/mnt/F/zt_global_elemapping02/visualization'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取灰度图（单通道）
grayscale = cv2.imread(grayscale_path, cv2.IMREAD_GRAYSCALE)
if grayscale is None:
    print(f"Failed to load grayscale image: {grayscale_path}")
    exit(1)

# 从文件名中提取 name（最大高程值）
filename = os.path.basename(grayscale_path)
try:
    height_str = filename.split('_height_')[1].split('_')[0]
    name = float(height_str) / 100  # name 是 max_height
    print(f"Extracted max height (name): {name}")
except (IndexError, ValueError):
    print("Could not extract 'height' from filename. Please ensure the filename format is correct.")
    exit(1)

# 从文件名中提取 distance_x 和 distance_y
try:
    distance_x = float(filename.split('_distancex_')[1].split('_')[0])
    distance_y = float(filename.split('_distancey_')[1].split('.')[0])
    print(f"Extracted distance_x: {distance_x}, distance_y: {distance_y}")
except (IndexError, ValueError):
    print("Could not extract 'distance_x' or 'distance_y' from filename. Please ensure the filename format is correct.")
    exit(1)

# 计算 rows 和 cols
rows = int(distance_x / 4)
cols = int(distance_y / 4)

# 验证图像尺寸
if grayscale.shape != (rows, cols):
    print(f"Image dimensions {grayscale.shape} do not match expected (rows, cols) = ({rows}, {cols}).")
    exit(1)

# 反归一化：恢复高程值
# pixel_value = (height / max_height) * 255 => height = (pixel_value / 255) * max_height
height_data = (grayscale.astype(np.float32) / 255.0) * name

# 将像素值为 0 的区域（第一类 NaN）设置为 NaN
height_data[grayscale == 0] = np.nan

# 创建 X 和 Y 坐标网格
x = np.arange(0, cols, 1)
y = np.arange(0, rows, 1)
X, Y = np.meshgrid(x, y)

# 创建三维表面图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制表面图，NaN 值会自动忽略
surface = ax.plot_surface(X, Y, height_data, cmap='viridis', edgecolor='none')

# 添加颜色条
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Height')

# 设置轴标签
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.set_zlabel('Height')
ax.set_title('3D Visualization of Elevation Map')

# 交互式展示，允许手动调整角度
plt.show()

# 手动保存（注释掉自动保存，提示用户手动保存）
# 如果需要保存，可以取消以下注释并运行，或在图形窗口点击保存按钮
"""
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"zt_3d_{current_time}_height_{name*100:.0f}_distancex_{distance_x}_distancey_{distance_y}.png"
output_path = os.path.join(output_dir, output_filename)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved 3D visualization: {output_path}")
"""

# 关闭图形（可选，手动关闭窗口后生效）
plt.close(fig)