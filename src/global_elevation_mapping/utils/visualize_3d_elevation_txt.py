import matplotlib
matplotlib.use('Qt5Agg')  # 使用交互式后端，支持显示和调整视角
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator

# 定义网格边长（单位：米）
GRID_CELL_SIZE = 0.04  # 每个网格单元的边长

# 读取数据并标记原始0的位置
data = np.loadtxt('/mnt/F/zt_global_elemapping02/underline_map/txt/20250506_162838/global_elevation_0004.txt')
original_zero_mask = (data == 0)

# 处理NaN插值
rows, cols = data.shape

# 获取非NaN点的坐标和值
valid_points = np.argwhere(~np.isnan(data))
x_coords = valid_points[:, 1]  # 列索引作为x坐标
y_coords = valid_points[:, 0]  # 行索引作为y坐标
values = data[~np.isnan(data)]

# 将网格索引转换为实际物理坐标（单位：米）
x_coords_physical = x_coords * GRID_CELL_SIZE
y_coords_physical = y_coords * GRID_CELL_SIZE

# 生成完整网格坐标（物理单位：米）
grid_x, grid_y = np.meshgrid(np.arange(cols) * GRID_CELL_SIZE, np.arange(rows) * GRID_CELL_SIZE)

# 双阶段插值（线性+最近邻）
filled_linear = griddata((x_coords_physical, y_coords_physical), values, (grid_x, grid_y), method='linear')
filled_nearest = griddata((x_coords_physical, y_coords_physical), values, (grid_x, grid_y), method='nearest')
filled_data = np.where(np.isnan(filled_linear), filled_nearest, filled_linear)

# 处理未填充区域（原始0值设为NaN以便在3D图中区分）
filled_data[original_zero_mask] = np.nan

# 创建3D表面图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 生成X, Y, Z数据
X = grid_x
Y = grid_y
Z = filled_data

# 绘制3D表面图
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# 设置轴标签（单位明确）
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Elevation (m)')
ax.set_title('3D Elevation Map')

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')

# 设置视角（可调整以优化初始视觉效果）
ax.view_init(elev=30, azim=45)

# 计算数据范围，动态设置Z轴范围
z_valid = Z[~np.isnan(Z)]
if z_valid.size > 0:
    z_min, z_max = np.min(z_valid), np.max(z_valid)
    z_margin = (z_max - z_min) * 0.1  # 添加10%的边界
    ax.set_zlim(z_min - z_margin, z_max + z_margin)
else:
    z_min, z_max = -1, 1  # 默认范围，防止空数据
    ax.set_zlim(z_min, z_max)

# 设置统一的刻度间隔（每1米一个刻度）
ax.xaxis.set_major_locator(MultipleLocator(1.0))  # X轴每1米一个刻度
ax.yaxis.set_major_locator(MultipleLocator(1.0))  # Y轴每1米一个刻度
ax.zaxis.set_major_locator(MultipleLocator(1.0))  # Z轴每1米一个刻度

# 设置轴的比例相等，确保1米在X、Y、Z轴上视觉长度一致
x_range = np.max(X) - np.min(X)
y_range = np.max(Y) - np.min(Y)
z_range = z_max - z_min if z_valid.size > 0 else 2
ax.set_box_aspect((x_range, y_range, z_range))  # 按实际数据范围设置比例

# 获取当前时间并格式化为字符串
now = datetime.now().strftime("%Y%m%d_%H%M%S")
# 图片保存路径和名称，包含时间
filename = f'output_3d_elevation_{now}.png'

# 保存3D图像
plt.savefig(filename, bbox_inches='tight', dpi=300, pad_inches=0)

# 显示交互式图形窗口
plt.show()

# 关闭图形，释放内存
plt.close(fig)

# 额外保存高程数据为txt（可选，用于验证）
np.savetxt(f'filled_elevation_{now}.txt', filled_data, fmt='%.6f')