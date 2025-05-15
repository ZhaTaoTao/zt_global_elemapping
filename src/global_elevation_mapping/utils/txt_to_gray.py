import os
import numpy as np
import cv2
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation
from datetime import datetime

""" 我在txt文件中存储了高程数据，现在要你将此txt文件转化成单通道灰度图，首先txt文件里有很多nan值，这些nan值分为两类，一类是机器人
局部探索范围不够大导致的周围区域的nan值，这类nan值一般在周围且比较集中，另一类是nan值为已探索区域中由于遮挡等因素造成的，这类值在空
间上夹杂在有效值之间。现在我的建议是第一类nan值（由于探索范围不够导致的），创建一个不包括第一类nan值的掩码，就是说暂时不对第一类nan
值进行处理，对于第二类nan值，我的建议是用线性插值进行填充。此外，由于要保存到单通道灰度图，因此不能出现负值，这里我建议所有值减去最小
值，这样就没有负值了，且相对高度不变。这样处理之后，先保存最大值至变量name，然后将最小值设置为0，最大值设置为255，中间的值就等于（原
来的值/（最大值-最小值）x 255），最后处理第一类nan值，将第一类nan值全部设置为255，最后保存灰度图，图片命名格式“zt_now(当下时间)_
height_{name×100}_distancex_{每一行像素个数×4}_distancey_{每一列像素个数×4}”，请写出代码 """


def identify_peripheral_nan_mask(data):
    """识别第一类 NaN（周围集中的 NaN），返回掩码"""
    # 创建 NaN 掩码
    nan_mask = np.isnan(data)
    
    # 使用形态学膨胀，识别周围集中的 NaN
    # 假设第一类 NaN 形成较大的连续区域
    structure = np.ones((5, 5))  # 5x5 核，捕捉较大连续 NaN 区域
    peripheral_nan_mask = binary_dilation(nan_mask, structure=structure, iterations=3)
    
    # 排除孤立的 NaN（第二类 NaN）
    # 如果一个 NaN 周围有较多有效值，认为它是第二类 NaN
    valid_mask = ~nan_mask
    neighbor_count = binary_dilation(valid_mask, structure=np.ones((3, 3))) & nan_mask
    peripheral_nan_mask = peripheral_nan_mask & ~neighbor_count
    
    return peripheral_nan_mask

def interpolate_scattered_nan(data, peripheral_nan_mask):
    """对第二类 NaN（分散的）进行线性插值，保留第一类 NaN"""
    # 创建有效值的坐标和值
    y, x = np.indices(data.shape)
    valid_mask = ~np.isnan(data) & ~peripheral_nan_mask
    points = np.vstack((y[valid_mask], x[valid_mask])).T
    values = data[valid_mask]
    
    # 创建需要插值的坐标（第二类 NaN）
    scattered_nan_mask = np.isnan(data) & ~peripheral_nan_mask
    if not np.any(scattered_nan_mask):
        return data  # 没有第二类 NaN，直接返回
    
    interp_points = np.vstack((y[scattered_nan_mask], x[scattered_nan_mask])).T
    
    # 线性插值
    interp_values = griddata(points, values, interp_points, method='linear', fill_value=0.0)
    
    # 填充插值结果
    data_out = data.copy()
    data_out[scattered_nan_mask] = interp_values
    
    return data_out

# 输入 TXT 文件路径
txt_file_path = '/mnt/F/zt_global_elemapping02/elevation_data/20250504_145020/global_elevation_final_20250504_145306.txt'  # 示例路径，需替换
output_dir = '/mnt/F/zt_global_elemapping02/map'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取 TXT 文件
data = np.loadtxt(txt_file_path, dtype=np.float32)

# 处理第一类 NaN：创建掩码
peripheral_nan_mask = identify_peripheral_nan_mask(data)

# 处理第二类 NaN：线性插值
data_interpolated = interpolate_scattered_nan(data, peripheral_nan_mask)

# 去除负值：减去最小值
valid_data = data_interpolated[~peripheral_nan_mask]  # 排除第一类 NaN
if valid_data.size == 0:
    print("No valid data after interpolation.")
    exit(1)

min_value = np.min(valid_data)
data_shifted = data_interpolated - min_value
data_shifted[peripheral_nan_mask] = np.nan  # 恢复第一类 NaN

# 保存最大值到 name
max_value = np.max(data_shifted[~peripheral_nan_mask])
name = max_value
print(f"Max value (name): {name}")

# 归一化到 [0, 255]
data_normalized = np.zeros_like(data_shifted, dtype=np.float32)
valid_mask = ~peripheral_nan_mask
data_normalized[valid_mask] = (data_shifted[valid_mask] / max_value) * 255
data_normalized[peripheral_nan_mask] = 0  # 第一类 NaN 设为 0

# 转换为 uint8
data_normalized = data_normalized.astype(np.uint8)

# 获取图像尺寸
rows, cols = data_normalized.shape
distance_x = rows * 4
distance_y = cols * 4

# 生成文件名
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"zt_{current_time}_height_{name*100:.0f}_distancex_{distance_x}_distancey_{distance_y}.png"
output_path = os.path.join(output_dir, filename)

# 保存单通道灰度图
cv2.imwrite(output_path, data_normalized)
print(f"Saved grayscale image: {output_path}")