import os
import numpy as np
import cv2
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation
from datetime import datetime

""" 对于之前的代码，有很多第一类nan值在周围，这些nan其实并没有用，而且在后续处理还会消耗计算量。所以在最后保存之前，
对第一类nan值所在的区域进行适当的裁剪，要求保留所有非nan值的部分，并且最后保留的数据形状也是正方形。rows, cols 取处理之后的shape """

max_height = 5

def identify_peripheral_nan_mask(data):
    """识别第一类 NaN（周围集中的 NaN），返回掩码"""
    nan_mask = np.isnan(data)
    structure = np.ones((5, 5))
    peripheral_nan_mask = binary_dilation(nan_mask, structure=structure, iterations=3)
    valid_mask = ~nan_mask
    neighbor_count = binary_dilation(valid_mask, structure=np.ones((3, 3))) & nan_mask
    peripheral_nan_mask = peripheral_nan_mask & ~neighbor_count
    return peripheral_nan_mask

def interpolate_scattered_nan(data, peripheral_nan_mask):
    """对第二类 NaN（分散的）进行线性插值，保留第一类 NaN"""
    y, x = np.indices(data.shape)
    valid_mask = ~np.isnan(data) & ~peripheral_nan_mask
    points = np.vstack((y[valid_mask], x[valid_mask])).T
    values = data[valid_mask]
    scattered_nan_mask = np.isnan(data) & ~peripheral_nan_mask
    if not np.any(scattered_nan_mask):
        return data
    interp_points = np.vstack((y[scattered_nan_mask], x[scattered_nan_mask])).T
    interp_values = griddata(points, values, interp_points, method='linear', fill_value=0.0)
    data_out = data.copy()
    data_out[scattered_nan_mask] = interp_values
    return data_out

def crop_to_valid_square(data, peripheral_nan_mask):
    """裁剪第一类 NaN 区域，保留非 NaN 部分，并确保正方形形状"""
    # 找到非 NaN 区域的边界框
    valid_mask = ~peripheral_nan_mask
    rows, cols = np.where(valid_mask)
    if len(rows) == 0 or len(cols) == 0:
        print("No valid data found for cropping.")
        return data, data.shape[0], data.shape[1]
    
    top, bottom = np.min(rows), np.max(rows) + 1
    left, right = np.min(cols), np.max(cols) + 1
    
    # 计算边界框尺寸
    height = bottom - top
    width = right - left
    side = max(height, width)  # 正方形边长取最大值
    
    # 裁剪数据
    cropped_data = np.zeros((side, side), dtype=data.dtype)
    start_y = (side - height) // 2
    start_x = (side - width) // 2
    end_y = start_y + height
    end_x = start_x + width
    cropped_data[start_y:end_y, start_x:end_x] = data[top:bottom, left:right]
    
    return cropped_data, side, side

# 输入 TXT 文件路径
txt_file_path = '/mnt/F/zt_global_elemapping/underline_map/txt/20250510_141247/global_elevation_0026.txt'
output_dir = '/mnt/F/zt_global_elemapping/map'

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
valid_data = data_interpolated[~peripheral_nan_mask]
if valid_data.size == 0:
    print("No valid data after interpolation.")
    exit(1)

min_value = np.min(valid_data)
data_shifted = data_interpolated - min_value
data_shifted[peripheral_nan_mask] = np.nan

# 保存最大值到 name
max_value = np.max(data_shifted[~peripheral_nan_mask])
# print(max_value, "----------------------")
if max_value > max_height:
    max_value = max_height
name = max_value
print(f"Max value (name): {name}")

# 归一化到 [0, 255]
data_normalized = np.zeros_like(data_shifted, dtype=np.float32)
valid_mask = ~peripheral_nan_mask
data_normalized[valid_mask] = (data_shifted[valid_mask] / max_value) * 255
data_normalized[peripheral_nan_mask] = 0  # 第一类 NaN 设为 0

# 裁剪第一类 NaN 区域并确保正方形
data_normalized, rows, cols = crop_to_valid_square(data_normalized, peripheral_nan_mask)

# 转换为 uint8
data_normalized = data_normalized.astype(np.uint8)

# 计算 distance_x 和 distance_y
distance_x = rows * 4
distance_y = cols * 4

# 生成文件名
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"zt_{current_time}_height_{name*100:.0f}_distancex_{distance_x}_distancey_{distance_y}.png"
output_path = os.path.join(output_dir, filename)

# 保存单通道灰度图
cv2.imwrite(output_path, data_normalized)
print(f"Saved grayscale image: {output_path}")