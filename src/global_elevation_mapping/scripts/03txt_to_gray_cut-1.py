import os
import numpy as np
import cv2
from scipy.interpolate import griddata
from scipy.ndimage import label
from datetime import datetime

max_height = 5       # 限制最大高度（防止最大高度过高，底层像素值普遍很小，导致灰度图表示的高度变化明显阶梯化）

def identify_peripheral_nan_mask(data):
    """识别第一类 NaN（连接到数据边界的 NaN），返回掩码"""
    nan_mask = np.isnan(data)
    # 标记 NaN 的连通区域
    labeled_array, num_features = label(nan_mask, structure=np.ones((3, 3)))
    peripheral_nan_mask = np.zeros_like(nan_mask, dtype=bool)
    
    # 检查每个连通区域是否接触边界
    rows, cols = data.shape
    for i in range(1, num_features + 1):
        region = (labeled_array == i)
        # 检查是否接触边界（第一行、最后一行、第一列、最后一列）
        touches_boundary = (
            np.any(region[0, :]) or  # 第一行
            np.any(region[-1, :]) or  # 最后一行
            np.any(region[:, 0]) or  # 第一列
            np.any(region[:, -1])     # 最后一列
        )
        if touches_boundary:
            peripheral_nan_mask |= region
    
    return peripheral_nan_mask

def interpolate_scattered_nan(data, peripheral_nan_mask):
    """对第二类 NaN（非外围）进行线性插值，保留第一类 NaN"""
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
    valid_mask = ~peripheral_nan_mask
    rows, cols = np.where(valid_mask)
    if len(rows) == 0 or len(cols) == 0:
        print("No valid data found for cropping.")
        return data, data.shape[0], data.shape[1]
    
    margin = 10  # 添加边距
    top, bottom = max(0, np.min(rows) - margin), min(data.shape[0], np.max(rows) + 1 + margin)
    left, right = max(0, np.min(cols) - margin), min(data.shape[1], np.max(cols) + 1 + margin)
    
    height = bottom - top
    width = right - left
    side = max(height, width)
    
    cropped_data = np.zeros((side, side), dtype=data.dtype)
    start_y = (side - height) // 2
    start_x = (side - width) // 2
    end_y = start_y + height
    end_x = start_x + width
    cropped_data[start_y:end_y, start_x:end_x] = data[top:bottom, left:right]
    
    return cropped_data, side, side

# 输入 TXT 文件路径
# txt_file_path = '/mnt/F/zt_global_elemapping/underline_map/txt/20250510_143955/global_elevation_0009.txt'
# txt_file_path = '/mnt/F/zt_global_elemapping/underline_map/txt/20250510_141247/global_elevation_0026.txt'
txt_file_path = '/mnt/F/zt_global_elemapping/underline_map/txt/20250510_135521/global_elevation_0043.txt'
# txt_file_path = 
output_dir = '/mnt/F/zt_global_elemapping/map'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取 TXT 文件
data = np.loadtxt(txt_file_path, dtype=np.float32)

# 处理第一类 NaN：创建掩码
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
peripheral_nan_mask = identify_peripheral_nan_mask(data)
# 调试：保存掩码
# cv2.imwrite(os.path.join(output_dir, f"peripheral_nan_mask_{current_time}.png"), 
#             (peripheral_nan_mask * 255).astype(np.uint8))
# cv2.imwrite(os.path.join(output_dir, f"scattered_nan_mask_{current_time}.png"), 
#             ((np.isnan(data) & ~peripheral_nan_mask) * 255).astype(np.uint8))

# 处理第二类 NaN：线性插值
data_interpolated = interpolate_scattered_nan(data, peripheral_nan_mask)

# 去负值：减去最小值
valid_data = data_interpolated[~peripheral_nan_mask]
if valid_data.size == 0:
    print("No valid data after interpolation, returning 1x1 zero image.")
    data_normalized = np.zeros((1, 1), dtype=np.uint8)
    rows, cols = 1, 1
    distance_x = distance_y = 4
else:
    min_value = np.min(valid_data)
    data_shifted = data_interpolated - min_value
    data_shifted[peripheral_nan_mask] = np.nan

    # 保存最大值到 name
    max_value = np.max(data_shifted[~peripheral_nan_mask])
    if max_value > max_height:
        max_value = max_height
    name = max_value
    print(f"Max value (name): {name}")

    # 归一化到 [0, 255]
    data_normalized = np.zeros_like(data_shifted, dtype=np.float32)
    valid_mask = ~peripheral_nan_mask
    data_normalized[valid_mask] = (data_shifted[valid_mask] / max_value) * 255
    data_normalized[peripheral_nan_mask] = 0

    # 裁剪第一类 NaN 区域并确保正方形
    data_normalized, rows, cols = crop_to_valid_square(data_normalized, peripheral_nan_mask)
    data_normalized = data_normalized.astype(np.uint8)

    # 计算 distance_x 和 distance_y
    resolution = 0.04  # 米
    distance_x = rows * (resolution * 100)  # 厘米
    distance_y = cols * (resolution * 100)

# 生成文件名
filename = f"zt_{current_time}_height_{name*100:.0f}_distancex_{distance_x:.0f}_distancey_{distance_y:.0f}.png"
output_path = os.path.join(output_dir, filename)

# 保存单通道灰度图
cv2.imwrite(output_path, data_normalized)
print(f"Saved grayscale image: {output_path}")