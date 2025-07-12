import os
import re
from PIL import Image
import numpy as np

def pad_image_to_size(input_path, output_dir, target_size=2500):
    """
    将输入图像填充为 target_size x target_size 像素，填充值为 0。
    
    参数：
    - input_path: 输入图像路径
    - output_dir: 输出目录
    - target_size: 目标尺寸（默认 2500x2500 像素）
    """
    # 读取图像
    try:
        img = Image.open(input_path)
    except Exception as e:
        print(f"无法打开图像 {input_path}: {e}")
        return

    # 转换为 numpy 数组
    img_array = np.array(img)

    # 获取当前尺寸
    current_height, current_width = img_array.shape[:2]

    # 如果图像已大于或等于目标尺寸，直接保存
    if current_height >= target_size and current_width >= target_size:
        print(f"图像 {input_path} 尺寸 ({current_height}x{current_width}) 已大于等于目标尺寸 {target_size}x{target_size}")
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        img.save(output_path, "PNG")
        print(f"已保存图像到: {output_path}")
        return

    # 计算需要填充的像素数
    pad_top = (target_size - current_height) // 2
    pad_bottom = target_size - current_height - pad_top
    pad_left = (target_size - current_width) // 2
    pad_right = target_size - current_width - pad_left

    # 根据图像维度设置 pad_width
    if img_array.ndim == 2:  # 灰度图像
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    else:  # RGB 图像
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

    # 填充 0 值像素
    padded_array = np.pad(img_array, pad_width, mode='constant', constant_values=0)

    # 转换为 PIL 图像
    padded_img = Image.fromarray(padded_array)
    # 左右翻转
    padded_img = padded_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保持原始文件名
    filename = os.path.basename(input_path)
    parts = filename.split("_")
    parts[-1] = "2500.png"
    parts[-3] = "2500"
    new_filename = "_".join(parts)
    output_path = os.path.join(output_dir, new_filename)

    # 保存图像
    padded_img.save(output_path, "PNG")
    print(f"已填充并保存图像到: {output_path}")

# 示例调用
if __name__ == "__main__":
    input_path = "/mnt/F/zt_global_elemapping/map/zt_20250519_193240_height_291_distancex_2444_distancey_2444.png"
    output_dir = "/mnt/F/zt_global_elemapping/map-padded"
    target_size = 625

    pad_image_to_size(input_path, output_dir, target_size)