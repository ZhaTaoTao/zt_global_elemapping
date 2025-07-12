import os
import re
from PIL import Image
import numpy as np

def crop_and_flip_image(input_path, output_dir, new_size, center_coords):
    """
    裁剪 PNG 图像并左右翻转，保存到指定目录。
    
    参数：
    - input_path: 输入图像路径（例如 'zt_20250516_191332_height_355_distancex_4540_distancey_4540.png'）
    - output_dir: 输出目录
    - new_size: 裁剪后正方形的边长（像素）
    - center_coords: 裁剪区域中心坐标 [center_x, center_y]，范围 0-1
    """
    # 读取图像
    try:
        img = Image.open(input_path)
    except Exception as e:
        print(f"无法打开图像 {input_path}: {e}")
        return

    # 获取原始图像尺寸
    orig_width, orig_height = img.size

    # 从文件名提取信息
    filename = os.path.basename(input_path)
    match = re.match(r'zt_(\d{8}_\d{6})_height_(\d+)_distancex_(\d+)_distancey_(\d+)\.png', filename)
    if not match:
        print(f"文件名格式不正确: {filename}")
        return

    timestamp, height, orig_distancex, orig_distancey = match.groups()
    # 将文件名中的尺寸除以 4，转换为实际像素尺寸
    expected_width = int(orig_distancex) // 4
    expected_height = int(orig_distancey) // 4
    if expected_width != orig_width or expected_height != orig_height:
        print(f"文件名中的尺寸 ({expected_width}x{expected_height}) 与实际尺寸 ({orig_width}x{orig_height}) 不匹配")
        return

    # 验证裁剪参数
    if new_size <= 0 or new_size > min(orig_width, orig_height):
        print(f"裁剪尺寸 {new_size} 无效，必须在 0 到 {min(orig_width, orig_height)} 之间")
        return

    center_x, center_y = center_coords
    if not (0 <= center_x <= 1 and 0 <= center_y <= 1):
        print(f"中心坐标 {center_coords} 无效，必须在 [0, 1] 范围内")
        return

    # 计算裁剪区域中心像素坐标
    center_pixel_x = int(center_x * orig_width)
    center_pixel_y = int(center_y * orig_height)

    # 计算裁剪区域的边界
    half_size = new_size // 2
    left = max(0, center_pixel_x - half_size)
    top = max(0, center_pixel_y - half_size)
    right = min(orig_width, center_pixel_x + half_size)
    bottom = min(orig_height, center_pixel_y + half_size)

    # 调整边界以确保裁剪区域为正方形
    if right - left < new_size:
        if left == 0:
            right = left + new_size
        else:
            left = right - new_size
    if bottom - top < new_size:
        if top == 0:
            bottom = top + new_size
        else:
            top = bottom - new_size

    # 裁剪图像
    cropped_img = img.crop((left, top, right, bottom))

    # 确保裁剪后是正方形
    cropped_img = cropped_img.resize((new_size, new_size), Image.Resampling.LANCZOS)

    # 左右翻转
    flipped_img = cropped_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成输出文件名，保存时使用实际像素值
    output_filename = f"zt_{timestamp}_height_{height}_distancex_{new_size*4}_distancey_{new_size*4}.png"
    output_path = os.path.join(output_dir, output_filename)

    # 保存图像
    flipped_img.save(output_path, "PNG")
    print(f"已保存裁剪并翻转后的图像到: {output_path}")

# 示例调用
if __name__ == "__main__":
    # input_path = "/mnt/F/zt_global_elemapping/map/zt_20250516_190501_height_301_distancex_3924_distancey_3924.png"
    # center_coords = [0.4, 0.6]  # 裁剪区域在原图的坐标（原图坐标系原点在左上角）

    # input_path = "/mnt/F/zt_global_elemapping/map/zt_20250516_190719_height_495_distancex_6648_distancey_6648.png"
    # center_coords = [0.3, 0.5]  # 裁剪区域在原图的坐标（原图坐标系原点在左上角）

    # input_path = "/mnt/F/zt_global_elemapping/map/zt_20250516_191126_height_300_distancex_3012_distancey_3012.png"
    # center_coords = [0.5, 0.5]  # 裁剪区域在原图的坐标（原图坐标系原点在左上角）

    # input_path = "/mnt/F/zt_global_elemapping/map/zt_20250516_191332_height_355_distancex_4540_distancey_4540.png"
    # center_coords = [0.4, 0.6]  # 裁剪区域在原图的坐标（原图坐标系原点在左上角）

    # input_path = "/mnt/F/zt_global_elemapping/map/zt_20250519_205413_height_302_distancex_3896_distancey_3896.png"
    # center_coords = [0.4, 0.6]  # 裁剪区域在原图的坐标（原图坐标系原点在左上角）

    input_path = "/mnt/F/zt_global_elemapping/map/zt_20250712_142359_height_291_distancex_3880_distancey_3880.png"
    center_coords = [0.4, 0.6]  # 裁剪区域在原图的坐标（原图坐标系原点在左上角）

    output_dir = "/mnt/F/zt_global_elemapping/map-cut"
    new_size = 2500  # 裁剪后正方形边长
    new_size_image = new_size // 4

    crop_and_flip_image(input_path, output_dir, new_size_image, center_coords)