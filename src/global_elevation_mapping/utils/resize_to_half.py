import cv2


def resize_and_save_image(input_path, output_path):
    # 读取图片
    image = cv2.imread(input_path)
    if image is None:
        print(f"无法读取图片: {input_path}")
        return
    # 获取原始图片的高度和宽度
    height, width = image.shape[:2]
    # 计算新的高度和宽度，变为原来的一半
    new_height = height // 2
    new_width = width // 2
    # 调整图片大小
    resized_image = cv2.resize(image, (new_width, new_height))
    # 保存调整后的图片
    cv2.imwrite(output_path, resized_image)
    print(f"图片已成功保存到: {output_path}")


if __name__ == "__main__":
    input_image_path = "/mnt/F/zt_global_elemapping02/map/zt_20250506_155300_height_203_distancex_4740_distancey_4740.png"
    output_image_path = "/mnt/F/zt_global_elemapping02/map1/zt_20250506_155300_height_203_distancex_4740_distancey_4740.png"
    resize_and_save_image(input_image_path, output_image_path)

