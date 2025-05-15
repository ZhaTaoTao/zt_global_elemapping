from PIL import Image


def get_png_info_pil(file_path):
    try:
        image = Image.open(file_path)
        width, height = image.size
        resolution = width * height
        mode = image.mode
        if mode == 'RGB':
            channels = 3
        elif mode == 'RGBA':
            channels = 4
        elif mode == 'L':
            channels = 1
        else:
            channels = None
        shape = (height, width, channels) if channels is not None else (height, width)
        return {
            "resolution": resolution,
            "shape": shape
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not open or find the image: {file_path}")


if __name__ == "__main__":
    file_path = "/mnt/F/zt_global_elemapping02/map/zt_20250504_163919_height_427_distancex_3000_distancey_3000.png"
    info = get_png_info_pil(file_path)
    print(f"Resolution: {info['resolution']}")
    print(f"Shape: {info['shape']}")
