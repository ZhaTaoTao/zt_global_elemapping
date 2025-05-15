import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob
from datetime import datetime

# 参数
RESOLUTION = 0.04  # 分辨率（米）
SIZE_X = 120.0     # 全局地图 X 维度（米）
SIZE_Y = 120.0     # 全局地图 Y 维度（米）
NUM_CELLS_X = int(SIZE_X / RESOLUTION)  # X 方向格子数
NUM_CELLS_Y = int(SIZE_Y / RESOLUTION)  # Y 方向格子数
FUSION_MODE = "mean"  # 融合模式：mean, overwrite, weighted, max, min

def save_elevation_map_as_image(elevation_data, frame_id, save_dir, prefix="global", mask=None):
    """将高程数据保存为热力图，NaN值和掩码区域透明，使用帧号命名"""
    data = elevation_data.copy()
    if mask is not None:
        data[mask] = np.nan  # 未观测区域设为 NaN

    min_val = np.nanmin(data)
    max_val = np.nanmax(data)

    if np.isnan(min_val) or np.isnan(max_val) or not np.isfinite(min_val) or not np.isfinite(max_val):
        print(f"{prefix} 热力图数据全为无效值，设置默认范围 [0, 0]")
        min_val, max_val = 0.0, 0.0
    if max_val <= min_val:
        print(f"{prefix} 热力图数据范围无效，设置归一化为0")
        normalized = np.zeros_like(data)
    else:
        normalized = (data - min_val) / (max_val - min_val)

    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='none')  # NaN 和掩码区域透明
    colored_image = cmap(normalized)
    colored_image = np.flipud(colored_image)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, f"{prefix}_frame_{frame_id:04d}.png")
    plt.imsave(filename, colored_image, format='png')
    print(f"已保存热力图: {filename}")

def save_elevation_to_txt(elevation_data, frame_id, save_dir, prefix="global_elevation", mask=None):
    """将高程数据保存为带帧号的.txt文件，掩码区域标记为NaN"""
    data = elevation_data.copy()
    if mask is not None:
        data[mask] = np.nan  # 未观测区域保存为 NaN
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, f"{prefix}_{frame_id:04d}.txt")
    np.savetxt(filename, data, fmt='%.6f')
    print(f"已保存高程数据到: {filename}")
    return filename

def load_odom_from_txt(filename):
    """读取里程计 txt 文件，返回 x, y, z（单位：米），忽略四元数"""
    try:
        with open(filename, 'r') as f:
            line = f.readline().strip()
            x, y, z, qx, qy, qz, qw = map(float, line.split())
            """ 这里取反主要是因为可视化时发现发布的odom坐标与实际的odom坐标系这两个方向反了 """
            """ 此外注意发布的局部地图没有进行旋转，因此这里整合全局地图时也不要进行旋转坐标变换（四元数都没用上） """
            x = -x
            y = -y
        position = np.array([x, y, z])  # 单位为米
        print(f"里程计 {filename}: position={position} m")
        return position
    except Exception as e:
        print(f"读取里程计文件 {filename} 失败: {e}")
        return None

def main(input_dir):
    """离线融合局部地图，仅处理坐标平移，生成全局地图"""
    # 初始化全局地图
    global_elevation = np.full((NUM_CELLS_Y, NUM_CELLS_X), np.nan, dtype=np.float32)
    count_map = np.zeros((NUM_CELLS_Y, NUM_CELLS_X), dtype=np.int32)  # 记录每个格子的有效值计数

    # 创建时间戳子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_save_dir = os.path.join("/mnt/F/zt_global_elemapping/underline_map/png", timestamp)
    txt_save_dir = os.path.join("/mnt/F/zt_global_elemapping/underline_map/txt", timestamp)

    # 查找所有局部地图和里程计文件
    elevation_files = list(set(
        glob.glob(os.path.join(input_dir, "local_elevation_*.txt")) +
        glob.glob(os.path.join(input_dir, "local_elevation_final_*.txt"))
    ))
    odom_files = list(set(
        glob.glob(os.path.join(input_dir, "local_odom_*.txt")) +
        glob.glob(os.path.join(input_dir, "local_odom_final_*.txt"))
    ))

    # 按时间戳排序
    elevation_files.sort()
    odom_files.sort()

    print(f"找到 {len(elevation_files)} 个局部地图文件，{len(odom_files)} 个里程计文件")

    # 检查文件数量是否匹配
    if len(elevation_files) != len(odom_files):
        print("警告：局部地图文件和里程计文件数量不匹配，可能导致配对错误")

    # 获取初始位置（用于平移）
    initial_position = None
    if odom_files:
        initial_position = load_odom_from_txt(odom_files[0])
        if initial_position is not None:
            print(f"初始位置: {initial_position}")

    # 配对文件
    frame_id = 1
    for elev_file in elevation_files:
        # 提取时间戳
        elev_timestamp = elev_file.split('_')[-1].replace('.txt', '')
        # 查找对应时间戳的里程计文件
        odom_file = None
        for ofile in odom_files:
            if elev_timestamp in ofile:
                odom_file = ofile
                break
        
        if odom_file is None:
            print(f"警告：未找到与 {elev_file} 对应的里程计文件，跳过")
            continue

        # 加载局部地图
        try:
            local_elevation = np.loadtxt(elev_file)
            valid_count = np.sum(np.isfinite(local_elevation))
            print(f"局部地图 {elev_file}: {local_elevation.shape}, 有效值数量: {valid_count}")
        except Exception as e:
            print(f"读取局部地图文件 {elev_file} 失败: {e}")
            continue

        # 加载里程计
        position = load_odom_from_txt(odom_file)
        if position is None:
            print(f"跳过文件 {elev_file}，由于里程计数据无效")
            continue
        # 平移坐标
        if initial_position is not None:
            position = position - initial_position
            print(f"里程计 {odom_file}: 原始 position={position + initial_position}, 平移后 position={position}")
        else:
            print(f"里程计 {odom_file}: position={position}")

        # 获取局部地图尺寸
        height_local, width_local = local_elevation.shape

        # 计算局部地图网格点坐标（10米 x 10米）
        i_local, j_local = np.indices((height_local, width_local))
        dx_local = (j_local - (width_local - 1) / 2.0) * RESOLUTION  # [-5.0, 4.96]
        dy_local = (i_local - (height_local - 1) / 2.0) * RESOLUTION  # [-5.0, 4.96]

        # 调试：检查局部网格范围
        print(f"dx_local 范围: [{np.min(dx_local):.2f}, {np.max(dx_local):.2f}]")
        print(f"dy_local 范围: [{np.min(dy_local):.2f}, {np.max(dy_local):.2f}]")

        # 创建齐次坐标（仅平移）
        dz_local = np.zeros_like(dx_local)
        local_points = np.stack([dx_local, dy_local, dz_local, np.ones_like(dx_local)], axis=0).reshape(4, -1)

        # 构建平移变换矩阵（无旋转）
        transform = np.eye(4)
        transform[:3, 3] = position

        # 变换到全局坐标系
        global_points = transform @ local_points
        x_global = global_points[0].reshape(height_local, width_local)
        y_global = global_points[1].reshape(height_local, width_local)

        # 调试：检查投影范围
        print(f"x_global 范围: [{np.min(x_global):.2f}, {np.max(x_global):.2f}]")
        print(f"y_global 范围: [{np.min(y_global):.2f}, {np.max(y_global):.2f}]")

        # 计算全局地图网格索引
        j_global = np.floor((x_global + SIZE_X / 2) / RESOLUTION).astype(int)
        i_global = np.floor((y_global + SIZE_Y / 2) / RESOLUTION).astype(int)

        # 调试：检查索引范围
        print(f"i_global 范围: [{np.min(i_global)}, {np.max(i_global)}]")
        print(f"j_global 范围: [{np.min(j_global)}, {np.max(j_global)}]")

        # 过滤有效索引
        valid = (0 <= i_global) & (i_global < NUM_CELLS_Y) & (0 <= j_global) & (j_global < NUM_CELLS_X)
        valid_local = np.isfinite(local_elevation)
        valid_count = np.sum(valid & valid_local)
        print(f"有效投影点数量: {valid_count}")

        # 投影和融合
        updated_cells = 0
        for i in range(height_local):
            for j in range(width_local):
                if valid[i, j] and valid_local[i, j]:
                    i_g, j_g = i_global[i, j], j_global[i, j]
                    z_local = local_elevation[i, j]
                    if np.isnan(global_elevation[i_g, j_g]):
                        global_elevation[i_g, j_g] = z_local
                        count_map[i_g, j_g] = 1
                    else:
                        if FUSION_MODE == "mean":
                            count = count_map[i_g, j_g]
                            global_elevation[i_g, j_g] = (global_elevation[i_g, j_g] * count + z_local) / (count + 1)
                        elif FUSION_MODE == "overwrite":
                            global_elevation[i_g, j_g] = z_local
                        elif FUSION_MODE == "weighted":
                            weight_new = 0.7
                            weight_old = 0.3
                            global_elevation[i_g, j_g] = (global_elevation[i_g, j_g] * weight_old + z_local * weight_new) / (weight_old + weight_new)
                        elif FUSION_MODE == "max":
                            global_elevation[i_g, j_g] = max(global_elevation[i_g, j_g], z_local)
                        elif FUSION_MODE == "min":
                            global_elevation[i_g, j_g] = min(global_elevation[i_g, j_g], z_local)
                        count_map[i_g, j_g] += 1
                    updated_cells += 1

        print(f"已处理文件: {elev_file} 和 {odom_file}, 更新格子数: {updated_cells}")

        # 保存当前全局地图(间隔 5 帧保存一次，最后一次必须保存)
        if frame_id % 5 == 0 or frame_id == len(elevation_files):
            global_valid_count = np.sum(np.isfinite(global_elevation))
            print(f"当前全局地图有效值数量: {global_valid_count}")
            save_elevation_to_txt(global_elevation, frame_id, txt_save_dir)
            save_elevation_map_as_image(global_elevation, frame_id, png_save_dir, prefix="global")
        frame_id += 1

if __name__ == "__main__":
    # 输入目录
    input_dir = "/mnt/F/zt_global_elemapping/underline_elevation_data/20250510_143812"
    main(input_dir)