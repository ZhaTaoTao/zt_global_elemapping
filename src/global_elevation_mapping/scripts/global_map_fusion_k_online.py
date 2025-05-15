import rospy
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Header
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from datetime import datetime
from collections import deque

# 参数
LOCAL_MAP_VARIANCE = 0.1  # 测量噪声协方差 R（标准差0.316米的方差）
PROCESS_NOISE_VARIANCE = 0.0001  # 过程噪声协方差 Q（标准差0.01米的方差）
INITIAL_GLOBAL_VARIANCE = 1000.0  # 初始协方差 P_init
MAX_FUSION_COUNT = 1  # 最大融合次数，减少动态物体残影
ELEVATION_THRESHOLD = 3.0  # 高程值阈值（米），过滤异常值

# Global variables
global_elevation = None
global_variance = None  # 存储每个网格的协方差
count_map = None
invalid_mask = None
odom_buffer = deque(maxlen=100)  # 存储最近的里程计消息
global_map = None
global_map_pub = None
frame_count = 0
init_pose = None
init_odom_set = False
last_message_time = None

global_map_x = 120.0
global_map_y = 120.0
global_map_resolution = 0.04

# 选择局部话题的图层
layer_name = "elevation"
basic_layer_name = "elevation"
num = 30                # 每间隔 30 帧保存一次
now = datetime.now().strftime("%Y%m%d_%H%M%S")

def init_global_map():
    """初始化全局高程地图：120x120米，分辨率0.04米，中心在(0,0)，高度为NaN"""
    global global_elevation, global_variance, count_map, invalid_mask, global_map
    resolution = global_map_resolution
    size_x = global_map_x
    size_y = global_map_y
    num_cells_x = int(size_x / resolution)
    num_cells_y = int(size_y / resolution)

    global_elevation = np.full((num_cells_y, num_cells_x), np.nan, dtype=np.float32)
    global_variance = np.full((num_cells_y, num_cells_x), INITIAL_GLOBAL_VARIANCE, dtype=np.float32)
    count_map = np.zeros((num_cells_y, num_cells_x), dtype=np.int32)
    invalid_mask = np.ones((num_cells_y, num_cells_x), dtype=bool)

    rospy.loginfo(f"初始化全局高程地图：{num_cells_y}x{num_cells_x}，分辨率 {resolution} 米，高程初始为 NaN")

    global_map = GridMap()
    global_map.info.header = Header()
    global_map.info.header.frame_id = "map"
    global_map.info.resolution = resolution
    global_map.info.length_x = size_x
    global_map.info.length_y = size_y
    global_map.info.pose.position.x = 0.0
    global_map.info.pose.position.y = 0.0
    global_map.info.pose.position.z = 0.0
    global_map.info.pose.orientation.w = 1.0

    elevation_multiarray = Float32MultiArray()
    elevation_multiarray.layout.dim = [
        MultiArrayDimension(label="column_index", size=num_cells_x, stride=1),
        MultiArrayDimension(label="row_index", size=num_cells_y, stride=num_cells_x)
    ]
    elevation_multiarray.data = np.nan_to_num(global_elevation, nan=0.0).flatten().tolist()

    global_map.layers = [layer_name]
    global_map.basic_layers = [basic_layer_name]
    global_map.data = [elevation_multiarray]

def odom_callback(msg):
    """存储里程计数据到缓冲区"""
    global odom_buffer, last_message_time
    odom_buffer.append((msg.header.stamp.to_sec(), msg))
    last_message_time = rospy.get_time()

def get_closest_odom(local_map_time):
    """查找与局部地图时间戳最接近的里程计消息"""
    global odom_buffer
    if not odom_buffer:
        return None
    closest_odom = min(odom_buffer, key=lambda x: abs(x[0] - local_map_time))
    return closest_odom[1]

def save_elevation_to_txt(elevation_data, filename_prefix, save_dir, mask=None):
    """将高程数据保存为带时间戳的.txt文件，掩码区域标记为NaN"""
    data = elevation_data.copy()
    if mask is not None:
        data[mask] = np.nan
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.txt")
    np.savetxt(filename, data, fmt='%.6f')
    rospy.loginfo(f"已保存高程数据到: {filename}")
    return filename

def save_elevation_map_as_image(elevation_data, frame_id, prefix="global", mask=None):
    """将高程数据保存为热力图，NaN值和掩码区域透明"""
    data = elevation_data.copy()
    if mask is not None:
        data[mask] = np.nan
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    if np.isnan(min_val) or np.isnan(max_val) or not np.isfinite(min_val) or not np.isfinite(max_val):
        rospy.logwarn(f"{prefix} 热力图数据全为无效值，设置默认范围 [0, 0]")
        min_val, max_val = 0.0, 0.0
    if max_val <= min_val:
        rospy.logwarn(f"{prefix} 热力图数据范围无效，设置归一化为0")
        normalized = np.zeros_like(data)
    else:
        normalized = (data - min_val) / (max_val - min_val)
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='none')
    colored_image = cmap(normalized)
    colored_image = np.flipud(colored_image)
    save_dir = f"/mnt/F/zt_global_elemapping/elevation_maps/{now}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, f"{prefix}_frame_{frame_id:04d}.png")
    plt.imsave(filename, colored_image, format='png')
    rospy.loginfo(f"已保存热力图: {filename}")

def local_map_callback(local_map):
    """处理局部高程地图，使用卡尔曼滤波融合投影到全局地图，仅平移，发布并保存"""
    global global_elevation, global_variance, count_map, invalid_mask, global_map, global_map_pub, frame_count
    global init_pose, init_odom_set, last_message_time

    last_message_time = rospy.get_time()

    try:
        # 获取与局部地图时间戳最接近的里程计
        local_map_time = local_map.info.header.stamp.to_sec()
        odom = get_closest_odom(local_map_time)
        if odom is None:
            rospy.logwarn("未接收到里程计数据，跳过局部地图处理")
            return

        # 设置初始位姿（恢复负号处理）
        if not init_odom_set:
            init_pose = odom.pose.pose
            init_odom_set = True
            rel_pos = np.array([0.0, 0.0, 0.0])
        else:
            curr_pos = np.array([
                -odom.pose.pose.position.x,
                -odom.pose.pose.position.y,
                odom.pose.pose.position.z
            ])
            init_pos = np.array([
                -init_pose.position.x,
                -init_pose.position.y,
                init_pose.position.z
            ])
            rel_pos = curr_pos - init_pos

        # 验证高程层
        if layer_name not in local_map.layers:
            rospy.logwarn(f"局部地图不包含 '{layer_name}' 层")
            return
        elevation_index = local_map.layers.index(layer_name)
        local_data = local_map.data[elevation_index]

        # 验证尺寸
        height_local = local_data.layout.dim[0].size
        width_local = local_data.layout.dim[1].size
        if height_local <= 0 or width_local <= 0 or height_local * width_local != len(local_data.data):
            rospy.logwarn(f"局部地图数据尺寸无效: {height_local}x{width_local}, 数据长度: {len(local_data.data)}")
            return

        # 验证分辨率
        res_local = local_map.info.resolution
        if abs(res_local - global_map.info.resolution) > 1e-6:
            rospy.logwarn(f"局部地图分辨率 {res_local} 与全局地图 {global_map.info.resolution} 不匹配")
            return

        local_elevation = np.array(local_data.data).reshape((height_local, width_local))
        valid_local = np.isfinite(local_elevation) & (np.abs(local_elevation) <= ELEVATION_THRESHOLD)
        invalid_count = np.sum(~valid_local)
        if invalid_count > 0:
            rospy.logwarn(f"局部高程地图包含 {invalid_count} 个无效值或超阈值 (NaN, inf, 或 |z|>{ELEVATION_THRESHOLD})，将跳过这些区域的更新")
        if invalid_count == height_local * width_local:
            rospy.logwarn("局部高程地图全为无效值，跳过")
            return

        # 计算局部网格
        i_local, j_local = np.indices((height_local, width_local))
        dx_local = (j_local - (width_local - 1) / 2.0) * res_local
        dy_local = (i_local - (height_local - 1) / 2.0) * res_local
        dz_local = np.zeros_like(dx_local)

        # 变换到全局坐标（仅平移）
        local_points = np.stack([dx_local, dy_local, dz_local, np.ones_like(dx_local)], axis=0).reshape(4, -1)
        transform = np.eye(4)
        transform[:3, 3] = rel_pos
        global_points = transform @ local_points
        x_global = global_points[0].reshape(height_local, width_local)
        y_global = global_points[1].reshape(height_local, width_local)

        # 计算全局索引
        num_cells_x = int(global_map.info.length_x / global_map.info.resolution)
        num_cells_y = int(global_map.info.length_y / global_map.info.resolution)
        j_global = np.floor((x_global + global_map.info.length_x / 2) / global_map.info.resolution).astype(int)
        i_global = np.floor((y_global + global_map.info.length_y / 2) / global_map.info.resolution).astype(int)

        # 过滤有效索引
        valid = (0 <= i_global) & (i_global < num_cells_y) & (0 <= j_global) & (j_global < num_cells_x)

        # 卡尔曼滤波融合，限制融合次数
        for i in range(height_local):
            for j in range(width_local):
                if valid[i, j] and valid_local[i, j]:
                    z_local = local_elevation[i, j]
                    i_g, j_g = i_global[i, j], j_global[i, j]
                    if count_map[i_g, j_g] < MAX_FUSION_COUNT:
                        if np.isnan(global_elevation[i_g, j_g]):
                            global_elevation[i_g, j_g] = z_local
                            global_variance[i_g, j_g] = INITIAL_GLOBAL_VARIANCE
                            count_map[i_g, j_g] = 1
                        else:
                            # 卡尔曼滤波更新
                            P = global_variance[i_g, j_g] + PROCESS_NOISE_VARIANCE  # 预测协方差
                            K = P / (P + LOCAL_MAP_VARIANCE)  # 卡尔曼增益
                            global_elevation[i_g, j_g] = global_elevation[i_g, j_g] + K * (z_local - global_elevation[i_g, j_g])
                            global_variance[i_g, j_g] = (1 - K) * P
                            count_map[i_g, j_g] += 1
                        invalid_mask[i_g, j_g] = False

        # 更新全局地图
        global_map.data[0].data = np.nan_to_num(global_elevation, nan=0.0).flatten().tolist()

        # 发布全局地图
        global_map.info.header.stamp = rospy.Time.now()
        global_map_pub.publish(global_map)

        # 保存数据（每100帧）
        global frame_count
        frame_count += 1
        save_dir = f"/mnt/F/zt_global_elemapping/elevation_data/{now}"
        if frame_count % num == 0:
            save_elevation_to_txt(local_elevation, "local_elevation", save_dir, mask=~valid_local)
            save_elevation_to_txt(global_elevation, "global_elevation", save_dir, mask=invalid_mask)
            save_elevation_map_as_image(global_elevation, frame_count, prefix="global", mask=invalid_mask)
            save_elevation_map_as_image(local_elevation, frame_count, prefix="local", mask=~valid_local)

    except Exception as e:
        rospy.logwarn(f"处理局部地图失败: {e}")
        return

def check_rosbag_completion(event):
    """定期检查rosbag播放是否完成"""
    global last_message_time, global_elevation, invalid_mask, frame_count
    current_time = rospy.get_time()
    if last_message_time is not None and (current_time - last_message_time) > 2.0:
        rospy.loginfo("未收到新消息，rosbag播放可能已完成")
        save_dir = f"/mnt/F/zt_global_elemapping/elevation_data/{now}"
        if np.any(~invalid_mask):
            save_elevation_map_as_image(global_elevation, frame_count + 1, prefix="global", mask=invalid_mask)
            save_elevation_to_txt(global_elevation, "global_elevation_final", save_dir, mask=invalid_mask)
            rospy.loginfo("已保存最终全局高程地图")
        else:
            rospy.logwarn("全局高程地图为空，未保存最终地图")
        rospy.signal_shutdown("rosbag播放完成，节点关闭")

if __name__ == '__main__':
    rospy.init_node('global_map_builder')
    init_global_map()
    global_map_pub = rospy.Publisher('/global_elevation_map', GridMap, queue_size=1)
    odom_sub = rospy.Subscriber('/fixposition/odometry_enu', Odometry, odom_callback)
    local_map_sub = rospy.Subscriber('/elevation_mapping/elevation_map_raw', GridMap, local_map_callback)
    rospy.Timer(rospy.Duration(1.0), check_rosbag_completion)
    rospy.spin()