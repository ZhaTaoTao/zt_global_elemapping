import rospy
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from datetime import datetime

# Global variables
latest_odom = None       # Latest odom data
latest_local_map = None  # Latest local map data
frame_count = 0          # Frame counter for unique filenames
last_message_time = None # Track last message time

# layer_name = "ele_traversability_input"
# basic_layer_name = "ele_traversability_input"

layer_name = "elevation"
basic_layer_name = "elevation"

# 记录每间隔多少帧保存一次数据（可根据数据频率调整，例如高频数据可设为 100）
num = 30

now = datetime.now().strftime("%Y%m%d_%H%M%S")

def odom_callback(msg):
    """存储最新的里程计数据并更新最后消息时间"""
    global latest_odom, last_message_time
    latest_odom = msg
    last_message_time = rospy.get_time()

def save_elevation_to_txt(elevation_data, filename_prefix, save_dir, mask=None):
    """将高程数据保存为带时间戳的.txt文件，掩码区域标记为NaN"""
    data = elevation_data.copy()
    if mask is not None:
        data[mask] = np.nan  # 未观测区域保存为 NaN
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.txt")
    np.savetxt(filename, data, fmt='%.6f')
    rospy.loginfo(f"已保存高程数据到: {filename}")
    return filename

def save_odom_to_txt(odom, filename_prefix, save_dir):
    """将里程计位姿保存为带时间戳的.txt文件"""
    if odom is None:
        rospy.logwarn("无有效的里程计数据，无法保存")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, f"{filename_prefix}_{timestamp}.txt")
    pose = odom.pose.pose
    with open(filename, 'w') as f:
        f.write(f"{pose.position.x:.6f} {pose.position.y:.6f} {pose.position.z:.6f} "
                f"{pose.orientation.x:.6f} {pose.orientation.y:.6f} {pose.orientation.z:.6f} {pose.orientation.w:.6f}\n")
    rospy.loginfo(f"已保存里程计数据到: {filename}")
    return filename

def save_elevation_map_as_image(elevation_data, frame_id, prefix="local", mask=None):
    """将高程数据保存为热力图，NaN值和掩码区域透明"""
    data = elevation_data.copy()
    if mask is not None:
        data[mask] = np.nan  # 未观测区域设为 NaN

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
    cmap.set_bad(color='none')  # NaN 和掩码区域透明
    colored_image = cmap(normalized)
    colored_image = np.flipud(colored_image)               # 这个函数会将图像上下翻转

    save_dir = "/mnt/F/zt_global_elemapping/underline_elevation_maps"
    save_dir = os.path.join(save_dir, now)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, f"{prefix}_frame_{frame_id:04d}.png")
    plt.imsave(filename, colored_image, format='png')
    rospy.loginfo(f"已保存热力图: {filename}")

def local_map_callback(local_map):
    """处理局部高程地图，保存每隔 num 帧的局部地图和最近的里程计数据"""
    global latest_odom, latest_local_map, frame_count, last_message_time

    last_message_time = rospy.get_time()  # 更新最后消息时间
    latest_local_map = local_map         # 更新最近的局部地图
    frame_count += 1                     # 增加帧计数

    # 检查局部地图是否包含高程层
    if layer_name not in local_map.layers:
        rospy.logwarn(f"局部地图不包含 '{layer_name}' 层")
        return
    elevation_index = local_map.layers.index(layer_name)
    local_data = local_map.data[elevation_index]

    # 验证局部地图数据尺寸
    if local_data.layout.dim[0].size <= 0 or local_data.layout.dim[1].size <= 0:
        rospy.logwarn("局部地图数据尺寸无效")
        return

    # 获取局部地图尺寸
    height_local = local_data.layout.dim[0].size
    print('---------------------', height_local)
    width_local = local_data.layout.dim[1].size
    local_elevation = np.array(local_data.data).reshape((height_local, width_local))

    # 检查局部地图中的无效值
    valid_local = np.isfinite(local_elevation)
    if not np.all(valid_local):
        rospy.logwarn(f"局部高程地图包含 {np.sum(~valid_local)} 个无效值 (NaN 或 inf)，将跳过这些区域的更新")

    # 保存数据（每 num 帧）
    save_dir = "/mnt/F/zt_global_elemapping/underline_elevation_data"
    save_dir = os.path.join(save_dir, now)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if frame_count % num == 0:
        # 保存局部高程数据到 txt（无效值保存为 NaN）
        save_elevation_to_txt(local_elevation, "local_elevation", save_dir, mask=~valid_local)
        # 保存局部高程地图为热力图（无效值透明）
        save_elevation_map_as_image(local_elevation, frame_count, prefix="local", mask=~valid_local)
        # 保存最近的里程计数据
        save_odom_to_txt(latest_odom, "local_odom", save_dir)

def check_rosbag_completion(event):
    """定期检查 rosbag 播放是否完成，并在结束时保存最后一次局部地图"""
    global last_message_time, latest_local_map, latest_odom, frame_count
    current_time = rospy.get_time()
    
    # 如果 2 秒内无新消息，假设 rosbag 播放完成
    if last_message_time is not None and (current_time - last_message_time) > 2.0:
        rospy.loginfo("未收到新消息，rosbag 播放已完成，保存最后一次局部地图")
        
        # 保存最后一次局部地图和里程计数据
        save_dir = "/mnt/F/zt_global_elemapping/underline_elevation_data"
        save_dir = os.path.join(save_dir, now)
        
        if latest_local_map is not None:
            # 检查局部地图是否包含高程层
            if layer_name in latest_local_map.layers:
                elevation_index = latest_local_map.layers.index(layer_name)
                local_data = latest_local_map.data[elevation_index]
                
                # 验证局部地图数据尺寸
                if local_data.layout.dim[0].size > 0 and local_data.layout.dim[1].size > 0:
                    height_local = local_data.layout.dim[0].size
                    width_local = local_data.layout.dim[1].size
                    local_elevation = np.array(local_data.data).reshape((height_local, width_local))
                    
                    # 检查无效值
                    valid_local = np.isfinite(local_elevation)
                    if not np.all(valid_local):
                        rospy.logwarn(f"最后一次局部高程地图包含 {np.sum(~valid_local)} 个无效值 (NaN 或 inf)")
                    
                    # 保存局部高程数据到 txt
                    save_elevation_to_txt(local_elevation, "local_elevation_final", save_dir, mask=~valid_local)
                    # 保存局部高程地图为热力图
                    save_elevation_map_as_image(local_elevation, frame_count + 1, prefix="local", mask=~valid_local)
                    # 保存最近的里程计数据
                    save_odom_to_txt(latest_odom, "local_odom_final", save_dir)
                else:
                    rospy.logwarn("最后一次局部地图数据尺寸无效")
            else:
                rospy.logwarn(f"最后一次局部地图不包含 '{layer_name}' 层")
        else:
            rospy.logwarn("无有效的最后一次局部地图数据")
        
        # 关闭节点
        rospy.signal_shutdown("rosbag 播放完成，节点关闭")

if __name__ == '__main__':
    # 初始化 ROS 节点
    rospy.init_node('local_map_saver')

    # 设置订阅者
    odom_sub = rospy.Subscriber('/fixposition/odometry_enu', Odometry, odom_callback)
    local_map_sub = rospy.Subscriber('/elevation_mapping/elevation_map_raw', GridMap, local_map_callback)

    # 设置定时器，每秒检查 rosbag 完成情况
    rospy.Timer(rospy.Duration(1.0), check_rosbag_completion)

    # 保持节点运行
    rospy.spin()