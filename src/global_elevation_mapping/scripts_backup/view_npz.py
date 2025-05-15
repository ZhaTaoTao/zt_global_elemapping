# import rospy
# import numpy as np
# from grid_map_msgs.msg import GridMap, GridMapInfo
# from std_msgs.msg import Float32MultiArray, Header
# from tf.transformations import quaternion_from_euler
# from geometry_msgs.msg import Pose, Quaternion  # 新增导入 Quaternion 类型

# def npz_to_gridmap(npz_path):
#     data = np.load(npz_path)
#     elevation_map = data['elevation']
#     resolution = data['resolution']
#     origin_x = data['origin_x']
#     origin_y = data['origin_y']

#     # 创建 GridMap 消息
#     grid_map = GridMap()
#     grid_map.info.header = Header(frame_id='map')
#     grid_map.info.resolution = resolution

#     # 设置地图尺寸（物理尺寸 = 像素数 × 分辨率）
#     grid_map.info.length_x = elevation_map.shape[1] * resolution  # width × resolution
#     grid_map.info.length_y = elevation_map.shape[0] * resolution  # height × resolution

#     # 设置地图原点（几何中心，使用 Pose 类型）
#     origin_pose = Pose()  # 修正为 Pose 类型
#     # 计算地图中心坐标（origin_x 和 origin_y 通常是地图左下角，需转换为中心）
#     center_x = origin_x + grid_map.info.length_x / 2
#     center_y = origin_y + grid_map.info.length_y / 2
#     origin_pose.position.x = center_x
#     origin_pose.position.y = center_y

#     # 四元数表示无旋转，转换为 Quaternion 类型
#     quaternion = quaternion_from_euler(0, 0, 0)
#     origin_pose.orientation = Quaternion(*quaternion)

#     grid_map.info.pose = origin_pose  # 将 Pose 赋值给 grid_map.info.pose

#     # 处理高程数据（NaN 转无效值 -1.0）
#     elevation_layer = Float32MultiArray()
#     # elevation_layer.data = np.nan_to_num(elevation_map, nan=-1.0).flatten().tolist()
#     elevation_layer.data = np.nan_to_num(elevation_map, nan=1.0).flatten().tolist()

#     # 组织消息结构
#     grid_map.layers = ['elevation']
#     grid_map.data = [elevation_layer]
#     return grid_map

# if __name__ == '__main__':
#     rospy.init_node('npz_to_rviz')
#     pub = rospy.Publisher('/grid_map', GridMap, queue_size=10)
#     rate = rospy.Rate(1)  # 1Hz 发布频率

#     # 加载并发布地图（请替换为实际文件路径）
#     npz_path = '/mnt/F/zt_global_elemapping02/map/final_global_map_20250428_094447.npz'
#     grid_map = npz_to_gridmap(npz_path)

#     while not rospy.is_shutdown():
#         grid_map.info.header.stamp = rospy.Time.now()
#         pub.publish(grid_map)
#         rate.sleep()


import rospy
import numpy as np
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, Header, MultiArrayDimension
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Pose, Quaternion

def npz_to_gridmap(npz_path):
    data = np.load(npz_path)
    print(data)
    elevation_map = data['elevation']
    print(elevation_map)
    resolution = data['resolution']
    print(resolution)
    origin_x = data['origin_x']
    print(origin_x)
    origin_y = data['origin_y']
    print(origin_y)

    grid_map = GridMap()
    grid_map.info.header = Header(frame_id='map')
    grid_map.info.resolution = resolution

    height, width = elevation_map.shape
    grid_map.info.length_x = width * resolution
    grid_map.info.length_y = height * resolution

    # 原点计算（左下角转中心）
    center_x = origin_x + (grid_map.info.length_x / 2.0)
    center_y = origin_y + (grid_map.info.length_y / 2.0)
    # center_x = origin_x
    # center_y = origin_y 

    origin_pose = Pose()
    origin_pose.position.x = center_x
    origin_pose.position.y = center_y
    origin_pose.position.z = 0.0
    origin_pose.orientation = Quaternion(*quaternion_from_euler(0, 0, 0))
    grid_map.info.pose = origin_pose

    elevation_layer = Float32MultiArray()

    # 关键修正：正确的行优先布局设置
    elevation_layer.layout.dim = [
        MultiArrayDimension(label="row", size=height, stride=width), 
        MultiArrayDimension(label="column", size=width, stride=1)     
    ]
    elevation_layer.layout.data_offset = 0

    # 处理数据顺序（必须使用C风格的行优先展平）
    # elevation_map = np.nan_to_num(elevation_map, nan=-9999.0)
    elevation_map = np.nan_to_num(elevation_map, nan=1.0)
    elevation_layer.data = elevation_map.astype(np.float32).flatten(order='C').tolist()

    grid_map.layers = ['elevation']
    grid_map.data = [elevation_layer]
    return grid_map

if __name__ == '__main__':
    rospy.init_node('npz_to_rviz')
    pub = rospy.Publisher('/grid_map', GridMap, queue_size=10)
    npz_path = '/mnt/F/zt_global_elemapping02/map/final_global_map_20250429_103626.npz'
    
    try:
        grid_map = npz_to_gridmap(npz_path)
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            grid_map.info.header.stamp = rospy.Time.now()
            pub.publish(grid_map)
            rate.sleep()
    except Exception as e:
        rospy.logerr(f"Error: {str(e)}")