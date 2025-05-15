编译：
catkin build global_elevation_mapping 

source ./devel/setup.bash


离线建图使用流程：
    1.编译好后，播放rosbag包，运行elevation_mapping_cupy_ws中的建图launch（设置好局部地图分辨率与范围），
    然后启动这个包的 global_mapping_underline.launch（也就是运行 01local_map_data_underline.py） 
    注意选好话题和图层，这是第一步，这里会采集局部地图的热力图、txt高程数据和odom坐标转换数据
    （注意：坐标转换数据中，x 和 y 轴方向是反的，四元数旋转用不上，因为发布的局部地图没有旋转）。

    2.运行 02global_map_fusion_underline.py，注意选择好局部地图 txt 高程数据和odom数据所在的文件夹，
    设置对应的分辨率，全局地图大小根据经验设置，这一步会生成全局地图的热力图和txt高程数据。

    3.运行03txt_to_gray_cut-1.py，注意选择对应的全局地图的txt高程数据文件，这一步会对生成的
    全局地图进行裁剪，裁剪掉周围多余的nan值，并将地图转换为灰度图。这里就要设置好合适的 限制最大高度，
    最终保存用灰度图表示的全局高程图。


在线建图使用流程：
    注意，使用在线建图的时候，elevation_mapping_cupy_ws中局部地图的发布频率不宜过大，否则会处理不过来导致效果不好，
    建议手动调小（在 elevation_mapping_cupy_ws 包的 src/elevation_mapping_cupy/elevation_mapping_cupy/
    config/core/example_setup.yaml 中设置），比如调到 0.5 Hz 左右。
    运行elevation_mapping_cupy_ws中的建图launch后，
    roslaunch global_elevation_mapping global_mapping_online.launch 

在线建图并发布使用流程：
    roslaunch global_elevation_mapping global_mapping_online_pub.launch 
    从 rviz 添加 /global_elevation_map 话题进行可视化
    注意：地图不要设置太大，否则 rviz 会死机。

最后提醒：代码中很多用的是笔者的绝对路径，因此克隆后记得修改相关路径。
