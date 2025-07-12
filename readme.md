编译：
catkin build global_elevation_mapping 

source ./devel/setup.bash

elevation_mapping_cupy_ws 代码 github 链接：https://github.com/leggedrobotics/elevation_mapping_cupy.git

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
    （global_map_fusion_k_online_rviz.py 和 global_map_fusion_k_online.py 并没有太大区别，前者设置的地图尺寸小，适合rviz可视化。）
    roslaunch global_elevation_mapping global_mapping_online_rviz.launch 
    从 rviz 添加 /global_elevation_map 话题进行可视化
    注意：地图不要设置太大，否则 rviz 会死机。

注意：
    使用在线方法生成全局地图时，代码里使用了卡尔曼滤波，因为之前默认是处理elevation层的原始高程数据。

    卡尔曼滤波的假设：
        系统模型是线性的（高程值随时间变化可以建模为线性过程）。
        噪声是高斯白噪声（过程噪声和测量噪声符合正态分布）。
        输入数据（z_local）是原始测量值，包含独立的高斯噪声。
        
    在切换其他图层（smooth、inpaint、min_filter）时，在 rviz 中添加 /global_elevation_map 话题后，
    height layer 和 color layer 都要选择你换的图层，此外，其他图层的数据已经经过相关处理，数据不再符合
    卡尔曼滤波的假设（独立高斯噪声），因此建议自行将卡尔曼滤波删除，使用其他的融合方法。

最后提醒：代码中很多用的是笔者的绝对路径，因此克隆后记得修改相关路径。
