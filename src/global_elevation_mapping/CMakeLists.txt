cmake_minimum_required(VERSION 3.0.2)
project(global_elevation_mapping)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  nav_msgs
  sensor_msgs
  tf2_ros
  message_generation
)

# add_service_files(
#   DIRECTORY srv
#   FILES SaveMap.srv
# )

# add_message_files(
#   FILES
#   CovarianceMap.msg
# )

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS 
  message_runtime 
  rospy 
  nav_msgs 
  tf2_ros
)

catkin_install_python(PROGRAMS
  scripts/01local_map_data_underline.py
  scripts/02global_map_fusion_underline.py
  scripts/03txt_to_gray_cut-1.py
  scripts/global_map_fusion_k_online.py
  scripts/global_map_fusion_k_online_rviz.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)