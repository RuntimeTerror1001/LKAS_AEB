cmake_minimum_required(VERSION 3.10)
project(lkas_aeb_perception)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters segmentation)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)

list(REMOVE_ITEM PCL_INCLUDE_DIRS "/usr/include/pcl-1.12")
list(REMOVE_ITEM PCL_LIBRARY_DIRS "/usr/lib/x86_64-linux-gnu")

add_library(lidar_preproc_node SHARED
  src/lidar_preproc_node.cpp
)

target_include_directories(lidar_preproc_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)

# Link directories for PCL
target_link_directories(lidar_preproc_node PUBLIC
  ${PCL_LIBRARY_DIRS}
)

# Add PCL definitions
target_compile_definitions(lidar_preproc_node PUBLIC
  ${PCL_DEFINITIONS}
)

ament_target_dependencies(lidar_preproc_node
  rclcpp
  rclcpp_components
  sensor_msgs
  vision_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  pcl_conversions
)

# Link PCL libraries - only link the specific components we need
target_link_libraries(lidar_preproc_node
  ${PCL_LIBRARIES}
)

rclcpp_components_register_nodes(lidar_preproc_node
  "lkas_aeb_perception::LidarPreprocNode"
)

# Install
install(TARGETS lidar_preproc_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()