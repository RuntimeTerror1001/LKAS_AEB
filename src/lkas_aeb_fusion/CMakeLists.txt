cmake_minimum_required(VERSION 3.8)
project(lkas_aeb_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(message_filters REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(lkas_aeb_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_geometry REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(pcl_conversions REQUIRED)

add_library(front_fusion_node SHARED
  src/front_fusion_node.cpp
)

target_include_directories(front_fusion_node PUBLIC
  include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

ament_target_dependencies(front_fusion_node
  rclcpp
  rclcpp_components
  message_filters
  vision_msgs
  sensor_msgs
  lkas_aeb_msgs
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
  cv_bridge
  image_geometry
  pcl_conversions
)

target_link_libraries(front_fusion_node
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

rclcpp_components_register_nodes(front_fusion_node
  PLUGIN "lkas_aeb_fusion::FrontFusionNode"
)

install(TARGETS front_fusion_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib 
  RUNTIME DESTINATION lib 
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
