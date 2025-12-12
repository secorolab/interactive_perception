cmake_minimum_required(VERSION 3.5)
project(robot_controller2)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(motion_specification_interfaces REQUIRED)

add_library(robot_controller2 SHARED src/motion_specification_server.cpp)

ament_target_dependencies(${PROJECT_NAME}
  "motion_specification_interfaces"

# Process action files
#rosidl_generate_interfaces(${PROJECT_NAME}
#  "action/MyAction.action"
#)

ament_package()
