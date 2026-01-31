cmake_minimum_required(VERSION 3.8)
project(robot_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(frame_id_converter_node scripts/frame_id_converter.cpp)
ament_target_dependencies(frame_id_converter_node rclcpp sensor_msgs)

add_executable(cmd_vel scripts/cmd_vel.cpp)
ament_target_dependencies(cmd_vel rclcpp geometry_msgs std_msgs)


install(
  DIRECTORY src launch config world scripts
  DESTINATION share/${PROJECT_NAME}
)

install(
    TARGETS 
    frame_id_converter_node
    cmd_vel
    DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
