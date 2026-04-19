cmake_minimum_required(VERSION 3.8)
project(samowl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(samowl src/samowl.cpp)
target_compile_features(samowl PRIVATE cxx_std_17)
target_compile_definitions(
  samowl
  PRIVATE
    SAMOWL_SOURCE_SCRIPT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/scripts/samowl_pipeline.py"
    SAMOWL_SOURCE_DAEMON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/scripts/samowl_daemon.py"
)
ament_target_dependencies(
  samowl
  ament_index_cpp
  cv_bridge
  geometry_msgs
  message_filters
  rclcpp
  sensor_msgs
  tf2
  tf2_ros
)
target_link_libraries(samowl ${OpenCV_LIBRARIES})

install(TARGETS samowl DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS
  scripts/samowl_pipeline.py
  scripts/samowl_daemon.py
  DESTINATION share/${PROJECT_NAME}/scripts
)
install(DIRECTORY data DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
