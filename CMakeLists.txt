cmake_minimum_required(VERSION 3.8)
project(samowl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nlohmann_json 3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(pcl_conversions REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(samowl src/samowl.cpp)
target_compile_features(samowl PRIVATE cxx_std_17)
target_compile_definitions(
  samowl
  PRIVATE
    SAMOWL_SOURCE_SCRIPT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/scripts/samowl_pipeline.py"
    SAMOWL_SOURCE_DAEMON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/scripts/samowl_daemon.py"
    SAMOWL_SOURCE_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/samowl.yaml"
)
ament_target_dependencies(
  samowl
  ament_index_cpp
  cv_bridge
  geometry_msgs
  message_filters
  pcl_conversions
  rclcpp
  sensor_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
)
target_include_directories(samowl PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(samowl ${OpenCV_LIBRARIES} yaml-cpp nlohmann_json::nlohmann_json ${PCL_LIBRARIES})

install(TARGETS samowl DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS
  scripts/samowl_pipeline.py
  scripts/samowl_daemon.py
  scripts/scene_graph_node.py
  scripts/scene_query_node.py
  scripts/scene_graph.py
  scripts/scene_query_cli.py
  DESTINATION share/${PROJECT_NAME}/scripts
)
install(DIRECTORY data DESTINATION share/${PROJECT_NAME})
install(FILES config/samowl.yaml DESTINATION share/${PROJECT_NAME}/config)
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
