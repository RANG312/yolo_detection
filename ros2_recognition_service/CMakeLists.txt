cmake_minimum_required(VERSION 3.8)
project(ros2_recognition_service)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SubmitRecognitionTask.srv"
  "srv/GetRecognitionTask.srv"
)

install(PROGRAMS
  ros2_recognition_service/node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME recognition_node
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
