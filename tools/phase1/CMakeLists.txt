cmake_minimum_required(VERSION 3.15)
project(phase1_reference LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
add_subdirectory(${VISION_ROOT}/3rdparty/tinympc tinympc)
add_executable(phase1-reference reference.cpp legacy/extrapolate.cpp
 ${VISION_ROOT}/src/modules/fire_control/fire_control.cpp
 ${VISION_ROOT}/src/modules/fire_control/fire_control_config.cpp
 ${VISION_ROOT}/src/modules/fire_control/gimbal_feedback_estimator.cpp
 ${VISION_ROOT}/src/modules/armor_predictor/detail/armor_motion_model.cpp
 ${VISION_ROOT}/src/modules/gimbal_trajectory_planner/gimbal_trajectory_planner.cpp
 ${VISION_ROOT}/src/modules/gimbal_trajectory_planner/gimbal_trajectory_planner_config.cpp
 ${VISION_ROOT}/src/modules/gimbal_trajectory_planner/detail/tinympc_workspace.cpp
 ${VISION_ROOT}/src/core/logger.cpp)
target_include_directories(phase1-reference PRIVATE ${VISION_ROOT}/src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(phase1-reference PRIVATE Eigen3::Eigen yaml-cpp mv-3rdparty-tinympc opencv_core fmt::fmt spdlog::spdlog Threads::Threads)

get_target_property(REFERENCE_SOURCES phase1-reference SOURCES)
list(REMOVE_ITEM REFERENCE_SOURCES reference.cpp)
add_executable(phase1-session-reference session_reference.cpp legacy/legacy_session.cpp ${REFERENCE_SOURCES})
target_include_directories(phase1-session-reference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/legacy ${VISION_ROOT}/src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(phase1-session-reference PRIVATE Eigen3::Eigen yaml-cpp mv-3rdparty-tinympc opencv_core fmt::fmt spdlog::spdlog Threads::Threads)
