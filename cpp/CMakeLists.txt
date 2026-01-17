cmake_minimum_required(VERSION 3.16)
project(templering_sfm_noopencv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(templering_sfm src/templering_sfm.cpp)
target_include_directories(templering_sfm PRIVATE include)

# Pure standard-library build; no OpenCV and no external deps.

add_executable(gt_keyframe_edge tools/gt_keyframe_edge.cpp)
target_include_directories(gt_keyframe_edge PRIVATE include)

add_executable(ate_two_frames tools/ate_two_frames.cpp)
target_include_directories(ate_two_frames PRIVATE include)

add_executable(ate_keyframes tools/ate_keyframes.cpp)
target_include_directories(ate_keyframes PRIVATE include)
