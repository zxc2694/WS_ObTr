project(TrackingProject)

cmake_minimum_required(VERSION 2.8)

SET("OpenCV_DIR" "D:/opencv/build")

find_package(OpenCV REQUIRED)

add_library(TrackingLibrary ObjectTracking.cpp motionDetection.cpp)

add_executable(Main main.cpp)

target_link_libraries(Main TrackingLibrary ${OpenCV_LIBS})

