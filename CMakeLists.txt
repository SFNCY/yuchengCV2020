cmake_minimum_required(VERSION 3.3)
project(test)

# set(CMAKE_PREFIX_PATH "/home/cvte/Software/opencv-3.4.2/build")

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

find_package(OpenCV 3.4.6)
list(APPEND _LINKER_LIBS ${OpenCV_LIBS})
add_definitions(-DUSE_OPENCV)

include_directories(SYSTEM "./include" ${OpenCV_INCLUDE_DIRS})

add_executable(
     TestApplication
    src/track_KLT.cpp
    src/feature_SIFT.cpp
    src/xopencl_kernels_video.cpp
    main.cpp)

target_link_libraries(TestApplication ${_LINKER_LIBS})