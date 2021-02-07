cmake_minimum_required(VERSION 3.3)
project(test)

# set(CMAKE_PREFIX_PATH "/home/cvte/Software/opencv-3.4.2/build")

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

# SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

find_package(OpenCV)
list(APPEND _LINKER_LIBS ${OpenCV_LIBS})
add_definitions(-DUSE_OPENCV)

include_directories(SYSTEM "./include" ${OpenCV_INCLUDE_DIRS})

add_executable(
    TestApplication
    src/track_KLT.cpp
    src/feature_SIFT.cpp
    src/feature_SURF.cpp
    src/feature_SURF.ocl.cpp
    src/feature_EAS.cpp
    src/feature_GoodFeature.cpp
    src/denoise_RANSAC.cpp
    src/descriptor_BRIEF.cpp
    src/xcv/xopencl_kernels_video.cpp
    src/xcv/xopencl_kernels_imgproc.cpp
    src/xcv/xopencl_kernels_xfeatures2d.cpp
    src/xcv/xrho.cpp
    src/xcv/xptsetreg.cpp
    main.cpp)

target_link_libraries(TestApplication ${_LINKER_LIBS})