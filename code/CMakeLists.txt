cmake_minimum_required(VERSION 2.8)
project(CUDA-stitch)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-all-loops")

list(APPEND CUDA_NVCC_FLAGS "-arch=sm_50;-std=c++11;-DVERBOSE")

#list(APPEND CUDA_NVCC_FLAGS "-arch=compute_20;-std=c++11;-DVERBOSE")
#set(CUDA_ARCH "-arch=compute_20" CACHE STRING "Value of the NVCC -arch option.")
#SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")

file(GLOB  cu  *.cu *.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

CUDA_ADD_EXECUTABLE(CUDA-Stitch main.cpp InterestPoint.cpp ResponseLayer.cpp FastHessian.cpp Timer.cpp Brief.cpp)
#add_executable(ParaPano ${SOURCE_FILES})

target_link_libraries(CUDA-Stitch ${CUDA_LIBRARIES} ${OpenCV_LIBS})
