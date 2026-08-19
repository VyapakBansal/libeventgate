cmake_minimum_required(VERSION 3.24)
project(libeventgate LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89) # Ada; override with -DCMAKE_CUDA_ARCHITECTURES=
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-DNDEBUG>
)

set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d videoio highgui)
find_package(Threads REQUIRED)

if(DEFINED ENV{TENSORRT_ROOT})
  set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
endif()
set(TENSORRT_ROOT "${TENSORRT_ROOT}" CACHE PATH "TensorRT install prefix")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT}/include /usr/include/x86_64-linux-gnu /usr/include)
find_library(TENSORRT_NVINFER nvinfer
  HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)
find_library(TENSORRT_NVONNX nvonnxparser
  HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER AND TENSORRT_NVONNX)
  set(LIBEVENTGATE_HAS_TRT ON)
  message(STATUS "TensorRT: ${TENSORRT_INCLUDE_DIR} / ${TENSORRT_NVINFER}")
else()
  set(LIBEVENTGATE_HAS_TRT OFF)
  message(WARNING "TensorRT not found; voxel/smoke path only. Set TENSORRT_ROOT for FireNet.")
endif()

set(EVENTGATE_SOURCES
  src/config.cpp
  src/hdf5_events.cpp
  src/imu.cpp
  src/gate.cpp
  src/keypoints.cpp
  src/vram.cpp
  src/voxel_cuda.cpp
  src/pipeline.cpp
  cuda/voxelize.cu
)
if(LIBEVENTGATE_HAS_TRT)
  list(APPEND EVENTGATE_SOURCES src/firenet_trt.cpp)
endif()

add_library(eventgate_core ${EVENTGATE_SOURCES})

target_include_directories(eventgate_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(eventgate_core
  PUBLIC
    CUDA::cudart
    ${HDF5_C_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
    ${OpenCV_LIBS}
    Threads::Threads
)

if(LIBEVENTGATE_HAS_TRT)
  target_include_directories(eventgate_core PUBLIC ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(eventgate_core PUBLIC ${TENSORRT_NVINFER} ${TENSORRT_NVONNX})
  target_compile_definitions(eventgate_core PUBLIC LIBEVENTGATE_HAS_TRT=1)
else()
  target_compile_definitions(eventgate_core PUBLIC LIBEVENTGATE_HAS_TRT=0)
endif()

set_target_properties(eventgate_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_executable(eventgate_smoke apps/eventgate_smoke.cpp)
target_link_libraries(eventgate_smoke PRIVATE eventgate_core)

add_executable(eventgate apps/eventgate.cpp)
target_link_libraries(eventgate PRIVATE eventgate_core)

if(LIBEVENTGATE_HAS_TRT)
  add_executable(build_engine apps/build_engine.cpp)
  target_link_libraries(build_engine PRIVATE eventgate_core)
endif()

install(TARGETS eventgate_core eventgate_smoke eventgate
  RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
