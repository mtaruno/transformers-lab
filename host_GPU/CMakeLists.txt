cmake_minimum_required(VERSION 3.10)

project(CSE599 LANGUAGES CXX CUDA)

# Language standards and compiler settings
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 90)

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "90")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[compute_90a,sm_90a] -Xcompiler=-fPIE -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(
  copy
  main.cu
  copy_first_column.cu
)

