cmake_minimum_required(VERSION 3.20)

# Quadro RTX 6000 (Turing, compute capability 7.5). Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(lab2 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Kernels, cuBLAS wrappers and file I/O, shared by both executables.
add_library(lab2_core STATIC src/kernels.cu src/reference.cu src/matrix_io.cpp)
target_include_directories(lab2_core PUBLIC include)
target_link_libraries(lab2_core PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(lab2_core PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)

# Graded CLI: build/submission
add_executable(submission src/main.cu)
target_link_libraries(submission PRIVATE lab2_core)
set_target_properties(submission PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)

# Experiments for the report: build/bench (not needed for grading)
add_executable(bench src/bench.cu src/bench_common.cu src/check.cu)
target_link_libraries(bench PRIVATE lab2_core)
set_target_properties(bench PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)
