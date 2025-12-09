cmake_minimum_required(VERSION 3.18)
project(DistributedSNP LANGUAGES CXX CUDA)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# Find MPI
find_package(MPI REQUIRED)

# Include Directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_PATH}
    /usr/local/cuda/include
)

# Source Files
set(COMMON_SOURCES
    src/common/cuda_utils.cu
)

set(LINALG_SOURCES
    src/linear_algebra/cpu/linear_algebra_cpu.cpp
    src/linear_algebra/gpu/linear_algebra_gpu.cpp
    src/linear_algebra/gpu/kernels.cu
    src/linear_algebra/distributed/linear_algebra_dist.cpp
    src/linear_algebra/distributed/mpi_utils.cpp
    src/linear_algebra/distributed/matrix_ops.cpp
)

set(SNP_SOURCES
    src/snp/snp_system.cpp
    src/snp/sorting.cpp
)

# Main Executable
add_executable(snp_system
    src/main.cpp
    ${COMMON_SOURCES}
    ${LINALG_SOURCES}
    ${SNP_SOURCES}
)

# Link Libraries
target_link_libraries(snp_system
    PRIVATE
    MPI::MPI_CXX
)

# Compiler Definitions
target_compile_definitions(snp_system PRIVATE
    ENABLE_GPU
    USE_GPU # For SNP system
)

# Benchmark Executable (CPU)
add_executable(snp_benchmark_cpu
    src/benchmark.cpp
    ${COMMON_SOURCES}
    ${LINALG_SOURCES}
    ${SNP_SOURCES}
)
target_link_libraries(snp_benchmark_cpu PRIVATE MPI::MPI_CXX)

# Benchmark Executable (GPU)
add_executable(snp_benchmark_gpu
    src/benchmark.cpp
    ${COMMON_SOURCES}
    ${LINALG_SOURCES}
    ${SNP_SOURCES}
)
target_link_libraries(snp_benchmark_gpu PRIVATE MPI::MPI_CXX)
target_compile_definitions(snp_benchmark_gpu PRIVATE ENABLE_GPU USE_GPU)

# Benchmark Executable (Distributed)
add_executable(snp_benchmark_dist
    src/benchmark.cpp
    ${COMMON_SOURCES}
    ${LINALG_SOURCES}
    ${SNP_SOURCES}
)
target_link_libraries(snp_benchmark_dist PRIVATE MPI::MPI_CXX)
target_compile_definitions(snp_benchmark_dist PRIVATE USE_DISTRIBUTED)

# Tests
enable_testing()

add_executable(test_mpi_ops
    tests/test_mpi_ops.cpp
    ${COMMON_SOURCES}
    ${LINALG_SOURCES}
)

target_link_libraries(test_mpi_ops
    PRIVATE
    MPI::MPI_CXX
)

# Output Directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
