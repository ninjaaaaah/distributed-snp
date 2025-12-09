# Makefile for Matrix Multiplication
# Supports CPU, GPU (CUDA), and Distributed (MPI+CUDA) implementations
#
# Usage:
#   make cpu          - Build CPU-only version
#   make gpu          - Build GPU (CUDA) version
#   make distributed  - Build distributed (MPI+CUDA) version
#   make all          - Build all versions
#   make clean        - Remove all build artifacts
#   make run-cpu      - Run CPU version
#   make run-gpu      - Run GPU version
#   make run-dist     - Run distributed version with MPI

# Compiler settings
CXX = g++
NVCC = nvcc
MPICXX = mpicxx

# Directories
SRC_DIR = src
MATRIX_MUL_DIR = $(SRC_DIR)/matrix_mul
CPU_DIR = $(MATRIX_MUL_DIR)/cpu
GPU_DIR = $(MATRIX_MUL_DIR)/gpu
DIST_DIR = $(MATRIX_MUL_DIR)/distributed
BUILD_DIR = build

# Target names
TARGET_CPU = $(BUILD_DIR)/matmul_cpu
TARGET_GPU = $(BUILD_DIR)/matmul_gpu
TARGET_DIST = $(BUILD_DIR)/matmul_dist

# CUDA settings
CUDA_ARCH = -arch=sm_75
CUDA_INCLUDE = -I/usr/local/cuda/include
CUDA_LIBS = -L/usr/local/cuda/lib64 -lcudart

# Compiler flags
CXX_FLAGS = -O3 -Wall -std=c++11 -I$(MATRIX_MUL_DIR)
NVCC_FLAGS = $(CUDA_ARCH) -O3 -Xcompiler -fPIC -std=c++11
MPICXX_FLAGS = -O3 -Wall -std=c++11 -I$(MATRIX_MUL_DIR) $(CUDA_INCLUDE)

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help:
	@echo "Matrix Multiplication Build System"
	@echo "==================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make cpu          - Build CPU-only version"
	@echo "  make gpu          - Build GPU (CUDA) version"
	@echo "  make distributed  - Build distributed (MPI+CUDA) version"
	@echo "  make all          - Build all versions"
	@echo "  make clean        - Remove all build artifacts"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-cpu [ARGS='M K N']     - Run CPU version"
	@echo "  make run-gpu [ARGS='M K N']     - Run GPU version"
	@echo "  make run-dist [ARGS='M K N']    - Run distributed version"
	@echo ""
	@echo "Example:"
	@echo "  make cpu && make run-cpu ARGS='512 512 512'"
	@echo ""

# Build all versions
.PHONY: all
all: cpu gpu distributed
	@echo "All versions built successfully!"

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

###############################
# CPU Implementation
###############################

.PHONY: cpu
cpu: $(TARGET_CPU)

$(TARGET_CPU): $(SRC_DIR)/main.cpp $(CPU_DIR)/matrix_mul_cpu.cpp $(MATRIX_MUL_DIR)/matrix_mul.h | $(BUILD_DIR)
	@echo "Building CPU version..."
	$(CXX) $(CXX_FLAGS) -o $@ $(SRC_DIR)/main.cpp $(CPU_DIR)/matrix_mul_cpu.cpp
	@echo "CPU version built: $@"

###############################
# GPU Implementation (CUDA)
###############################

.PHONY: gpu
gpu: $(TARGET_GPU)

# Compile CUDA kernel
$(BUILD_DIR)/cuda_kernels.o: $(GPU_DIR)/cuda_kernels.cu $(GPU_DIR)/cuda_utils.h | $(BUILD_DIR)
	@echo "Compiling CUDA kernels..."
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Build GPU version
$(TARGET_GPU): $(SRC_DIR)/main.cpp $(GPU_DIR)/matrix_mul_gpu.cpp $(BUILD_DIR)/cuda_kernels.o $(MATRIX_MUL_DIR)/matrix_mul.h | $(BUILD_DIR)
	@echo "Building GPU version..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SRC_DIR)/main.cpp $(GPU_DIR)/matrix_mul_gpu.cpp $(BUILD_DIR)/cuda_kernels.o $(CUDA_LIBS)
	@echo "GPU version built: $@"

###############################
# Distributed Implementation (MPI+CUDA)
###############################

.PHONY: distributed
distributed: $(TARGET_DIST)

# Compile CUDA kernels for distributed version
$(BUILD_DIR)/cuda_kernels_dist.o: $(DIST_DIR)/cuda_kernels.cu $(DIST_DIR)/cuda_utils.h | $(BUILD_DIR)
	@echo "Compiling CUDA kernels for distributed version..."
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile MPI utilities
$(BUILD_DIR)/mpi_utils.o: $(DIST_DIR)/mpi_utils.cpp $(DIST_DIR)/mpi_utils.h | $(BUILD_DIR)
	@echo "Compiling MPI utilities..."
	$(MPICXX) $(MPICXX_FLAGS) -c $< -o $@

# Compile matrix operations
$(BUILD_DIR)/matrix_ops.o: $(DIST_DIR)/matrix_ops.cpp $(DIST_DIR)/matrix_ops.h | $(BUILD_DIR)
	@echo "Compiling matrix operations..."
	$(MPICXX) $(MPICXX_FLAGS) -c $< -o $@

# Compile distributed main
$(BUILD_DIR)/main_dist.o: $(DIST_DIR)/main.cpp $(DIST_DIR)/mpi_utils.h $(DIST_DIR)/cuda_utils.h $(DIST_DIR)/matrix_ops.h | $(BUILD_DIR)
	@echo "Compiling distributed main..."
	$(MPICXX) $(MPICXX_FLAGS) -c $< -o $@

# Link distributed version
$(TARGET_DIST): $(BUILD_DIR)/main_dist.o $(BUILD_DIR)/mpi_utils.o $(BUILD_DIR)/matrix_ops.o $(BUILD_DIR)/cuda_kernels_dist.o | $(BUILD_DIR)
	@echo "Linking distributed version..."
	$(MPICXX) $(MPICXX_FLAGS) -o $@ $^ $(CUDA_LIBS)
	@echo "Distributed version built: $@"

###############################
# Run Targets
###############################

# Default matrix dimensions
ARGS ?= 512 512 512

.PHONY: run-cpu
run-cpu: $(TARGET_CPU)
	@echo "Running CPU version with dimensions: $(ARGS)"
	@echo "=========================================="
	./$(TARGET_CPU) $(ARGS)

.PHONY: run-gpu
run-gpu: $(TARGET_GPU)
	@echo "Running GPU version with dimensions: $(ARGS)"
	@echo "=========================================="
	./$(TARGET_GPU) $(ARGS)

.PHONY: run-dist
run-dist: $(TARGET_DIST)
	@echo "Running distributed version with dimensions: $(ARGS)"
	@echo "Note: Using 2 MPI processes by default"
	@echo "=========================================="
	mpirun -np 2 \
		--host localhost,10.0.0.2 \
	 	--mca btl_tcp_if_include ens5 \
		--mca oob_tcp_if_include ens5 \
		./$(TARGET_DIST) $(ARGS)

###############################
# Clean
###############################

.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	@echo "Clean complete!"

###############################
# Info
###############################

.PHONY: info
info:
	@echo "Build Configuration:"
	@echo "==================="
	@echo "CXX       = $(CXX)"
	@echo "NVCC      = $(NVCC)"
	@echo "MPICXX    = $(MPICXX)"
	@echo "CUDA_ARCH = $(CUDA_ARCH)"
	@echo "BUILD_DIR = $(BUILD_DIR)"
	@echo ""
	@echo "Targets:"
	@echo "  CPU:         $(TARGET_CPU)"
	@echo "  GPU:         $(TARGET_GPU)"
	@echo "  Distributed: $(TARGET_DIST)"
