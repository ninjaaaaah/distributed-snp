# Makefile for Distributed Matrix Multiplication (CUDA + MPI)
# Targets: all, clean, deploy, run

# Compiler settings
NVCC = nvcc
MPICXX = mpicxx
TARGET = matmul_dist

# Directories
SRC_DIR = src
BUILD_DIR = build

# CUDA compute capability for T4 GPU (7.5)
CUDA_ARCH = -arch=sm_75

# Compiler flags
NVCC_FLAGS = $(CUDA_ARCH) -O3 -Xcompiler -fPIC
MPICXX_FLAGS = -O3 -Wall -I$(SRC_DIR)
CUDA_INCLUDE = -I/usr/local/cuda/include
CUDA_LIBS = -L/usr/local/cuda/lib64 -lcudart

# Source files
CU_SOURCES = $(SRC_DIR)/cuda_kernels.cu
CPP_SOURCES = $(SRC_DIR)/main.cpp $(SRC_DIR)/mpi_utils.cpp $(SRC_DIR)/matrix_ops.cpp

# Object files
CU_OBJECTS = $(BUILD_DIR)/cuda_kernels.o
CPP_OBJECTS = $(BUILD_DIR)/main.o $(BUILD_DIR)/mpi_utils.o $(BUILD_DIR)/matrix_ops.o
ALL_OBJECTS = $(CU_OBJECTS) $(CPP_OBJECTS)

# Deployment settings (modify these for your environment)
SECONDARY_HOST ?= secondary
SECONDARY_USER ?= shared
SECONDARY_PATH ?= /home/shared/distributed-snp/cuda
HOSTFILE ?= hostfile

# Default target
all: $(BUILD_DIR)/$(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile CUDA source files
$(BUILD_DIR)/cuda_kernels.o: $(SRC_DIR)/cuda_kernels.cu $(SRC_DIR)/cuda_utils.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/mpi_utils.h $(SRC_DIR)/cuda_utils.h $(SRC_DIR)/matrix_ops.h | $(BUILD_DIR)
	$(MPICXX) $(MPICXX_FLAGS) $(CUDA_INCLUDE) -c $< -o $@

$(BUILD_DIR)/mpi_utils.o: $(SRC_DIR)/mpi_utils.cpp $(SRC_DIR)/mpi_utils.h | $(BUILD_DIR)
	$(MPICXX) $(MPICXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/matrix_ops.o: $(SRC_DIR)/matrix_ops.cpp $(SRC_DIR)/matrix_ops.h | $(BUILD_DIR)
	$(MPICXX) $(MPICXX_FLAGS) -c $< -o $@

# Link all object files
$(BUILD_DIR)/$(TARGET): $(ALL_OBJECTS)
	$(MPICXX) $(MPICXX_FLAGS) -o $@ $^ $(CUDA_LIBS)
	@echo ""
	@echo "==============================================="
	@echo "Build completed successfully!"
	@echo "Executable: $(BUILD_DIR)/$(TARGET)"
	@echo "==============================================="

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	@echo "Build directory cleaned."

# Deploy to secondary node
deploy: all
	@echo ""
	@echo "==============================================="
	@echo "Deploying to secondary node..."
	@echo "==============================================="
	@ssh $(SECONDARY_USER)@$(SECONDARY_HOST) "mkdir -p $(SECONDARY_PATH)/build"
	@scp $(BUILD_DIR)/$(TARGET) $(SECONDARY_USER)@$(SECONDARY_HOST):$(SECONDARY_PATH)/build/
	@if [ -f $(HOSTFILE) ]; then \
		scp $(HOSTFILE) $(SECONDARY_USER)@$(SECONDARY_HOST):$(SECONDARY_PATH)/; \
	fi
	@echo "Deployment completed!"
	@echo "==============================================="

# Run the program locally (single node, useful for testing)
run-local: all
	@echo ""
	@echo "==============================================="
	@echo "Running on local node only..."
	@echo "==============================================="
	mpirun -np 1 $(BUILD_DIR)/$(TARGET)

# Run the program across both nodes
run: all
	@echo ""
	@echo "==============================================="
	@echo "Running distributed across nodes..."
	@echo "==============================================="
	mpirun -np 2 \
		--host localhost,$(SECONDARY_HOST) \
	 	--mca btl_tcp_if_include ens5 \
		--mca oob_tcp_if_include ens5 \
		$(BUILD_DIR)/$(TARGET)

# Run with custom matrix dimensions (M K N)
run-custom: all
	@if [ -z "$(M)" ] || [ -z "$(K)" ] || [ -z "$(N)" ]; then \
		echo "Error: Please specify matrix dimensions: M=<rows> K=<cols> N=<cols>"; \
		echo "Example: make run-custom M=4096 K=4096 N=4096"; \
		exit 1; \
	fi
	@echo ""
	@echo "==============================================="
	@echo "Running with custom dimensions: $(M)x$(K) * $(K)x$(N)"
	@echo "==============================================="
	mpirun -np 2 \
		--host localhost,$(SECONDARY_HOST) \
	 	--mca btl_tcp_if_include ens5 \
		--mca oob_tcp_if_include ens5 \
		$(BUILD_DIR)/$(TARGET)

# Deploy and run
deploy-run: deploy run

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the project (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  run-local    - Run on single node (testing)"
	@echo "  run          - Run distributed across nodes"
	@echo "  deploy       - Copy executable to secondary node"
	@echo "  deploy-run   - Deploy and run"
	@echo "  run-custom   - Run with custom matrix dimensions"
	@echo "                 Usage: make run-custom M=2048 K=2048 N=2048"
	@echo ""
	@echo "Configuration variables:"
	@echo "  SECONDARY_HOST - Hostname of secondary node (default: secondary)"
	@echo "  SECONDARY_USER - Username on secondary node (default: shared)"
	@echo "  SECONDARY_PATH - Path on secondary node (default: /home/shared/distributed-snp/cuda)"
	@echo "  HOSTFILE       - MPI hostfile (default: hostfile)"
	@echo ""
	@echo "Example:"
	@echo "  make SECONDARY_HOST=node2 HOSTFILE=myhosts run"

.PHONY: all clean run run-local deploy deploy-run run-custom help
