# Default backend
BACKEND ?= cpu

CXX = g++
CXXFLAGS = -Wall -g -std=c++14

BUILD_DIR = ./build
TARGET = $(BUILD_DIR)/snp_system

# Common sources
SRCS = snp.cpp snp_system.cpp sorting.cpp
OBJS = $(BUILD_DIR)/snp.o $(BUILD_DIR)/snp_system.o $(BUILD_DIR)/sorting.o

ifeq ($(BACKEND), gpu)
    CXXFLAGS += -DUSE_GPU
    # Add CUDA paths and libraries
    CUDA_PATH ?= /usr/local/cuda
    CXXFLAGS += -I$(CUDA_PATH)/include
    LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart
    
    # GPU specific sources
    OBJS += $(BUILD_DIR)/linear_algebra_gpu.o $(BUILD_DIR)/cuda_kernels.o
else ifeq ($(BACKEND), distributed)
    CXXFLAGS += -DUSE_DISTRIBUTED
    CXX = mpic++
    # Distributed specific sources
    OBJS += $(BUILD_DIR)/linear_algebra_distributed.o $(BUILD_DIR)/linear_algebra_cpu.o
else
    # CPU specific sources
    OBJS += $(BUILD_DIR)/linear_algebra_cpu.o
endif

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Generic rule for local cpp files
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rules for external dependencies
$(BUILD_DIR)/linear_algebra_cpu.o: ../linear_algebra/cpu/linear_algebra_cpu.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/linear_algebra_gpu.o: ../linear_algebra/gpu/linear_algebra_gpu.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/linear_algebra_distributed.o: ../linear_algebra/distributed/linear_algebra_distributed.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for CUDA kernels
$(BUILD_DIR)/cuda_kernels.o: ../linear_algebra/gpu/cuda_kernels.cu
	@mkdir -p $(BUILD_DIR)
	nvcc -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

run: $(TARGET)
	@echo "Running simulation..."
	$(TARGET)

help:
	@echo "Available targets:"
	@echo "  make run-cpu [ARGS='...']     - Run CPU version"
	@echo "  make run-gpu [ARGS='...']     - Run GPU version"
	@echo "  make run-dist [ARGS='...']    - Run distributed version"

run-cpu:
	$(MAKE) clean
	$(MAKE) all BACKEND=cpu
	$(TARGET) $(ARGS)

run-gpu:
	$(MAKE) clean
	$(MAKE) all BACKEND=gpu
	$(TARGET) $(ARGS)

run-dist:
	$(MAKE) clean
	$(MAKE) all BACKEND=distributed
	mpirun --oversubscribe -np 4 $(TARGET) $(ARGS)

.PHONY: all run clean help run-cpu run-gpu run-dist