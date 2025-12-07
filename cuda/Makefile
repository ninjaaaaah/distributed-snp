# ==========================================
# Configuration
# ==========================================

# 1. Compiler Settings
NVCC        := nvcc
MPICXX      := mpic++ # Used only to extract flags, not to compile
BINARY      := matmul_distributed
SOURCE      := matmul_mpi_cuda.cu

# 2. MPI Paths (Adjust these if auto-detection fails)
# We try to auto-detect MPI paths using mpic++. 
# If this fails, manually set MPI_INCLUDE and MPI_LIB.
MPI_INCLUDE := $(shell $(MPICXX) -showme:compile | sed 's/-I//g' | awk '{print $$1}')
MPI_LINK    := $(shell $(MPICXX) -showme:link | sed 's/-pthread/-Xcompiler -pthread/g')

# 3. CUDA Architecture (Change based on your GPU, e.g., sm_70 for Volta, sm_80 for Ampere)
CUDA_ARCH   := -arch=sm_75

# 4. Cluster Settings for 'distribute' and 'run'
# List your node IP addresses or hostnames here, separated by spaces
NODES       := 10.140.0.2 10.0.0.2
# The number of processes to run (total)
NP          := 2
# Remote directory to copy the binary to
REMOTE_DIR  := ~/mpi_cuda_test

# ==========================================
# Targets
# ==========================================

.PHONY: all clean distribute run help

# Default target
all: $(BINARY)

# Compile
$(BINARY): $(SOURCE)
	@echo "Compiling CUDA + MPI..."
	$(NVCC) $(CUDA_ARCH) -I$(MPI_INCLUDE) $(MPI_LINK) -o $(BINARY) $(SOURCE)
	@echo "Build successful: $(BINARY)"

# Clean
clean:
	@echo "Cleaning up..."
	rm -f $(BINARY)

# Distribute: SCP the binary to all nodes defined in $(NODES)
distribute: all
	@echo "Distributing binary to nodes..."
	@for node in $(NODES); do \
		echo "Copying to $$node..."; \
		ssh $$node "mkdir -p $(REMOTE_DIR)"; \
		scp $(BINARY) $$node:$(REMOTE_DIR); \
	done
	@echo "Distribution complete."

# Run: Execute mpirun
run:
	@echo "Running MPI job on: $(NODES)"
	# We convert space-separated NODES to comma-separated for mpirun
	mpirun -np $(NP) --host $(shell echo $(NODES) | tr ' ' ',') $(REMOTE_DIR)/$(BINARY)

# Help
help:
	@echo "Available targets:"
	@echo "  make           : Compile the code"
	@echo "  make clean     : Remove the binary"
	@echo "  make distribute: Compile and SCP binary to all NODES"
	@echo "  make run       : Run the binary across NODES using mpirun"