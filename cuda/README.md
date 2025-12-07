# Distributed Matrix Multiplication with CUDA + MPI

A high-performance distributed parallel matrix multiplication implementation using CUDA for GPU acceleration and MPI for inter-node communication.

## Overview

This project implements matrix multiplication (C = A × B) using a hybrid approach:
- **MPI** for distributing work across multiple nodes
- **CUDA** for GPU-accelerated computation on each node

## Infrastructure

- **Platform**: Google Cloud Platform (GCP)
- **Configuration**: 2 Virtual Machines
- **GPU**: 1 × NVIDIA T4 per VM
- **Network**: TCP/IP

## Project Structure

```
cuda/
├── Makefile              # Build and deployment automation
├── hostfile              # MPI host configuration (you need to create this)
├── README.md             # This file
├── src/                  # Source code
│   ├── main.cpp          # Main program
│   ├── cuda_kernels.cu   # CUDA kernel implementations
│   ├── cuda_utils.h      # CUDA utilities and error checking
│   ├── mpi_utils.cpp     # MPI communication functions
│   ├── mpi_utils.h       # MPI utilities header
│   ├── matrix_ops.cpp    # Matrix operations (init, verify)
│   └── matrix_ops.h      # Matrix operations header
└── build/                # Compiled executables (auto-generated)
```

## Prerequisites

### Software Requirements

1. **CUDA Toolkit** (11.0 or later)
   ```bash
   nvcc --version
   ```

2. **MPI Implementation** (OpenMPI or MPICH)
   ```bash
   mpirun --version
   ```

3. **C++ Compiler** (g++ 7.0 or later)
   ```bash
   g++ --version
   ```

4. **SSH Keys** (for passwordless communication between nodes)
   ```bash
   ssh-keygen -t rsa
   ssh-copy-id user@secondary-node
   ```

## Configuration

### 1. Create Hostfile

Create a file named `hostfile` in the `cuda/` directory:

```bash
# hostfile
primary-node slots=1
secondary-node slots=1
```

Or use IP addresses:

```bash
# hostfile
10.0.0.1 slots=1
10.0.0.2 slots=1
```

**Important**: Ensure that both hostnames can be resolved via `/etc/hosts` or DNS.

### 2. Environment Variables

Set these variables in your shell or modify the Makefile:

```bash
export SECONDARY_HOST=secondary-node    # Hostname of second VM
export SECONDARY_USER=shared            # Username on second VM
export SECONDARY_PATH=/home/shared/distributed-snp/cuda
```

### 3. Test SSH Connectivity

Verify passwordless SSH access:

```bash
ssh $SECONDARY_USER@$SECONDARY_HOST "hostname"
```

This should return the hostname without prompting for a password.

### 4. Configure Network

Ensure both nodes can communicate:

```bash
# On primary node
ping secondary-node

# Test MPI connectivity
mpirun --hostfile hostfile hostname
```

## Building the Project

### Build on Primary Node

```bash
cd /home/shared/distributed-snp/cuda
make
```

This compiles:
- CUDA kernels (`cuda_kernels.cu`)
- MPI utilities (`mpi_utils.cpp`)
- Matrix operations (`matrix_ops.cpp`)
- Main program (`main.cpp`)

The executable will be in `build/matmul_dist`.

### Clean Build Artifacts

```bash
make clean
```

## Deployment

### Deploy to Secondary Node

```bash
make deploy
```

This automatically:
1. Creates necessary directories on the secondary node
2. Copies the executable via `scp`
3. Copies the hostfile

### Manual Deployment

```bash
# Create directories
ssh $SECONDARY_USER@$SECONDARY_HOST "mkdir -p /home/shared/distributed-snp/cuda/build"

# Copy executable
scp build/matmul_dist $SECONDARY_USER@$SECONDARY_HOST:/home/shared/distributed-snp/cuda/build/

# Copy hostfile
scp hostfile $SECONDARY_USER@$SECONDARY_HOST:/home/shared/distributed-snp/cuda/
```

## Running the Program

### Run Distributed (2 Nodes)

```bash
make run
```

This runs the program across both nodes with default matrix size (2048×2048).

### Run with Custom Matrix Dimensions

```bash
make run-custom M=4096 K=4096 N=4096
```

Where:
- `M` = rows of matrix A
- `K` = columns of A / rows of B
- `N` = columns of matrix B

### Run Locally (Single Node Testing)

```bash
make run-local
```

### Deploy and Run in One Command

```bash
make deploy-run
```

## Performance Tuning

### CUDA Optimization

The implementation uses:
- **Shared memory tiling** (16×16 tiles) for better memory coalescing
- **Optimized memory access patterns**
- **Compute capability 7.5** (NVIDIA T4)

To change GPU architecture, modify `CUDA_ARCH` in Makefile:

```makefile
CUDA_ARCH = -arch=sm_75  # For T4 (7.5)
```

### MPI Optimization

- **Data distribution**: Matrix A is distributed by rows using `MPI_Scatterv`
- **Broadcasting**: Matrix B is broadcast to all nodes using `MPI_Bcast`
- **Result gathering**: Results collected using `MPI_Gatherv`

### Benchmark Results

The program automatically outputs:
- Total execution time
- Communication time and percentage
- Computation time and percentage
- GFLOPS (billions of floating-point operations per second)
- Effective memory bandwidth

Example output:
```
=================================================
Performance Metrics
=================================================
Total execution time:     0.523456 seconds
Communication time:       0.089234 seconds (17.05%)
Computation time:         0.421087 seconds (80.45%)
Computational throughput: 40.89 GFLOPS
Effective bandwidth:      12.34 GB/s
=================================================
```

## Troubleshooting

### MPI Errors

**Problem**: "Cannot find hostfile"
```bash
# Create hostfile in cuda/ directory
echo "primary-node slots=1" > hostfile
echo "secondary-node slots=1" >> hostfile
```

**Problem**: "Connection refused"
```bash
# Check SSH connectivity
ssh secondary-node hostname

# Check firewall rules
sudo ufw status
```

**Problem**: "Unknown host"
```bash
# Add to /etc/hosts
echo "10.0.0.2 secondary-node" | sudo tee -a /etc/hosts
```

### CUDA Errors

**Problem**: "No CUDA devices found"
```bash
# Check GPU visibility
nvidia-smi

# Verify CUDA installation
nvcc --version
```

**Problem**: "CUDA out of memory"
```bash
# Reduce matrix size
make run-custom M=1024 K=1024 N=1024
```

### Build Errors

**Problem**: "nvcc: command not found"
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Problem**: "mpicxx: command not found"
```bash
# Install MPI
sudo apt-get install libopenmpi-dev openmpi-bin
```

## Advanced Usage

### Custom Hostfile Location

```bash
make run HOSTFILE=/path/to/custom/hostfile
```

### Different Secondary Host

```bash
make deploy SECONDARY_HOST=node2 SECONDARY_USER=ubuntu
```

### Run with More Processes

Modify the Makefile `run` target:

```makefile
mpirun -np 4 --hostfile $(HOSTFILE) $(BUILD_DIR)/$(TARGET)
```

### Enable MPI Verbose Output

```bash
mpirun -np 2 --hostfile hostfile --display-map --display-allocation build/matmul_dist
```

## Code Structure

### Main Components

1. **main.cpp**: Orchestrates the entire workflow
   - MPI initialization and finalization
   - Matrix distribution
   - CUDA kernel invocation
   - Result gathering and verification
   - Performance metrics

2. **cuda_kernels.cu**: GPU computation
   - Optimized matrix multiplication kernel with shared memory
   - CUDA device initialization
   - Error checking wrappers

3. **mpi_utils.cpp**: Inter-node communication
   - Scatter matrix A rows to processes
   - Broadcast matrix B to all processes
   - Gather result matrix C

4. **matrix_ops.cpp**: Utility functions
   - Random matrix initialization
   - Result verification
   - Matrix printing (for debugging)

### Error Handling

All CUDA and MPI calls are wrapped with error-checking macros:

```cpp
CUDA_CHECK(cudaMalloc(...));    // Checks CUDA errors
MPI_CHECK(MPI_Send(...));       // Checks MPI errors
```

## Performance Expectations

For 2048×2048 matrices on 2 × T4 GPUs:
- **Computation time**: ~0.4 seconds
- **Total time**: ~0.5 seconds
- **Throughput**: ~40 GFLOPS
- **Speedup**: ~1.8× compared to single node

## License

This project is for educational and research purposes.

## Contact

For issues or questions, please refer to the project repository.
