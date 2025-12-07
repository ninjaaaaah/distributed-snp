# Role
Act as an expert in High-Performance Computing (HPC), specializing in CUDA (Compute Unified Device Architecture) and MPI (Message Passing Interface).

# Objective
Create a complete, robust, and distributable C++ project that performs distributed parallel matrix multiplication using a hybrid CUDA + MPI approach.

# Infrastructure Context
* **Hardware:** 2 Virtual Machines on Google Cloud Platform (GCP).
* **Accelerators:** Each VM is equipped with 1 NVIDIA T4 GPU.
* **Network:** Standard TCP/IP between nodes.

# Technical Requirements

## 1. Core Implementation
* **Algorithm:** Implement parallel matrix multiplication (C = A * B).
* **Hybrid Model:**
    * Use **MPI** for inter-node communication (distributing blocks of the matrix between the two VMs).
    * Use **CUDA** for intra-node acceleration (performing the actual multiplication on the T4 GPU).
* **Data Types:** Use `float` (single-precision floating-point) for matrix elements.
* **Initialization:** Populate matrices A and B with random floating-point numbers.

## 2. Architecture & Code Structure
* **Separation of Concerns:** Decompose the program into logical files (e.g., `main.cpp`, `mpi_utils.cpp`, `cuda_kernels.cu`, `matrix_ops.cpp`).
* **Directory Structure:**
    * `./src/`: All source code (`.cpp`, `.cu`, `.h`).
    * `./build/`: Compiled executables and artifacts.
* **Error Handling:**
    * Implement robust error checking for all MPI calls (e.g., `MPI_Comm_rank`).
    * Implement robust error checking for all CUDA calls (e.g., `cudaMalloc`, `cudaMemcpy`, kernel launches).
    * Use a macro or helper function to wrap CUDA calls and catch errors immediately.
* **Logging:** Implement a logging system that outputs process status, errors, and performance metrics to `stdout` or a log file.

## 3. Build & Deployment System
* **Makefile:** Create a comprehensive `Makefile` that:
    * Compiles C++ code with `mpicxx`.
    * Compiles CUDA code with `nvcc`.
    * Links object files correctly.
    * Includes a `clean` target.
    * **Crucial:** Includes a `deploy` or `run` target that uses `scp` to distribute the executable to the second node and `mpirun` to execute the job across both nodes (assume SSH keys are set up).

## 4. Benchmarking
* Instrument the code to measure:
    * Total execution time.
    * Communication time (MPI overhead).
    * Computation time (CUDA kernel execution).
* Output the performance metrics clearly at the end of execution.

# Deliverables
1.  The complete file structure and content for all source files.
2.  The `Makefile`.
3.  A brief `README` guide explaining how to configure the hostfile and environment variables for the multi-node run.