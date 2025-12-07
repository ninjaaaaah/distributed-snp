#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

// Error handling macro for CUDA
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) MPI_Abort(MPI_COMM_WORLD, code);
   }
}

// CUDA Kernel: Computes a partial C = A_sub * B
// Each thread computes one element of the sub-matrix C
__global__ void matMulKernel(float *d_A, float *d_B, float *d_C, int N, int rowsPerProcess) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsPerProcess && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += d_A[row * N + k] * d_B[k * N + col];
        }
        d_C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 1. Setup Parameters
    int N = 1024; // Matrix size (1024 x 1024)
    
    // Ensure N is divisible by world_size for this simple example
    if (N % world_size != 0) {
        if (world_rank == 0) std::cerr << "Error: N must be divisible by world_size" << std::endl;
        MPI_Finalize();
        return -1;
    }

    int rowsPerProcess = N / world_size;
    int elementsPerProcess = rowsPerProcess * N;
    int elementsB = N * N;

    // 2. Assign GPU to MPI Process
    // If you have multiple GPUs on one node, this ensures round-robin assignment
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(world_rank % num_devices);

    // 3. Allocate Host Memory
    std::vector<float> h_A;
    std::vector<float> h_B(elementsB);
    std::vector<float> h_C;
    std::vector<float> h_A_local(elementsPerProcess);
    std::vector<float> h_C_local(elementsPerProcess);

    // Master initializes data
    if (world_rank == 0) {
        h_A.resize(N * N);
        h_C.resize(N * N);
        for (int i = 0; i < N * N; ++i) {
            h_A[i] = 1.0f; // Simplified for verification
            h_B[i] = 1.0f;
        }
    }

    // 4. Distribute Data (MPI)
    
    // Broadcast Matrix B to ALL processes (B is needed entirely)
    // Note: In Bcast, root sends, others receive.
    if (world_rank == 0) {
        MPI_Bcast(h_B.data(), elementsB, MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(h_B.data(), elementsB, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // Scatter Matrix A (Break A into horizontal bands)
    MPI_Scatter(h_A.data(), elementsPerProcess, MPI_FLOAT, 
                h_A_local.data(), elementsPerProcess, MPI_FLOAT, 
                0, MPI_COMM_WORLD);

    // 5. CUDA Computation
    float *d_A, *d_B, *d_C;

    // Allocate Device Memory
    cudaCheckError(cudaMalloc(&d_A, elementsPerProcess * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_B, elementsB * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_C, elementsPerProcess * sizeof(float)));

    // Copy Data to Device
    cudaCheckError(cudaMemcpy(d_A, h_A_local.data(), elementsPerProcess * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B.data(), elementsB * sizeof(float), cudaMemcpyHostToDevice));

    // Define Grid/Block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (rowsPerProcess + blockSize.y - 1) / blockSize.y);

    // Launch Kernel
    matMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, rowsPerProcess);
    cudaCheckError(cudaDeviceSynchronize());

    // Copy Result back to Host
    cudaCheckError(cudaMemcpy(h_C_local.data(), d_C, elementsPerProcess * sizeof(float), cudaMemcpyDeviceToHost));

    // 6. Gather Results (MPI)
    MPI_Gather(h_C_local.data(), elementsPerProcess, MPI_FLOAT, 
               h_C.data(), elementsPerProcess, MPI_FLOAT, 
               0, MPI_COMM_WORLD);

    // 7. Cleanup & Verify
    if (world_rank == 0) {
        // Simple check: C[0] should be N (since 1.0 * 1.0 * N times)
        std::cout << "Computation Complete." << std::endl;
        std::cout << "Verification: C[0] = " << h_C[0] << " (Expected: " << (float)N << ")" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    MPI_Finalize();
    return 0;
}