#include "cuda_utils.h"
#include <stdio.h>

// CUDA kernel for matrix multiplication
// C[i][j] = sum(A[i][k] * B[k][j]) for k in [0, colsA)
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C,
                                     int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// Optimized kernel using shared memory
__global__ void matrixMultiplyKernelShared(const float* A, const float* B, float* C,
                                           int rowsA, int colsA, int colsB) {
    const int TILE_SIZE = 16;
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (colsA + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < rowsA && (t * TILE_SIZE + threadIdx.x) < colsA) {
            As[threadIdx.y][threadIdx.x] = A[row * colsA + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((t * TILE_SIZE + threadIdx.y) < colsA && col < colsB) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * colsB + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = sum;
    }
}

// Host function to launch CUDA matrix multiplication
void matrixMultiplyCUDA(const float* d_A, const float* d_B, float* d_C,
                        int rowsA, int colsA, int colsB) {
    // Use shared memory optimization for better performance
    const int TILE_SIZE = 16;
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((colsB + TILE_SIZE - 1) / TILE_SIZE, 
                 (rowsA + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel
    matrixMultiplyKernelShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, 
                                                       rowsA, colsA, colsB);
    
    // Check for kernel launch errors
    CUDA_CHECK_KERNEL();
}

// Initialize CUDA device for the current MPI rank
void initializeCUDA(int rank) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found on rank %d\n", rank);
        exit(EXIT_FAILURE);
    }
    
    // Set device for this rank (useful if multiple GPUs per node)
    int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("[Rank %d] Using CUDA device %d: %s (Compute %d.%d)\n",
           rank, device, prop.name, prop.major, prop.minor);
    printf("[Rank %d] Total Global Memory: %.2f GB\n",
           rank, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}
