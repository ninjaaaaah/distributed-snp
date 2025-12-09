#include "../matrix_mul.h"
#include "cuda_utils.h"
#include <stdio.h>
#include <stdlib.h>

// GPU-based matrix multiplication implementation using CUDA
void matrixMultiply(const float* A, const float* B, float* C,
                    int rowsA, int colsA, int colsB) {
    // Initialize CUDA
    static bool initialized = false;
    if (!initialized) {
        initializeCUDA();
        initialized = true;
    }
    
    // Calculate sizes
    size_t sizeA = rowsA * colsA * sizeof(float);
    size_t sizeB = colsA * colsB * sizeof(float);
    size_t sizeC = rowsA * colsB * sizeof(float);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));
    
    // Copy input matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice));
    
    // Perform matrix multiplication on GPU
    matrixMultiplyCUDA(d_A, d_B, d_C, rowsA, colsA, colsB);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}
