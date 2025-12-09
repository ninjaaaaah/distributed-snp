#include "distributed_snp/linear_algebra/gpu_backend.h"
#include "distributed_snp/common/cuda_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

namespace LinAlg {
    // GPU-based matrix multiplication implementation using CUDA
    void LinearAlgebraGPU::multiply(const float* A, const float* B, float* C,
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

    // SNP System Operations (Int) - Fallback to CPU for now
    void LinearAlgebraGPU::multiplyVectorMatrix(const int* vec, const int* mat, int* result, int rows, int cols) {
        // Initialize result to 0
        std::memset(result, 0, cols * sizeof(int));

        for (int c = 0; c < cols; ++c) {        // Iterate Columns
            for (int r = 0; r < rows; ++r) {    // Iterate Rows
                // Standard dot product logic
                if (vec[r] != 0) { // Optimization for sparse vectors
                    result[c] += vec[r] * mat[r * cols + c];
                }
            }
        }
    }

    void LinearAlgebraGPU::hadamardProduct(const int* a, const int* b, int* result, int size) {
        for (int i = 0; i < size; ++i) {
            result[i] = a[i] * b[i];
        }
    }

    void LinearAlgebraGPU::addVectors(const int* a, const int* b, int* result, int size) {
        for (int i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
}
