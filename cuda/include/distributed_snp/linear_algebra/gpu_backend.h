#ifndef LINEAR_ALGEBRA_GPU_H
#define LINEAR_ALGEBRA_GPU_H

#include "linear_algebra.h"

namespace LinAlg {
    // GPU-based matrix multiplication implementation using CUDA
    class LinearAlgebraGPU : public LinearAlgebra {
    public:
        LinearAlgebraGPU() = default;
        virtual ~LinearAlgebraGPU() = default;
        
        // Implements GPU-based matrix multiplication using CUDA
        void multiply(const float* A, const float* B, float* C,
                      int rowsA, int colsA, int colsB) override;

        // SNP System Operations (Int) - Currently falling back to CPU or throwing
        void multiplyVectorMatrix(const int* vec, const int* mat, int* result, int rows, int cols) override;
        void hadamardProduct(const int* a, const int* b, int* result, int size) override;
        void addVectors(const int* a, const int* b, int* result, int size) override;
    };
}

#endif // LINEAR_ALGEBRA_GPU_H
