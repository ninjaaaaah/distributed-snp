#ifndef LINEAR_ALGEBRA_CPU_H
#define LINEAR_ALGEBRA_CPU_H

#include "../linear_algebra.h"

namespace LinAlg {
    // CPU-based matrix multiplication implementation
    class LinearAlgebraCPU : public LinearAlgebra {
    public:
        LinearAlgebraCPU() = default;
        virtual ~LinearAlgebraCPU() = default;
        
        // Implements CPU-based matrix multiplication
        void multiply(const float* A, const float* B, float* C,
                      int rowsA, int colsA, int colsB) override;

        void hadamardProduct(const float* A, const float* B, float* C, int size) override;
        void addVectors(const float* A, const float* B, float* C, int size) override;

        // SNP System Operations (Int)
        void multiplyVectorMatrix(const int* vec, const int* mat, int* result, int rows, int cols) override;
        void hadamardProduct(const int* a, const int* b, int* result, int size) override;
        void addVectors(const int* a, const int* b, int* result, int size) override;
    };
}

#endif // LINEAR_ALGEBRA_CPU_H
