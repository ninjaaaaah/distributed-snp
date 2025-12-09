#ifndef COMMON_LINEAR_ALGEBRA_H
#define COMMON_LINEAR_ALGEBRA_H

#include <stdexcept>

namespace LinAlg {
    // Abstract base class for linear algebra operations
    // Supports both Float (MatrixMul) and Int (SNP) operations
    class LinearAlgebra {
    public:
        virtual ~LinearAlgebra() = default;

        // --- Float Operations (Matrix Multiplication) ---
        
        // Performs matrix multiplication: C = A * B
        virtual void multiply(const float* A, const float* B, float* C,
                              int rowsA, int colsA, int colsB) {
            throw std::runtime_error("multiply (float) not implemented");
        }

        virtual void hadamardProduct(const float* A, const float* B, float* C, int size) {
            throw std::runtime_error("hadamardProduct (float) not implemented");
        }

        virtual void addVectors(const float* A, const float* B, float* C, int size) {
            throw std::runtime_error("addVectors (float) not implemented");
        }

        // --- Int Operations (SNP System) ---

        // Performs Vector-Matrix Multiplication: result = vec * mat
        virtual void multiplyVectorMatrix(const int* vec, const int* mat, int* result, int rows, int cols) {
            throw std::runtime_error("multiplyVectorMatrix (int) not implemented");
        }

        // Performs Element-wise Multiplication (Hadamard Product): result = a (*) b
        virtual void hadamardProduct(const int* a, const int* b, int* result, int size) {
            throw std::runtime_error("hadamardProduct (int) not implemented");
        }

        // Vector Addition: result = a + b
        virtual void addVectors(const int* a, const int* b, int* result, int size) {
            throw std::runtime_error("addVectors (int) not implemented");
        }
    };
}

#endif // COMMON_LINEAR_ALGEBRA_H
