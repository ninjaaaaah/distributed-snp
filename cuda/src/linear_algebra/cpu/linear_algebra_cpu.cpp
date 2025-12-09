#include "distributed_snp/linear_algebra/cpu_backend.h"
#include <stdio.h>
#include <cstring>

namespace LinAlg {
    // CPU-based matrix multiplication implementation
    // Computes C = A * B where:
    //   A is a matrix of size (rowsA x colsA)
    //   B is a matrix of size (colsA x colsB)
    //   C is a matrix of size (rowsA x colsB)
    // Matrices are stored in row-major order
    void LinearAlgebraCPU::multiply(const float* A, const float* B, float* C,
                                    int rowsA, int colsA, int colsB) {
        // Standard matrix multiplication algorithm
        // C[i][j] = sum(A[i][k] * B[k][j]) for k in [0, colsA)
        
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                float sum = 0.0f;
                
                // Compute dot product of row i of A with column j of B
                for (int k = 0; k < colsA; k++) {
                    sum += A[i * colsA + k] * B[k * colsB + j];
                }
                
                C[i * colsB + j] = sum;
            }
        }
    }

    void LinearAlgebraCPU::hadamardProduct(const float* A, const float* B, float* C, int size) {
        for (int i = 0; i < size; ++i) {
            C[i] = A[i] * B[i];
        }
    }

    void LinearAlgebraCPU::addVectors(const float* A, const float* B, float* C, int size) {
        for (int i = 0; i < size; ++i) {
            C[i] = A[i] + B[i];
        }
    }

    // SNP System Operations (Int)
    void LinearAlgebraCPU::multiplyVectorMatrix(const int* vec, const int* mat, int* result, int rows, int cols) {
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

    void LinearAlgebraCPU::hadamardProduct(const int* a, const int* b, int* result, int size) {
        for (int i = 0; i < size; ++i) {
            result[i] = a[i] * b[i];
        }
    }

    void LinearAlgebraCPU::addVectors(const int* a, const int* b, int* result, int size) {
        for (int i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
}
