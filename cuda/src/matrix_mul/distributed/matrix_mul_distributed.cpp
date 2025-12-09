#include "../matrix_mul.h"
#include <stdio.h>

// CPU-based matrix multiplication implementation
// Computes C = A * B where:
//   A is a matrix of size (rowsA x colsA)
//   B is a matrix of size (colsA x colsB)
//   C is a matrix of size (rowsA x colsB)
// Matrices are stored in row-major order
void matrixMultiply(const float* A, const float* B, float* C,
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
