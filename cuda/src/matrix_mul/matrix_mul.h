#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Matrix multiplication function declaration
// Computes C = A * B where:
//   A is a matrix of size (rowsA x colsA)
//   B is a matrix of size (colsA x colsB)
//   C is a matrix of size (rowsA x colsB)
void matrixMultiply(const float* A, const float* B, float* C,
                    int rowsA, int colsA, int colsB);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_MUL_H
