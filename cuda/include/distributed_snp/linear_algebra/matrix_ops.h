#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stdlib.h>

// Function declarations
void initializeMatrix(float* matrix, int rows, int cols, unsigned int seed);
void printMatrix(const float* matrix, int rows, int cols, const char* name);
bool verifyResult(const float* A, const float* B, const float* C, 
                  int rowsA, int colsA, int colsB, float tolerance);

#endif // MATRIX_OPS_H
