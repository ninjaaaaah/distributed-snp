#include "distributed_snp/linear_algebra/matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Initialize matrix with random floating-point numbers
void initializeMatrix(float* matrix, int rows, int cols, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / (float)RAND_MAX * 10.0f; // Random values [0, 10)
    }
}

// Print a matrix (for debugging small matrices)
void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    printf("\nMatrix %s (%dx%d):\n", name, rows, cols);
    
    // Only print small matrices to avoid flooding the output
    int maxPrint = 10;
    int printRows = rows < maxPrint ? rows : maxPrint;
    int printCols = cols < maxPrint ? cols : maxPrint;
    
    for (int i = 0; i < printRows; i++) {
        for (int j = 0; j < printCols; j++) {
            printf("%8.3f ", matrix[i * cols + j]);
        }
        if (cols > maxPrint) printf("...");
        printf("\n");
    }
    
    if (rows > maxPrint) {
        printf("...\n");
    }
    printf("\n");
}

// Verify the result by computing a few elements on CPU
bool verifyResult(const float* A, const float* B, const float* C, 
                  int rowsA, int colsA, int colsB, float tolerance) {
    // Check a few random elements
    int numChecks = (rowsA < 10 && colsB < 10) ? rowsA * colsB : 100;
    
    for (int check = 0; check < numChecks; check++) {
        int i = rand() % rowsA;
        int j = rand() % colsB;
        
        float expected = 0.0f;
        for (int k = 0; k < colsA; k++) {
            expected += A[i * colsA + k] * B[k * colsB + j];
        }
        
        float actual = C[i * colsB + j];
        float diff = fabs(expected - actual);
        
        if (diff > tolerance) {
            printf("Verification failed at C[%d][%d]: expected %.6f, got %.6f (diff: %.6f)\n",
                   i, j, expected, actual, diff);
            return false;
        }
    }
    
    printf("Verification passed! Checked %d random elements.\n", numChecks);
    return true;
}
