#include "distributed_snp/linear_algebra/linear_algebra.h"
#include "distributed_snp/linear_algebra/matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cmath>
#include <memory>

// Include the appropriate implementation based on what's being compiled
#ifdef ENABLE_GPU
    #include "distributed_snp/linear_algebra/gpu_backend.h"
#else
    #include "distributed_snp/linear_algebra/cpu_backend.h"
#endif

// Get current time in seconds (with microsecond precision)
double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    // Matrix dimensions (can be passed as command-line arguments)
    int M = 4;  // Rows of A and C
    int K = 4;  // Cols of A, Rows of B
    int N = 4;  // Cols of B and C
    
    if (argc >= 4) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    }
    
    printf("=================================================\n");
    printf("CPU Matrix Multiplication Example\n");
    printf("=================================================\n");
    printf("Matrix dimensions: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", 
           M, K, K, N, M, N);
    printf("Total elements to compute: %d\n", M * N);
    printf("=================================================\n\n");
    
    // Allocate matrices
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C = (float*)malloc(M * N * sizeof(float));
    
    if (!A || !B || !C) {
        fprintf(stderr, "Error: Failed to allocate memory for matrices\n");
        return EXIT_FAILURE;
    }
    
    // Initialize matrices with random values
    printf("Initializing matrices...\n");
    initializeMatrix(A, M, K, 12345);
    initializeMatrix(B, K, N, 67890);
    
    // Print matrices if they're small enough
    if (M <= 10 && K <= 10 && N <= 10) {
        printMatrix(A, M, K, "A");
        printMatrix(B, K, N, "B");
    }
    
    // Perform matrix multiplication
    printf("Performing matrix multiplication...\n");
    double start = getTime();
    
    // Create the appropriate implementation
    std::unique_ptr<LinAlg::LinearAlgebra> multiplier;
#ifdef ENABLE_GPU
    multiplier = std::make_unique<LinAlg::LinearAlgebraGPU>();
    printf("Using GPU implementation\n");
#else
    multiplier = std::make_unique<LinAlg::LinearAlgebraCPU>();
    printf("Using CPU implementation\n");
#endif
    
    multiplier->multiply(A, B, C, M, K, N);
    double end = getTime();
    
    double elapsed = end - start;
    printf("Computation time: %.6f seconds\n", elapsed);
    
    // Calculate performance metrics
    double gflops = (2.0 * M * K * N) / (elapsed * 1e9);
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print result if matrix is small enough
    if (M <= 10 && N <= 10) {
        printMatrix(C, M, N, "C");
    }
    
    // Verify a sample element (middle of matrix)
    if (M > 0 && N > 0) {
        int i = M / 2;
        int j = N / 2;
        float expected = 0.0f;
        for (int k = 0; k < K; k++) {
            expected += A[i * K + k] * B[k * N + j];
        }
        float actual = C[i * N + j];
        printf("\nVerification (element C[%d][%d]):\n", i, j);
        printf("  Expected: %.6f\n", expected);
        printf("  Actual:   %.6f\n", actual);
        printf("  Match:    %s\n", (fabs(expected - actual) < 1e-3) ? "YES" : "NO");
    }
    
    // Clean up
    free(A);
    free(B);
    free(C);
    
    printf("\n=================================================\n");
    printf("Matrix multiplication completed successfully!\n");
    printf("=================================================\n");
    
    return EXIT_SUCCESS;
}
