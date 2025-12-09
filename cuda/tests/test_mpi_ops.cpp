#include "distributed_snp/linear_algebra/mpi_utils.h"
#include "distributed_snp/common/cuda_utils.h"
#include "distributed_snp/linear_algebra/matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// Get current time in seconds (with microsecond precision)
double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    int rank, size;
    double total_start, total_end;
    double comm_time = 0.0, comp_time = 0.0;
    
    // Initialize MPI
    initializeMPI(&argc, &argv, &rank, &size);
    total_start = getTime();
    
    // Matrix dimensions (can be passed as command-line arguments)
    int M = 2048;  // Rows of A and C
    int K = 2048;  // Cols of A, Rows of B
    int N = 2048;  // Cols of B and C
    
    if (argc >= 4) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    }
    
    if (rank == 0) {
        printf("Matrix dimensions: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", 
               M, K, K, N, M, N);
        printf("Total elements to compute: %d\n", M * N);
        printf("=================================================\n\n");
    }
    
    // Initialize CUDA
    initializeCUDA(rank);
    
    // Calculate local dimensions
    int rows_per_proc = M / size;
    int remainder = M % size;
    int localRows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    if (rank == 0) {
        printf("\n=================================================\n");
        printf("Work Distribution:\n");
        printf("=================================================\n");
    }
    printf("[Rank %d] Processing %d rows (%.2f%% of total)\n", 
           rank, localRows, 100.0 * localRows / M);
    
    // Allocate host memory
    float *A = NULL, *B = NULL, *C = NULL;
    float *localA = (float*)malloc(localRows * K * sizeof(float));
    float *localB = (float*)malloc(K * N * sizeof(float));
    float *localC = (float*)malloc(localRows * N * sizeof(float));
    
    // Master process initializes matrices
    if (rank == 0) {
        A = (float*)malloc(M * K * sizeof(float));
        B = (float*)malloc(K * N * sizeof(float));
        C = (float*)malloc(M * N * sizeof(float));
        
        printf("\n[Rank 0] Initializing matrices...\n");
        initializeMatrix(A, M, K, time(NULL));
        initializeMatrix(B, K, N, time(NULL) + 1);
        
        // Optionally print small matrices
        if (M <= 10 && K <= 10 && N <= 10) {
            printMatrix(A, M, K, "A");
            printMatrix(B, K, N, "B");
        }
    }
    
    // Distribute data
    printf("\n=================================================\n");
    printf("Data Distribution Phase\n");
    printf("=================================================\n");
    
    double comm_start = getTime();
    distributeMatrixA(A, localA, M, K, localRows, rank, size);
    distributeMatrixB(B, localB, K, N, rank);
    comm_time += getTime() - comm_start;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, localRows * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, localRows * N * sizeof(float)));
    
    // Copy data to device
    printf("\n[Rank %d] Copying data to GPU...\n", rank);
    comm_start = getTime();
    CUDA_CHECK(cudaMemcpy(d_A, localA, localRows * K * sizeof(float), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, localB, K * N * sizeof(float), 
                          cudaMemcpyHostToDevice));
    comm_time += getTime() - comm_start;
    
    // Perform matrix multiplication on GPU
    printf("\n=================================================\n");
    printf("GPU Computation Phase\n");
    printf("=================================================\n");
    printf("[Rank %d] Starting CUDA matrix multiplication...\n", rank);
    
    double comp_start = getTime();
    matrixMultiplyCUDA(d_A, d_B, d_C, localRows, K, N);
    comp_time += getTime() - comp_start;
    
    printf("[Rank %d] CUDA computation completed.\n", rank);
    
    // Copy result back to host
    comm_start = getTime();
    CUDA_CHECK(cudaMemcpy(localC, d_C, localRows * N * sizeof(float), 
                          cudaMemcpyDeviceToHost));
    comm_time += getTime() - comm_start;
    
    // Gather results
    printf("\n=================================================\n");
    printf("Result Gathering Phase\n");
    printf("=================================================\n");
    
    comm_start = getTime();
    gatherResultMatrix(localC, C, M, N, localRows, rank, size);
    comm_time += getTime() - comm_start;
    
    total_end = getTime();
    
    // Master process verifies and prints results
    if (rank == 0) {
        printf("\n=================================================\n");
        printf("Verification Phase\n");
        printf("=================================================\n");
        
        if (M <= 1000 && N <= 1000) {
            bool correct = verifyResult(A, B, C, M, K, N, 0.01f);
            if (!correct) {
                printf("WARNING: Result verification failed!\n");
            }
        } else {
            printf("Skipping verification for large matrices.\n");
        }
        
        // Optionally print result
        if (M <= 10 && N <= 10) {
            printMatrix(C, M, N, "C (Result)");
        }
        
        // Print performance metrics
        printf("\n=================================================\n");
        printf("Performance Metrics\n");
        printf("=================================================\n");
        printf("Total execution time:     %.6f seconds\n", total_end - total_start);
        printf("Communication time:       %.6f seconds (%.2f%%)\n", 
               comm_time, 100.0 * comm_time / (total_end - total_start));
        printf("Computation time:         %.6f seconds (%.2f%%)\n", 
               comp_time, 100.0 * comp_time / (total_end - total_start));
        
        // Calculate GFLOPS
        double flops = 2.0 * M * N * K; // 2 operations per element (multiply + add)
        double gflops = (flops / comp_time) / 1e9;
        printf("Computational throughput: %.2f GFLOPS\n", gflops);
        
        // Memory bandwidth
        double memory_bytes = (M * K + K * N + M * N) * sizeof(float);
        double bandwidth = (memory_bytes / (total_end - total_start)) / (1024.0 * 1024.0 * 1024.0);
        printf("Effective bandwidth:      %.2f GB/s\n", bandwidth);
        
        printf("=================================================\n");
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    free(localA);
    free(localB);
    free(localC);
    
    if (rank == 0) {
        free(A);
        free(B);
        free(C);
        printf("\nProgram completed successfully!\n");
    }
    
    finalizeMPI();
    return 0;
}
