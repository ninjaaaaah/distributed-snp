#include "mpi_utils.h"
#include <stdio.h>
#include <stdlib.h>

// Initialize MPI environment
void initializeMPI(int* argc, char*** argv, int* rank, int* size) {
    MPI_CHECK(MPI_Init(argc, argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, size));
    
    if (*rank == 0) {
        printf("=================================================\n");
        printf("MPI Distributed Matrix Multiplication with CUDA\n");
        printf("=================================================\n");
        printf("Number of MPI processes: %d\n", *size);
    }
    
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_CHECK(MPI_Get_processor_name(processor_name, &name_len));
    printf("[Rank %d] Running on processor: %s\n", *rank, processor_name);
}

// Finalize MPI environment
void finalizeMPI() {
    MPI_CHECK(MPI_Finalize());
}

// Distribute rows of matrix A to all processes
void distributeMatrixA(const float* A, float* localA, int totalRows, 
                       int cols, int localRows, int rank, int size) {
    // Calculate send counts and displacements for Scatterv
    int* sendcounts = NULL;
    int* displs = NULL;
    
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        int rows_per_proc = totalRows / size;
        int remainder = totalRows % size;
        int offset = 0;
        
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * cols;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    
    int recvcount = localRows * cols;
    
    MPI_CHECK(MPI_Scatterv(A, sendcounts, displs, MPI_FLOAT,
                          localA, recvcount, MPI_FLOAT,
                          0, MPI_COMM_WORLD));
    
    if (rank == 0) {
        free(sendcounts);
        free(displs);
    }
    
    printf("[Rank %d] Received %d rows of matrix A (%d elements)\n", 
           rank, localRows, recvcount);
}

// Broadcast entire matrix B to all processes
void distributeMatrixB(const float* B, float* localB, int rows, int cols, 
                       int rank) {
    int size = rows * cols;
    
    MPI_CHECK(MPI_Bcast(localB, size, MPI_FLOAT, 0, MPI_COMM_WORLD));
    
    printf("[Rank %d] Received matrix B (%dx%d = %d elements)\n", 
           rank, rows, cols, size);
}

// Gather local result matrices from all processes
void gatherResultMatrix(const float* localC, float* C, int totalRows, 
                        int cols, int localRows, int rank, int size) {
    // Calculate receive counts and displacements for Gatherv
    int* recvcounts = NULL;
    int* displs = NULL;
    
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        int rows_per_proc = totalRows / size;
        int remainder = totalRows % size;
        int offset = 0;
        
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = rows * cols;
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }
    
    int sendcount = localRows * cols;
    
    MPI_CHECK(MPI_Gatherv(localC, sendcount, MPI_FLOAT,
                         C, recvcounts, displs, MPI_FLOAT,
                         0, MPI_COMM_WORLD));
    
    printf("[Rank %d] Sent %d rows of result matrix C (%d elements)\n", 
           rank, localRows, sendcount);
    
    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }
}
