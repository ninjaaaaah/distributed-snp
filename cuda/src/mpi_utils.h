#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <mpi.h>

// MPI error checking macro
#define MPI_CHECK(call) \
    do { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            char error_string[MPI_MAX_ERROR_STRING]; \
            int length; \
            MPI_Error_string(err, error_string, &length); \
            fprintf(stderr, "MPI Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, error_string); \
            MPI_Abort(MPI_COMM_WORLD, err); \
        } \
    } while(0)

// Function declarations
void initializeMPI(int* argc, char*** argv, int* rank, int* size);
void finalizeMPI();
void distributeMatrixA(const float* A, float* localA, int totalRows, 
                       int cols, int localRows, int rank, int size);
void distributeMatrixB(const float* B, float* localB, int rows, int cols, 
                       int rank);
void gatherResultMatrix(const float* localC, float* C, int totalRows, 
                        int cols, int localRows, int rank, int size);

#endif // MPI_UTILS_H
