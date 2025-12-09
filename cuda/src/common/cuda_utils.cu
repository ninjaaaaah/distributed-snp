#include "distributed_snp/common/cuda_utils.h"
#include <stdio.h>

// Initialize CUDA device
void initializeCUDA(int rank) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        exit(EXIT_FAILURE);
    }
    
    int deviceId = 0;
    if (rank >= 0) {
        // Simple round-robin assignment for distributed mode
        deviceId = rank % deviceCount;
    }
    
    // Use selected device
    CUDA_CHECK(cudaSetDevice(deviceId));
    
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    
    if (rank >= 0) {
        printf("[Rank %d] Using CUDA device %d: %s\n", rank, deviceId, prop.name);
    } else {
        printf("Using CUDA device: %s (Compute %d.%d)\n",
               prop.name, prop.major, prop.minor);
    }
}

