#include "distributed_snp/snp/sorting.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

// Get current time in seconds (with microsecond precision)
double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    int N = 10; // Default size
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("=================================================\n");
    printf("SNP Sorting Benchmark\n");
    printf("=================================================\n");
    printf("Number of elements to sort: %d\n", N);
    printf("=================================================\n\n");

    // Generate random data
    printf("Initializing data...\n");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 50); // Values between 1 and 50

    std::vector<int> input(N);
    for(int i=0; i<N; ++i) {
        input[i] = dis(gen);
    }

    // Print input if small
    if (N <= 20) {
        printf("Input:    ");
        for(int x : input) printf("%d ", x);
        printf("\n");
    }

    printf("Performing sort...\n");
    
#ifdef USE_GPU
    printf("Using GPU implementation\n");
#elif defined(USE_DISTRIBUTED)
    printf("Using Distributed implementation\n");
#else
    printf("Using CPU implementation\n");
#endif

    double start = getTime();
    std::vector<int> result = runSort(input);
    double end = getTime();

    double elapsed = end - start;
    printf("Computation time: %.6f seconds\n", elapsed);

    // Calculate performance metrics
    double elemsPerSec = N / elapsed;
    printf("Performance: %.3f Elements/sec\n", elemsPerSec);

    // Print result if small
    if (N <= 20) {
        printf("Sorted:   ");
        for(int x : result) printf("%d ", x);
        printf("\n");
    }

    // Verify
    std::vector<int> expected = input;
    std::sort(expected.begin(), expected.end());

    bool correct = (result == expected);
    
    printf("\nVerification:\n");
    if (correct) {
        printf("  Match:    YES\n");
    } else {
        printf("  Match:    NO\n");
        if (N > 20) {
             // Print mismatch details if not already printed
             // (If N<=20 we already printed the result above)
             // But let's print the first mismatch
             for(size_t i=0; i<result.size(); ++i) {
                 if (result[i] != expected[i]) {
                     printf("  Mismatch at index %lu: Expected %d, Got %d\n", i, expected[i], result[i]);
                     break;
                 }
             }
        }
    }

    printf("\n=================================================\n");
    printf("Benchmark completed %s!\n", correct ? "successfully" : "with errors");
    printf("=================================================\n");

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
