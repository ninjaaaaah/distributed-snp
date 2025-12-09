#include "sorting.h"
#include "snp_system.h"
#include <iostream>
#include <algorithm>

std::vector<int> runSort(const std::vector<int>& inputNumbers) {
    int N = inputNumbers.size();
    if (N == 0) return {};

    // Determine max value to calculate simulation ticks needed
    int maxVal = 0;
    for(int n : inputNumbers) if(n > maxVal) maxVal = n;
    
    // Layout: [ Inputs (0..N-1) | Sorters (N..2N-1) | Outputs (2N..3N-1) ]
    int startInput = 0;
    int startSorter = N;
    int startOutput = 2 * N;
    int totalNeurons = 3 * N;

    SNPSystem sys(totalNeurons);

    // --- A. Build Topology ---
    // Inputs -> All Sorters
    for(int i = 0; i < N; ++i) {
        for(int s = 0; s < N; ++s) {
            sys.addSynapse(startInput + i, startSorter + s);
        }
    }

    // Sorters -> Outputs
    // Sorter s (detects N-s spikes) connects to Outputs [s...N-1]
    for(int s = 0; s < N; ++s) {
        for(int o = s; o < N; ++o) {
            sys.addSynapse(startSorter + s, startOutput + o);
        }
    }

    // --- B. Define Rules ---
    // Input Streams
    for(int i = 0; i < N; ++i) {
        sys.addRule(startInput + i, 1, 1, 0, 1, "stream");
        sys.setSpikes(startInput + i, inputNumbers[i]); 
    }

    // Sorters
    for(int s = 0; s < N; ++s) {
        int sorterID = startSorter + s;
        int target = N - s;

        // 1. Forget High (Greedy priority)
        for(int k = N; k > target; --k) 
            sys.addRule(sorterID, k, 0, 0, k, "forget_high");
        
        // 2. Fire Exact
        sys.addRule(sorterID, target, 1, 0, target, "fire_exact");

        // 3. Forget Low
        for(int k = target - 1; k >= 1; --k)
            sys.addRule(sorterID, k, 0, 0, k, "forget_low");
    }

    sys.buildMatrix();

    // --- C. Execute ---
    int ticks = maxVal + 3; // Buffer for signal propagation
    for(int t = 0; t < ticks; ++t) {
        sys.step();
    }

    // --- D. Collect Results ---
    std::vector<int> result;
    std::cout << "Original Array: [ ";
    for(int n : inputNumbers) std::cout << n << " ";
    std::cout << "]\n";

    std::cout << "Sorted Array:   [ ";
    for(int o = 0; o < N; ++o) {
        int val = sys.getSpikes(startOutput + o);
        result.push_back(val);
        std::cout << val << " ";
    }
    std::cout << "]\n-------------------------------\n";
    
    return result;
}
