#include <iostream>
#include <vector>
#include <iomanip>
#include <stdexcept>

// ==========================================
// 1. Generic Linear Algebra Functions
//    (Context Agnostic)
// ==========================================
namespace LinAlg {

    // Performs Vector-Matrix Multiplication: result = vec * mat
    // vec: 1 x R (Row vector)
    // mat: R x C (Matrix)
    // result: 1 x C (Row vector)
    std::vector<int> multiplyVectorMatrix(const std::vector<int>& vec, const std::vector<std::vector<int>>& mat) {
        if (mat.empty()) return {};
        size_t R = vec.size();
        size_t C = mat[0].size();
        
        if (mat.size() != R) {
            throw std::invalid_argument("Dimension mismatch: Vector size must match Matrix rows.");
        }

        std::vector<int> result(C, 0);

        for (size_t c = 0; c < C; ++c) {        // Iterate Columns
            for (size_t r = 0; r < R; ++r) {    // Iterate Rows
                // Standard dot product logic
                if (vec[r] != 0) { // Optimization for sparse vectors
                    result[c] += vec[r] * mat[r][c];
                }
            }
        }
        return result;
    }

    // Performs Element-wise Multiplication (Hadamard Product): result = a (*) b
    // Used for applying the Status Vector (St) mask
    std::vector<int> hadamardProduct(const std::vector<int>& a, const std::vector<int>& b) {
        if (a.size() != b.size()) throw std::invalid_argument("Vector sizes must match for Hadamard product.");
        
        std::vector<int> result = a;
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] *= b[i];
        }
        return result;
    }

    // Vector Addition: result = a + b
    std::vector<int> addVectors(const std::vector<int>& a, const std::vector<int>& b) {
        if (a.size() != b.size()) throw std::invalid_argument("Vector sizes must match for addition.");

        std::vector<int> result = a;
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] += b[i];
        }
        return result;
    }
}

// ==========================================
// 2. SN P System Simulation Engine
//    (Context Specific)
// ==========================================

struct Rule {
    int id;
    int neuronIdx;
    int consumed;  // Spikes consumed
    int produced;  // Spikes produced
    int delay;     // Delay ticks
    int minSpikes; // Firing condition (E)
    std::string label; // Debug label
};

class SNPSystem {
private:
    int m_neurons;
    int n_rules;

    // --- The Vectors (as defined in Definitions 2-10) ---
    std::vector<int> C;   // Configuration (Spikes) [cite: 96]
    std::vector<int> St;  // Status (1=Open, 0=Closed) [cite: 110]
    std::vector<int> Iv;  // Indicator (Firing now) [cite: 117]
    std::vector<int> Dv;  // Decision (Selected now) [cite: 133]
    std::vector<int> DIv; // Delayed Indicator (Waiting) [cite: 140]
    std::vector<int> DSv; // Delay Status (Ticks remaining) [cite: 152]

    // --- The Matrix ---
    std::vector<std::vector<int>> adj; // Adjacency List (Graph connections)
    std::vector<Rule> ruleList;
    std::vector<std::vector<int>> neuronToRules;
    std::vector<std::vector<int>> M;   // Spiking Transition Matrix

public:
    std::vector<int> STv; // Spike Train Vector (External Input) [cite: 176]

    SNPSystem(int neurons) : m_neurons(neurons), n_rules(0) {
        // Initialize Vectors
        C.resize(m_neurons, 0);
        St.resize(m_neurons, 1);
        STv.resize(m_neurons, 0);
        
        adj.resize(m_neurons);
        neuronToRules.resize(m_neurons);
    }

    void addSynapse(int from, int to) { adj[from].push_back(to); }
    void setSpikes(int n, int count) { if(n < m_neurons) C[n] = count; }
    int getSpikes(int n) { return (n < m_neurons) ? C[n] : 0; }

    // Add a firing rule
    void addRule(int neuronIdx, int consumed, int produced, int delay, int minSpikes, std::string label) {
        Rule r = {n_rules, neuronIdx, consumed, produced, delay, minSpikes, label};
        ruleList.push_back(r);
        neuronToRules[neuronIdx].push_back(n_rules);
        n_rules++;

        // Expand rule-dependent vectors
        Iv.push_back(0);
        Dv.push_back(0);
        DIv.push_back(0);
        DSv.push_back(0);
    }
    
    // Construct Matrix M based on Definition 3 [cite: 106]
    void buildMatrix() {
        M.assign(n_rules, std::vector<int>(m_neurons, 0));
        
        for (const auto& r : ruleList) {
            // 1. Consumption: Rule i removes spikes from its own Neuron j
            M[r.id][r.neuronIdx] -= r.consumed; 
            
            // 2. Production: Rule i adds spikes to ALL connected neighbors
            if (r.produced > 0) {
                for (int neighbor : adj[r.neuronIdx]) {
                    M[r.id][neighbor] += r.produced;
                }
            }
        }
    }

    // The Main Simulation Step
    void step(int tick) {
        std::cout << "\n--- Calculation for Tick " << tick << " ---\n";

        // 0. Update Status Vector (St)
        // A neuron is closed (0) if any of its rules are currently delayed (DSv > 0)
        // [cite: 114, 288]
        std::fill(St.begin(), St.end(), 1);
        for(int i=0; i<m_neurons; ++i) {
            for(int rId : neuronToRules[i]) {
                if(DSv[rId] > 0) {
                    St[i] = 0;
                    break;
                }
            }
        }

        // 1. Calculate Decision Vector (Dv) [cite: 130]
        // Which rules CAN fire and are CHOSEN?
        std::fill(Dv.begin(), Dv.end(), 0);
        for (int i = 0; i < m_neurons; ++i) {
            if (St[i] == 0) continue; // Closed neurons cannot decide to fire

            // Logic: Find applicable rules (C >= E)
            for (int rId : neuronToRules[i]) {
                if (C[i] >= ruleList[rId].minSpikes) {
                    Dv[rId] = 1; 
                    break; // Deterministic: Pick first valid rule (Simplification)
                }
            }
        }

        // 2. Compute Indicator Vector (Iv)
        // Algorithm 1[cite: 770]: Iv = (Dv OR DIv) AND (NOT DSv)
        // Rules that are either chosen now or were waiting, AND have 0 delay left.
        for(int i=0; i<n_rules; ++i) {
            bool active = (Dv[i] == 1) || (DIv[i] == 1);
            bool ready = (DSv[i] == 0);
            Iv[i] = (active && ready) ? 1 : 0;
        }

        // 3. Compute Delayed Indicator Vector (DIv)
        // Algorithm 2 [cite: 783]
        for(int i=0; i<n_rules; ++i) {
            // Keep waiting if in DIv but not firing yet
            bool keepWaiting = (DIv[i] == 1 && Iv[i] == 0);
            // Start waiting if chosen and has delay
            bool startWaiting = (Dv[i] == 1 && ruleList[i].delay > 0);
            
            DIv[i] = (keepWaiting || startWaiting) ? 1 : 0;
        }

        // 4. Calculate Next Configuration (C)
        // Algorithm 3 / Definition 10 
        // Formula: C_new = C + St * (Iv * M + STv)
        
        // A. Generic Matrix Multiplication: NetGain = Iv * M
        std::vector<int> netGainFromRules = LinAlg::multiplyVectorMatrix(Iv, M);

        // B. Add Input Spike Train: TotalGain = NetGain + STv
        std::vector<int> totalGain = LinAlg::addVectors(netGainFromRules, STv);

        // C. Apply Status Mask (Hadamard): FilteredGain = St * TotalGain
        // "Closed nodes... spikes are dropped and lost" [cite: 287]
        std::vector<int> filteredGain = LinAlg::hadamardProduct(St, totalGain);

        // D. Final Update: C = C + FilteredGain
        C = LinAlg::addVectors(C, filteredGain);


        // 5. Update Delay Status Vector (DSv)
        // Algorithm 4 [cite: 824]
        for(int i=0; i<n_rules; ++i) {
            if (DSv[i] > 0) DSv[i]--; // Decrement timer

            // If rule chosen and has delay, set timer
            if (Dv[i] == 1 && ruleList[i].delay > 0) {
                DSv[i] = ruleList[i].delay; 
                
                // Lock sibling rules (Neuron becomes closed)
                int nIdx = ruleList[i].neuronIdx;
                for(int sib : neuronToRules[nIdx]) {
                    DSv[sib] = ruleList[i].delay;
                }
            }
        }

        // --- Debug Output ---
        std::cout << "Tick " << std::setw(2) << tick << " | ";
        std::cout << "In(STv): " << STv[0] << " | ";
        std::cout << "Spikes: ";
        std::cout << "one: " << C[0] << "  ";
        std::cout << "both: " << C[1] << "  ";
        std::cout << "MAX: " << C[2] << "  ";
        std::cout << "MIN: " << C[3] << "\n";
    }
};

// =========================================================
// 3. Dynamic Sorting Builder
// =========================================================
int main() {
    // --- User Configuration ---
    std::vector<int> inputNumbers = { 
                                    530, 225, 726, 148, 61, 137, 659, 153, 743, 131, 171, 126, 834, 359, 266, 50, 639, 857, 647, 492, 43, 879, 559, 916, 812, 725, 19, 177, 98, 280, 409, 458, 583, 275, 890, 747, 112, 775, 63, 384, 376, 240, 109, 514, 915, 572, 435, 680, 182, 638, 935, 291, 955, 520, 426, 389, 622, 566, 592, 419, 482, 40, 100, 826, 770, 599, 345, 274, 620, 786, 54, 709, 848, 443, 16, 204, 489, 253, 396, 14, 538, 544, 994, 468, 328, 721, 288, 586, 439, 390, 909, 626, 121, 296, 625, 329, 907, 875, 367, 684
                                 }; // Change this array to sort different sets
    int N = inputNumbers.size();
    
    // Calculate total neurons needed
    // Inputs (N) + Sorters (N) + Outputs (N)
    int totalNeurons = 3 * N;
    SNPSystem sys(totalNeurons);

    // ID Ranges
    int startInput = 0;
    int startSorter = N;
    int startOutput = 2 * N;

    // --- 1. Construct Topology ---
    
    // A. Connect Inputs to ALL Sorters
    for(int i = 0; i < N; ++i) {       // Input i
        for(int s = 0; s < N; ++s) {   // Sorter s
            sys.addSynapse(startInput + i, startSorter + s);
        }
    }

    // B. Connect Sorters to Outputs
    // Sorter k (detects k spikes) connects to outputs [k...N] (Median/Max logic)
    // IMPORTANT: The paper's logic is inverted in indexing vs standard array.
    // If we have 4 numbers, Sorter 4 (detects 4 spikes) is the "Min" detector.
    // It should connect to Output 1 (Min), Output 2, Output 3, Output 4.
    // Sorter 1 (detects 1 spike) is the "Max" detector. It connects ONLY to Output 4 (Max).
    
    // We map: Sorter s (0..N-1) corresponds to detecting (N-s) spikes.
    // Wait, let's stick to the paper's diagram logic exactly:
    // s1 (detects N spikes) -> connects to o1...oN
    // sN (detects 1 spike)  -> connects to oN
    
    for(int s = 0; s < N; ++s) {
        int sorterID = startSorter + s;
        // This sorter detects (N - s) spikes.
        // Let's call index 0 "s1" (detects N), index N-1 "sN" (detects 1)
        
        // s1 connects to o1...oN
        // s2 connects to o2...oN
        for(int o = s; o < N; ++o) {
            sys.addSynapse(sorterID, startOutput + o);
        }
    }

    // --- 2. Create Rules ---

    // A. Input Neurons: "a*/a -> a;0"
    for(int i = 0; i < N; ++i) {
        sys.addRule(startInput + i, 1, 1, 0, 1, "a*->a");
        sys.setSpikes(startInput + i, inputNumbers[i]); // Load Data
    }

    // B. Sorter Neurons
    // Sorter s (at index s) corresponds to checking for `target` spikes.
    // target = N - s. (e.g., if N=4, s=0 checks for 4, s=3 checks for 1)
    for(int s = 0; s < N; ++s) {
        int sorterID = startSorter + s;
        int target = N - s; 

        // 1. FORGET rules for spikes > target
        // We need rules for target+1 up to N (max possible spikes received)
        for(int k = N; k > target; --k) {
            sys.addRule(sorterID, k, 0, 0, k, "forget_high");
        }

        // 2. FIRE rule for exactly `target` spikes
        // a^k -> a;0
        sys.addRule(sorterID, target, 1, 0, target, "fire_exact");

        // 3. FORGET rules for spikes < target
        // a^(target-1) -> lambda ... a -> lambda
        // Just adding one rule "a -> lambda" with priority lower than above is sufficient
        // IF we add them in correct order. But our engine is greedy.
        // We must add explicit rules for k = target-1 down to 1.
        for(int k = target - 1; k >= 1; --k) {
            sys.addRule(sorterID, k, 0, 0, k, "forget_low");
        }
    }

    sys.buildMatrix();

    // --- 3. Run Simulation ---
    std::cout << "--- Sorting Array: { ";
    for(int n : inputNumbers) std::cout << n << " ";
    std::cout << "} ---\n";
    
    int maxVal = 0;
    for(int n : inputNumbers) if(n > maxVal) maxVal = n;
    int ticksToRun = maxVal + 2; // Need enough ticks to drain inputs

    for(int t = 0; t < ticksToRun; ++t) {
        sys.step(t);
    }

    std::cout << "\nFinal Sorted Output:\n";
    for(int o = 0; o < N; ++o) {
        std::cout << "Position " << (o+1) << ": " << sys.getSpikes(startOutput + o) << "\n";
    }

    return 0;
}