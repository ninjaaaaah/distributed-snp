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
        
        std::cout << C.size() << std::endl; 
        std::cout << St.size() << std::endl; 
        std::cout << Iv.size() << std::endl; 
        std::cout << Dv.size() << std::endl; 
        std::cout << DIv.size() << std::endl; 
        std::cout << DSv.size() << std::endl; 
        std::cout << neuronToRules.size() << std::endl;
        std::cout << ruleList.size() << std::endl;
        std::cout << M.size() << std::endl;
    }

    void step() {
        std::fill(St.begin(), St.end(), 1);
        for(int i=0; i<m_neurons; ++i) {
            for(int rId : neuronToRules[i]) if(DSv[rId] > 0) { St[i] = 0; break; }
        }

        std::fill(Dv.begin(), Dv.end(), 0);
        for (int i = 0; i < m_neurons; ++i) {
            if (St[i] == 0) continue;
            for (int rId : neuronToRules[i]) {
                if (C[i] >= ruleList[rId].minSpikes) {
                    Dv[rId] = 1; break; 
                }
            }
        }

        for(int i=0; i<n_rules; ++i) {
            bool active = (Dv[i] == 1) || (DIv[i] == 1);
            Iv[i] = (active && (DSv[i] == 0)) ? 1 : 0;
            bool keepWaiting = (DIv[i] == 1 && Iv[i] == 0);
            bool startWaiting = (Dv[i] == 1 && ruleList[i].delay > 0);
            DIv[i] = (keepWaiting || startWaiting) ? 1 : 0;
        }

        std::vector<int> netGain = LinAlg::multiplyVectorMatrix(Iv, M);
        std::vector<int> totalInflow = LinAlg::addVectors(netGain, STv);
        std::vector<int> effectiveGain = LinAlg::hadamardProduct(St, totalInflow);
        C = LinAlg::addVectors(C, effectiveGain);

        for(int i=0; i<n_rules; ++i) {
            if (DSv[i] > 0) DSv[i]--;
            if (Dv[i] == 1 && ruleList[i].delay > 0) {
                DSv[i] = ruleList[i].delay;
                int nIdx = ruleList[i].neuronIdx;
                for(int sib : neuronToRules[nIdx]) DSv[sib] = ruleList[i].delay;
            }
        }
    }
};

// =========================================================
// 3. Sorting Abstraction
// =========================================================

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

// =========================================================
// 4. Main
// =========================================================
int main() {
    // Test Case 1
    std::vector<int> sorted1 = runSort({ 445, 179, 305, 544, 240, 397, 372, 848, 710, 497, 707, 288, 682, 470, 247, 690, 206, 64, 956, 741 });

    return 0;
}