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

int main() {
// Mapping IDs:
    // Inputs: i1(0), i2(1), i3(2)
    // Sorters: s1(3), s2(4), s3(5)
    // Outputs: o1(6), o2(7), o3(8)
    SNPSystem sys(9);

    // --- 1. Topology ---
    
    // Inputs connect to ALL Sorters
    for(int i=0; i<=2; ++i) {
        sys.addSynapse(i, 3); // to s1
        sys.addSynapse(i, 4); // to s2
        sys.addSynapse(i, 5); // to s3
    }

    // s1 connects to o1, o2, o3 (accumulates to Min, Med, Max)
    sys.addSynapse(3, 6);
    sys.addSynapse(3, 7);
    sys.addSynapse(3, 8);

    // s2 connects to o2, o3 (accumulates to Med, Max)
    sys.addSynapse(4, 7);
    sys.addSynapse(4, 8);

    // s3 connects to o3 (accumulates to Max)
    sys.addSynapse(5, 8);

    // --- 2. Rules (Priority: Highest spikes first) ---
    
    // Inputs (0,1,2): Rule a*/a -> a;0
    // "If >= 1 spike, consume 1, produce 1"
    for(int i=0; i<=2; ++i) {
        sys.addRule(i, 1, 1, 0, 1, "a*->a");
    }

    // Sorter s1 (3): Detects 3 spikes
    sys.addRule(3, 3, 1, 0, 3, "a^3->a"); // Fire if 3
    sys.addRule(3, 2, 0, 0, 2, "a^2->L"); // Forget if 2
    sys.addRule(3, 1, 0, 0, 1, "a->L");   // Forget if 1

    // Sorter s2 (4): Detects 2 spikes
    sys.addRule(4, 3, 0, 0, 3, "a^3->L"); // Forget if 3 (Too many)
    sys.addRule(4, 2, 1, 0, 2, "a^2->a"); // Fire if 2
    sys.addRule(4, 1, 0, 0, 1, "a->L");   // Forget if 1

    // Sorter s3 (5): Detects 1 spike
    sys.addRule(5, 3, 0, 0, 3, "a^3->L"); // Forget if 3
    sys.addRule(5, 2, 0, 0, 2, "a^2->L"); // Forget if 2
    sys.addRule(5, 1, 1, 0, 1, "a->a");   // Fire if 1

    sys.buildMatrix();

    // --- 3. Initial State (1, 3, 2) ---
    sys.setSpikes(0, 6); // i1
    sys.setSpikes(1, 10); // i2
    sys.setSpikes(2, 2); // i3

    std::cout << "--- Sorting Inputs: 1, 3, 2 ---\n";
    std::cout << "Outputs should settle to: Min(1), Med(2), Max(3)\n\n";

    // Run for 5 ticks (enough to drain i2's 3 spikes)
    for(int t=0; t<100; ++t) {
        std::cout << "Tick " << t << " | ";
        
        // Inputs
        std::cout << "i: " << sys.getSpikes(0) << "," << sys.getSpikes(1) << "," << sys.getSpikes(2) << " | ";
        
        // Sorters (Transient)
        std::cout << "s: " << sys.getSpikes(3) << "," << sys.getSpikes(4) << "," << sys.getSpikes(5) << " | ";
        
        // Outputs (Accumulated)
        std::cout << "OUT: "
                  << "o1(Min)=" << sys.getSpikes(6) << " "
                  << "o2(Med)=" << sys.getSpikes(7) << " "
                  << "o3(Max)=" << sys.getSpikes(8) << "\n";

        sys.step(t);
    }

    return 0;
}