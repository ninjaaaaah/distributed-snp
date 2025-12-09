#ifndef SNP_SYSTEM_H
#define SNP_SYSTEM_H

#include <vector>
#include <string>
#include "rule.h"
#include "../linear_algebra/linear_algebra.h"

class SNPSystem {
private:
    LinAlg::LinearAlgebra* linAlg;
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
    std::vector<int> M;   // Spiking Transition Matrix (Flattened)

public:
    std::vector<int> STv; // Spike Train Vector (External Input) [cite: 176]

    SNPSystem(int neurons);
    ~SNPSystem();

    void addSynapse(int from, int to);
    void setSpikes(int n, int count);
    int getSpikes(int n);

    // Add a firing rule
    void addRule(int neuronIdx, int consumed, int produced, int delay, int minSpikes, std::string label);
    
    // Construct Matrix M based on Definition 3 [cite: 106]
    void buildMatrix();

    // Execute one simulation step
    void step();
};

#endif // SNP_SYSTEM_H
