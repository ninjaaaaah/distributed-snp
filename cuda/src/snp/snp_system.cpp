#include "snp_system.h"
#include "linear_algebra.h"
#include <iostream>
#include <algorithm>

SNPSystem::SNPSystem(int neurons) : m_neurons(neurons), n_rules(0) {
    // Initialize Vectors
    C.resize(m_neurons, 0);
    St.resize(m_neurons, 1);
    STv.resize(m_neurons, 0);
    
    adj.resize(m_neurons);
    neuronToRules.resize(m_neurons);
}

void SNPSystem::addSynapse(int from, int to) { 
    adj[from].push_back(to); 
}

void SNPSystem::setSpikes(int n, int count) { 
    if(n < m_neurons) C[n] = count; 
}

int SNPSystem::getSpikes(int n) { 
    return (n < m_neurons) ? C[n] : 0; 
}

void SNPSystem::addRule(int neuronIdx, int consumed, int produced, int delay, int minSpikes, std::string label) {
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

void SNPSystem::buildMatrix() {
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

void SNPSystem::step() {
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
