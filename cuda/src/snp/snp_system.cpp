#include "snp_system.h"
#include "../linear_algebra/linear_algebra.h"
#ifdef USE_GPU
#include "../linear_algebra/gpu/linear_algebra_gpu.h"
#elif defined(USE_DISTRIBUTED)
#include "../linear_algebra/distributed/linear_algebra_distributed.h"
#else
#include "../linear_algebra/cpu/linear_algebra_cpu.h"
#endif
#include <iostream>
#include <algorithm>

SNPSystem::SNPSystem(int neurons) : m_neurons(neurons), n_rules(0) {
#ifdef USE_GPU
    linAlg = new LinAlg::LinearAlgebraGPU();
#elif defined(USE_DISTRIBUTED)
    linAlg = new LinAlg::LinearAlgebraDistributed();
#else
    linAlg = new LinAlg::LinearAlgebraCPU();
#endif
    // Initialize Vectors
    C.resize(m_neurons, 0);
    St.resize(m_neurons, 1);
    STv.resize(m_neurons, 0);
    
    adj.resize(m_neurons);
    neuronToRules.resize(m_neurons);
}

SNPSystem::~SNPSystem() {
    delete linAlg;
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
    // M is n_rules x m_neurons
    M.assign(n_rules * m_neurons, 0);
    
    for (const auto& r : ruleList) {
        // 1. Consumption: Rule i removes spikes from its own Neuron j
        // M[r.id][r.neuronIdx] -= r.consumed; 
        M[r.id * m_neurons + r.neuronIdx] -= r.consumed;
        
        // 2. Production: Rule i adds spikes to ALL connected neighbors
        if (r.produced > 0) {
            for (int neighbor : adj[r.neuronIdx]) {
                // M[r.id][neighbor] += r.produced;
                M[r.id * m_neurons + neighbor] += r.produced;
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

    // std::vector<int> netGain = LinAlg::multiplyVectorMatrix(Iv, M);
    // std::vector<int> totalInflow = LinAlg::addVectors(netGain, STv);
    // std::vector<int> effectiveGain = LinAlg::hadamardProduct(St, totalInflow);
    // C = LinAlg::addVectors(C, effectiveGain);

    std::vector<int> netGain(m_neurons);
    // Iv is 1 x n_rules
    // M is n_rules x m_neurons
    // netGain is 1 x m_neurons
    linAlg->multiplyVectorMatrix(Iv.data(), M.data(), netGain.data(), n_rules, m_neurons);

    std::vector<int> totalInflow(m_neurons);
    linAlg->addVectors(netGain.data(), STv.data(), totalInflow.data(), m_neurons);

    std::vector<int> effectiveGain(m_neurons);
    linAlg->hadamardProduct(St.data(), totalInflow.data(), effectiveGain.data(), m_neurons);

    std::vector<int> newC(m_neurons);
    linAlg->addVectors(C.data(), effectiveGain.data(), newC.data(), m_neurons);
    C = newC;

    for(int i=0; i<n_rules; ++i) {
        if (DSv[i] > 0) DSv[i]--;
        if (Dv[i] == 1 && ruleList[i].delay > 0) {
            DSv[i] = ruleList[i].delay;
            int nIdx = ruleList[i].neuronIdx;
            for(int sib : neuronToRules[nIdx]) DSv[sib] = ruleList[i].delay;
        }
    }
}
