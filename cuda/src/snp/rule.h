#ifndef RULE_H
#define RULE_H

#include <string>

struct Rule {
    int id;
    int neuronIdx;
    int consumed;  // Spikes consumed
    int produced;  // Spikes produced
    int delay;     // Delay ticks
    int minSpikes; // Firing condition (E)
    std::string label; // Debug label
};

#endif // RULE_H
