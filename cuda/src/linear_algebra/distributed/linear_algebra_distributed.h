#ifndef LINEAR_ALGEBRA_DISTRIBUTED_H
#define LINEAR_ALGEBRA_DISTRIBUTED_H

#include "../cpu/linear_algebra_cpu.h"

namespace LinAlg {
    // Distributed matrix multiplication implementation using MPI
    // Inherits from LinearAlgebraCPU to reuse CPU implementations as fallback or base
    class LinearAlgebraDistributed : public LinearAlgebraCPU {
    public:
        LinearAlgebraDistributed() = default;
        virtual ~LinearAlgebraDistributed() = default;
        
        // We can override multiply here if we want to implement distributed logic later
        // For now, it will use the CPU implementation from the base class
    };
}

#endif // LINEAR_ALGEBRA_DISTRIBUTED_H
