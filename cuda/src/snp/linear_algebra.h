#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <vector>

namespace LinAlg {
    // Performs Vector-Matrix Multiplication: result = vec * mat
    // vec: 1 x R (Row vector)
    // mat: R x C (Matrix)
    // result: 1 x C (Row vector)
    std::vector<int> multiplyVectorMatrix(const std::vector<int>& vec, const std::vector<std::vector<int>>& mat);

    // Performs Element-wise Multiplication (Hadamard Product): result = a (*) b
    // Used for applying the Status Vector (St) mask
    std::vector<int> hadamardProduct(const std::vector<int>& a, const std::vector<int>& b);

    // Vector Addition: result = a + b
    std::vector<int> addVectors(const std::vector<int>& a, const std::vector<int>& b);
}

#endif // LINEAR_ALGEBRA_H
