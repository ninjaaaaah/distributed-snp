#include "linear_algebra.h"
#include <stdexcept>

namespace LinAlg {
    
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

    std::vector<int> hadamardProduct(const std::vector<int>& a, const std::vector<int>& b) {
        if (a.size() != b.size()) throw std::invalid_argument("Vector sizes must match for Hadamard product.");
        
        std::vector<int> result = a;
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] *= b[i];
        }
        return result;
    }

    std::vector<int> addVectors(const std::vector<int>& a, const std::vector<int>& b) {
        if (a.size() != b.size()) throw std::invalid_argument("Vector sizes must match for addition.");

        std::vector<int> result = a;
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] += b[i];
        }
        return result;
    }
}
