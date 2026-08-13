#ifndef NN_OVERFLOW_H
#define NN_OVERFLOW_H

#include "nn/result.h"

#include <cstdint>
#include <limits>

namespace nn {

inline bool mul_overflow_u64(uint64_t a, uint64_t b, uint64_t* out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return true;
    }
    *out = a * b;
    return false;
}

inline bool add_overflow_u64(uint64_t a, uint64_t b, uint64_t* out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return true;
    }
    *out = a + b;
    return false;
}

Result<uint64_t> checked_mul_u64(uint64_t a, uint64_t b);
Result<uint64_t> checked_add_u64(uint64_t a, uint64_t b);
Result<uint64_t> checked_mul_many(const uint64_t* values, std::size_t n);

}  // namespace nn

#endif
