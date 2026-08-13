#include "util/overflow.h"

namespace nn {

Result<uint64_t> checked_mul_u64(uint64_t a, uint64_t b) {
    uint64_t out = 0;
    if (mul_overflow_u64(a, b, &out)) {
        return error(ErrorCode::Overflow, "unsigned multiplication overflow");
    }
    return out;
}

Result<uint64_t> checked_add_u64(uint64_t a, uint64_t b) {
    uint64_t out = 0;
    if (add_overflow_u64(a, b, &out)) {
        return error(ErrorCode::Overflow, "unsigned addition overflow");
    }
    return out;
}

Result<uint64_t> checked_mul_many(const uint64_t* values, std::size_t n) {
    uint64_t acc = 1;
    for (std::size_t i = 0; i < n; ++i) {
        auto r = checked_mul_u64(acc, values[i]);
        if (!r) {
            return r.error();
        }
        acc = r.value();
    }
    return acc;
}

}  // namespace nn
