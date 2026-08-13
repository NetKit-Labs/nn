#include "test.h"
#include "util/overflow.h"

TEST(overflow_mul) {
    auto ok = nn::checked_mul_u64(3, 4);
    CHECK(ok);
    CHECK(ok.value() == 12);
    auto bad = nn::checked_mul_u64(UINT64_C(1) << 32, UINT64_C(1) << 33);
    CHECK(!bad);
}
