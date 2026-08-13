#include "nn/target.h"
#include "test.h"

TEST(builtin_targets) {
    auto ts = nn::builtin_targets();
    CHECK(!ts.empty());
    CHECK(nn::find_builtin_target("cortex-m4f") != nullptr);
    CHECK(nn::find_builtin_target("no-such-chip") == nullptr);
}
