#include "nn/quantization.h"
#include "test.h"

TEST(quant_valid) {
    nn::QuantizationInfo q;
    CHECK(q.valid());
    q.quantized = true;
    CHECK(!q.valid());
    q.scales = {0.1};
    CHECK(q.valid());
    q.per_channel = true;
    CHECK(!q.valid());
    q.axis = 0;
    CHECK(q.valid());
}
