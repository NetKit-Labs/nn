#include "nn/shape.h"
#include "test.h"

TEST(shape_static) {
    auto s = nn::shape_from_ints({1, 3, 224, 224});
    CHECK(s.rank() == 4);
    CHECK(s.is_static());
    auto n = s.element_count();
    CHECK(n);
    CHECK(n.value() == 1ull * 3 * 224 * 224);
    CHECK(s.to_string() == "1x3x224x224");
}

TEST(shape_dynamic) {
    nn::Shape s;
    nn::Dimension d;
    d.symbol = "N";
    s.dims.push_back(d);
    CHECK(!s.is_static());
    CHECK(!s.element_count());
}
