#include "nn/analysis.h"
#include "test.h"

#include <cstring>
#include <memory>
#include <vector>

TEST(sparsity_float) {
    nn::ModelIR m;
    nn::Graph g;
    nn::Tensor t;
    t.id = 0;
    t.name = "w";
    t.constant = true;
    t.dtype = nn::DataType::Float32;
    t.shape = nn::shape_from_ints({4});
    auto data = std::make_shared<std::vector<uint8_t>>(16);
    float vals[4] = {0.f, 0.f, 1.f, 2.f};
    std::memcpy(data->data(), vals, 16);
    nn::TensorDataReference ref;
    ref.owned = data;
    ref.length = 16;
    t.data = ref;
    g.tensors.push_back(t);
    m.graphs.push_back(g);
    auto r = nn::analyze_sparsity(m, {});
    CHECK(r.tensors_computed == 1);
    CHECK(r.total_zeros == 2);
}
