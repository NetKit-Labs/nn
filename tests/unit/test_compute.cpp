#include "nn/analysis.h"
#include "test.h"

TEST(compute_matmul) {
    nn::Graph g;
    nn::Tensor a;
    a.id = 0;
    a.dtype = nn::DataType::Float32;
    a.shape = nn::shape_from_ints({4, 8});
    nn::Tensor b;
    b.id = 1;
    b.dtype = nn::DataType::Float32;
    b.shape = nn::shape_from_ints({8, 16});
    nn::Tensor c;
    c.id = 2;
    c.dtype = nn::DataType::Float32;
    c.shape = nn::shape_from_ints({4, 16});
    g.tensors = {a, b, c};
    nn::Node n;
    n.id = 0;
    n.op_type = "MatMul";
    n.canonical = nn::CanonicalOp::MatMul;
    n.inputs = {0, 1};
    n.outputs = {2};
    g.nodes.push_back(n);
    auto cost = nn::estimate_node_compute(g, n);
    CHECK(cost.macs.known);
    CHECK(cost.macs.value == 4ull * 16 * 8);
}
