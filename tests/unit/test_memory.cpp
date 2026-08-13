#include "nn/analysis.h"
#include "test.h"

TEST(memory_plan) {
    nn::ModelIR m;
    nn::Graph g;
    nn::Tensor a;
    a.id = 0;
    a.name = "a";
    a.storage_bytes = 64;
    a.dtype = nn::DataType::Float32;
    a.shape = nn::shape_from_ints({16});
    nn::Tensor b;
    b.id = 1;
    b.name = "b";
    b.storage_bytes = 64;
    b.dtype = nn::DataType::Float32;
    b.shape = nn::shape_from_ints({16});
    g.tensors = {a, b};
    nn::Node n0;
    n0.id = 0;
    n0.outputs = {0};
    nn::Node n1;
    n1.id = 1;
    n1.inputs = {0};
    n1.outputs = {1};
    g.nodes = {n0, n1};
    m.graphs.push_back(g);
    nn::MemoryOptions opt;
    opt.plan = true;
    auto r = nn::analyze_memory(m, opt);
    CHECK(r.peak_known);
}
