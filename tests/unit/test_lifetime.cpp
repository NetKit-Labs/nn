#include "nn/analysis.h"
#include "test.h"

TEST(lifetime_peak) {
    nn::Graph g;
    nn::Tensor in;
    in.id = 0;
    in.model_input = true;
    in.storage_bytes = 100;
    nn::Tensor mid;
    mid.id = 1;
    mid.storage_bytes = 200;
    nn::Tensor out;
    out.id = 2;
    out.model_output = true;
    out.storage_bytes = 50;
    g.tensors = {in, mid, out};
    g.inputs = {0};
    g.outputs = {2};
    nn::Node n0;
    n0.id = 0;
    n0.inputs = {0};
    n0.outputs = {1};
    nn::Node n1;
    n1.id = 1;
    n1.inputs = {1};
    n1.outputs = {2};
    g.nodes = {n0, n1};
    auto lt = nn::analyze_lifetimes(g);
    CHECK(lt.size() == 3);
}
