#include "nn/analysis.h"
#include "test.h"

TEST(canonical_text) {
    nn::ModelIR m;
    m.source_format = "onnx";
    nn::Graph g;
    g.name = "g";
    nn::Tensor t;
    t.id = 0;
    t.name = "in";
    t.dtype = nn::DataType::Float32;
    t.shape = nn::shape_from_ints({1, 4});
    t.model_input = true;
    g.tensors.push_back(t);
    g.inputs.push_back(0);
    m.graphs.push_back(g);
    auto s = nn::canonicalize_graph_text(m);
    CHECK(s.find("format=onnx") != std::string::npos);
}
