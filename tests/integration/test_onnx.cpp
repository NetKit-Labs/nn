#include "formats/onnx/writer.h"
#include "nn/format.h"
#include "test.h"

#include <filesystem>

TEST(onnx_add_roundtrip) {
    const auto path = std::filesystem::temp_directory_path() / "nn_add.onnx";
    nn::SimpleOnnxSpec spec;
    spec.op_type = "Add";
    spec.inputs = 2;
    spec.input_shape = {1, 4};
    auto st = nn::write_simple_onnx(path, spec);
    CHECK(st);
    auto m = nn::load_model(path);
    CHECK(m);
    CHECK(m.value().source_format == "onnx");
    const nn::Graph* g = nn::primary_graph(m.value());
    CHECK(g);
    CHECK(g->nodes.size() == 1);
    CHECK(g->nodes[0].op_type == "Add");
    CHECK(g->inputs.size() == 2);
    CHECK(g->outputs.size() == 1);
}
