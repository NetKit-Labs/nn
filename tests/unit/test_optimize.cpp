#include "formats/onnx/writer.h"
#include "nn/format.h"
#include "nn/optimize.h"
#include "test.h"

#include <filesystem>

TEST(optimize_identity_rewire) {
    const auto path = std::filesystem::temp_directory_path() / "nn_ident.onnx";
    nn::SimpleOnnxSpec spec;
    spec.op_type = "Identity";
    spec.inputs = 1;
    spec.input_shape = {1, 4};
    CHECK(nn::write_simple_onnx(path, spec));
    auto model = nn::load_model(path);
    CHECK(model);
    auto report = nn::optimize_model(std::move(model.value()));
    CHECK(!report.changes.empty());
    const nn::Graph* g = nn::primary_graph(report.model);
    CHECK(g);
    CHECK(g->nodes.empty());
}

TEST(convert_onnx_rewrite) {
    const auto in = std::filesystem::temp_directory_path() / "nn_cvt_in.onnx";
    const auto out = std::filesystem::temp_directory_path() / "nn_cvt_out.onnx";
    CHECK(nn::write_simple_onnx(in, {}));
    auto model = nn::load_model(in);
    CHECK(model);
    CHECK(nn::write_onnx_model(out, model.value()));
    auto round = nn::load_model(out);
    CHECK(round);
    CHECK(round.value().source_format == "onnx");
}
