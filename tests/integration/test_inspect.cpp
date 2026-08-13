#include "formats/onnx/writer.h"
#include "nn/analysis.h"
#include "nn/format.h"
#include "test.h"

TEST(inspect_add) {
    const auto path = std::filesystem::temp_directory_path() / "nn_inspect.onnx";
    nn::SimpleOnnxSpec spec;
    CHECK(nn::write_simple_onnx(path, spec));
    auto m = nn::load_model(path);
    CHECK(m);
    auto r = nn::inspect_model(m.value());
    CHECK(r);
    CHECK(r.value().model.source_format == "onnx");
}
