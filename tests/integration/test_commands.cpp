#include "formats/onnx/writer.h"
#include "nn/diff.h"
#include "nn/format.h"
#include "test.h"

TEST(diff_copies) {
    const auto a = std::filesystem::temp_directory_path() / "nn_a.onnx";
    const auto b = std::filesystem::temp_directory_path() / "nn_b.onnx";
    CHECK(nn::write_simple_onnx(a, {}));
    CHECK(nn::write_simple_onnx(b, {}));
    auto ma = nn::load_model(a);
    auto mb = nn::load_model(b);
    CHECK(ma && mb);
    auto d = nn::diff_models(ma.value(), mb.value());
    CHECK(d);
    CHECK(d.value().identical);
}
