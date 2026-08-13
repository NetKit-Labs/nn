#include "runtime/tensor_io.h"
#include "test.h"

#include <cstring>
#include <vector>

namespace {

nn::RuntimeTensor f32(std::string name, std::vector<int64_t> shape, const std::vector<float>& v) {
    nn::RuntimeTensor t;
    t.name = std::move(name);
    t.dtype = nn::DataType::Float32;
    t.shape = nn::shape_from_ints(shape);
    t.bytes.resize(v.size() * sizeof(float));
    std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
    return t;
}

}  // namespace

TEST(normalize_tensor_name_strips_tflite_noise) {
    CHECK(nn::normalize_tensor_name("serving_default_input:0") == "input");
    CHECK(nn::normalize_tensor_name("Input3") == "input3");
    CHECK(nn::normalize_tensor_name("foo/bar:0") == "bar");
}

TEST(align_outputs_by_normalized_name) {
    std::vector<nn::RuntimeTensor> a;
    a.push_back(f32("serving_default_x:0", {2}, {1, 2}));
    std::vector<nn::RuntimeTensor> b;
    b.push_back(f32("x", {2}, {1, 2}));
    auto pairs = nn::align_runtime_tensors(a, b);
    CHECK(pairs.size() == 1);
    CHECK(pairs[0].a != nullptr);
    CHECK(pairs[0].b != nullptr);
}

TEST(align_outputs_by_index_when_names_differ) {
    std::vector<nn::RuntimeTensor> a;
    a.push_back(f32("Plus214_Output_0", {2}, {1, 2}));
    std::vector<nn::RuntimeTensor> b;
    b.push_back(f32("Identity", {2}, {1, 2}));
    auto pairs = nn::align_runtime_tensors(a, b);
    CHECK(pairs.size() == 1);
    CHECK(pairs[0].a != nullptr && pairs[0].b != nullptr);
}

TEST(nchw_nhwc_numeric_align) {
    const int64_t N = 1, C = 2, H = 3, W = 4;
    std::vector<float> nchw(static_cast<std::size_t>(N * C * H * W));
    std::vector<float> nhwc(nchw.size());
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                const float v = static_cast<float>(c * 100 + h * 10 + w);
                nchw[static_cast<std::size_t>(((0 * C + c) * H + h) * W + w)] = v;
                nhwc[static_cast<std::size_t>(((0 * H + h) * W + w) * C + c)] = v;
            }
        }
    }
    auto a = f32("a", {N, C, H, W}, nchw);
    auto b = f32("b", {N, H, W, C}, nhwc);
    auto raw = nn::compare_numeric(a, b, 1e-5, 1e-5);
    CHECK(raw.comparable);
    CHECK(raw.changed > 0);
    auto aligned = nn::compare_numeric_aligned(a, b, 1e-5, 1e-5);
    CHECK(aligned.comparable);
    CHECK(aligned.changed == 0);
}
