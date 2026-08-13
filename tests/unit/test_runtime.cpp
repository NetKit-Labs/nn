#include "formats/onnx/writer.h"
#include "nn/format.h"
#include "runtime/tensor_io.h"
#include "test.h"
#include "util/npy.h"

#include <cstring>
#include <filesystem>
#include <map>
#include <vector>

namespace {

nn::RuntimeTensor f32_tensor(std::string name, std::vector<int64_t> shape, const std::vector<float>& v) {
    nn::RuntimeTensor t;
    t.name = std::move(name);
    t.dtype = nn::DataType::Float32;
    t.shape = nn::shape_from_ints(shape);
    t.bytes.resize(v.size() * sizeof(float));
    std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
    return t;
}

}  // namespace

TEST(reference_add) {
    const auto model_path = std::filesystem::temp_directory_path() / "nn_ref_add.onnx";
    nn::SimpleOnnxSpec spec;
    spec.op_type = "Add";
    spec.inputs = 2;
    spec.input_shape = {1, 4};
    CHECK(nn::write_simple_onnx(model_path, spec));
    auto model = nn::load_model(model_path);
    CHECK(model);
    std::map<std::string, nn::RuntimeTensor> ins;
    ins["input0"] = f32_tensor("input0", {1, 4}, {1, 2, 3, 4});
    ins["input1"] = f32_tensor("input1", {1, 4}, {10, 20, 30, 40});
    auto rr = nn::eval_model(model.value(), ins, {}, "reference");
    CHECK(rr);
    CHECK(rr.value().outputs.size() == 1);
    auto got = nn::as_f64(rr.value().outputs.front());
    CHECK(got.size() == 4);
    CHECK(got[0] == 11);
    CHECK(got[3] == 44);
}

TEST(reference_matmul_weight) {
    const auto model_path = std::filesystem::temp_directory_path() / "nn_ref_mm.onnx";
    nn::SimpleOnnxSpec spec;
    spec.op_type = "MatMul";
    spec.inputs = 1;
    spec.with_weight = true;
    spec.input_shape = {1, 4};
    spec.weight_shape = {4, 4};
    spec.output_shape = {1, 4};
    spec.weight = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    CHECK(nn::write_simple_onnx(model_path, spec));
    auto model = nn::load_model(model_path);
    CHECK(model);
    std::map<std::string, nn::RuntimeTensor> ins;
    ins["input"] = f32_tensor("input", {1, 4}, {1, 2, 3, 4});
    auto rr = nn::eval_model(model.value(), ins, {}, "reference");
    CHECK(rr);
    auto got = nn::as_f64(rr.value().outputs.front());
    CHECK(got.size() == 4);
    CHECK(got[0] == 1);
    CHECK(got[1] == 2);
    CHECK(got[2] == 3);
    CHECK(got[3] == 4);
}

TEST(numeric_compare_detects_diff) {
    auto a = f32_tensor("a", {2}, {1, 2});
    auto b = f32_tensor("b", {2}, {1, 3});
    auto c = nn::compare_numeric(a, b, 1e-5, 1e-5);
    CHECK(c.comparable);
    CHECK(c.changed == 1);
}

TEST(npy_bind_roundtrip) {
    auto t = f32_tensor("x", {2}, {5, 6});
    const auto path = std::filesystem::temp_directory_path() / "nn_bind.npy";
    CHECK(nn::save_npy(path, t));
    auto loaded = nn::load_tensor_file(path);
    CHECK(loaded);
    CHECK(nn::as_f64(loaded.value())[1] == 6);
}

TEST(npz_bind_two_inputs) {
    const auto dir = std::filesystem::temp_directory_path();
    const auto model = dir / "nn_npz_bind.onnx";
    const auto npz = dir / "nn_npz_bind.npz";
    nn::SimpleOnnxSpec spec;
    spec.op_type = "Add";
    spec.inputs = 2;
    spec.input_shape = {1, 2};
    CHECK(nn::write_simple_onnx(model, spec));
    auto a = f32_tensor("input0", {1, 2}, {1, 2});
    auto b = f32_tensor("input1", {1, 2}, {3, 4});
    CHECK(nn::save_npz(npz, {{"input0", a}, {"input1", b}}));
    auto m = nn::load_model(model);
    CHECK(m);
    auto ins = nn::bind_model_inputs(m.value(), {npz.string()}, 0, false);
    CHECK(ins);
    CHECK(ins.value().count("input0") == 1);
    CHECK(ins.value().count("input1") == 1);
    auto rr = nn::eval_model(m.value(), ins.value(), {}, "reference");
    CHECK(rr);
    auto got = nn::as_f64(rr.value().outputs.front());
    CHECK(got.size() == 2);
    CHECK(got[0] == 4);
    CHECK(got[1] == 6);
}

TEST(compare_activations_same_add) {
    const auto path = std::filesystem::temp_directory_path() / "nn_act_add.onnx";
    nn::SimpleOnnxSpec spec;
    spec.op_type = "Add";
    spec.inputs = 2;
    spec.input_shape = {1, 4};
    CHECK(nn::write_simple_onnx(path, spec));
    auto m = nn::load_model(path);
    CHECK(m);
    std::map<std::string, nn::RuntimeTensor> ins;
    ins["input0"] = f32_tensor("input0", {1, 4}, {1, 2, 3, 4});
    ins["input1"] = f32_tensor("input1", {1, 4}, {1, 1, 1, 1});
    nn::RuntimeOptions opt;
    opt.dump_all = true;
    auto ra = nn::eval_model(m.value(), ins, opt, "reference");
    auto rb = nn::eval_model(m.value(), ins, opt, "reference");
    CHECK(ra && rb);
    CHECK(!ra.value().dumps.empty());
    auto pairs = nn::align_activation_dumps(m.value(), ra.value(), m.value(), rb.value());
    CHECK(!pairs.empty());
    for (const auto& p : pairs) {
        CHECK(p.a && p.b);
        auto c = nn::compare_numeric(*p.a, *p.b, 1e-5, 1e-5);
        CHECK(c.comparable);
        CHECK(c.changed == 0);
    }
}
