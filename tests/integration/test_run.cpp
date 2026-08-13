#include "formats/onnx/writer.h"
#include "nn/compat.h"
#include "nn/convert.h"
#include "nn/format.h"
#include "nn/optimize.h"
#include "runtime/tensor_io.h"
#include "test.h"
#include "util/npy.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <vector>

TEST(run_add_with_npy) {
    const auto dir = std::filesystem::temp_directory_path();
    const auto model = dir / "nn_run_add.onnx";
    const auto a = dir / "nn_run_a.npy";
    const auto b = dir / "nn_run_b.npy";
    nn::SimpleOnnxSpec spec;
    spec.op_type = "Add";
    spec.inputs = 2;
    spec.input_shape = {1, 4};
    CHECK(nn::write_simple_onnx(model, spec));
    nn::RuntimeTensor t;
    t.dtype = nn::DataType::Float32;
    t.shape = nn::shape_from_ints({1, 4});
    float v[4] = {1, 2, 3, 4};
    t.bytes.resize(sizeof(v));
    std::memcpy(t.bytes.data(), v, sizeof(v));
    CHECK(nn::save_npy(a, t));
    float w[4] = {1, 1, 1, 1};
    std::memcpy(t.bytes.data(), w, sizeof(w));
    CHECK(nn::save_npy(b, t));
    auto m = nn::load_model(model);
    CHECK(m);
    auto ins = nn::bind_model_inputs(m.value(), {"input0=" + a.string(), "input1=" + b.string()}, 0, false);
    CHECK(ins);
    auto rr = nn::eval_model(m.value(), ins.value(), {}, {});
    CHECK(rr);
    auto got = nn::as_f64(rr.value().outputs.front());
    CHECK(got.size() == 4);
    CHECK(got[0] == 2);
}

#if defined(NN_HAS_ONNXRUNTIME)
TEST(onnxruntime_add_matches_reference) {
    const auto dir = std::filesystem::temp_directory_path();
    const auto model = dir / "nn_ort_add.onnx";
    nn::SimpleOnnxSpec spec;
    spec.op_type = "Add";
    spec.inputs = 2;
    spec.input_shape = {1, 4};
    CHECK(nn::write_simple_onnx(model, spec));
    auto m = nn::load_model(model);
    CHECK(m);
    std::map<std::string, nn::RuntimeTensor> ins;
    auto make = [](const char* name, std::initializer_list<float> v) {
        nn::RuntimeTensor t;
        t.name = name;
        t.dtype = nn::DataType::Float32;
        t.shape = nn::shape_from_ints({1, 4});
        t.bytes.resize(v.size() * sizeof(float));
        std::memcpy(t.bytes.data(), std::data(v), t.bytes.size());
        return t;
    };
    ins["input0"] = make("input0", {1, 2, 3, 4});
    ins["input1"] = make("input1", {10, 20, 30, 40});
    auto ort = nn::eval_model(m.value(), ins, {}, "onnxruntime");
    auto ref = nn::eval_model(m.value(), ins, {}, "reference");
    CHECK(ort);
    CHECK(ref);
    auto c = nn::compare_numeric(ort.value().outputs.front(), ref.value().outputs.front(), 1e-5, 1e-5);
    CHECK(c.comparable);
    CHECK(c.changed == 0);
}

#ifdef NN_TEST_MNIST_ONNX
TEST(onnxruntime_mnist_zeros) {
    auto m = nn::load_model(NN_TEST_MNIST_ONNX);
    CHECK(m);
    CHECK(m.value().source_format == "onnx");
    auto* be = nn::default_runtime_registry().find("onnxruntime");
    CHECK(be);
    CHECK(be->supports(m.value()));
    auto ins = nn::bind_model_inputs(m.value(), {}, 0, false);
    CHECK(ins);
    auto rr = nn::eval_model(m.value(), ins.value(), {}, "onnxruntime");
    CHECK(rr);
    CHECK(!rr.value().outputs.empty());
    CHECK(!rr.value().outputs.front().bytes.empty());
}
#endif
#endif

#if defined(NN_HAS_LITERT) && defined(NN_TEST_TFLITE_ADD)
TEST(litert_add_zeros) {
    auto m = nn::load_model(NN_TEST_TFLITE_ADD);
    CHECK(m);
    CHECK(m.value().source_format == "tflite");
    auto* be = nn::default_runtime_registry().find("litert");
    CHECK(be);
    CHECK(be->supports(m.value()));
    auto ins = nn::bind_model_inputs(m.value(), {}, 0, false);
    CHECK(ins);
    auto rr = nn::eval_model(m.value(), ins.value(), {}, "litert");
    CHECK(rr);
    CHECK(!rr.value().outputs.empty());
    CHECK(!rr.value().outputs.front().bytes.empty());
}

TEST(litert_add_default_backend) {
    auto m = nn::load_model(NN_TEST_TFLITE_ADD);
    CHECK(m);
    auto ins = nn::bind_model_inputs(m.value(), {}, 0, false);
    CHECK(ins);
    auto rr = nn::eval_model(m.value(), ins.value(), {}, {});
    CHECK(rr);
    CHECK(!rr.value().outputs.empty());
}

#if defined(NN_HAS_ONNXRUNTIME)
TEST(compare_cross_format_zeros) {
    auto tfl = nn::load_model(NN_TEST_TFLITE_ADD);
    CHECK(tfl);
    const nn::Graph* g = nn::primary_graph(tfl.value());
    CHECK(g && !g->inputs.empty());
    const nn::Tensor* in = g->find_tensor(g->inputs.front());
    CHECK(in);
    std::vector<int64_t> dims;
    for (const auto& d : in->shape.dims) {
        dims.push_back(d.value.value_or(1));
    }
    const auto onnx_path = std::filesystem::temp_directory_path() / "nn_cross_add.onnx";
    nn::SimpleOnnxSpec spec;
    spec.op_type = "Add";
    spec.inputs = 2;
    spec.input_shape = dims;
    CHECK(nn::write_simple_onnx(onnx_path, spec));
    auto onnx = nn::load_model(onnx_path);
    CHECK(onnx);
    auto ins_t = nn::bind_model_inputs(tfl.value(), {}, 0, false);
    auto ins_o = nn::bind_model_inputs(onnx.value(), {}, 0, false);
    CHECK(ins_t && ins_o);
    auto rt = nn::eval_model(tfl.value(), ins_t.value(), {}, {});
    auto ro = nn::eval_model(onnx.value(), ins_o.value(), {}, {});
    CHECK(rt && ro);
    auto pairs = nn::align_runtime_tensors(rt.value().outputs, ro.value().outputs);
    CHECK(!pairs.empty());
    CHECK(pairs.front().a && pairs.front().b);
    auto c = nn::compare_numeric_aligned(*pairs.front().a, *pairs.front().b, 1e-4, 1e-4);
    CHECK(c.comparable);
    CHECK(c.changed == 0);
}
#endif
#endif

TEST(compat_reference_add) {
    const auto path = std::filesystem::temp_directory_path() / "nn_compat.onnx";
    CHECK(nn::write_simple_onnx(path, {}));
    auto m = nn::load_model(path);
    CHECK(m);
    auto r = nn::check_compat(m.value(), "reference", "0.1.0");
    CHECK(r.compatible);
}

TEST(convert_routes_list_onnx) {
    bool found = false;
    for (const auto& r : nn::conversion_routes()) {
        if (r.from == "onnx" && r.to == "onnx") {
            found = r.available;
        }
    }
    CHECK(found);
}

TEST(test_manifest_eval) {
    const auto dir = std::filesystem::temp_directory_path() / "nn_tests";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto model = dir / "model.onnx";
    nn::SimpleOnnxSpec spec;
    spec.op_type = "Add";
    spec.inputs = 2;
    spec.input_shape = {1, 2};
    CHECK(nn::write_simple_onnx(model, spec));
    nn::RuntimeTensor t;
    t.dtype = nn::DataType::Float32;
    t.shape = nn::shape_from_ints({1, 2});
    float v[2] = {1, 2};
    t.bytes.resize(sizeof(v));
    std::memcpy(t.bytes.data(), v, sizeof(v));
    CHECK(nn::save_npy(dir / "a.npy", t));
    CHECK(nn::save_npy(dir / "b.npy", t));
    float e[2] = {2, 4};
    std::memcpy(t.bytes.data(), e, sizeof(e));
    CHECK(nn::save_npy(dir / "exp.npy", t));
    std::ofstream mf(dir / "manifest.json");
    mf << R"({"tests":[{"name":"add","inputs":{"input0":"a.npy","input1":"b.npy"},"expected":{"output":"exp.npy"}}]})";
    mf.close();
    auto cases = nn::load_test_manifest(dir);
    CHECK(cases);
    CHECK(cases.value().size() == 1);
    auto m = nn::load_model(model);
    CHECK(m);
    auto ins = nn::bind_model_inputs(m.value(),
                                     {"input0=" + (dir / "a.npy").string(),
                                      "input1=" + (dir / "b.npy").string()},
                                     0, false);
    CHECK(ins);
    auto rr = nn::eval_model(m.value(), ins.value(), {}, {});
    CHECK(rr);
    auto exp = nn::load_tensor_file(dir / "exp.npy");
    CHECK(exp);
    auto c = nn::compare_numeric(rr.value().outputs.front(), exp.value(), 1e-5, 1e-5);
    CHECK(c.comparable);
    CHECK(c.changed == 0);
}
