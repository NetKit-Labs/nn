#include "nn/analysis.h"
#include "test.h"

#include <cstring>
#include <memory>
#include <vector>

namespace {

nn::Tensor make_f32(nn::TensorId id, const std::string& name, const std::vector<int64_t>& dims,
                    const std::vector<float>& vals) {
    nn::Tensor t;
    t.id = id;
    t.name = name;
    t.constant = true;
    t.dtype = nn::DataType::Float32;
    t.shape = nn::shape_from_ints(dims);
    auto data = std::make_shared<std::vector<uint8_t>>(vals.size() * sizeof(float));
    std::memcpy(data->data(), vals.data(), data->size());
    nn::TensorDataReference ref;
    ref.owned = data;
    ref.length = data->size();
    t.data = ref;
    t.storage_bytes = data->size();
    return t;
}

nn::Tensor make_act(nn::TensorId id, const std::string& name, const std::vector<int64_t>& dims) {
    nn::Tensor t;
    t.id = id;
    t.name = name;
    t.dtype = nn::DataType::Float32;
    t.shape = nn::shape_from_ints(dims);
    return t;
}

}  // namespace

TEST(sparsity_float) {
    nn::ModelIR m;
    nn::Graph g;
    g.tensors.push_back(make_f32(0, "w", {4}, {0.f, 0.f, 1.f, 2.f}));
    m.graphs.push_back(g);
    auto r = nn::analyze_sparsity(m, {});
    CHECK(r.tensors_computed == 1);
    CHECK(r.total_zeros == 2);
    CHECK(r.total_near_zeros == 2);
}

TEST(sparsity_near_zeros_threshold) {
    nn::ModelIR m;
    nn::Graph g;
    g.tensors.push_back(make_f32(0, "w", {4}, {0.f, 1e-8f, 1.f, 2.f}));
    m.graphs.push_back(g);
    nn::SparsityOptions opt;
    opt.threshold = 1e-6;
    auto r = nn::analyze_sparsity(m, opt);
    CHECK(r.tensors_computed == 1);
    CHECK(r.total_zeros == 1);
    CHECK(r.total_near_zeros == 2);
    CHECK(r.tensors[0].near_zeros == 2);
}

TEST(sparsity_conv_weak_channels) {
    nn::ModelIR m;
    nn::Graph g;
    // ONNX [Co, Ci, Kh, Kw] = [4, 1, 1, 1]: channels 0 and 1 tiny, 2 and 3 large.
    g.tensors.push_back(make_act(0, "x", {1, 1, 8, 8}));
    g.tensors.push_back(make_f32(1, "w", {4, 1, 1, 1}, {0.001f, 0.001f, 1.f, 1.f}));
    g.tensors.push_back(make_act(2, "y", {1, 4, 8, 8}));
    nn::Node n;
    n.id = 0;
    n.name = "conv1";
    n.op_type = "Conv";
    n.canonical = nn::CanonicalOp::Convolution;
    n.inputs = {0, 1};
    n.outputs = {2};
    g.nodes.push_back(n);
    m.graphs.push_back(g);

    auto r = nn::analyze_sparsity(m, {});
    CHECK(r.tensors_computed == 1);
    const nn::TensorSparsity* w = nullptr;
    for (const auto& t : r.tensors) {
        if (t.name == "w") {
            w = &t;
        }
    }
    CHECK(w != nullptr);
    CHECK(w->layer == "conv1");
    CHECK(w->layout == "conv-onnx");
    CHECK(w->channels == 4);
    CHECK(w->weak_channels == 2);
    CHECK(w->macs_known);
    CHECK(w->score > 0);
    CHECK(w->estimated_saved_bytes > 0);
    CHECK(w->estimated_saved_macs > 0);
}

TEST(sparsity_dense_out_cols) {
    nn::ModelIR m;
    nn::Graph g;
    // Gemm transB=0: weight [in, out] = [2, 3]. Last dim is out.
    // Column 0 all ~0, columns 1 and 2 large.
    g.tensors.push_back(make_act(0, "x", {4, 2}));
    g.tensors.push_back(make_f32(1, "w", {2, 3}, {0.001f, 1.f, 1.f, 0.001f, 1.f, 1.f}));
    g.tensors.push_back(make_act(2, "y", {4, 3}));
    nn::Node n;
    n.id = 0;
    n.name = "fc";
    n.op_type = "Gemm";
    n.canonical = nn::CanonicalOp::Gemm;
    n.inputs = {0, 1};
    n.outputs = {2};
    g.nodes.push_back(n);
    m.graphs.push_back(g);

    auto r = nn::analyze_sparsity(m, {});
    const nn::TensorSparsity* w = nullptr;
    for (const auto& t : r.tensors) {
        if (t.name == "w") {
            w = &t;
        }
    }
    CHECK(w != nullptr);
    CHECK(w->layout == "dense-out-cols");
    CHECK(w->channels == 3);
    CHECK(w->weak_channels == 1);
}

TEST(sparsity_sort_by_score) {
    nn::ModelIR m;
    nn::Graph g;
    g.tensors.push_back(make_act(0, "x0", {1, 1, 4, 4}));
    g.tensors.push_back(make_f32(1, "w_small_mac", {4, 1, 1, 1}, {0.001f, 0.001f, 1.f, 1.f}));
    g.tensors.push_back(make_act(2, "y0", {1, 4, 4, 4}));
    g.tensors.push_back(make_act(3, "x1", {1, 1, 16, 16}));
    g.tensors.push_back(make_f32(4, "w_big_mac", {4, 1, 1, 1}, {0.001f, 1.f, 1.f, 1.f}));
    g.tensors.push_back(make_act(5, "y1", {1, 4, 16, 16}));
    nn::Node a;
    a.id = 0;
    a.name = "conv_small";
    a.op_type = "Conv";
    a.canonical = nn::CanonicalOp::Convolution;
    a.inputs = {0, 1};
    a.outputs = {2};
    nn::Node b;
    b.id = 1;
    b.name = "conv_big";
    b.op_type = "Conv";
    b.canonical = nn::CanonicalOp::Convolution;
    b.inputs = {3, 4};
    b.outputs = {5};
    g.nodes = {a, b};
    m.graphs.push_back(g);

    auto r = nn::analyze_sparsity(m, {});
    CHECK(r.tensors.size() >= 2);
    CHECK(r.tensors[0].name == "w_big_mac");
    CHECK(r.tensors[0].score >= r.tensors[1].score);
}

TEST(sparsity_skip_coupled_add) {
    nn::ModelIR m;
    nn::Graph g;
    g.tensors.push_back(make_act(0, "x", {1, 1, 4, 4}));
    g.tensors.push_back(make_f32(1, "w", {2, 1, 1, 1}, {1.f, 1.f}));
    g.tensors.push_back(make_act(2, "y", {1, 2, 4, 4}));
    g.tensors.push_back(make_act(3, "skip", {1, 2, 4, 4}));
    g.tensors.push_back(make_act(4, "out", {1, 2, 4, 4}));
    nn::Node conv;
    conv.id = 0;
    conv.name = "conv1";
    conv.op_type = "Conv";
    conv.canonical = nn::CanonicalOp::Convolution;
    conv.inputs = {0, 1};
    conv.outputs = {2};
    nn::Node add;
    add.id = 1;
    add.name = "add1";
    add.op_type = "Add";
    add.canonical = nn::CanonicalOp::Elementwise;
    add.inputs = {2, 3};
    add.outputs = {4};
    g.nodes = {conv, add};
    m.graphs.push_back(g);

    auto r = nn::analyze_sparsity(m, {});
    bool found = false;
    for (const auto& t : r.tensors) {
        if (t.name == "w") {
            CHECK(t.skip_coupled);
            found = true;
        }
    }
    CHECK(found);
    bool note = false;
    for (const auto& n : r.notes) {
        if (n.find("Add/Concat") != std::string::npos) {
            note = true;
        }
    }
    CHECK(note);
}
