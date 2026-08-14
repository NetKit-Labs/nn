#include "nn/analysis.h"

#include "nn/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nn {
namespace {

int64_t dim_at(const Shape& s, std::size_t i) {
    if (i >= s.dims.size() || !s.dims[i].value) {
        return 0;
    }
    return *s.dims[i].value;
}

int64_t attr_int(const Node& n, const char* key, int64_t fallback = 0) {
    auto it = n.attributes.find(key);
    if (it == n.attributes.end()) {
        return fallback;
    }
    if (it->second.kind == Attribute::Kind::Int) {
        return it->second.i;
    }
    return fallback;
}

bool is_compute_weight_op(CanonicalOp c) {
    return c == CanonicalOp::Convolution || c == CanonicalOp::GroupedConvolution ||
           c == CanonicalOp::DepthwiseConvolution || c == CanonicalOp::Dense ||
           c == CanonicalOp::Gemm || c == CanonicalOp::MatMul;
}

bool is_weight_input(const Node& n, TensorId id) {
    if (n.inputs.size() >= 3 && n.inputs[2] == id) {
        return false;  // bias
    }
    if (n.inputs.size() >= 2 && n.inputs[1] == id) {
        return true;
    }
    if (n.inputs.size() == 1 && n.inputs[0] == id) {
        return true;
    }
    return false;
}

bool looks_spatial(int64_t a, int64_t b) { return a > 0 && b > 0 && a <= 32 && b <= 32; }

enum class WeightLayout { None, ConvOnnx, ConvTflite, DenseRows, DenseCols };

const char* layout_name(WeightLayout l) {
    switch (l) {
        case WeightLayout::ConvOnnx:
            return "conv-onnx";
        case WeightLayout::ConvTflite:
            return "conv-tflite";
        case WeightLayout::DenseRows:
            return "dense-out-rows";
        case WeightLayout::DenseCols:
            return "dense-out-cols";
        default:
            return "unstructured";
    }
}

WeightLayout detect_layout(const Tensor& t, const Node* consumer) {
    if (!t.shape.is_static()) {
        return WeightLayout::None;
    }
    const int rnk = static_cast<int>(t.shape.rank());
    const int64_t d0 = dim_at(t.shape, 0);
    const int64_t d1 = dim_at(t.shape, 1);
    const int64_t d2 = dim_at(t.shape, 2);
    const int64_t d3 = dim_at(t.shape, 3);
    const CanonicalOp c = consumer ? consumer->canonical : CanonicalOp::Unknown;

    if (c == CanonicalOp::DepthwiseConvolution && rnk == 4) {
        if (d0 == 1) {
            return WeightLayout::ConvTflite;
        }
        return WeightLayout::ConvOnnx;
    }
    if ((c == CanonicalOp::Convolution || c == CanonicalOp::GroupedConvolution) && rnk == 4) {
        if (looks_spatial(d2, d3)) {
            return WeightLayout::ConvOnnx;
        }
        if (looks_spatial(d0, d1)) {
            return WeightLayout::ConvTflite;
        }
        return WeightLayout::ConvOnnx;
    }
    if ((c == CanonicalOp::Dense || c == CanonicalOp::Gemm || c == CanonicalOp::MatMul) &&
        rnk == 2) {
        const std::string op = consumer->op_type;
        std::string lower = op;
        for (char& ch : lower) {
            if (ch >= 'A' && ch <= 'Z') {
                ch = static_cast<char>(ch - 'A' + 'a');
            }
        }
        if (lower.find("fullyconnected") != std::string::npos ||
            lower.find("fully_connected") != std::string::npos || lower == "dense") {
            return WeightLayout::DenseRows;
        }
        if (c == CanonicalOp::Gemm || c == CanonicalOp::Dense) {
            return attr_int(*consumer, "transB", 0) ? WeightLayout::DenseRows
                                                    : WeightLayout::DenseCols;
        }
        return WeightLayout::DenseCols;
    }
    return WeightLayout::None;
}

std::vector<double> values_f64(const Tensor& t, const std::vector<uint8_t>& bytes) {
    std::vector<double> v;
    auto take = [&](auto* p, std::size_t n) {
        v.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = static_cast<double>(p[i]);
        }
    };
    if (t.dtype == DataType::Float32) {
        take(reinterpret_cast<const float*>(bytes.data()), bytes.size() / sizeof(float));
    } else if (t.dtype == DataType::Float64) {
        take(reinterpret_cast<const double*>(bytes.data()), bytes.size() / sizeof(double));
    } else if (t.dtype == DataType::Int8) {
        take(reinterpret_cast<const int8_t*>(bytes.data()), bytes.size());
    } else if (t.dtype == DataType::Int32) {
        take(reinterpret_cast<const int32_t*>(bytes.data()), bytes.size() / sizeof(int32_t));
    }
    return v;
}

void count_magnitude(const std::vector<double>& v, double threshold, TensorSparsity& ts) {
    ts.elements = v.size();
    ts.computed = true;
    for (double x : v) {
        if (x == 0.0) {
            ++ts.zeros;
        }
        if (std::fabs(x) <= threshold) {
            ++ts.near_zeros;
        }
    }
    ts.zero_fraction =
        ts.elements ? static_cast<double>(ts.zeros) / static_cast<double>(ts.elements) : 0.0;
    ts.near_zero_fraction =
        ts.elements ? static_cast<double>(ts.near_zeros) / static_cast<double>(ts.elements) : 0.0;
}

std::vector<double> channel_l1(const std::vector<double>& v, const Tensor& t, WeightLayout layout) {
    const int64_t d0 = dim_at(t.shape, 0);
    const int64_t d1 = dim_at(t.shape, 1);
    const int64_t d2 = dim_at(t.shape, 2);
    const int64_t d3 = dim_at(t.shape, 3);
    std::vector<double> l1;
    if (layout == WeightLayout::ConvOnnx || layout == WeightLayout::DenseRows) {
        const int64_t nch = d0;
        if (nch <= 0) {
            return {};
        }
        const std::size_t inner = v.size() / static_cast<std::size_t>(nch);
        if (inner == 0 || inner * static_cast<std::size_t>(nch) != v.size()) {
            return {};
        }
        l1.assign(static_cast<std::size_t>(nch), 0);
        for (int64_t c = 0; c < nch; ++c) {
            double s = 0;
            const std::size_t off = static_cast<std::size_t>(c) * inner;
            for (std::size_t j = 0; j < inner; ++j) {
                s += std::fabs(v[off + j]);
            }
            l1[static_cast<std::size_t>(c)] = s;
        }
        return l1;
    }
    if (layout == WeightLayout::DenseCols) {
        const int64_t rows = d0;
        const int64_t cols = d1;
        if (rows <= 0 || cols <= 0 ||
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) != v.size()) {
            return {};
        }
        l1.assign(static_cast<std::size_t>(cols), 0);
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                l1[static_cast<std::size_t>(c)] +=
                    std::fabs(v[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
                                static_cast<std::size_t>(c)]);
            }
        }
        return l1;
    }
    if (layout == WeightLayout::ConvTflite) {
        const int64_t kh = d0;
        const int64_t kw = d1;
        const int64_t ci = d2;
        const int64_t co = d3;
        if (kh <= 0 || kw <= 0 || ci <= 0 || co <= 0) {
            return {};
        }
        const std::size_t want = static_cast<std::size_t>(kh) * static_cast<std::size_t>(kw) *
                                 static_cast<std::size_t>(ci) * static_cast<std::size_t>(co);
        if (want != v.size()) {
            return {};
        }
        l1.assign(static_cast<std::size_t>(co), 0);
        for (int64_t y = 0; y < kh; ++y) {
            for (int64_t x = 0; x < kw; ++x) {
                for (int64_t i = 0; i < ci; ++i) {
                    for (int64_t o = 0; o < co; ++o) {
                        const std::size_t idx =
                            static_cast<std::size_t>(((y * kw + x) * ci + i) * co + o);
                        l1[static_cast<std::size_t>(o)] += std::fabs(v[idx]);
                    }
                }
            }
        }
        return l1;
    }
    return {};
}

const Node* primary_consumer(const Graph& g, TensorId id) {
    const Node* best = nullptr;
    for (const auto& n : g.nodes) {
        bool uses = false;
        for (TensorId in : n.inputs) {
            if (in == id) {
                uses = true;
                break;
            }
        }
        if (!uses) {
            continue;
        }
        if (is_compute_weight_op(n.canonical) && is_weight_input(n, id)) {
            return &n;
        }
        if (!best) {
            best = &n;
        }
    }
    return best;
}

bool output_feeds_merge(const Graph& g, const Node& n) {
    for (TensorId oid : n.outputs) {
        for (const auto& m : g.nodes) {
            if (m.id == n.id) {
                continue;
            }
            for (TensorId iid : m.inputs) {
                if (iid != oid) {
                    continue;
                }
                if (m.canonical == CanonicalOp::Concatenation ||
                    m.canonical == CanonicalOp::Elementwise) {
                    return true;
                }
            }
        }
    }
    return false;
}

}  // namespace

SparsityReport analyze_sparsity(const ModelIR& model, const SparsityOptions& options) {
    SparsityReport r;
    r.threshold = options.threshold;
    r.channel_l1_frac = options.channel_l1_frac;
    const Graph* g = primary_graph(model);
    if (!g) {
        return r;
    }

    const ComputeReport compute = analyze_compute(model);
    r.total_macs_known = compute.total.macs.known;
    r.total_macs = compute.total.macs.known ? compute.total.macs.value : 0;
    std::map<NodeId, uint64_t> macs_by_node;
    for (const auto& n : compute.nodes) {
        if (n.cost.macs.known) {
            macs_by_node[n.node_id] = n.cost.macs.value;
        }
    }

    bool any_structured = false;
    bool any_skip = false;
    bool any_dw = false;

    for (const auto& t : g->tensors) {
        if (!t.constant) {
            continue;
        }
        ++r.tensors_considered;
        TensorSparsity ts;
        ts.name = t.name;
        ts.shape = t.shape.to_string();
        if (auto b = tensor_storage_bytes(t)) {
            ts.bytes = b.value();
        }

        const Node* consumer = primary_consumer(*g, t.id);
        if (consumer) {
            ts.layer = consumer->name.empty() ? consumer->op_type : consumer->name;
            ts.op_type = consumer->op_type;
            ts.canonical = canonical_op_name(consumer->canonical);
            ts.depthwise = consumer->canonical == CanonicalOp::DepthwiseConvolution;
            ts.skip_coupled = output_feeds_merge(*g, *consumer);
            if (is_weight_input(*consumer, t.id)) {
                auto mit = macs_by_node.find(consumer->id);
                if (mit != macs_by_node.end()) {
                    ts.macs_known = true;
                    ts.macs = mit->second;
                }
            }
        }
        if (r.total_macs_known && r.total_macs > 0 && ts.macs_known) {
            ts.mac_share = static_cast<double>(ts.macs) / static_cast<double>(r.total_macs);
        }

        auto bytes = tensor_payload_bytes(t);
        if (!bytes || bytes.value().empty()) {
            ts.computed = false;
            ts.layout = layout_name(WeightLayout::None);
            r.tensors.push_back(std::move(ts));
            r.notes.push_back("weights not in memory for " + t.name + "; sparsity not computed");
            continue;
        }
        const std::vector<double> values = values_f64(t, bytes.value());
        if (values.empty()) {
            ts.computed = false;
            ts.layout = layout_name(WeightLayout::None);
            r.notes.push_back("unsupported dtype for sparsity on " + t.name);
            r.tensors.push_back(std::move(ts));
            continue;
        }

        count_magnitude(values, options.threshold, ts);
        ++r.tensors_computed;
        r.total_elements += ts.elements;
        r.total_zeros += ts.zeros;
        r.total_near_zeros += ts.near_zeros;

        WeightLayout layout = WeightLayout::None;
        if (consumer && is_weight_input(*consumer, t.id)) {
            layout = detect_layout(t, consumer);
        }
        ts.layout = layout_name(layout);

        if (layout != WeightLayout::None) {
            const std::vector<double> l1 = channel_l1(values, t, layout);
            if (!l1.empty()) {
                any_structured = true;
                ts.channels = l1.size();
                ts.max_channel_l1 = 0;
                for (double x : l1) {
                    ts.max_channel_l1 = std::max(ts.max_channel_l1, x);
                }
                const double cut = options.channel_l1_frac * ts.max_channel_l1;
                for (double x : l1) {
                    if (x <= cut) {
                        ++ts.weak_channels;
                    }
                }
                ts.weak_channel_frac = static_cast<double>(ts.weak_channels) /
                                       static_cast<double>(ts.channels);
                if (ts.skip_coupled) {
                    any_skip = true;
                }
                if (ts.depthwise) {
                    any_dw = true;
                }
            }
        }

        ts.score = ts.weak_channel_frac * ts.mac_share;
        if (ts.channels > 0 && ts.weak_channels > 0) {
            ts.estimated_saved_bytes =
                ts.bytes * ts.weak_channels / ts.channels;
            if (ts.macs_known) {
                ts.estimated_saved_macs = ts.macs * ts.weak_channels / ts.channels;
            }
            r.estimated_saved_bytes += ts.estimated_saved_bytes;
            r.estimated_saved_macs += ts.estimated_saved_macs;
        }

        r.tensors.push_back(std::move(ts));
    }

    r.overall_zero_fraction =
        r.total_elements ? static_cast<double>(r.total_zeros) / static_cast<double>(r.total_elements)
                         : 0.0;
    r.overall_near_zero_fraction =
        r.total_elements
            ? static_cast<double>(r.total_near_zeros) / static_cast<double>(r.total_elements)
            : 0.0;

    std::sort(r.tensors.begin(), r.tensors.end(), [](const TensorSparsity& a, const TensorSparsity& b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        if (a.weak_channel_frac != b.weak_channel_frac) {
            return a.weak_channel_frac > b.weak_channel_frac;
        }
        if (a.mac_share != b.mac_share) {
            return a.mac_share > b.mac_share;
        }
        if (a.near_zero_fraction != b.near_zero_fraction) {
            return a.near_zero_fraction > b.near_zero_fraction;
        }
        return a.bytes > b.bytes;
    });

    if (any_structured) {
        r.notes.push_back(
            "estimated savings are an upper bound; residual Add, Concat, and "
            "depthwise→pointwise couples reduce the real save");
    }
    if (any_skip) {
        r.notes.push_back(
            "one or more layers feed Add/Concat; channel prune is coupled to the other branch");
    }
    if (any_dw) {
        r.notes.push_back("depthwise channels are coupled to the following pointwise");
    }
    return r;
}

}  // namespace nn
