#include "nn/analysis.h"

#include "util/overflow.h"

#include <algorithm>

namespace nn {
namespace {

std::optional<int64_t> attr_int(const Node& n, const char* key) {
    auto it = n.attributes.find(key);
    if (it == n.attributes.end()) {
        return std::nullopt;
    }
    if (it->second.kind == Attribute::Kind::Int) {
        return it->second.i;
    }
    if (it->second.kind == Attribute::Kind::Ints && !it->second.ints.empty()) {
        return it->second.ints.front();
    }
    return std::nullopt;
}

std::vector<int64_t> attr_ints(const Node& n, const char* key) {
    auto it = n.attributes.find(key);
    if (it == n.attributes.end()) {
        return {};
    }
    if (it->second.kind == Attribute::Kind::Ints) {
        return it->second.ints;
    }
    if (it->second.kind == Attribute::Kind::Int) {
        return {it->second.i};
    }
    return {};
}

bool static_shape(const Tensor* t) { return t && t->shape.is_static(); }

Result<uint64_t> elems(const Tensor* t) {
    if (!t) {
        return error(ErrorCode::InvalidArgument, "missing tensor");
    }
    return t->shape.element_count();
}

ComputeCost macs_only(uint64_t macs) {
    ComputeCost c;
    c.macs = {true, macs};
    c.flops = {true, macs * 2};
    c.float_ops = c.flops;
    return c;
}

ComputeCost elemwise(uint64_t n, bool floating) {
    ComputeCost c;
    if (floating) {
        c.float_ops = {true, n};
        c.flops = c.float_ops;
    } else {
        c.int_ops = {true, n};
    }
    return c;
}

uint64_t last_static(const Shape& s, std::size_t n) {
    if (s.dims.size() < n) {
        return 0;
    }
    uint64_t acc = 1;
    for (std::size_t i = s.dims.size() - n; i < s.dims.size(); ++i) {
        if (!s.dims[i].value || *s.dims[i].value < 0) {
            return 0;
        }
        acc *= static_cast<uint64_t>(*s.dims[i].value);
    }
    return acc;
}

ComputeCost conv_cost(const Graph& g, const Node& n, bool depthwise) {
    if (n.inputs.empty()) {
        return {};
    }
    const Tensor* x = g.find_tensor(n.inputs[0]);
    const Tensor* w = n.inputs.size() > 1 ? g.find_tensor(n.inputs[1]) : nullptr;
    if (!static_shape(x)) {
        return {};
    }
    const Shape& xs = x->shape;
    if (xs.rank() < 3) {
        return {};
    }
    // Assume NCHW (ONNX) or NHWC (TFLite). Use weight tensor when present.
    int64_t n_batch = xs.dims[0].value.value_or(1);
    int64_t c_in = 0;
    int64_t spatial = 1;
    bool nhwc = false;
    if (w && w->shape.rank() >= 4 && w->shape.is_static()) {
        // ONNX weight: M, C/g, kH, kW. TFLite OHWI-ish varies; use spatial from input.
        nhwc = (xs.rank() == 4 && xs.dims.back().value && w->shape.dims[0].value &&
                *xs.dims.back().value == *w->shape.dims[0].value);
    }
    if (xs.rank() == 4) {
        if (nhwc) {
            c_in = *xs.dims[3].value;
            spatial = *xs.dims[1].value * *xs.dims[2].value;
        } else {
            c_in = *xs.dims[1].value;
            spatial = *xs.dims[2].value * *xs.dims[3].value;
        }
    } else if (xs.rank() == 3) {
        c_in = *xs.dims[1].value;
        spatial = *xs.dims[2].value;
    } else if (xs.rank() == 5) {
        c_in = *xs.dims[1].value;
        spatial = *xs.dims[2].value * *xs.dims[3].value * *xs.dims[4].value;
    }
    int64_t c_out = c_in;
    int64_t kh = 1, kw = 1, kd = 1;
    auto ks = attr_ints(n, "kernel_shape");
    if (ks.size() >= 2) {
        kh = ks[0];
        kw = ks[1];
        if (ks.size() >= 3) {
            kd = ks[2];
        }
    } else if (w && w->shape.rank() >= 4 && w->shape.is_static()) {
        // ONNX: [M, C/g, kH, kW]
        c_out = *w->shape.dims[0].value;
        kh = *w->shape.dims[2].value;
        kw = *w->shape.dims[3].value;
        if (w->shape.rank() == 5) {
            kd = *w->shape.dims[4].value;
        }
    }
    int64_t group = attr_int(n, "group").value_or(1);
    if (group <= 0) {
        group = 1;
    }
    if (depthwise) {
        group = c_in;
        c_out = c_in;
        auto dm = attr_int(n, "depth_multiplier");
        if (dm) {
            c_out = c_in * *dm;
        }
    }
    if (n.outputs.empty()) {
        return {};
    }
    const Tensor* y = g.find_tensor(n.outputs[0]);
    uint64_t out_spatial = spatial;
    if (y && y->shape.is_static() && y->shape.rank() >= 3) {
        if (y->shape.rank() == 4) {
            if (nhwc) {
                out_spatial = static_cast<uint64_t>(*y->shape.dims[1].value * *y->shape.dims[2].value);
                c_out = *y->shape.dims[3].value;
                n_batch = *y->shape.dims[0].value;
            } else {
                out_spatial = static_cast<uint64_t>(*y->shape.dims[2].value * *y->shape.dims[3].value);
                c_out = *y->shape.dims[1].value;
                n_batch = *y->shape.dims[0].value;
            }
        }
    }
    const uint64_t cin_per_group = static_cast<uint64_t>(c_in / group);
    uint64_t macs = 0;
    uint64_t t = 0;
    if (mul_overflow_u64(static_cast<uint64_t>(n_batch), static_cast<uint64_t>(c_out), &t) ||
        mul_overflow_u64(t, out_spatial, &t) || mul_overflow_u64(t, cin_per_group, &t) ||
        mul_overflow_u64(t, static_cast<uint64_t>(kh), &t) ||
        mul_overflow_u64(t, static_cast<uint64_t>(kw), &t) ||
        mul_overflow_u64(t, static_cast<uint64_t>(kd), &macs)) {
        return {};
    }
    return macs_only(macs);
}

ComputeCost matmul_cost(const Graph& g, const Node& n) {
    if (n.inputs.size() < 2) {
        return {};
    }
    const Tensor* a = g.find_tensor(n.inputs[0]);
    const Tensor* b = g.find_tensor(n.inputs[1]);
    if (!static_shape(a) || !static_shape(b) || a->shape.rank() < 2 || b->shape.rank() < 2) {
        return {};
    }
    const auto& ad = a->shape.dims;
    const auto& bd = b->shape.dims;
    const int64_t k = *ad.back().value;
    const int64_t m = *ad[ad.size() - 2].value;
    const int64_t n_dim = *bd.back().value;
    if (*bd[bd.size() - 2].value != k && *bd.back().value != k) {
        // still compute with declared shapes
    }
    uint64_t batch = 1;
    const std::size_t ar = ad.size();
    if (ar > 2) {
        for (std::size_t i = 0; i + 2 < ar; ++i) {
            batch *= static_cast<uint64_t>(*ad[i].value);
        }
    }
    uint64_t macs = 0;
    uint64_t t = 0;
    if (mul_overflow_u64(batch, static_cast<uint64_t>(m), &t) ||
        mul_overflow_u64(t, static_cast<uint64_t>(n_dim), &t) ||
        mul_overflow_u64(t, static_cast<uint64_t>(k), &macs)) {
        return {};
    }
    return macs_only(macs);
}

ComputeCost gemm_cost(const Graph& g, const Node& n) {
    // Treat like MatMul of A and B. transA/transB swap the last two dims.
    if (n.inputs.size() < 2) {
        return {};
    }
    Tensor a = *g.find_tensor(n.inputs[0]);
    Tensor b = *g.find_tensor(n.inputs[1]);
    const int64_t transA = attr_int(n, "transA").value_or(0);
    const int64_t transB = attr_int(n, "transB").value_or(0);
    auto swap_last2 = [](Shape& s) {
        if (s.rank() >= 2) {
            std::swap(s.dims[s.rank() - 1], s.dims[s.rank() - 2]);
        }
    };
    if (transA) {
        swap_last2(a.shape);
    }
    if (transB) {
        swap_last2(b.shape);
    }
    Graph tmp = g;
    // Use local tensors
    Node nn = n;
    return matmul_cost(g, n);
}

}  // namespace

ComputeCost estimate_node_compute(const Graph& graph, const Node& node) {
    switch (node.canonical) {
        case CanonicalOp::Convolution:
        case CanonicalOp::GroupedConvolution:
            return conv_cost(graph, node, false);
        case CanonicalOp::DepthwiseConvolution:
            return conv_cost(graph, node, true);
        case CanonicalOp::MatMul:
            return matmul_cost(graph, node);
        case CanonicalOp::Dense:
        case CanonicalOp::Gemm:
            return matmul_cost(graph, node);
        case CanonicalOp::Activation:
        case CanonicalOp::Elementwise:
        case CanonicalOp::Softmax: {
            if (node.outputs.empty()) {
                return {};
            }
            const Tensor* y = graph.find_tensor(node.outputs[0]);
            auto e = elems(y);
            if (!e) {
                return {};
            }
            const bool fl = y && datatype_is_float(y->dtype);
            auto c = elemwise(e.value(), fl);
            if (node.canonical == CanonicalOp::Softmax) {
                c.flops = {true, e.value() * 3};  // exp + sum + div (approx)
                c.float_ops = c.flops;
            }
            return c;
        }
        case CanonicalOp::Pooling:
        case CanonicalOp::Normalization:
        case CanonicalOp::Reshape:
        case CanonicalOp::Transpose:
        case CanonicalOp::Concatenation:
        case CanonicalOp::Split:
        case CanonicalOp::Identity:
        case CanonicalOp::Cast:
        case CanonicalOp::Pad:
        case CanonicalOp::Slice:
        case CanonicalOp::Quantize:
        case CanonicalOp::Dequantize:
        case CanonicalOp::Constant:
        case CanonicalOp::Dropout:
            return {};  // known-zero extra MACs; not unknown
        default:
            return {};
    }
}

ComputeReport analyze_compute(const ModelIR& model) {
    ComputeReport report;
    const Graph* g = primary_graph(model);
    if (!g) {
        return report;
    }
    auto add = [](OptionalCount& dst, const OptionalCount& src) {
        if (!src.known) {
            return;
        }
        if (!dst.known) {
            dst.known = true;
            dst.value = 0;
        }
        dst.value += src.value;
    };
    for (const auto& n : g->nodes) {
        NodeCompute nc;
        nc.node_id = n.id;
        nc.name = n.name;
        nc.op_type = n.op_type;
        nc.canonical = n.canonical;
        nc.cost = estimate_node_compute(*g, n);
        const bool any = nc.cost.macs.known || nc.cost.flops.known || nc.cost.int_ops.known ||
                         nc.cost.float_ops.known;
        if (!any && n.canonical != CanonicalOp::Reshape && n.canonical != CanonicalOp::Identity &&
            n.canonical != CanonicalOp::Constant && n.canonical != CanonicalOp::Transpose &&
            n.canonical != CanonicalOp::Cast && n.canonical != CanonicalOp::Dropout &&
            n.canonical != CanonicalOp::Pad && n.canonical != CanonicalOp::Slice &&
            n.canonical != CanonicalOp::Concatenation && n.canonical != CanonicalOp::Split &&
            n.canonical != CanonicalOp::Pooling && n.canonical != CanonicalOp::Quantize &&
            n.canonical != CanonicalOp::Dequantize && n.canonical != CanonicalOp::Normalization) {
            ++report.unknown_node_count;
        }
        add(report.total.macs, nc.cost.macs);
        add(report.total.flops, nc.cost.flops);
        add(report.total.int_ops, nc.cost.int_ops);
        add(report.total.float_ops, nc.cost.float_ops);
        report.nodes.push_back(std::move(nc));
    }
    (void)last_static;
    (void)gemm_cost;
    (void)attr_int;
    return report;
}

}  // namespace nn
