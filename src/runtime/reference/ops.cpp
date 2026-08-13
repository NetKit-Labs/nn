#include "runtime/reference/ops.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace nn {
namespace {

std::string lower_op(std::string_view s) {
    std::string o(s);
    const auto dot = o.find_last_of('.');
    if (dot != std::string::npos) {
        o = o.substr(dot + 1);
    }
    std::string compact;
    compact.reserve(o.size());
    for (char ch : o) {
        if (ch == '_' || ch == '-') {
            continue;
        }
        compact.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return compact;
}

std::vector<int64_t> dims_of(const Shape& s) {
    std::vector<int64_t> d;
    d.reserve(s.dims.size());
    for (const auto& dim : s.dims) {
        d.push_back(dim.value.value_or(0));
    }
    return d;
}

uint64_t numel(const std::vector<int64_t>& d) {
    uint64_t n = 1;
    for (int64_t v : d) {
        if (v < 0) {
            return 0;
        }
        n *= static_cast<uint64_t>(v);
    }
    return n;
}

int64_t attr_int(const Node& n, const char* key, int64_t def) {
    auto it = n.attributes.find(key);
    if (it == n.attributes.end()) {
        return def;
    }
    if (it->second.kind == Attribute::Kind::Int) {
        return it->second.i;
    }
    if (it->second.kind == Attribute::Kind::Ints && !it->second.ints.empty()) {
        return it->second.ints.front();
    }
    return def;
}

double attr_float(const Node& n, const char* key, double def) {
    auto it = n.attributes.find(key);
    if (it == n.attributes.end()) {
        return def;
    }
    if (it->second.kind == Attribute::Kind::Float) {
        return it->second.f;
    }
    if (it->second.kind == Attribute::Kind::Floats && !it->second.floats.empty()) {
        return it->second.floats.front();
    }
    return def;
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

Result<std::vector<float>> as_f32(const RuntimeTensor& t) {
    if (t.dtype != DataType::Float32) {
        return error(ErrorCode::UnsupportedOperator, "reference interpreter supports float32 only");
    }
    if (t.bytes.size() % sizeof(float) != 0) {
        return error(ErrorCode::ExecutionFailure, "float32 tensor size is not a multiple of 4");
    }
    std::vector<float> v(t.bytes.size() / sizeof(float));
    if (!v.empty()) {
        std::memcpy(v.data(), t.bytes.data(), v.size() * sizeof(float));
    }
    return v;
}

RuntimeTensor from_f32(std::string name, Shape shape, const std::vector<float>& v) {
    RuntimeTensor rt;
    rt.name = std::move(name);
    rt.dtype = DataType::Float32;
    rt.shape = std::move(shape);
    rt.bytes.resize(v.size() * sizeof(float));
    if (!v.empty()) {
        std::memcpy(rt.bytes.data(), v.data(), rt.bytes.size());
    }
    return rt;
}

Result<std::vector<int64_t>> as_i64(const RuntimeTensor& t) {
    std::vector<int64_t> v;
    if (t.dtype == DataType::Int64) {
        if (t.bytes.size() % sizeof(int64_t) != 0) {
            return error(ErrorCode::ExecutionFailure, "int64 tensor size is invalid");
        }
        v.resize(t.bytes.size() / sizeof(int64_t));
        if (!v.empty()) {
            std::memcpy(v.data(), t.bytes.data(), v.size() * sizeof(int64_t));
        }
        return v;
    }
    if (t.dtype == DataType::Int32) {
        if (t.bytes.size() % sizeof(int32_t) != 0) {
            return error(ErrorCode::ExecutionFailure, "int32 tensor size is invalid");
        }
        const std::size_t n = t.bytes.size() / sizeof(int32_t);
        v.resize(n);
        const auto* p = reinterpret_cast<const int32_t*>(t.bytes.data());
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = p[i];
        }
        return v;
    }
    return error(ErrorCode::UnsupportedOperator, "shape tensor must be int32 or int64");
}

Result<std::vector<int64_t>> broadcast_shape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    const std::size_t r = std::max(a.size(), b.size());
    std::vector<int64_t> o(r, 1);
    for (std::size_t i = 0; i < r; ++i) {
        const int64_t da = i < r - a.size() ? 1 : a[i - (r - a.size())];
        const int64_t db = i < r - b.size() ? 1 : b[i - (r - b.size())];
        if (da != db && da != 1 && db != 1) {
            return error(ErrorCode::ExecutionFailure, "shapes are not broadcastable");
        }
        o[i] = std::max(da, db);
    }
    return o;
}

std::size_t broadcast_index(std::size_t linear, const std::vector<int64_t>& out_shape,
                            const std::vector<int64_t>& in_shape) {
    if (in_shape.empty()) {
        return 0;
    }
    std::size_t idx = 0;
    std::size_t stride = 1;
    std::size_t remaining = linear;
    for (std::size_t dim = out_shape.size(); dim-- > 0;) {
        const int64_t od = out_shape[dim];
        const std::size_t coord = od > 0 ? remaining % static_cast<std::size_t>(od) : 0;
        remaining = od > 0 ? remaining / static_cast<std::size_t>(od) : 0;
        const int64_t id =
            dim < out_shape.size() - in_shape.size() ? 1 : in_shape[dim - (out_shape.size() - in_shape.size())];
        const std::size_t ic = (id == 1) ? 0 : coord;
        idx += ic * stride;
        stride *= static_cast<std::size_t>(id < 1 ? 1 : id);
    }
    return idx;
}

const RuntimeTensor* require(const std::map<TensorId, RuntimeTensor>& vals, TensorId id,
                             const char* what) {
    auto it = vals.find(id);
    if (it == vals.end()) {
        return nullptr;
    }
    (void)what;
    return &it->second;
}

Result<RuntimeTensor> binary_op(const Graph& graph, const Node& node,
                                const std::map<TensorId, RuntimeTensor>& vals, char op) {
    if (node.inputs.size() < 2 || node.outputs.empty()) {
        return error(ErrorCode::InvalidGraph, "binary op missing edges");
    }
    const auto* a = require(vals, node.inputs[0], "a");
    const auto* b = require(vals, node.inputs[1], "b");
    if (!a || !b) {
        return error(ErrorCode::ExecutionFailure, "missing input for " + node.op_type);
    }
    auto fa = as_f32(*a);
    auto fb = as_f32(*b);
    if (!fa) {
        return fa.error();
    }
    if (!fb) {
        return fb.error();
    }
    auto sa = dims_of(a->shape);
    auto sb = dims_of(b->shape);
    auto so = broadcast_shape(sa, sb);
    if (!so) {
        return so.error();
    }
    const uint64_t n = numel(so.value());
    std::vector<float> out(static_cast<std::size_t>(n));
    for (uint64_t i = 0; i < n; ++i) {
        const float x = fa.value()[broadcast_index(static_cast<std::size_t>(i), so.value(), sa)];
        const float y = fb.value()[broadcast_index(static_cast<std::size_t>(i), so.value(), sb)];
        float z = 0;
        switch (op) {
            case '+':
                z = x + y;
                break;
            case '-':
                z = x - y;
                break;
            case '*':
                z = x * y;
                break;
            case '/':
                z = x / y;
                break;
            case 'M':
                z = std::max(x, y);
                break;
            case 'm':
                z = std::min(x, y);
                break;
            case 'p':
                z = std::pow(x, y);
                break;
            default:
                break;
        }
        out[static_cast<std::size_t>(i)] = z;
    }
    Shape shape = shape_from_ints(so.value());
    if (const Tensor* t = graph.find_tensor(node.outputs[0]); t && t->shape.is_static()) {
        shape = t->shape;
    }
    return from_f32(graph.find_tensor(node.outputs[0]) ? graph.find_tensor(node.outputs[0])->name : "",
                    std::move(shape), out);
}

Result<RuntimeTensor> unary_f32(const Graph& graph, const Node& node,
                                const std::map<TensorId, RuntimeTensor>& vals,
                                float (*fn)(float)) {
    if (node.inputs.empty() || node.outputs.empty()) {
        return error(ErrorCode::InvalidGraph, "unary op missing edges");
    }
    const auto* a = require(vals, node.inputs[0], "x");
    if (!a) {
        return error(ErrorCode::ExecutionFailure, "missing input for " + node.op_type);
    }
    auto fa = as_f32(*a);
    if (!fa) {
        return fa.error();
    }
    for (float& v : fa.value()) {
        v = fn(v);
    }
    Shape shape = a->shape;
    if (const Tensor* t = graph.find_tensor(node.outputs[0]); t && t->shape.is_static()) {
        shape = t->shape;
    }
    return from_f32(graph.find_tensor(node.outputs[0]) ? graph.find_tensor(node.outputs[0])->name : "",
                    std::move(shape), fa.value());
}

void transpose_2d_to(const std::vector<float>& in, int64_t rows, int64_t cols, std::vector<float>& out) {
    out.resize(static_cast<std::size_t>(rows * cols));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            out[static_cast<std::size_t>(c * rows + r)] = in[static_cast<std::size_t>(r * cols + c)];
        }
    }
}

Result<RuntimeTensor> exec_matmul(const Graph& graph, const Node& node,
                                  const std::map<TensorId, RuntimeTensor>& vals) {
    if (node.inputs.size() < 2 || node.outputs.empty()) {
        return error(ErrorCode::InvalidGraph, "MatMul missing edges");
    }
    const auto* a = require(vals, node.inputs[0], "a");
    const auto* b = require(vals, node.inputs[1], "b");
    if (!a || !b) {
        return error(ErrorCode::ExecutionFailure, "missing MatMul input");
    }
    auto fa = as_f32(*a);
    auto fb = as_f32(*b);
    if (!fa) {
        return fa.error();
    }
    if (!fb) {
        return fb.error();
    }
    auto sa = dims_of(a->shape);
    auto sb = dims_of(b->shape);
    if (sa.empty()) {
        sa.push_back(1);
    }
    if (sb.empty()) {
        sb.push_back(1);
    }
    const bool a1d = sa.size() == 1;
    const bool b1d = sb.size() == 1;
    if (a1d) {
        sa.insert(sa.begin(), 1);
    }
    if (b1d) {
        sb.push_back(1);
    }
    if (sa.size() < 2 || sb.size() < 2) {
        return error(ErrorCode::ExecutionFailure, "MatMul requires rank >= 1");
    }
    const int64_t m = sa[sa.size() - 2];
    const int64_t k = sa[sa.size() - 1];
    const int64_t k2 = sb[sb.size() - 2];
    const int64_t n = sb[sb.size() - 1];
    if (k != k2) {
        return error(ErrorCode::ExecutionFailure, "MatMul inner dimensions do not match");
    }
    std::vector<int64_t> batch_a(sa.begin(), sa.end() - 2);
    std::vector<int64_t> batch_b(sb.begin(), sb.end() - 2);
    auto batch = broadcast_shape(batch_a, batch_b);
    if (!batch) {
        return batch.error();
    }
    const uint64_t batches = numel(batch.value());
    std::vector<int64_t> so = batch.value();
    if (!a1d) {
        so.push_back(m);
    }
    if (!b1d) {
        so.push_back(n);
    }
    std::vector<float> out(static_cast<std::size_t>(batches * static_cast<uint64_t>(m) * static_cast<uint64_t>(n)));
    const uint64_t a_mat = static_cast<uint64_t>(m) * static_cast<uint64_t>(k);
    const uint64_t b_mat = static_cast<uint64_t>(k) * static_cast<uint64_t>(n);
    const uint64_t o_mat = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
    for (uint64_t bi = 0; bi < batches; ++bi) {
        const std::size_t ia = broadcast_index(static_cast<std::size_t>(bi), batch.value(), batch_a);
        const std::size_t ib = broadcast_index(static_cast<std::size_t>(bi), batch.value(), batch_b);
        const float* ap = fa.value().data() + ia * static_cast<std::size_t>(a_mat);
        const float* bp = fb.value().data() + ib * static_cast<std::size_t>(b_mat);
        float* op = out.data() + static_cast<std::size_t>(bi * o_mat);
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float acc = 0;
                for (int64_t t = 0; t < k; ++t) {
                    acc += ap[i * k + t] * bp[t * n + j];
                }
                op[i * n + j] = acc;
            }
        }
    }
    Shape shape = shape_from_ints(so);
    if (const Tensor* t = graph.find_tensor(node.outputs[0]); t && t->shape.is_static() && !t->shape.dims.empty()) {
        shape = t->shape;
    }
    return from_f32(graph.find_tensor(node.outputs[0]) ? graph.find_tensor(node.outputs[0])->name : "",
                    std::move(shape), out);
}

Result<RuntimeTensor> exec_gemm(const Graph& graph, const Node& node,
                                const std::map<TensorId, RuntimeTensor>& vals) {
    if (node.inputs.size() < 2 || node.outputs.empty()) {
        return error(ErrorCode::InvalidGraph, "Gemm missing edges");
    }
    const auto* a = require(vals, node.inputs[0], "a");
    const auto* b = require(vals, node.inputs[1], "b");
    if (!a || !b) {
        return error(ErrorCode::ExecutionFailure, "missing Gemm input");
    }
    auto fa = as_f32(*a);
    auto fb = as_f32(*b);
    if (!fa) {
        return fa.error();
    }
    if (!fb) {
        return fb.error();
    }
    auto sa = dims_of(a->shape);
    auto sb = dims_of(b->shape);
    if (sa.size() != 2 || sb.size() != 2) {
        return error(ErrorCode::ExecutionFailure, "Gemm expects 2D inputs");
    }
    const int64_t transA = attr_int(node, "transA", 0);
    const int64_t transB = attr_int(node, "transB", 0);
    const float alpha = static_cast<float>(attr_float(node, "alpha", 1.0));
    const float beta = static_cast<float>(attr_float(node, "beta", 1.0));
    std::vector<float> A = fa.value();
    std::vector<float> B = fb.value();
    int64_t m = sa[0];
    int64_t k = sa[1];
    int64_t k2 = sb[0];
    int64_t n = sb[1];
    if (transA) {
        transpose_2d_to(fa.value(), sa[0], sa[1], A);
        m = sa[1];
        k = sa[0];
    }
    if (transB) {
        transpose_2d_to(fb.value(), sb[0], sb[1], B);
        k2 = sb[1];
        n = sb[0];
    }
    if (k != k2) {
        return error(ErrorCode::ExecutionFailure, "Gemm inner dimensions do not match");
    }
    std::vector<float> out(static_cast<std::size_t>(m * n), 0.0f);
    if (node.inputs.size() >= 3) {
        const auto* c = require(vals, node.inputs[2], "c");
        if (c) {
            auto fc = as_f32(*c);
            if (!fc) {
                return fc.error();
            }
            auto sc = dims_of(c->shape);
            auto so = std::vector<int64_t>{m, n};
            for (int64_t i = 0; i < m * n; ++i) {
                const float cv = fc.value()[broadcast_index(static_cast<std::size_t>(i), so, sc)];
                out[static_cast<std::size_t>(i)] = beta * cv;
            }
        }
    }
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0;
            for (int64_t t = 0; t < k; ++t) {
                acc += A[static_cast<std::size_t>(i * k + t)] * B[static_cast<std::size_t>(t * n + j)];
            }
            out[static_cast<std::size_t>(i * n + j)] += alpha * acc;
        }
    }
    return from_f32(graph.find_tensor(node.outputs[0]) ? graph.find_tensor(node.outputs[0])->name : "",
                    shape_from_ints({m, n}), out);
}

Result<RuntimeTensor> exec_conv(const Graph& graph, const Node& node,
                                const std::map<TensorId, RuntimeTensor>& vals) {
    if (node.inputs.size() < 2 || node.outputs.empty()) {
        return error(ErrorCode::InvalidGraph, "Conv missing edges");
    }
    const auto* x = require(vals, node.inputs[0], "x");
    const auto* w = require(vals, node.inputs[1], "w");
    if (!x || !w) {
        return error(ErrorCode::ExecutionFailure, "missing Conv input");
    }
    auto fx = as_f32(*x);
    auto fw = as_f32(*w);
    if (!fx) {
        return fx.error();
    }
    if (!fw) {
        return fw.error();
    }
    auto sx = dims_of(x->shape);
    auto sw = dims_of(w->shape);
    if (sx.size() != 4 || sw.size() != 4) {
        return error(ErrorCode::ExecutionFailure, "reference Conv supports NCHW 4D tensors only");
    }
    const int64_t n = sx[0];
    const int64_t c = sx[1];
    const int64_t h = sx[2];
    const int64_t ww = sx[3];
    const int64_t m = sw[0];
    const int64_t kh = sw[2];
    const int64_t kw = sw[3];
    const int64_t group = attr_int(node, "group", 1);
    if (group < 1 || c % group != 0 || m % group != 0) {
        return error(ErrorCode::ExecutionFailure, "invalid Conv group");
    }
    auto strides = attr_ints(node, "strides");
    auto pads = attr_ints(node, "pads");
    auto dilations = attr_ints(node, "dilations");
    const int64_t sh = strides.size() >= 2 ? strides[0] : (strides.empty() ? 1 : strides[0]);
    const int64_t sws = strides.size() >= 2 ? strides[1] : (strides.empty() ? 1 : strides[0]);
    const int64_t dh = dilations.size() >= 2 ? dilations[0] : 1;
    const int64_t dw = dilations.size() >= 2 ? dilations[1] : 1;
    int64_t pt = 0, pl = 0, pb = 0, pr = 0;
    if (pads.size() >= 4) {
        pt = pads[0];
        pl = pads[1];
        pb = pads[2];
        pr = pads[3];
    } else if (pads.size() == 2) {
        pt = pb = pads[0];
        pl = pr = pads[1];
    }
    const int64_t khd = dh * (kh - 1) + 1;
    const int64_t kwd = dw * (kw - 1) + 1;
    const int64_t oh = (h + pt + pb - khd) / sh + 1;
    const int64_t ow = (ww + pl + pr - kwd) / sws + 1;
    if (oh < 1 || ow < 1) {
        return error(ErrorCode::ExecutionFailure, "Conv produced an empty spatial shape");
    }
    std::vector<float> bias(static_cast<std::size_t>(m), 0.0f);
    if (node.inputs.size() >= 3) {
        const auto* b = require(vals, node.inputs[2], "b");
        if (b) {
            auto fb = as_f32(*b);
            if (!fb) {
                return fb.error();
            }
            for (std::size_t i = 0; i < bias.size() && i < fb.value().size(); ++i) {
                bias[i] = fb.value()[i];
            }
        }
    }
    const int64_t cg = c / group;
    std::vector<float> out(static_cast<std::size_t>(n * m * oh * ow));
    auto atx = [&](int64_t ni, int64_t ci, int64_t hi, int64_t wi) -> float {
        if (hi < 0 || wi < 0 || hi >= h || wi >= ww) {
            return 0.0f;
        }
        return fx.value()[static_cast<std::size_t>(((ni * c + ci) * h + hi) * ww + wi)];
    };
    auto atw = [&](int64_t oc, int64_t ic, int64_t y, int64_t x) -> float {
        return fw.value()[static_cast<std::size_t>(((oc * cg + ic) * kh + y) * kw + x)];
    };
    for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t oc = 0; oc < m; ++oc) {
            const int64_t g = oc / (m / group);
            for (int64_t oy = 0; oy < oh; ++oy) {
                for (int64_t ox = 0; ox < ow; ++ox) {
                    float acc = bias[static_cast<std::size_t>(oc)];
                    for (int64_t ic = 0; ic < cg; ++ic) {
                        const int64_t ci = g * cg + ic;
                        for (int64_t ky = 0; ky < kh; ++ky) {
                            for (int64_t kx = 0; kx < kw; ++kx) {
                                const int64_t ih = oy * sh - pt + ky * dh;
                                const int64_t iw = ox * sws - pl + kx * dw;
                                acc += atx(ni, ci, ih, iw) * atw(oc, ic, ky, kx);
                            }
                        }
                    }
                    out[static_cast<std::size_t>(((ni * m + oc) * oh + oy) * ow + ox)] = acc;
                }
            }
        }
    }
    Shape shape = shape_from_ints({n, m, oh, ow});
    if (const Tensor* t = graph.find_tensor(node.outputs[0]); t && t->shape.is_static() && t->shape.rank() == 4) {
        shape = t->shape;
    }
    return from_f32(graph.find_tensor(node.outputs[0]) ? graph.find_tensor(node.outputs[0])->name : "",
                    std::move(shape), out);
}

Result<RuntimeTensor> exec_softmax(const Graph& graph, const Node& node,
                                   const std::map<TensorId, RuntimeTensor>& vals) {
    if (node.inputs.empty() || node.outputs.empty()) {
        return error(ErrorCode::InvalidGraph, "Softmax missing edges");
    }
    const auto* a = require(vals, node.inputs[0], "x");
    if (!a) {
        return error(ErrorCode::ExecutionFailure, "missing Softmax input");
    }
    auto fa = as_f32(*a);
    if (!fa) {
        return fa.error();
    }
    auto s = dims_of(a->shape);
    if (s.empty()) {
        s.push_back(1);
    }
    int64_t axis = attr_int(node, "axis", -1);
    if (axis < 0) {
        axis += static_cast<int64_t>(s.size());
    }
    if (axis < 0 || axis >= static_cast<int64_t>(s.size())) {
        return error(ErrorCode::ExecutionFailure, "Softmax axis out of range");
    }
    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= s[static_cast<std::size_t>(i)];
    }
    const int64_t axis_n = s[static_cast<std::size_t>(axis)];
    for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < s.size(); ++i) {
        inner *= s[i];
    }
    std::vector<float> out = fa.value();
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            float m = -std::numeric_limits<float>::infinity();
            for (int64_t aidx = 0; aidx < axis_n; ++aidx) {
                const std::size_t idx = static_cast<std::size_t>((o * axis_n + aidx) * inner + i);
                m = std::max(m, out[idx]);
            }
            float sum = 0;
            for (int64_t aidx = 0; aidx < axis_n; ++aidx) {
                const std::size_t idx = static_cast<std::size_t>((o * axis_n + aidx) * inner + i);
                out[idx] = std::exp(out[idx] - m);
                sum += out[idx];
            }
            for (int64_t aidx = 0; aidx < axis_n; ++aidx) {
                const std::size_t idx = static_cast<std::size_t>((o * axis_n + aidx) * inner + i);
                out[idx] = out[idx] / (sum == 0 ? 1.0f : sum);
            }
        }
    }
    return from_f32(graph.find_tensor(node.outputs[0]) ? graph.find_tensor(node.outputs[0])->name : "",
                    a->shape, out);
}

Result<RuntimeTensor> exec_concat(const Graph& graph, const Node& node,
                                  const std::map<TensorId, RuntimeTensor>& vals) {
    if (node.inputs.empty() || node.outputs.empty()) {
        return error(ErrorCode::InvalidGraph, "Concat missing edges");
    }
    std::vector<const RuntimeTensor*> ins;
    std::vector<std::vector<float>> f32s;
    for (TensorId id : node.inputs) {
        const auto* t = require(vals, id, "x");
        if (!t) {
            return error(ErrorCode::ExecutionFailure, "missing Concat input");
        }
        auto f = as_f32(*t);
        if (!f) {
            return f.error();
        }
        ins.push_back(t);
        f32s.push_back(std::move(f.value()));
    }
    auto base = dims_of(ins.front()->shape);
    int64_t axis = attr_int(node, "axis", 0);
    if (axis < 0) {
        axis += static_cast<int64_t>(base.size());
    }
    if (axis < 0 || axis >= static_cast<int64_t>(base.size())) {
        return error(ErrorCode::ExecutionFailure, "Concat axis out of range");
    }
    int64_t axis_sum = 0;
    for (const auto* t : ins) {
        auto d = dims_of(t->shape);
        if (d.size() != base.size()) {
            return error(ErrorCode::ExecutionFailure, "Concat rank mismatch");
        }
        for (std::size_t i = 0; i < d.size(); ++i) {
            if (static_cast<int64_t>(i) != axis && d[i] != base[i]) {
                return error(ErrorCode::ExecutionFailure, "Concat shape mismatch");
            }
        }
        axis_sum += d[static_cast<std::size_t>(axis)];
    }
    auto out_shape = base;
    out_shape[static_cast<std::size_t>(axis)] = axis_sum;
    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= base[static_cast<std::size_t>(i)];
    }
    for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < base.size(); ++i) {
        inner *= base[i];
    }
    std::vector<float> out(static_cast<std::size_t>(numel(out_shape)));
    for (int64_t o = 0; o < outer; ++o) {
        int64_t offset = 0;
        for (std::size_t t = 0; t < ins.size(); ++t) {
            const int64_t ad = dims_of(ins[t]->shape)[static_cast<std::size_t>(axis)];
            for (int64_t aidx = 0; aidx < ad; ++aidx) {
                for (int64_t i = 0; i < inner; ++i) {
                    const std::size_t si = static_cast<std::size_t>((o * ad + aidx) * inner + i);
                    const std::size_t di =
                        static_cast<std::size_t>((o * axis_sum + offset + aidx) * inner + i);
                    out[di] = f32s[t][si];
                }
            }
            offset += ad;
        }
    }
    return from_f32(graph.find_tensor(node.outputs[0]) ? graph.find_tensor(node.outputs[0])->name : "",
                    shape_from_ints(out_shape), out);
}

Result<RuntimeTensor> exec_transpose(const Graph& graph, const Node& node,
                                     const std::map<TensorId, RuntimeTensor>& vals) {
    if (node.inputs.empty() || node.outputs.empty()) {
        return error(ErrorCode::InvalidGraph, "Transpose missing edges");
    }
    const auto* a = require(vals, node.inputs[0], "x");
    if (!a) {
        return error(ErrorCode::ExecutionFailure, "missing Transpose input");
    }
    auto fa = as_f32(*a);
    if (!fa) {
        return fa.error();
    }
    auto s = dims_of(a->shape);
    std::vector<int64_t> perm = attr_ints(node, "perm");
    if (perm.empty()) {
        perm.resize(s.size());
        for (std::size_t i = 0; i < s.size(); ++i) {
            perm[i] = static_cast<int64_t>(s.size() - 1 - i);
        }
    }
    if (perm.size() != s.size()) {
        return error(ErrorCode::ExecutionFailure, "Transpose perm rank mismatch");
    }
    std::vector<int64_t> out_shape(s.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] < 0 || perm[i] >= static_cast<int64_t>(s.size())) {
            return error(ErrorCode::ExecutionFailure, "Transpose perm out of range");
        }
        out_shape[i] = s[static_cast<std::size_t>(perm[i])];
    }
    std::vector<int64_t> in_stride(s.size(), 1);
    for (std::size_t i = s.size(); i-- > 1;) {
        in_stride[i - 1] = in_stride[i] * s[i];
    }
    std::vector<float> out(fa.value().size());
    const uint64_t n = numel(out_shape);
    for (uint64_t oi = 0; oi < n; ++oi) {
        uint64_t rem = oi;
        int64_t ii = 0;
        for (std::size_t d = 0; d < out_shape.size(); ++d) {
            const int64_t od = out_shape[d];
            const int64_t coord = od > 0 ? static_cast<int64_t>(rem % static_cast<uint64_t>(od)) : 0;
            rem = od > 0 ? rem / static_cast<uint64_t>(od) : 0;
            ii += coord * in_stride[static_cast<std::size_t>(perm[d])];
        }
        out[static_cast<std::size_t>(oi)] = fa.value()[static_cast<std::size_t>(ii)];
    }
    return from_f32(graph.find_tensor(node.outputs[0]) ? graph.find_tensor(node.outputs[0])->name : "",
                    shape_from_ints(out_shape), out);
}

Result<RuntimeTensor> exec_reshape(const Graph& graph, const Node& node,
                                   const std::map<TensorId, RuntimeTensor>& vals) {
    if (node.inputs.empty() || node.outputs.empty()) {
        return error(ErrorCode::InvalidGraph, "Reshape missing edges");
    }
    const auto* a = require(vals, node.inputs[0], "x");
    if (!a) {
        return error(ErrorCode::ExecutionFailure, "missing Reshape input");
    }
    std::vector<int64_t> new_shape;
    if (node.inputs.size() >= 2) {
        const auto* sh = require(vals, node.inputs[1], "shape");
        if (sh) {
            auto dims = as_i64(*sh);
            if (!dims) {
                return dims.error();
            }
            new_shape = std::move(dims.value());
        }
    }
    if (new_shape.empty()) {
        if (const Tensor* t = graph.find_tensor(node.outputs[0]); t && t->shape.is_static()) {
            new_shape = dims_of(t->shape);
        } else {
            new_shape = dims_of(a->shape);
        }
    }
    int64_t minus = -1;
    int64_t prod = 1;
    for (std::size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            minus = static_cast<int64_t>(i);
        } else if (new_shape[i] == 0) {
            if (i < a->shape.dims.size() && a->shape.dims[i].value) {
                new_shape[i] = *a->shape.dims[i].value;
            }
            prod *= new_shape[i] == 0 ? 1 : new_shape[i];
        } else {
            prod *= new_shape[i];
        }
    }
    auto n = a->shape.element_count();
    const uint64_t elems = n ? n.value() : static_cast<uint64_t>(a->bytes.size() / std::max<std::size_t>(1, datatype_size(a->dtype)));
    if (minus >= 0) {
        if (prod == 0) {
            return error(ErrorCode::ExecutionFailure, "invalid Reshape");
        }
        new_shape[static_cast<std::size_t>(minus)] = static_cast<int64_t>(elems / static_cast<uint64_t>(prod));
    }
    RuntimeTensor rt = *a;
    rt.shape = shape_from_ints(new_shape);
    if (const Tensor* t = graph.find_tensor(node.outputs[0])) {
        rt.name = t->name;
    }
    return rt;
}

Result<RuntimeTensor> exec_clip(const Graph& graph, const Node& node,
                                const std::map<TensorId, RuntimeTensor>& vals) {
    if (node.inputs.empty() || node.outputs.empty()) {
        return error(ErrorCode::InvalidGraph, "Clip missing edges");
    }
    const auto* a = require(vals, node.inputs[0], "x");
    if (!a) {
        return error(ErrorCode::ExecutionFailure, "missing Clip input");
    }
    auto fa = as_f32(*a);
    if (!fa) {
        return fa.error();
    }
    float lo = static_cast<float>(attr_float(node, "min", -std::numeric_limits<double>::infinity()));
    float hi = static_cast<float>(attr_float(node, "max", std::numeric_limits<double>::infinity()));
    if (node.inputs.size() >= 2) {
        if (const auto* mn = require(vals, node.inputs[1], "min")) {
            auto f = as_f32(*mn);
            if (f && !f.value().empty()) {
                lo = f.value().front();
            }
        }
    }
    if (node.inputs.size() >= 3) {
        if (const auto* mx = require(vals, node.inputs[2], "max")) {
            auto f = as_f32(*mx);
            if (f && !f.value().empty()) {
                hi = f.value().front();
            }
        }
    }
    for (float& v : fa.value()) {
        v = std::min(hi, std::max(lo, v));
    }
    return from_f32(graph.find_tensor(node.outputs[0]) ? graph.find_tensor(node.outputs[0])->name : "",
                    a->shape, fa.value());
}

}  // namespace

bool reference_op_supported(const Node& node) {
    const std::string op = lower_op(node.op_type);
    static const char* k[] = {
        "add",        "sub",     "mul",        "div",     "relu",     "relu6",     "tanh",
        "sigmoid",    "clip",    "softmax",    "identity","dropout",  "reshape",   "squeeze",
        "unsqueeze",  "flatten", "transpose",  "concat",  "concatenation", "matmul", "gemm",
        "conv",       "conv2d",  "convolution","constant","abs",      "neg",       "max",
        "min",        "pow"};
    for (const char* s : k) {
        if (op == s) {
            return true;
        }
    }
    return node.canonical == CanonicalOp::Identity || node.canonical == CanonicalOp::Dropout ||
           node.canonical == CanonicalOp::Constant;
}

Result<RuntimeTensor> reference_exec_node(const Graph& graph, const Node& node,
                                          const std::map<TensorId, RuntimeTensor>& vals) {
    const std::string op = lower_op(node.op_type);
    if (op == "add") {
        return binary_op(graph, node, vals, '+');
    }
    if (op == "sub") {
        return binary_op(graph, node, vals, '-');
    }
    if (op == "mul") {
        return binary_op(graph, node, vals, '*');
    }
    if (op == "div") {
        return binary_op(graph, node, vals, '/');
    }
    if (op == "max") {
        return binary_op(graph, node, vals, 'M');
    }
    if (op == "min") {
        return binary_op(graph, node, vals, 'm');
    }
    if (op == "pow") {
        return binary_op(graph, node, vals, 'p');
    }
    if (op == "relu") {
        return unary_f32(graph, node, vals, [](float v) { return v < 0 ? 0.0f : v; });
    }
    if (op == "relu6") {
        return unary_f32(graph, node, vals, [](float v) { return std::min(6.0f, std::max(0.0f, v)); });
    }
    if (op == "tanh") {
        return unary_f32(graph, node, vals, [](float v) { return std::tanh(v); });
    }
    if (op == "sigmoid") {
        return unary_f32(graph, node, vals, [](float v) { return 1.0f / (1.0f + std::exp(-v)); });
    }
    if (op == "abs") {
        return unary_f32(graph, node, vals, [](float v) { return std::fabs(v); });
    }
    if (op == "neg") {
        return unary_f32(graph, node, vals, [](float v) { return -v; });
    }
    if (op == "clip") {
        return exec_clip(graph, node, vals);
    }
    if (op == "softmax") {
        return exec_softmax(graph, node, vals);
    }
    if (op == "identity" || op == "dropout" || node.canonical == CanonicalOp::Identity ||
        node.canonical == CanonicalOp::Dropout) {
        if (node.inputs.empty() || node.outputs.empty()) {
            return error(ErrorCode::InvalidGraph, "Identity missing edges");
        }
        const auto* a = require(vals, node.inputs[0], "x");
        if (!a) {
            return error(ErrorCode::ExecutionFailure, "missing Identity input");
        }
        RuntimeTensor rt = *a;
        if (const Tensor* t = graph.find_tensor(node.outputs[0])) {
            rt.name = t->name;
        }
        return rt;
    }
    if (op == "reshape" || op == "squeeze" || op == "unsqueeze" || op == "flatten") {
        return exec_reshape(graph, node, vals);
    }
    if (op == "transpose") {
        return exec_transpose(graph, node, vals);
    }
    if (op == "concat" || op == "concatenation") {
        return exec_concat(graph, node, vals);
    }
    if (op == "matmul") {
        return exec_matmul(graph, node, vals);
    }
    if (op == "gemm") {
        return exec_gemm(graph, node, vals);
    }
    if (op == "conv" || op == "conv2d" || op == "convolution") {
        return exec_conv(graph, node, vals);
    }
    if (op == "constant" || node.canonical == CanonicalOp::Constant) {
        auto it = node.attributes.find("value");
        if (it != node.attributes.end() && it->second.kind == Attribute::Kind::Tensor) {
            const Tensor& src = it->second.tensor;
            RuntimeTensor rt;
            rt.name = graph.find_tensor(node.outputs[0]) ? graph.find_tensor(node.outputs[0])->name : src.name;
            rt.dtype = src.dtype;
            rt.shape = src.shape;
            auto bytes = tensor_payload_bytes(src);
            if (!bytes) {
                return bytes.error();
            }
            rt.bytes = std::move(bytes.value());
            return rt;
        }
        if (!node.outputs.empty()) {
            auto vit = vals.find(node.outputs[0]);
            if (vit != vals.end()) {
                return vit->second;
            }
        }
        return error(ErrorCode::ExecutionFailure, "Constant node has no tensor payload");
    }
    return error(ErrorCode::UnsupportedOperator, "unsupported op " + node.op_type);
}

}  // namespace nn
