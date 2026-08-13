#include "runtime/tensor_io.h"

#include "nn/analysis.h"
#include "nn/json.h"
#include "nn/mmap.h"
#include "util/npy.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>

namespace nn {
namespace {

std::string lower_ext(const std::filesystem::path& path) {
    std::string e = path.extension().string();
    std::transform(e.begin(), e.end(), e.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return e;
}

Result<RuntimeTensor> load_csv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        return error(ErrorCode::FileNotFound, "cannot open " + path.string());
    }
    std::vector<std::vector<float>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<float> row;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            row.push_back(static_cast<float>(std::strtod(tok.c_str(), nullptr)));
        }
        if (!row.empty()) {
            rows.push_back(std::move(row));
        }
    }
    if (rows.empty()) {
        return error(ErrorCode::InvalidFormat, "empty CSV: " + path.string());
    }
    const std::size_t cols = rows.front().size();
    for (const auto& r : rows) {
        if (r.size() != cols) {
            return error(ErrorCode::InvalidFormat, "ragged CSV: " + path.string());
        }
    }
    RuntimeTensor t;
    t.name = path.stem().string();
    t.dtype = DataType::Float32;
    if (rows.size() == 1) {
        t.shape = shape_from_ints({static_cast<int64_t>(cols)});
    } else {
        t.shape = shape_from_ints({static_cast<int64_t>(rows.size()), static_cast<int64_t>(cols)});
    }
    t.bytes.resize(rows.size() * cols * sizeof(float));
    float* p = reinterpret_cast<float*>(t.bytes.data());
    std::size_t i = 0;
    for (const auto& r : rows) {
        for (float v : r) {
            p[i++] = v;
        }
    }
    return t;
}

Result<RuntimeTensor> load_raw(const std::filesystem::path& path) {
    auto mapped = MappedFile::open(path);
    if (!mapped) {
        return mapped.error();
    }
    RuntimeTensor t;
    t.name = path.stem().string();
    t.dtype = DataType::UInt8;
    t.shape = shape_from_ints({static_cast<int64_t>(mapped.value().size())});
    auto sp = mapped.value().span();
    t.bytes.assign(sp.begin(), sp.end());
    return t;
}

uint32_t splitmix(uint64_t& s) {
    s += 0x9e3779b97f4a7c15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<uint32_t>(z ^ (z >> 31));
}

}  // namespace

Result<RuntimeTensor> load_tensor_file(const std::filesystem::path& path) {
    const std::string ext = lower_ext(path);
    if (ext == ".npy") {
        return load_npy(path);
    }
    if (ext == ".npz") {
        auto arrays = load_npz(path);
        if (!arrays) {
            return arrays.error();
        }
        if (arrays.value().size() != 1) {
            return error(ErrorCode::InvalidArgument,
                         "npz '" + path.string() + "' has " +
                             std::to_string(arrays.value().size()) +
                             " arrays; use --input NAME=" + path.string() +
                             " or a single-array npz");
        }
        return arrays.value().begin()->second;
    }
    if (ext == ".csv" || ext == ".txt") {
        return load_csv(path);
    }
    if (ext == ".raw" || ext == ".bin") {
        return load_raw(path);
    }
    auto npy = load_npy(path);
    if (npy) {
        return npy;
    }
    return load_csv(path);
}

Status save_tensor_file(const std::filesystem::path& path, const RuntimeTensor& tensor) {
    const std::string ext = lower_ext(path);
    if (ext == ".csv" || ext == ".txt") {
        auto vals = as_f64(tensor);
        std::ofstream out(path);
        if (!out) {
            return Status::err(error(ErrorCode::FileError, "cannot write " + path.string()));
        }
        int64_t cols = 1;
        if (tensor.shape.rank() >= 1 && tensor.shape.dims.back().value) {
            cols = *tensor.shape.dims.back().value;
        }
        if (cols < 1) {
            cols = 1;
        }
        for (std::size_t i = 0; i < vals.size(); ++i) {
            if (i && (i % static_cast<std::size_t>(cols) == 0)) {
                out << '\n';
            } else if (i) {
                out << ',';
            }
            out << vals[i];
        }
        out << '\n';
        return Status::ok();
    }
    if (ext == ".raw" || ext == ".bin") {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            return Status::err(error(ErrorCode::FileError, "cannot write " + path.string()));
        }
        if (!tensor.bytes.empty()) {
            out.write(reinterpret_cast<const char*>(tensor.bytes.data()),
                      static_cast<std::streamsize>(tensor.bytes.size()));
        }
        return Status::ok();
    }
    return save_npy(path, tensor);
}

Result<RuntimeTensor> runtime_from_ir_tensor(const Tensor& t) {
    RuntimeTensor rt;
    rt.name = t.name;
    rt.dtype = t.dtype;
    rt.shape = t.shape;
    auto bytes = tensor_payload_bytes(t);
    if (!bytes) {
        return bytes.error();
    }
    rt.bytes = std::move(bytes.value());
    return rt;
}

RuntimeTensor make_input_tensor(const Tensor& spec, uint64_t seed, bool random) {
    RuntimeTensor rt;
    rt.name = spec.name;
    rt.dtype = spec.dtype == DataType::Unknown ? DataType::Float32 : spec.dtype;
    rt.shape = spec.shape;
    uint64_t nbytes = spec.storage_bytes;
    if (nbytes == 0) {
        auto b = tensor_storage_bytes(spec);
        nbytes = b ? b.value() : 0;
        if (nbytes == 0 && spec.shape.is_static()) {
            auto n = spec.shape.element_count();
            const std::size_t es = datatype_size(rt.dtype);
            nbytes = n ? n.value() * es : 0;
        }
    }
    rt.bytes.assign(static_cast<std::size_t>(nbytes), 0);
    if (random && rt.dtype == DataType::Float32) {
        uint64_t s = seed ? seed : 1;
        float* p = reinterpret_cast<float*>(rt.bytes.data());
        const std::size_t n = rt.bytes.size() / sizeof(float);
        for (std::size_t i = 0; i < n; ++i) {
            const uint32_t u = splitmix(s);
            p[i] = static_cast<float>(u) / static_cast<float>(UINT32_MAX) * 2.0f - 1.0f;
        }
    }
    return rt;
}

std::string normalize_tensor_name(std::string_view name) {
    std::string s(name);
    auto slash = s.find_last_of("/\\");
    if (slash != std::string::npos) {
        s = s.substr(slash + 1);
    }
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    constexpr const char* prefixes[] = {"serving_default_", "statefulpartitionedcall_"};
    for (const char* p : prefixes) {
        const std::size_t n = std::strlen(p);
        if (s.size() > n && s.compare(0, n, p) == 0) {
            s.erase(0, n);
            break;
        }
    }
    const auto colon = s.rfind(':');
    if (colon != std::string::npos && colon + 1 < s.size()) {
        bool digits = true;
        for (std::size_t i = colon + 1; i < s.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
                digits = false;
                break;
            }
        }
        if (digits) {
            s.resize(colon);
        }
    }
    std::string alnum;
    alnum.reserve(s.size());
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            alnum.push_back(c);
        }
    }
    return alnum.empty() ? s : alnum;
}

namespace {

void adapt_loaded_to_spec(RuntimeTensor& loaded, const Graph& g, const std::string& name,
                          const std::filesystem::path& path) {
    if (const Tensor* spec_t = g.find_tensor_by_name(name)) {
        const std::string ext = lower_ext(path);
        if (loaded.dtype == DataType::UInt8 && spec_t->dtype == DataType::Float32 &&
            ext != ".npy" && ext != ".npz") {
            auto want = tensor_storage_bytes(*spec_t);
            if (want && want.value() == loaded.bytes.size()) {
                loaded.dtype = spec_t->dtype;
                loaded.shape = spec_t->shape;
            }
        } else if (loaded.shape.dims.empty() && spec_t->shape.is_static()) {
            loaded.shape = spec_t->shape;
            loaded.dtype = spec_t->dtype;
        }
    }
}

const Tensor* find_unbound_input(const std::vector<const Tensor*>& inputs,
                                 const std::map<std::string, RuntimeTensor>& out,
                                 std::string_view key) {
    const std::string nk = normalize_tensor_name(key);
    for (const auto* t : inputs) {
        if (out.count(t->name)) {
            continue;
        }
        if (t->name == key || normalize_tensor_name(t->name) == nk) {
            return t;
        }
    }
    return nullptr;
}

}  // namespace

Result<std::map<std::string, RuntimeTensor>> bind_model_inputs(
    const ModelIR& model, const std::vector<std::string>& specs, uint64_t seed, bool random_fill) {
    const Graph* g = primary_graph(model);
    if (!g) {
        return error(ErrorCode::InvalidGraph, "no graph");
    }
    std::map<std::string, RuntimeTensor> out;
    std::vector<const Tensor*> inputs;
    for (TensorId id : g->inputs) {
        if (const Tensor* t = g->find_tensor(id)) {
            inputs.push_back(t);
        }
    }

    auto store = [&](std::string name, RuntimeTensor loaded, const std::filesystem::path& path) {
        loaded.name = name;
        adapt_loaded_to_spec(loaded, *g, name, path);
        out[name] = std::move(loaded);
    };

    for (const auto& spec : specs) {
        std::string name;
        std::filesystem::path path;
        const auto eq = spec.find('=');
        if (eq != std::string::npos) {
            name = spec.substr(0, eq);
            path = spec.substr(eq + 1);
        } else {
            path = spec;
        }
        if (lower_ext(path) == ".npz") {
            auto arrays = load_npz(path);
            if (!arrays) {
                return arrays.error();
            }
            if (!name.empty()) {
                auto it = arrays.value().find(name);
                if (it == arrays.value().end()) {
                    const std::string nk = normalize_tensor_name(name);
                    for (auto& kv : arrays.value()) {
                        if (normalize_tensor_name(kv.first) == nk) {
                            it = arrays.value().find(kv.first);
                            break;
                        }
                    }
                }
                if (it == arrays.value().end() && arrays.value().size() == 1) {
                    it = arrays.value().begin();
                }
                if (it == arrays.value().end()) {
                    return error(ErrorCode::MissingArgument,
                                 "npz has no array named '" + name + "'");
                }
                store(name, std::move(it->second), path);
                continue;
            }
            std::vector<std::pair<std::string, RuntimeTensor>> leftover;
            leftover.reserve(arrays.value().size());
            for (auto& kv : arrays.value()) {
                if (const Tensor* t = find_unbound_input(inputs, out, kv.first)) {
                    store(t->name, std::move(kv.second), path);
                } else {
                    leftover.emplace_back(kv.first, std::move(kv.second));
                }
            }
            std::vector<const Tensor*> unbound;
            for (const auto* t : inputs) {
                if (!out.count(t->name)) {
                    unbound.push_back(t);
                }
            }
            if (leftover.size() == 1 && unbound.size() == 1) {
                store(unbound.front()->name, std::move(leftover.front().second), path);
            } else if (leftover.size() == unbound.size() && !leftover.empty()) {
                for (std::size_t i = 0; i < leftover.size(); ++i) {
                    store(unbound[i]->name, std::move(leftover[i].second), path);
                }
            } else if (!leftover.empty() && !unbound.empty()) {
                return error(ErrorCode::InvalidArgument,
                             "could not match npz arrays to inputs in " + path.string());
            }
            continue;
        }

        auto loaded = load_tensor_file(path);
        if (!loaded) {
            return loaded.error();
        }
        if (name.empty()) {
            if (inputs.size() == 1) {
                name = inputs.front()->name;
            } else {
                const std::string stem = path.stem().string();
                if (const Tensor* t = find_unbound_input(inputs, out, stem)) {
                    name = t->name;
                } else {
                    for (const auto* in : inputs) {
                        if (!out.count(in->name)) {
                            name = in->name;
                            break;
                        }
                    }
                }
                if (name.empty()) {
                    name = stem;
                }
            }
        }
        store(std::move(name), std::move(loaded.value()), path);
    }
    for (const auto* t : inputs) {
        if (out.count(t->name)) {
            continue;
        }
        if (!t->shape.is_static()) {
            return error(ErrorCode::MissingArgument,
                         "missing input '" + t->name + "' and shape is not static");
        }
        out[t->name] = make_input_tensor(*t, seed + static_cast<uint64_t>(t->id), random_fill);
    }
    return out;
}

std::vector<double> as_f64(const RuntimeTensor& t) {
    std::vector<double> v;
    if (t.dtype == DataType::Float32) {
        const std::size_t n = t.bytes.size() / sizeof(float);
        v.resize(n);
        const auto* p = reinterpret_cast<const float*>(t.bytes.data());
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = p[i];
        }
    } else if (t.dtype == DataType::Float64) {
        const std::size_t n = t.bytes.size() / sizeof(double);
        v.resize(n);
        const auto* p = reinterpret_cast<const double*>(t.bytes.data());
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = p[i];
        }
    } else if (t.dtype == DataType::Int32) {
        const std::size_t n = t.bytes.size() / sizeof(int32_t);
        v.resize(n);
        const auto* p = reinterpret_cast<const int32_t*>(t.bytes.data());
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = p[i];
        }
    } else if (t.dtype == DataType::Int64) {
        const std::size_t n = t.bytes.size() / sizeof(int64_t);
        v.resize(n);
        const auto* p = reinterpret_cast<const int64_t*>(t.bytes.data());
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = static_cast<double>(p[i]);
        }
    } else if (t.dtype == DataType::UInt8) {
        v.resize(t.bytes.size());
        for (std::size_t i = 0; i < t.bytes.size(); ++i) {
            v[i] = t.bytes[i];
        }
    }
    return v;
}

NumericCompare compare_numeric(const RuntimeTensor& a, const RuntimeTensor& b, double atol,
                               double rtol) {
    NumericCompare c;
    auto fa = as_f64(a);
    auto fb = as_f64(b);
    if (fa.empty() || fb.empty()) {
        c.note = "one or both tensors have a non-numeric dtype";
        return c;
    }
    if (fa.size() != fb.size()) {
        c.note = "element counts differ (" + std::to_string(fa.size()) + " vs " +
                 std::to_string(fb.size()) + ")";
        return c;
    }
    c.comparable = true;
    c.total = fa.size();
    double sum_abs = 0;
    double sum_sq = 0;
    double dot = 0;
    double na = 0;
    double nb = 0;
    for (std::size_t i = 0; i < fa.size(); ++i) {
        const double d = std::fabs(fa[i] - fb[i]);
        const double tol = atol + rtol * std::fabs(fb[i]);
        if (d > tol) {
            ++c.changed;
        }
        if (d > c.max_abs) {
            c.max_abs = d;
        }
        sum_abs += d;
        sum_sq += d * d;
        dot += fa[i] * fb[i];
        na += fa[i] * fa[i];
        nb += fb[i] * fb[i];
    }
    c.mean_abs = sum_abs / static_cast<double>(fa.size());
    c.rmse = std::sqrt(sum_sq / static_cast<double>(fa.size()));
    const double denom = std::sqrt(na) * std::sqrt(nb);
    c.cosine = denom == 0 ? 1.0 : dot / denom;
    return c;
}

namespace {

std::optional<int64_t> dim_at(const Shape& s, std::size_t i) {
    if (i >= s.dims.size() || !s.dims[i].value) {
        return std::nullopt;
    }
    return *s.dims[i].value;
}

bool nchw_nhwc_pair(const Shape& nchw, const Shape& nhwc) {
    auto n = dim_at(nchw, 0);
    auto c = dim_at(nchw, 1);
    auto h = dim_at(nchw, 2);
    auto w = dim_at(nchw, 3);
    auto n2 = dim_at(nhwc, 0);
    auto h2 = dim_at(nhwc, 1);
    auto w2 = dim_at(nhwc, 2);
    auto c2 = dim_at(nhwc, 3);
    return n && c && h && w && n2 && h2 && w2 && c2 && *n == *n2 && *c == *c2 && *h == *h2 &&
           *w == *w2;
}

Result<RuntimeTensor> permute_nchw_nhwc(const RuntimeTensor& src, bool to_nhwc) {
    if (src.shape.rank() != 4) {
        return error(ErrorCode::InvalidArgument, "layout permute requires rank 4");
    }
    auto n = dim_at(src.shape, 0);
    auto d1 = dim_at(src.shape, 1);
    auto d2 = dim_at(src.shape, 2);
    auto d3 = dim_at(src.shape, 3);
    if (!n || !d1 || !d2 || !d3) {
        return error(ErrorCode::InvalidArgument, "layout permute requires static shape");
    }
    const std::size_t es = datatype_size(src.dtype);
    if (es == 0) {
        return error(ErrorCode::InvalidArgument, "layout permute unsupported dtype");
    }
    const int64_t N = *n;
    int64_t C = 0, H = 0, W = 0;
    if (to_nhwc) {
        C = *d1;
        H = *d2;
        W = *d3;
    } else {
        H = *d1;
        W = *d2;
        C = *d3;
    }
    RuntimeTensor out;
    out.name = src.name;
    out.dtype = src.dtype;
    out.shape = to_nhwc ? shape_from_ints({N, H, W, C}) : shape_from_ints({N, C, H, W});
    out.bytes.resize(src.bytes.size());
    const auto* in = src.bytes.data();
    auto* dst = out.bytes.data();
    for (int64_t ni = 0; ni < N; ++ni) {
        for (int64_t ci = 0; ci < C; ++ci) {
            for (int64_t hi = 0; hi < H; ++hi) {
                for (int64_t wi = 0; wi < W; ++wi) {
                    const int64_t src_i =
                        to_nhwc ? (((ni * C + ci) * H + hi) * W + wi)
                                : (((ni * H + hi) * W + wi) * C + ci);
                    const int64_t dst_i =
                        to_nhwc ? (((ni * H + hi) * W + wi) * C + ci)
                                : (((ni * C + ci) * H + hi) * W + wi);
                    std::memcpy(dst + static_cast<std::size_t>(dst_i) * es,
                                in + static_cast<std::size_t>(src_i) * es, es);
                }
            }
        }
    }
    return out;
}

int match_score(std::string_view a, std::string_view b) {
    if (a == b) {
        return 3;
    }
    const std::string na = normalize_tensor_name(a);
    const std::string nb = normalize_tensor_name(b);
    if (!na.empty() && na == nb) {
        return 2;
    }
    if (!na.empty() && !nb.empty() && (na.find(nb) != std::string::npos || nb.find(na) != std::string::npos)) {
        return 1;
    }
    return 0;
}

}  // namespace

NumericCompare compare_numeric_aligned(const RuntimeTensor& a, const RuntimeTensor& b, double atol,
                                       double rtol) {
    if (a.shape.rank() == 4 && b.shape.rank() == 4 && a.shape != b.shape) {
        if (nchw_nhwc_pair(a.shape, b.shape)) {
            auto perm = permute_nchw_nhwc(b, false);
            if (perm) {
                auto c = compare_numeric(a, perm.value(), atol, rtol);
                if (c.comparable && c.note.empty()) {
                    c.note = "aligned NHWC to NCHW";
                }
                return c;
            }
        }
        if (nchw_nhwc_pair(b.shape, a.shape)) {
            auto perm = permute_nchw_nhwc(a, false);
            if (perm) {
                auto c = compare_numeric(perm.value(), b, atol, rtol);
                if (c.comparable && c.note.empty()) {
                    c.note = "aligned NHWC to NCHW";
                }
                return c;
            }
        }
    }
    return compare_numeric(a, b, atol, rtol);
}

std::vector<AlignedTensorPair> align_runtime_tensors(const std::vector<RuntimeTensor>& a,
                                                     const std::vector<RuntimeTensor>& b) {
    std::vector<AlignedTensorPair> out;
    std::vector<bool> used(b.size(), false);
    for (std::size_t i = 0; i < a.size(); ++i) {
        AlignedTensorPair p;
        p.a = &a[i];
        p.label = a[i].name.empty() ? ("output" + std::to_string(i)) : a[i].name;
        int best = -1;
        int best_score = 0;
        for (std::size_t j = 0; j < b.size(); ++j) {
            if (used[j]) {
                continue;
            }
            const int s = match_score(a[i].name, b[j].name);
            if (s > best_score) {
                best_score = s;
                best = static_cast<int>(j);
            }
        }
        if (best < 0 && i < b.size() && !used[i]) {
            best = static_cast<int>(i);
        }
        if (best >= 0) {
            used[static_cast<std::size_t>(best)] = true;
            p.b = &b[static_cast<std::size_t>(best)];
            if (p.label.empty()) {
                p.label = b[static_cast<std::size_t>(best)].name;
            }
            if (best_score == 0 && p.a->name != p.b->name) {
                p.label = p.a->name + " ~ " + p.b->name;
            }
        }
        out.push_back(p);
    }
    for (std::size_t j = 0; j < b.size(); ++j) {
        if (used[j]) {
            continue;
        }
        AlignedTensorPair p;
        p.b = &b[j];
        p.label = b[j].name.empty() ? ("output" + std::to_string(j)) : b[j].name;
        out.push_back(p);
    }
    return out;
}

std::vector<AlignedTensorPair> align_activation_dumps(const ModelIR& model_a, const RunResult& ra,
                                                      const ModelIR& model_b, const RunResult& rb) {
    std::vector<AlignedTensorPair> out;
    std::map<std::string, bool> used_a;
    std::map<std::string, bool> used_b;
    const Graph* ga = primary_graph(model_a);
    const Graph* gb = primary_graph(model_b);
    if (ga && gb) {
        for (const auto& m : match_graphs(*ga, *gb)) {
            if (!m.a || !m.b || m.a->outputs.empty() || m.b->outputs.empty()) {
                continue;
            }
            const Tensor* ta = ga->find_tensor(m.a->outputs[0]);
            const Tensor* tb = gb->find_tensor(m.b->outputs[0]);
            if (!ta || !tb) {
                continue;
            }
            auto ia = ra.dumps.find(ta->name);
            auto ib = rb.dumps.find(tb->name);
            if (ia == ra.dumps.end() || ib == rb.dumps.end()) {
                continue;
            }
            AlignedTensorPair p;
            p.label = std::string(canonical_op_name(m.a->canonical)) + " " + ta->name;
            if (ta->name != tb->name) {
                p.label += " ~ " + tb->name;
            }
            p.a = &ia->second;
            p.b = &ib->second;
            used_b[tb->name] = true;
            used_a[ta->name] = true;
            out.push_back(p);
        }
    }
    std::vector<RuntimeTensor> leftover_a;
    std::vector<RuntimeTensor> leftover_b;
    leftover_a.reserve(ra.dumps.size());
    leftover_b.reserve(rb.dumps.size());
    for (const auto& [name, t] : ra.dumps) {
        if (!used_a.count(name) && !used_a.count(t.name)) {
            leftover_a.push_back(t);
            leftover_a.back().name = name;
        }
    }
    for (const auto& [name, t] : rb.dumps) {
        if (!used_b.count(name)) {
            leftover_b.push_back(t);
            leftover_b.back().name = name;
        }
    }
    auto rest = align_runtime_tensors(leftover_a, leftover_b);
    for (const auto& p : rest) {
        if (!p.a || !p.b) {
            continue;
        }
        AlignedTensorPair q;
        q.label = p.label;
        auto ia = ra.dumps.find(p.a->name);
        auto ib = rb.dumps.find(p.b->name);
        if (ia == ra.dumps.end() || ib == rb.dumps.end()) {
            continue;
        }
        q.a = &ia->second;
        q.b = &ib->second;
        out.push_back(q);
    }
    return out;
}

RuntimeBackend* select_backend(const ModelIR& model, std::string_view name) {
    RuntimeRegistry& reg = default_runtime_registry();
    if (!name.empty()) {
        return reg.find(name);
    }
    return reg.default_backend(model);
}

RuntimeBackend* select_backend(const ModelIR& model, std::string_view name,
                               const RuntimeOptions& options) {
    RuntimeRegistry& reg = default_runtime_registry();
    if (!name.empty()) {
        return reg.find(name);
    }
    if (options.dump_all || !options.dump_names.empty()) {
        if (auto* ref = reg.find("reference"); ref && ref->available() && ref->supports(model)) {
            return ref;
        }
    }
    return reg.default_backend(model);
}

Result<RunResult> eval_model(const ModelIR& model,
                             const std::map<std::string, RuntimeTensor>& inputs,
                             const RuntimeOptions& options, std::string_view backend_name) {
    RuntimeBackend* be = select_backend(model, backend_name, options);
    if (!be || !be->available()) {
        return error(ErrorCode::BackendUnavailable, "backend unavailable");
    }
    if (!be->supports(model)) {
        return error(ErrorCode::UnsupportedOperator,
                     "backend '" + be->name() + "' does not support this model");
    }
    auto sess = be->create_session(model, options);
    if (!sess) {
        return sess.error();
    }
    return sess.value()->run(inputs);
}

Result<std::vector<TestCase>> load_test_manifest(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::path manifest = path;
    std::filesystem::path root = path.parent_path();
    if (std::filesystem::is_directory(path, ec)) {
        root = path;
        if (std::filesystem::exists(path / "manifest.json", ec)) {
            manifest = path / "manifest.json";
        } else {
            TestCase tc;
            tc.name = "default";
            if (std::filesystem::exists(path / "input.npy", ec)) {
                tc.inputs["input"] = (path / "input.npy").string();
            }
            if (std::filesystem::exists(path / "expected.npy", ec)) {
                tc.expected["output"] = (path / "expected.npy").string();
            }
            if (tc.inputs.empty()) {
                return error(ErrorCode::FileNotFound, "no manifest.json or input.npy in " + path.string());
            }
            return std::vector<TestCase>{std::move(tc)};
        }
    }
    std::ifstream in(manifest);
    if (!in) {
        return error(ErrorCode::FileNotFound, "cannot open " + manifest.string());
    }
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto j = parse_json(s);
    if (!j) {
        return j.error();
    }
    const Json* tests = nullptr;
    if (j.value().contains("tests") && j.value().at("tests").is_array()) {
        tests = &j.value().at("tests");
    }
    std::vector<TestCase> out;
    auto resolve = [&](const std::string& p) {
        std::filesystem::path fp(p);
        if (fp.is_absolute()) {
            return fp.string();
        }
        return (root / fp).string();
    };
    auto fill_map = [&](const Json& obj, std::map<std::string, std::string>& dest) {
        if (!obj.is_object()) {
            return;
        }
        for (const auto& [k, v] : obj.as_object()) {
            if (v.is_string()) {
                dest[k] = resolve(v.as_string());
            }
        }
    };
    if (tests) {
        int i = 0;
        for (const auto& t : tests->as_array()) {
            TestCase tc;
            tc.name = t.contains("name") && t.at("name").is_string()
                          ? t.at("name").as_string()
                          : ("test" + std::to_string(i));
            if (t.contains("inputs")) {
                fill_map(t.at("inputs"), tc.inputs);
            }
            if (t.contains("input") && t.at("input").is_string()) {
                tc.inputs["input"] = resolve(t.at("input").as_string());
            }
            if (t.contains("expected")) {
                fill_map(t.at("expected"), tc.expected);
            }
            if (t.contains("outputs")) {
                fill_map(t.at("outputs"), tc.expected);
            }
            out.push_back(std::move(tc));
            ++i;
        }
    }
    if (out.empty()) {
        return error(ErrorCode::InvalidFormat, "manifest missing tests array");
    }
    return out;
}

}  // namespace nn
