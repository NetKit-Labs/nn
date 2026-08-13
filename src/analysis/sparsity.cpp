#include "nn/analysis.h"

#include <cmath>
#include <cstring>
#include <span>

namespace nn {
namespace {

template <typename T>
void count_zeros(std::span<const uint8_t> bytes, double threshold, TensorSparsity& ts) {
    const std::size_t n = bytes.size() / sizeof(T);
    ts.elements = n;
    ts.computed = true;
    const T* p = reinterpret_cast<const T*>(bytes.data());
    for (std::size_t i = 0; i < n; ++i) {
        const double v = static_cast<double>(p[i]);
        if (v == 0.0) {
            ++ts.zeros;
        }
        if (std::fabs(v) <= threshold) {
            ++ts.near_zeros;
        }
    }
    ts.zero_fraction = n ? static_cast<double>(ts.zeros) / static_cast<double>(n) : 0.0;
}

std::span<const uint8_t> tensor_bytes(const Tensor& t) {
    if (!t.data) {
        return {};
    }
    if (t.data->owned) {
        return std::span<const uint8_t>(t.data->owned->data(), t.data->owned->size());
    }
    return {};
}

}  // namespace

SparsityReport analyze_sparsity(const ModelIR& model, const SparsityOptions& options) {
    SparsityReport r;
    const Graph* g = primary_graph(model);
    if (!g) {
        return r;
    }
    for (const auto& t : g->tensors) {
        if (!t.constant) {
            continue;
        }
        ++r.tensors_considered;
        TensorSparsity ts;
        ts.name = t.name;
        auto bytes = tensor_bytes(t);
        if (bytes.empty()) {
            ts.computed = false;
            r.tensors.push_back(ts);
            r.notes.push_back("weights not in memory for " + t.name + "; sparsity not computed");
            continue;
        }
        if (t.dtype == DataType::Float32) {
            count_zeros<float>(bytes, options.threshold, ts);
        } else if (t.dtype == DataType::Float64) {
            count_zeros<double>(bytes, options.threshold, ts);
        } else if (t.dtype == DataType::Int8) {
            count_zeros<int8_t>(bytes, options.threshold, ts);
        } else if (t.dtype == DataType::Int32) {
            count_zeros<int32_t>(bytes, options.threshold, ts);
        } else {
            ts.computed = false;
            r.notes.push_back("unsupported dtype for sparsity on " + t.name);
            r.tensors.push_back(ts);
            continue;
        }
        ++r.tensors_computed;
        r.total_elements += ts.elements;
        r.total_zeros += ts.zeros;
        r.tensors.push_back(ts);
    }
    r.overall_zero_fraction =
        r.total_elements ? static_cast<double>(r.total_zeros) / static_cast<double>(r.total_elements)
                         : 0.0;
    return r;
}

}  // namespace nn
