#include "nn/diff.h"

#include "nn/analysis.h"
#include "nn/hash.h"
#include "nn/mmap.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

namespace nn {
namespace {

uint64_t param_count(const Graph& g) {
    uint64_t n = 0;
    for (const auto& t : g.tensors) {
        if (!t.constant) {
            continue;
        }
        if (auto e = t.shape.element_count()) {
            n += e.value();
        }
    }
    return n;
}

uint64_t weight_bytes(const Graph& g) {
    uint64_t n = 0;
    for (const auto& t : g.tensors) {
        if (t.constant) {
            n += t.storage_bytes ? t.storage_bytes : tensor_storage_bytes(t).value_or(0);
        }
    }
    return n;
}

std::string shape_of(const Graph& g, TensorId id) {
    if (const Tensor* t = g.find_tensor(id)) {
        return t->shape.to_string() + " " + datatype_name(t->dtype);
    }
    return "?";
}

WeightDiffStats compare_weights(const Tensor& a, const Tensor& b, double atol, double rtol) {
    WeightDiffStats s;
    s.name = a.name;
    if (a.dtype != DataType::Float32 || b.dtype != DataType::Float32) {
        s.note = "numeric weight diff implemented for float32 only";
        return s;
    }
    auto bytes_of = [](const Tensor& t) -> std::vector<uint8_t> {
        if (t.data && t.data->owned) {
            return *t.data->owned;
        }
        if (t.data && !t.data->file.empty()) {
            auto mapped = MappedFile::open(t.data->file);
            if (!mapped) {
                return {};
            }
            auto sl = mapped.value().slice(t.data->offset, t.data->length);
            return {sl.begin(), sl.end()};
        }
        return {};
    };
    const auto ba = bytes_of(a);
    const auto bb = bytes_of(b);
    const std::size_t na = ba.size() / sizeof(float);
    const std::size_t nb = bb.size() / sizeof(float);
    const std::size_t n = std::min(na, nb);
    if (n == 0) {
        s.note = "weight payload unavailable";
        return s;
    }
    const float* pa = reinterpret_cast<const float*>(ba.data());
    const float* pb = reinterpret_cast<const float*>(bb.data());
    double sum_abs = 0;
    double sum_sq = 0;
    double dot = 0;
    double na2 = 0;
    double nb2 = 0;
    double max_abs = 0;
    uint64_t changed = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double da = static_cast<double>(pa[i]);
        const double db = static_cast<double>(pb[i]);
        const double d = std::fabs(da - db);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        sum_sq += d * d;
        dot += da * db;
        na2 += da * da;
        nb2 += db * db;
        const double tol = atol + rtol * std::fabs(db);
        if (d > tol) {
            ++changed;
        }
    }
    s.compared = true;
    s.total_elements = n;
    s.changed_elements = changed;
    s.changed_fraction = static_cast<double>(changed) / static_cast<double>(n);
    s.max_abs = max_abs;
    s.mean_abs = sum_abs / static_cast<double>(n);
    s.rmse = std::sqrt(sum_sq / static_cast<double>(n));
    const double denom = std::sqrt(na2) * std::sqrt(nb2);
    s.cosine = denom > 0 ? dot / denom : 0.0;
    s.relative = (std::sqrt(nb2) > 0) ? s.rmse / (std::sqrt(nb2) / std::sqrt(static_cast<double>(n)))
                                      : 0.0;
    return s;
}

}  // namespace

Result<DiffReport> diff_models(const ModelIR& old_model, const ModelIR& new_model,
                               const DiffOptions& options) {
    DiffReport r;
    const Graph* ga = primary_graph(old_model);
    const Graph* gb = primary_graph(new_model);
    if (!ga || !gb) {
        r.identical = false;
        r.notes.push_back("one or both models have no graph");
        return r;
    }
    r.old_nodes = ga->nodes.size();
    r.new_nodes = gb->nodes.size();
    r.old_parameters = param_count(*ga);
    r.new_parameters = param_count(*gb);
    r.old_weight_bytes = weight_bytes(*ga);
    r.new_weight_bytes = weight_bytes(*gb);
    const auto ca = analyze_compute(old_model);
    const auto cb = analyze_compute(new_model);
    r.old_macs = ca.total.macs;
    r.new_macs = cb.total.macs;
    const auto ma = analyze_memory(old_model);
    const auto mb = analyze_memory(new_model);
    r.old_peak_activations = ma.peak_activation_bytes;
    r.new_peak_activations = mb.peak_activation_bytes;
    r.peak_known = ma.peak_known && mb.peak_known;

    auto io_list = [](const Graph& g, const std::vector<TensorId>& ids) {
        std::vector<std::string> s;
        for (TensorId id : ids) {
            if (const Tensor* t = g.find_tensor(id)) {
                s.push_back(t->name + ":" + t->shape.to_string() + ":" + datatype_name(t->dtype));
            }
        }
        return s;
    };
    if (io_list(*ga, ga->inputs) != io_list(*gb, gb->inputs)) {
        r.identical = false;
        ArchitectureChange c;
        c.subject = "inputs";
        c.kind = "modified";
        c.detail = "input contract changed";
        r.io.push_back(c);
    }
    if (io_list(*ga, ga->outputs) != io_list(*gb, gb->outputs)) {
        r.identical = false;
        ArchitectureChange c;
        c.subject = "outputs";
        c.kind = "modified";
        c.detail = "output contract changed";
        r.io.push_back(c);
    }

    std::map<std::string, const Node*> b_by_name;
    for (const auto& n : gb->nodes) {
        if (!n.name.empty()) {
            b_by_name[n.name] = &n;
        }
    }
    std::vector<bool> used(gb->nodes.size(), false);
    for (const auto& n : ga->nodes) {
        const Node* m = nullptr;
        if (!n.name.empty()) {
            auto it = b_by_name.find(n.name);
            if (it != b_by_name.end()) {
                m = it->second;
            }
        }
        if (m) {
            const auto idx = static_cast<std::size_t>(m - gb->nodes.data());
            if (idx < used.size()) {
                used[idx] = true;
            }
            if (m->op_type != n.op_type) {
                r.identical = false;
                ArchitectureChange c;
                c.subject = n.name;
                c.kind = "modified";
                c.detail = n.op_type + " -> " + m->op_type;
                r.architecture.push_back(c);
            }
            auto ashape = n.outputs.empty() ? std::string() : shape_of(*ga, n.outputs[0]);
            auto bshape = m->outputs.empty() ? std::string() : shape_of(*gb, m->outputs[0]);
            if (ashape != bshape) {
                r.identical = false;
                ArchitectureChange c;
                c.subject = n.name.empty() ? n.op_type : n.name;
                c.kind = "modified";
                c.detail = "output " + ashape + " -> " + bshape;
                r.architecture.push_back(c);
            }
        } else {
            r.identical = false;
            ArchitectureChange c;
            c.subject = n.name.empty() ? n.op_type : n.name;
            c.kind = "removed";
            r.architecture.push_back(c);
        }
    }
    for (std::size_t i = 0; i < gb->nodes.size(); ++i) {
        if (!used[i]) {
            r.identical = false;
            ArchitectureChange c;
            c.subject = gb->nodes[i].name.empty() ? gb->nodes[i].op_type : gb->nodes[i].name;
            c.kind = "added";
            r.architecture.push_back(c);
        }
    }

    if (options.weights && !options.ignore_weights) {
        for (const auto& ta : ga->tensors) {
            if (!ta.constant) {
                continue;
            }
            const Tensor* tb = gb->find_tensor_by_name(ta.name);
            if (!tb || !tb->constant) {
                continue;
            }
            r.weights.push_back(compare_weights(ta, *tb, options.atol, options.rtol));
            if (r.weights.back().compared && r.weights.back().changed_elements > 0) {
                r.identical = false;
            }
        }
    }

    if (!options.ignore_metadata) {
        if (old_model.producer != new_model.producer) {
            ArchitectureChange c;
            c.subject = "producer";
            c.kind = "modified";
            c.detail = old_model.producer + " -> " + new_model.producer;
            r.metadata_changes.push_back(c);
            // metadata-only differences still count unless ignored
            if (options.metadata) {
                r.identical = false;
            }
        }
    }

    if (r.old_nodes == r.new_nodes && r.architecture.empty() && r.io.empty() &&
        r.old_parameters == r.new_parameters) {
        // keep identical unless weights/metadata said otherwise
    } else {
        r.identical = false;
    }
    return r;
}

}  // namespace nn
