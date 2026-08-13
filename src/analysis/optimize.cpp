#include "nn/optimize.h"

#include "runtime/reference/ops.h"

#include <unordered_set>

namespace nn {
namespace {

void replace_tensor_id(Graph& g, TensorId from, TensorId to) {
    if (from == to) {
        return;
    }
    for (auto& n : g.nodes) {
        for (auto& in : n.inputs) {
            if (in == from) {
                in = to;
            }
        }
    }
    for (auto& id : g.inputs) {
        if (id == from) {
            id = to;
        }
    }
    for (auto& id : g.outputs) {
        if (id == from) {
            id = to;
        }
    }
}

bool is_pass_through(const Node& n) {
    return n.canonical == CanonicalOp::Identity || n.canonical == CanonicalOp::Dropout ||
           n.op_type == "Identity" || n.op_type == "Dropout";
}

bool tensor_is_used(const Graph& g, TensorId id, const std::unordered_set<NodeId>& skip) {
    for (TensorId out : g.outputs) {
        if (out == id) {
            return true;
        }
    }
    for (const auto& n : g.nodes) {
        if (skip.count(n.id)) {
            continue;
        }
        for (TensorId in : n.inputs) {
            if (in == id) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

OptimizeReport optimize_model(ModelIR model) {
    OptimizeReport r;
    Graph* g = primary_graph(model);
    if (!g) {
        r.model = std::move(model);
        return r;
    }

    std::vector<Node> kept;
    kept.reserve(g->nodes.size());
    for (auto& n : g->nodes) {
        if (is_pass_through(n) && n.inputs.size() >= 1 && n.outputs.size() >= 1) {
            replace_tensor_id(*g, n.outputs[0], n.inputs[0]);
            r.changes.push_back("rewire " + n.op_type + " node " +
                                (n.name.empty() ? std::to_string(n.id) : n.name));
            continue;
        }
        kept.push_back(std::move(n));
    }
    g->nodes = std::move(kept);

    bool folded = true;
    while (folded) {
        folded = false;
        std::map<TensorId, RuntimeTensor> consts;
        for (const auto& t : g->tensors) {
            if (t.constant && t.data) {
                RuntimeTensor rt;
                rt.name = t.name;
                rt.dtype = t.dtype;
                rt.shape = t.shape;
                auto bytes = tensor_payload_bytes(t);
                if (bytes) {
                    rt.bytes = std::move(bytes.value());
                    consts[t.id] = std::move(rt);
                }
            }
        }
        for (std::size_t i = 0; i < g->nodes.size(); ++i) {
            const Node& n = g->nodes[i];
            if (n.op_type != "Add" && n.op_type != "Mul" && n.op_type != "Sub" && n.op_type != "Div") {
                continue;
            }
            if (n.inputs.size() < 2 || n.outputs.empty()) {
                continue;
            }
            if (!consts.count(n.inputs[0]) || !consts.count(n.inputs[1])) {
                continue;
            }
            auto out = reference_exec_node(*g, n, consts);
            if (!out) {
                continue;
            }
            if (Tensor* t = g->find_tensor(n.outputs[0])) {
                t->constant = true;
                t->dtype = out.value().dtype;
                t->shape = out.value().shape;
                TensorDataReference ref;
                ref.owned = std::make_shared<const std::vector<uint8_t>>(std::move(out.value().bytes));
                ref.length = ref.owned->size();
                t->data = std::move(ref);
                t->storage_bytes = ref.length;
            }
            r.changes.push_back("fold " + n.op_type + " node " +
                                (n.name.empty() ? std::to_string(n.id) : n.name));
            g->nodes.erase(g->nodes.begin() + static_cast<std::ptrdiff_t>(i));
            folded = true;
            break;
        }
    }

    bool dropped = true;
    while (dropped) {
        dropped = false;
        for (std::size_t i = 0; i < g->nodes.size(); ++i) {
            const Node& n = g->nodes[i];
            bool used = false;
            for (TensorId out : n.outputs) {
                if (tensor_is_used(*g, out, {n.id})) {
                    used = true;
                    break;
                }
            }
            if (!used && !n.outputs.empty()) {
                r.changes.push_back("eliminate dead node " +
                                    (n.name.empty() ? n.op_type : n.name));
                g->nodes.erase(g->nodes.begin() + static_cast<std::ptrdiff_t>(i));
                dropped = true;
                break;
            }
        }
    }

    g->rebuild_use_lists();
    r.model = std::move(model);
    return r;
}

Result<ModelIR> extract_subgraph(const ModelIR& model, std::string_view from, std::string_view to) {
    const Graph* src = primary_graph(model);
    if (!src) {
        return error(ErrorCode::InvalidGraph, "no graph");
    }
    int from_i = 0;
    int to_i = static_cast<int>(src->nodes.size()) - 1;
    if (!from.empty()) {
        from_i = -1;
        for (int i = 0; i < static_cast<int>(src->nodes.size()); ++i) {
            if (src->nodes[static_cast<std::size_t>(i)].name == from ||
                src->nodes[static_cast<std::size_t>(i)].op_type == from) {
                from_i = i;
                break;
            }
        }
        if (from_i < 0) {
            return error(ErrorCode::InvalidArgument, "start node not found: " + std::string(from));
        }
    }
    if (!to.empty()) {
        to_i = -1;
        for (int i = static_cast<int>(src->nodes.size()) - 1; i >= 0; --i) {
            if (src->nodes[static_cast<std::size_t>(i)].name == to ||
                src->nodes[static_cast<std::size_t>(i)].op_type == to) {
                to_i = i;
                break;
            }
        }
        if (to_i < 0) {
            return error(ErrorCode::InvalidArgument, "end node not found: " + std::string(to));
        }
    }
    if (from_i > to_i) {
        return error(ErrorCode::InvalidArgument, "subgraph range is empty");
    }
    ModelIR out = model;
    Graph* g = primary_graph(out);
    if (!g) {
        return error(ErrorCode::InvalidGraph, "no graph");
    }
    std::vector<Node> nodes;
    for (int i = from_i; i <= to_i; ++i) {
        nodes.push_back(src->nodes[static_cast<std::size_t>(i)]);
    }
    g->nodes = std::move(nodes);
    std::unordered_set<TensorId> produced;
    std::unordered_set<TensorId> consumed;
    for (const auto& n : g->nodes) {
        for (TensorId id : n.outputs) {
            produced.insert(id);
        }
        for (TensorId id : n.inputs) {
            consumed.insert(id);
        }
    }
    g->inputs.clear();
    for (TensorId id : consumed) {
        if (!produced.count(id)) {
            if (const Tensor* t = g->find_tensor(id); t && !t->constant) {
                g->inputs.push_back(id);
            }
        }
    }
    g->outputs.clear();
    if (to_i >= 0 && to_i < static_cast<int>(src->nodes.size())) {
        g->outputs = src->nodes[static_cast<std::size_t>(to_i)].outputs;
    }
    g->rebuild_use_lists();
    return out;
}

}  // namespace nn
