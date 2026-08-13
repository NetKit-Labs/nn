#include "nn/model.h"
#include "nn/operator.h"

#include "nn/hash.h"

#include <algorithm>
#include <sstream>

namespace nn {

// Produce a stable textual canonicalization of graph structure (no weights).
std::string canonicalize_graph_text(const ModelIR& model) {
    std::ostringstream os;
    os << "format=" << model.source_format << '\n';
    for (const auto& g : model.graphs) {
        os << "graph " << g.name << '\n';
        std::vector<std::string> ins;
        for (TensorId id : g.inputs) {
            if (const Tensor* t = g.find_tensor(id)) {
                ins.push_back(t->name + ":" + t->shape.to_string() + ":" + datatype_name(t->dtype));
            }
        }
        std::sort(ins.begin(), ins.end());
        os << "inputs";
        for (const auto& s : ins) {
            os << ' ' << s;
        }
        os << '\n';
        std::vector<std::string> outs;
        for (TensorId id : g.outputs) {
            if (const Tensor* t = g.find_tensor(id)) {
                outs.push_back(t->name + ":" + t->shape.to_string() + ":" + datatype_name(t->dtype));
            }
        }
        std::sort(outs.begin(), outs.end());
        os << "outputs";
        for (const auto& s : outs) {
            os << ' ' << s;
        }
        os << '\n';
        std::vector<std::string> nodes;
        for (const auto& n : g.nodes) {
            std::ostringstream ns;
            ns << canonical_op_name(n.canonical) << '|' << n.op_type;
            for (TensorId id : n.inputs) {
                if (const Tensor* t = g.find_tensor(id)) {
                    ns << " in:" << t->shape.to_string() << '/' << datatype_name(t->dtype);
                }
            }
            for (TensorId id : n.outputs) {
                if (const Tensor* t = g.find_tensor(id)) {
                    ns << " out:" << t->shape.to_string() << '/' << datatype_name(t->dtype);
                }
            }
            for (const auto& [k, a] : n.attributes) {
                ns << ' ' << k << '=' << a.to_string();
            }
            nodes.push_back(ns.str());
        }
        std::sort(nodes.begin(), nodes.end());
        for (const auto& s : nodes) {
            os << s << '\n';
        }
    }
    return os.str();
}

std::string canonical_graph_hash(const ModelIR& model) {
    const std::string text = canonicalize_graph_text(model);
    return sha256_hex(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}

}  // namespace nn
