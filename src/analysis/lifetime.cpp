#include "nn/analysis.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace nn {

std::vector<TensorLifetime> analyze_lifetimes(const Graph& graph) {
    std::unordered_map<TensorId, int> birth;
    std::unordered_map<TensorId, int> death;
    std::unordered_set<TensorId> persistent;

    for (TensorId id : graph.inputs) {
        persistent.insert(id);
        birth[id] = -1;
    }
    for (const auto& t : graph.tensors) {
        if (t.constant) {
            persistent.insert(t.id);
            birth[t.id] = -1;
        }
    }

    for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
        const auto& n = graph.nodes[static_cast<std::size_t>(i)];
        for (TensorId id : n.outputs) {
            if (!birth.count(id)) {
                birth[id] = i;
            }
        }
        for (TensorId id : n.inputs) {
            death[id] = i;
        }
    }

    std::vector<TensorLifetime> out;
    out.reserve(graph.tensors.size());
    for (const auto& t : graph.tensors) {
        TensorLifetime lt;
        lt.tensor_id = t.id;
        lt.name = t.name;
        lt.persistent = persistent.count(t.id) != 0;
        lt.birth = birth.count(t.id) ? birth[t.id] : -1;
        lt.death = death.count(t.id) ? death[t.id] : lt.birth;
        if (auto b = tensor_storage_bytes(t)) {
            lt.bytes = b.value();
        }
        if (t.model_output) {
            lt.death = static_cast<int>(graph.nodes.size());
        }
        out.push_back(lt);
    }
    return out;
}

}  // namespace nn
