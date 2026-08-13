#include "nn/target.h"

#include <algorithm>
#include <cctype>

namespace nn {

PartitionReport partition_model(const ModelIR& model, const Accelerator& accel) {
    PartitionReport r;
    const Graph* g = primary_graph(model);
    if (!g) {
        return r;
    }
    auto supported = [&](const Node& n) {
        std::string op = n.op_type;
        for (const auto& s : accel.supported_ops) {
            if (s == n.op_type || s == canonical_op_name(n.canonical)) {
                return true;
            }
        }
        return false;
    };
    Partition current;
    current.device = "accelerator";
    current.index = 0;
    for (const auto& n : g->nodes) {
        const bool acc = supported(n);
        const std::string want = acc ? "accelerator" : "cpu";
        if (current.nodes.empty()) {
            current.device = want;
            if (!acc) {
                current.reason = "unsupported " + n.op_type;
            }
        } else if (current.device != want) {
            r.partitions.push_back(current);
            current = Partition{};
            current.index = static_cast<int>(r.partitions.size());
            current.device = want;
            if (!acc) {
                current.reason = "unsupported " + n.op_type;
            }
        }
        current.nodes.push_back(n.id);
    }
    if (!current.nodes.empty()) {
        r.partitions.push_back(current);
    }
    for (const auto& p : r.partitions) {
        if (p.device == "accelerator") {
            ++r.accelerator_launches;
        }
    }
    return r;
}

}  // namespace nn
