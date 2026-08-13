#include "nn/graph.h"

namespace nn {

const Tensor* Graph::find_tensor(TensorId id) const {
    for (const auto& t : tensors) {
        if (t.id == id) {
            return &t;
        }
    }
    return nullptr;
}

Tensor* Graph::find_tensor(TensorId id) {
    for (auto& t : tensors) {
        if (t.id == id) {
            return &t;
        }
    }
    return nullptr;
}

const Tensor* Graph::find_tensor_by_name(std::string_view name) const {
    for (const auto& t : tensors) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

const Node* Graph::find_node(NodeId id) const {
    for (const auto& n : nodes) {
        if (n.id == id) {
            return &n;
        }
    }
    return nullptr;
}

const Node* Graph::find_node_by_name(std::string_view name) const {
    for (const auto& n : nodes) {
        if (n.name == name) {
            return &n;
        }
    }
    return nullptr;
}

void Graph::rebuild_use_lists() {
    for (auto& t : tensors) {
        t.producer_node.clear();
        t.consumer_nodes.clear();
    }
    for (const auto& n : nodes) {
        for (TensorId id : n.outputs) {
            if (auto* t = find_tensor(id)) {
                t->producer_node = n.name.empty() ? n.op_type : n.name;
            }
        }
        for (TensorId id : n.inputs) {
            if (auto* t = find_tensor(id)) {
                t->consumer_nodes.push_back(n.name.empty() ? n.op_type : n.name);
            }
        }
    }
}

}  // namespace nn
