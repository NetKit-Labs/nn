#include "nn/graph.h"

namespace nn {

const Tensor* Graph::find_tensor(TensorId tensor_id) const {
    for (const auto& t : tensors) {
        if (t.id == tensor_id) {
            return &t;
        }
    }
    return nullptr;
}

Tensor* Graph::find_tensor(TensorId tensor_id) {
    for (auto& t : tensors) {
        if (t.id == tensor_id) {
            return &t;
        }
    }
    return nullptr;
}

const Tensor* Graph::find_tensor_by_name(std::string_view tensor_name) const {
    for (const auto& t : tensors) {
        if (t.name == tensor_name) {
            return &t;
        }
    }
    return nullptr;
}

const Node* Graph::find_node(NodeId node_id) const {
    for (const auto& n : nodes) {
        if (n.id == node_id) {
            return &n;
        }
    }
    return nullptr;
}

const Node* Graph::find_node_by_name(std::string_view node_name) const {
    for (const auto& n : nodes) {
        if (n.name == node_name) {
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
        for (TensorId tid : n.outputs) {
            if (auto* t = find_tensor(tid)) {
                t->producer_node = n.name.empty() ? n.op_type : n.name;
            }
        }
        for (TensorId tid : n.inputs) {
            if (auto* t = find_tensor(tid)) {
                t->consumer_nodes.push_back(n.name.empty() ? n.op_type : n.name);
            }
        }
    }
}

}  // namespace nn
