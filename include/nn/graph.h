#ifndef NN_GRAPH_H
#define NN_GRAPH_H

#include "nn/node.h"
#include "nn/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nn {

using GraphId = uint32_t;
inline constexpr GraphId kInvalidGraphId = ~GraphId{0};

struct Graph {
    GraphId id = 0;
    std::string name;
    std::vector<Node> nodes;
    std::vector<Tensor> tensors;
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
    std::string doc;

    const Tensor* find_tensor(TensorId tensor_id) const;
    Tensor* find_tensor(TensorId tensor_id);
    const Tensor* find_tensor_by_name(std::string_view tensor_name) const;
    const Node* find_node(NodeId node_id) const;
    const Node* find_node_by_name(std::string_view node_name) const;

    // Rebuild producer/consumer fields on tensors from node edges.
    void rebuild_use_lists();
};

}  // namespace nn

#endif
