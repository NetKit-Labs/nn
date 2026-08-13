#include "nn/model.h"

namespace nn {

const Graph* primary_graph(const ModelIR& model) {
    if (model.graphs.empty()) {
        return nullptr;
    }
    return &model.graphs.front();
}

Graph* primary_graph(ModelIR& model) {
    if (model.graphs.empty()) {
        return nullptr;
    }
    return &model.graphs.front();
}

}  // namespace nn
