#include "nn/analysis.h"

namespace nn {

Result<InspectReport> inspect_model(const ModelIR& model) {
    InspectReport r;
    r.model = model;
    r.compute = analyze_compute(model);
    r.memory = analyze_memory(model);
    const Graph* g = primary_graph(model);
    if (!g) {
        return r;
    }
    uint64_t params = 0;
    bool known = true;
    for (const auto& t : g->tensors) {
        if (!t.constant) {
            continue;
        }
        auto n = t.shape.element_count();
        if (!n) {
            known = false;
            continue;
        }
        params += n.value();
    }
    r.parameter_count = params;
    r.parameter_count_known = known;
    return r;
}

}  // namespace nn
