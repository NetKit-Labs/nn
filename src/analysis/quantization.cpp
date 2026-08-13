#include "nn/analysis.h"

namespace nn {

QuantReport analyze_quantization(const ModelIR& model) {
    QuantReport r;
    const Graph* g = primary_graph(model);
    if (!g) {
        return r;
    }
    for (const auto& t : g->tensors) {
        if (t.quantization.quantized) {
            ++r.quantized_tensors;
            if (t.quantization.per_channel) {
                ++r.per_channel;
            } else {
                ++r.per_tensor;
            }
            if (!t.quantization.valid()) {
                r.issues.push_back("invalid quantization on tensor " + t.name);
            }
            if (t.quantization.quantized && t.quantization.scales.empty()) {
                r.issues.push_back("missing scale on tensor " + t.name);
            }
        }
        if (datatype_is_float(t.dtype)) {
            ++r.float_tensors;
        }
        if (datatype_is_integer(t.dtype)) {
            ++r.integer_tensors;
        }
    }
    for (const auto& n : g->nodes) {
        if (n.canonical == CanonicalOp::Quantize) {
            ++r.quantize_nodes;
        }
        if (n.canonical == CanonicalOp::Dequantize) {
            ++r.dequantize_nodes;
        }
    }
    if (r.quantize_nodes + r.dequantize_nodes > 0 &&
        r.quantize_nodes + r.dequantize_nodes > g->nodes.size() / 3) {
        r.issues.push_back("excessive quantize/dequantize boundaries relative to graph size");
    }
    return r;
}

}  // namespace nn
