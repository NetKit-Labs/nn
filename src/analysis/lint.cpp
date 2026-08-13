#include "nn/analysis.h"

#include <unordered_map>
#include <unordered_set>

namespace nn {

LintReport lint_model(const ModelIR& model) {
    LintReport r;
    auto add = [&](LintSeverity sev, std::string code, std::string msg, std::string loc) {
        LintIssue i;
        i.severity = sev;
        i.code = std::move(code);
        i.message = std::move(msg);
        i.location = std::move(loc);
        if (sev == LintSeverity::Error) {
            ++r.errors;
        } else if (sev == LintSeverity::Warning) {
            ++r.warnings;
        } else {
            ++r.infos;
        }
        r.issues.push_back(std::move(i));
    };

    const Graph* g = primary_graph(model);
    if (!g) {
        add(LintSeverity::Error, "no-graph", "model contains no graph", "");
        return r;
    }
    if (g->nodes.empty() && model.source_format != "safetensors" &&
        model.source_format != "gguf") {
        add(LintSeverity::Warning, "empty-graph", "graph has no nodes", g->name);
    }

    std::unordered_set<TensorId> produced;
    std::unordered_set<TensorId> consumed;
    for (TensorId id : g->inputs) {
        produced.insert(id);
    }
    for (const auto& t : g->tensors) {
        if (t.constant) {
            produced.insert(t.id);
        }
        if (!t.shape.dims.empty()) {
            for (const auto& d : t.shape.dims) {
                if (d.value && *d.value < 0) {
                    add(LintSeverity::Error, "neg-dim", "negative dimension", t.name);
                }
            }
        }
        if (t.quantization.quantized && !t.quantization.valid()) {
            add(LintSeverity::Error, "bad-quant", "invalid quantization parameters", t.name);
        }
        if (t.quantization.quantized && t.quantization.scales.empty()) {
            add(LintSeverity::Warning, "missing-scale", "quantized tensor missing scale", t.name);
        }
    }
    for (const auto& n : g->nodes) {
        if (n.op_type.empty()) {
            add(LintSeverity::Error, "missing-op", "node missing op type", n.name);
        }
        if (n.canonical == CanonicalOp::Unknown && n.op_type != "Constant") {
            add(LintSeverity::Info, "unknown-op", "unrecognized operator " + n.op_type, n.name);
        }
        for (TensorId id : n.outputs) {
            produced.insert(id);
        }
        for (TensorId id : n.inputs) {
            consumed.insert(id);
            if (!g->find_tensor(id)) {
                add(LintSeverity::Error, "dangling-input", "node input tensor id not in graph",
                    n.name);
            }
        }
        if (n.canonical == CanonicalOp::Reshape) {
            add(LintSeverity::Info, "reshape", "reshape present; verify layout assumptions", n.name);
        }
    }
    for (const auto& t : g->tensors) {
        if (!consumed.count(t.id) && !t.model_output && !t.model_input) {
            add(LintSeverity::Warning, "unused-tensor", "tensor is never consumed", t.name);
        }
        if (t.constant && !consumed.count(t.id)) {
            add(LintSeverity::Warning, "unused-constant", "constant is unused", t.name);
        }
        if (auto e = t.shape.element_count()) {
            if (e.value() > (uint64_t{64} << 20) && !t.constant) {
                add(LintSeverity::Warning, "large-activation",
                    "intermediate tensor exceeds 64M elements", t.name);
            }
        } else if (!t.shape.is_static() && t.model_input) {
            add(LintSeverity::Warning, "dynamic-input",
                "dynamic input shape may prevent embedded deployment", t.name);
        }
    }
    for (TensorId id : g->outputs) {
        if (!produced.count(id) && !g->find_tensor(id)) {
            add(LintSeverity::Error, "missing-output", "output tensor not produced", "");
        }
    }
    return r;
}

}  // namespace nn
