#include "cli/commands.h"

#include "commands/common.h"
#include "nn/analysis.h"
#include "util/format_text.h"

namespace nn {

int cmd_inspect(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "inspect");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    auto rep = inspect_model(model.value());
    if (!rep) {
        return cmd_fail(p, rep.error());
    }
    const InspectReport& r = rep.value();
    const Graph* graph = primary_graph(r.model);
    const bool all = flag_set(args.value(), "all");
    const bool summary_only = flag_set(args.value(), "summary");

    if (g.output_format == OutputFormat::Json || g.output_format == OutputFormat::Yaml) {
        Json j = Json::object();
        j["schema_version"] = 1;
        j["file"] = r.model.source_path.string();
        j["format"] = r.model.source_format;
        j["format_version"] = r.model.source_format_version;
        j["producer"] = r.model.producer;
        j["file_size"] = r.model.file_size;
        j["sha256"] = r.model.sha256;
        if (graph) {
            j["graphs"] = static_cast<int64_t>(r.model.graphs.size());
            j["nodes"] = static_cast<int64_t>(graph->nodes.size());
            j["tensors"] = static_cast<int64_t>(graph->tensors.size());
        }
        if (r.parameter_count_known) {
            j["parameters"] = r.parameter_count;
        }
        if (r.compute.total.macs.known) {
            j["macs"] = r.compute.total.macs.value;
        }
        if (g.output_format == OutputFormat::Yaml) {
            p.yaml(j);
        } else {
            p.json(j);
        }
        return kExitOk;
    }

    p.println("Model");
    p.kv("File:", r.model.source_path.string());
    p.kv("Format:", r.model.source_format);
    p.kv("Format version:", r.model.source_format_version.empty() ? std::string("-")
                                                                  : r.model.source_format_version);
    p.kv("Framework:", r.model.framework_version.empty() ? std::string("-")
                                                         : r.model.framework_version);
    p.kv("Producer:", r.model.producer.empty() ? std::string("-") : r.model.producer);
    p.kv("File size:", format_bytes(r.model.file_size));
    p.kv("SHA-256:", r.model.sha256.empty() ? std::string("-") : r.model.sha256);

    if (graph && !summary_only) {
        p.heading("Graph");
        p.kv("Graphs:", std::to_string(r.model.graphs.size()));
        p.kv("Nodes:", std::to_string(graph->nodes.size()));
        p.kv("Tensors:", std::to_string(graph->tensors.size()));
    }
    p.heading("Parameters");
    if (r.parameter_count_known) {
        p.kv("Count:", format_count(r.parameter_count));
    } else {
        p.kv("Count:", "unknown");
    }
    p.kv("Storage:", format_bytes(r.memory.weight_bytes));

    p.heading("Compute");
    if (r.compute.total.macs.known) {
        p.kv("MACs:", human_si(static_cast<double>(r.compute.total.macs.value)));
    } else {
        p.kv("MACs:", "unknown");
    }
    if (r.compute.total.flops.known) {
        p.kv("FLOPs:", human_si(static_cast<double>(r.compute.total.flops.value)));
    } else {
        p.kv("FLOPs:", "unknown");
    }

    if (graph && (all || !flag_set(args.value(), "outputs") || true)) {
        p.heading("Inputs");
        if (graph->inputs.empty()) {
            p.println("  (none)");
        }
        for (TensorId id : graph->inputs) {
            if (const Tensor* t = graph->find_tensor(id)) {
                p.println("  " + t->name);
                p.kv("    shape:", t->shape.to_string());
                p.kv("    dtype:", datatype_name(t->dtype));
            }
        }
        p.heading("Outputs");
        if (graph->outputs.empty()) {
            p.println("  (none)");
        }
        for (TensorId id : graph->outputs) {
            if (const Tensor* t = graph->find_tensor(id)) {
                p.println("  " + t->name);
                p.kv("    shape:", t->shape.to_string());
                p.kv("    dtype:", datatype_name(t->dtype));
            }
        }
    }
    if (flag_set(args.value(), "metadata") || all) {
        p.heading("Metadata");
        for (const auto& [k, v] : r.model.metadata) {
            p.kv(k + ":", v);
        }
    }
    for (const auto& w : r.model.warnings) {
        p.errln("nn: warning: " + w);
    }
    return kExitOk;
}

}  // namespace nn
