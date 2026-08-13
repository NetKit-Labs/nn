#include "cli/commands.h"
#include "commands/common.h"

namespace nn {
int cmd_io(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "io");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    const Graph* graph = primary_graph(model.value());
    if (!graph) {
        return cmd_fail(p, error(ErrorCode::InvalidGraph, "no graph"));
    }
    auto dump = [&](const char* title, const std::vector<TensorId>& ids) {
        p.heading(title);
        for (TensorId id : ids) {
            const Tensor* t = graph->find_tensor(id);
            if (!t) {
                continue;
            }
            p.println(t->name);
            p.kv("  shape:", t->shape.to_string());
            p.kv("  dtype:", datatype_name(t->dtype));
            p.kv("  layout:", t->layout.empty() ? std::string("-") : t->layout);
            p.kv("  quant:", t->quantization.describe());
        }
    };
    if (g.output_format == OutputFormat::Json) {
        Json root = Json::object();
        root["schema_version"] = 1;
        auto arr = [&](const std::vector<TensorId>& ids) {
            Json a = Json::array();
            for (TensorId id : ids) {
                if (const Tensor* t = graph->find_tensor(id)) {
                    Json o = Json::object();
                    o["name"] = t->name;
                    o["dtype"] = datatype_name(t->dtype);
                    o["shape"] = t->shape.to_string();
                    a.push(std::move(o));
                }
            }
            return a;
        };
        root["inputs"] = arr(graph->inputs);
        root["outputs"] = arr(graph->outputs);
        p.json(root);
        return kExitOk;
    }
    dump("Inputs", graph->inputs);
    dump("Outputs", graph->outputs);
    return kExitOk;
}
}  // namespace nn
