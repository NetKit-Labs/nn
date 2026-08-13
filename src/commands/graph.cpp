#include "cli/commands.h"
#include "commands/common.h"

namespace nn {
int cmd_graph(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "graph");
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
    std::string fmt = flag_value(args.value(), "format").value_or("text");
    auto opf = flag_value(args.value(), "op");
    if (fmt == "dot") {
        p.println("digraph G {");
        for (const auto& n : graph->nodes) {
            if (opf && n.op_type != *opf) {
                continue;
            }
            p.println("  n" + std::to_string(n.id) + " [label=\"" +
                      (n.name.empty() ? n.op_type : n.name) + "\"];");
        }
        for (const auto& n : graph->nodes) {
            for (TensorId tid : n.outputs) {
                if (const Tensor* t = graph->find_tensor(tid)) {
                    for (const auto& m : graph->nodes) {
                        for (TensorId in : m.inputs) {
                            if (in == tid) {
                                p.println("  n" + std::to_string(n.id) + " -> n" +
                                          std::to_string(m.id) + " [label=\"" + t->name + "\"];");
                            }
                        }
                    }
                }
            }
        }
        p.println("}");
        return kExitOk;
    }
    if (fmt == "mermaid") {
        p.println("graph TD");
        for (const auto& n : graph->nodes) {
            p.println("  n" + std::to_string(n.id) + "[" +
                      (n.name.empty() ? n.op_type : n.name) + "]");
        }
        return kExitOk;
    }
    if (fmt == "json" || g.output_format == OutputFormat::Json) {
        Json arr = Json::array();
        for (const auto& n : graph->nodes) {
            Json o = Json::object();
            o["id"] = static_cast<int64_t>(n.id);
            o["name"] = n.name;
            o["op"] = n.op_type;
            arr.push(std::move(o));
        }
        Json root = Json::object();
        root["schema_version"] = 1;
        root["nodes"] = std::move(arr);
        p.json(root);
        return kExitOk;
    }
    for (const auto& n : graph->nodes) {
        p.println(std::to_string(n.id) + "  " + n.op_type + "  " + n.name);
    }
    return kExitOk;
}
}  // namespace nn
