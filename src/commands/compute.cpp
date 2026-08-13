#include "cli/commands.h"
#include "commands/common.h"
#include "nn/analysis.h"
#include "util/format_text.h"

namespace nn {
int cmd_compute(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "compute");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    auto r = analyze_compute(model.value());
    auto show = [](const OptionalCount& c) {
        return c.known ? human_si(static_cast<double>(c.value)) : std::string("unknown");
    };
    if (g.output_format == OutputFormat::Json) {
        Json o = Json::object();
        o["schema_version"] = 1;
        if (r.total.macs.known) {
            o["macs"] = r.total.macs.value;
        }
        if (r.total.flops.known) {
            o["flops"] = r.total.flops.value;
        }
        o["unknown_nodes"] = r.unknown_node_count;
        p.json(o);
        return kExitOk;
    }
    p.kv("MACs:", show(r.total.macs));
    p.kv("FLOPs:", show(r.total.flops));
    p.kv("Integer ops:", show(r.total.int_ops));
    p.kv("Float ops:", show(r.total.float_ops));
    p.kv("Unknown nodes:", std::to_string(r.unknown_node_count));
    if (flag_set(args.value(), "per-node")) {
        for (const auto& n : r.nodes) {
            p.println(n.name + "  " + n.op_type + "  " + show(n.cost.macs));
        }
    }
    return kExitOk;
}
}  // namespace nn
