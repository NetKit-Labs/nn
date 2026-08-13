#include "cli/commands.h"
#include "commands/common.h"

namespace nn {
int cmd_metadata(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "metadata");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    if (g.output_format == OutputFormat::Json) {
        Json o = Json::object();
        o["schema_version"] = 1;
        o["format"] = model.value().source_format;
        o["producer"] = model.value().producer;
        Json md = Json::object();
        for (const auto& [k, v] : model.value().metadata) {
            md[k] = v;
        }
        o["metadata"] = std::move(md);
        p.json(o);
        return kExitOk;
    }
    p.kv("Format:", model.value().source_format);
    p.kv("Producer:", model.value().producer);
    p.kv("Version:", model.value().source_format_version);
    for (const auto& [k, v] : model.value().metadata) {
        p.kv(k + ":", v);
    }
    return kExitOk;
}
}  // namespace nn
