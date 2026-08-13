#include "cli/commands.h"
#include "commands/common.h"
#include "nn/format.h"

#include <cstdio>

namespace nn {
int cmd_formats(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "formats");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto infos = default_format_registry().formats();
    if (!args.value().positionals.empty()) {
        const std::string want = args.value().positionals[0];
        bool found = false;
        for (const auto& f : infos) {
            if (f.name == want || f.display_name == want) {
                found = true;
                p.kv("Name:", f.display_name);
                p.kv("Read:", yes_no(f.capabilities.read));
                p.kv("Graph:", yes_no(f.capabilities.graph));
                p.kv("Weights:", yes_no(f.capabilities.weights));
                p.kv("Execute:", yes_no(f.capabilities.execute));
                p.kv("Convert:", yes_no(f.capabilities.convert));
                if (!f.capabilities.notes.empty()) {
                    p.kv("Notes:", f.capabilities.notes);
                }
            }
        }
        if (!found) {
            p.errln("nn: unknown format '" + want + "'");
            return kExitUsage;
        }
        return kExitOk;
    }
    if (g.output_format == OutputFormat::Json) {
        Json arr = Json::array();
        for (const auto& f : infos) {
            Json o = Json::object();
            o["name"] = f.name;
            o["display"] = f.display_name;
            o["read"] = f.capabilities.read;
            o["graph"] = f.capabilities.graph;
            o["weights"] = f.capabilities.weights;
            o["execute"] = f.capabilities.execute;
            o["convert"] = f.capabilities.convert;
            o["notes"] = f.capabilities.notes;
            arr.push(std::move(o));
        }
        Json root = Json::object();
        root["schema_version"] = 1;
        root["formats"] = std::move(arr);
        p.json(root);
        return kExitOk;
    }
    p.println("Format             Read   Graph   Weights   Execute   Convert");
    p.println("---------------------------------------------------------------");
    for (const auto& f : infos) {
        char line[96];
        std::snprintf(line, sizeof(line), "%-18.18s %-6s %-7s %-9s %-9s %s", f.display_name.c_str(),
                      yes_no(f.capabilities.read).c_str(), yes_no(f.capabilities.graph).c_str(),
                      yes_no(f.capabilities.weights).c_str(),
                      yes_no(f.capabilities.execute).c_str(),
                      yes_no(f.capabilities.convert).c_str());
        p.println(line);
    }
    return kExitOk;
}
}  // namespace nn
