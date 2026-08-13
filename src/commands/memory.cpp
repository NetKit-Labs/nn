#include "cli/commands.h"
#include "commands/common.h"
#include "nn/analysis.h"
#include "util/format_text.h"

#include <cstdio>

namespace nn {
int cmd_memory(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "memory");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    MemoryOptions opt;
    opt.plan = flag_set(args.value(), "plan");
    auto r = analyze_memory(model.value(), opt);
    if (g.output_format == OutputFormat::Json) {
        Json o = Json::object();
        o["schema_version"] = 1;
        o["weights"] = r.weight_bytes;
        o["peak_activations"] = r.peak_activation_bytes;
        o["estimated_ram"] = r.estimated_ram_bytes;
        o["file_size"] = r.file_size;
        if (opt.plan) {
            Json plan = Json::array();
            for (const auto& e : r.plan) {
                Json pe = Json::object();
                pe["name"] = e.name;
                pe["offset"] = e.offset;
                pe["size"] = e.size;
                pe["birth"] = e.birth;
                pe["death"] = e.death;
                plan.push(std::move(pe));
            }
            o["plan"] = std::move(plan);
        }
        p.json(o);
        return kExitOk;
    }
    p.println("Memory Analysis");
    p.println();
    p.kv("Weights", human_bytes(r.weight_bytes));
    p.kv("Persistent tensors", human_bytes(r.persistent_bytes));
    p.kv("Peak live activations",
         r.peak_known ? human_bytes(r.peak_activation_bytes) : std::string("unknown"));
    p.kv("Estimated scratch", human_bytes(r.scratch_bytes));
    p.println("--------------------------------------");
    p.kv("Estimated RAM requirement", human_bytes(r.estimated_ram_bytes));
    p.kv("Flash/model storage", human_bytes(r.file_size));
    if (flag_set(args.value(), "timeline")) {
        p.heading("Timeline");
        for (const auto& lt : r.lifetimes) {
            char line[128];
            std::snprintf(line, sizeof(line), "%-32.32s  %8s  life %d-%d%s", lt.name.c_str(),
                          human_bytes(lt.bytes).c_str(), lt.birth, lt.death,
                          lt.persistent ? "  persistent" : "");
            p.println(line);
        }
    }
    if (opt.plan) {
        p.heading("Tensor                   Start       Size        Lifetime");
        p.println("---------------------------------------------------------");
        for (const auto& e : r.plan) {
            char line[128];
            std::snprintf(line, sizeof(line), "%-24.24s %8llu %10llu      %d-%d", e.name.c_str(),
                          static_cast<unsigned long long>(e.offset),
                          static_cast<unsigned long long>(e.size), e.birth, e.death);
            p.println(line);
        }
    }
    return kExitOk;
}
}  // namespace nn
