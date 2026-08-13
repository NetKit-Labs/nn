#include "cli/commands.h"
#include "commands/common.h"
#include "nn/diff.h"
#include "util/format_text.h"

#include <cstdio>
#include <cstdlib>

namespace nn {
int cmd_diff(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "diff");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    if (args.value().positionals.size() < 2) {
        return p.usage_error("nn: missing model operands", "nn diff [options] <old> <new>");
    }
    auto a = load_model(args.value().positionals[0]);
    if (!a) {
        return cmd_fail(p, a.error());
    }
    auto b = load_model(args.value().positionals[1]);
    if (!b) {
        return cmd_fail(p, b.error());
    }
    DiffOptions opt;
    opt.weights = flag_set(args.value(), "weights") || flag_set(args.value(), "numeric");
    opt.ignore_weights = flag_set(args.value(), "ignore-weights");
    opt.ignore_metadata = flag_set(args.value(), "ignore-metadata");
    if (auto v = flag_value(args.value(), "atol")) {
        opt.atol = std::strtod(v->c_str(), nullptr);
    }
    if (auto v = flag_value(args.value(), "rtol")) {
        opt.rtol = std::strtod(v->c_str(), nullptr);
    }
    auto d = diff_models(a.value(), b.value(), opt);
    if (!d) {
        return cmd_fail(p, d.error());
    }
    const auto& r = d.value();
    if (g.output_format == OutputFormat::Json) {
        Json o = Json::object();
        o["schema_version"] = 1;
        o["identical"] = r.identical;
        o["old_nodes"] = r.old_nodes;
        o["new_nodes"] = r.new_nodes;
        o["old_parameters"] = r.old_parameters;
        o["new_parameters"] = r.new_parameters;
        p.json(o);
        return r.identical ? kExitOk : kExitDifference;
    }
    p.println("MODEL DIFFERENCE");
    p.heading("Summary");
    p.println("                              OLD             NEW");
    p.println("---------------------------------------------------------");
    char line[96];
    std::snprintf(line, sizeof(line), "Nodes               %12llu %15llu",
                  static_cast<unsigned long long>(r.old_nodes),
                  static_cast<unsigned long long>(r.new_nodes));
    p.println(line);
    std::snprintf(line, sizeof(line), "Parameters          %12s %15s",
                  human_si(static_cast<double>(r.old_parameters)).c_str(),
                  human_si(static_cast<double>(r.new_parameters)).c_str());
    p.println(line);
    std::snprintf(line, sizeof(line), "Weight storage      %12s %15s",
                  human_bytes(r.old_weight_bytes).c_str(), human_bytes(r.new_weight_bytes).c_str());
    p.println(line);
    if (!r.architecture.empty()) {
        p.heading("Architecture changes");
        for (const auto& c : r.architecture) {
            p.println("- " + c.subject + "  " + c.kind + (c.detail.empty() ? "" : "  " + c.detail));
        }
    }
    if (r.io.empty()) {
        p.heading("Inputs/Outputs");
        p.println("    unchanged");
    } else {
        p.heading("Inputs/Outputs");
        for (const auto& c : r.io) {
            p.println(c.subject + ": " + c.detail);
        }
    }
    for (const auto& w : r.weights) {
        if (w.compared) {
            p.println(w.name + "  max_abs=" + std::to_string(w.max_abs) +
                      " rmse=" + std::to_string(w.rmse));
        }
    }
    return r.identical ? kExitOk : kExitDifference;
}
}  // namespace nn
