#include "cli/commands.h"
#include "commands/common.h"
#include "nn/compat.h"

namespace nn {
int cmd_compat(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "compat");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    const std::string runtime = flag_value(args.value(), "runtime").value_or("reference");
    const std::string ver = flag_value(args.value(), "runtime-version").value_or("");
    auto report = check_compat(model.value(), runtime, ver);
    p.kv("Runtime:", report.runtime);
    p.kv("Capability table:", report.table_version.empty() ? "(none)" : report.table_version);
    if (const auto* table = find_compat_table(runtime, ver)) {
        if (!table->notes.empty()) {
            p.kv("Notes:", table->notes);
        }
    }
    p.kv("Nodes supported:", std::to_string(report.supported) + " / " + std::to_string(report.total));
    if (!report.unsupported.empty()) {
        p.heading("Unsupported:");
        for (const auto& b : report.unsupported) {
            p.println("  " + b);
        }
        p.println("RESULT: incompatible");
        return kExitDifference;
    }
    p.println("RESULT: compatible (vs " + report.runtime + " table " + report.table_version + ")");
    return kExitOk;
}
}  // namespace nn
