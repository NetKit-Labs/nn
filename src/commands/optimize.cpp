#include "cli/commands.h"
#include "commands/common.h"
#include "formats/onnx/writer.h"
#include "nn/optimize.h"

namespace nn {
int cmd_optimize(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "optimize");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto loaded = cmd_load(args.value());
    if (!loaded) {
        return cmd_fail(p, loaded.error());
    }
    auto report = optimize_model(std::move(loaded.value()));
    p.println("Proposed changes: " + std::to_string(report.changes.size()));
    for (const auto& c : report.changes) {
        p.println("  " + c);
    }
    if (flag_set(args.value(), "dry-run")) {
        return kExitOk;
    }
    std::optional<std::string> out = flag_value(args.value(), "output");
    if (!out && g.output_file) {
        out = g.output_file->string();
    }
    if (!out) {
        return kExitOk;
    }
    if (report.model.source_format != "onnx" && report.model.source_format != "ONNX") {
        p.println("writing optimized graph as ONNX regardless of source format");
    }
    auto st = write_onnx_model(*out, report.model);
    if (!st) {
        return cmd_fail(p, st.error());
    }
    p.println("wrote " + *out);
    return kExitOk;
}
}  // namespace nn
