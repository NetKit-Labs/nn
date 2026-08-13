#include "cli/commands.h"
#include "commands/common.h"
#include "nn/target.h"
#include "util/format_text.h"

namespace nn {
int cmd_target(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "target");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    Target tgt;
    if (auto f = flag_value(args.value(), "target-file")) {
        auto loaded = load_target_file(*f);
        if (!loaded) {
            return cmd_fail(p, loaded.error());
        }
        tgt = std::move(loaded.value());
    } else {
        const std::string name = flag_value(args.value(), "target").value_or("cortex-m4f");
        const Target* b = find_builtin_target(name);
        if (!b) {
            p.errln("nn: unknown target '" + name + "'");
            p.errln("Try 'nn targets' for the list.");
            return kExitUsage;
        }
        tgt = *b;
    }
    auto fit = evaluate_target(model.value(), tgt);
    p.println("Target: " + fit.target_name);
    p.kv("RAM:", human_bytes(tgt.ram_bytes));
    p.kv("Flash:", human_bytes(tgt.storage_bytes));
    p.heading("Storage");
    p.kv("    Model:", human_bytes(fit.model_bytes) + (fit.fits_storage ? "       PASS" : "       FAIL"));
    p.heading("RAM");
    p.kv("    Activations:", human_bytes(fit.activation_bytes));
    p.kv("    Scratch:", human_bytes(fit.scratch_bytes));
    p.kv("    Runtime overhead:", human_bytes(fit.runtime_overhead_bytes));
    p.kv("    Total:", human_bytes(fit.total_ram_bytes) + (fit.fits_ram ? "       PASS" : "       FAIL"));
    if (!fit.unsupported_ops.empty()) {
        p.heading("Unsupported operators");
        for (const auto& o : fit.unsupported_ops) {
            p.println("    " + o);
        }
    }
    p.heading("Result:");
    p.println(fit.overall_fit ? "    MODEL FITS" : "    MODEL DOES NOT CURRENTLY FIT");
    for (const auto& n : fit.notes) {
        p.println("    " + n);
    }
    return fit.overall_fit ? kExitOk : kExitDifference;
}
}  // namespace nn
