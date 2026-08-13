#include "cli/commands.h"
#include "commands/common.h"
#include "formats/onnx/writer.h"
#include "nn/convert.h"

#include <algorithm>
#include <cctype>

namespace nn {
int cmd_convert(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "convert");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    if (flag_set(args.value(), "list") || args.value().positionals.empty()) {
        for (const auto& r : conversion_routes()) {
            const std::string status = r.available ? "available" : "unavailable";
            p.println(r.from + " -> " + r.to + "               " + status);
            if (!r.notes.empty()) {
                p.println("  " + r.notes);
            }
        }
        return kExitOk;
    }
    std::string to = flag_value(args.value(), "to").value_or("onnx");
    std::transform(to.begin(), to.end(), to.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    auto out = flag_value(args.value(), "output");
    if (!out && g.output_file) {
        out = g.output_file->string();
    }
    if (!out) {
        return p.usage_error("nn: --output is required", "nn convert <input> --to onnx -o FILE");
    }
    if (to == "onnx") {
        auto st = write_onnx_model(*out, model.value());
        if (!st) {
            return cmd_fail(p, st.error());
        }
        p.println("wrote " + *out + " (onnx)");
        return kExitOk;
    }
    p.errln("nn: conversion " + model.value().source_format + " -> " + to +
            " requires an external conversion adapter");
    return kExitBackendUnavailable;
}
}  // namespace nn
