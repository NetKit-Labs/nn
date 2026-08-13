#include "cli/commands.h"
#include "commands/common.h"
#include "nn/analysis.h"

namespace nn {
int cmd_quant(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "quant");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    if (!args.value().positionals.empty() && args.value().positionals[0] == "compare") {
        if (args.value().positionals.size() < 3) {
            return p.usage_error("nn: quant compare requires two models",
                                 "nn quant compare <float> <quant>");
        }
        auto a = load_model(args.value().positionals[1]);
        auto b = load_model(args.value().positionals[2]);
        if (!a) {
            return cmd_fail(p, a.error());
        }
        if (!b) {
            return cmd_fail(p, b.error());
        }
        auto qa = analyze_quantization(a.value());
        auto qb = analyze_quantization(b.value());
        p.println("Quantization comparison");
        p.kv("Float tensors (A/B):", std::to_string(qa.float_tensors) + " / " +
                                         std::to_string(qb.float_tensors));
        p.kv("Quantized tensors (A/B):", std::to_string(qa.quantized_tensors) + " / " +
                                             std::to_string(qb.quantized_tensors));
        p.println("Per-layer activation error: unavailable (no test vectors supplied)");
        return kExitOk;
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    auto r = analyze_quantization(model.value());
    if (g.output_format == OutputFormat::Json) {
        Json o = Json::object();
        o["schema_version"] = 1;
        o["quantized_tensors"] = r.quantized_tensors;
        o["float_tensors"] = r.float_tensors;
        o["integer_tensors"] = r.integer_tensors;
        o["per_channel"] = r.per_channel;
        o["per_tensor"] = r.per_tensor;
        o["quantize_nodes"] = r.quantize_nodes;
        o["dequantize_nodes"] = r.dequantize_nodes;
        p.json(o);
        return kExitOk;
    }
    p.kv("Quantized tensors:", std::to_string(r.quantized_tensors));
    p.kv("Float tensors:", std::to_string(r.float_tensors));
    p.kv("Integer tensors:", std::to_string(r.integer_tensors));
    p.kv("Per-channel:", std::to_string(r.per_channel));
    p.kv("Per-tensor:", std::to_string(r.per_tensor));
    p.kv("Quantize nodes:", std::to_string(r.quantize_nodes));
    p.kv("Dequantize nodes:", std::to_string(r.dequantize_nodes));
    for (const auto& i : r.issues) {
        p.errln("nn: warning: " + i);
    }
    return kExitOk;
}
}  // namespace nn
