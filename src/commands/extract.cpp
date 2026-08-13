#include "cli/commands.h"
#include "commands/common.h"
#include "formats/onnx/writer.h"
#include "nn/optimize.h"
#include "runtime/tensor_io.h"

namespace nn {
int cmd_extract(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "extract");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    auto out = flag_value(args.value(), "output");
    if (!out && g.output_file) {
        out = g.output_file->string();
    }
    if (!out) {
        return p.usage_error("nn: -o is required", "nn extract <model> --tensor NAME -o FILE");
    }
    auto from = flag_value(args.value(), "from");
    auto to = flag_value(args.value(), "to");
    if (from || to) {
        auto sub = extract_subgraph(model.value(), from.value_or(""), to.value_or(""));
        if (!sub) {
            return cmd_fail(p, sub.error());
        }
        auto st = write_onnx_model(*out, sub.value());
        if (!st) {
            return cmd_fail(p, st.error());
        }
        p.println("wrote subgraph " + *out);
        return kExitOk;
    }
    auto tensor = flag_value(args.value(), "tensor");
    if (!tensor) {
        return p.usage_error("nn: --tensor and -o are required",
                             "nn extract <model> --tensor NAME -o FILE");
    }
    const Graph* graph = primary_graph(model.value());
    if (!graph) {
        return kExitMalformed;
    }
    const Tensor* t = graph->find_tensor_by_name(*tensor);
    if (!t) {
        p.errln("nn: tensor not found: " + *tensor);
        return kExitMalformed;
    }
    auto rt = runtime_from_ir_tensor(*t);
    if (!rt) {
        return cmd_fail(p, rt.error());
    }
    auto st = save_tensor_file(*out, rt.value());
    if (!st) {
        return cmd_fail(p, st.error());
    }
    p.println("wrote " + *out);
    return kExitOk;
}
}  // namespace nn
