#include "cli/commands.h"
#include "commands/common.h"
#include "runtime/tensor_io.h"

namespace nn {
int cmd_profile(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "profile");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    const std::string backend_name = flag_value(args.value(), "backend").value_or("");
    RuntimeOptions opt;
    opt.threads = g.threads;
    opt.profile = true;
    auto ins = bind_model_inputs(model.value(), flag_values(args.value(), "input"), 0, false);
    if (!ins) {
        return cmd_fail(p, ins.error());
    }
    auto rr = eval_model(model.value(), ins.value(), opt, backend_name);
    if (!rr) {
        return cmd_fail(p, rr.error());
    }
    if (!rr.value().profiled || rr.value().profile.empty()) {
        p.errln("nn: backend cannot expose operator timings");
        return kExitBackendUnavailable;
    }
    p.kv("latency_ms:", std::to_string(rr.value().latency_ms));
    p.println("node                         op              time_ms");
    for (const auto& ev : rr.value().profile) {
        p.println(ev.node + "  " + ev.op_type + "  " + std::to_string(ev.time_ms));
    }
    return kExitOk;
}
}  // namespace nn
