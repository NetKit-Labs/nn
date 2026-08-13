#include "cli/commands.h"
#include "commands/common.h"
#include "nn/target.h"

namespace nn {
int cmd_partition(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "partition");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    auto accf = flag_value(args.value(), "accelerator");
    if (!accf) {
        return p.usage_error("nn: --accelerator is required",
                             "nn partition <model> --accelerator FILE");
    }
    auto acc = load_accelerator_file(*accf);
    if (!acc) {
        return cmd_fail(p, acc.error());
    }
    auto r = partition_model(model.value(), acc.value());
    for (const auto& part : r.partitions) {
        p.println("Partition " + std::to_string(part.index));
        p.println("    device: " + part.device);
        p.println("    nodes: " + std::to_string(part.nodes.size()));
        if (!part.reason.empty()) {
            p.println("    reason: " + part.reason);
        }
    }
    p.kv("Accelerator launches:", std::to_string(r.accelerator_launches));
    return kExitOk;
}
}  // namespace nn
