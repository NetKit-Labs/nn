#include "cli/commands.h"
#include "commands/common.h"
#include "nn/analysis.h"

#include <cstdlib>

namespace nn {
int cmd_sparsity(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "sparsity");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    SparsityOptions opt;
    if (auto t = flag_value(args.value(), "threshold")) {
        opt.threshold = std::strtod(t->c_str(), nullptr);
    }
    auto r = analyze_sparsity(model.value(), opt);
    p.kv("Tensors considered:", std::to_string(r.tensors_considered));
    p.kv("Tensors computed:", std::to_string(r.tensors_computed));
    p.kv("Zero fraction:", std::to_string(r.overall_zero_fraction));
    for (const auto& t : r.tensors) {
        if (t.computed) {
            p.println(t.name + "  zeros=" + std::to_string(t.zeros) + "/" +
                      std::to_string(t.elements));
        }
    }
    for (const auto& n : r.notes) {
        p.errln("nn: warning: " + n);
    }
    return kExitOk;
}
}  // namespace nn
