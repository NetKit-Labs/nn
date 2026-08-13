#include "cli/commands.h"
#include "commands/common.h"
#include "nn/analysis.h"

namespace nn {
int cmd_canonicalize(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "canonicalize");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    p.print(canonicalize_graph_text(model.value()));
    return kExitOk;
}
}  // namespace nn
