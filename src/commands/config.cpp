#include "cli/commands.h"
#include "commands/common.h"

namespace nn {
int cmd_config(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "config");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto cfg = load_config();
    if (flag_set(args.value(), "list") || args.value().positionals.empty()) {
        for (const auto& [k, v] : cfg) {
            p.println(k + " = " + v);
        }
        if (cfg.empty()) {
            p.println("# no configuration keys set");
        }
        return kExitOk;
    }
    if (args.value().positionals.size() == 1) {
        auto it = cfg.find(args.value().positionals[0]);
        if (it == cfg.end()) {
            return kExitOk;
        }
        p.println(it->second);
        return kExitOk;
    }
    cfg[args.value().positionals[0]] = args.value().positionals[1];
    auto st = save_user_config(cfg);
    if (!st) {
        return cmd_fail(p, st.error());
    }
    return kExitOk;
}
}  // namespace nn
