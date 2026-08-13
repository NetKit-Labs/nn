#include "cli/commands.h"
#include "commands/common.h"

namespace nn {
int cmd_help(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "help");
    if (args && !args.value().positionals.empty()) {
        const CommandSpec* spec = find_command(args.value().positionals[0]);
        if (!spec) {
            p.errln("nn: unknown command '" + args.value().positionals[0] + "'");
            return kExitUsage;
        }
        print_command_help(p, *spec);
        return kExitOk;
    }
    print_top_help(p);
    return kExitOk;
}
}  // namespace nn
