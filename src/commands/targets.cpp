#include "cli/commands.h"
#include "nn/target.h"
#include "util/format_text.h"

#include <cstdio>

namespace nn {
int cmd_targets(const GlobalOptions& g) {
    Printer p(g);
    p.println("NAME                 CPU              RAM         FLASH");
    p.println("-------------------------------------------------------");
    for (const auto& t : builtin_targets()) {
        char line[96];
        std::snprintf(line, sizeof(line), "%-20.20s %-16.16s %-11s %s", t.name.c_str(), t.cpu.c_str(),
                      human_bytes(t.ram_bytes).c_str(), human_bytes(t.storage_bytes).c_str());
        p.println(line);
    }
    return kExitOk;
}
}  // namespace nn
