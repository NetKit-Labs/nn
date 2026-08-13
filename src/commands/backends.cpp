#include "cli/commands.h"
#include "nn/runtime.h"

namespace nn {
int cmd_backends(const GlobalOptions& g) {
    Printer p(g);
    p.println("NAME            AVAILABLE   VERSION");
    p.println("-----------------------------------");
    for (auto* be : default_runtime_registry().list()) {
        p.println(be->name() + "    " + (be->available() ? "yes" : "no") + "    " + be->version());
    }
    return kExitOk;
}
}  // namespace nn
