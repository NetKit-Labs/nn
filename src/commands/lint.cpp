#include "cli/commands.h"
#include "commands/common.h"
#include "nn/analysis.h"

namespace nn {
int cmd_lint(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "lint");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    auto r = lint_model(model.value());
    for (const auto& i : r.issues) {
        const char* sev = "info";
        if (i.severity == LintSeverity::Warning) {
            sev = "warning";
        }
        if (i.severity == LintSeverity::Error) {
            sev = "error";
        }
        p.println(std::string(sev) + ": [" + i.code + "] " + i.message +
                  (i.location.empty() ? "" : " (" + i.location + ")"));
    }
    p.kv("errors:", std::to_string(r.errors));
    p.kv("warnings:", std::to_string(r.warnings));
    return r.errors ? kExitValidation : kExitOk;
}
}  // namespace nn
