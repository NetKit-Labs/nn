#include "cli/commands.h"
#include "commands/common.h"
#include "nn/version.h"

namespace nn {
int cmd_version(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "version");
    p.println("nn " + version_string());
    if (args && flag_set(args.value(), "build-options")) {
        const auto b = build_info();
        p.kv("commit:", b.git_commit);
        p.kv("compiler:", b.compiler);
        p.kv("os:", b.os);
        p.kv("arch:", b.architecture);
        std::string fm;
        for (const auto& f : b.enabled_formats) {
            if (!fm.empty()) {
                fm += ", ";
            }
            fm += f;
        }
        p.kv("formats:", fm);
        std::string rt;
        for (const auto& r : b.enabled_runtimes) {
            if (!rt.empty()) {
                rt += ", ";
            }
            rt += r;
        }
        p.kv("runtimes:", rt);
    }
    return kExitOk;
}
}  // namespace nn
