#include "cli/commands.h"
#include "nn/format.h"
#include "nn/runtime.h"
#include "nn/version.h"

#ifdef __linux__
#include <unistd.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

namespace nn {
int cmd_doctor(const GlobalOptions& g) {
    Printer p(g);
    const auto b = build_info();
    p.kv("NN version", b.version);
    p.println();
    p.println("Core");
    auto formats = default_format_registry().formats();
    for (const auto& f : formats) {
        p.kv("  " + f.display_name, f.capabilities.read ? "yes" : "no");
    }
    p.println();
    p.println("Optional runtimes");
    for (auto* be : default_runtime_registry().list()) {
        p.kv("  " + be->name(), be->available() ? (be->version().empty() ? "available" : be->version())
                                                : "unavailable");
    }
    p.println();
    p.println("System");
    p.kv("  OS", b.os);
    p.kv("  Architecture", b.architecture);
    p.kv("  Compiler", b.compiler);
    p.kv("  Git commit", b.git_commit);
    int cpus = 0;
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    cpus = static_cast<int>(si.dwNumberOfProcessors);
#elif defined(__APPLE__)
    size_t sz = sizeof(cpus);
    sysctlbyname("hw.ncpu", &cpus, &sz, nullptr, 0);
#elif defined(__linux__)
    cpus = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
#endif
    if (cpus > 0) {
        p.kv("  CPUs", std::to_string(cpus));
    }
    return kExitOk;
}
}  // namespace nn
