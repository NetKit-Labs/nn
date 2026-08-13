#ifndef NN_COMPAT_H
#define NN_COMPAT_H

#include "nn/model.h"

#include <string>
#include <vector>

namespace nn {

struct CompatTable {
    std::string runtime;
    std::string version;
    std::vector<std::string> ops;  // native or canonical names, lowercased
    std::string notes;
};

struct CompatReport {
    std::string runtime;
    std::string table_version;
    uint64_t supported = 0;
    uint64_t total = 0;
    std::vector<std::string> unsupported;
    bool compatible = false;
};

std::vector<CompatTable> builtin_compat_tables();
const CompatTable* find_compat_table(std::string_view runtime, std::string_view version);
CompatReport check_compat(const ModelIR& model, std::string_view runtime, std::string_view version);

}  // namespace nn

#endif
