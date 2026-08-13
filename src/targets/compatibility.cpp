#include "nn/target.h"

#include "nn/analysis.h"

#include <algorithm>

namespace nn {

TargetFit evaluate_target(const ModelIR& model, const Target& target) {
    TargetFit f;
    f.target_name = target.name;
    f.model_bytes = model.file_size;
    auto mem = analyze_memory(model);
    f.activation_bytes = mem.peak_activation_bytes;
    f.scratch_bytes = mem.scratch_bytes;
    f.runtime_overhead_bytes = 24 * 1024;  // documented estimate of interpreter overhead
    f.notes.push_back("runtime overhead is estimated (24 KiB), not measured");
    f.total_ram_bytes = f.activation_bytes + f.scratch_bytes + f.runtime_overhead_bytes;
    f.fits_storage = target.storage_bytes == 0 || f.model_bytes <= target.storage_bytes;
    f.fits_ram = target.ram_bytes == 0 || f.total_ram_bytes <= target.ram_bytes;
    const Graph* g = primary_graph(model);
    if (g) {
        for (const auto& n : g->nodes) {
            if (n.canonical == CanonicalOp::Unknown) {
                f.unsupported_ops.push_back(n.op_type);
            }
        }
        for (const auto& t : g->tensors) {
            if (t.dtype == DataType::Unknown) {
                continue;
            }
            bool ok = false;
            for (auto d : target.native_types) {
                if (d == t.dtype) {
                    ok = true;
                    break;
                }
            }
            if (!ok && t.constant == false) {
                const std::string name = datatype_name(t.dtype);
                if (std::find(f.unsupported_types.begin(), f.unsupported_types.end(), name) ==
                    f.unsupported_types.end()) {
                    f.unsupported_types.push_back(name);
                }
            }
        }
    }
    if (!f.fits_ram) {
        f.notes.push_back("peak RAM exceeds target RAM");
    }
    if (!f.fits_storage) {
        f.notes.push_back("model file exceeds target flash/storage");
    }
    f.overall_fit = f.fits_ram && f.fits_storage && f.unsupported_ops.empty();
    return f;
}

}  // namespace nn
