#ifndef NN_DIFF_H
#define NN_DIFF_H

#include "nn/analysis.h"
#include "nn/model.h"
#include "nn/result.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nn {

struct WeightDiffStats {
    std::string name;
    bool compared = false;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rmse = 0.0;
    double cosine = 0.0;
    double relative = 0.0;
    uint64_t changed_elements = 0;
    uint64_t total_elements = 0;
    double changed_fraction = 0.0;
    std::string note;
};

struct DiffOptions {
    bool summary = true;
    bool graph = true;
    bool weights = false;
    bool tensors = true;
    bool ops = true;
    bool quantization = true;
    bool metadata = true;
    bool numeric = false;
    bool structural = true;
    bool ignore_weights = false;
    bool ignore_metadata = false;
    double atol = 1e-5;
    double rtol = 1e-5;
};

struct ArchitectureChange {
    std::string subject;
    std::string kind;  // added, removed, modified
    std::string detail;
};

struct DiffReport {
    bool identical = true;
    uint64_t old_nodes = 0;
    uint64_t new_nodes = 0;
    uint64_t old_parameters = 0;
    uint64_t new_parameters = 0;
    uint64_t old_weight_bytes = 0;
    uint64_t new_weight_bytes = 0;
    OptionalCount old_macs;
    OptionalCount new_macs;
    uint64_t old_peak_activations = 0;
    uint64_t new_peak_activations = 0;
    bool peak_known = false;
    std::vector<ArchitectureChange> architecture;
    std::vector<ArchitectureChange> quantization;
    std::vector<ArchitectureChange> io;
    std::vector<ArchitectureChange> metadata_changes;
    std::vector<WeightDiffStats> weights;
    std::vector<std::string> notes;
};

Result<DiffReport> diff_models(const ModelIR& old_model,
                               const ModelIR& new_model,
                               const DiffOptions& options = {});

}  // namespace nn

#endif
