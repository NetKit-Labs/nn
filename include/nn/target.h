#ifndef NN_TARGET_H
#define NN_TARGET_H

#include "nn/datatype.h"
#include "nn/model.h"
#include "nn/result.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nn {

struct TargetCapabilities {
    bool simd = false;
    bool fpu = false;
    std::string simd_name;
    std::string fpu_name;
    std::vector<std::string> supported_canonical_ops;
    std::vector<DataType> native_types;
};

struct PerformanceModel {
    uint64_t clock_hz = 0;
    uint64_t macs_per_cycle = 1;
    bool has_estimate = false;
};

struct Target {
    std::string name;
    std::string cpu;
    uint64_t ram_bytes = 0;
    uint64_t storage_bytes = 0;
    int cores = 1;
    std::vector<DataType> native_types;
    TargetCapabilities capabilities;
    PerformanceModel performance;
};

struct Accelerator {
    std::string name;
    std::vector<std::string> supported_ops;
    std::vector<DataType> data_types;
    uint64_t macs_per_cycle = 0;
    uint64_t clock_hz = 0;
    uint64_t local_memory_bytes = 0;
    uint64_t alignment = 1;
};

struct TargetFit {
    std::string target_name;
    bool fits_storage = false;
    bool fits_ram = false;
    uint64_t model_bytes = 0;
    uint64_t activation_bytes = 0;
    uint64_t scratch_bytes = 0;
    uint64_t runtime_overhead_bytes = 0;
    uint64_t total_ram_bytes = 0;
    std::vector<std::string> unsupported_ops;
    std::vector<std::string> unsupported_types;
    std::vector<std::string> notes;
    bool overall_fit = false;
};

struct Partition {
    int index = 0;
    std::string device;  // accelerator or cpu
    std::vector<NodeId> nodes;
    std::string reason;
};

struct PartitionReport {
    std::vector<Partition> partitions;
    uint64_t transfer_bytes = 0;
    uint64_t accelerator_launches = 0;
};

std::vector<Target> builtin_targets();
const Target* find_builtin_target(std::string_view name);
Result<Target> load_target_file(const std::filesystem::path& path);
Result<Accelerator> load_accelerator_file(const std::filesystem::path& path);
TargetFit evaluate_target(const ModelIR& model, const Target& target);
PartitionReport partition_model(const ModelIR& model, const Accelerator& accel);

}  // namespace nn

#endif
