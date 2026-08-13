#ifndef NN_ANALYSIS_H
#define NN_ANALYSIS_H

#include "nn/model.h"
#include "nn/result.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nn {

struct OptionalCount {
    bool known = false;
    uint64_t value = 0;
};

struct ComputeCost {
    OptionalCount macs;
    OptionalCount flops;
    OptionalCount int_ops;
    OptionalCount float_ops;
    OptionalCount memory_reads;
    OptionalCount memory_writes;
};

struct NodeCompute {
    NodeId node_id = kInvalidNodeId;
    std::string name;
    std::string op_type;
    CanonicalOp canonical = CanonicalOp::Unknown;
    ComputeCost cost;
};

struct ComputeReport {
    ComputeCost total;
    std::vector<NodeCompute> nodes;
    uint64_t unknown_node_count = 0;
};

ComputeReport analyze_compute(const ModelIR& model);
ComputeCost estimate_node_compute(const Graph& graph, const Node& node);

struct TensorLifetime {
    TensorId tensor_id = kInvalidTensorId;
    std::string name;
    int birth = -1;  // producing node index, -1 = input/constant
    int death = -1;  // last consuming node index
    uint64_t bytes = 0;
    bool persistent = false;
};

struct MemoryPlanEntry {
    TensorId tensor_id = kInvalidTensorId;
    std::string name;
    uint64_t offset = 0;
    uint64_t size = 0;
    int birth = -1;
    int death = -1;
};

struct MemoryReport {
    uint64_t file_size = 0;
    uint64_t weight_bytes = 0;
    uint64_t constant_bytes = 0;
    uint64_t persistent_bytes = 0;
    uint64_t peak_activation_bytes = 0;
    uint64_t scratch_bytes = 0;
    uint64_t estimated_ram_bytes = 0;
    bool peak_known = false;
    std::vector<TensorLifetime> lifetimes;
    std::vector<MemoryPlanEntry> plan;
    std::vector<std::string> notes;
};

struct MemoryOptions {
    bool plan = false;
    uint64_t alignment = 16;
};

MemoryReport analyze_memory(const ModelIR& model, const MemoryOptions& options = {});
std::vector<TensorLifetime> analyze_lifetimes(const Graph& graph);

struct QuantReport {
    uint64_t quantized_tensors = 0;
    uint64_t float_tensors = 0;
    uint64_t integer_tensors = 0;
    uint64_t per_channel = 0;
    uint64_t per_tensor = 0;
    uint64_t quantize_nodes = 0;
    uint64_t dequantize_nodes = 0;
    std::vector<std::string> issues;
};

QuantReport analyze_quantization(const ModelIR& model);

struct SparsityOptions {
    double threshold = 0.0;
};

struct TensorSparsity {
    std::string name;
    uint64_t elements = 0;
    uint64_t zeros = 0;
    uint64_t near_zeros = 0;
    double zero_fraction = 0.0;
    bool computed = false;
};

struct SparsityReport {
    uint64_t tensors_considered = 0;
    uint64_t tensors_computed = 0;
    uint64_t total_elements = 0;
    uint64_t total_zeros = 0;
    double overall_zero_fraction = 0.0;
    std::vector<TensorSparsity> tensors;
    std::vector<std::string> notes;
};

SparsityReport analyze_sparsity(const ModelIR& model, const SparsityOptions& options = {});

struct MatchedNode {
    const Node* a = nullptr;
    const Node* b = nullptr;
};

std::vector<MatchedNode> match_graphs(const Graph& a, const Graph& b);

std::string canonicalize_graph_text(const ModelIR& model);

enum class LintSeverity { Info, Warning, Error };

struct LintIssue {
    LintSeverity severity = LintSeverity::Warning;
    std::string code;
    std::string message;
    std::string location;
};

struct LintReport {
    std::vector<LintIssue> issues;
    uint64_t errors = 0;
    uint64_t warnings = 0;
    uint64_t infos = 0;
};

LintReport lint_model(const ModelIR& model);

struct InspectOptions {
    bool summary = false;
    bool all = false;
    bool metadata = false;
    bool inputs = true;
    bool outputs = true;
    bool ops = false;
    bool tensors = false;
    bool weights = false;
    bool quantization = false;
    bool subgraphs = false;
    bool raw = false;
};

struct InspectReport {
    ModelIR model;
    ComputeReport compute;
    MemoryReport memory;
    uint64_t parameter_count = 0;
    bool parameter_count_known = false;
};

Result<InspectReport> inspect_model(const ModelIR& model);

struct HashOptions {
    bool graph = false;
    bool weights = false;
    std::string tensor_name;
    bool canonical = false;
};

struct HashReport {
    std::string artifact_sha256;
    std::string graph_sha256;
    std::string weights_sha256;
    std::string canonical_sha256;
    std::string tensor_name;
    std::string tensor_sha256;
};

Result<HashReport> hash_model(const ModelIR& model, const HashOptions& options);

}  // namespace nn

#endif
