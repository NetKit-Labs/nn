#ifndef NN_TENSOR_IO_H
#define NN_TENSOR_IO_H

#include "nn/runtime.h"

#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace nn {

Result<RuntimeTensor> load_tensor_file(const std::filesystem::path& path);
Status save_tensor_file(const std::filesystem::path& path, const RuntimeTensor& tensor);
Result<RuntimeTensor> runtime_from_ir_tensor(const Tensor& t);

RuntimeTensor make_input_tensor(const Tensor& spec, uint64_t seed, bool random);

// Parse --input NAME=FILE or FILE. Multiple specs allowed. FILE may be npy/npz/csv/raw.
Result<std::map<std::string, RuntimeTensor>> bind_model_inputs(
    const ModelIR& model, const std::vector<std::string>& specs, uint64_t seed, bool random_fill);

struct NumericCompare {
    bool comparable = false;
    std::string note;
    double max_abs = 0;
    double mean_abs = 0;
    double rmse = 0;
    double cosine = 0;
    uint64_t changed = 0;
    uint64_t total = 0;
};

NumericCompare compare_numeric(const RuntimeTensor& a, const RuntimeTensor& b, double atol,
                               double rtol);
// Same as compare_numeric, but permutes rank-4 NCHW/NHWC pairs when shapes match that way.
NumericCompare compare_numeric_aligned(const RuntimeTensor& a, const RuntimeTensor& b, double atol,
                                       double rtol);

std::string normalize_tensor_name(std::string_view name);

struct AlignedTensorPair {
    std::string label;
    const RuntimeTensor* a = nullptr;
    const RuntimeTensor* b = nullptr;
};

std::vector<AlignedTensorPair> align_runtime_tensors(const std::vector<RuntimeTensor>& a,
                                                     const std::vector<RuntimeTensor>& b);
std::vector<AlignedTensorPair> align_activation_dumps(const ModelIR& model_a, const RunResult& ra,
                                                      const ModelIR& model_b, const RunResult& rb);

Result<RunResult> eval_model(const ModelIR& model,
                             const std::map<std::string, RuntimeTensor>& inputs,
                             const RuntimeOptions& options, std::string_view backend_name = {});

RuntimeBackend* select_backend(const ModelIR& model, std::string_view name);
RuntimeBackend* select_backend(const ModelIR& model, std::string_view name,
                               const RuntimeOptions& options);

std::vector<double> as_f64(const RuntimeTensor& t);

struct TestCase {
    std::string name;
    std::map<std::string, std::string> inputs;
    std::map<std::string, std::string> expected;
};

Result<std::vector<TestCase>> load_test_manifest(const std::filesystem::path& path);

}  // namespace nn

#endif
