#ifndef NN_NPY_H
#define NN_NPY_H

#include "nn/datatype.h"
#include "nn/result.h"
#include "nn/runtime.h"

#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nn {

Result<RuntimeTensor> load_npy(const std::filesystem::path& path);
Result<RuntimeTensor> load_npy_bytes(std::span<const uint8_t> data, std::string_view name = {});
Status save_npy(const std::filesystem::path& path, const RuntimeTensor& tensor);
Result<std::vector<uint8_t>> encode_npy(const RuntimeTensor& tensor);
Result<std::map<std::string, RuntimeTensor>> load_npz(const std::filesystem::path& path);
Status save_npz(const std::filesystem::path& path,
                const std::map<std::string, RuntimeTensor>& tensors);

}  // namespace nn

#endif
