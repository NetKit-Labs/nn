#ifndef NN_DETECT_H
#define NN_DETECT_H

#include "nn/result.h"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace nn {

std::string extension_lower(const std::filesystem::path& p);
bool has_magic(std::span<const uint8_t> data, std::span<const uint8_t> magic);
bool file_has_magic(const std::filesystem::path& path, std::span<const uint8_t> magic);
Result<std::vector<std::string>> zip_list_names(std::span<const uint8_t> data);
bool looks_like_zip(std::span<const uint8_t> data);
bool looks_like_hdf5(std::span<const uint8_t> data);
bool looks_like_pickle(std::span<const uint8_t> data);

}  // namespace nn

#endif
