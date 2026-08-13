#ifndef NN_ZIP_H
#define NN_ZIP_H

#include "nn/result.h"

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace nn {

struct ZipMember {
    std::string name;
    std::vector<uint8_t> bytes;
};

// Read uncompressed (ZIP stored) members. Deflated entries return an error.
Result<std::vector<ZipMember>> zip_read_stored(std::span<const uint8_t> data);

// Write a ZIP archive using store (no compression).
std::vector<uint8_t> zip_write_stored(const std::vector<ZipMember>& members);

uint32_t crc32_ieee(std::span<const uint8_t> data);

}  // namespace nn

#endif
