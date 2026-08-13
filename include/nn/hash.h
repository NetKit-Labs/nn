#ifndef NN_HASH_H
#define NN_HASH_H

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace nn {

class Sha256 {
public:
    Sha256();
    void update(std::span<const uint8_t> data);
    void update(std::string_view text);
    std::array<uint8_t, 32> digest();
    std::string hex_digest();

private:
    void transform(const uint8_t block[64]);
    uint64_t bit_len_ = 0;
    uint32_t state_[8]{};
    uint8_t buffer_[64]{};
    std::size_t buffer_len_ = 0;
};

std::string sha256_hex(std::span<const uint8_t> data);
std::string sha256_hex_file(const std::string& path);

}  // namespace nn

#endif
