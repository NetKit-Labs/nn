#ifndef NN_BINARY_H
#define NN_BINARY_H

#include "nn/result.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace nn {

class BinaryReader {
public:
    explicit BinaryReader(std::span<const uint8_t> data, std::string source = {})
        : data_(data), source_(std::move(source)) {}

    uint64_t size() const { return data_.size(); }
    uint64_t position() const { return pos_; }
    uint64_t remaining() const { return pos_ <= data_.size() ? data_.size() - pos_ : 0; }
    const uint8_t* data() const { return data_.data(); }

    void seek(uint64_t pos) { pos_ = pos; }
    Status require(uint64_t n) const;

    Result<uint8_t> u8();
    Result<uint16_t> u16le();
    Result<uint32_t> u32le();
    Result<uint64_t> u64le();
    Result<uint16_t> u16be();
    Result<uint32_t> u32be();
    Result<uint64_t> u64be();
    Result<int32_t> i32le();
    Result<int64_t> i64le();
    Result<float> f32le();
    Result<double> f64le();
    Result<std::span<const uint8_t>> bytes(uint64_t n);
    Result<std::string> string(uint64_t n);

    std::span<const uint8_t> remaining_span() const {
        if (pos_ >= data_.size()) {
            return {};
        }
        return data_.subspan(static_cast<std::size_t>(pos_));
    }

private:
    Error bounds(std::string_view what) const;
    std::span<const uint8_t> data_;
    uint64_t pos_ = 0;
    std::string source_;
};

}  // namespace nn

#endif
