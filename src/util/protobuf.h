#ifndef NN_PROTOBUF_H
#define NN_PROTOBUF_H

#include "nn/result.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nn {

enum class ProtoWire : int { Varint = 0, Fixed64 = 1, Length = 2, Fixed32 = 5 };

struct ProtoField {
    uint32_t number = 0;
    ProtoWire wire = ProtoWire::Varint;
    uint64_t varint = 0;
    uint64_t fixed64 = 0;
    uint32_t fixed32 = 0;
    std::span<const uint8_t> bytes;
};

class ProtoReader {
public:
    explicit ProtoReader(std::span<const uint8_t> data) : data_(data) {}

    bool done() const { return pos_ >= data_.size(); }
    Result<ProtoField> next();

    static Result<uint64_t> read_varint(std::span<const uint8_t> data, std::size_t& pos);

private:
    std::span<const uint8_t> data_;
    std::size_t pos_ = 0;
};

class ProtoWriter {
public:
    void varint_field(uint32_t number, uint64_t value);
    void bytes_field(uint32_t number, std::span<const uint8_t> value);
    void string_field(uint32_t number, std::string_view value);
    void message_field(uint32_t number, const std::vector<uint8_t>& msg);
    void packed_varint_field(uint32_t number, const std::vector<uint64_t>& values);
    void packed_fixed32_field(uint32_t number, const std::vector<uint32_t>& values);
    void packed_fixed64_field(uint32_t number, const std::vector<uint64_t>& values);
    void fixed32_field(uint32_t number, uint32_t value);
    void fixed64_field(uint32_t number, uint64_t value);

    const std::vector<uint8_t>& data() const { return buf_; }
    std::vector<uint8_t> take() { return std::move(buf_); }

private:
    void write_varint(uint64_t v);
    void write_tag(uint32_t number, ProtoWire wire);
    std::vector<uint8_t> buf_;
};

}  // namespace nn

#endif
