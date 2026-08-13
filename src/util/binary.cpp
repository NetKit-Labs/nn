#include "util/binary.h"

#include <cstring>

namespace nn {

Error BinaryReader::bounds(std::string_view what) const {
    std::string msg = "read past end of buffer";
    if (!what.empty()) {
        msg += " while reading ";
        msg += what;
    }
    if (!source_.empty()) {
        msg += " in ";
        msg += source_;
    }
    msg += " (offset " + std::to_string(pos_) + ", size " + std::to_string(data_.size()) + ")";
    return error(ErrorCode::ParseError, std::move(msg));
}

Status BinaryReader::require(uint64_t n) const {
    if (n > remaining()) {
        return Status::err(bounds("bytes"));
    }
    return Status::ok();
}

Result<uint8_t> BinaryReader::u8() {
    if (remaining() < 1) {
        return bounds("u8");
    }
    return data_[static_cast<std::size_t>(pos_++)];
}

Result<uint16_t> BinaryReader::u16le() {
    auto a = u8();
    if (!a) {
        return a.error();
    }
    auto b = u8();
    if (!b) {
        return b.error();
    }
    return static_cast<uint16_t>(static_cast<uint16_t>(a.value()) |
                                 (static_cast<uint16_t>(b.value()) << 8));
}

Result<uint32_t> BinaryReader::u32le() {
    auto lo = u16le();
    if (!lo) {
        return lo.error();
    }
    auto hi = u16le();
    if (!hi) {
        return hi.error();
    }
    return static_cast<uint32_t>(lo.value()) | (static_cast<uint32_t>(hi.value()) << 16);
}

Result<uint64_t> BinaryReader::u64le() {
    auto lo = u32le();
    if (!lo) {
        return lo.error();
    }
    auto hi = u32le();
    if (!hi) {
        return hi.error();
    }
    return static_cast<uint64_t>(lo.value()) | (static_cast<uint64_t>(hi.value()) << 32);
}

Result<uint16_t> BinaryReader::u16be() {
    auto a = u8();
    if (!a) {
        return a.error();
    }
    auto b = u8();
    if (!b) {
        return b.error();
    }
    return static_cast<uint16_t>((static_cast<uint16_t>(a.value()) << 8) |
                                 static_cast<uint16_t>(b.value()));
}

Result<uint32_t> BinaryReader::u32be() {
    auto hi = u16be();
    if (!hi) {
        return hi.error();
    }
    auto lo = u16be();
    if (!lo) {
        return lo.error();
    }
    return (static_cast<uint32_t>(hi.value()) << 16) | static_cast<uint32_t>(lo.value());
}

Result<uint64_t> BinaryReader::u64be() {
    auto hi = u32be();
    if (!hi) {
        return hi.error();
    }
    auto lo = u32be();
    if (!lo) {
        return lo.error();
    }
    return (static_cast<uint64_t>(hi.value()) << 32) | static_cast<uint64_t>(lo.value());
}

Result<int32_t> BinaryReader::i32le() {
    auto v = u32le();
    if (!v) {
        return v.error();
    }
    return static_cast<int32_t>(v.value());
}

Result<int64_t> BinaryReader::i64le() {
    auto v = u64le();
    if (!v) {
        return v.error();
    }
    return static_cast<int64_t>(v.value());
}

Result<float> BinaryReader::f32le() {
    auto v = u32le();
    if (!v) {
        return v.error();
    }
    float f = 0;
    std::memcpy(&f, &v.value(), sizeof(f));
    return f;
}

Result<double> BinaryReader::f64le() {
    auto v = u64le();
    if (!v) {
        return v.error();
    }
    double d = 0;
    std::memcpy(&d, &v.value(), sizeof(d));
    return d;
}

Result<std::span<const uint8_t>> BinaryReader::bytes(uint64_t n) {
    if (n > remaining()) {
        return bounds("bytes");
    }
    auto out = data_.subspan(static_cast<std::size_t>(pos_), static_cast<std::size_t>(n));
    pos_ += n;
    return out;
}

Result<std::string> BinaryReader::string(uint64_t n) {
    auto b = bytes(n);
    if (!b) {
        return b.error();
    }
    return std::string(reinterpret_cast<const char*>(b.value().data()), b.value().size());
}

}  // namespace nn
