#include "util/protobuf.h"

namespace nn {

Result<uint64_t> ProtoReader::read_varint(std::span<const uint8_t> data, std::size_t& pos) {
    uint64_t result = 0;
    int shift = 0;
    while (true) {
        if (pos >= data.size()) {
            return error(ErrorCode::ParseError, "truncated protobuf varint");
        }
        if (shift >= 64) {
            return error(ErrorCode::ParseError, "protobuf varint overflow");
        }
        const uint8_t byte = data[pos++];
        result |= static_cast<uint64_t>(byte & 0x7Fu) << shift;
        if ((byte & 0x80u) == 0) {
            return result;
        }
        shift += 7;
    }
}

Result<ProtoField> ProtoReader::next() {
    if (done()) {
        return error(ErrorCode::ParseError, "protobuf reader exhausted");
    }
    auto tag_r = read_varint(data_, pos_);
    if (!tag_r) {
        return tag_r.error();
    }
    const uint64_t tag = tag_r.value();
    ProtoField field;
    field.number = static_cast<uint32_t>(tag >> 3);
    const int wt = static_cast<int>(tag & 7u);
    if (field.number == 0) {
        return error(ErrorCode::ParseError, "protobuf field number 0 is invalid");
    }
    switch (wt) {
        case 0: {
            field.wire = ProtoWire::Varint;
            auto v = read_varint(data_, pos_);
            if (!v) {
                return v.error();
            }
            field.varint = v.value();
            return field;
        }
        case 1: {
            field.wire = ProtoWire::Fixed64;
            if (pos_ + 8 > data_.size()) {
                return error(ErrorCode::ParseError, "truncated protobuf fixed64");
            }
            uint64_t v = 0;
            for (int i = 0; i < 8; ++i) {
                v |= static_cast<uint64_t>(data_[pos_++]) << (8 * i);
            }
            field.fixed64 = v;
            return field;
        }
        case 2: {
            field.wire = ProtoWire::Length;
            auto len_r = read_varint(data_, pos_);
            if (!len_r) {
                return len_r.error();
            }
            const uint64_t len = len_r.value();
            if (len > data_.size() - pos_) {
                return error(ErrorCode::ParseError, "protobuf length-delimited field exceeds buffer");
            }
            field.bytes = data_.subspan(pos_, static_cast<std::size_t>(len));
            pos_ += static_cast<std::size_t>(len);
            return field;
        }
        case 5: {
            field.wire = ProtoWire::Fixed32;
            if (pos_ + 4 > data_.size()) {
                return error(ErrorCode::ParseError, "truncated protobuf fixed32");
            }
            uint32_t v = 0;
            for (int i = 0; i < 4; ++i) {
                v |= static_cast<uint32_t>(data_[pos_++]) << (8 * i);
            }
            field.fixed32 = v;
            return field;
        }
        default:
            return error(ErrorCode::ParseError,
                         "unsupported protobuf wire type " + std::to_string(wt));
    }
}

void ProtoWriter::write_varint(uint64_t v) {
    while (v >= 0x80) {
        buf_.push_back(static_cast<uint8_t>(v | 0x80));
        v >>= 7;
    }
    buf_.push_back(static_cast<uint8_t>(v));
}

void ProtoWriter::write_tag(uint32_t number, ProtoWire wire) {
    write_varint((static_cast<uint64_t>(number) << 3) | static_cast<uint64_t>(wire));
}

void ProtoWriter::varint_field(uint32_t number, uint64_t value) {
    write_tag(number, ProtoWire::Varint);
    write_varint(value);
}

void ProtoWriter::bytes_field(uint32_t number, std::span<const uint8_t> value) {
    write_tag(number, ProtoWire::Length);
    write_varint(value.size());
    buf_.insert(buf_.end(), value.begin(), value.end());
}

void ProtoWriter::string_field(uint32_t number, std::string_view value) {
    bytes_field(number, std::span<const uint8_t>(
                            reinterpret_cast<const uint8_t*>(value.data()), value.size()));
}

void ProtoWriter::message_field(uint32_t number, const std::vector<uint8_t>& msg) {
    bytes_field(number, std::span<const uint8_t>(msg.data(), msg.size()));
}

void ProtoWriter::packed_varint_field(uint32_t number, const std::vector<uint64_t>& values) {
    std::vector<uint8_t> packed;
    ProtoWriter inner;
    for (uint64_t v : values) {
        inner.write_varint(v);
    }
    packed = inner.take();
    bytes_field(number, packed);
}

void ProtoWriter::packed_fixed32_field(uint32_t number, const std::vector<uint32_t>& values) {
    std::vector<uint8_t> packed;
    packed.reserve(values.size() * 4);
    for (uint32_t v : values) {
        for (int i = 0; i < 4; ++i) {
            packed.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
        }
    }
    bytes_field(number, packed);
}

void ProtoWriter::packed_fixed64_field(uint32_t number, const std::vector<uint64_t>& values) {
    std::vector<uint8_t> packed;
    packed.reserve(values.size() * 8);
    for (uint64_t v : values) {
        for (int i = 0; i < 8; ++i) {
            packed.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
        }
    }
    bytes_field(number, packed);
}

void ProtoWriter::fixed32_field(uint32_t number, uint32_t value) {
    write_tag(number, ProtoWire::Fixed32);
    for (int i = 0; i < 4; ++i) {
        buf_.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFFu));
    }
}

void ProtoWriter::fixed64_field(uint32_t number, uint64_t value) {
    write_tag(number, ProtoWire::Fixed64);
    for (int i = 0; i < 8; ++i) {
        buf_.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFFu));
    }
}

}  // namespace nn
