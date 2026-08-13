#include "util/flatbuffer.h"

#include <cstring>

namespace nn {

bool FlatBuffer::in_bounds(uint64_t off, uint64_t n) const {
    if (n > data_.size()) {
        return false;
    }
    return off <= data_.size() - n;
}

Result<uint32_t> FlatBuffer::root_offset() const {
    auto o = u32(0);
    if (!o) {
        return o.error();
    }
    if (!in_bounds(o.value(), 4)) {
        return error(ErrorCode::ParseError, "FlatBuffer root offset out of range");
    }
    return o;
}

Result<int32_t> FlatBuffer::i32(uint64_t off) const {
    if (!in_bounds(off, 4)) {
        return error(ErrorCode::ParseError, "FlatBuffer i32 out of range");
    }
    int32_t v = 0;
    std::memcpy(&v, data_.data() + off, 4);
    return v;
}

Result<uint32_t> FlatBuffer::u32(uint64_t off) const {
    if (!in_bounds(off, 4)) {
        return error(ErrorCode::ParseError, "FlatBuffer u32 out of range");
    }
    uint32_t v = 0;
    std::memcpy(&v, data_.data() + off, 4);
    return v;
}

Result<int16_t> FlatBuffer::i16(uint64_t off) const {
    if (!in_bounds(off, 2)) {
        return error(ErrorCode::ParseError, "FlatBuffer i16 out of range");
    }
    int16_t v = 0;
    std::memcpy(&v, data_.data() + off, 2);
    return v;
}

Result<uint16_t> FlatBuffer::u16(uint64_t off) const {
    if (!in_bounds(off, 2)) {
        return error(ErrorCode::ParseError, "FlatBuffer u16 out of range");
    }
    uint16_t v = 0;
    std::memcpy(&v, data_.data() + off, 2);
    return v;
}

Result<uint8_t> FlatBuffer::u8(uint64_t off) const {
    if (!in_bounds(off, 1)) {
        return error(ErrorCode::ParseError, "FlatBuffer u8 out of range");
    }
    return data_[static_cast<std::size_t>(off)];
}

Result<float> FlatBuffer::f32(uint64_t off) const {
    if (!in_bounds(off, 4)) {
        return error(ErrorCode::ParseError, "FlatBuffer f32 out of range");
    }
    float v = 0;
    std::memcpy(&v, data_.data() + off, 4);
    return v;
}

Result<std::string> FlatBuffer::string_at(uint64_t off) const {
    auto len = i32(off);
    if (!len) {
        return len.error();
    }
    if (len.value() < 0) {
        return error(ErrorCode::ParseError, "negative FlatBuffer string length");
    }
    const uint64_t n = static_cast<uint64_t>(len.value());
    if (!in_bounds(off + 4, n)) {
        return error(ErrorCode::ParseError, "FlatBuffer string exceeds buffer");
    }
    return std::string(reinterpret_cast<const char*>(data_.data() + off + 4),
                       static_cast<std::size_t>(n));
}

Result<int16_t> FlatBuffer::field_offset(uint64_t table_off, uint16_t field_id) const {
    auto soff = i32(table_off);
    if (!soff) {
        return soff.error();
    }
    // Official FlatBuffers stores soffset_t as the distance to subtract:
    // vtable = table - soffset (positive when the vtable sits before the table).
    const int64_t vtable = static_cast<int64_t>(table_off) - soff.value();
    if (vtable < 0) {
        return error(ErrorCode::ParseError, "FlatBuffer vtable offset negative");
    }
    auto vts = i16(static_cast<uint64_t>(vtable));
    if (!vts) {
        return vts.error();
    }
    const int field_off = 4 + static_cast<int>(field_id) * 2;
    if (field_off + 2 > vts.value()) {
        return static_cast<int16_t>(0);
    }
    return i16(static_cast<uint64_t>(vtable) + static_cast<uint64_t>(field_off));
}

bool FlatBuffer::has_field(uint64_t table_off, uint16_t field_id) const {
    auto o = field_offset(table_off, field_id);
    return o && o.value() != 0;
}

Result<int32_t> FlatBuffer::table_i32(uint64_t table_off, uint16_t field_id, int32_t def) const {
    auto o = field_offset(table_off, field_id);
    if (!o) {
        return o.error();
    }
    if (o.value() == 0) {
        return def;
    }
    return i32(table_off + static_cast<uint64_t>(o.value()));
}

Result<uint32_t> FlatBuffer::table_u32(uint64_t table_off, uint16_t field_id, uint32_t def) const {
    auto o = field_offset(table_off, field_id);
    if (!o) {
        return o.error();
    }
    if (o.value() == 0) {
        return def;
    }
    return u32(table_off + static_cast<uint64_t>(o.value()));
}

Result<int16_t> FlatBuffer::table_i16(uint64_t table_off, uint16_t field_id, int16_t def) const {
    auto o = field_offset(table_off, field_id);
    if (!o) {
        return o.error();
    }
    if (o.value() == 0) {
        return def;
    }
    return i16(table_off + static_cast<uint64_t>(o.value()));
}

Result<uint8_t> FlatBuffer::table_u8(uint64_t table_off, uint16_t field_id, uint8_t def) const {
    auto o = field_offset(table_off, field_id);
    if (!o) {
        return o.error();
    }
    if (o.value() == 0) {
        return def;
    }
    return u8(table_off + static_cast<uint64_t>(o.value()));
}

Result<float> FlatBuffer::table_f32(uint64_t table_off, uint16_t field_id, float def) const {
    auto o = field_offset(table_off, field_id);
    if (!o) {
        return o.error();
    }
    if (o.value() == 0) {
        return def;
    }
    return f32(table_off + static_cast<uint64_t>(o.value()));
}

Result<uint64_t> FlatBuffer::table_offset(uint64_t table_off, uint16_t field_id) const {
    auto o = field_offset(table_off, field_id);
    if (!o) {
        return o.error();
    }
    if (o.value() == 0) {
        return static_cast<uint64_t>(0);
    }
    const uint64_t field_pos = table_off + static_cast<uint64_t>(o.value());
    auto rel = u32(field_pos);
    if (!rel) {
        return rel.error();
    }
    return field_pos + rel.value();
}

Result<std::string> FlatBuffer::table_string(uint64_t table_off, uint16_t field_id) const {
    auto off = table_offset(table_off, field_id);
    if (!off) {
        return off.error();
    }
    if (off.value() == 0) {
        return std::string();
    }
    return string_at(off.value());
}

Result<uint32_t> FlatBuffer::vec_len(uint64_t vec_off) const {
    auto n = u32(vec_off);
    if (!n) {
        return n.error();
    }
    return n;
}

Result<uint64_t> FlatBuffer::vec_elem_offset(uint64_t vec_off, uint32_t index, uint32_t elem_size) const {
    auto n = vec_len(vec_off);
    if (!n) {
        return n.error();
    }
    if (index >= n.value()) {
        return error(ErrorCode::ParseError, "FlatBuffer vector index out of range");
    }
    const uint64_t off = vec_off + 4 + static_cast<uint64_t>(index) * elem_size;
    if (!in_bounds(off, elem_size)) {
        return error(ErrorCode::ParseError, "FlatBuffer vector element out of range");
    }
    return off;
}

Result<uint64_t> FlatBuffer::vec_uoffset(uint64_t vec_off, uint32_t index) const {
    auto elem = vec_elem_offset(vec_off, index, 4);
    if (!elem) {
        return elem.error();
    }
    auto rel = u32(elem.value());
    if (!rel) {
        return rel.error();
    }
    return elem.value() + rel.value();
}

}  // namespace nn
