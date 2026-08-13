#ifndef NN_FLATBUFFER_H
#define NN_FLATBUFFER_H

#include "nn/result.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace nn {

class FlatBuffer {
public:
    explicit FlatBuffer(std::span<const uint8_t> data) : data_(data) {}

    Result<uint32_t> root_offset() const;
    bool in_bounds(uint64_t off, uint64_t n) const;

    Result<int32_t> i32(uint64_t off) const;
    Result<uint32_t> u32(uint64_t off) const;
    Result<int16_t> i16(uint64_t off) const;
    Result<uint16_t> u16(uint64_t off) const;
    Result<uint8_t> u8(uint64_t off) const;
    Result<float> f32(uint64_t off) const;
    Result<std::string> string_at(uint64_t off) const;

    // Table at absolute offset `table_off`.
    Result<int16_t> field_offset(uint64_t table_off, uint16_t field_id) const;
    bool has_field(uint64_t table_off, uint16_t field_id) const;

    Result<int32_t> table_i32(uint64_t table_off, uint16_t field_id, int32_t def = 0) const;
    Result<uint32_t> table_u32(uint64_t table_off, uint16_t field_id, uint32_t def = 0) const;
    Result<int16_t> table_i16(uint64_t table_off, uint16_t field_id, int16_t def = 0) const;
    Result<uint8_t> table_u8(uint64_t table_off, uint16_t field_id, uint8_t def = 0) const;
    Result<float> table_f32(uint64_t table_off, uint16_t field_id, float def = 0) const;
    Result<std::string> table_string(uint64_t table_off, uint16_t field_id) const;
    Result<uint64_t> table_offset(uint64_t table_off, uint16_t field_id) const;  // indirect

    // Vector whose offset field is stored at `vec_off` (start of vector object).
    Result<uint32_t> vec_len(uint64_t vec_off) const;
    Result<uint64_t> vec_elem_offset(uint64_t vec_off, uint32_t index, uint32_t elem_size) const;
    Result<uint64_t> vec_uoffset(uint64_t vec_off, uint32_t index) const;

    std::span<const uint8_t> span() const { return data_; }

private:
    std::span<const uint8_t> data_;
};

}  // namespace nn

#endif
