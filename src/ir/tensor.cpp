#include "nn/tensor.h"

#include "nn/mmap.h"
#include "util/overflow.h"

namespace nn {

Result<uint64_t> tensor_element_count(const Tensor& t) {
    return t.shape.element_count();
}

Result<uint64_t> tensor_storage_bytes(const Tensor& t) {
    if (t.storage_bytes > 0) {
        return t.storage_bytes;
    }
    auto n = t.shape.element_count();
    if (!n) {
        return n.error();
    }
    const int bits = datatype_bits(t.dtype);
    if (bits <= 0) {
        return error(ErrorCode::InvalidArgument, "unknown dtype size for tensor " + t.name);
    }
    uint64_t bits_total = 0;
    if (mul_overflow_u64(n.value(), static_cast<uint64_t>(bits), &bits_total)) {
        return error(ErrorCode::Overflow, "tensor bit-size overflow: " + t.name);
    }
    return (bits_total + 7u) / 8u;
}

Result<std::vector<uint8_t>> tensor_payload_bytes(const Tensor& t) {
    if (!t.data) {
        return error(ErrorCode::FileError, "tensor has no payload: " + t.name);
    }
    if (t.data->owned) {
        return *t.data->owned;
    }
    if (t.data->file.empty()) {
        return error(ErrorCode::FileError, "tensor payload is not resident: " + t.name);
    }
    auto mapped = MappedFile::open(t.data->file);
    if (!mapped) {
        return mapped.error();
    }
    const uint64_t length = t.data->length ? t.data->length : t.storage_bytes;
    auto sl = mapped.value().slice(t.data->offset, length);
    if (sl.empty() && length != 0) {
        return error(ErrorCode::FileError, "tensor payload slice is out of range: " + t.name);
    }
    return std::vector<uint8_t>(sl.begin(), sl.end());
}

}  // namespace nn
