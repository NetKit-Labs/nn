#ifndef NN_DATATYPE_H
#define NN_DATATYPE_H

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace nn {

enum class DataType {
    Unknown = 0,
    Float64,
    Float32,
    Float16,
    BFloat16,
    Float8E4M3,
    Float8E5M2,
    Int64,
    Int32,
    Int16,
    Int8,
    Int4,
    UInt64,
    UInt32,
    UInt16,
    UInt8,
    UInt4,
    Bool,
    Complex64,
    Complex128,
    String
};

const char* datatype_name(DataType t);
std::optional<DataType> datatype_from_name(std::string_view name);

// Storage size of one element in bits. Returns 0 for Unknown/String.
int datatype_bits(DataType t);

// Storage size of one element in bytes, rounded up for sub-byte types.
// Returns 0 for Unknown/String.
std::size_t datatype_size(DataType t);

bool datatype_is_float(DataType t);
bool datatype_is_integer(DataType t);
bool datatype_is_signed_integer(DataType t);

}  // namespace nn

#endif
