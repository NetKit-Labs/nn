#include "nn/datatype.h"

#include <algorithm>
#include <cctype>

namespace nn {

const char* datatype_name(DataType t) {
    switch (t) {
        case DataType::Unknown:
            return "unknown";
        case DataType::Float64:
            return "float64";
        case DataType::Float32:
            return "float32";
        case DataType::Float16:
            return "float16";
        case DataType::BFloat16:
            return "bfloat16";
        case DataType::Float8E4M3:
            return "float8e4m3";
        case DataType::Float8E5M2:
            return "float8e5m2";
        case DataType::Int64:
            return "int64";
        case DataType::Int32:
            return "int32";
        case DataType::Int16:
            return "int16";
        case DataType::Int8:
            return "int8";
        case DataType::Int4:
            return "int4";
        case DataType::UInt64:
            return "uint64";
        case DataType::UInt32:
            return "uint32";
        case DataType::UInt16:
            return "uint16";
        case DataType::UInt8:
            return "uint8";
        case DataType::UInt4:
            return "uint4";
        case DataType::Bool:
            return "bool";
        case DataType::Complex64:
            return "complex64";
        case DataType::Complex128:
            return "complex128";
        case DataType::String:
            return "string";
    }
    return "unknown";
}

std::optional<DataType> datatype_from_name(std::string_view name) {
    std::string s(name);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "float64" || s == "double" || s == "f64") {
        return DataType::Float64;
    }
    if (s == "float32" || s == "float" || s == "f32") {
        return DataType::Float32;
    }
    if (s == "float16" || s == "fp16" || s == "f16" || s == "half") {
        return DataType::Float16;
    }
    if (s == "bfloat16" || s == "bf16") {
        return DataType::BFloat16;
    }
    if (s == "int64" || s == "i64" || s == "long") {
        return DataType::Int64;
    }
    if (s == "int32" || s == "i32") {
        return DataType::Int32;
    }
    if (s == "int16" || s == "i16") {
        return DataType::Int16;
    }
    if (s == "int8" || s == "i8") {
        return DataType::Int8;
    }
    if (s == "int4" || s == "i4") {
        return DataType::Int4;
    }
    if (s == "uint64" || s == "u64") {
        return DataType::UInt64;
    }
    if (s == "uint32" || s == "u32") {
        return DataType::UInt32;
    }
    if (s == "uint16" || s == "u16") {
        return DataType::UInt16;
    }
    if (s == "uint8" || s == "u8") {
        return DataType::UInt8;
    }
    if (s == "uint4" || s == "u4") {
        return DataType::UInt4;
    }
    if (s == "bool" || s == "boolean") {
        return DataType::Bool;
    }
    if (s == "string" || s == "str") {
        return DataType::String;
    }
    return std::nullopt;
}

int datatype_bits(DataType t) {
    switch (t) {
        case DataType::Float64:
        case DataType::Int64:
        case DataType::UInt64:
        case DataType::Complex64:
            return 64;
        case DataType::Float32:
        case DataType::Int32:
        case DataType::UInt32:
            return 32;
        case DataType::Float16:
        case DataType::BFloat16:
        case DataType::Int16:
        case DataType::UInt16:
            return 16;
        case DataType::Float8E4M3:
        case DataType::Float8E5M2:
        case DataType::Int8:
        case DataType::UInt8:
        case DataType::Bool:
            return 8;
        case DataType::Int4:
        case DataType::UInt4:
            return 4;
        case DataType::Complex128:
            return 128;
        case DataType::Unknown:
        case DataType::String:
            return 0;
    }
    return 0;
}

std::size_t datatype_size(DataType t) {
    const int bits = datatype_bits(t);
    if (bits <= 0) {
        return 0;
    }
    return static_cast<std::size_t>((bits + 7) / 8);
}

bool datatype_is_float(DataType t) {
    return t == DataType::Float64 || t == DataType::Float32 || t == DataType::Float16 ||
           t == DataType::BFloat16 || t == DataType::Float8E4M3 || t == DataType::Float8E5M2;
}

bool datatype_is_integer(DataType t) {
    return t == DataType::Int64 || t == DataType::Int32 || t == DataType::Int16 ||
           t == DataType::Int8 || t == DataType::Int4 || t == DataType::UInt64 ||
           t == DataType::UInt32 || t == DataType::UInt16 || t == DataType::UInt8 ||
           t == DataType::UInt4;
}

bool datatype_is_signed_integer(DataType t) {
    return t == DataType::Int64 || t == DataType::Int32 || t == DataType::Int16 ||
           t == DataType::Int8 || t == DataType::Int4;
}

}  // namespace nn
