#include "nn/datatype.h"
#include "test.h"

TEST(datatype_sizes) {
    CHECK(nn::datatype_size(nn::DataType::Float32) == 4);
    CHECK(nn::datatype_size(nn::DataType::Float16) == 2);
    CHECK(nn::datatype_size(nn::DataType::Int8) == 1);
    CHECK(nn::datatype_bits(nn::DataType::Int4) == 4);
    CHECK(nn::datatype_from_name("float32").value() == nn::DataType::Float32);
    CHECK(nn::datatype_is_float(nn::DataType::Float16));
    CHECK(nn::datatype_is_integer(nn::DataType::UInt8));
}
