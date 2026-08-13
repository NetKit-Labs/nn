#ifndef NN_ATTRIBUTE_H
#define NN_ATTRIBUTE_H

#include "nn/tensor.h"

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace nn {

struct Attribute {
    enum class Kind {
        Int,
        Float,
        String,
        Ints,
        Floats,
        Strings,
        Tensor
    };

    Kind kind = Kind::Int;
    int64_t i = 0;
    double f = 0.0;
    std::string s;
    std::vector<int64_t> ints;
    std::vector<double> floats;
    std::vector<std::string> strings;
    Tensor tensor;

    static Attribute from_int(int64_t v);
    static Attribute from_float(double v);
    static Attribute from_string(std::string v);
    static Attribute from_ints(std::vector<int64_t> v);
    static Attribute from_floats(std::vector<double> v);
    static Attribute from_strings(std::vector<std::string> v);

    std::string to_string() const;
};

using AttributeMap = std::map<std::string, Attribute>;

}  // namespace nn

#endif
