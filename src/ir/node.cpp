#include "nn/attribute.h"

#include <sstream>

namespace nn {

Attribute Attribute::from_int(int64_t v) {
    Attribute a;
    a.kind = Kind::Int;
    a.i = v;
    return a;
}

Attribute Attribute::from_float(double v) {
    Attribute a;
    a.kind = Kind::Float;
    a.f = v;
    return a;
}

Attribute Attribute::from_string(std::string v) {
    Attribute a;
    a.kind = Kind::String;
    a.s = std::move(v);
    return a;
}

Attribute Attribute::from_ints(std::vector<int64_t> v) {
    Attribute a;
    a.kind = Kind::Ints;
    a.ints = std::move(v);
    return a;
}

Attribute Attribute::from_floats(std::vector<double> v) {
    Attribute a;
    a.kind = Kind::Floats;
    a.floats = std::move(v);
    return a;
}

Attribute Attribute::from_strings(std::vector<std::string> v) {
    Attribute a;
    a.kind = Kind::Strings;
    a.strings = std::move(v);
    return a;
}

std::string Attribute::to_string() const {
    std::ostringstream os;
    switch (kind) {
        case Kind::Int:
            os << i;
            break;
        case Kind::Float:
            os << f;
            break;
        case Kind::String:
            os << s;
            break;
        case Kind::Ints:
            os << '[';
            for (std::size_t n = 0; n < ints.size(); ++n) {
                if (n) {
                    os << ", ";
                }
                os << ints[n];
            }
            os << ']';
            break;
        case Kind::Floats:
            os << '[';
            for (std::size_t n = 0; n < floats.size(); ++n) {
                if (n) {
                    os << ", ";
                }
                os << floats[n];
            }
            os << ']';
            break;
        case Kind::Strings:
            os << '[';
            for (std::size_t n = 0; n < strings.size(); ++n) {
                if (n) {
                    os << ", ";
                }
                os << strings[n];
            }
            os << ']';
            break;
        case Kind::Tensor:
            os << "<tensor " << tensor.name << '>';
            break;
    }
    return os.str();
}

}  // namespace nn
