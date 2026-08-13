#include "nn/shape.h"

#include "util/overflow.h"

#include <sstream>

namespace nn {

std::string Dimension::to_string() const {
    if (value) {
        return std::to_string(*value);
    }
    if (!symbol.empty()) {
        return symbol;
    }
    return "?";
}

bool Shape::is_static() const {
    for (const auto& d : dims) {
        if (!d.is_static()) {
            return false;
        }
    }
    return true;
}

Result<uint64_t> Shape::element_count() const {
    uint64_t acc = 1;
    for (const auto& d : dims) {
        if (!d.value) {
            return error(ErrorCode::InvalidArgument, "shape has dynamic dimension");
        }
        if (*d.value < 0) {
            return error(ErrorCode::InvalidArgument, "negative dimension");
        }
        auto r = checked_mul_u64(acc, static_cast<uint64_t>(*d.value));
        if (!r) {
            return r.error();
        }
        acc = r.value();
    }
    return acc;
}

std::string Shape::to_string() const {
    if (dims.empty()) {
        return "scalar";
    }
    std::ostringstream os;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i) {
            os << 'x';
        }
        os << dims[i].to_string();
    }
    return os.str();
}

Shape scalar_shape() { return {}; }

Shape shape_from_ints(const std::vector<int64_t>& dims) {
    Shape s;
    s.dims.reserve(dims.size());
    for (int64_t d : dims) {
        Dimension dim;
        dim.value = d;
        s.dims.push_back(std::move(dim));
    }
    return s;
}

}  // namespace nn
