#ifndef NN_SHAPE_H
#define NN_SHAPE_H

#include "nn/result.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nn {

struct Dimension {
    std::optional<int64_t> value;
    std::string symbol;

    bool is_static() const { return value.has_value(); }
    bool is_dynamic() const { return !value.has_value(); }

    std::string to_string() const;
    bool operator==(const Dimension& o) const {
        return value == o.value && symbol == o.symbol;
    }
};

struct Shape {
    std::vector<Dimension> dims;

    std::size_t rank() const { return dims.size(); }
    bool is_static() const;
    bool is_scalar() const { return dims.empty(); }

    // Product of static dimensions. Error if any dimension is dynamic or
    // the product would overflow.
    Result<uint64_t> element_count() const;

    std::string to_string() const;  // e.g. 1x3x224x224 or 1xCxHxW
    bool operator==(const Shape& o) const { return dims == o.dims; }
};

Shape scalar_shape();
Shape shape_from_ints(const std::vector<int64_t>& dims);

}  // namespace nn

#endif
