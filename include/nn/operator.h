#ifndef NN_OPERATOR_H
#define NN_OPERATOR_H

#include "nn/node.h"

#include <string>
#include <string_view>

namespace nn {

struct OperatorTaxonomy {
    std::string format;
    std::string native;
    CanonicalOp canonical;
};

CanonicalOp lookup_canonical_op(std::string_view format, std::string_view op_type);

}  // namespace nn

#endif
