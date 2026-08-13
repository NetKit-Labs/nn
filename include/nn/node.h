#ifndef NN_NODE_H
#define NN_NODE_H

#include "nn/attribute.h"
#include "nn/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nn {

using NodeId = uint32_t;
inline constexpr NodeId kInvalidNodeId = ~NodeId{0};

enum class CanonicalOp {
    Unknown = 0,
    Convolution,
    DepthwiseConvolution,
    GroupedConvolution,
    Dense,  // Gemm / MatMul / FullyConnected
    MatMul,
    Gemm,
    Activation,
    Pooling,
    Normalization,
    Attention,
    Softmax,
    Reshape,
    Transpose,
    Concatenation,
    Split,
    Gather,
    Embedding,
    Elementwise,
    Reduction,
    Recurrent,
    Quantize,
    Dequantize,
    Pad,
    Resize,
    Slice,
    Cast,
    Identity,
    Constant,
    Dropout,
    Custom
};

const char* canonical_op_name(CanonicalOp op);

struct Node {
    NodeId id = kInvalidNodeId;
    std::string name;
    std::string domain;
    std::string op_type;  // native name
    CanonicalOp canonical = CanonicalOp::Unknown;
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
    AttributeMap attributes;
    std::string doc;
};

CanonicalOp canonicalize_op(std::string_view format, std::string_view op_type);

}  // namespace nn

#endif
