#include "nn/operator.h"

#include <algorithm>
#include <cctype>

namespace nn {
namespace {

std::string lower(std::string_view s) {
    std::string o(s);
    std::transform(o.begin(), o.end(), o.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return o;
}

CanonicalOp from_lowered(const std::string& op) {
    if (op == "conv" || op == "conv1d" || op == "conv2d" || op == "conv3d" ||
        op == "convolution" || op == "conv_2d") {
        return CanonicalOp::Convolution;
    }
    if (op == "depthwiseconv" || op == "depthwiseconv2d" || op == "depthwise_conv2d" ||
        op == "depthwiseconvolution" || op == "depthwise_conv_2d") {
        return CanonicalOp::DepthwiseConvolution;
    }
    if (op == "gemm" || op == "fullyconnected" || op == "fully_connected" || op == "dense" ||
        op == "linear") {
        return CanonicalOp::Dense;
    }
    if (op == "matmul" || op == "batchmatmul") {
        return CanonicalOp::MatMul;
    }
    if (op == "relu" || op == "relu6" || op == "leakyrelu" || op == "prelu" || op == "elu" ||
        op == "selu" || op == "sigmoid" || op == "tanh" || op == "gelu" || op == "silu" ||
        op == "swish" || op == "hardswish" || op == "hardsigmoid" || op == "clip" ||
        op == "thresholdedrelu" || op == "mish") {
        return CanonicalOp::Activation;
    }
    if (op == "maxpool" || op == "maxpool1d" || op == "maxpool2d" || op == "maxpool3d" ||
        op == "averagepool" || op == "averagepool2d" || op == "avgpool" || op == "globalaveragepool" ||
        op == "globalmaxpool" || op == "max_pool_2d" || op == "average_pool_2d" ||
        op == "mean_pooling") {
        return CanonicalOp::Pooling;
    }
    if (op == "batchnormalization" || op == "batchnorm" || op == "layernormalization" ||
        op == "layernorm" || op == "instancenormalization" || op == "groupnormalization" ||
        op == "lrn" || op == "rmsnorm") {
        return CanonicalOp::Normalization;
    }
    if (op == "attention" || op == "multiheadattention" || op == "scaleddotproductattention" ||
        op == "sdpa") {
        return CanonicalOp::Attention;
    }
    if (op == "softmax" || op == "logsoftmax") {
        return CanonicalOp::Softmax;
    }
    if (op == "reshape" || op == "squeeze" || op == "unsqueeze" || op == "flatten" ||
        op == "view") {
        return CanonicalOp::Reshape;
    }
    if (op == "transpose" || op == "permute") {
        return CanonicalOp::Transpose;
    }
    if (op == "concat" || op == "concatenation" || op == "concatv2") {
        return CanonicalOp::Concatenation;
    }
    if (op == "split" || op == "splitv") {
        return CanonicalOp::Split;
    }
    if (op == "gather" || op == "gatherelements" || op == "gathernd") {
        return CanonicalOp::Gather;
    }
    if (op == "gather" || op == "embedding" || op == "embed") {
        return CanonicalOp::Embedding;
    }
    if (op == "add" || op == "sub" || op == "mul" || op == "div" || op == "max" || op == "min" ||
        op == "pow" || op == "abs" || op == "neg" || op == "sqrt" || op == "exp" || op == "log" ||
        op == "and" || op == "or" || op == "xor" || op == "equal" || op == "greater" ||
        op == "less" || op == "where" || op == "mod") {
        return CanonicalOp::Elementwise;
    }
    if (op == "reduce" || op == "reducesum" || op == "reducemean" || op == "reducemax" ||
        op == "reducemin" || op == "reduceprod" || op == "sum" || op == "mean") {
        return CanonicalOp::Reduction;
    }
    if (op == "lstm" || op == "gru" || op == "rnn") {
        return CanonicalOp::Recurrent;
    }
    if (op == "quantize" || op == "quantizelinear" || op == "dynamicquantize" ||
        op == "quantize_v2") {
        return CanonicalOp::Quantize;
    }
    if (op == "dequantize" || op == "dequantizelinear") {
        return CanonicalOp::Dequantize;
    }
    if (op == "pad" || op == "padding") {
        return CanonicalOp::Pad;
    }
    if (op == "resize" || op == "upsample" || op == "resizebilinear" || op == "resizenearest") {
        return CanonicalOp::Resize;
    }
    if (op == "slice" || op == "stridedslice") {
        return CanonicalOp::Slice;
    }
    if (op == "cast" || op == "astype") {
        return CanonicalOp::Cast;
    }
    if (op == "identity" || op == "noop") {
        return CanonicalOp::Identity;
    }
    if (op == "constant" || op == "const") {
        return CanonicalOp::Constant;
    }
    if (op == "dropout") {
        return CanonicalOp::Dropout;
    }
    return CanonicalOp::Unknown;
}

}  // namespace

const char* canonical_op_name(CanonicalOp op) {
    switch (op) {
        case CanonicalOp::Unknown:
            return "Unknown";
        case CanonicalOp::Convolution:
            return "Convolution";
        case CanonicalOp::DepthwiseConvolution:
            return "DepthwiseConv";
        case CanonicalOp::GroupedConvolution:
            return "GroupedConv";
        case CanonicalOp::Dense:
            return "Dense";
        case CanonicalOp::MatMul:
            return "MatMul";
        case CanonicalOp::Gemm:
            return "Gemm";
        case CanonicalOp::Activation:
            return "Activation";
        case CanonicalOp::Pooling:
            return "Pooling";
        case CanonicalOp::Normalization:
            return "Normalization";
        case CanonicalOp::Attention:
            return "Attention";
        case CanonicalOp::Softmax:
            return "Softmax";
        case CanonicalOp::Reshape:
            return "Reshape";
        case CanonicalOp::Transpose:
            return "Transpose";
        case CanonicalOp::Concatenation:
            return "Concatenation";
        case CanonicalOp::Split:
            return "Split";
        case CanonicalOp::Gather:
            return "Gather";
        case CanonicalOp::Embedding:
            return "Embedding";
        case CanonicalOp::Elementwise:
            return "Elementwise";
        case CanonicalOp::Reduction:
            return "Reduction";
        case CanonicalOp::Recurrent:
            return "Recurrent";
        case CanonicalOp::Quantize:
            return "Quantize";
        case CanonicalOp::Dequantize:
            return "Dequantize";
        case CanonicalOp::Pad:
            return "Pad";
        case CanonicalOp::Resize:
            return "Resize";
        case CanonicalOp::Slice:
            return "Slice";
        case CanonicalOp::Cast:
            return "Cast";
        case CanonicalOp::Identity:
            return "Identity";
        case CanonicalOp::Constant:
            return "Constant";
        case CanonicalOp::Dropout:
            return "Dropout";
        case CanonicalOp::Custom:
            return "Custom";
    }
    return "Unknown";
}

CanonicalOp lookup_canonical_op(std::string_view format, std::string_view op_type) {
    (void)format;
    std::string op = lower(op_type);
    // Strip domain prefixes such as "ai.onnx.Conv" leftovers and tflite enum names.
    const auto pos = op.find_last_of('.');
    if (pos != std::string::npos) {
        op = op.substr(pos + 1);
    }
    // Remove underscores for matching CONV_2D etc. after specific aliases.
    CanonicalOp c = from_lowered(op);
    if (c != CanonicalOp::Unknown) {
        return c;
    }
    std::string compact;
    compact.reserve(op.size());
    for (char ch : op) {
        if (ch != '_' && ch != '-') {
            compact.push_back(ch);
        }
    }
    return from_lowered(compact);
}

CanonicalOp canonicalize_op(std::string_view format, std::string_view op_type) {
    return lookup_canonical_op(format, op_type);
}

}  // namespace nn
