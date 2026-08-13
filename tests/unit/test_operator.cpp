#include "nn/operator.h"
#include "test.h"

TEST(canonical_conv) {
    CHECK(nn::lookup_canonical_op("onnx", "Conv") == nn::CanonicalOp::Convolution);
    CHECK(nn::lookup_canonical_op("tflite", "CONV_2D") == nn::CanonicalOp::Convolution);
    CHECK(nn::lookup_canonical_op("onnx", "Relu") == nn::CanonicalOp::Activation);
    CHECK(nn::lookup_canonical_op("onnx", "Add") == nn::CanonicalOp::Elementwise);
}
