#include "nn/convert.h"

namespace nn {

std::vector<ConversionRoute> conversion_routes() {
    return {
        {"onnx", "onnx", true, "re-serialize the loaded graph as ONNX"},
        {"ir", "onnx", true, "write any loaded graph IR as ONNX"},
        {"onnx", "tflite", false, "requires an external conversion adapter"},
        {"onnx", "coreml", false, "requires an external conversion adapter"},
        {"pytorch", "onnx", false, "requires torch.onnx; pickle load is refused"},
        {"savedmodel", "tflite", false, "requires an external conversion adapter"},
        {"tflite", "onnx", false, "requires an external conversion adapter"},
        {"keras", "onnx", false, "requires an external conversion adapter"},
    };
}

}  // namespace nn
