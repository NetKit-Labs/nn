#ifndef NN_ONNX_PROTO_H
#define NN_ONNX_PROTO_H

#include "nn/datatype.h"
#include "nn/format.h"
#include "nn/model.h"
#include "nn/result.h"

#include "util/protobuf.h"

#include <span>

namespace nn {
namespace onnx {

DataType tensor_dtype_from_onnx(int32_t t);
int32_t onnx_from_tensor_dtype(DataType t);

Result<ModelIR> parse_model_proto(std::span<const uint8_t> bytes, const LoadOptions& options,
                                  const std::filesystem::path& source);

}  // namespace onnx
}  // namespace nn

#endif
