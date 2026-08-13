#ifndef NN_TENSOR_H
#define NN_TENSOR_H

#include "nn/datatype.h"
#include "nn/quantization.h"
#include "nn/result.h"
#include "nn/shape.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace nn {

using TensorId = uint32_t;
inline constexpr TensorId kInvalidTensorId = ~TensorId{0};

// Reference to tensor payload in a backing artifact. The bytes are not
// copied into the IR. Offsets are 64-bit so multi-gigabyte files work.
struct TensorDataReference {
    std::shared_ptr<const std::vector<uint8_t>> owned;
    std::filesystem::path file;
    uint64_t offset = 0;
    uint64_t length = 0;
    bool external = false;

    bool empty() const { return length == 0 && (!owned || owned->empty()); }
};

struct Tensor {
    TensorId id = kInvalidTensorId;
    std::string name;
    DataType dtype = DataType::Unknown;
    Shape shape;
    QuantizationInfo quantization;
    bool constant = false;
    bool model_input = false;
    bool model_output = false;
    uint64_t storage_bytes = 0;
    std::optional<TensorDataReference> data;
    std::string layout;  // "NCHW", "NHWC", ... when known
    std::string producer_node;
    std::vector<std::string> consumer_nodes;
};

Result<uint64_t> tensor_element_count(const Tensor& t);
Result<uint64_t> tensor_storage_bytes(const Tensor& t);

// Copy payload bytes from an owned buffer or a file-backed slice. Does not
// invent data when the tensor has no backing store.
Result<std::vector<uint8_t>> tensor_payload_bytes(const Tensor& t);

}  // namespace nn

#endif
