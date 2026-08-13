#ifndef NN_ONNX_WRITER_H
#define NN_ONNX_WRITER_H

#include "nn/model.h"
#include "nn/result.h"

#include <filesystem>
#include <vector>

namespace nn {

struct SimpleOnnxSpec {
    std::string name = "graph";
    std::string op_type = "Add";
    std::vector<int64_t> input_shape = {1, 4};
    DataType dtype = DataType::Float32;
    int inputs = 2;
    bool with_weight = false;
    std::vector<int64_t> weight_shape;
    std::vector<float> weight;
    std::vector<int64_t> output_shape;
};

Result<std::vector<uint8_t>> encode_simple_onnx(const SimpleOnnxSpec& spec);
Status write_simple_onnx(const std::filesystem::path& path, const SimpleOnnxSpec& spec);
Status write_onnx_model(const std::filesystem::path& path, const ModelIR& model);

}  // namespace nn

#endif
