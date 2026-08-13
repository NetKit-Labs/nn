#ifndef NN_FORMAT_READERS_H
#define NN_FORMAT_READERS_H

#include "nn/format.h"

#include <memory>

namespace nn {

std::unique_ptr<ModelReader> make_onnx_reader();
std::unique_ptr<ModelReader> make_tflite_reader();
std::unique_ptr<ModelReader> make_gguf_reader();
std::unique_ptr<ModelReader> make_safetensors_reader();
std::unique_ptr<ModelReader> make_pytorch_reader();
std::unique_ptr<ModelReader> make_executorch_reader();
std::unique_ptr<ModelReader> make_coreml_reader();
std::unique_ptr<ModelReader> make_openvino_reader();
std::unique_ptr<ModelReader> make_tensorrt_reader();
std::unique_ptr<ModelReader> make_ncnn_reader();
std::unique_ptr<ModelReader> make_mnn_reader();
std::unique_ptr<ModelReader> make_tfjs_reader();
std::unique_ptr<ModelReader> make_caffe_reader();
std::unique_ptr<ModelReader> make_darknet_reader();
std::unique_ptr<ModelReader> make_mxnet_reader();
std::unique_ptr<ModelReader> make_paddle_reader();
std::unique_ptr<ModelReader> make_keras_reader();
std::unique_ptr<ModelReader> make_tensorflow_reader();

}  // namespace nn

#endif
