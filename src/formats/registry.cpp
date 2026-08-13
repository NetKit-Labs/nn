#include "nn/format.h"

#include "formats/readers.h"

#include "nn/hash.h"
#include "nn/mmap.h"

#include <algorithm>

namespace nn {

void FormatRegistry::register_reader(std::unique_ptr<ModelReader> reader) {
    readers_.push_back(std::move(reader));
}

ModelReader* FormatRegistry::detect(const FileSet& files) const {
    for (const auto& r : readers_) {
        if (r->probe(files)) {
            return r.get();
        }
    }
    return nullptr;
}

Result<ModelReader*> FormatRegistry::require(const FileSet& files) const {
    if (auto* r = detect(files)) {
        return r;
    }
    return error(ErrorCode::UnsupportedFormat,
                 "unrecognized model format: " + files.primary().string());
}

std::vector<FormatInfo> FormatRegistry::formats() const {
    std::vector<FormatInfo> out;
    out.reserve(readers_.size());
    for (const auto& r : readers_) {
        FormatInfo info;
        info.name = r->name();
        info.display_name = r->display_name();
        info.extensions = r->extensions();
        info.capabilities = r->capabilities();
        out.push_back(std::move(info));
    }
    return out;
}

ModelReader* FormatRegistry::find_by_name(std::string_view name) const {
    for (const auto& r : readers_) {
        if (r->name() == name) {
            return r.get();
        }
    }
    return nullptr;
}

void register_builtin_formats(FormatRegistry& registry) {
#ifdef NN_ENABLE_ONNX
    registry.register_reader(make_onnx_reader());
#endif
#ifdef NN_ENABLE_TFLITE
    registry.register_reader(make_tflite_reader());
#endif
#ifdef NN_ENABLE_GGUF
    registry.register_reader(make_gguf_reader());
#endif
#ifdef NN_ENABLE_SAFETENSORS
    registry.register_reader(make_safetensors_reader());
#endif
#ifdef NN_ENABLE_EXECUTORCH
    registry.register_reader(make_executorch_reader());
#endif
#ifdef NN_ENABLE_COREML
    registry.register_reader(make_coreml_reader());
#endif
#ifdef NN_ENABLE_OPENVINO
    registry.register_reader(make_openvino_reader());
#endif
#ifdef NN_ENABLE_NCNN
    registry.register_reader(make_ncnn_reader());
#endif
#ifdef NN_ENABLE_MNN
    registry.register_reader(make_mnn_reader());
#endif
#ifdef NN_ENABLE_TENSORFLOW
    registry.register_reader(make_tensorflow_reader());
#endif
#ifdef NN_ENABLE_KERAS
    registry.register_reader(make_keras_reader());
#endif
#ifdef NN_ENABLE_PYTORCH
    registry.register_reader(make_pytorch_reader());
#endif
#ifdef NN_ENABLE_TFJS
    registry.register_reader(make_tfjs_reader());
#endif
#ifdef NN_ENABLE_TENSORRT
    registry.register_reader(make_tensorrt_reader());
#endif
#ifdef NN_ENABLE_LEGACY
    registry.register_reader(make_caffe_reader());
    registry.register_reader(make_darknet_reader());
    registry.register_reader(make_mxnet_reader());
    registry.register_reader(make_paddle_reader());
#endif
}

FormatRegistry& default_format_registry() {
    static FormatRegistry registry = [] {
        FormatRegistry r;
        register_builtin_formats(r);
        return r;
    }();
    return registry;
}

Result<ModelIR> load_model(const std::filesystem::path& path, const LoadOptions& options) {
    auto files = FileSet::from_path(path);
    if (!files) {
        return files.error();
    }
    auto reader = default_format_registry().require(files.value());
    if (!reader) {
        return reader.error();
    }
    auto model = reader.value()->load(files.value(), options);
    if (!model) {
        return model.error();
    }
    ModelIR& m = model.value();
    m.source_path = path;
    std::error_code ec;
    if (std::filesystem::is_regular_file(path, ec)) {
        m.file_size = static_cast<uint64_t>(std::filesystem::file_size(path, ec));
        m.sha256 = sha256_hex_file(path.string());
    } else if (std::filesystem::is_directory(path, ec)) {
        uint64_t total = 0;
        Sha256 h;
        std::vector<std::filesystem::path> parts;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path, ec)) {
            if (entry.is_regular_file()) {
                parts.push_back(entry.path());
                total += static_cast<uint64_t>(entry.file_size());
            }
        }
        std::sort(parts.begin(), parts.end());
        for (const auto& p : parts) {
            const auto mapped = MappedFile::open(p);
            if (mapped) {
                h.update(mapped.value().span());
            }
        }
        m.file_size = total;
        m.sha256 = const_cast<Sha256&>(h).hex_digest();
    }
    return model;
}

}  // namespace nn
