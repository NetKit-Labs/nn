#include "nn/runtime.h"

#include "formats/onnx/writer.h"
#include "nn/json.h"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>

namespace nn {
namespace {

Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "nn");
    return env;
}

ONNXTensorElementDataType to_ort_dtype(DataType t) {
    switch (t) {
        case DataType::Float32:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case DataType::Float64:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        case DataType::Float16:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case DataType::BFloat16:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
        case DataType::Int64:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case DataType::Int32:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case DataType::Int16:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case DataType::Int8:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case DataType::UInt64:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
        case DataType::UInt32:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
        case DataType::UInt16:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        case DataType::UInt8:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case DataType::Bool:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        default:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

DataType from_ort_dtype(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return DataType::Float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return DataType::Float64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return DataType::Float16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return DataType::BFloat16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return DataType::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return DataType::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return DataType::Int16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return DataType::Int8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            return DataType::UInt64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return DataType::UInt32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return DataType::UInt16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return DataType::UInt8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return DataType::Bool;
        default:
            return DataType::Unknown;
    }
}

std::vector<int64_t> shape_ints(const Shape& s) {
    std::vector<int64_t> d;
    d.reserve(s.dims.size());
    for (const auto& dim : s.dims) {
        d.push_back(dim.value.value_or(1));
    }
    return d;
}

Error ort_error(const Ort::Exception& e) {
    return error(ErrorCode::ExecutionFailure, std::string("onnxruntime: ") + e.what());
}

class OrtNnSession final : public Session {
public:
    OrtNnSession(std::filesystem::path path, const RuntimeOptions& options)
        : path_(std::move(path)), options_(options) {}

    Status init() {
        try {
            Ort::SessionOptions opts;
            if (options_.threads > 0) {
                opts.SetIntraOpNumThreads(options_.threads);
            }
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (options_.profile) {
                profile_prefix_ = (std::filesystem::temp_directory_path() / "nn-ort-profile").string();
                opts.EnableProfiling(profile_prefix_.c_str());
            }
#ifdef _WIN32
            session_ = std::make_unique<Ort::Session>(ort_env(), path_.wstring().c_str(), opts);
#else
            session_ = std::make_unique<Ort::Session>(ort_env(), path_.c_str(), opts);
#endif
            Ort::AllocatorWithDefaultOptions alloc;
            const std::size_t nin = session_->GetInputCount();
            const std::size_t nout = session_->GetOutputCount();
            in_names_.reserve(nin);
            out_names_.reserve(nout);
            for (std::size_t i = 0; i < nin; ++i) {
                auto n = session_->GetInputNameAllocated(i, alloc);
                in_names_.emplace_back(n.get());
            }
            for (std::size_t i = 0; i < nout; ++i) {
                auto n = session_->GetOutputNameAllocated(i, alloc);
                out_names_.emplace_back(n.get());
            }
            in_c_.reserve(in_names_.size());
            out_c_.reserve(out_names_.size());
            for (const auto& s : in_names_) {
                in_c_.push_back(s.c_str());
            }
            for (const auto& s : out_names_) {
                out_c_.push_back(s.c_str());
            }
            return Status::ok();
        } catch (const Ort::Exception& e) {
            return Status::err(ort_error(e));
        }
    }

    std::vector<std::string> input_names() const override { return in_names_; }
    std::vector<std::string> output_names() const override { return out_names_; }

    Result<RunResult> run(const std::map<std::string, RuntimeTensor>& inputs) override {
        try {
            Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<Ort::Value> in_vals;
            in_vals.reserve(in_names_.size());
            std::vector<RuntimeTensor> held;
            held.reserve(in_names_.size());
            for (const auto& name : in_names_) {
                auto it = inputs.find(name);
                if (it == inputs.end()) {
                    return error(ErrorCode::MissingArgument, "missing input tensor " + name);
                }
                held.push_back(it->second);
                RuntimeTensor& t = held.back();
                const auto dtype = to_ort_dtype(t.dtype);
                if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
                    return error(ErrorCode::UnsupportedOperator,
                                 "onnxruntime cannot bind dtype for " + name);
                }
                auto shape = shape_ints(t.shape);
                if (t.bytes.empty()) {
                    return error(ErrorCode::ExecutionFailure, "empty input tensor " + name);
                }
                in_vals.push_back(Ort::Value::CreateTensor(
                    static_cast<const OrtMemoryInfo*>(mem), t.bytes.data(), t.bytes.size(),
                    shape.data(), shape.size(), dtype));
            }
            const auto t0 = std::chrono::steady_clock::now();
            auto out_vals = session_->Run(Ort::RunOptions{nullptr}, in_c_.data(), in_vals.data(),
                                          in_vals.size(), out_c_.data(), out_c_.size());
            const auto t1 = std::chrono::steady_clock::now();
            RunResult rr;
            rr.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            for (std::size_t i = 0; i < out_vals.size(); ++i) {
                auto info = out_vals[i].GetTensorTypeAndShapeInfo();
                RuntimeTensor t;
                t.name = i < out_names_.size() ? out_names_[i] : "";
                t.dtype = from_ort_dtype(info.GetElementType());
                t.shape = shape_from_ints(info.GetShape());
                const void* raw = out_vals[i].GetTensorRawData();
                const std::size_t nbytes = info.GetElementCount() * datatype_size(t.dtype);
                t.bytes.resize(nbytes);
                if (nbytes && raw) {
                    std::memcpy(t.bytes.data(), raw, nbytes);
                }
                rr.outputs.push_back(std::move(t));
            }
            if (options_.profile) {
                collect_profile(rr);
            }
            return rr;
        } catch (const Ort::Exception& e) {
            return ort_error(e);
        }
    }

private:
    void collect_profile(RunResult& rr) {
        try {
            Ort::AllocatorWithDefaultOptions alloc;
            auto file_ptr = session_->EndProfilingAllocated(alloc);
            const std::string file = file_ptr.get() ? file_ptr.get() : "";
            if (file.empty()) {
                return;
            }
            std::ifstream in(file);
            if (!in) {
                return;
            }
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            auto j = parse_json(text);
            if (!j || !j.value().is_array()) {
                return;
            }
            for (const auto& ev : j.value().as_array()) {
                if (!ev.is_object()) {
                    continue;
                }
                const std::string cat = ev.contains("cat") && ev.at("cat").is_string()
                                            ? ev.at("cat").as_string()
                                            : "";
                if (cat != "Node" && cat != "node") {
                    continue;
                }
                ProfileEvent pe;
                pe.node = ev.contains("name") && ev.at("name").is_string() ? ev.at("name").as_string()
                                                                           : "";
                if (ev.contains("args") && ev.at("args").is_object()) {
                    const auto& args = ev.at("args");
                    if (args.contains("op_name") && args.at("op_name").is_string()) {
                        pe.op_type = args.at("op_name").as_string();
                    }
                }
                if (ev.contains("dur") && ev.at("dur").is_number()) {
                    pe.time_ms = ev.at("dur").as_number() / 1000.0;  // Chrome trace: microseconds
                }
                rr.profile.push_back(std::move(pe));
            }
            rr.profiled = !rr.profile.empty();
            std::error_code ec;
            std::filesystem::remove(file, ec);
        } catch (...) {
            // Profiling is best-effort; the run itself succeeded.
        }
    }

    std::filesystem::path path_;
    RuntimeOptions options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> in_names_;
    std::vector<std::string> out_names_;
    std::vector<const char*> in_c_;
    std::vector<const char*> out_c_;
    std::string profile_prefix_;
};

class OnnxRuntimeBackend final : public RuntimeBackend {
public:
    std::string name() const override { return "onnxruntime"; }
    std::string version() const override {
        try {
            return Ort::GetVersionString();
        } catch (...) {
            return "unknown";
        }
    }
    bool compiled() const override { return true; }
    bool available() const override { return true; }

    bool supports(const ModelIR& model) const override {
        if (model.source_format == "onnx") {
            return true;
        }
        std::error_code ec;
        return !model.source_path.empty() && model.source_path.extension() == ".onnx" &&
               std::filesystem::exists(model.source_path, ec);
    }

    Result<std::unique_ptr<Session>> create_session(const ModelIR& model,
                                                    const RuntimeOptions& options) override {
        std::filesystem::path path = model.source_path;
        std::error_code ec;
        if (path.empty() || !std::filesystem::exists(path, ec) || path.extension() != ".onnx") {
            path = std::filesystem::temp_directory_path() /
                   ("nn-ort-" + std::to_string(reinterpret_cast<uintptr_t>(&model)) + ".onnx");
            auto st = write_onnx_model(path, model);
            if (!st) {
                return st.error();
            }
        }
        try {
            auto sess = std::make_unique<OrtNnSession>(path, options);
            auto st = sess->init();
            if (!st) {
                return st.error();
            }
            return std::unique_ptr<Session>(std::move(sess));
        } catch (const Ort::Exception& e) {
            return ort_error(e);
        }
    }
};

}  // namespace

std::unique_ptr<RuntimeBackend> make_onnxruntime_backend() {
    return std::make_unique<OnnxRuntimeBackend>();
}

}  // namespace nn
