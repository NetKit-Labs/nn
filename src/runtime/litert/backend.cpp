#include "nn/runtime.h"

#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_profiler.h"
#include "litert/c/litert_tensor_buffer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>

namespace nn {
namespace {

Error litert_error(LiteRtStatus st, std::string_view what) {
    const char* name = LiteRtGetStatusString(st);
    std::ostringstream os;
    os << "litert: " << what;
    if (name && *name) {
        os << ": " << name;
    } else {
        os << " (status " << static_cast<int>(st) << ")";
    }
    return error(ErrorCode::ExecutionFailure, os.str());
}

bool litert_ok(LiteRtStatus st, std::string_view what, Status& out) {
    if (st == kLiteRtStatusOk) {
        return true;
    }
    out = Status::err(litert_error(st, what));
    return false;
}

DataType from_litert_dtype(LiteRtElementType t) {
    switch (t) {
        case kLiteRtElementTypeFloat32:
            return DataType::Float32;
        case kLiteRtElementTypeFloat64:
            return DataType::Float64;
        case kLiteRtElementTypeFloat16:
            return DataType::Float16;
        case kLiteRtElementTypeBFloat16:
            return DataType::BFloat16;
        case kLiteRtElementTypeFloat8E4M3FN:
            return DataType::Float8E4M3;
        case kLiteRtElementTypeFloat8E5M2:
            return DataType::Float8E5M2;
        case kLiteRtElementTypeInt64:
            return DataType::Int64;
        case kLiteRtElementTypeInt32:
            return DataType::Int32;
        case kLiteRtElementTypeInt16:
            return DataType::Int16;
        case kLiteRtElementTypeInt8:
            return DataType::Int8;
        case kLiteRtElementTypeInt4:
            return DataType::Int4;
        case kLiteRtElementTypeUInt64:
            return DataType::UInt64;
        case kLiteRtElementTypeUInt32:
            return DataType::UInt32;
        case kLiteRtElementTypeUInt16:
            return DataType::UInt16;
        case kLiteRtElementTypeUInt8:
            return DataType::UInt8;
        case kLiteRtElementTypeUInt4:
            return DataType::UInt4;
        case kLiteRtElementTypeBool:
            return DataType::Bool;
        case kLiteRtElementTypeComplex64:
            return DataType::Complex64;
        case kLiteRtElementTypeComplex128:
            return DataType::Complex128;
        case kLiteRtElementTypeTfString:
            return DataType::String;
        default:
            return DataType::Unknown;
    }
}

Shape shape_from_layout(const LiteRtLayout& layout) {
    std::vector<int64_t> dims;
    dims.reserve(layout.rank);
    for (unsigned i = 0; i < layout.rank; ++i) {
        dims.push_back(layout.dimensions[i]);
    }
    return shape_from_ints(dims);
}

const RuntimeTensor* find_input(const std::map<std::string, RuntimeTensor>& inputs,
                                const std::string& name, std::size_t index,
                                const std::vector<std::string>& fallback_names) {
    auto it = inputs.find(name);
    if (it != inputs.end()) {
        return &it->second;
    }
    if (index < fallback_names.size()) {
        it = inputs.find(fallback_names[index]);
        if (it != inputs.end()) {
            return &it->second;
        }
    }
    if (inputs.size() == 1 && fallback_names.size() <= 1) {
        return &inputs.begin()->second;
    }
    return nullptr;
}

struct EnvDeleter {
    void operator()(std::remove_pointer_t<LiteRtEnvironment>* p) const {
        if (p) {
            LiteRtDestroyEnvironment(p);
        }
    }
};
struct ModelDeleter {
    void operator()(std::remove_pointer_t<LiteRtModel>* p) const {
        if (p) {
            LiteRtDestroyModel(p);
        }
    }
};
struct OptionsDeleter {
    void operator()(std::remove_pointer_t<LiteRtOptions>* p) const {
        if (p) {
            LiteRtDestroyOptions(p);
        }
    }
};
struct CompiledDeleter {
    void operator()(std::remove_pointer_t<LiteRtCompiledModel>* p) const {
        if (p) {
            LiteRtDestroyCompiledModel(p);
        }
    }
};
struct BufferDeleter {
    void operator()(std::remove_pointer_t<LiteRtTensorBuffer>* p) const {
        if (p) {
            LiteRtDestroyTensorBuffer(p);
        }
    }
};

using EnvPtr = std::unique_ptr<std::remove_pointer_t<LiteRtEnvironment>, EnvDeleter>;
using ModelPtr = std::unique_ptr<std::remove_pointer_t<LiteRtModel>, ModelDeleter>;
using OptionsPtr = std::unique_ptr<std::remove_pointer_t<LiteRtOptions>, OptionsDeleter>;
using CompiledPtr = std::unique_ptr<std::remove_pointer_t<LiteRtCompiledModel>, CompiledDeleter>;
using BufferPtr = std::unique_ptr<std::remove_pointer_t<LiteRtTensorBuffer>, BufferDeleter>;

void overlay_ir_shape(LiteRtRankedTensorType& t, const ModelIR& model, const std::string& name,
                      std::size_t index, bool is_input) {
    const Graph* g = primary_graph(model);
    if (!g) {
        return;
    }
    const Tensor* ir = g->find_tensor_by_name(name);
    if (!ir) {
        const auto& ids = is_input ? g->inputs : g->outputs;
        if (index < ids.size()) {
            ir = g->find_tensor(ids[index]);
        }
    }
    if (!ir || !ir->shape.is_static() || ir->shape.dims.empty()) {
        return;
    }
    bool needs = t.layout.rank == 0;
    for (unsigned i = 0; !needs && i < t.layout.rank; ++i) {
        needs = t.layout.dimensions[i] <= 0;
    }
    if (!needs && t.layout.rank == ir->shape.rank()) {
        return;
    }
    const auto& dims = ir->shape.dims;
    t.layout.rank = static_cast<unsigned>(
        std::min(dims.size(), static_cast<std::size_t>(LITERT_TENSOR_MAX_RANK)));
    t.layout.has_strides = false;
    for (unsigned i = 0; i < t.layout.rank; ++i) {
        t.layout.dimensions[i] = static_cast<int32_t>(dims[i].value.value_or(1));
        t.layout.strides[i] = 0;
    }
}

Result<std::size_t> packed_bytes(const LiteRtRankedTensorType& t) {
    const DataType dt = from_litert_dtype(t.element_type);
    const std::size_t elem = datatype_size(dt);
    if (elem == 0) {
        return error(ErrorCode::UnsupportedOperator, "litert: unsupported tensor element type");
    }
    std::size_t n = 1;
    for (unsigned i = 0; i < t.layout.rank; ++i) {
        if (t.layout.dimensions[i] <= 0) {
            return error(ErrorCode::ExecutionFailure,
                         "litert: cannot allocate a buffer for a dynamic dimension");
        }
        n *= static_cast<std::size_t>(t.layout.dimensions[i]);
    }
    return n * elem;
}

Status make_cpu_buffer(LiteRtEnvironment env, const LiteRtRankedTensorType& type,
                       LiteRtTensorBufferRequirements reqs, BufferPtr& out, std::string_view what) {
    LiteRtTensorBuffer buf = nullptr;
    if (reqs && LiteRtCreateManagedTensorBufferFromRequirements(env, &type, reqs, &buf) ==
                    kLiteRtStatusOk &&
        buf) {
        out.reset(buf);
        return Status::ok();
    }
    auto nbytes = packed_bytes(type);
    if (!nbytes) {
        return Status::err(nbytes.error());
    }
    Status st;
    if (!litert_ok(LiteRtCreateManagedTensorBuffer(env, kLiteRtTensorBufferTypeHostMemory, &type,
                                                   nbytes.value(), &buf),
                   std::string("create ") + std::string(what), st)) {
        return st;
    }
    out.reset(buf);
    return Status::ok();
}

class LitertSession final : public Session {
public:
    LitertSession(ModelIR model, RuntimeOptions options)
        : model_ir_(std::move(model)), options_(std::move(options)) {}

    Status init() {
        if (LiteRtLogger logger = LiteRtGetDefaultLogger()) {
            LiteRtSetMinLoggerSeverity(logger, kLiteRtLogSeverityError);
        }

        LiteRtEnvOption env_opts[2] = {};
        env_opts[0].tag = kLiteRtEnvOptionTagAutoRegisterAccelerators;
        env_opts[0].value.type = kLiteRtAnyTypeInt;
        env_opts[0].value.int_value = kLiteRtHwAcceleratorCpu;
        env_opts[1].tag = kLiteRtEnvOptionTagMinLoggerSeverity;
        env_opts[1].value.type = kLiteRtAnyTypeInt;
        env_opts[1].value.int_value = kLiteRtLogSeverityError;

        LiteRtEnvironment env = nullptr;
        Status st;
        if (!litert_ok(LiteRtCreateEnvironment(2, env_opts, &env), "create environment", st)) {
            return st;
        }
        env_.reset(env);

        LiteRtModel model = nullptr;
        if (!litert_ok(LiteRtCreateModelFromFile(env_.get(), model_ir_.source_path.string().c_str(),
                                                 &model),
                       "load model", st)) {
            return st;
        }
        model_.reset(model);

        LiteRtOptions opts = nullptr;
        if (!litert_ok(LiteRtCreateOptions(&opts), "create options", st)) {
            return st;
        }
        options_handle_.reset(opts);
        if (!litert_ok(LiteRtSetOptionsHardwareAccelerators(options_handle_.get(),
                                                            kLiteRtHwAcceleratorCpu),
                       "set CPU accelerator", st)) {
            return st;
        }

        LiteRtCompiledModel compiled = nullptr;
        if (!litert_ok(LiteRtCreateCompiledModel(env_.get(), model_.get(), options_handle_.get(),
                                                 &compiled),
                       "compile model", st)) {
            return st;
        }
        compiled_.reset(compiled);

        LiteRtParamIndex nsig = 0;
        if (!litert_ok(LiteRtGetNumModelSignatures(model_.get(), &nsig), "count signatures", st)) {
            return st;
        }
        if (nsig == 0) {
            return Status::err(error(ErrorCode::InvalidGraph, "litert: model has no signatures"));
        }

        LiteRtSignature sig = nullptr;
        if (!litert_ok(LiteRtGetModelSignature(model_.get(), 0, &sig), "get signature", st)) {
            return st;
        }

        LiteRtParamIndex nin = 0;
        LiteRtParamIndex nout = 0;
        if (!litert_ok(LiteRtGetNumSignatureInputs(sig, &nin), "count inputs", st) ||
            !litert_ok(LiteRtGetNumSignatureOutputs(sig, &nout), "count outputs", st)) {
            return st;
        }

        in_names_.resize(nin);
        out_names_.resize(nout);
        in_bufs_.resize(nin);
        out_bufs_.resize(nout);
        in_types_.resize(nin);
        out_types_.resize(nout);

        for (LiteRtParamIndex i = 0; i < nin; ++i) {
            const char* name = "";
            if (!litert_ok(LiteRtGetSignatureInputName(sig, i, &name), "input name", st)) {
                return st;
            }
            in_names_[i] = name ? name : "";
            LiteRtTensor tensor = nullptr;
            if (!litert_ok(LiteRtGetSignatureInputTensorByIndex(sig, i, &tensor), "input tensor",
                           st)) {
                return st;
            }
            if (!litert_ok(LiteRtGetRankedTensorType(tensor, &in_types_[i]), "input type", st)) {
                return st;
            }
            overlay_ir_shape(in_types_[i], model_ir_, in_names_[i], i, true);
            std::vector<int> resize_dims(in_types_[i].layout.rank);
            bool can_resize = in_types_[i].layout.rank > 0;
            for (unsigned d = 0; d < in_types_[i].layout.rank; ++d) {
                if (in_types_[i].layout.dimensions[d] <= 0) {
                    can_resize = false;
                    break;
                }
                resize_dims[d] = in_types_[i].layout.dimensions[d];
            }
            if (can_resize) {
                LiteRtCompiledModelResizeInputTensor(compiled_.get(), 0, i, resize_dims.data(),
                                                     resize_dims.size());
            }
            LiteRtTensorBufferRequirements reqs = nullptr;
            LiteRtGetCompiledModelInputBufferRequirements(compiled_.get(), 0, i, &reqs);
            if (auto bst = make_cpu_buffer(env_.get(), in_types_[i], reqs, in_bufs_[i],
                                           "input buffer");
                !bst) {
                return bst;
            }
        }
        for (LiteRtParamIndex i = 0; i < nout; ++i) {
            const char* name = "";
            if (!litert_ok(LiteRtGetSignatureOutputName(sig, i, &name), "output name", st)) {
                return st;
            }
            out_names_[i] = name ? name : "";
            LiteRtTensor tensor = nullptr;
            if (!litert_ok(LiteRtGetSignatureOutputTensorByIndex(sig, i, &tensor), "output tensor",
                           st)) {
                return st;
            }
            if (!litert_ok(LiteRtGetRankedTensorType(tensor, &out_types_[i]), "output type", st)) {
                return st;
            }
            overlay_ir_shape(out_types_[i], model_ir_, out_names_[i], i, false);
            LiteRtTensorBufferRequirements reqs = nullptr;
            LiteRtGetCompiledModelOutputBufferRequirements(compiled_.get(), 0, i, &reqs);
            if (auto bst = make_cpu_buffer(env_.get(), out_types_[i], reqs, out_bufs_[i],
                                           "output buffer");
                !bst) {
                return bst;
            }
        }

        if (options_.profile) {
            LiteRtProfiler prof = nullptr;
            if (LiteRtCompiledModelGetProfiler(compiled_.get(), &prof) == kLiteRtStatusOk && prof) {
                profiler_ = prof;
            }
        }
        return Status::ok();
    }

    std::vector<std::string> input_names() const override { return in_names_; }
    std::vector<std::string> output_names() const override { return out_names_; }

    Result<RunResult> run(const std::map<std::string, RuntimeTensor>& inputs) override {
        std::vector<LiteRtTensorBuffer> in_raw(in_bufs_.size());
        std::vector<LiteRtTensorBuffer> out_raw(out_bufs_.size());
        for (std::size_t i = 0; i < in_bufs_.size(); ++i) {
            const RuntimeTensor* t = find_input(inputs, in_names_[i], i, in_names_);
            if (!t) {
                return error(ErrorCode::MissingArgument, "missing input tensor " + in_names_[i]);
            }
            if (t->bytes.empty()) {
                return error(ErrorCode::ExecutionFailure, "empty input tensor " + in_names_[i]);
            }
            void* host = nullptr;
            LiteRtStatus lst =
                LiteRtLockTensorBuffer(in_bufs_[i].get(), &host, kLiteRtTensorBufferLockModeWrite);
            if (lst != kLiteRtStatusOk) {
                return litert_error(lst, "lock input " + in_names_[i]);
            }
            size_t packed = 0;
            lst = LiteRtGetTensorBufferPackedSize(in_bufs_[i].get(), &packed);
            if (lst != kLiteRtStatusOk) {
                LiteRtUnlockTensorBuffer(in_bufs_[i].get());
                return litert_error(lst, "input packed size");
            }
            const std::size_t n = std::min(packed, t->bytes.size());
            if (host && n) {
                std::memcpy(host, t->bytes.data(), n);
            }
            lst = LiteRtUnlockTensorBuffer(in_bufs_[i].get());
            if (lst != kLiteRtStatusOk) {
                return litert_error(lst, "unlock input " + in_names_[i]);
            }
            in_raw[i] = in_bufs_[i].get();
        }
        for (std::size_t i = 0; i < out_bufs_.size(); ++i) {
            out_raw[i] = out_bufs_[i].get();
        }

        if (profiler_) {
            LiteRtResetProfiler(profiler_);
            LiteRtStartProfiler(profiler_);
        }
        const auto t0 = std::chrono::steady_clock::now();
        LiteRtStatus rst = LiteRtRunCompiledModel(compiled_.get(), 0, in_raw.size(), in_raw.data(),
                                                   out_raw.size(), out_raw.data());
        const auto t1 = std::chrono::steady_clock::now();
        if (profiler_) {
            LiteRtStopProfiler(profiler_);
        }
        if (rst != kLiteRtStatusOk) {
            return litert_error(rst, "run");
        }

        RunResult rr;
        rr.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        for (std::size_t i = 0; i < out_bufs_.size(); ++i) {
            RuntimeTensor t;
            t.name = i < out_names_.size() ? out_names_[i] : "";
            t.dtype = from_litert_dtype(out_types_[i].element_type);
            t.shape = shape_from_layout(out_types_[i].layout);
            void* host = nullptr;
            LiteRtStatus lst =
                LiteRtLockTensorBuffer(out_bufs_[i].get(), &host, kLiteRtTensorBufferLockModeRead);
            if (lst != kLiteRtStatusOk) {
                return litert_error(lst, "lock output " + t.name);
            }
            size_t packed = 0;
            lst = LiteRtGetTensorBufferPackedSize(out_bufs_[i].get(), &packed);
            if (lst != kLiteRtStatusOk) {
                LiteRtUnlockTensorBuffer(out_bufs_[i].get());
                return litert_error(lst, "output packed size");
            }
            t.bytes.resize(packed);
            if (host && packed) {
                std::memcpy(t.bytes.data(), host, packed);
            }
            lst = LiteRtUnlockTensorBuffer(out_bufs_[i].get());
            if (lst != kLiteRtStatusOk) {
                return litert_error(lst, "unlock output " + t.name);
            }
            rr.outputs.push_back(std::move(t));
        }
        if (profiler_) {
            collect_profile(rr);
        }
        return rr;
    }

private:
    void collect_profile(RunResult& rr) {
        int n = 0;
        if (LiteRtGetNumProfilerEvents(profiler_, &n) != kLiteRtStatusOk || n <= 0) {
            return;
        }
        std::vector<ProfiledEventData> events(static_cast<std::size_t>(n));
        if (LiteRtGetProfilerEvents(profiler_, n, events.data()) != kLiteRtStatusOk) {
            return;
        }
        for (const auto& ev : events) {
            if (ev.event_type != OPERATOR_INVOKE_EVENT &&
                ev.event_type != DELEGATE_OPERATOR_INVOKE_EVENT &&
                ev.event_type != DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT) {
                continue;
            }
            ProfileEvent pe;
            pe.node = ev.tag ? ev.tag : "";
            pe.op_type = pe.node;
            pe.time_ms = static_cast<double>(ev.elapsed_time_us) / 1000.0;
            rr.profile.push_back(std::move(pe));
        }
        rr.profiled = !rr.profile.empty();
    }

    ModelIR model_ir_;
    RuntimeOptions options_;
    EnvPtr env_;
    ModelPtr model_;
    OptionsPtr options_handle_;
    CompiledPtr compiled_;
    std::vector<BufferPtr> in_bufs_;
    std::vector<BufferPtr> out_bufs_;
    std::vector<std::string> in_names_;
    std::vector<std::string> out_names_;
    std::vector<LiteRtRankedTensorType> in_types_;
    std::vector<LiteRtRankedTensorType> out_types_;
    LiteRtProfiler profiler_ = nullptr;
};

class LitertBackend final : public RuntimeBackend {
public:
    std::string name() const override { return "litert"; }
    std::string version() const override {
#ifdef NN_LITERT_VERSION
        return NN_LITERT_VERSION;
#else
        return "unknown";
#endif
    }
    bool compiled() const override { return true; }
    bool available() const override { return true; }

    bool supports(const ModelIR& model) const override {
        if (model.source_format == "tflite") {
            return true;
        }
        std::error_code ec;
        const auto ext = model.source_path.extension().string();
        return !model.source_path.empty() && (ext == ".tflite" || ext == ".lite") &&
               std::filesystem::exists(model.source_path, ec);
    }

    Result<std::unique_ptr<Session>> create_session(const ModelIR& model,
                                                    const RuntimeOptions& options) override {
        std::error_code ec;
        if (model.source_path.empty() || !std::filesystem::exists(model.source_path, ec)) {
            return error(ErrorCode::FileNotFound, "litert: model file is missing");
        }
        auto sess = std::make_unique<LitertSession>(model, options);
        auto st = sess->init();
        if (!st) {
            return st.error();
        }
        return std::unique_ptr<Session>(std::move(sess));
    }
};

}  // namespace

std::unique_ptr<RuntimeBackend> make_litert_backend() {
    return std::make_unique<LitertBackend>();
}

}  // namespace nn
