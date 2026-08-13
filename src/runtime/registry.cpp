#include "nn/runtime.h"

namespace nn {

void RuntimeRegistry::register_backend(std::unique_ptr<RuntimeBackend> backend) {
    backends_.push_back(std::move(backend));
}

RuntimeBackend* RuntimeRegistry::find(std::string_view name) const {
    for (const auto& b : backends_) {
        if (b->name() == name) {
            return b.get();
        }
    }
    return nullptr;
}

std::vector<RuntimeBackend*> RuntimeRegistry::list() const {
    std::vector<RuntimeBackend*> out;
    for (const auto& b : backends_) {
        out.push_back(b.get());
    }
    return out;
}

RuntimeBackend* RuntimeRegistry::default_backend(const ModelIR& model) const {
    for (const auto& b : backends_) {
        if (b->available() && b->supports(model)) {
            return b.get();
        }
    }
    return nullptr;
}

std::unique_ptr<RuntimeBackend> make_reference_backend();
#if defined(NN_HAS_ONNXRUNTIME)
std::unique_ptr<RuntimeBackend> make_onnxruntime_backend();
#endif
#if defined(NN_HAS_LITERT)
std::unique_ptr<RuntimeBackend> make_litert_backend();
#endif

void register_builtin_backends(RuntimeRegistry& registry) {
#if defined(NN_HAS_ONNXRUNTIME)
    registry.register_backend(make_onnxruntime_backend());
#endif
#if defined(NN_HAS_LITERT)
    registry.register_backend(make_litert_backend());
#endif
    registry.register_backend(make_reference_backend());
}

RuntimeRegistry& default_runtime_registry() {
    static RuntimeRegistry r = [] {
        RuntimeRegistry x;
        register_builtin_backends(x);
        return x;
    }();
    return r;
}

}  // namespace nn
