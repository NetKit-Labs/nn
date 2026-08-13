#ifndef NN_RUNTIME_H
#define NN_RUNTIME_H

#include "nn/model.h"
#include "nn/result.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nn {

struct RuntimeTensor {
    std::string name;
    DataType dtype = DataType::Unknown;
    Shape shape;
    std::vector<uint8_t> bytes;
};

struct RuntimeOptions {
    int threads = 0;
    uint64_t seed = 0;
    int iterations = 1;
    bool dump_all = false;
    std::vector<std::string> dump_names;
    bool profile = false;
};

struct ProfileEvent {
    std::string node;
    std::string op_type;
    double time_ms = 0.0;
};

struct RunResult {
    std::vector<RuntimeTensor> outputs;
    std::map<std::string, RuntimeTensor> dumps;
    double latency_ms = 0.0;
    bool profiled = false;
    std::vector<ProfileEvent> profile;
};

class Session {
public:
    virtual ~Session() = default;
    virtual Result<RunResult> run(const std::map<std::string, RuntimeTensor>& inputs) = 0;
    virtual std::vector<std::string> input_names() const = 0;
    virtual std::vector<std::string> output_names() const = 0;
};

class RuntimeBackend {
public:
    virtual ~RuntimeBackend() = default;
    virtual std::string name() const = 0;
    virtual std::string version() const { return ""; }
    virtual bool compiled() const { return true; }
    virtual bool available() const { return compiled(); }
    virtual bool supports(const ModelIR& model) const = 0;
    virtual Result<std::unique_ptr<Session>> create_session(
        const ModelIR& model, const RuntimeOptions& options) = 0;
};

class RuntimeRegistry {
public:
    void register_backend(std::unique_ptr<RuntimeBackend> backend);
    RuntimeBackend* find(std::string_view name) const;
    std::vector<RuntimeBackend*> list() const;
    RuntimeBackend* default_backend(const ModelIR& model) const;

private:
    std::vector<std::unique_ptr<RuntimeBackend>> backends_;
};

RuntimeRegistry& default_runtime_registry();
void register_builtin_backends(RuntimeRegistry& registry);

}  // namespace nn

#endif
