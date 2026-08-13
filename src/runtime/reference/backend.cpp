#include "nn/runtime.h"

#include "runtime/reference/ops.h"

#include <chrono>

namespace nn {
namespace {

class ReferenceSession final : public Session {
public:
    ReferenceSession(ModelIR model, RuntimeOptions options)
        : model_(std::move(model)), options_(std::move(options)) {}

    std::vector<std::string> input_names() const override {
        std::vector<std::string> n;
        if (const Graph* g = primary_graph(model_)) {
            for (TensorId id : g->inputs) {
                if (const Tensor* t = g->find_tensor(id)) {
                    n.push_back(t->name);
                }
            }
        }
        return n;
    }
    std::vector<std::string> output_names() const override {
        std::vector<std::string> n;
        if (const Graph* g = primary_graph(model_)) {
            for (TensorId id : g->outputs) {
                if (const Tensor* t = g->find_tensor(id)) {
                    n.push_back(t->name);
                }
            }
        }
        return n;
    }

    Result<RunResult> run(const std::map<std::string, RuntimeTensor>& inputs) override;

private:
    ModelIR model_;
    RuntimeOptions options_;
};

class ReferenceBackend final : public RuntimeBackend {
public:
    std::string name() const override { return "reference"; }
    std::string version() const override { return "0.1.0"; }
    bool supports(const ModelIR& model) const override {
        const Graph* g = primary_graph(model);
        if (!g || g->nodes.empty()) {
            return false;
        }
        for (const auto& n : g->nodes) {
            if (!reference_op_supported(n)) {
                return false;
            }
        }
        return true;
    }
    Result<std::unique_ptr<Session>> create_session(const ModelIR& model,
                                                    const RuntimeOptions& options) override {
        if (!supports(model)) {
            return error(ErrorCode::UnsupportedOperator,
                         "reference backend does not support one or more operators");
        }
        return std::unique_ptr<Session>(std::make_unique<ReferenceSession>(model, options));
    }
};

Result<RuntimeTensor> load_constant(const Tensor& t) {
    RuntimeTensor rt;
    rt.name = t.name;
    rt.dtype = t.dtype;
    rt.shape = t.shape;
    auto bytes = tensor_payload_bytes(t);
    if (!bytes) {
        return bytes.error();
    }
    rt.bytes = std::move(bytes.value());
    return rt;
}

Result<RunResult> ReferenceSession::run(const std::map<std::string, RuntimeTensor>& inputs) {
    const Graph* g = primary_graph(model_);
    if (!g) {
        return error(ErrorCode::InvalidGraph, "no graph");
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::map<TensorId, RuntimeTensor> vals;
    for (const auto& t : g->tensors) {
        if (t.constant && t.data) {
            auto rt = load_constant(t);
            if (!rt) {
                return rt.error();
            }
            vals[t.id] = std::move(rt.value());
        }
    }
    for (TensorId id : g->inputs) {
        const Tensor* t = g->find_tensor(id);
        if (!t) {
            continue;
        }
        auto it = inputs.find(t->name);
        if (it == inputs.end()) {
            return error(ErrorCode::MissingArgument, "missing input tensor " + t->name);
        }
        vals[id] = it->second;
        vals[id].name = t->name;
    }
    RunResult rr;
    rr.profiled = true;
    for (const auto& n : g->nodes) {
        const auto n0 = std::chrono::steady_clock::now();
        auto out = reference_exec_node(*g, n, vals);
        if (!out) {
            return out.error();
        }
        const auto n1 = std::chrono::steady_clock::now();
        ProfileEvent ev;
        ev.node = n.name.empty() ? n.op_type : n.name;
        ev.op_type = n.op_type;
        ev.time_ms = std::chrono::duration<double, std::milli>(n1 - n0).count();
        rr.profile.push_back(ev);
        if (!n.outputs.empty()) {
            vals[n.outputs[0]] = std::move(out.value());
            if (const Tensor* t = g->find_tensor(n.outputs[0])) {
                vals[n.outputs[0]].name = t->name;
            }
            for (std::size_t i = 1; i < n.outputs.size(); ++i) {
                return error(ErrorCode::UnsupportedOperator,
                             "reference backend supports a single output per node");
            }
        }
    }
    for (TensorId id : g->outputs) {
        auto it = vals.find(id);
        if (it == vals.end()) {
            return error(ErrorCode::ExecutionFailure, "output tensor was not produced");
        }
        rr.outputs.push_back(it->second);
        if (const Tensor* t = g->find_tensor(id)) {
            rr.outputs.back().name = t->name;
        }
    }
    auto maybe_dump = [&](const Tensor& t, const RuntimeTensor& rt) {
        if (options_.dump_all) {
            rr.dumps[t.name] = rt;
            return;
        }
        for (const auto& name : options_.dump_names) {
            if (name == t.name) {
                rr.dumps[t.name] = rt;
            }
        }
    };
    if (options_.dump_all || !options_.dump_names.empty()) {
        for (const auto& t : g->tensors) {
            auto it = vals.find(t.id);
            if (it != vals.end()) {
                maybe_dump(t, it->second);
            }
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    rr.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return rr;
}

}  // namespace

std::unique_ptr<RuntimeBackend> make_reference_backend() {
    return std::make_unique<ReferenceBackend>();
}

}  // namespace nn
