#include "cli/commands.h"
#include "commands/common.h"
#include "runtime/tensor_io.h"
#include "util/npy.h"

#include <cstdlib>
#include <filesystem>

namespace nn {
namespace {

int parse_int_flag(const ParsedArgs& a, std::string_view name, int def) {
    if (auto v = flag_value(a, name)) {
        return std::atoi(v->c_str());
    }
    return def;
}

}  // namespace

int cmd_run(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "run");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    const std::string backend_name = flag_value(args.value(), "backend").value_or("");
    RuntimeOptions opt;
    opt.threads = g.threads;
    opt.dump_all = flag_set(args.value(), "dump-all");
    opt.dump_names = flag_values(args.value(), "dump");
    opt.iterations = parse_int_flag(args.value(), "iterations", 1);
    if (opt.iterations < 1) {
        opt.iterations = 1;
    }
    if (auto s = flag_value(args.value(), "seed")) {
        opt.seed = static_cast<uint64_t>(std::strtoull(s->c_str(), nullptr, 10));
    }
    RuntimeBackend* be = select_backend(model.value(), backend_name, opt);
    if (!backend_name.empty() && !be) {
        p.errln("nn: unknown backend '" + backend_name + "'");
        return kExitBackendUnavailable;
    }
    if (!be || !be->available()) {
        p.errln("nn: backend unavailable");
        return kExitBackendUnavailable;
    }
    if (!be->supports(model.value())) {
        p.errln("nn: backend '" + be->name() + "' does not support this model");
        return kExitUnsupportedOperator;
    }
    const bool random_fill = flag_value(args.value(), "seed").has_value();
    auto inputs = bind_model_inputs(model.value(), flag_values(args.value(), "input"), opt.seed,
                                    random_fill);
    if (!inputs) {
        return cmd_fail(p, inputs.error());
    }
    auto sess = be->create_session(model.value(), opt);
    if (!sess) {
        return cmd_fail(p, sess.error());
    }
    Result<RunResult> result = error(ErrorCode::ExecutionFailure, "no iterations");
    for (int i = 0; i < opt.iterations; ++i) {
        result = sess.value()->run(inputs.value());
        if (!result) {
            return cmd_fail(p, result.error());
        }
    }
    if (g.output_format == OutputFormat::Json) {
        Json root = Json::object();
        root["schema_version"] = 1;
        root["backend"] = be->name();
        root["latency_ms"] = result.value().latency_ms;
        Json outs = Json::array();
        for (const auto& t : result.value().outputs) {
            Json o = Json::object();
            o["name"] = t.name;
            o["dtype"] = datatype_name(t.dtype);
            o["shape"] = json_shape(t.shape);
            o["bytes"] = static_cast<int64_t>(t.bytes.size());
            outs.push(std::move(o));
        }
        root["outputs"] = std::move(outs);
        p.json(root);
    } else {
        p.kv("backend:", be->name());
        p.kv("latency_ms:", std::to_string(result.value().latency_ms));
        p.kv("outputs:", std::to_string(result.value().outputs.size()));
        for (const auto& t : result.value().outputs) {
            p.println("  " + t.name + "  " + datatype_name(t.dtype) + "  " + t.shape.to_string());
        }
    }
    if (auto out = flag_value(args.value(), "output")) {
        if (!result.value().outputs.empty()) {
            auto st = save_tensor_file(*out, result.value().outputs.front());
            if (!st) {
                return cmd_fail(p, st.error());
            }
        }
    } else if (g.output_file && !result.value().outputs.empty()) {
        auto st = save_tensor_file(*g.output_file, result.value().outputs.front());
        if (!st) {
            return cmd_fail(p, st.error());
        }
    }
    const auto dump_dir = [&]() {
        if (auto o = flag_value(args.value(), "output")) {
            return std::filesystem::path(*o).parent_path();
        }
        if (g.output_file) {
            return g.output_file->parent_path();
        }
        return std::filesystem::current_path();
    }();
    for (const auto& [name, tensor] : result.value().dumps) {
        auto path = dump_dir / (name + ".npy");
        auto st = save_npy(path, tensor);
        if (!st) {
            return cmd_fail(p, st.error());
        }
        if (g.output_format != OutputFormat::Json) {
            p.println("dump " + path.string());
        }
    }
    return kExitOk;
}
}  // namespace nn
