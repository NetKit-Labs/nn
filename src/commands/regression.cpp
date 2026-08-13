#include "cli/commands.h"
#include "commands/common.h"
#include "nn/analysis.h"
#include "runtime/tensor_io.h"

#include <cstdlib>

namespace nn {
int cmd_regression(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "regression");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    if (args.value().positionals.size() < 3) {
        return p.usage_error("nn: regression requires old, new, and tests",
                             "nn regression <old> <new> <tests>");
    }
    auto a = load_model(args.value().positionals[0]);
    auto b = load_model(args.value().positionals[1]);
    if (!a) {
        return cmd_fail(p, a.error());
    }
    if (!b) {
        return cmd_fail(p, b.error());
    }
    const auto ma = analyze_memory(a.value());
    const auto mb = analyze_memory(b.value());
    p.println("Regression Analysis");
    p.kv("Old RAM (est):", std::to_string(ma.estimated_ram_bytes));
    p.kv("New RAM (est):", std::to_string(mb.estimated_ram_bytes));
    p.kv("Old size:", std::to_string(a.value().file_size));
    p.kv("New size:", std::to_string(b.value().file_size));

    bool fail = false;
    auto frac = [](uint64_t oldv, uint64_t newv) -> double {
        if (oldv == 0) {
            return newv == 0 ? 0.0 : 1.0;
        }
        return static_cast<double>(newv) / static_cast<double>(oldv) - 1.0;
    };
    if (auto v = flag_value(args.value(), "max-memory-growth")) {
        const double max_mem = std::strtod(v->c_str(), nullptr);
        const double growth = frac(ma.estimated_ram_bytes, mb.estimated_ram_bytes);
        p.kv("Memory growth:", std::to_string(growth));
        if (growth > max_mem) {
            p.println("Reason: memory growth exceeds --max-memory-growth");
            fail = true;
        }
    }
    if (auto v = flag_value(args.value(), "max-model-growth")) {
        const double maxg = std::strtod(v->c_str(), nullptr);
        const double growth = frac(a.value().file_size, b.value().file_size);
        p.kv("Model size growth:", std::to_string(growth));
        if (growth > maxg) {
            p.println("Reason: model size growth exceeds --max-model-growth");
            fail = true;
        }
    }

    auto cases = load_test_manifest(args.value().positionals[2]);
    double old_lat = 0;
    double new_lat = 0;
    double old_mse = 0;
    double new_mse = 0;
    int n = 0;
    if (cases) {
        RuntimeOptions opt;
        opt.threads = g.threads;
        for (const auto& tc : cases.value()) {
            std::vector<std::string> specs;
            for (const auto& [name, path] : tc.inputs) {
                specs.push_back(name + "=" + path);
            }
            auto ins_a = bind_model_inputs(a.value(), specs, 0, false);
            auto ins_b = bind_model_inputs(b.value(), specs, 0, false);
            if (!ins_a || !ins_b) {
                continue;
            }
            auto ra = eval_model(a.value(), ins_a.value(), opt, {});
            auto rb = eval_model(b.value(), ins_b.value(), opt, {});
            if (!ra || !rb) {
                continue;
            }
            old_lat += ra.value().latency_ms;
            new_lat += rb.value().latency_ms;
            if (!tc.expected.empty() && !ra.value().outputs.empty() && !rb.value().outputs.empty()) {
                auto exp = load_tensor_file(tc.expected.begin()->second);
                if (exp) {
                    auto ca = compare_numeric(ra.value().outputs.front(), exp.value(), 0, 0);
                    auto cb = compare_numeric(rb.value().outputs.front(), exp.value(), 0, 0);
                    if (ca.comparable && cb.comparable) {
                        old_mse += ca.rmse * ca.rmse;
                        new_mse += cb.rmse * cb.rmse;
                    }
                }
            }
            ++n;
        }
    } else {
        p.println("Tests: " + cases.error().message() + " (memory/size policy still applied)");
    }
    if (n > 0) {
        old_lat /= static_cast<double>(n);
        new_lat /= static_cast<double>(n);
        old_mse /= static_cast<double>(n);
        new_mse /= static_cast<double>(n);
        p.kv("Old latency_ms:", std::to_string(old_lat));
        p.kv("New latency_ms:", std::to_string(new_lat));
        p.kv("Old mse:", std::to_string(old_mse));
        p.kv("New mse:", std::to_string(new_mse));
        if (auto v = flag_value(args.value(), "max-latency-growth")) {
            const double maxg = std::strtod(v->c_str(), nullptr);
            const double growth = old_lat == 0 ? 0.0 : new_lat / old_lat - 1.0;
            p.kv("Latency growth:", std::to_string(growth));
            if (growth > maxg) {
                p.println("Reason: latency growth exceeds --max-latency-growth");
                fail = true;
            }
        }
        if (auto v = flag_value(args.value(), "max-accuracy-loss")) {
            const double maxl = std::strtod(v->c_str(), nullptr);
            const double loss = new_mse - old_mse;
            p.kv("MSE increase:", std::to_string(loss));
            if (loss > maxl) {
                p.println("Reason: accuracy loss (MSE increase) exceeds --max-accuracy-loss");
                fail = true;
            }
        }
    } else {
        p.println("Accuracy: unavailable (no executable test cases)");
        p.println("Latency: unavailable (not measured)");
    }
    p.println(fail ? "RESULT: FAIL" : "RESULT: PASS");
    return fail ? kExitDifference : kExitOk;
}
}  // namespace nn
