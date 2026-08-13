#include "cli/commands.h"
#include "commands/common.h"
#include "runtime/tensor_io.h"

#include <cmath>

namespace nn {
namespace {

double argmax_agree(const RuntimeTensor& got, const RuntimeTensor& exp) {
    auto a = as_f64(got);
    auto b = as_f64(exp);
    if (a.empty() || a.size() != b.size()) {
        return 0;
    }
    auto imax = [](const std::vector<double>& v) {
        std::size_t i = 0;
        for (std::size_t n = 1; n < v.size(); ++n) {
            if (v[n] > v[i]) {
                i = n;
            }
        }
        return i;
    };
    return imax(a) == imax(b) ? 1.0 : 0.0;
}

}  // namespace

int cmd_validate(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "validate");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    if (args.value().positionals.size() < 2) {
        return p.usage_error("nn: validate requires model and dataset", "nn validate <model> <dataset>");
    }
    auto model = load_model(args.value().positionals[0]);
    if (!model) {
        return cmd_fail(p, model.error());
    }
    auto cases = load_test_manifest(args.value().positionals[1]);
    if (!cases) {
        return cmd_fail(p, cases.error());
    }
    const std::string metric = flag_value(args.value(), "metric").value_or("mse");
    RuntimeOptions opt;
    opt.threads = g.threads;
    double sum_mse = 0;
    double sum_mae = 0;
    double sum_rmse = 0;
    double sum_acc = 0;
    int n = 0;
    int failed = 0;
    for (const auto& tc : cases.value()) {
        std::vector<std::string> specs;
        for (const auto& [name, path] : tc.inputs) {
            specs.push_back(name + "=" + path);
        }
        auto ins = bind_model_inputs(model.value(), specs, 0, false);
        if (!ins) {
            p.errln(tc.name + ": " + ins.error().format());
            ++failed;
            continue;
        }
        auto rr = eval_model(model.value(), ins.value(), opt, {});
        if (!rr) {
            p.errln(tc.name + ": " + rr.error().format());
            ++failed;
            continue;
        }
        if (tc.expected.empty() || rr.value().outputs.empty()) {
            continue;
        }
        auto exp = load_tensor_file(tc.expected.begin()->second);
        if (!exp) {
            p.errln(tc.name + ": " + exp.error().format());
            ++failed;
            continue;
        }
        const auto& got = rr.value().outputs.front();
        auto c = compare_numeric(got, exp.value(), 0, 0);
        if (!c.comparable) {
            ++failed;
            continue;
        }
        sum_mse += c.rmse * c.rmse;
        sum_mae += c.mean_abs;
        sum_rmse += c.rmse;
        sum_acc += argmax_agree(got, exp.value());
        ++n;
    }
    if (n == 0 && failed) {
        p.errln("nn: validation did not produce any comparable cases");
        return kExitValidation;
    }
    const double inv = n ? 1.0 / static_cast<double>(n) : 0;
    p.kv("cases:", std::to_string(n));
    p.kv("metric:", metric);
    p.kv("mse:", std::to_string(sum_mse * inv));
    p.kv("mae:", std::to_string(sum_mae * inv));
    p.kv("rmse:", std::to_string(sum_rmse * inv));
    p.kv("accuracy:", std::to_string(sum_acc * inv) + " (argmax agreement)");
    if (failed) {
        p.println("RESULT: FAIL");
        return kExitValidation;
    }
    p.println("RESULT: PASS");
    return kExitOk;
}
}  // namespace nn
