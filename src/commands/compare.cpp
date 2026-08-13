#include "cli/commands.h"
#include "commands/common.h"
#include "nn/diff.h"
#include "runtime/tensor_io.h"

#include <algorithm>
#include <cstdlib>

namespace nn {
int cmd_compare(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "compare");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    if (args.value().positionals.size() < 2) {
        return p.usage_error("nn: compare requires two models", "nn compare <a> <b>");
    }
    auto a = load_model(args.value().positionals[0]);
    auto b = load_model(args.value().positionals[1]);
    if (!a) {
        return cmd_fail(p, a.error());
    }
    if (!b) {
        return cmd_fail(p, b.error());
    }
    const double atol = std::strtod(flag_value(args.value(), "atol").value_or("1e-5").c_str(), nullptr);
    const double rtol = std::strtod(flag_value(args.value(), "rtol").value_or("1e-5").c_str(), nullptr);
    const double threshold =
        std::strtod(flag_value(args.value(), "threshold").value_or("1e-5").c_str(), nullptr);
    const auto input_specs = flag_values(args.value(), "input");
    if (input_specs.empty() && !flag_set(args.value(), "activations")) {
        p.println("Structural comparison (no inputs supplied)");
        auto d = diff_models(a.value(), b.value());
        if (!d) {
            return cmd_fail(p, d.error());
        }
        p.kv("identical:", d.value().identical ? "yes" : "no");
        return d.value().identical ? kExitOk : kExitDifference;
    }
    RuntimeOptions opt;
    opt.threads = g.threads;
    opt.dump_all = flag_set(args.value(), "activations");
    auto ins_a = bind_model_inputs(a.value(), input_specs, 0, false);
    if (!ins_a) {
        return cmd_fail(p, ins_a.error());
    }
    auto ins_b = bind_model_inputs(b.value(), input_specs, 0, false);
    if (!ins_b) {
        return cmd_fail(p, ins_b.error());
    }
    const std::string be1 = flag_value(args.value(), "backend").value_or("");
    const std::string be2 = flag_value(args.value(), "backend2").value_or(be1);
    auto ra = eval_model(a.value(), ins_a.value(), opt, be1);
    auto rb = eval_model(b.value(), ins_b.value(), opt, be2);
    if (!ra) {
        return cmd_fail(p, ra.error());
    }
    if (!rb) {
        return cmd_fail(p, rb.error());
    }
    bool differ = false;
    auto compare_pair = [&](const std::string& name, const RuntimeTensor* x, const RuntimeTensor* y) {
        if (!x || !y) {
            p.println(name);
            p.kv("  comparable:", "no");
            p.kv("  note:", !x && !y ? "missing on both sides" : (!x ? "missing on first model" : "missing on second model"));
            differ = true;
            return;
        }
        auto c = compare_numeric_aligned(*x, *y, atol, rtol);
        p.println(name);
        if (!c.comparable) {
            p.kv("  comparable:", "no");
            p.kv("  note:", c.note);
            differ = true;
            return;
        }
        p.kv("  max_abs:", std::to_string(c.max_abs));
        p.kv("  mean_abs:", std::to_string(c.mean_abs));
        p.kv("  rmse:", std::to_string(c.rmse));
        p.kv("  cosine:", std::to_string(c.cosine));
        p.kv("  changed:", std::to_string(c.changed) + " / " + std::to_string(c.total));
        if (!c.note.empty()) {
            p.kv("  note:", c.note);
        }
        if (c.max_abs > threshold) {
            differ = true;
        }
    };
    const auto pairs = align_runtime_tensors(ra.value().outputs, rb.value().outputs);
    if (pairs.empty()) {
        p.println("no outputs to compare");
        differ = true;
    }
    for (const auto& pair : pairs) {
        compare_pair(pair.label, pair.a, pair.b);
    }
    if (flag_set(args.value(), "activations")) {
        p.heading("Activations");
        if (ra.value().dumps.empty() && rb.value().dumps.empty()) {
            p.println("neither backend dumped intermediates (only the reference backend supports --activations)");
        } else if (ra.value().dumps.empty() || rb.value().dumps.empty()) {
            p.println("one backend did not dump intermediates; comparing available activations only");
        }
        const auto acts = align_activation_dumps(a.value(), ra.value(), b.value(), rb.value());
        if (acts.empty() && (!ra.value().dumps.empty() || !rb.value().dumps.empty())) {
            p.println("no matching activations by name or graph correspondence");
        }
        for (const auto& pair : acts) {
            compare_pair(pair.label, pair.a, pair.b);
        }
    }
    p.println(differ ? "RESULT: different" : "RESULT: similar");
    return differ ? kExitDifference : kExitOk;
}
}  // namespace nn
