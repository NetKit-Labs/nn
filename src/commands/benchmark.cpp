#include "cli/commands.h"
#include "commands/common.h"
#include "runtime/tensor_io.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace nn {
int cmd_benchmark(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "benchmark");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    const std::string backend_name = flag_value(args.value(), "backend").value_or("");
    RuntimeBackend* be = select_backend(model.value(), backend_name);
    if (!be || !be->available() || !be->supports(model.value())) {
        p.errln("nn: backend unavailable");
        return kExitBackendUnavailable;
    }
    const int warmup = std::max(0, std::atoi(flag_value(args.value(), "warmup").value_or("5").c_str()));
    const int iters = std::max(1, std::atoi(flag_value(args.value(), "iterations").value_or("20").c_str()));
    RuntimeOptions opt;
    opt.threads = g.threads;
    auto ins = bind_model_inputs(model.value(), flag_values(args.value(), "input"), 0, false);
    if (!ins) {
        return cmd_fail(p, ins.error());
    }
    auto sess = be->create_session(model.value(), opt);
    if (!sess) {
        return cmd_fail(p, sess.error());
    }
    for (int i = 0; i < warmup; ++i) {
        auto r = sess.value()->run(ins.value());
        if (!r) {
            return cmd_fail(p, r.error());
        }
    }
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        auto r = sess.value()->run(ins.value());
        if (!r) {
            return cmd_fail(p, r.error());
        }
        samples.push_back(r.value().latency_ms);
    }
    std::sort(samples.begin(), samples.end());
    auto pct = [&](double q) {
        if (samples.empty()) {
            return 0.0;
        }
        const double idx = q * static_cast<double>(samples.size() - 1);
        const std::size_t i = static_cast<std::size_t>(idx);
        return samples[i];
    };
    double sum = 0;
    for (double s : samples) {
        sum += s;
    }
    p.kv("backend:", be->name());
    p.kv("warmup:", std::to_string(warmup));
    p.kv("iterations:", std::to_string(iters));
    p.kv("mean_ms:", std::to_string(sum / static_cast<double>(samples.size())));
    p.kv("min_ms:", std::to_string(samples.front()));
    p.kv("p50_ms:", std::to_string(pct(0.50)));
    p.kv("p90_ms:", std::to_string(pct(0.90)));
    p.kv("p99_ms:", std::to_string(pct(0.99)));
    p.kv("max_ms:", std::to_string(samples.back()));
    return kExitOk;
}
}  // namespace nn
