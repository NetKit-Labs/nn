#include "cli/commands.h"
#include "commands/common.h"
#include "runtime/tensor_io.h"

namespace nn {
int cmd_test(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "test");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    if (args.value().positionals.size() < 2) {
        return p.usage_error("nn: test requires model and tests path", "nn test <model> <tests>");
    }
    auto model = load_model(args.value().positionals[0]);
    if (!model) {
        return cmd_fail(p, model.error());
    }
    auto cases = load_test_manifest(args.value().positionals[1]);
    if (!cases) {
        return cmd_fail(p, cases.error());
    }
    RuntimeOptions opt;
    opt.threads = g.threads;
    int failed = 0;
    int ran = 0;
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
        ++ran;
        bool ok = true;
        if (tc.expected.empty()) {
            p.println(tc.name + ": PASS (executed; no expected tensors)");
            continue;
        }
        for (const auto& [name, path] : tc.expected) {
            auto exp = load_tensor_file(path);
            if (!exp) {
                p.errln(tc.name + ": " + exp.error().format());
                ok = false;
                continue;
            }
            const RuntimeTensor* got = nullptr;
            for (const auto& o : rr.value().outputs) {
                if (o.name == name || name == "output") {
                    got = &o;
                    if (o.name == name) {
                        break;
                    }
                }
            }
            if (!got && !rr.value().outputs.empty()) {
                got = &rr.value().outputs.front();
            }
            if (!got) {
                p.errln(tc.name + ": missing output " + name);
                ok = false;
                continue;
            }
            auto c = compare_numeric(*got, exp.value(), 1e-5, 1e-5);
            if (!c.comparable || c.changed > 0) {
                p.println(tc.name + ": FAIL " + name + " max_abs=" + std::to_string(c.max_abs));
                ok = false;
            }
        }
        if (ok) {
            p.println(tc.name + ": PASS");
        } else {
            ++failed;
        }
    }
    p.kv("ran:", std::to_string(ran));
    p.kv("failed:", std::to_string(failed));
    if (failed) {
        p.println("RESULT: FAIL");
        return kExitValidation;
    }
    p.println("RESULT: PASS");
    return kExitOk;
}
}  // namespace nn
