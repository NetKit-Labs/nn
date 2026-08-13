#include "cli/commands.h"
#include "commands/common.h"
#include "nn/analysis.h"

namespace nn {
int cmd_hash(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "hash");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    HashOptions opt;
    opt.graph = flag_set(args.value(), "graph");
    opt.weights = flag_set(args.value(), "weights");
    opt.canonical = flag_set(args.value(), "canonical");
    if (auto t = flag_value(args.value(), "tensor")) {
        opt.tensor_name = *t;
    }
    auto h = hash_model(model.value(), opt);
    if (!h) {
        return cmd_fail(p, h.error());
    }
    if (g.output_format == OutputFormat::Json) {
        Json o = Json::object();
        o["schema_version"] = 1;
        o["artifact"] = h.value().artifact_sha256;
        o["graph"] = h.value().graph_sha256;
        o["weights"] = h.value().weights_sha256;
        if (!h.value().tensor_sha256.empty()) {
            o["tensor"] = h.value().tensor_sha256;
        }
        p.json(o);
        return kExitOk;
    }
    if (opt.graph || opt.canonical) {
        p.kv("graph:", h.value().graph_sha256);
    } else if (opt.weights) {
        p.kv("weights:", h.value().weights_sha256);
    } else if (!opt.tensor_name.empty()) {
        p.kv(opt.tensor_name + ":", h.value().tensor_sha256);
    } else {
        p.kv("artifact:", h.value().artifact_sha256);
        p.kv("graph:", h.value().graph_sha256);
        p.kv("weights:", h.value().weights_sha256);
    }
    return kExitOk;
}
}  // namespace nn
