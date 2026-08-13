#include "cli/commands.h"

#include "commands/common.h"
#include "util/format_text.h"

#include <algorithm>
#include <cstdio>

namespace nn {

int cmd_tensors(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "tensors");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    const Graph* graph = primary_graph(model.value());
    if (!graph) {
        p.errln("nn: model has no graph");
        return kExitMalformed;
    }
    std::vector<const Tensor*> ts;
    for (const auto& t : graph->tensors) {
        if (flag_set(args.value(), "weights") && !t.constant) {
            continue;
        }
        if (flag_set(args.value(), "activations") && t.constant) {
            continue;
        }
        if (flag_set(args.value(), "inputs") && !t.model_input) {
            continue;
        }
        if (flag_set(args.value(), "outputs") && !t.model_output) {
            continue;
        }
        if (auto dt = flag_value(args.value(), "dtype")) {
            auto want = datatype_from_name(*dt);
            if (want && t.dtype != *want) {
                continue;
            }
        }
        if (auto pat = flag_value(args.value(), "name")) {
            if (t.name.find(*pat) == std::string::npos) {
                continue;
            }
        }
        ts.push_back(&t);
    }
    if (flag_set(args.value(), "largest")) {
        std::sort(ts.begin(), ts.end(), [](const Tensor* a, const Tensor* b) {
            return a->storage_bytes > b->storage_bytes;
        });
    }
    if (g.output_format == OutputFormat::Json) {
        Json arr = Json::array();
        for (const Tensor* t : ts) {
            Json o = Json::object();
            o["name"] = t->name;
            o["dtype"] = datatype_name(t->dtype);
            o["shape"] = t->shape.to_string();
            o["bytes"] = t->storage_bytes;
            o["constant"] = t->constant;
            arr.push(std::move(o));
        }
        Json root = Json::object();
        root["schema_version"] = 1;
        root["tensors"] = std::move(arr);
        p.json(root);
        return kExitOk;
    }
    p.println("NAME                             SHAPE            DTYPE      BYTES  CONST  QUANT");
    p.println("-------------------------------------------------------------------------------");
    for (const Tensor* t : ts) {
        char line[160];
        std::snprintf(line, sizeof(line), "%-32.32s %-16.16s %-10s %8s %5s  %s", t->name.c_str(),
                      t->shape.to_string().c_str(), datatype_name(t->dtype),
                      human_bytes(t->storage_bytes).c_str(), t->constant ? "yes" : "no",
                      t->quantization.describe().c_str());
        p.println(line);
    }
    return kExitOk;
}

}  // namespace nn
