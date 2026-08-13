#include "cli/commands.h"

#include "commands/common.h"
#include "nn/analysis.h"
#include "util/format_text.h"

#include <algorithm>
#include <cstdio>
#include <map>

namespace nn {

int cmd_ops(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "ops");
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
    auto compute = analyze_compute(model.value());
    const bool canonical = flag_set(args.value(), "canonical") || !flag_set(args.value(), "native");
    auto filter = flag_value(args.value(), "op");

    if (flag_set(args.value(), "details") || filter) {
        for (const auto& n : graph->nodes) {
            const std::string shown = canonical ? canonical_op_name(n.canonical) : n.op_type;
            if (filter && n.op_type != *filter && shown != *filter && n.name != *filter) {
                continue;
            }
            p.println(n.name.empty() ? shown : n.name);
            p.kv("  op:", n.op_type);
            p.kv("  canonical:", canonical_op_name(n.canonical));
        }
        return kExitOk;
    }

    struct Row {
        std::string name;
        uint64_t count = 0;
        OptionalCount macs;
    };
    std::map<std::string, Row> rows;
    for (std::size_t i = 0; i < graph->nodes.size(); ++i) {
        const auto& n = graph->nodes[i];
        std::string key = canonical ? canonical_op_name(n.canonical) : n.op_type;
        if (n.canonical == CanonicalOp::Unknown && canonical) {
            key = n.op_type;
        }
        auto& row = rows[key];
        row.name = key;
        ++row.count;
        if (i < compute.nodes.size() && compute.nodes[i].cost.macs.known) {
            if (!row.macs.known) {
                row.macs.known = true;
            }
            row.macs.value += compute.nodes[i].cost.macs.value;
        }
    }
    std::vector<Row> list;
    for (auto& [_, r] : rows) {
        list.push_back(r);
    }
    if (flag_set(args.value(), "by-cost")) {
        std::sort(list.begin(), list.end(), [](const Row& a, const Row& b) {
            return a.macs.value > b.macs.value;
        });
    }
    if (g.output_format == OutputFormat::Json) {
        Json arr = Json::array();
        for (const auto& r : list) {
            Json o = Json::object();
            o["op"] = r.name;
            o["count"] = r.count;
            if (r.macs.known) {
                o["macs"] = r.macs.value;
            }
            arr.push(std::move(o));
        }
        Json root = Json::object();
        root["schema_version"] = 1;
        root["operators"] = std::move(arr);
        p.json(root);
        return kExitOk;
    }
    p.println("OPERATOR                 COUNT          MACs");
    p.println("------------------------------------------------");
    for (const auto& r : list) {
        char line[96];
        const std::string macs =
            r.macs.known ? human_si(static_cast<double>(r.macs.value)) : std::string("-");
        std::snprintf(line, sizeof(line), "%-24s %6llu %14s", r.name.c_str(),
                      static_cast<unsigned long long>(r.count), macs.c_str());
        p.println(line);
    }
    return kExitOk;
}

}  // namespace nn
