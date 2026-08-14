#include "cli/commands.h"
#include "commands/common.h"
#include "nn/analysis.h"
#include "util/format_text.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace nn {
namespace {

std::string pct(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f%%", v * 100.0);
    return buf;
}

std::string score_text(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4f", v);
    return buf;
}

Json layer_json(const TensorSparsity& t) {
    Json o = Json::object();
    o["tensor"] = t.name;
    o["layer"] = t.layer;
    o["op"] = t.op_type;
    o["canonical"] = t.canonical;
    o["shape"] = t.shape;
    o["bytes"] = static_cast<int64_t>(t.bytes);
    o["elements"] = static_cast<int64_t>(t.elements);
    o["zeros"] = static_cast<int64_t>(t.zeros);
    o["near_zeros"] = static_cast<int64_t>(t.near_zeros);
    o["zero_fraction"] = t.zero_fraction;
    o["near_zero_fraction"] = t.near_zero_fraction;
    o["computed"] = t.computed;
    o["layout"] = t.layout;
    o["channels"] = static_cast<int64_t>(t.channels);
    o["weak_channels"] = static_cast<int64_t>(t.weak_channels);
    o["weak_channel_frac"] = t.weak_channel_frac;
    o["max_channel_l1"] = t.max_channel_l1;
    o["macs"] = static_cast<int64_t>(t.macs);
    o["macs_known"] = t.macs_known;
    o["mac_share"] = t.mac_share;
    o["score"] = t.score;
    o["estimated_saved_bytes"] = static_cast<int64_t>(t.estimated_saved_bytes);
    o["estimated_saved_macs"] = static_cast<int64_t>(t.estimated_saved_macs);
    o["skip_coupled"] = t.skip_coupled;
    o["depthwise"] = t.depthwise;
    return o;
}

}  // namespace

int cmd_sparsity(const GlobalOptions& g) {
    Printer p(g);
    auto args = cmd_args(g, "sparsity");
    if (!args) {
        return cmd_fail(p, args.error());
    }
    auto model = cmd_load(args.value());
    if (!model) {
        return cmd_fail(p, model.error());
    }
    SparsityOptions opt;
    if (auto t = flag_value(args.value(), "threshold")) {
        opt.threshold = std::strtod(t->c_str(), nullptr);
    }
    if (auto f = flag_value(args.value(), "channel-frac")) {
        opt.channel_l1_frac = std::strtod(f->c_str(), nullptr);
        if (opt.channel_l1_frac < 0) {
            opt.channel_l1_frac = 0;
        }
    }
    auto r = analyze_sparsity(model.value(), opt);

    if (g.output_format == OutputFormat::Json) {
        Json o = Json::object();
        o["schema_version"] = 1;
        o["tensors_considered"] = static_cast<int64_t>(r.tensors_considered);
        o["tensors_computed"] = static_cast<int64_t>(r.tensors_computed);
        o["overall_zero_fraction"] = r.overall_zero_fraction;
        o["overall_near_zero_fraction"] = r.overall_near_zero_fraction;
        o["threshold"] = r.threshold;
        o["channel_l1_frac"] = r.channel_l1_frac;
        o["total_macs"] = static_cast<int64_t>(r.total_macs);
        o["total_macs_known"] = r.total_macs_known;
        o["estimated_saved_bytes"] = static_cast<int64_t>(r.estimated_saved_bytes);
        o["estimated_saved_macs"] = static_cast<int64_t>(r.estimated_saved_macs);
        o["savings_are_upper_bound"] = true;
        Json layers = Json::array();
        for (const auto& t : r.tensors) {
            layers.push(layer_json(t));
        }
        o["layers"] = std::move(layers);
        Json notes = Json::array();
        for (const auto& n : r.notes) {
            notes.push(Json(n));
        }
        o["notes"] = std::move(notes);
        p.json(o);
        return kExitOk;
    }

    p.kv("Tensors considered:", std::to_string(r.tensors_considered));
    p.kv("Tensors computed:", std::to_string(r.tensors_computed));
    p.kv("Zero fraction:", std::to_string(r.overall_zero_fraction));
    p.kv("Near-zero fraction:", std::to_string(r.overall_near_zero_fraction));
    p.kv("Near-zero |w| <=:", std::to_string(r.threshold));
    char frac[32];
    std::snprintf(frac, sizeof(frac), "%.2f%% of max channel L1", r.channel_l1_frac * 100.0);
    p.kv("Weak-channel cut:", frac);
    p.kv("Total MACs:", r.total_macs_known ? human_si(static_cast<double>(r.total_macs)) : "unknown");
    p.kv("Est. saved bytes:", human_bytes(r.estimated_saved_bytes) + " (upper bound)");
    p.kv("Est. saved MACs:",
         r.total_macs_known ? human_si(static_cast<double>(r.estimated_saved_macs)) + " (upper bound)"
                            : std::string("unknown"));
    p.println();
    p.println("LAYER                OP         SHAPE            ZEROS       NEAR        WEAK     "
              "MAC%    SCORE");
    p.println("--------------------------------------------------------------------------------"
              "---------------");
    for (const auto& t : r.tensors) {
        const std::string layer = t.layer.empty() ? (t.name.empty() ? "-" : t.name) : t.layer;
        const std::string op = t.op_type.empty() ? "-" : t.op_type;
        const std::string zeros =
            t.computed ? (std::to_string(t.zeros) + "/" + std::to_string(t.elements)) : "-";
        const std::string near =
            t.computed ? (std::to_string(t.near_zeros) + "/" + std::to_string(t.elements)) : "-";
        const std::string weak = t.channels
                                     ? (std::to_string(t.weak_channels) + "/" + std::to_string(t.channels))
                                     : "-";
        char line[220];
        std::snprintf(line, sizeof(line), "%-20.20s %-10.10s %-16.16s %-11.11s %-11.11s %-8.8s %7s %8s",
                      layer.c_str(), op.c_str(), t.shape.c_str(), zeros.c_str(), near.c_str(),
                      weak.c_str(), t.macs_known ? pct(t.mac_share).c_str() : "-",
                      t.computed ? score_text(t.score).c_str() : "-");
        p.println(line);
    }
    p.println();
    p.println("Candidates only; nn does not prune. Magnitude is a where-to-look hint, not "
              "accuracy.");
    for (const auto& n : r.notes) {
        if (n.find("weights not in memory") == 0 || n.find("unsupported dtype") == 0) {
            p.errln("nn: warning: " + n);
        } else {
            p.println("nn: note: " + n);
        }
    }
    return kExitOk;
}
}  // namespace nn
