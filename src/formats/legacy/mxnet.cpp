#include "formats/readers.h"

#include "nn/json.h"

#include <fstream>

namespace nn {
namespace {

class MxnetReader final : public ModelReader {
public:
    std::string name() const override { return "mxnet"; }
    std::string display_name() const override { return "MXNet"; }
    std::vector<std::string> extensions() const override { return {".json", ".params"}; }

    bool probe(const FileSet& files) const override {
        auto p = files.primary();
        if (p.extension() == ".params") {
            p.replace_extension(".json");
        }
        std::ifstream in(p);
        if (!in) {
            return false;
        }
        std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto j = parse_json(s.size() > 2'000'000 ? s.substr(0, 2'000'000) : s);
        if (!j || !j.value().is_object()) {
            return false;
        }
        return j.value().contains("nodes") && j.value().contains("arg_nodes");
    }

    ModelCapabilities capabilities() const override {
        ModelCapabilities c;
        c.read = true;
        c.graph = true;
        c.weights = true;
        c.execute = false;
        c.convert = false;
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions&) override {
        auto p = files.primary();
        if (p.extension() == ".params") {
            p.replace_extension(".json");
        }
        std::ifstream in(p);
        if (!in) {
            return error(ErrorCode::FileNotFound, "cannot open MXNet symbol JSON: " + p.string());
        }
        std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto j = parse_json(s);
        if (!j) {
            return j.error();
        }
        ModelIR model;
        model.source_format = "mxnet";
        Graph graph;
        graph.name = p.stem().string();
        if (j.value().contains("nodes") && j.value().at("nodes").is_array()) {
            NodeId nid = 0;
            for (const auto& n : j.value().at("nodes").as_array()) {
                Node node;
                node.id = nid++;
                if (n.contains("name") && n.at("name").is_string()) {
                    node.name = n.at("name").as_string();
                }
                if (n.contains("op") && n.at("op").is_string()) {
                    node.op_type = n.at("op").as_string();
                }
                node.canonical = canonicalize_op("mxnet", node.op_type);
                graph.nodes.push_back(std::move(node));
            }
        }
        graph.rebuild_use_lists();
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_mxnet_reader() { return std::make_unique<MxnetReader>(); }

}  // namespace nn
