#include "formats/readers.h"

#include "formats/detect.h"
#include "nn/json.h"
#include "nn/mmap.h"

#include <fstream>

namespace nn {
namespace {

class TfjsReader final : public ModelReader {
public:
    std::string name() const override { return "tfjs"; }
    std::string display_name() const override { return "TensorFlow.js"; }
    std::vector<std::string> extensions() const override { return {".json"}; }

    bool probe(const FileSet& files) const override {
        auto path = files.primary();
        if (files.is_directory) {
            path /= "model.json";
        }
        std::ifstream in(path);
        if (!in) {
            return false;
        }
        std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto j = parse_json(s.size() > 1'000'000 ? s.substr(0, 1'000'000) : s);
        if (!j || !j.value().is_object()) {
            return false;
        }
        return j.value().contains("modelTopology") || j.value().contains("weightsManifest") ||
               j.value().contains("format");
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
        auto path = files.primary();
        if (files.is_directory) {
            path /= "model.json";
        }
        std::ifstream in(path);
        if (!in) {
            return error(ErrorCode::FileNotFound, "cannot open " + path.string());
        }
        std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto j = parse_json(s);
        if (!j) {
            return j.error();
        }
        ModelIR model;
        model.source_format = "tfjs";
        Graph graph;
        graph.name = path.parent_path().filename().string();
        if (j.value().contains("format") && j.value().at("format").is_string()) {
            model.source_format_version = j.value().at("format").as_string();
        }
        if (j.value().contains("generatedBy") && j.value().at("generatedBy").is_string()) {
            model.producer = j.value().at("generatedBy").as_string();
        }
        const Json* topo = &j.value().at("modelTopology");
        if (topo->is_object() && topo->contains("node") && topo->at("node").is_array()) {
            NodeId nid = 0;
            for (const auto& n : topo->at("node").as_array()) {
                Node node;
                node.id = nid++;
                if (n.contains("name") && n.at("name").is_string()) {
                    node.name = n.at("name").as_string();
                }
                if (n.contains("op") && n.at("op").is_string()) {
                    node.op_type = n.at("op").as_string();
                }
                node.canonical = canonicalize_op("tensorflow", node.op_type);
                graph.nodes.push_back(std::move(node));
            }
        }
        graph.rebuild_use_lists();
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_tfjs_reader() { return std::make_unique<TfjsReader>(); }

}  // namespace nn
