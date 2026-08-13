#include "formats/readers.h"

#include "formats/detect.h"

#include <fstream>
#include <sstream>

namespace nn {
namespace {

class DarknetReader final : public ModelReader {
public:
    std::string name() const override { return "darknet"; }
    std::string display_name() const override { return "Darknet"; }
    std::vector<std::string> extensions() const override { return {".cfg", ".weights"}; }

    bool probe(const FileSet& files) const override {
        auto p = files.primary();
        if (extension_lower(p) == ".weights") {
            p.replace_extension(".cfg");
        }
        std::ifstream in(p);
        if (!in) {
            return false;
        }
        std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return s.find("[net]") != std::string::npos || s.find("[convolutional]") != std::string::npos;
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
        if (extension_lower(p) == ".weights") {
            p.replace_extension(".cfg");
        }
        std::ifstream in(p);
        if (!in) {
            return error(ErrorCode::FileNotFound, "cannot open Darknet cfg: " + p.string());
        }
        ModelIR model;
        model.source_format = "darknet";
        Graph graph;
        graph.name = p.stem().string();
        std::string line;
        NodeId nid = 0;
        std::string current;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            if (line.front() == '[' && line.back() == ']') {
                current = line.substr(1, line.size() - 2);
                if (current != "net") {
                    Node node;
                    node.id = nid++;
                    node.op_type = current;
                    node.name = current + "_" + std::to_string(node.id);
                    node.canonical = canonicalize_op("darknet", current);
                    graph.nodes.push_back(std::move(node));
                }
            }
        }
        graph.rebuild_use_lists();
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_darknet_reader() { return std::make_unique<DarknetReader>(); }

}  // namespace nn
