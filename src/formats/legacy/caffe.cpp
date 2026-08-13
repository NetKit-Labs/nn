#include "formats/readers.h"

#include "formats/detect.h"

#include <fstream>

namespace nn {
namespace {

class CaffeReader final : public ModelReader {
public:
    std::string name() const override { return "caffe"; }
    std::string display_name() const override { return "Caffe"; }
    std::vector<std::string> extensions() const override { return {".prototxt", ".caffemodel"}; }

    bool probe(const FileSet& files) const override {
        auto p = files.primary();
        if (extension_lower(p) == ".caffemodel") {
            p.replace_extension(".prototxt");
        }
        std::ifstream in(p);
        if (!in) {
            return false;
        }
        std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return s.find("layer") != std::string::npos &&
               (s.find("name:") != std::string::npos || s.find("type:") != std::string::npos);
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
        if (extension_lower(p) == ".caffemodel") {
            p.replace_extension(".prototxt");
        }
        std::ifstream in(p);
        if (!in) {
            return error(ErrorCode::FileNotFound, "cannot open Caffe prototxt: " + p.string());
        }
        std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        ModelIR model;
        model.source_format = "caffe";
        Graph graph;
        graph.name = p.stem().string();
        std::size_t pos = 0;
        NodeId nid = 0;
        while ((pos = s.find("layer", pos)) != std::string::npos) {
            Node node;
            node.id = nid++;
            auto name_p = s.find("name:", pos);
            auto type_p = s.find("type:", pos);
            auto next = s.find("layer", pos + 5);
            auto grab = [&](std::size_t at) {
                if (at == std::string::npos || (next != std::string::npos && at > next)) {
                    return std::string();
                }
                auto q1 = s.find('"', at);
                auto q2 = s.find('"', q1 + 1);
                if (q1 == std::string::npos || q2 == std::string::npos) {
                    return std::string();
                }
                return s.substr(q1 + 1, q2 - q1 - 1);
            };
            node.name = grab(name_p);
            node.op_type = grab(type_p);
            node.canonical = canonicalize_op("caffe", node.op_type);
            graph.nodes.push_back(std::move(node));
            pos += 5;
        }
        graph.rebuild_use_lists();
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_caffe_reader() { return std::make_unique<CaffeReader>(); }

}  // namespace nn
