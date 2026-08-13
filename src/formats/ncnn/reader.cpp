#include "formats/readers.h"

#include "formats/detect.h"

#include <fstream>
#include <sstream>

namespace nn {
namespace {

class NcnnReader final : public ModelReader {
public:
    std::string name() const override { return "ncnn"; }
    std::string display_name() const override { return "NCNN"; }
    std::vector<std::string> extensions() const override { return {".param", ".bin"}; }

    bool probe(const FileSet& files) const override {
        auto param = files.primary();
        if (extension_lower(param) == ".bin") {
            param.replace_extension(".param");
        }
        std::ifstream in(param);
        if (!in) {
            return false;
        }
        std::string line;
        std::getline(in, line);
        // Magic: 7767517
        return line.find("7767517") != std::string::npos;
    }

    ModelCapabilities capabilities() const override {
        ModelCapabilities c;
        c.read = true;
        c.graph = true;
        c.weights = true;
        c.execute = false;
        c.convert = false;
        c.notes = "param graph inspection; execution requires NCNN runtime";
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions&) override {
        auto param = files.primary();
        std::filesystem::path bin = param;
        if (extension_lower(param) == ".bin") {
            bin = param;
            param.replace_extension(".param");
        } else {
            bin.replace_extension(".bin");
        }
        std::ifstream in(param);
        if (!in) {
            return error(ErrorCode::FileNotFound, "cannot open NCNN param: " + param.string());
        }
        ModelIR model;
        model.source_format = "ncnn";
        Graph graph;
        graph.name = param.stem().string();
        std::string line;
        std::getline(in, line);  // magic
        int layer_count = 0;
        int blob_count = 0;
        if (std::getline(in, line)) {
            std::istringstream ls(line);
            ls >> layer_count >> blob_count;
        }
        model.metadata["layer_count"] = std::to_string(layer_count);
        model.metadata["blob_count"] = std::to_string(blob_count);
        NodeId nid = 0;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream ls(line);
            Node node;
            node.id = nid++;
            ls >> node.op_type >> node.name;
            node.canonical = canonicalize_op("ncnn", node.op_type);
            graph.nodes.push_back(std::move(node));
        }
        std::error_code ec;
        if (std::filesystem::exists(bin, ec)) {
            model.metadata["weights_bin"] = bin.string();
        }
        graph.rebuild_use_lists();
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_ncnn_reader() { return std::make_unique<NcnnReader>(); }

}  // namespace nn
