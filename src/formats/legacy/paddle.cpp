#include "formats/readers.h"

#include "nn/json.h"

#include <fstream>

namespace nn {
namespace {

class PaddleReader final : public ModelReader {
public:
    std::string name() const override { return "paddle"; }
    std::string display_name() const override { return "PaddlePaddle"; }
    std::vector<std::string> extensions() const override { return {".pdmodel", ".json"}; }

    bool probe(const FileSet& files) const override {
        auto p = files.primary();
        if (files.is_directory) {
            std::error_code ec;
            return std::filesystem::exists(p / "inference.pdmodel", ec) ||
                   std::filesystem::exists(p / "model.pdmodel", ec) ||
                   std::filesystem::exists(p / "inference.json", ec);
        }
        const auto ext = p.extension().string();
        if (ext == ".pdmodel") {
            return true;
        }
        if (ext == ".json") {
            std::ifstream in(p);
            if (!in) {
                return false;
            }
            std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            auto j = parse_json(s.size() > 1'000'000 ? s.substr(0, 1'000'000) : s);
            return j && j.value().is_object() &&
                   (j.value().contains("program") || j.value().contains("op_types") ||
                    j.value().contains("ops"));
        }
        return false;
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
        ModelIR model;
        model.source_format = "paddle";
        Graph graph;
        graph.name = files.primary().filename().string();
        model.metadata["kind"] = "paddle_inference";
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_paddle_reader() { return std::make_unique<PaddleReader>(); }

}  // namespace nn
