#include "formats/readers.h"

#include "formats/detect.h"
#include "nn/mmap.h"
#include "util/protobuf.h"

namespace nn {
namespace {

class TensorflowReader final : public ModelReader {
public:
    std::string name() const override { return "tensorflow"; }
    std::string display_name() const override { return "TensorFlow"; }
    std::vector<std::string> extensions() const override { return {".pb", ".pbtxt"}; }

    bool probe(const FileSet& files) const override {
        if (files.paths.empty()) {
            return false;
        }
        auto p = files.primary();
        if (files.is_directory || std::filesystem::is_directory(p)) {
            std::error_code ec;
            return std::filesystem::exists(p / "saved_model.pb", ec) ||
                   std::filesystem::exists(p / "saved_model.pbtxt", ec);
        }
        if (p.filename() == "saved_model.pb" || p.filename() == "saved_model.pbtxt") {
            return true;
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
        c.notes = "SavedModel directory inspection; TensorFlow runtime is optional and not compiled in";
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions&) override {
        auto p = files.primary();
        if (std::filesystem::is_regular_file(p) && p.filename() != "saved_model.pb") {
            p = p.parent_path();
        }
        if (std::filesystem::is_regular_file(files.primary()) &&
            files.primary().filename() == "saved_model.pb") {
            p = files.primary().parent_path();
        }
        const auto pb = std::filesystem::is_directory(p) ? (p / "saved_model.pb") : files.primary();
        auto mapped = MappedFile::open(pb);
        if (!mapped) {
            return mapped.error();
        }
        ModelIR model;
        model.source_format = "tensorflow";
        Graph graph;
        graph.name = p.filename().string();
        std::error_code ec;
        if (std::filesystem::exists(p / "variables", ec)) {
            model.metadata["variables"] = "present";
        }
        ProtoReader r(mapped.value().span());
        int fields = 0;
        while (!r.done() && fields < 64) {
            auto f = r.next();
            if (!f) {
                break;
            }
            ++fields;
            if (f.value().number == 2 && f.value().wire == ProtoWire::Length) {
                model.metadata["meta_graph_bytes"] = std::to_string(f.value().bytes.size());
            }
        }
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_tensorflow_reader() {
    return std::make_unique<TensorflowReader>();
}

}  // namespace nn
