#include "formats/readers.h"

#include "formats/detect.h"
#include "nn/mmap.h"
#include "util/flatbuffer.h"

namespace nn {
namespace {

class MnnReader final : public ModelReader {
public:
    std::string name() const override { return "mnn"; }
    std::string display_name() const override { return "MNN"; }
    std::vector<std::string> extensions() const override { return {".mnn"}; }

    bool probe(const FileSet& files) const override {
        if (files.paths.empty() || files.is_directory) {
            return false;
        }
        auto mapped = MappedFile::open(files.primary());
        if (!mapped || mapped.value().size() < 8) {
            return false;
        }
        const auto sp = mapped.value().span();
        if (sp.size() >= 8 && sp[4] == 'M' && sp[5] == 'N' && sp[6] == 'N') {
            return true;
        }
        FlatBuffer fb(sp);
        return static_cast<bool>(fb.root_offset()) && extension_lower(files.primary()) == ".mnn";
    }

    ModelCapabilities capabilities() const override {
        ModelCapabilities c;
        c.read = true;
        c.graph = true;
        c.weights = true;
        c.execute = false;
        c.convert = false;
        c.notes = "container inspection; execution requires MNN runtime";
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions&) override {
        auto mapped = MappedFile::open(files.primary());
        if (!mapped) {
            return mapped.error();
        }
        ModelIR model;
        model.source_format = "mnn";
        Graph graph;
        graph.name = files.primary().filename().string();
        FlatBuffer fb(mapped.value().span());
        auto root = fb.root_offset();
        if (root) {
            model.metadata["flatbuffer_root"] = std::to_string(root.value());
        }
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_mnn_reader() { return std::make_unique<MnnReader>(); }

}  // namespace nn
