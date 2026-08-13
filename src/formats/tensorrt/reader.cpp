#include "formats/readers.h"

#include "nn/mmap.h"
#include "util/binary.h"

namespace nn {
namespace {

class TensorRTReader final : public ModelReader {
public:
    std::string name() const override { return "tensorrt"; }
    std::string display_name() const override { return "TensorRT Engine"; }
    std::vector<std::string> extensions() const override { return {".engine", ".plan"}; }

    bool probe(const FileSet& files) const override {
        if (files.paths.empty() || files.is_directory) {
            return false;
        }
        auto mapped = MappedFile::open(files.primary());
        if (!mapped || mapped.value().size() < 8) {
            return false;
        }
        const auto sp = mapped.value().span();
        // Serialized engines commonly start with "ptrt" / "ftrt" / similar ASCII tags.
        const bool ascii = (sp[0] == 'p' || sp[0] == 'f') && sp[1] == 't' && sp[2] == 'r' &&
                           sp[3] == 't';
        return ascii;
    }

    ModelCapabilities capabilities() const override {
        ModelCapabilities c;
        c.read = true;
        c.graph = false;
        c.weights = false;
        c.execute = false;
        c.convert = false;
        c.notes = "metadata only; cannot convert a TensorRT engine back to a training graph";
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions&) override {
        auto mapped = MappedFile::open(files.primary());
        if (!mapped) {
            return mapped.error();
        }
        ModelIR model;
        model.source_format = "tensorrt";
        Graph graph;
        graph.name = files.primary().filename().string();
        BinaryReader r(mapped.value().span());
        auto mag = r.string(4);
        if (mag) {
            model.metadata["magic"] = mag.value();
        }
        model.warnings.push_back(
            "TensorRT engines are compiled artifacts; graph reconstruction is not supported");
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_tensorrt_reader() { return std::make_unique<TensorRTReader>(); }

}  // namespace nn
