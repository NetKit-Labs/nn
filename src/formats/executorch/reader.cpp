#include "formats/readers.h"

#include "formats/detect.h"
#include "nn/mmap.h"
#include "util/flatbuffer.h"

#include <cstring>

namespace nn {
namespace {

class ExecuTorchReader final : public ModelReader {
public:
    std::string name() const override { return "executorch"; }
    std::string display_name() const override { return "ExecuTorch"; }
    std::vector<std::string> extensions() const override { return {".pte"}; }

    bool probe(const FileSet& files) const override {
        if (files.paths.empty() || files.is_directory) {
            return false;
        }
        auto mapped = MappedFile::open(files.primary());
        if (!mapped || mapped.value().size() < 8) {
            return false;
        }
        const auto sp = mapped.value().span();
        // Common identifiers: ET12 / extended header. Also accept FlatBuffer-looking files
        // with a .pte extension only as a weak fallback after identifier check.
        if (sp.size() >= 8) {
            const char* id = reinterpret_cast<const char*>(sp.data() + 4);
            if (std::strncmp(id, "ET12", 4) == 0 || std::strncmp(id, "XNNP", 4) == 0) {
                return true;
            }
        }
        FlatBuffer fb(sp);
        return static_cast<bool>(fb.root_offset()) && extension_lower(files.primary()) == ".pte";
    }

    ModelCapabilities capabilities() const override {
        ModelCapabilities c;
        c.read = true;
        c.graph = true;
        c.weights = true;
        c.execute = false;
        c.convert = false;
        c.notes = "serialized program inspection; execution requires an ExecuTorch backend";
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions&) override {
        auto mapped = MappedFile::open(files.primary());
        if (!mapped) {
            return mapped.error();
        }
        ModelIR model;
        model.source_format = "executorch";
        Graph graph;
        graph.name = files.primary().filename().string();
        FlatBuffer fb(mapped.value().span());
        auto root = fb.root_offset();
        if (!root) {
            return root.error();
        }
        model.metadata["flatbuffer_root"] = std::to_string(root.value());
        model.warnings.push_back(
            "ExecuTorch schema details beyond the container header are parsed conservatively");
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_executorch_reader() {
    return std::make_unique<ExecuTorchReader>();
}

}  // namespace nn
