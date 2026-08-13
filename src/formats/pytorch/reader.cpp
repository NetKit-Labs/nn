#include "formats/readers.h"

#include "formats/detect.h"
#include "nn/mmap.h"

namespace nn {
namespace {

class PytorchReader final : public ModelReader {
public:
    std::string name() const override { return "pytorch"; }
    std::string display_name() const override { return "PyTorch"; }
    std::vector<std::string> extensions() const override { return {".pt", ".pth", ".pt2"}; }

    bool probe(const FileSet& files) const override {
        if (files.paths.empty() || files.is_directory) {
            return false;
        }
        auto mapped = MappedFile::open(files.primary());
        if (!mapped) {
            return false;
        }
        const auto sp = mapped.value().span();
        if (looks_like_zip(sp)) {
            auto names = zip_list_names(sp);
            if (!names) {
                return false;
            }
            for (const auto& n : names.value()) {
                if (n.find("data.pkl") != std::string::npos || n.find("byteorder") != std::string::npos ||
                    n.find("model.json") != std::string::npos ||
                    n.find(".safetensors") != std::string::npos || n.find("version") != std::string::npos) {
                    return true;
                }
            }
        }
        return looks_like_pickle(sp);
    }

    ModelCapabilities capabilities() const override {
        ModelCapabilities c;
        c.read = true;
        c.graph = false;
        c.weights = false;
        c.execute = false;
        c.convert = false;
        c.notes = "safe inspection only; pickle deserialization is refused by default";
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions& options) override {
        auto mapped = MappedFile::open(files.primary());
        if (!mapped) {
            return mapped.error();
        }
        ModelIR model;
        model.source_format = "pytorch";
        Graph graph;
        graph.name = files.primary().filename().string();
        const auto sp = mapped.value().span();
        if (looks_like_pickle(sp)) {
            model.warnings.push_back(
                "artifact is a Python pickle; nn will not unpickle untrusted files");
            if (!options.allow_unsafe_deserialize) {
                model.metadata["unsafe"] = "pickle";
                model.metadata["load"] = "refused";
                model.graphs.push_back(std::move(graph));
                return model;
            }
            return error(ErrorCode::UnsafeOperation,
                         "unsafe pickle deserialize is not implemented; refused");
        }
        if (looks_like_zip(sp)) {
            auto names = zip_list_names(sp);
            if (!names) {
                return names.error();
            }
            bool has_pkl = false;
            bool has_constants = false;
            for (const auto& n : names.value()) {
                model.metadata["zip." + n] = "present";
                if (n.find("data.pkl") != std::string::npos) {
                    has_pkl = true;
                }
                if (n.find("constants.pkl") != std::string::npos) {
                    has_constants = true;
                }
            }
            if (has_pkl && has_constants) {
                model.source_format = "torchscript";
                model.metadata["kind"] = "torchscript_zip";
            } else if (has_pkl) {
                model.metadata["kind"] = "torch_save_zip";
            }
            model.warnings.push_back(
                "PyTorch zip archives may contain pickle payloads; graph/weight decode requires "
                "unsafe framework loading and is not performed");
        }
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_pytorch_reader() { return std::make_unique<PytorchReader>(); }

}  // namespace nn
