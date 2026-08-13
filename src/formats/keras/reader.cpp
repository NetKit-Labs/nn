#include "formats/readers.h"

#include "formats/detect.h"
#include "nn/mmap.h"

namespace nn {
namespace {

class KerasReader final : public ModelReader {
public:
    std::string name() const override { return "keras"; }
    std::string display_name() const override { return "Keras"; }
    std::vector<std::string> extensions() const override { return {".keras", ".h5", ".hdf5"}; }

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
                if (n.find("config.json") != std::string::npos ||
                    n.find("model.weights") != std::string::npos ||
                    n.find("metadata.json") != std::string::npos) {
                    return true;
                }
            }
        }
        return looks_like_hdf5(sp);
    }

    ModelCapabilities capabilities() const override {
        ModelCapabilities c;
        c.read = true;
        c.graph = true;
        c.weights = true;
        c.execute = false;
        c.convert = false;
        c.notes = "recognizes .keras zip and HDF5; some HDF5 files are weights-only";
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions&) override {
        auto mapped = MappedFile::open(files.primary());
        if (!mapped) {
            return mapped.error();
        }
        ModelIR model;
        model.source_format = "keras";
        Graph graph;
        graph.name = files.primary().filename().string();
        const auto sp = mapped.value().span();
        if (looks_like_zip(sp)) {
            auto names = zip_list_names(sp);
            if (names) {
                bool has_config = false;
                bool has_weights = false;
                for (const auto& n : names.value()) {
                    model.metadata["zip." + n] = "present";
                    if (n.find("config.json") != std::string::npos) {
                        has_config = true;
                    }
                    if (n.find("weights") != std::string::npos) {
                        has_weights = true;
                    }
                }
                model.metadata["has_config"] = has_config ? "yes" : "no";
                model.metadata["has_weights"] = has_weights ? "yes" : "no";
            }
        } else if (looks_like_hdf5(sp)) {
            model.metadata["container"] = "hdf5";
            model.warnings.push_back(
                "HDF5 Keras files may contain weights only; full graph decode is not claimed");
        }
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_keras_reader() { return std::make_unique<KerasReader>(); }

}  // namespace nn
