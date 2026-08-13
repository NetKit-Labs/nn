#include "formats/readers.h"

#include "formats/detect.h"
#include "nn/mmap.h"
#include "util/protobuf.h"

#include <cstring>
#include <fstream>

namespace nn {
namespace {

bool looks_like_mlmodel(std::span<const uint8_t> data) {
    // Core ML .mlmodel is a protobuf Model. Weak structural check: field 1 (specificationVersion)
    // as varint is common. Also accept XML/plist-looking packages via directory.
    if (data.size() < 4) {
        return false;
    }
    return data[0] == 0x08;  // protobuf field 1 varint — common for spec version
}

class CoreMLReader final : public ModelReader {
public:
    std::string name() const override { return "coreml"; }
    std::string display_name() const override { return "Core ML"; }
    std::vector<std::string> extensions() const override { return {".mlmodel", ".mlpackage"}; }

    bool probe(const FileSet& files) const override {
        if (files.paths.empty()) {
            return false;
        }
        const auto& p = files.primary();
        if (files.is_directory || std::filesystem::is_directory(p)) {
            std::error_code ec;
            return std::filesystem::exists(p / "Manifest.json", ec) ||
                   std::filesystem::exists(p / "Data" / "com.apple.CoreML" / "model.mlmodel", ec);
        }
        auto mapped = MappedFile::open(p);
        if (!mapped) {
            return false;
        }
        return looks_like_mlmodel(mapped.value().span()) &&
               (extension_lower(p) == ".mlmodel");
    }

    ModelCapabilities capabilities() const override {
        ModelCapabilities c;
        c.read = true;
        c.graph = true;
        c.weights = true;
        c.execute = false;
        c.convert = false;
#ifdef NN_PLATFORM_APPLE
        c.notes = "inspection on all platforms; execution optional via Core ML on macOS (not compiled)";
#else
        c.notes = "file inspection only; Core ML execution is macOS-only and not compiled in";
#endif
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions&) override {
        ModelIR model;
        model.source_format = "coreml";
        Graph graph;
        auto path = files.primary();
        if (std::filesystem::is_directory(path)) {
            graph.name = path.filename().string();
            std::error_code ec;
            const auto manifest = path / "Manifest.json";
            if (std::filesystem::exists(manifest, ec)) {
                std::ifstream in(manifest);
                std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                model.metadata["manifest_present"] = "yes";
                if (s.size() > 2000) {
                    s.resize(2000);
                }
                model.metadata["manifest_prefix"] = s;
            }
        } else {
            graph.name = path.filename().string();
            auto mapped = MappedFile::open(path);
            if (!mapped) {
                return mapped.error();
            }
            ProtoReader r(mapped.value().span());
            while (!r.done()) {
                auto f = r.next();
                if (!f) {
                    break;
                }
                if (f.value().number == 1 && f.value().wire == ProtoWire::Varint) {
                    model.source_format_version = std::to_string(f.value().varint);
                    break;
                }
            }
        }
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_coreml_reader() { return std::make_unique<CoreMLReader>(); }

}  // namespace nn
