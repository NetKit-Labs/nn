#include "formats/readers.h"

#include "formats/onnx/proto.h"

#include "nn/mmap.h"

namespace nn {
namespace {

class OnnxReader final : public ModelReader {
public:
    std::string name() const override { return "onnx"; }
    std::string display_name() const override { return "ONNX"; }
    std::vector<std::string> extensions() const override { return {".onnx"}; }

    bool probe(const FileSet& files) const override {
        if (files.paths.empty() || files.is_directory) {
            return false;
        }
        const auto mapped = MappedFile::open(files.primary());
        if (!mapped || mapped.value().size() < 8) {
            return false;
        }
        // ONNX is protobuf ModelProto. Require ir_version (field 1) in a plausible range.
        ProtoReader r(mapped.value().span());
        bool saw_ir = false;
        bool saw_graph = false;
        int fields = 0;
        while (!r.done() && fields < 32) {
            auto f = r.next();
            if (!f) {
                return false;
            }
            ++fields;
            if (f.value().number == 1 && f.value().wire == ProtoWire::Varint) {
                const uint64_t v = f.value().varint;
                if (v >= 1 && v <= 12) {
                    saw_ir = true;
                }
            }
            if (f.value().number == 7 && f.value().wire == ProtoWire::Length) {
                saw_graph = true;
            }
            if (saw_ir && saw_graph) {
                return true;
            }
        }
        return saw_ir;
    }

    ModelCapabilities capabilities() const override {
        ModelCapabilities c;
        c.read = true;
        c.graph = true;
        c.weights = true;
#if defined(NN_HAS_ONNXRUNTIME)
        c.execute = true;
        c.execute_compiled = true;
        c.notes = "graph and weights; ONNX rewrite via nn convert --to onnx; "
                  "execution via ONNX Runtime";
#else
        c.execute = false;
        c.execute_compiled = false;
        c.notes = "graph and weights; ONNX rewrite via nn convert --to onnx; "
                  "execution via the reference backend when operators are supported";
#endif
        c.convert = true;
        c.convert_compiled = true;
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions& options) override {
        const auto mapped = MappedFile::open(files.primary());
        if (!mapped) {
            return mapped.error();
        }
        if (mapped.value().size() > options.max_allocation_bytes) {
            return error(ErrorCode::LimitExceeded, "ONNX file exceeds allocation limit");
        }
        return onnx::parse_model_proto(mapped.value().span(), options, files.primary());
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_onnx_reader() {
    return std::make_unique<OnnxReader>();
}

}  // namespace nn
