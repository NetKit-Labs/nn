#include "formats/readers.h"

#include "formats/detect.h"
#include "nn/mmap.h"
#include "util/binary.h"

#include <cstdlib>

namespace nn {
namespace {

enum class GgufType : uint32_t {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12
};

DataType gguf_tensor_type(uint32_t t) {
    switch (t) {
        case 0:
            return DataType::Float32;  // F32
        case 1:
            return DataType::Float16;  // F16
        case 2:
        case 3:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 33:
        case 34:
        case 35:
        case 36:
            return DataType::Int8;  // quantized types occupy integer storage
        default:
            return DataType::Unknown;
    }
}

const char* gguf_type_name(uint32_t t) {
    switch (t) {
        case 0:
            return "F32";
        case 1:
            return "F16";
        case 2:
            return "Q4_0";
        case 3:
            return "Q4_1";
        case 6:
            return "Q5_0";
        case 7:
            return "Q5_1";
        case 8:
            return "Q8_0";
        case 9:
            return "Q8_1";
        case 10:
            return "Q2_K";
        case 11:
            return "Q3_K";
        case 12:
            return "Q4_K";
        case 13:
            return "Q5_K";
        case 14:
            return "Q6_K";
        case 15:
            return "Q8_K";
        case 24:
            return "I8";
        case 25:
            return "I16";
        case 26:
            return "I32";
        case 27:
            return "I64";
        case 28:
            return "F64";
        case 30:
            return "BF16";
        default:
            return "unknown";
    }
}

Result<std::string> read_gguf_string(BinaryReader& r) {
    auto n = r.u64le();
    if (!n) {
        return n.error();
    }
    if (n.value() > (uint64_t{1} << 20)) {
        return error(ErrorCode::ParseError, "GGUF string length exceeds 1 MiB");
    }
    return r.string(n.value());
}

Result<std::string> skip_or_read_value(BinaryReader& r, uint32_t type, int depth) {
    if (depth > 8) {
        return error(ErrorCode::ParseError, "GGUF metadata nesting too deep");
    }
    switch (static_cast<GgufType>(type)) {
        case GgufType::Uint8:
        case GgufType::Int8:
        case GgufType::Bool: {
            auto v = r.u8();
            if (!v) {
                return v.error();
            }
            return std::to_string(v.value());
        }
        case GgufType::Uint16:
        case GgufType::Int16: {
            auto v = r.u16le();
            if (!v) {
                return v.error();
            }
            return std::to_string(v.value());
        }
        case GgufType::Uint32:
        case GgufType::Int32: {
            auto v = r.u32le();
            if (!v) {
                return v.error();
            }
            return std::to_string(v.value());
        }
        case GgufType::Float32: {
            auto v = r.f32le();
            if (!v) {
                return v.error();
            }
            return std::to_string(v.value());
        }
        case GgufType::Uint64: {
            auto v = r.u64le();
            if (!v) {
                return v.error();
            }
            return std::to_string(v.value());
        }
        case GgufType::Int64: {
            auto v = r.i64le();
            if (!v) {
                return v.error();
            }
            return std::to_string(v.value());
        }
        case GgufType::Float64: {
            auto v = r.f64le();
            if (!v) {
                return v.error();
            }
            return std::to_string(v.value());
        }
        case GgufType::String:
            return read_gguf_string(r);
        case GgufType::Array: {
            auto et = r.u32le();
            if (!et) {
                return et.error();
            }
            auto n = r.u64le();
            if (!n) {
                return n.error();
            }
            if (n.value() > 1'000'000) {
                return error(ErrorCode::ParseError, "GGUF array too large");
            }
            std::string joined;
            const uint64_t show = std::min<uint64_t>(n.value(), 8);
            for (uint64_t i = 0; i < n.value(); ++i) {
                auto item = skip_or_read_value(r, et.value(), depth + 1);
                if (!item) {
                    return item.error();
                }
                if (i < show) {
                    if (i) {
                        joined += ",";
                    }
                    joined += item.value();
                }
            }
            if (n.value() > show) {
                joined += ",...";
            }
            return "[" + joined + "]";
        }
        default:
            return error(ErrorCode::ParseError, "unknown GGUF value type " + std::to_string(type));
    }
}

class GgufReader final : public ModelReader {
public:
    std::string name() const override { return "gguf"; }
    std::string display_name() const override { return "GGUF"; }
    std::vector<std::string> extensions() const override { return {".gguf"}; }

    bool probe(const FileSet& files) const override {
        if (files.paths.empty() || files.is_directory) {
            return false;
        }
        const uint8_t mag[] = {'G', 'G', 'U', 'F'};
        return file_has_magic(files.primary(), mag);
    }

    ModelCapabilities capabilities() const override {
        ModelCapabilities c;
        c.read = true;
        c.graph = true;  // architecture metadata + tensor inventory, not a compute graph
        c.weights = true;
        c.execute = false;
        c.convert = false;
        c.notes = "container/tensor inspection; execution requires a GGUF runtime";
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions& options) override {
        auto mapped = MappedFile::open(files.primary());
        if (!mapped) {
            return mapped.error();
        }
        BinaryReader r(mapped.value().span(), files.primary().string());
        auto magic = r.bytes(4);
        if (!magic) {
            return magic.error();
        }
        auto version = r.u32le();
        if (!version) {
            return version.error();
        }
        if (version.value() < 1 || version.value() > 3) {
            return error(ErrorCode::ParseError,
                         "unsupported GGUF version " + std::to_string(version.value()));
        }
        auto tensor_count = r.u64le();
        if (!tensor_count) {
            return tensor_count.error();
        }
        auto kv_count = r.u64le();
        if (!kv_count) {
            return kv_count.error();
        }
        if (tensor_count.value() > 10'000'000 || kv_count.value() > 1'000'000) {
            return error(ErrorCode::ParseError, "GGUF counts implausibly large");
        }

        ModelIR model;
        model.source_format = "gguf";
        model.source_format_version = std::to_string(version.value());
        Graph graph;
        graph.name = files.primary().filename().string();

        for (uint64_t i = 0; i < kv_count.value(); ++i) {
            auto key = read_gguf_string(r);
            if (!key) {
                return key.error();
            }
            auto ty = r.u32le();
            if (!ty) {
                return ty.error();
            }
            auto val = skip_or_read_value(r, ty.value(), 0);
            if (!val) {
                return val.error();
            }
            model.metadata[key.value()] = val.value();
            if (key.value() == "general.architecture") {
                model.producer = val.value();
            }
            if (key.value() == "general.name") {
                graph.name = val.value();
            }
        }

        struct TInfo {
            std::string name;
            std::vector<int64_t> dims;
            uint32_t type = 0;
            uint64_t offset = 0;
        };
        std::vector<TInfo> infos;
        infos.reserve(static_cast<std::size_t>(tensor_count.value()));
        for (uint64_t i = 0; i < tensor_count.value(); ++i) {
            TInfo t;
            auto name = read_gguf_string(r);
            if (!name) {
                return name.error();
            }
            t.name = std::move(name.value());
            auto n_dims = r.u32le();
            if (!n_dims) {
                return n_dims.error();
            }
            if (n_dims.value() > 8) {
                return error(ErrorCode::ParseError, "GGUF tensor rank too high: " + t.name);
            }
            t.dims.resize(n_dims.value());
            for (uint32_t d = 0; d < n_dims.value(); ++d) {
                auto dim = r.u64le();
                if (!dim) {
                    return dim.error();
                }
                t.dims[d] = static_cast<int64_t>(dim.value());
            }
            auto ty = r.u32le();
            if (!ty) {
                return ty.error();
            }
            t.type = ty.value();
            auto off = r.u64le();
            if (!off) {
                return off.error();
            }
            t.offset = off.value();
            infos.push_back(std::move(t));
        }

        // Data section is aligned; GGUF v2/v3 typically align to 32.
        uint64_t alignment = 32;
        auto it = model.metadata.find("general.alignment");
        if (it != model.metadata.end()) {
            alignment = static_cast<uint64_t>(std::strtoull(it->second.c_str(), nullptr, 10));
            if (alignment == 0) {
                alignment = 32;
            }
        }
        const uint64_t data_start = (r.position() + alignment - 1) / alignment * alignment;

        for (std::size_t i = 0; i < infos.size(); ++i) {
            Tensor t;
            t.id = static_cast<TensorId>(i);
            t.name = infos[i].name;
            t.dtype = gguf_tensor_type(infos[i].type);
            t.shape = shape_from_ints(infos[i].dims);
            t.constant = true;
            t.quantization.quantized = (infos[i].type >= 2 && infos[i].type < 24);
            t.quantization.scheme = gguf_type_name(infos[i].type);
            if (auto b = tensor_storage_bytes(t)) {
                t.storage_bytes = b.value();
            }
            TensorDataReference ref;
            ref.file = files.primary();
            ref.offset = data_start + infos[i].offset;
            ref.length = t.storage_bytes;
            if (options.load_weights) {
                t.data = std::move(ref);
            }
            model.metadata["tensor." + t.name + ".ggml_type"] = gguf_type_name(infos[i].type);
            graph.tensors.push_back(std::move(t));
        }

        graph.rebuild_use_lists();
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_gguf_reader() { return std::make_unique<GgufReader>(); }

}  // namespace nn
