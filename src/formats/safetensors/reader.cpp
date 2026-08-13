#include "formats/readers.h"

#include "nn/json.h"
#include "nn/mmap.h"
#include "util/binary.h"

#include <algorithm>

namespace nn {
namespace {

DataType safetensors_dtype(std::string_view s) {
    if (s == "F64") {
        return DataType::Float64;
    }
    if (s == "F32") {
        return DataType::Float32;
    }
    if (s == "F16") {
        return DataType::Float16;
    }
    if (s == "BF16") {
        return DataType::BFloat16;
    }
    if (s == "I64") {
        return DataType::Int64;
    }
    if (s == "I32") {
        return DataType::Int32;
    }
    if (s == "I16") {
        return DataType::Int16;
    }
    if (s == "I8") {
        return DataType::Int8;
    }
    if (s == "U8") {
        return DataType::UInt8;
    }
    if (s == "BOOL") {
        return DataType::Bool;
    }
    if (s == "F8_E4M3") {
        return DataType::Float8E4M3;
    }
    if (s == "F8_E5M2") {
        return DataType::Float8E5M2;
    }
    return DataType::Unknown;
}

class SafeTensorsReader final : public ModelReader {
public:
    std::string name() const override { return "safetensors"; }
    std::string display_name() const override { return "SafeTensors"; }
    std::vector<std::string> extensions() const override { return {".safetensors"}; }

    bool probe(const FileSet& files) const override {
        if (files.paths.empty() || files.is_directory) {
            return false;
        }
        auto mapped = MappedFile::open(files.primary());
        if (!mapped || mapped.value().size() < 8) {
            return false;
        }
        BinaryReader r(mapped.value().span());
        auto n = r.u64le();
        if (!n) {
            return false;
        }
        const uint64_t header = n.value();
        if (header < 2 || header > mapped.value().size() - 8) {
            return false;
        }
        if (header > (uint64_t{100} << 20)) {
            return false;
        }
        auto bytes = r.bytes(std::min<uint64_t>(header, 16));
        if (!bytes) {
            return false;
        }
        // Header is JSON object.
        return !bytes.value().empty() && bytes.value()[0] == '{';
    }

    ModelCapabilities capabilities() const override {
        ModelCapabilities c;
        c.read = true;
        c.graph = false;
        c.weights = true;
        c.execute = false;
        c.convert = false;
        c.notes = "weight/tensor storage only; no computational graph";
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions& options) override {
        auto mapped = MappedFile::open(files.primary());
        if (!mapped) {
            return mapped.error();
        }
        BinaryReader r(mapped.value().span(), files.primary().string());
        auto header_len = r.u64le();
        if (!header_len) {
            return header_len.error();
        }
        if (header_len.value() > options.max_allocation_bytes ||
            header_len.value() > mapped.value().size() - 8) {
            return error(ErrorCode::ParseError, "SafeTensors header length invalid");
        }
        auto header_bytes = r.bytes(header_len.value());
        if (!header_bytes) {
            return header_bytes.error();
        }
        const std::string header_text(reinterpret_cast<const char*>(header_bytes.value().data()),
                                      header_bytes.value().size());
        auto json = parse_json(header_text);
        if (!json) {
            return json.error();
        }
        if (!json.value().is_object()) {
            return error(ErrorCode::ParseError, "SafeTensors header is not a JSON object");
        }

        ModelIR model;
        model.source_format = "safetensors";
        Graph graph;
        graph.name = files.primary().filename().string();
        const uint64_t data_start = 8 + header_len.value();

        TensorId tid = 0;
        for (const auto& [name, info] : json.value().as_object()) {
            if (name == "__metadata__") {
                if (info.is_object()) {
                    for (const auto& [k, v] : info.as_object()) {
                        model.metadata[k] = v.is_string() ? v.as_string() : v.dump(false);
                    }
                }
                continue;
            }
            if (!info.is_object()) {
                return error(ErrorCode::ParseError, "tensor entry is not an object: " + name);
            }
            Tensor t;
            t.id = tid++;
            t.name = name;
            t.constant = true;
            if (info.contains("dtype") && info.at("dtype").is_string()) {
                t.dtype = safetensors_dtype(info.at("dtype").as_string());
            }
            if (info.contains("shape") && info.at("shape").is_array()) {
                std::vector<int64_t> dims;
                for (const auto& d : info.at("shape").as_array()) {
                    dims.push_back(static_cast<int64_t>(d.as_number()));
                }
                t.shape = shape_from_ints(dims);
            }
            uint64_t begin = 0;
            uint64_t end = 0;
            if (info.contains("data_offsets") && info.at("data_offsets").is_array() &&
                info.at("data_offsets").as_array().size() >= 2) {
                begin = static_cast<uint64_t>(info.at("data_offsets").as_array()[0].as_number());
                end = static_cast<uint64_t>(info.at("data_offsets").as_array()[1].as_number());
            }
            if (end < begin) {
                return error(ErrorCode::ParseError, "invalid data_offsets for " + name);
            }
            t.storage_bytes = end - begin;
            if (data_start + end > mapped.value().size()) {
                return error(ErrorCode::ParseError, "tensor data exceeds file: " + name);
            }
            TensorDataReference ref;
            ref.file = files.primary();
            ref.offset = data_start + begin;
            ref.length = t.storage_bytes;
            if (options.load_weights) {
                t.data = std::move(ref);
            }
            graph.tensors.push_back(std::move(t));
        }
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_safetensors_reader() {
    return std::make_unique<SafeTensorsReader>();
}

}  // namespace nn
