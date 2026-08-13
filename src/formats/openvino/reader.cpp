#include "formats/readers.h"

#include "formats/detect.h"
#include "nn/mmap.h"

#include <fstream>

namespace nn {
namespace {

bool xml_contains(std::span<const uint8_t> data, std::string_view needle) {
    const auto* b = reinterpret_cast<const char*>(data.data());
    std::string_view sv(b, data.size() > 4096 ? 4096 : data.size());
    return sv.find(needle) != std::string_view::npos;
}

class OpenVinoReader final : public ModelReader {
public:
    std::string name() const override { return "openvino"; }
    std::string display_name() const override { return "OpenVINO IR"; }
    std::vector<std::string> extensions() const override { return {".xml", ".bin"}; }

    bool probe(const FileSet& files) const override {
        if (files.paths.empty()) {
            return false;
        }
        auto xml = files.primary();
        if (extension_lower(xml) == ".bin") {
            xml.replace_extension(".xml");
        }
        auto mapped = MappedFile::open(xml);
        if (!mapped) {
            return false;
        }
        return xml_contains(mapped.value().span(), "<net") &&
               (xml_contains(mapped.value().span(), "layers") ||
                xml_contains(mapped.value().span(), "<layer"));
    }

    ModelCapabilities capabilities() const override {
        ModelCapabilities c;
        c.read = true;
        c.graph = true;
        c.weights = true;
        c.execute = false;
        c.convert = false;
        c.notes = "IR XML inspection; execution requires OpenVINO Runtime";
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions&) override {
        auto xml = files.primary();
        std::filesystem::path bin;
        if (extension_lower(xml) == ".bin") {
            bin = xml;
            xml.replace_extension(".xml");
        } else {
            bin = xml;
            bin.replace_extension(".bin");
        }
        std::ifstream in(xml);
        if (!in) {
            return error(ErrorCode::FileNotFound, "cannot open OpenVINO XML: " + xml.string());
        }
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        ModelIR model;
        model.source_format = "openvino";
        Graph graph;
        graph.name = xml.stem().string();

        auto find_attr = [&](std::string_view tag, std::string_view key) -> std::string {
            const auto p = text.find(tag);
            if (p == std::string::npos) {
                return {};
            }
            const auto k = text.find(key, p);
            if (k == std::string::npos) {
                return {};
            }
            const auto q1 = text.find('"', k);
            if (q1 == std::string::npos) {
                return {};
            }
            const auto q2 = text.find('"', q1 + 1);
            if (q2 == std::string::npos) {
                return {};
            }
            return text.substr(q1 + 1, q2 - q1 - 1);
        };
        const std::string ver = find_attr("<net", "version=");
        if (!ver.empty()) {
            model.source_format_version = ver;
        }
        const std::string name = find_attr("<net", "name=");
        if (!name.empty()) {
            graph.name = name;
        }

        std::size_t pos = 0;
        NodeId nid = 0;
        while (true) {
            pos = text.find("<layer ", pos);
            if (pos == std::string::npos) {
                break;
            }
            Node node;
            node.id = nid++;
            auto grab = [&](std::string_view key) {
                const auto k = text.find(key, pos);
                const auto endtag = text.find('>', pos);
                if (k == std::string::npos || k > endtag) {
                    return std::string();
                }
                const auto q1 = text.find('"', k);
                const auto q2 = text.find('"', q1 + 1);
                if (q1 == std::string::npos || q2 == std::string::npos) {
                    return std::string();
                }
                return text.substr(q1 + 1, q2 - q1 - 1);
            };
            node.name = grab("name=");
            node.op_type = grab("type=");
            node.canonical = canonicalize_op("openvino", node.op_type);
            graph.nodes.push_back(std::move(node));
            ++pos;
        }
        std::error_code ec;
        if (std::filesystem::exists(bin, ec)) {
            model.metadata["weights_bin"] = bin.string();
            model.metadata["weights_bytes"] =
                std::to_string(static_cast<uint64_t>(std::filesystem::file_size(bin, ec)));
        }
        graph.rebuild_use_lists();
        model.graphs.push_back(std::move(graph));
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_openvino_reader() { return std::make_unique<OpenVinoReader>(); }

}  // namespace nn
