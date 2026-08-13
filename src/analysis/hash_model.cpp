#include "nn/analysis.h"

#include "nn/hash.h"
#include "nn/mmap.h"

#include <algorithm>
#include <sstream>

namespace nn {

Result<HashReport> hash_model(const ModelIR& model, const HashOptions& options) {
    HashReport r;
    r.artifact_sha256 = model.sha256;
    r.graph_sha256 = sha256_hex(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(canonicalize_graph_text(model).data()),
        canonicalize_graph_text(model).size()));
    // Recompute once
    const std::string canon = canonicalize_graph_text(model);
    r.graph_sha256 = sha256_hex(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(canon.data()), canon.size()));
    r.canonical_sha256 = r.graph_sha256;

    Sha256 weights;
    const Graph* g = primary_graph(model);
    if (g) {
        std::vector<const Tensor*> consts;
        for (const auto& t : g->tensors) {
            if (t.constant) {
                consts.push_back(&t);
            }
        }
        std::sort(consts.begin(), consts.end(),
                  [](const Tensor* a, const Tensor* b) { return a->name < b->name; });
        for (const Tensor* t : consts) {
            weights.update(t->name);
            if (t->data && t->data->owned) {
                weights.update(std::span<const uint8_t>(t->data->owned->data(), t->data->owned->size()));
            } else if (t->data && !t->data->file.empty()) {
                // Chunked file hash of the tensor slice.
                auto mapped = MappedFile::open(t->data->file);
                if (mapped) {
                    auto sl = mapped.value().slice(t->data->offset, t->data->length);
                    weights.update(sl);
                }
            }
        }
    }
    r.weights_sha256 = weights.hex_digest();

    if (!options.tensor_name.empty() && g) {
        if (const Tensor* t = g->find_tensor_by_name(options.tensor_name)) {
            Sha256 th;
            if (t->data && t->data->owned) {
                th.update(std::span<const uint8_t>(t->data->owned->data(), t->data->owned->size()));
            } else if (t->data && !t->data->file.empty()) {
                auto mapped = MappedFile::open(t->data->file);
                if (mapped) {
                    th.update(mapped.value().slice(t->data->offset, t->data->length));
                }
            }
            r.tensor_name = options.tensor_name;
            r.tensor_sha256 = th.hex_digest();
        } else {
            return error(ErrorCode::InvalidArgument, "tensor not found: " + options.tensor_name);
        }
    }
    (void)options;
    return r;
}

}  // namespace nn
