#include "nn/compat.h"

#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace nn {
namespace {

std::string lower(std::string_view s) {
    std::string o(s);
    std::transform(o.begin(), o.end(), o.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return o;
}

std::string compact(std::string_view s) {
    std::string o = lower(s);
    const auto dot = o.find_last_of('.');
    if (dot != std::string::npos) {
        o = o.substr(dot + 1);
    }
    std::string c;
    for (char ch : o) {
        if (ch != '_' && ch != '-') {
            c.push_back(ch);
        }
    }
    return c;
}

CompatTable table(std::string runtime, std::string version, std::vector<std::string> ops,
                  std::string notes) {
    CompatTable t;
    t.runtime = std::move(runtime);
    t.version = std::move(version);
    t.ops = std::move(ops);
    t.notes = std::move(notes);
    return t;
}

}  // namespace

std::vector<CompatTable> builtin_compat_tables() {
    static const std::vector<std::string> kRef = {
        "add",    "sub",     "mul",     "div",        "relu",     "relu6",   "tanh",
        "sigmoid","clip",    "softmax", "identity",   "dropout",  "reshape", "squeeze",
        "unsqueeze","flatten","transpose","concat",   "matmul",   "gemm",    "conv",
        "constant","abs",    "neg",     "max",        "min",      "pow"};
    // Documented ONNX Runtime / TFLite operator names. These tables do not
    // imply that those runtimes are linked into this binary.
    static const std::vector<std::string> kOrt = {
        "abs", "add", "and", "argmax", "argmin", "averagepool", "batchnormalization",
        "cast", "clip", "concat", "constant", "conv", "convtranspose", "div", "dropout",
        "equal", "exp", "flatten", "gather", "gemm", "globalaveragepool", "identity",
        "leakyrelu", "log", "matmul", "maxpool", "mul", "neg", "pad", "pow", "reducemean",
        "reducesum", "relu", "reshape", "resize", "sigmoid", "slice", "softmax", "split",
        "sqrt", "squeeze", "sub", "tanh", "transpose", "unsqueeze", "where"};
    static const std::vector<std::string> kTflite = {
        "add", "average_pool_2d", "batch_to_space_nd", "cast", "concatenation", "conv_2d",
        "depthwise_conv_2d", "dequantize", "div", "fully_connected", "logistic", "max_pool_2d",
        "mean", "mul", "pad", "quantize", "relu", "relu6", "reshape", "resize_bilinear",
        "softmax", "squeeze", "strided_slice", "sub", "tanh", "transpose"};
    return {
        table("reference", "0.1.0", kRef,
              "built-in reference interpreter; compiled into this binary"),
        table("onnxruntime", "1.16.0", kOrt,
              "documented ONNX Runtime opset coverage; runtime is not linked unless "
              "NN_ENABLE_ONNXRUNTIME is on and the SDK is present"),
        table("tflite", "2.1.6", kTflite,
              "documented TFLite builtin ops; LiteRT runtime is not linked unless "
              "NN_ENABLE_LITERT_RUNTIME is on and the SDK is present"),
        table("litert", "2.1.6", kTflite,
              "alias of the tflite capability table; linked when NN_HAS_LITERT is set"),
    };
}

const CompatTable* find_compat_table(std::string_view runtime, std::string_view version) {
    static const auto kTables = builtin_compat_tables();
    const std::string rt = lower(runtime);
    const std::string ver = std::string(version);
    const CompatTable* fallback = nullptr;
    for (const auto& t : kTables) {
        if (lower(t.runtime) != rt) {
            continue;
        }
        if (ver.empty() || t.version == ver) {
            return &t;
        }
        fallback = &t;
    }
    return fallback;
}

CompatReport check_compat(const ModelIR& model, std::string_view runtime, std::string_view version) {
    CompatReport r;
    r.runtime = std::string(runtime);
    const CompatTable* table = find_compat_table(runtime, version);
    if (!table) {
        r.table_version = std::string(version);
        r.compatible = false;
        r.unsupported.push_back("no capability table for runtime '" + r.runtime + "'");
        return r;
    }
    r.table_version = table->version;
    const Graph* g = primary_graph(model);
    if (!g) {
        r.compatible = false;
        r.unsupported.push_back("model has no graph");
        return r;
    }
    std::unordered_set<std::string> ops;
    for (const auto& o : table->ops) {
        ops.insert(compact(o));
    }
    for (const auto& n : g->nodes) {
        ++r.total;
        const std::string op = compact(n.op_type);
        if (n.canonical == CanonicalOp::Constant || ops.count(op)) {
            ++r.supported;
        } else {
            r.unsupported.push_back(n.name.empty() ? n.op_type : n.name + " (" + n.op_type + ")");
        }
    }
    r.compatible = r.unsupported.empty();
    return r;
}

}  // namespace nn
