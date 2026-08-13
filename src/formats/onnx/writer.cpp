#include "formats/onnx/writer.h"

#include "formats/onnx/proto.h"

#include "util/protobuf.h"

#include <cstring>
#include <fstream>

namespace nn {
namespace {

std::vector<uint8_t> encode_tensor_type(DataType dtype, const std::vector<int64_t>& dims) {
    ProtoWriter shape;
    for (int64_t d : dims) {
        ProtoWriter dim;
        dim.varint_field(1, static_cast<uint64_t>(d));
        shape.message_field(1, dim.data());
    }
    ProtoWriter tensor;
    tensor.varint_field(1, static_cast<uint64_t>(onnx::onnx_from_tensor_dtype(dtype)));
    tensor.message_field(2, shape.data());
    ProtoWriter type;
    type.message_field(1, tensor.data());
    return type.take();
}

std::vector<uint8_t> encode_value_info(const std::string& name, DataType dtype,
                                      const std::vector<int64_t>& dims) {
    ProtoWriter v;
    v.string_field(1, name);
    v.message_field(2, encode_tensor_type(dtype, dims));
    return v.take();
}

std::vector<uint8_t> encode_tensor_proto(const std::string& name, DataType dtype,
                                        const std::vector<int64_t>& dims,
                                        const std::vector<float>& data) {
    ProtoWriter t;
    for (int64_t d : dims) {
        t.varint_field(1, static_cast<uint64_t>(d));
    }
    t.varint_field(2, static_cast<uint64_t>(onnx::onnx_from_tensor_dtype(dtype)));
    t.string_field(8, name);
    if (!data.empty()) {
        std::vector<uint8_t> raw(data.size() * sizeof(float));
        std::memcpy(raw.data(), data.data(), raw.size());
        t.bytes_field(9, raw);
    }
    return t.take();
}

}  // namespace

Result<std::vector<uint8_t>> encode_simple_onnx(const SimpleOnnxSpec& spec) {
    ProtoWriter graph;
    graph.string_field(2, spec.name);

    std::vector<std::string> input_names;
    for (int i = 0; i < spec.inputs; ++i) {
        const std::string n = spec.inputs == 1 ? "input" : ("input" + std::to_string(i));
        input_names.push_back(n);
        graph.message_field(11, encode_value_info(n, spec.dtype, spec.input_shape));
    }
    graph.message_field(12, encode_value_info("output", spec.dtype,
                                              spec.output_shape.empty() ? spec.input_shape
                                                                        : spec.output_shape));

    if (spec.with_weight) {
        const auto wshape = spec.weight_shape.empty() ? spec.input_shape : spec.weight_shape;
        graph.message_field(5, encode_tensor_proto("weight", spec.dtype, wshape, spec.weight));
    }

    ProtoWriter node;
    for (const auto& n : input_names) {
        node.string_field(1, n);
    }
    if (spec.with_weight) {
        node.string_field(1, "weight");
    }
    node.string_field(2, "output");
    node.string_field(3, spec.op_type + "_0");
    node.string_field(4, spec.op_type);
    graph.message_field(1, node.data());

    ProtoWriter model;
    model.varint_field(1, 8);  // ir_version
    model.string_field(2, "nn-test");
    model.string_field(3, "0.1.0");
    model.message_field(7, graph.data());

    ProtoWriter opset;
    opset.string_field(1, "");
    opset.varint_field(2, 13);
    model.message_field(8, opset.data());
    return model.take();
}

Status write_simple_onnx(const std::filesystem::path& path, const SimpleOnnxSpec& spec) {
    auto bytes = encode_simple_onnx(spec);
    if (!bytes) {
        return Status::err(bytes.error());
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return Status::err(error(ErrorCode::FileError, "cannot write " + path.string()));
    }
    out.write(reinterpret_cast<const char*>(bytes.value().data()),
              static_cast<std::streamsize>(bytes.value().size()));
    return Status::ok();
}

Status write_onnx_model(const std::filesystem::path& path, const ModelIR& model) {
    const Graph* g = primary_graph(model);
    if (!g) {
        return Status::err(error(ErrorCode::InvalidGraph, "model has no graph"));
    }
    ProtoWriter graph;
    graph.string_field(2, g->name);

    auto tensor_dims = [](const Tensor& t) {
        std::vector<int64_t> d;
        for (const auto& dim : t.shape.dims) {
            d.push_back(dim.value.value_or(0));
        }
        return d;
    };

    for (TensorId id : g->inputs) {
        if (const Tensor* t = g->find_tensor(id)) {
            graph.message_field(11, encode_value_info(t->name, t->dtype, tensor_dims(*t)));
        }
    }
    for (TensorId id : g->outputs) {
        if (const Tensor* t = g->find_tensor(id)) {
            graph.message_field(12, encode_value_info(t->name, t->dtype, tensor_dims(*t)));
        }
    }
    for (const auto& t : g->tensors) {
        if (!t.constant) {
            continue;
        }
        auto payload = tensor_payload_bytes(t);
        if (!payload) {
            continue;
        }
        ProtoWriter tw;
        for (int64_t d : tensor_dims(t)) {
            tw.varint_field(1, static_cast<uint64_t>(d));
        }
        tw.varint_field(2, static_cast<uint64_t>(onnx::onnx_from_tensor_dtype(t.dtype)));
        tw.string_field(8, t.name);
        tw.bytes_field(9, payload.value());
        graph.message_field(5, tw.data());
    }
    for (const auto& n : g->nodes) {
        ProtoWriter node;
        for (TensorId id : n.inputs) {
            if (const Tensor* t = g->find_tensor(id)) {
                node.string_field(1, t->name);
            }
        }
        for (TensorId id : n.outputs) {
            if (const Tensor* t = g->find_tensor(id)) {
                node.string_field(2, t->name);
            }
        }
        if (!n.name.empty()) {
            node.string_field(3, n.name);
        }
        node.string_field(4, n.op_type);
        if (!n.domain.empty()) {
            node.string_field(7, n.domain);
        }
        for (const auto& [k, a] : n.attributes) {
            ProtoWriter attr;
            attr.string_field(1, k);
            switch (a.kind) {
                case Attribute::Kind::Int:
                    attr.varint_field(3, static_cast<uint64_t>(a.i));
                    break;
                case Attribute::Kind::Float: {
                    float fv = static_cast<float>(a.f);
                    uint32_t bits = 0;
                    std::memcpy(&bits, &fv, 4);
                    attr.fixed32_field(2, bits);
                    break;
                }
                case Attribute::Kind::String:
                    attr.string_field(4, a.s);
                    break;
                case Attribute::Kind::Ints:
                    for (int64_t v : a.ints) {
                        attr.varint_field(8, static_cast<uint64_t>(v));
                    }
                    break;
                default:
                    break;
            }
            node.message_field(5, attr.data());
        }
        graph.message_field(1, node.data());
    }

    ProtoWriter mp;
    mp.varint_field(1, 8);
    if (!model.producer.empty()) {
        mp.string_field(2, model.producer);
    } else {
        mp.string_field(2, "nn");
    }
    mp.message_field(7, graph.data());
    ProtoWriter opset;
    opset.string_field(1, "");
    opset.varint_field(2, 13);
    mp.message_field(8, opset.data());

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return Status::err(error(ErrorCode::FileError, "cannot write " + path.string()));
    }
    out.write(reinterpret_cast<const char*>(mp.data().data()),
              static_cast<std::streamsize>(mp.data().size()));
    return Status::ok();
}

}  // namespace nn
