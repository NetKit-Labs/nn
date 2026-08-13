#include "formats/onnx/proto.h"

#include "nn/operator.h"

#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace nn {
namespace onnx {
namespace {

int64_t zigzag(uint64_t v) {
    // protobuf int64 is stored as varint of the raw two's complement bits, not zigzag
    // (sint64 is zigzag). ONNX uses int64 / int32 proto types which are varint-encoded
    // as unsigned reinterpretation of the signed value.
    return static_cast<int64_t>(v);
}

DataType map_dtype(int32_t t) {
    switch (t) {
        case 1:
            return DataType::Float32;
        case 2:
            return DataType::UInt8;
        case 3:
            return DataType::Int8;
        case 4:
            return DataType::UInt16;
        case 5:
            return DataType::Int16;
        case 6:
            return DataType::Int32;
        case 7:
            return DataType::Int64;
        case 8:
            return DataType::String;
        case 9:
            return DataType::Bool;
        case 10:
            return DataType::Float16;
        case 11:
            return DataType::Float64;
        case 12:
            return DataType::UInt32;
        case 13:
            return DataType::UInt64;
        case 14:
            return DataType::Complex64;
        case 15:
            return DataType::Complex128;
        case 16:
            return DataType::BFloat16;
        default:
            return DataType::Unknown;
    }
}

Result<std::vector<int64_t>> packed_int64s(std::span<const uint8_t> bytes) {
    std::vector<int64_t> out;
    std::size_t pos = 0;
    while (pos < bytes.size()) {
        auto v = ProtoReader::read_varint(bytes, pos);
        if (!v) {
            return v.error();
        }
        out.push_back(zigzag(v.value()));
    }
    return out;
}

Result<Shape> parse_shape(std::span<const uint8_t> bytes) {
    Shape shape;
    ProtoReader r(bytes);
    while (!r.done()) {
        auto f = r.next();
        if (!f) {
            return f.error();
        }
        if (f.value().number != 1 || f.value().wire != ProtoWire::Length) {
            continue;
        }
        Dimension dim;
        ProtoReader dr(f.value().bytes);
        while (!dr.done()) {
            auto df = dr.next();
            if (!df) {
                return df.error();
            }
            if (df.value().number == 1 && df.value().wire == ProtoWire::Varint) {
                dim.value = zigzag(df.value().varint);
            } else if (df.value().number == 2 && df.value().wire == ProtoWire::Length) {
                dim.symbol.assign(reinterpret_cast<const char*>(df.value().bytes.data()),
                                  df.value().bytes.size());
            }
        }
        shape.dims.push_back(std::move(dim));
    }
    return shape;
}

Result<std::pair<DataType, Shape>> parse_type_proto(std::span<const uint8_t> bytes) {
    DataType dtype = DataType::Unknown;
    Shape shape;
    ProtoReader r(bytes);
    while (!r.done()) {
        auto f = r.next();
        if (!f) {
            return f.error();
        }
        if (f.value().number == 1 && f.value().wire == ProtoWire::Length) {
            ProtoReader tr(f.value().bytes);
            while (!tr.done()) {
                auto tf = tr.next();
                if (!tf) {
                    return tf.error();
                }
                if (tf.value().number == 1 && tf.value().wire == ProtoWire::Varint) {
                    dtype = map_dtype(static_cast<int32_t>(tf.value().varint));
                } else if (tf.value().number == 2 && tf.value().wire == ProtoWire::Length) {
                    auto sh = parse_shape(tf.value().bytes);
                    if (!sh) {
                        return sh.error();
                    }
                    shape = std::move(sh.value());
                }
            }
        }
    }
    return std::make_pair(dtype, shape);
}

Result<Tensor> parse_value_info(std::span<const uint8_t> bytes, TensorId id) {
    Tensor t;
    t.id = id;
    ProtoReader r(bytes);
    while (!r.done()) {
        auto f = r.next();
        if (!f) {
            return f.error();
        }
        if (f.value().number == 1 && f.value().wire == ProtoWire::Length) {
            t.name.assign(reinterpret_cast<const char*>(f.value().bytes.data()),
                          f.value().bytes.size());
        } else if (f.value().number == 2 && f.value().wire == ProtoWire::Length) {
            auto ty = parse_type_proto(f.value().bytes);
            if (!ty) {
                return ty.error();
            }
            t.dtype = ty.value().first;
            t.shape = std::move(ty.value().second);
        }
    }
    return t;
}

Result<Tensor> parse_tensor_proto(std::span<const uint8_t> bytes, TensorId id,
                                 const LoadOptions& options, const std::filesystem::path& source) {
    Tensor t;
    t.id = id;
    t.constant = true;
    int32_t onnx_dtype = 0;
    std::vector<int64_t> dims;
    std::span<const uint8_t> raw;
    bool external = false;
    std::string ext_location;
    uint64_t ext_offset = 0;
    uint64_t ext_length = 0;

    ProtoReader r(bytes);
    while (!r.done()) {
        auto f = r.next();
        if (!f) {
            return f.error();
        }
        const auto& field = f.value();
        switch (field.number) {
            case 1: {  // dims
                if (field.wire == ProtoWire::Varint) {
                    dims.push_back(zigzag(field.varint));
                } else if (field.wire == ProtoWire::Length) {
                    auto more = packed_int64s(field.bytes);
                    if (!more) {
                        return more.error();
                    }
                    dims.insert(dims.end(), more.value().begin(), more.value().end());
                }
                break;
            }
            case 2:
                if (field.wire == ProtoWire::Varint) {
                    onnx_dtype = static_cast<int32_t>(field.varint);
                }
                break;
            case 8:
                if (field.wire == ProtoWire::Length) {
                    t.name.assign(reinterpret_cast<const char*>(field.bytes.data()),
                                  field.bytes.size());
                }
                break;
            case 9:
                if (field.wire == ProtoWire::Length) {
                    raw = field.bytes;
                }
                break;
            case 13:
                if (field.wire == ProtoWire::Length) {
                    std::string key;
                    std::string value;
                    ProtoReader er(field.bytes);
                    while (!er.done()) {
                        auto ef = er.next();
                        if (!ef) {
                            return ef.error();
                        }
                        if (ef.value().number == 1 && ef.value().wire == ProtoWire::Length) {
                            key.assign(reinterpret_cast<const char*>(ef.value().bytes.data()),
                                       ef.value().bytes.size());
                        } else if (ef.value().number == 2 && ef.value().wire == ProtoWire::Length) {
                            value.assign(reinterpret_cast<const char*>(ef.value().bytes.data()),
                                         ef.value().bytes.size());
                        }
                    }
                    if (key == "location") {
                        ext_location = value;
                    } else if (key == "offset") {
                        ext_offset = static_cast<uint64_t>(std::strtoull(value.c_str(), nullptr, 10));
                    } else if (key == "length") {
                        ext_length = static_cast<uint64_t>(std::strtoull(value.c_str(), nullptr, 10));
                    }
                }
                break;
            case 14:
                if (field.wire == ProtoWire::Varint && field.varint == 1) {
                    external = true;
                }
                break;
            default:
                break;
        }
    }

    t.dtype = map_dtype(onnx_dtype);
    t.shape = shape_from_ints(dims);
    if (auto bytes_r = tensor_storage_bytes(t)) {
        t.storage_bytes = bytes_r.value();
    }

    TensorDataReference ref;
    if (external) {
        ref.external = true;
        auto loc = source.parent_path() / ext_location;
        ref.file = loc;
        ref.offset = ext_offset;
        ref.length = ext_length ? ext_length : t.storage_bytes;
        if (options.follow_external_data && options.load_weights) {
            t.data = std::move(ref);
        } else {
            t.data = std::move(ref);
        }
    } else if (!raw.empty() && options.load_weights) {
        if (raw.size() > options.max_allocation_bytes) {
            return error(ErrorCode::LimitExceeded, "initializer exceeds allocation limit: " + t.name);
        }
        ref.owned = std::make_shared<const std::vector<uint8_t>>(raw.begin(), raw.end());
        ref.length = raw.size();
        t.data = std::move(ref);
        t.storage_bytes = raw.size();
    }
    return t;
}

}  // namespace

DataType tensor_dtype_from_onnx(int32_t t) { return map_dtype(t); }

int32_t onnx_from_tensor_dtype(DataType t) {
    switch (t) {
        case DataType::Float32:
            return 1;
        case DataType::UInt8:
            return 2;
        case DataType::Int8:
            return 3;
        case DataType::UInt16:
            return 4;
        case DataType::Int16:
            return 5;
        case DataType::Int32:
            return 6;
        case DataType::Int64:
            return 7;
        case DataType::String:
            return 8;
        case DataType::Bool:
            return 9;
        case DataType::Float16:
            return 10;
        case DataType::Float64:
            return 11;
        case DataType::UInt32:
            return 12;
        case DataType::UInt64:
            return 13;
        case DataType::Complex64:
            return 14;
        case DataType::Complex128:
            return 15;
        case DataType::BFloat16:
            return 16;
        default:
            return 0;
    }
}

Result<ModelIR> parse_model_proto(std::span<const uint8_t> bytes, const LoadOptions& options,
                                  const std::filesystem::path& source) {
    ModelIR model;
    model.source_format = "onnx";
    Graph graph;
    graph.id = 0;
    std::vector<std::string> opsets;

    struct PendingNode {
        Node node;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
    };
    std::vector<PendingNode> pending;
    std::unordered_map<std::string, TensorId> by_name;
    TensorId next_tid = 0;

    auto intern = [&](const std::string& name) -> TensorId {
        if (name.empty()) {
            const TensorId id = next_tid++;
            Tensor t;
            t.id = id;
            graph.tensors.push_back(std::move(t));
            return id;
        }
        auto it = by_name.find(name);
        if (it != by_name.end()) {
            return it->second;
        }
        Tensor t;
        t.id = next_tid++;
        t.name = name;
        by_name[name] = t.id;
        graph.tensors.push_back(std::move(t));
        return by_name[name];
    };

    auto merge_tensor = [&](Tensor incoming, bool is_input, bool is_output, bool is_const) {
        TensorId id;
        auto it = by_name.find(incoming.name);
        if (it == by_name.end()) {
            incoming.id = next_tid++;
            by_name[incoming.name] = incoming.id;
            id = incoming.id;
            graph.tensors.push_back(std::move(incoming));
        } else {
            id = it->second;
            Tensor* existing = graph.find_tensor(id);
            if (existing) {
                if (existing->dtype == DataType::Unknown) {
                    existing->dtype = incoming.dtype;
                }
                if (existing->shape.dims.empty()) {
                    existing->shape = incoming.shape;
                }
                if (incoming.data) {
                    existing->data = incoming.data;
                    existing->storage_bytes = incoming.storage_bytes;
                    existing->constant = true;
                }
            }
        }
        Tensor* t = graph.find_tensor(id);
        if (t) {
            if (is_input) {
                t->model_input = true;
            }
            if (is_output) {
                t->model_output = true;
            }
            if (is_const) {
                t->constant = true;
            }
        }
        return id;
    };

    ProtoReader r(bytes);
    while (!r.done()) {
        auto f = r.next();
        if (!f) {
            return f.error();
        }
        const auto& field = f.value();
        auto str = [&]() {
            return std::string(reinterpret_cast<const char*>(field.bytes.data()), field.bytes.size());
        };
        switch (field.number) {
            case 1:
                if (field.wire == ProtoWire::Varint) {
                    model.source_format_version = std::to_string(field.varint);
                }
                break;
            case 2:
                if (field.wire == ProtoWire::Length) {
                    model.producer = str();
                }
                break;
            case 3:
                if (field.wire == ProtoWire::Length) {
                    model.framework_version = str();
                }
                break;
            case 4:
                if (field.wire == ProtoWire::Length) {
                    model.domain = str();
                }
                break;
            case 5:
                if (field.wire == ProtoWire::Varint) {
                    model.model_version = zigzag(field.varint);
                }
                break;
            case 6:
                if (field.wire == ProtoWire::Length) {
                    model.doc = str();
                }
                break;
            case 8:
                if (field.wire == ProtoWire::Length) {
                    std::string domain;
                    int64_t version = 0;
                    ProtoReader osr(field.bytes);
                    while (!osr.done()) {
                        auto of = osr.next();
                        if (!of) {
                            return of.error();
                        }
                        if (of.value().number == 1 && of.value().wire == ProtoWire::Length) {
                            domain.assign(reinterpret_cast<const char*>(of.value().bytes.data()),
                                          of.value().bytes.size());
                        } else if (of.value().number == 2 && of.value().wire == ProtoWire::Varint) {
                            version = zigzag(of.value().varint);
                        }
                    }
                    opsets.push_back((domain.empty() ? "ai.onnx" : domain) + ":" +
                                     std::to_string(version));
                }
                break;
            case 14:
                if (field.wire == ProtoWire::Length) {
                    std::string key, value;
                    ProtoReader mr(field.bytes);
                    while (!mr.done()) {
                        auto mf = mr.next();
                        if (!mf) {
                            return mf.error();
                        }
                        if (mf.value().number == 1 && mf.value().wire == ProtoWire::Length) {
                            key.assign(reinterpret_cast<const char*>(mf.value().bytes.data()),
                                       mf.value().bytes.size());
                        } else if (mf.value().number == 2 && mf.value().wire == ProtoWire::Length) {
                            value.assign(reinterpret_cast<const char*>(mf.value().bytes.data()),
                                         mf.value().bytes.size());
                        }
                    }
                    if (!key.empty()) {
                        model.metadata[key] = value;
                    }
                }
                break;
            case 7:
                if (field.wire == ProtoWire::Length) {
                    ProtoReader gr(field.bytes);
                    while (!gr.done()) {
                        auto gf = gr.next();
                        if (!gf) {
                            return gf.error();
                        }
                        const auto& gfield = gf.value();
                        auto gstr = [&]() {
                            return std::string(reinterpret_cast<const char*>(gfield.bytes.data()),
                                               gfield.bytes.size());
                        };
                        switch (gfield.number) {
                            case 2:
                                if (gfield.wire == ProtoWire::Length) {
                                    graph.name = gstr();
                                }
                                break;
                            case 10:
                                if (gfield.wire == ProtoWire::Length) {
                                    graph.doc = gstr();
                                }
                                break;
                            case 11:
                                if (gfield.wire == ProtoWire::Length) {
                                    auto vi = parse_value_info(gfield.bytes, 0);
                                    if (!vi) {
                                        return vi.error();
                                    }
                                    const TensorId id = merge_tensor(std::move(vi.value()), true, false, false);
                                    graph.inputs.push_back(id);
                                }
                                break;
                            case 12:
                                if (gfield.wire == ProtoWire::Length) {
                                    auto vi = parse_value_info(gfield.bytes, 0);
                                    if (!vi) {
                                        return vi.error();
                                    }
                                    const TensorId id = merge_tensor(std::move(vi.value()), false, true, false);
                                    graph.outputs.push_back(id);
                                }
                                break;
                            case 13:
                                if (gfield.wire == ProtoWire::Length) {
                                    auto vi = parse_value_info(gfield.bytes, 0);
                                    if (!vi) {
                                        return vi.error();
                                    }
                                    merge_tensor(std::move(vi.value()), false, false, false);
                                }
                                break;
                            case 5:
                                if (gfield.wire == ProtoWire::Length) {
                                    auto init = parse_tensor_proto(gfield.bytes, 0, options, source);
                                    if (!init) {
                                        return init.error();
                                    }
                                    merge_tensor(std::move(init.value()), false, false, true);
                                }
                                break;
                            case 1:
                                if (gfield.wire == ProtoWire::Length) {
                                    PendingNode pn;
                                    pn.node.id = static_cast<NodeId>(pending.size());
                                    ProtoReader nr(gfield.bytes);
                                    while (!nr.done()) {
                                        auto nf = nr.next();
                                        if (!nf) {
                                            return nf.error();
                                        }
                                        const auto& nfield = nf.value();
                                        auto nstr = [&]() {
                                            return std::string(
                                                reinterpret_cast<const char*>(nfield.bytes.data()),
                                                nfield.bytes.size());
                                        };
                                        switch (nfield.number) {
                                            case 1:
                                                if (nfield.wire == ProtoWire::Length) {
                                                    pn.inputs.push_back(nstr());
                                                }
                                                break;
                                            case 2:
                                                if (nfield.wire == ProtoWire::Length) {
                                                    pn.outputs.push_back(nstr());
                                                }
                                                break;
                                            case 3:
                                                if (nfield.wire == ProtoWire::Length) {
                                                    pn.node.name = nstr();
                                                }
                                                break;
                                            case 4:
                                                if (nfield.wire == ProtoWire::Length) {
                                                    pn.node.op_type = nstr();
                                                }
                                                break;
                                            case 7:
                                                if (nfield.wire == ProtoWire::Length) {
                                                    pn.node.domain = nstr();
                                                }
                                                break;
                                            case 6:
                                                if (nfield.wire == ProtoWire::Length) {
                                                    pn.node.doc = nstr();
                                                }
                                                break;
                                            case 5:
                                                if (nfield.wire == ProtoWire::Length) {
                                                    std::string aname;
                                                    Attribute attr;
                                                    ProtoReader ar(nfield.bytes);
                                                    while (!ar.done()) {
                                                        auto af = ar.next();
                                                        if (!af) {
                                                            return af.error();
                                                        }
                                                        const auto& afield = af.value();
                                                        if (afield.number == 1 &&
                                                            afield.wire == ProtoWire::Length) {
                                                            aname.assign(reinterpret_cast<const char*>(
                                                                             afield.bytes.data()),
                                                                         afield.bytes.size());
                                                        } else if (afield.number == 3 &&
                                                                   afield.wire == ProtoWire::Varint) {
                                                            attr = Attribute::from_int(
                                                                zigzag(afield.varint));
                                                        } else if (afield.number == 2 &&
                                                                   afield.wire == ProtoWire::Fixed32) {
                                                            float fv = 0;
                                                            std::memcpy(&fv, &afield.fixed32, 4);
                                                            attr = Attribute::from_float(
                                                                static_cast<double>(fv));
                                                        } else if (afield.number == 4 &&
                                                                   afield.wire == ProtoWire::Length) {
                                                            attr = Attribute::from_string(std::string(
                                                                reinterpret_cast<const char*>(
                                                                    afield.bytes.data()),
                                                                afield.bytes.size()));
                                                        } else if (afield.number == 8 &&
                                                                   afield.wire == ProtoWire::Varint) {
                                                            if (attr.kind != Attribute::Kind::Ints) {
                                                                attr = Attribute::from_ints({});
                                                            }
                                                            attr.ints.push_back(zigzag(afield.varint));
                                                        } else if (afield.number == 8 &&
                                                                   afield.wire == ProtoWire::Length) {
                                                            auto ints = packed_int64s(afield.bytes);
                                                            if (!ints) {
                                                                return ints.error();
                                                            }
                                                            attr = Attribute::from_ints(
                                                                std::move(ints.value()));
                                                        } else if (afield.number == 7 &&
                                                                   afield.wire == ProtoWire::Length) {
                                                            attr = Attribute::from_floats({});
                                                            for (std::size_t i = 0;
                                                                 i + 4 <= afield.bytes.size();
                                                                 i += 4) {
                                                                float fv = 0;
                                                                std::memcpy(&fv, afield.bytes.data() + i,
                                                                            4);
                                                                attr.floats.push_back(
                                                                    static_cast<double>(fv));
                                                            }
                                                        }
                                                    }
                                                    if (!aname.empty()) {
                                                        pn.node.attributes.emplace(aname, attr);
                                                    }
                                                }
                                                break;
                                            default:
                                                break;
                                        }
                                    }
                                    pending.push_back(std::move(pn));
                                }
                                break;
                            default:
                                break;
                        }
                    }
                }
                break;
            default:
                break;
        }
    }

    if (model.source_format_version.empty()) {
        return error(ErrorCode::ParseError, "not a valid ONNX ModelProto (missing ir_version)");
    }

    for (auto& pn : pending) {
        for (const auto& in : pn.inputs) {
            pn.node.inputs.push_back(intern(in));
        }
        for (const auto& out : pn.outputs) {
            pn.node.outputs.push_back(intern(out));
        }
        pn.node.canonical = canonicalize_op("onnx", pn.node.op_type);
        graph.nodes.push_back(std::move(pn.node));
    }

    for (std::size_t i = 0; i < opsets.size(); ++i) {
        model.metadata["opset." + std::to_string(i)] = opsets[i];
    }
    graph.rebuild_use_lists();
    for (auto& t : graph.tensors) {
        if (t.storage_bytes == 0) {
            if (auto b = tensor_storage_bytes(t)) {
                t.storage_bytes = b.value();
            }
        }
    }
    model.graphs.push_back(std::move(graph));
    return model;
}

}  // namespace onnx
}  // namespace nn
