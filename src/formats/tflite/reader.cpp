#include "formats/readers.h"

#include "formats/detect.h"
#include "nn/mmap.h"
#include "nn/operator.h"
#include "util/flatbuffer.h"

namespace nn {
namespace {

DataType tflite_dtype(int32_t t) {
    switch (t) {
        case 0:
            return DataType::Float32;
        case 1:
            return DataType::Float16;
        case 2:
            return DataType::Int32;
        case 3:
            return DataType::UInt8;
        case 4:
            return DataType::Int64;
        case 5:
            return DataType::String;
        case 6:
            return DataType::Bool;
        case 7:
            return DataType::Int16;
        case 9:
            return DataType::Int8;
        case 10:
            return DataType::Float64;
        case 12:
            return DataType::UInt64;
        case 15:
            return DataType::UInt32;
        case 16:
            return DataType::UInt16;
        case 17:
            return DataType::Int4;
        case 18:
            return DataType::BFloat16;
        default:
            return DataType::Unknown;
    }
}

const char* tflite_builtin_name(int32_t code) {
    switch (code) {
        case 0:
            return "ADD";
        case 1:
            return "AVERAGE_POOL_2D";
        case 2:
            return "CONCATENATION";
        case 3:
            return "CONV_2D";
        case 4:
            return "DEPTHWISE_CONV_2D";
        case 5:
            return "DEPTH_TO_SPACE";
        case 6:
            return "DEQUANTIZE";
        case 7:
            return "EMBEDDING_LOOKUP";
        case 8:
            return "FLOOR";
        case 9:
            return "FULLY_CONNECTED";
        case 10:
            return "HASHTABLE_LOOKUP";
        case 11:
            return "L2_NORMALIZATION";
        case 12:
            return "L2_POOL_2D";
        case 13:
            return "LOCAL_RESPONSE_NORMALIZATION";
        case 14:
            return "LOGISTIC";
        case 15:
            return "LSH_PROJECTION";
        case 16:
            return "LSTM";
        case 17:
            return "MAX_POOL_2D";
        case 18:
            return "MUL";
        case 19:
            return "RELU";
        case 20:
            return "RELU_N1_TO_1";
        case 21:
            return "RELU6";
        case 22:
            return "RESHAPE";
        case 23:
            return "RESIZE_BILINEAR";
        case 24:
            return "RNN";
        case 25:
            return "SOFTMAX";
        case 26:
            return "SPACE_TO_DEPTH";
        case 27:
            return "SVDF";
        case 28:
            return "TANH";
        case 32:
            return "CONCAT_EMBEDDINGS";
        case 33:
            return "SKIP_GRAM";
        case 34:
            return "CALL";
        case 35:
            return "CUSTOM";
        case 39:
            return "GATHER";
        case 40:
            return "TRANSPOSE";
        case 47:
            return "MEAN";
        case 48:
            return "SUB";
        case 49:
            return "DIV";
        case 50:
            return "SQUEEZE";
        case 57:
            return "QUANTIZE";
        case 59:
            return "PAD";
        case 80:
            return "TRANSPOSE_CONV";
        default:
            return nullptr;
    }
}

class TfliteReader final : public ModelReader {
public:
    std::string name() const override { return "tflite"; }
    std::string display_name() const override { return "TFLite/LiteRT"; }
    std::vector<std::string> extensions() const override { return {".tflite"}; }

    bool probe(const FileSet& files) const override {
        if (files.paths.empty() || files.is_directory) {
            return false;
        }
        auto mapped = MappedFile::open(files.primary());
        if (!mapped || mapped.value().size() < 12) {
            return false;
        }
        const auto sp = mapped.value().span();
        // File identifier "TFL3" at offset 4 is common; also accept root-offset sanity.
        if (sp.size() >= 8 && sp[4] == 'T' && sp[5] == 'F' && sp[6] == 'L' && sp[7] == '3') {
            return true;
        }
        if (extension_lower(files.primary()) != ".tflite") {
            return false;
        }
        FlatBuffer fb(sp);
        auto root = fb.root_offset();
        return static_cast<bool>(root);
    }

    ModelCapabilities capabilities() const override {
        ModelCapabilities c;
        c.read = true;
        c.graph = true;
        c.weights = true;
#if defined(NN_HAS_LITERT)
        c.execute = true;
        c.execute_compiled = true;
        c.notes = "graph and weights; execution via LiteRT";
#else
        c.execute = false;
        c.execute_compiled = false;
        c.notes = "FlatBuffer graph inspection; execution requires LiteRT/TFLite runtime";
#endif
        c.convert = false;
        return c;
    }

    Result<ModelIR> load(const FileSet& files, const LoadOptions& options) override {
        (void)options;
        auto mapped = MappedFile::open(files.primary());
        if (!mapped) {
            return mapped.error();
        }
        FlatBuffer fb(mapped.value().span());
        auto root_off = fb.root_offset();
        if (!root_off) {
            return root_off.error();
        }
        const uint64_t model_table = root_off.value();

        ModelIR model;
        model.source_format = "tflite";
        auto ver = fb.table_u32(model_table, 0, 0);
        if (ver) {
            model.source_format_version = std::to_string(ver.value());
        }
        auto desc = fb.table_string(model_table, 3);
        if (desc) {
            model.doc = desc.value();
        }

        std::vector<std::string> opcodes;
        auto oc_vec = fb.table_offset(model_table, 1);
        if (oc_vec && oc_vec.value() != 0) {
            auto n = fb.vec_len(oc_vec.value());
            if (!n) {
                return n.error();
            }
            opcodes.resize(n.value());
            for (uint32_t i = 0; i < n.value(); ++i) {
                auto table = fb.vec_uoffset(oc_vec.value(), i);
                if (!table) {
                    return table.error();
                }
                auto custom = fb.table_string(table.value(), 1);
                auto builtin = fb.table_i32(table.value(), 3, 0);
                auto deprecated = fb.table_i16(table.value(), 0, 0);
                int32_t code = builtin ? builtin.value() : 0;
                if (code == 0 && deprecated) {
                    code = deprecated.value();
                }
                if (custom && !custom.value().empty()) {
                    opcodes[i] = custom.value();
                } else if (const char* nm = tflite_builtin_name(code)) {
                    opcodes[i] = nm;
                } else {
                    opcodes[i] = "opcode_" + std::to_string(code);
                }
            }
        }

        auto sg_vec = fb.table_offset(model_table, 2);
        if (!sg_vec || sg_vec.value() == 0) {
            return error(ErrorCode::ParseError, "TFLite model has no subgraphs");
        }
        auto nsg = fb.vec_len(sg_vec.value());
        if (!nsg) {
            return nsg.error();
        }
        for (uint32_t s = 0; s < nsg.value(); ++s) {
            auto sg = fb.vec_uoffset(sg_vec.value(), s);
            if (!sg) {
                return sg.error();
            }
            Graph graph;
            graph.id = s;
            auto gname = fb.table_string(sg.value(), 4);
            if (gname) {
                graph.name = gname.value();
            }
            if (graph.name.empty()) {
                graph.name = "subgraph" + std::to_string(s);
            }

            auto tensors_off = fb.table_offset(sg.value(), 0);
            if (tensors_off && tensors_off.value() != 0) {
                auto nt = fb.vec_len(tensors_off.value());
                if (!nt) {
                    return nt.error();
                }
                for (uint32_t ti = 0; ti < nt.value(); ++ti) {
                    auto tt = fb.vec_uoffset(tensors_off.value(), ti);
                    if (!tt) {
                        return tt.error();
                    }
                    Tensor t;
                    t.id = ti;
                    auto name = fb.table_string(tt.value(), 3);
                    if (name) {
                        t.name = name.value();
                    }
                    if (t.name.empty()) {
                        t.name = "t" + std::to_string(ti);
                    }
                    auto ty = fb.table_i32(tt.value(), 1, 0);
                    if (ty) {
                        t.dtype = tflite_dtype(ty.value());
                    }
                    auto shape_off = fb.table_offset(tt.value(), 0);
                    if (shape_off && shape_off.value() != 0) {
                        auto ns = fb.vec_len(shape_off.value());
                        if (!ns) {
                            return ns.error();
                        }
                        std::vector<int64_t> dims;
                        dims.reserve(ns.value());
                        for (uint32_t di = 0; di < ns.value(); ++di) {
                            auto eoff = fb.vec_elem_offset(shape_off.value(), di, 4);
                            if (!eoff) {
                                return eoff.error();
                            }
                            auto v = fb.i32(eoff.value());
                            if (!v) {
                                return v.error();
                            }
                            dims.push_back(v.value());
                        }
                        t.shape = shape_from_ints(dims);
                    }
                    auto qoff = fb.table_offset(tt.value(), 4);
                    if (qoff && qoff.value() != 0) {
                        t.quantization.quantized = true;
                        auto scale_off = fb.table_offset(qoff.value(), 1);
                        if (scale_off && scale_off.value() != 0) {
                            auto ns = fb.vec_len(scale_off.value());
                            if (ns) {
                                for (uint32_t qi = 0; qi < ns.value(); ++qi) {
                                    auto e = fb.vec_elem_offset(scale_off.value(), qi, 4);
                                    if (e) {
                                        auto fv = fb.f32(e.value());
                                        if (fv) {
                                            t.quantization.scales.push_back(
                                                static_cast<double>(fv.value()));
                                        }
                                    }
                                }
                            }
                        }
                        t.quantization.per_channel = t.quantization.scales.size() > 1;
                        t.quantization.bits = datatype_bits(t.dtype);
                    }
                    if (auto b = tensor_storage_bytes(t)) {
                        t.storage_bytes = b.value();
                    }
                    graph.tensors.push_back(std::move(t));
                }
            }

            auto mark_io = [&](uint16_t field, bool input) -> Status {
                auto off = fb.table_offset(sg.value(), field);
                if (!off || off.value() == 0) {
                    return Status::ok();
                }
                auto n = fb.vec_len(off.value());
                if (!n) {
                    return Status::err(n.error());
                }
                for (uint32_t i = 0; i < n.value(); ++i) {
                    auto e = fb.vec_elem_offset(off.value(), i, 4);
                    if (!e) {
                        return Status::err(e.error());
                    }
                    auto idx = fb.i32(e.value());
                    if (!idx) {
                        return Status::err(idx.error());
                    }
                    if (idx.value() < 0) {
                        continue;
                    }
                    const auto tid = static_cast<TensorId>(idx.value());
                    if (auto* t = graph.find_tensor(tid)) {
                        if (input) {
                            t->model_input = true;
                            graph.inputs.push_back(tid);
                        } else {
                            t->model_output = true;
                            graph.outputs.push_back(tid);
                        }
                    }
                }
                return Status::ok();
            };
            if (auto st = mark_io(1, true); !st) {
                return st.error();
            }
            if (auto st = mark_io(2, false); !st) {
                return st.error();
            }

            auto ops_off = fb.table_offset(sg.value(), 3);
            if (ops_off && ops_off.value() != 0) {
                auto no = fb.vec_len(ops_off.value());
                if (!no) {
                    return no.error();
                }
                for (uint32_t oi = 0; oi < no.value(); ++oi) {
                    auto ot = fb.vec_uoffset(ops_off.value(), oi);
                    if (!ot) {
                        return ot.error();
                    }
                    Node node;
                    node.id = oi;
                    auto opcode_index = fb.table_u32(ot.value(), 0, 0);
                    const uint32_t oci = opcode_index ? opcode_index.value() : 0;
                    node.op_type = oci < opcodes.size() ? opcodes[oci] : "unknown";
                    node.name = node.op_type + "_" + std::to_string(oi);
                    node.canonical = canonicalize_op("tflite", node.op_type);
                    auto add_edges = [&](uint16_t field, std::vector<TensorId>& dest) -> Status {
                        auto voff = fb.table_offset(ot.value(), field);
                        if (!voff || voff.value() == 0) {
                            return Status::ok();
                        }
                        auto n = fb.vec_len(voff.value());
                        if (!n) {
                            return Status::err(n.error());
                        }
                        for (uint32_t i = 0; i < n.value(); ++i) {
                            auto e = fb.vec_elem_offset(voff.value(), i, 4);
                            if (!e) {
                                return Status::err(e.error());
                            }
                            auto idx = fb.i32(e.value());
                            if (!idx) {
                                return Status::err(idx.error());
                            }
                            if (idx.value() < 0) {
                                continue;
                            }
                            dest.push_back(static_cast<TensorId>(idx.value()));
                        }
                        return Status::ok();
                    };
                    if (auto st = add_edges(1, node.inputs); !st) {
                        return st.error();
                    }
                    if (auto st = add_edges(2, node.outputs); !st) {
                        return st.error();
                    }
                    graph.nodes.push_back(std::move(node));
                }
            }
            graph.rebuild_use_lists();
            model.graphs.push_back(std::move(graph));
        }
        return model;
    }
};

}  // namespace

std::unique_ptr<ModelReader> make_tflite_reader() { return std::make_unique<TfliteReader>(); }

}  // namespace nn
