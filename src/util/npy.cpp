#include "util/npy.h"

#include "nn/mmap.h"
#include "util/binary.h"
#include "util/zip.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string_view>

namespace nn {
namespace {

std::string npy_descr(DataType t) {
    switch (t) {
        case DataType::Float32:
            return "<f4";
        case DataType::Float64:
            return "<f8";
        case DataType::Int8:
            return "|i1";
        case DataType::Int16:
            return "<i2";
        case DataType::Int32:
            return "<i4";
        case DataType::Int64:
            return "<i8";
        case DataType::UInt8:
            return "|u1";
        case DataType::UInt16:
            return "<u2";
        case DataType::UInt32:
            return "<u4";
        case DataType::UInt64:
            return "<u8";
        default:
            return "";
    }
}

DataType dtype_from_descr(std::string_view d) {
    if (d.find("f4") != std::string_view::npos) {
        return DataType::Float32;
    }
    if (d.find("f8") != std::string_view::npos) {
        return DataType::Float64;
    }
    if (d.find("i8") != std::string_view::npos) {
        return DataType::Int64;
    }
    if (d.find("i4") != std::string_view::npos) {
        return DataType::Int32;
    }
    if (d.find("i2") != std::string_view::npos) {
        return DataType::Int16;
    }
    if (d.find("i1") != std::string_view::npos) {
        return DataType::Int8;
    }
    if (d.find("u1") != std::string_view::npos) {
        return DataType::UInt8;
    }
    return DataType::Unknown;
}

}  // namespace

Result<RuntimeTensor> load_npy_bytes(std::span<const uint8_t> data, std::string_view name) {
    BinaryReader r(data, std::string(name));
    auto mag = r.bytes(6);
    if (!mag) {
        return mag.error();
    }
    const uint8_t expect[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (!std::equal(std::begin(expect), std::end(expect), mag.value().begin())) {
        return error(ErrorCode::InvalidFormat, "not an NPY file");
    }
    auto major = r.u8();
    auto minor = r.u8();
    if (!major || !minor) {
        return major ? minor.error() : major.error();
    }
    uint32_t hlen = 0;
    if (major.value() == 1) {
        auto n = r.u16le();
        if (!n) {
            return n.error();
        }
        hlen = n.value();
    } else {
        auto n = r.u32le();
        if (!n) {
            return n.error();
        }
        hlen = n.value();
    }
    auto header = r.string(hlen);
    if (!header) {
        return header.error();
    }
    RuntimeTensor t;
    t.name = std::string(name);
    const auto& h = header.value();
    auto descr_at = h.find("'descr'");
    if (descr_at == std::string::npos) {
        descr_at = h.find("\"descr\"");
    }
    if (descr_at != std::string::npos) {
        auto colon = h.find(':', descr_at);
        if (colon != std::string::npos) {
            auto q1 = h.find('\'', colon);
            if (q1 == std::string::npos) {
                q1 = h.find('"', colon);
            }
            if (q1 != std::string::npos) {
                auto q2 = h.find(h[q1], q1 + 1);
                if (q2 != std::string::npos) {
                    t.dtype = dtype_from_descr(h.substr(q1 + 1, q2 - q1 - 1));
                }
            }
        }
    }
    auto shape_at = h.find("shape");
    if (shape_at != std::string::npos) {
        auto p1 = h.find('(', shape_at);
        auto p2 = h.find(')', p1);
        if (p1 != std::string::npos && p2 != std::string::npos) {
            std::vector<int64_t> dims;
            std::stringstream ss(h.substr(p1 + 1, p2 - p1 - 1));
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (tok.find_first_of("0123456789") == std::string::npos) {
                    continue;
                }
                dims.push_back(std::strtoll(tok.c_str(), nullptr, 10));
            }
            t.shape = shape_from_ints(dims);
        }
    }
    auto rest = r.remaining_span();
    t.bytes.assign(rest.begin(), rest.end());
    return t;
}

Result<RuntimeTensor> load_npy(const std::filesystem::path& path) {
    auto mapped = MappedFile::open(path);
    if (!mapped) {
        return mapped.error();
    }
    return load_npy_bytes(mapped.value().span(), path.stem().string());
}

Result<std::vector<uint8_t>> encode_npy(const RuntimeTensor& tensor) {
    const std::string descr = npy_descr(tensor.dtype);
    if (descr.empty()) {
        return error(ErrorCode::InvalidArgument, "unsupported npy dtype");
    }
    std::ostringstream shape;
    shape << "(";
    for (std::size_t i = 0; i < tensor.shape.dims.size(); ++i) {
        if (i) {
            shape << ", ";
        }
        shape << tensor.shape.dims[i].value.value_or(0);
    }
    if (tensor.shape.dims.size() == 1) {
        shape << ",";
    }
    shape << ")";
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " +
                         shape.str() + ", }";
    while ((10 + header.size()) % 16 != 0) {
        header.push_back(' ');
    }
    header.back() = '\n';
    std::vector<uint8_t> out;
    const uint8_t mag[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    out.insert(out.end(), std::begin(mag), std::end(mag));
    const uint16_t hlen = static_cast<uint16_t>(header.size());
    out.push_back(static_cast<uint8_t>(hlen));
    out.push_back(static_cast<uint8_t>(hlen >> 8));
    out.insert(out.end(), header.begin(), header.end());
    out.insert(out.end(), tensor.bytes.begin(), tensor.bytes.end());
    return out;
}

Status save_npy(const std::filesystem::path& path, const RuntimeTensor& tensor) {
    auto bytes = encode_npy(tensor);
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

namespace {

std::string npz_key(std::string name) {
    auto slash = name.find_last_of("/\\");
    if (slash != std::string::npos) {
        name = name.substr(slash + 1);
    }
    constexpr std::string_view ext = ".npy";
    if (name.size() >= ext.size() &&
        name.compare(name.size() - ext.size(), ext.size(), ext.data()) == 0) {
        name.resize(name.size() - ext.size());
    }
    return name;
}

}  // namespace

Result<std::map<std::string, RuntimeTensor>> load_npz(const std::filesystem::path& path) {
    auto mapped = MappedFile::open(path);
    if (!mapped) {
        return mapped.error();
    }
    auto members = zip_read_stored(mapped.value().span());
    if (!members) {
        return members.error();
    }
    std::map<std::string, RuntimeTensor> out;
    for (const auto& m : members.value()) {
        const std::string key = npz_key(m.name);
        if (key.empty() || key == "." || key[0] == '.') {
            continue;
        }
        auto t = load_npy_bytes(m.bytes, key);
        if (!t) {
            return t.error();
        }
        out[key] = std::move(t.value());
    }
    if (out.empty()) {
        return error(ErrorCode::InvalidFormat, "npz contains no arrays: " + path.string());
    }
    return out;
}

Status save_npz(const std::filesystem::path& path,
                const std::map<std::string, RuntimeTensor>& tensors) {
    if (tensors.empty()) {
        return Status::err(error(ErrorCode::InvalidArgument, "npz has no arrays"));
    }
    std::vector<ZipMember> members;
    members.reserve(tensors.size());
    for (const auto& [name, t] : tensors) {
        auto bytes = encode_npy(t);
        if (!bytes) {
            return Status::err(bytes.error());
        }
        ZipMember m;
        m.name = npz_key(name) + ".npy";
        m.bytes = std::move(bytes.value());
        members.push_back(std::move(m));
    }
    const auto zip = zip_write_stored(members);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return Status::err(error(ErrorCode::FileError, "cannot write " + path.string()));
    }
    out.write(reinterpret_cast<const char*>(zip.data()), static_cast<std::streamsize>(zip.size()));
    return Status::ok();
}

}  // namespace nn
