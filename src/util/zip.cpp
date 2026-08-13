#include "util/zip.h"

#include <cstring>

namespace nn {
namespace {

uint16_t u16le(std::span<const uint8_t> d, std::size_t off) {
    return static_cast<uint16_t>(d[off] | (static_cast<uint16_t>(d[off + 1]) << 8));
}

uint32_t u32le(std::span<const uint8_t> d, std::size_t off) {
    return static_cast<uint32_t>(d[off]) | (static_cast<uint32_t>(d[off + 1]) << 8) |
           (static_cast<uint32_t>(d[off + 2]) << 16) | (static_cast<uint32_t>(d[off + 3]) << 24);
}

void put_u16le(std::vector<uint8_t>& o, uint16_t v) {
    o.push_back(static_cast<uint8_t>(v));
    o.push_back(static_cast<uint8_t>(v >> 8));
}

void put_u32le(std::vector<uint8_t>& o, uint32_t v) {
    o.push_back(static_cast<uint8_t>(v));
    o.push_back(static_cast<uint8_t>(v >> 8));
    o.push_back(static_cast<uint8_t>(v >> 16));
    o.push_back(static_cast<uint8_t>(v >> 24));
}

constexpr uint32_t kLocal = 0x04034b50u;
constexpr uint32_t kCentral = 0x02014b50u;
constexpr uint32_t kEocd = 0x06054b50u;

}  // namespace

uint32_t crc32_ieee(std::span<const uint8_t> data) {
    uint32_t c = 0xffffffffu;
    for (uint8_t b : data) {
        c ^= b;
        for (int i = 0; i < 8; ++i) {
            c = (c & 1u) ? (c >> 1) ^ 0xedb88320u : (c >> 1);
        }
    }
    return ~c;
}

Result<std::vector<ZipMember>> zip_read_stored(std::span<const uint8_t> data) {
    if (data.size() < 4 || u32le(data, 0) != kLocal) {
        return error(ErrorCode::InvalidFormat, "not a ZIP archive");
    }
    std::vector<ZipMember> out;
    std::size_t pos = 0;
    int files = 0;
    while (pos + 30 <= data.size() && files < 4096) {
        const uint32_t sig = u32le(data, pos);
        if (sig == kCentral || sig == kEocd) {
            break;
        }
        if (sig != kLocal) {
            return error(ErrorCode::InvalidFormat, "ZIP local header is malformed");
        }
        const uint16_t flags = u16le(data, pos + 6);
        const uint16_t method = u16le(data, pos + 8);
        const uint32_t comp = u32le(data, pos + 18);
        const uint32_t uncomp = u32le(data, pos + 22);
        const uint16_t name_len = u16le(data, pos + 26);
        const uint16_t extra_len = u16le(data, pos + 28);
        if (pos + 30 + name_len + extra_len > data.size()) {
            return error(ErrorCode::InvalidFormat, "ZIP name/extra overruns buffer");
        }
        std::string name(reinterpret_cast<const char*>(data.data() + pos + 30), name_len);
        const std::size_t data_off = pos + 30 + name_len + extra_len;
        if (flags & 0x8) {
            return error(ErrorCode::UnsupportedFormat,
                         "ZIP data descriptors are not supported: " + name);
        }
        if (data_off + comp > data.size()) {
            return error(ErrorCode::InvalidFormat, "ZIP payload overruns buffer: " + name);
        }
        pos = data_off + comp;
        ++files;
        if (name.empty() || name.back() == '/') {
            continue;
        }
        if (method != 0) {
            return error(ErrorCode::UnsupportedFormat,
                         "compressed ZIP member '" + name +
                             "' (deflate); use uncompressed npz (numpy.savez) or extract .npy files");
        }
        if (comp != uncomp) {
            return error(ErrorCode::InvalidFormat, "stored ZIP size mismatch: " + name);
        }
        ZipMember m;
        m.name = std::move(name);
        m.bytes.assign(data.begin() + static_cast<std::ptrdiff_t>(data_off),
                       data.begin() + static_cast<std::ptrdiff_t>(data_off + comp));
        out.push_back(std::move(m));
    }
    return out;
}

std::vector<uint8_t> zip_write_stored(const std::vector<ZipMember>& members) {
    std::vector<uint8_t> out;
    struct Meta {
        std::string name;
        uint32_t crc = 0;
        uint32_t size = 0;
        uint32_t local_off = 0;
    };
    std::vector<Meta> meta;
    meta.reserve(members.size());
    for (const auto& m : members) {
        Meta md;
        md.name = m.name;
        md.crc = crc32_ieee(m.bytes);
        md.size = static_cast<uint32_t>(m.bytes.size());
        md.local_off = static_cast<uint32_t>(out.size());
        put_u32le(out, kLocal);
        put_u16le(out, 20);
        put_u16le(out, 0);
        put_u16le(out, 0);
        put_u16le(out, 0);
        put_u16le(out, 0);
        put_u32le(out, md.crc);
        put_u32le(out, md.size);
        put_u32le(out, md.size);
        put_u16le(out, static_cast<uint16_t>(md.name.size()));
        put_u16le(out, 0);
        out.insert(out.end(), md.name.begin(), md.name.end());
        out.insert(out.end(), m.bytes.begin(), m.bytes.end());
        meta.push_back(std::move(md));
    }
    const uint32_t cd_off = static_cast<uint32_t>(out.size());
    for (const auto& md : meta) {
        put_u32le(out, kCentral);
        put_u16le(out, 20);
        put_u16le(out, 20);
        put_u16le(out, 0);
        put_u16le(out, 0);
        put_u16le(out, 0);
        put_u16le(out, 0);
        put_u32le(out, md.crc);
        put_u32le(out, md.size);
        put_u32le(out, md.size);
        put_u16le(out, static_cast<uint16_t>(md.name.size()));
        put_u16le(out, 0);
        put_u16le(out, 0);
        put_u16le(out, 0);
        put_u16le(out, 0);
        put_u32le(out, 0);
        put_u32le(out, md.local_off);
        out.insert(out.end(), md.name.begin(), md.name.end());
    }
    const uint32_t cd_size = static_cast<uint32_t>(out.size()) - cd_off;
    put_u32le(out, kEocd);
    put_u16le(out, 0);
    put_u16le(out, 0);
    put_u16le(out, static_cast<uint16_t>(meta.size()));
    put_u16le(out, static_cast<uint16_t>(meta.size()));
    put_u32le(out, cd_size);
    put_u32le(out, cd_off);
    put_u16le(out, 0);
    return out;
}

}  // namespace nn
