#include "formats/detect.h"

#include "nn/mmap.h"

#include <algorithm>
#include <cctype>
#include <cstring>

namespace nn {

std::string extension_lower(const std::filesystem::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return e;
}

bool has_magic(std::span<const uint8_t> data, std::span<const uint8_t> magic) {
    if (data.size() < magic.size()) {
        return false;
    }
    return std::equal(magic.begin(), magic.end(), data.begin());
}

bool file_has_magic(const std::filesystem::path& path, std::span<const uint8_t> magic) {
    const auto mapped = MappedFile::open(path);
    if (!mapped) {
        return false;
    }
    return has_magic(mapped.value().span(), magic);
}

bool looks_like_zip(std::span<const uint8_t> data) {
    const uint8_t mag[] = {'P', 'K', 0x03, 0x04};
    return has_magic(data, mag);
}

bool looks_like_hdf5(std::span<const uint8_t> data) {
    const uint8_t mag[] = {0x89, 'H', 'D', 'F', 0x0d, 0x0a, 0x1a, 0x0a};
    return has_magic(data, mag);
}

bool looks_like_pickle(std::span<const uint8_t> data) {
    return data.size() >= 2 && data[0] == 0x80 && data[1] <= 0x05;
}

Result<std::vector<std::string>> zip_list_names(std::span<const uint8_t> data) {
    std::vector<std::string> names;
    if (!looks_like_zip(data)) {
        return error(ErrorCode::InvalidFormat, "not a ZIP archive");
    }
    std::size_t pos = 0;
    int files = 0;
    while (pos + 30 <= data.size() && files < 4096) {
        if (data[pos] == 'P' && data[pos + 1] == 'K' && data[pos + 2] == 0x03 &&
            data[pos + 3] == 0x04) {
            uint16_t name_len = 0;
            std::memcpy(&name_len, data.data() + pos + 26, 2);
            if (pos + 30 + name_len > data.size()) {
                break;
            }
            names.emplace_back(reinterpret_cast<const char*>(data.data() + pos + 30), name_len);
            pos += 30u + name_len;
            ++files;
            continue;
        }
        ++pos;
    }
    return names;
}

}  // namespace nn
