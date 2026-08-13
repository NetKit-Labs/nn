#include "nn/result.h"

#include <filesystem>
#include <fstream>
#include <vector>

namespace nn {

Result<std::vector<uint8_t>> read_file_limited(const std::filesystem::path& path,
                                               uint64_t max_bytes) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return error(ErrorCode::FileNotFound, "file not found: " + path.string());
    }
    const uint64_t sz = static_cast<uint64_t>(std::filesystem::file_size(path, ec));
    if (ec) {
        return error(ErrorCode::FileError, "cannot stat " + path.string());
    }
    if (sz > max_bytes) {
        return error(ErrorCode::LimitExceeded,
                     "file exceeds allocation limit: " + path.string());
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return error(ErrorCode::FileError, "cannot open " + path.string());
    }
    std::vector<uint8_t> buf(static_cast<std::size_t>(sz));
    if (sz > 0) {
        in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
        if (static_cast<uint64_t>(in.gcount()) != sz) {
            return error(ErrorCode::FileError, "short read: " + path.string());
        }
    }
    return buf;
}

std::string read_text_file(const std::filesystem::path& path, std::error_code& ec) {
    std::ifstream in(path);
    if (!in) {
        ec = std::make_error_code(std::errc::io_error);
        return {};
    }
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return s;
}

}  // namespace nn
