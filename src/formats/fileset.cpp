#include "nn/format.h"

namespace nn {

Result<FileSet> FileSet::from_path(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return error(ErrorCode::FileNotFound, "file not found: " + path.string());
    }
    FileSet fs;
    if (std::filesystem::is_directory(path, ec)) {
        fs.is_directory = true;
        fs.paths.push_back(path);
        for (const auto& entry : std::filesystem::directory_iterator(path, ec)) {
            if (entry.is_regular_file()) {
                fs.paths.push_back(entry.path());
            }
        }
        return fs;
    }
    fs.paths.push_back(path);
    return fs;
}

const std::filesystem::path& FileSet::primary() const {
    static const std::filesystem::path empty;
    return paths.empty() ? empty : paths.front();
}

std::filesystem::path FileSet::sibling(std::string_view replacement_ext) const {
    auto p = primary();
    p.replace_extension(std::string(replacement_ext));
    return p;
}

}  // namespace nn
