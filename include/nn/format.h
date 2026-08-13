#ifndef NN_FORMAT_H
#define NN_FORMAT_H

#include "nn/model.h"
#include "nn/result.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace nn {

struct FileSet {
    std::vector<std::filesystem::path> paths;
    bool is_directory = false;

    static Result<FileSet> from_path(const std::filesystem::path& path);
    const std::filesystem::path& primary() const;
    std::filesystem::path sibling(std::string_view replacement_ext) const;
};

struct LoadOptions {
    bool load_weights = true;
    bool follow_external_data = true;
    uint64_t max_allocation_bytes = uint64_t{1} << 40;  // 1 TiB default cap
    bool allow_unsafe_deserialize = false;
};

struct ModelCapabilities {
    bool read = false;
    bool graph = false;
    bool weights = false;
    bool execute = false;
    bool convert = false;
    bool execute_compiled = false;  // compiled in this binary
    bool convert_compiled = false;
    std::string notes;
};

struct FormatInfo {
    std::string name;
    std::string display_name;
    std::vector<std::string> extensions;
    ModelCapabilities capabilities;
};

class ModelReader {
public:
    virtual ~ModelReader() = default;
    virtual std::string name() const = 0;
    virtual std::string display_name() const { return name(); }
    virtual std::vector<std::string> extensions() const = 0;
    virtual bool probe(const FileSet& files) const = 0;
    virtual ModelCapabilities capabilities() const = 0;
    virtual Result<ModelIR> load(const FileSet& files, const LoadOptions& options) = 0;
};

class FormatRegistry {
public:
    void register_reader(std::unique_ptr<ModelReader> reader);
    ModelReader* detect(const FileSet& files) const;
    Result<ModelReader*> require(const FileSet& files) const;
    std::vector<FormatInfo> formats() const;
    ModelReader* find_by_name(std::string_view name) const;
    const std::vector<std::unique_ptr<ModelReader>>& readers() const { return readers_; }

private:
    std::vector<std::unique_ptr<ModelReader>> readers_;
};

FormatRegistry& default_format_registry();
void register_builtin_formats(FormatRegistry& registry);

Result<ModelIR> load_model(const std::filesystem::path& path,
                           const LoadOptions& options = {});

}  // namespace nn

#endif
