#ifndef NN_MMAP_H
#define NN_MMAP_H

#include "nn/result.h"

#include <cstdint>
#include <filesystem>
#include <span>

namespace nn {

class MappedFile {
public:
    MappedFile() = default;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    static Result<MappedFile> open(const std::filesystem::path& path);

    const uint8_t* data() const { return data_; }
    uint64_t size() const { return size_; }
    std::span<const uint8_t> span() const {
        return {data_, static_cast<std::size_t>(size_)};
    }
    std::span<const uint8_t> slice(uint64_t offset, uint64_t length) const;
    const std::filesystem::path& path() const { return path_; }

private:
    void close() noexcept;
    const uint8_t* data_ = nullptr;
    uint64_t size_ = 0;
    std::filesystem::path path_;
#ifdef _WIN32
    void* file_handle_ = nullptr;
    void* map_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}  // namespace nn

#endif
