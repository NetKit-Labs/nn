#include "nn/mmap.h"

#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nn {

MappedFile::MappedFile(MappedFile&& other) noexcept { *this = std::move(other); }

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    close();
    data_ = other.data_;
    size_ = other.size_;
    path_ = std::move(other.path_);
    other.data_ = nullptr;
    other.size_ = 0;
#ifdef _WIN32
    file_handle_ = other.file_handle_;
    map_handle_ = other.map_handle_;
    other.file_handle_ = nullptr;
    other.map_handle_ = nullptr;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif
    return *this;
}

MappedFile::~MappedFile() { close(); }

void MappedFile::close() noexcept {
#ifdef _WIN32
    if (data_) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (map_handle_) {
        CloseHandle(map_handle_);
        map_handle_ = nullptr;
    }
    if (file_handle_) {
        CloseHandle(file_handle_);
        file_handle_ = nullptr;
    }
#else
    if (data_ && data_ != MAP_FAILED && size_ > 0) {
        munmap(const_cast<uint8_t*>(data_), static_cast<std::size_t>(size_));
    }
    data_ = nullptr;
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    size_ = 0;
}

Result<MappedFile> MappedFile::open(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return error(ErrorCode::FileNotFound, "file not found: " + path.string());
    }
    const uint64_t sz = static_cast<uint64_t>(std::filesystem::file_size(path, ec));
    if (ec) {
        return error(ErrorCode::FileError, "cannot stat " + path.string() + ": " + ec.message());
    }

    MappedFile m;
    m.path_ = path;
    m.size_ = sz;

    if (sz == 0) {
        m.data_ = nullptr;
        return m;
    }

#ifdef _WIN32
    HANDLE fh = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        return error(ErrorCode::FileError, "cannot open " + path.string());
    }
    HANDLE mh = CreateFileMappingW(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) {
        CloseHandle(fh);
        return error(ErrorCode::FileError, "cannot map " + path.string());
    }
    void* view = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        CloseHandle(mh);
        CloseHandle(fh);
        return error(ErrorCode::FileError, "cannot map view of " + path.string());
    }
    m.file_handle_ = fh;
    m.map_handle_ = mh;
    m.data_ = static_cast<const uint8_t*>(view);
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return error(ErrorCode::FileError, "cannot open " + path.string());
    }
    void* p = mmap(nullptr, static_cast<std::size_t>(sz), PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        ::close(fd);
        return error(ErrorCode::FileError, "cannot mmap " + path.string());
    }
    m.fd_ = fd;
    m.data_ = static_cast<const uint8_t*>(p);
#endif
    return m;
}

std::span<const uint8_t> MappedFile::slice(uint64_t offset, uint64_t length) const {
    if (offset > size_ || length > size_ - offset) {
        return {};
    }
    if (!data_) {
        return {};
    }
    return {data_ + offset, static_cast<std::size_t>(length)};
}

}  // namespace nn
