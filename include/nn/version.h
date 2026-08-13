#ifndef NN_VERSION_H
#define NN_VERSION_H

#include <string>
#include <vector>

namespace nn {

inline constexpr int kVersionMajor = NN_VERSION_MAJOR;
inline constexpr int kVersionMinor = NN_VERSION_MINOR;
inline constexpr int kVersionPatch = NN_VERSION_PATCH;

struct BuildInfo {
    std::string version;
    std::string git_commit;
    std::string compiler;
    std::string os;
    std::string architecture;
    std::vector<std::string> enabled_formats;
    std::vector<std::string> enabled_runtimes;
};

std::string version_string();
BuildInfo build_info();

}  // namespace nn

#endif
