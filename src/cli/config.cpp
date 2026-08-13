#include "nn/cli.h"

#include "nn/json.h"

#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <pwd.h>
#endif

namespace nn {
namespace {

std::filesystem::path user_config_path() {
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME")) {
        return std::filesystem::path(xdg) / "nn" / "config";
    }
#ifdef _WIN32
    if (const char* app = std::getenv("APPDATA")) {
        return std::filesystem::path(app) / "nn" / "config";
    }
#endif
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".config" / "nn" / "config";
    }
    return "nn.config";
}

std::filesystem::path repo_config_path() {
    std::error_code ec;
    auto p = std::filesystem::current_path(ec) / ".nnconfig";
    return p;
}

std::map<std::string, std::string> parse_config_file(const std::filesystem::path& path) {
    std::map<std::string, std::string> m;
    std::ifstream in(path);
    if (!in) {
        return m;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        m[line.substr(0, eq)] = line.substr(eq + 1);
    }
    return m;
}

}  // namespace

std::map<std::string, std::string> load_config() {
    auto m = parse_config_file(user_config_path());
    auto repo = parse_config_file(repo_config_path());
    for (auto& [k, v] : repo) {
        m[k] = v;
    }
    return m;
}

Status save_user_config(const std::map<std::string, std::string>& m) {
    auto path = user_config_path();
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream out(path);
    if (!out) {
        return Status::err(error(ErrorCode::FileError, "cannot write " + path.string()));
    }
    for (const auto& [k, v] : m) {
        out << k << '=' << v << '\n';
    }
    return Status::ok();
}

std::filesystem::path nn_user_config_path() { return user_config_path(); }

}  // namespace nn
