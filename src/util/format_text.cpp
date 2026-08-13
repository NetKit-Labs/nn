#include "util/format_text.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace nn {

std::string human_count(uint64_t n) {
    std::string s = std::to_string(n);
    std::string out;
    const int first = static_cast<int>(s.size() % 3);
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i > 0 && (static_cast<int>(i) - first) % 3 == 0) {
            out.push_back(',');
        }
        out.push_back(s[i]);
    }
    return out;
}

std::string human_bytes(uint64_t n) {
    static const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    double v = static_cast<double>(n);
    int u = 0;
    while (v >= 1024.0 && u < 5) {
        v /= 1024.0;
        ++u;
    }
    std::ostringstream os;
    if (u == 0) {
        os << n << " B";
    } else {
        os << std::fixed << std::setprecision(v >= 10.0 ? 1 : 2) << v << " " << units[u];
    }
    return os.str();
}

std::string human_si(double n, std::string_view suffix) {
    if (!std::isfinite(n)) {
        return "unknown";
    }
    const char* prefixes[] = {"", "K", "M", "G", "T"};
    int p = 0;
    double v = n;
    const double absn = std::fabs(v);
    if (absn >= 1000.0) {
        while (std::fabs(v) >= 1000.0 && p < 4) {
            v /= 1000.0;
            ++p;
        }
    }
    std::ostringstream os;
    os << std::fixed << std::setprecision(v >= 10.0 ? 1 : 2) << v;
    if (prefixes[p][0] != '\0') {
        os << ' ' << prefixes[p];
    }
    if (!suffix.empty()) {
        if (prefixes[p][0] == '\0') {
            os << ' ';
        }
        os << suffix;
    }
    return os.str();
}

std::string optional_count_text(bool known, uint64_t value, std::string_view suffix) {
    if (!known) {
        return "unknown";
    }
    return human_si(static_cast<double>(value), suffix);
}

}  // namespace nn
