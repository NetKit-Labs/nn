#include "nn/json.h"

#include <cctype>
#include <sstream>

namespace nn {
namespace {

void yaml_impl(const Json& j, std::ostringstream& os, int indent, bool in_array) {
    auto pad = [&](int n) {
        for (int i = 0; i < n; ++i) {
            os << "  ";
        }
    };
    switch (j.type()) {
        case Json::Type::Null:
            os << "null";
            break;
        case Json::Type::Bool:
            os << (j.as_bool() ? "true" : "false");
            break;
        case Json::Type::Number:
            os << Json(j.as_number()).dump(false);
            // dump includes no newline for non-pretty numbers; Json dump for number is fine
            break;
        case Json::Type::String: {
            const std::string& s = j.as_string();
            bool simple = !s.empty();
            for (char c : s) {
                if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' ||
                      c == '.' || c == '/')) {
                    simple = false;
                    break;
                }
            }
            if (simple) {
                os << s;
            } else {
                os << j.dump(false);
            }
            break;
        }
        case Json::Type::Array: {
            const auto& a = j.as_array();
            if (a.empty()) {
                os << "[]";
                break;
            }
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (i > 0 || !in_array) {
                    if (i > 0) {
                        os << '\n';
                    }
                    pad(indent);
                }
                os << "- ";
                if (a[i].is_object() || a[i].is_array()) {
                    os << '\n';
                    yaml_impl(a[i], os, indent + 1, a[i].is_array());
                } else {
                    yaml_impl(a[i], os, indent + 1, false);
                }
            }
            break;
        }
        case Json::Type::Object: {
            const auto& o = j.as_object();
            if (o.empty()) {
                os << "{}";
                break;
            }
            std::size_t i = 0;
            for (const auto& [k, v] : o) {
                if (i > 0) {
                    os << '\n';
                }
                pad(indent);
                os << k << ":";
                if (v.is_object() || v.is_array()) {
                    os << '\n';
                    yaml_impl(v, os, indent + 1, v.is_array());
                } else {
                    os << ' ';
                    yaml_impl(v, os, indent + 1, false);
                }
                ++i;
            }
            break;
        }
    }
}

}  // namespace

std::string to_yaml(const Json& json) {
    std::ostringstream os;
    yaml_impl(json, os, 0, json.is_array());
    os << '\n';
    return os.str();
}

}  // namespace nn
