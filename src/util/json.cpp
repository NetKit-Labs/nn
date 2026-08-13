#include "nn/json.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace nn {
namespace {

const Json kNull;

void dump_impl(const Json& j, std::ostringstream& os, bool pretty, int indent) {
    auto pad = [&](int n) {
        if (pretty) {
            os << '\n';
            for (int i = 0; i < n; ++i) {
                os << "  ";
            }
        }
    };
    switch (j.type()) {
        case Json::Type::Null:
            os << "null";
            break;
        case Json::Type::Bool:
            os << (j.as_bool() ? "true" : "false");
            break;
        case Json::Type::Number: {
            const double v = j.as_number();
            if (std::isfinite(v) && v == std::floor(v) && v >= -9.0e15 && v <= 9.0e15) {
                os << static_cast<int64_t>(v);
            } else {
                os << v;
            }
            break;
        }
        case Json::Type::String: {
            os << '"';
            for (char ch : j.as_string()) {
                const unsigned char c = static_cast<unsigned char>(ch);
                switch (c) {
                    case '"':
                        os << "\\\"";
                        break;
                    case '\\':
                        os << "\\\\";
                        break;
                    case '\n':
                        os << "\\n";
                        break;
                    case '\r':
                        os << "\\r";
                        break;
                    case '\t':
                        os << "\\t";
                        break;
                    default:
                        if (c < 0x20) {
                            char buf[8];
                            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                            os << buf;
                        } else {
                            os << static_cast<char>(c);
                        }
                        break;
                }
            }
            os << '"';
            break;
        }
        case Json::Type::Array: {
            const auto& a = j.as_array();
            os << '[';
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (pretty) {
                    pad(indent + 1);
                }
                dump_impl(a[i], os, pretty, indent + 1);
                if (i + 1 < a.size()) {
                    os << ',';
                    if (!pretty) {
                        os << ' ';
                    }
                }
            }
            if (pretty && !a.empty()) {
                pad(indent);
            }
            os << ']';
            break;
        }
        case Json::Type::Object: {
            const auto& o = j.as_object();
            os << '{';
            std::size_t i = 0;
            for (const auto& [k, v] : o) {
                if (pretty) {
                    pad(indent + 1);
                }
                Json key(k);
                dump_impl(key, os, false, 0);
                os << (pretty ? ": " : ":");
                dump_impl(v, os, pretty, indent + 1);
                if (i + 1 < o.size()) {
                    os << ',';
                    if (!pretty) {
                        os << ' ';
                    }
                }
                ++i;
            }
            if (pretty && !o.empty()) {
                pad(indent);
            }
            os << '}';
            break;
        }
    }
}

class Parser {
public:
    explicit Parser(std::string_view t) : t_(t) {}

    Result<Json> parse() {
        skip();
        auto v = value();
        if (!v) {
            return v.error();
        }
        skip();
        if (i_ != t_.size()) {
            return error(ErrorCode::ParseError, "trailing JSON content");
        }
        return v;
    }

private:
    void skip() {
        while (i_ < t_.size() && std::isspace(static_cast<unsigned char>(t_[i_]))) {
            ++i_;
        }
    }

    Error unexpected(const char* what) const {
        return error(ErrorCode::ParseError,
                     std::string("JSON parse error at offset ") + std::to_string(i_) + ": " + what);
    }

    Result<Json> value() {
        skip();
        if (i_ >= t_.size()) {
            return unexpected("unexpected end");
        }
        const char c = t_[i_];
        if (c == 'n') {
            return lit("null", Json());
        }
        if (c == 't') {
            return lit("true", Json(true));
        }
        if (c == 'f') {
            return lit("false", Json(false));
        }
        if (c == '"') {
            return parse_string();
        }
        if (c == '{') {
            return parse_object();
        }
        if (c == '[') {
            return parse_array();
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return parse_number();
        }
        return unexpected("invalid value");
    }

    Result<Json> lit(const char* s, Json v) {
        const std::size_t n = std::strlen(s);
        if (t_.substr(i_, n) != s) {
            return unexpected("invalid literal");
        }
        i_ += n;
        return v;
    }

    Result<Json> parse_string() {
        ++i_;
        std::string out;
        while (i_ < t_.size()) {
            const char c = t_[i_++];
            if (c == '"') {
                return Json(std::move(out));
            }
            if (c == '\\') {
                if (i_ >= t_.size()) {
                    return unexpected("unterminated escape");
                }
                const char e = t_[i_++];
                switch (e) {
                    case '"':
                    case '\\':
                    case '/':
                        out.push_back(e);
                        break;
                    case 'b':
                        out.push_back('\b');
                        break;
                    case 'f':
                        out.push_back('\f');
                        break;
                    case 'n':
                        out.push_back('\n');
                        break;
                    case 'r':
                        out.push_back('\r');
                        break;
                    case 't':
                        out.push_back('\t');
                        break;
                    case 'u': {
                        if (i_ + 4 > t_.size()) {
                            return unexpected("truncated unicode escape");
                        }
                        unsigned code = 0;
                        for (int k = 0; k < 4; ++k) {
                            const char h = t_[i_++];
                            code <<= 4;
                            if (h >= '0' && h <= '9') {
                                code |= static_cast<unsigned>(h - '0');
                            } else if (h >= 'a' && h <= 'f') {
                                code |= static_cast<unsigned>(h - 'a' + 10);
                            } else if (h >= 'A' && h <= 'F') {
                                code |= static_cast<unsigned>(h - 'A' + 10);
                            } else {
                                return unexpected("invalid hex in unicode escape");
                            }
                        }
                        if (code < 0x80) {
                            out.push_back(static_cast<char>(code));
                        } else if (code < 0x800) {
                            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        } else {
                            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        }
                        break;
                    }
                    default:
                        return unexpected("invalid escape");
                }
            } else {
                out.push_back(c);
            }
        }
        return unexpected("unterminated string");
    }

    Result<Json> parse_number() {
        const std::size_t start = i_;
        if (t_[i_] == '-') {
            ++i_;
        }
        if (i_ >= t_.size() || !std::isdigit(static_cast<unsigned char>(t_[i_]))) {
            return unexpected("invalid number");
        }
        if (t_[i_] == '0') {
            ++i_;
        } else {
            while (i_ < t_.size() && std::isdigit(static_cast<unsigned char>(t_[i_]))) {
                ++i_;
            }
        }
        if (i_ < t_.size() && t_[i_] == '.') {
            ++i_;
            if (i_ >= t_.size() || !std::isdigit(static_cast<unsigned char>(t_[i_]))) {
                return unexpected("invalid fraction");
            }
            while (i_ < t_.size() && std::isdigit(static_cast<unsigned char>(t_[i_]))) {
                ++i_;
            }
        }
        if (i_ < t_.size() && (t_[i_] == 'e' || t_[i_] == 'E')) {
            ++i_;
            if (i_ < t_.size() && (t_[i_] == '+' || t_[i_] == '-')) {
                ++i_;
            }
            if (i_ >= t_.size() || !std::isdigit(static_cast<unsigned char>(t_[i_]))) {
                return unexpected("invalid exponent");
            }
            while (i_ < t_.size() && std::isdigit(static_cast<unsigned char>(t_[i_]))) {
                ++i_;
            }
        }
        const std::string s(t_.substr(start, i_ - start));
        char* end = nullptr;
        const double v = std::strtod(s.c_str(), &end);
        return Json(v);
    }

    Result<Json> parse_array() {
        ++i_;
        Json arr = Json::array();
        skip();
        if (i_ < t_.size() && t_[i_] == ']') {
            ++i_;
            return arr;
        }
        while (true) {
            auto v = value();
            if (!v) {
                return v.error();
            }
            arr.push(std::move(v.value()));
            skip();
            if (i_ >= t_.size()) {
                return unexpected("unterminated array");
            }
            if (t_[i_] == ',') {
                ++i_;
                continue;
            }
            if (t_[i_] == ']') {
                ++i_;
                return arr;
            }
            return unexpected("expected comma or ']'");
        }
    }

    Result<Json> parse_object() {
        ++i_;
        Json obj = Json::object();
        skip();
        if (i_ < t_.size() && t_[i_] == '}') {
            ++i_;
            return obj;
        }
        while (true) {
            skip();
            if (i_ >= t_.size() || t_[i_] != '"') {
                return unexpected("expected object key");
            }
            auto key = parse_string();
            if (!key) {
                return key.error();
            }
            skip();
            if (i_ >= t_.size() || t_[i_] != ':') {
                return unexpected("expected ':'");
            }
            ++i_;
            auto v = value();
            if (!v) {
                return v.error();
            }
            obj[key.value().as_string()] = std::move(v.value());
            skip();
            if (i_ >= t_.size()) {
                return unexpected("unterminated object");
            }
            if (t_[i_] == ',') {
                ++i_;
                continue;
            }
            if (t_[i_] == '}') {
                ++i_;
                return obj;
            }
            return unexpected("expected comma or '}'");
        }
    }

    std::string_view t_;
    std::size_t i_ = 0;
};

}  // namespace

Json::Type Json::type() const {
    return static_cast<Type>(data_.index());
}

bool Json::as_bool() const {
    return std::holds_alternative<bool>(data_) && std::get<bool>(data_);
}

double Json::as_number() const {
    return std::holds_alternative<double>(data_) ? std::get<double>(data_) : 0.0;
}

const std::string& Json::as_string() const {
    static const std::string empty;
    return std::holds_alternative<std::string>(data_) ? std::get<std::string>(data_) : empty;
}

const std::vector<Json>& Json::as_array() const {
    static const std::vector<Json> empty;
    return std::holds_alternative<std::vector<Json>>(data_) ? std::get<std::vector<Json>>(data_)
                                                           : empty;
}

const std::map<std::string, Json>& Json::as_object() const {
    static const std::map<std::string, Json> empty;
    return std::holds_alternative<std::map<std::string, Json>>(data_)
               ? std::get<std::map<std::string, Json>>(data_)
               : empty;
}

Json& Json::operator[](const std::string& key) {
    if (!std::holds_alternative<std::map<std::string, Json>>(data_)) {
        data_ = std::map<std::string, Json>{};
    }
    return std::get<std::map<std::string, Json>>(data_)[key];
}

const Json& Json::at(const std::string& key) const {
    if (!is_object()) {
        return kNull;
    }
    const auto& o = as_object();
    const auto it = o.find(key);
    return it == o.end() ? kNull : it->second;
}

bool Json::contains(const std::string& key) const {
    if (!is_object()) {
        return false;
    }
    return as_object().count(key) != 0;
}

void Json::push(Json v) {
    if (!std::holds_alternative<std::vector<Json>>(data_)) {
        data_ = std::vector<Json>{};
    }
    std::get<std::vector<Json>>(data_).push_back(std::move(v));
}

std::string Json::dump(bool pretty, int indent) const {
    std::ostringstream os;
    dump_impl(*this, os, pretty, indent);
    if (pretty) {
        os << '\n';
    }
    return os.str();
}

Result<Json> parse_json(std::string_view text) {
    Parser p(text);
    return p.parse();
}

}  // namespace nn
