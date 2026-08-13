#ifndef NN_JSON_H
#define NN_JSON_H

#include "nn/result.h"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace nn {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool v) : data_(v) {}
    Json(int v) : data_(static_cast<double>(v)) {}
    Json(int64_t v) : data_(static_cast<double>(v)) {}
    Json(uint64_t v) : data_(static_cast<double>(v)) {}
    Json(double v) : data_(v) {}
    Json(const char* v) : data_(std::string(v)) {}
    Json(std::string v) : data_(std::move(v)) {}
    Json(std::vector<Json> v) : data_(std::move(v)) {}
    Json(std::map<std::string, Json> v) : data_(std::move(v)) {}

    static Json object() { return Json(std::map<std::string, Json>{}); }
    static Json array() { return Json(std::vector<Json>{}); }

    Type type() const;
    bool is_null() const { return type() == Type::Null; }
    bool is_bool() const { return type() == Type::Bool; }
    bool is_number() const { return type() == Type::Number; }
    bool is_string() const { return type() == Type::String; }
    bool is_array() const { return type() == Type::Array; }
    bool is_object() const { return type() == Type::Object; }

    bool as_bool() const;
    double as_number() const;
    const std::string& as_string() const;
    const std::vector<Json>& as_array() const;
    const std::map<std::string, Json>& as_object() const;

    Json& operator[](const std::string& key);
    const Json& at(const std::string& key) const;
    bool contains(const std::string& key) const;
    void push(Json v);

    std::string dump(bool pretty = false, int indent = 0) const;

private:
    std::variant<std::monostate, bool, double, std::string, std::vector<Json>,
                 std::map<std::string, Json>>
        data_;
};

Result<Json> parse_json(std::string_view text);
std::string to_yaml(const Json& json);

}  // namespace nn

#endif
