#ifndef NN_RESULT_H
#define NN_RESULT_H

#include "nn/error.h"

#include <optional>
#include <utility>
#include <variant>

namespace nn {

template <typename T>
class [[nodiscard]] Result {
public:
    Result(T value) : data_(std::in_place_index<0>, std::move(value)) {}
    Result(Error err) : data_(std::in_place_index<1>, std::move(err)) {}

    static Result ok(T value) { return Result(std::move(value)); }
    static Result err(Error e) { return Result(std::move(e)); }

    bool is_ok() const { return data_.index() == 0; }
    bool is_err() const { return data_.index() == 1; }
    explicit operator bool() const { return is_ok(); }

    T& value() & { return std::get<0>(data_); }
    const T& value() const& { return std::get<0>(data_); }
    T&& value() && { return std::get<0>(std::move(data_)); }

    const Error& error() const { return std::get<1>(data_); }
    Error&& take_error() && { return std::get<1>(std::move(data_)); }

    T value_or(T fallback) const {
        return is_ok() ? std::get<0>(data_) : std::move(fallback);
    }

private:
    std::variant<T, Error> data_;
};

class [[nodiscard]] Status {
public:
    Status() = default;
    Status(Error err) : error_(std::move(err)) {}

    static Status ok() { return Status(); }
    static Status err(Error e) { return Status(std::move(e)); }

    bool is_ok() const { return !error_.has_value(); }
    bool is_err() const { return error_.has_value(); }
    explicit operator bool() const { return is_ok(); }

    const Error& error() const { return *error_; }

private:
    std::optional<Error> error_;
};

}  // namespace nn

#endif
