#ifndef NN_ERROR_H
#define NN_ERROR_H

#include "nn/exit_status.h"

#include <string>
#include <utility>
#include <vector>

namespace nn {

enum class ErrorCode {
    Ok = 0,
    InvalidArgument,
    UnknownOption,
    UnknownCommand,
    MissingArgument,
    FileNotFound,
    FileError,
    InvalidFormat,
    UnsupportedFormat,
    ParseError,
    InvalidGraph,
    UnsupportedOperator,
    BackendUnavailable,
    ExecutionFailure,
    ValidationFailure,
    ConversionUnavailable,
    LimitExceeded,
    Overflow,
    UnsafeOperation,
    InternalError
};

class Error {
public:
    Error() = default;
    Error(ErrorCode code, std::string message)
        : code_(code), message_(std::move(message)) {}

    ErrorCode code() const { return code_; }
    const std::string& message() const { return message_; }
    const std::vector<std::string>& context() const { return context_; }

    Error& with(std::string item) {
        context_.push_back(std::move(item));
        return *this;
    }

    std::string format() const;
    ExitStatus exit_status() const;

private:
    ErrorCode code_ = ErrorCode::InternalError;
    std::string message_;
    std::vector<std::string> context_;
};

Error error(ErrorCode code, std::string message);

const char* error_code_name(ErrorCode code);

}  // namespace nn

#endif
