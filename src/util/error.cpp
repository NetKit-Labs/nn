#include "nn/error.h"

#include <sstream>

namespace nn {

Error error(ErrorCode code, std::string message) {
    return Error(code, std::move(message));
}

const char* error_code_name(ErrorCode code) {
    switch (code) {
        case ErrorCode::Ok:
            return "ok";
        case ErrorCode::InvalidArgument:
            return "invalid_argument";
        case ErrorCode::UnknownOption:
            return "unknown_option";
        case ErrorCode::UnknownCommand:
            return "unknown_command";
        case ErrorCode::MissingArgument:
            return "missing_argument";
        case ErrorCode::FileNotFound:
            return "file_not_found";
        case ErrorCode::FileError:
            return "file_error";
        case ErrorCode::InvalidFormat:
            return "invalid_format";
        case ErrorCode::UnsupportedFormat:
            return "unsupported_format";
        case ErrorCode::ParseError:
            return "parse_error";
        case ErrorCode::InvalidGraph:
            return "invalid_graph";
        case ErrorCode::UnsupportedOperator:
            return "unsupported_operator";
        case ErrorCode::BackendUnavailable:
            return "backend_unavailable";
        case ErrorCode::ExecutionFailure:
            return "execution_failure";
        case ErrorCode::ValidationFailure:
            return "validation_failure";
        case ErrorCode::ConversionUnavailable:
            return "conversion_unavailable";
        case ErrorCode::LimitExceeded:
            return "limit_exceeded";
        case ErrorCode::Overflow:
            return "overflow";
        case ErrorCode::UnsafeOperation:
            return "unsafe_operation";
        case ErrorCode::InternalError:
            return "internal_error";
    }
    return "internal_error";
}

std::string Error::format() const {
    std::ostringstream os;
    os << message_;
    for (const auto& c : context_) {
        os << "\n  " << c;
    }
    return os.str();
}

ExitStatus Error::exit_status() const {
    switch (code_) {
        case ErrorCode::Ok:
            return kExitOk;
        case ErrorCode::InvalidArgument:
        case ErrorCode::UnknownOption:
        case ErrorCode::UnknownCommand:
        case ErrorCode::MissingArgument:
            return kExitUsage;
        case ErrorCode::FileNotFound:
        case ErrorCode::FileError:
            return kExitFile;
        case ErrorCode::InvalidFormat:
        case ErrorCode::ParseError:
        case ErrorCode::InvalidGraph:
            return kExitMalformed;
        case ErrorCode::UnsupportedFormat:
            return kExitUnsupportedFormat;
        case ErrorCode::UnsupportedOperator:
            return kExitUnsupportedOperator;
        case ErrorCode::BackendUnavailable:
        case ErrorCode::ConversionUnavailable:
            return kExitBackendUnavailable;
        case ErrorCode::ExecutionFailure:
            return kExitExecution;
        case ErrorCode::ValidationFailure:
            return kExitValidation;
        case ErrorCode::UnsafeOperation:
        case ErrorCode::LimitExceeded:
        case ErrorCode::Overflow:
        case ErrorCode::InternalError:
            return kExitInternal;
    }
    return kExitInternal;
}

}  // namespace nn
