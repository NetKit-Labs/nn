#include "nn/logging.h"

#include <iostream>

namespace nn {
namespace {
LogLevel g_level = LogLevel::Warning;
}

void set_log_level(LogLevel level) { g_level = level; }
LogLevel log_level() { return g_level; }

void log_message(LogLevel level, std::string_view message) {
    if (static_cast<int>(level) > static_cast<int>(g_level)) {
        return;
    }
    const char* tag = "info";
    switch (level) {
        case LogLevel::Error:
            tag = "error";
            break;
        case LogLevel::Warning:
            tag = "warning";
            break;
        case LogLevel::Info:
            tag = "info";
            break;
        case LogLevel::Debug:
            tag = "debug";
            break;
        case LogLevel::Trace:
            tag = "trace";
            break;
    }
    std::cerr << "nn: " << tag << ": " << message << '\n';
}

void log_error(std::string_view message) { log_message(LogLevel::Error, message); }
void log_warn(std::string_view message) { log_message(LogLevel::Warning, message); }
void log_info(std::string_view message) { log_message(LogLevel::Info, message); }
void log_debug(std::string_view message) { log_message(LogLevel::Debug, message); }
void log_trace(std::string_view message) { log_message(LogLevel::Trace, message); }

}  // namespace nn
