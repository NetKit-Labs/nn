#ifndef NN_LOGGING_H
#define NN_LOGGING_H

#include <string_view>

namespace nn {

enum class LogLevel { Error, Warning, Info, Debug, Trace };

void set_log_level(LogLevel level);
LogLevel log_level();
void log_message(LogLevel level, std::string_view message);

void log_error(std::string_view message);
void log_warn(std::string_view message);
void log_info(std::string_view message);
void log_debug(std::string_view message);
void log_trace(std::string_view message);

}  // namespace nn

#endif
