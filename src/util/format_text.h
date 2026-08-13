#ifndef NN_FORMAT_TEXT_H
#define NN_FORMAT_TEXT_H

#include <cstdint>
#include <string>
#include <string_view>

namespace nn {

std::string human_count(uint64_t n);
std::string human_bytes(uint64_t n);
std::string human_si(double n, std::string_view suffix = "");
std::string optional_count_text(bool known, uint64_t value, std::string_view suffix = "");

}  // namespace nn

#endif
