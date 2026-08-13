#include "nn/cli.h"

#include <cstdint>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    nn::CommandSpec spec;
    spec.name = "inspect";
    spec.flags = {{"summary", '\0', false, "", ""},
                  {"all", '\0', false, "", ""},
                  {"metadata", '\0', false, "", ""}};
    std::vector<std::string> args;
    std::string cur;
    for (size_t i = 0; i < size && args.size() < 16; ++i) {
        const unsigned char c = data[i];
        if (c == 0 || c == ' ' || c == '\n') {
            if (!cur.empty()) {
                args.push_back(cur);
                cur.clear();
            }
        } else if (c >= 32 && c < 127 && cur.size() < 64) {
            cur.push_back(static_cast<char>(c));
        }
    }
    if (!cur.empty()) {
        args.push_back(cur);
    }
    (void)nn::parse_command_args(args, spec);
    return 0;
}
