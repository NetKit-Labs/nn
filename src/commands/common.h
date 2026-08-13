#ifndef NN_CMD_COMMON_H
#define NN_CMD_COMMON_H

#include "nn/cli.h"
#include "nn/format.h"

namespace nn {

inline int cmd_fail(Printer& p, const Error& e) {
    p.errln(std::string("nn: ") + e.format());
    return e.exit_status();
}

inline Result<ParsedArgs> cmd_args(const GlobalOptions& g, const char* name) {
    const CommandSpec* spec = find_command(name);
    if (!spec) {
        return error(ErrorCode::InternalError, "missing command spec");
    }
    return parse_command_args(g.args, *spec);
}

inline Result<ModelIR> cmd_load(const ParsedArgs& a, std::size_t index = 0) {
    if (a.positionals.size() <= index) {
        return error(ErrorCode::MissingArgument, "missing model path");
    }
    return load_model(a.positionals[index]);
}

inline Json json_shape(const Shape& s) {
    Json a = Json::array();
    for (const auto& d : s.dims) {
        if (d.value) {
            a.push(Json(*d.value));
        } else {
            a.push(Json(d.symbol.empty() ? "?" : d.symbol));
        }
    }
    return a;
}

}  // namespace nn

#endif
