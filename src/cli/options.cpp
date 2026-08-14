#include "nn/cli.h"

#include "util/terminal.h"

#include <algorithm>
#include <cstdlib>

namespace nn {
namespace {

bool command_owns_output_flag(std::string_view command) {
    const CommandSpec* spec = find_command(command);
    if (!spec) {
        return false;
    }
    for (const auto& f : spec->flags) {
        if (f.long_name == "output" || f.short_name == 'o') {
            return true;
        }
    }
    return false;
}

}  // namespace

Result<GlobalOptions> parse_global_args(int argc, char** argv) {
    GlobalOptions g;
    std::vector<std::string> rest;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i] ? argv[i] : "";
        if (a == "-h" || a == "--help") {
            g.help = true;
            continue;
        }
        if (a == "--version") {
            g.version = true;
            continue;
        }
        if (a == "-v" || a == "--verbose") {
            ++g.verbose;
            continue;
        }
        if (a == "-q" || a == "--quiet") {
            g.quiet = true;
            continue;
        }
        if (a == "--json") {
            g.output_format = OutputFormat::Json;
            continue;
        }
        if (a == "--yaml") {
            g.output_format = OutputFormat::Yaml;
            continue;
        }
        if (a == "--porcelain") {
            g.output_format = OutputFormat::Porcelain;
            continue;
        }
        if (a == "--color=auto" || a == "--color=always" || a == "--color=never") {
            if (a.ends_with("always")) {
                g.color = ColorMode::Always;
            } else if (a.ends_with("never")) {
                g.color = ColorMode::Never;
            } else {
                g.color = ColorMode::Auto;
            }
            continue;
        }
        if (a == "--color") {
            if (i + 1 >= argc) {
                return error(ErrorCode::MissingArgument, "nn: option '--color' requires a value");
            }
            const std::string v = argv[++i];
            if (v == "always") {
                g.color = ColorMode::Always;
            } else if (v == "never") {
                g.color = ColorMode::Never;
            } else if (v == "auto") {
                g.color = ColorMode::Auto;
            } else {
                return error(ErrorCode::InvalidArgument, "nn: invalid --color value '" + v + "'");
            }
            continue;
        }
        if (a == "--output" || a == "-o" || a.rfind("--output=", 0) == 0) {
            if (!g.command.empty() && command_owns_output_flag(g.command)) {
                rest.push_back(a);
                if (a == "--output" || a == "-o") {
                    if (i + 1 >= argc) {
                        return error(ErrorCode::MissingArgument, "nn: option '--output' requires a value");
                    }
                    rest.push_back(argv[++i] ? argv[i] : "");
                }
                continue;
            }
            if (a.rfind("--output=", 0) == 0) {
                g.output_file = a.substr(9);
                continue;
            }
            if (i + 1 >= argc) {
                return error(ErrorCode::MissingArgument, "nn: option '--output' requires a value");
            }
            g.output_file = argv[++i];
            continue;
        }
        if (a == "--threads") {
            if (i + 1 >= argc) {
                return error(ErrorCode::MissingArgument, "nn: option '--threads' requires a value");
            }
            g.threads = std::atoi(argv[++i]);
            continue;
        }
        if (a.rfind("--threads=", 0) == 0) {
            g.threads = std::atoi(a.c_str() + 10);
            continue;
        }
        if (g.command.empty() && !a.empty() && a[0] != '-') {
            g.command = a;
            continue;
        }
        rest.push_back(a);
    }
    g.args = std::move(rest);
    return g;
}

Result<ParsedArgs> parse_command_args(const std::vector<std::string>& args, const CommandSpec& spec) {
    ParsedArgs out;
    std::map<std::string, const FlagSpec*> by_long;
    std::map<char, const FlagSpec*> by_short;
    for (const auto& f : spec.flags) {
        by_long[f.long_name] = &f;
        if (f.short_name) {
            by_short[f.short_name] = &f;
        }
    }
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--") {
            for (std::size_t j = i + 1; j < args.size(); ++j) {
                out.positionals.push_back(args[j]);
            }
            break;
        }
        if (a == "-h" || a == "--help") {
            out.help = true;
            continue;
        }
        if (a.rfind("--", 0) == 0) {
            std::string name = a.substr(2);
            std::string value;
            bool has_eq = false;
            const auto eq = name.find('=');
            if (eq != std::string::npos) {
                value = name.substr(eq + 1);
                name = name.substr(0, eq);
                has_eq = true;
            }
            auto it = by_long.find(name);
            if (it == by_long.end()) {
                return error(ErrorCode::UnknownOption, "nn: unknown option '--" + name + "'");
            }
            const FlagSpec* f = it->second;
            if (f->takes_value) {
                if (!has_eq) {
                    if (i + 1 >= args.size()) {
                        return error(ErrorCode::MissingArgument,
                                     "nn: option '--" + name + "' requires a value");
                    }
                    value = args[++i];
                }
                out.options[name] = value;
                out.multi[name].push_back(value);
            } else {
                if (has_eq) {
                    return error(ErrorCode::InvalidArgument,
                                 "nn: option '--" + name + "' does not take a value");
                }
                out.options[name] = "1";
                out.multi[name].push_back("1");
            }
            continue;
        }
        if (a.size() >= 2 && a[0] == '-' && a[1] != '-') {
            for (std::size_t c = 1; c < a.size(); ++c) {
                auto it = by_short.find(a[c]);
                if (it == by_short.end()) {
                    return error(ErrorCode::UnknownOption,
                                 std::string("nn: unknown option '-") + a[c] + "'");
                }
                const FlagSpec* f = it->second;
                if (f->takes_value) {
                    std::string value;
                    if (c + 1 < a.size()) {
                        value = a.substr(c + 1);
                    } else {
                        if (i + 1 >= args.size()) {
                            return error(ErrorCode::MissingArgument,
                                         std::string("nn: option '-") + a[c] + "' requires a value");
                        }
                        value = args[++i];
                    }
                    out.options[f->long_name] = value;
                    out.multi[f->long_name].push_back(value);
                    break;
                }
                out.options[f->long_name] = "1";
                out.multi[f->long_name].push_back("1");
            }
            continue;
        }
        out.positionals.push_back(a);
    }
    return out;
}

bool flag_set(const ParsedArgs& a, std::string_view name) {
    return a.options.find(std::string(name)) != a.options.end();
}

std::optional<std::string> flag_value(const ParsedArgs& a, std::string_view name) {
    auto it = a.options.find(std::string(name));
    if (it == a.options.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::vector<std::string> flag_values(const ParsedArgs& a, std::string_view name) {
    auto it = a.multi.find(std::string(name));
    if (it != a.multi.end()) {
        return it->second;
    }
    if (auto v = flag_value(a, name)) {
        return {*v};
    }
    return {};
}

}  // namespace nn
