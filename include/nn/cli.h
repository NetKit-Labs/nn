#ifndef NN_CLI_H
#define NN_CLI_H

#include "nn/error.h"
#include "nn/exit_status.h"
#include "nn/json.h"
#include "nn/result.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace nn {

enum class OutputFormat { Text, Json, Yaml, Porcelain, Csv };

enum class ColorMode { Auto, Always, Never };

struct GlobalOptions {
    bool help = false;
    bool version = false;
    int verbose = 0;
    bool quiet = false;
    ColorMode color = ColorMode::Auto;
    OutputFormat output_format = OutputFormat::Text;
    std::optional<std::filesystem::path> output_file;
    int threads = 0;
    std::string command;
    std::vector<std::string> args;  // remaining after global flags
};

struct FlagSpec {
    std::string long_name;
    char short_name = '\0';
    bool takes_value = false;
    std::string value_name;
    std::string help;
};

struct CommandSpec {
    std::string name;
    std::string synopsis;
    std::string description;
    std::vector<std::string> arguments;
    std::vector<FlagSpec> flags;
    std::vector<std::string> examples;
    std::string exit_status;
    std::function<int(const GlobalOptions&)> run;
};

struct ParsedArgs {
    std::vector<std::string> positionals;
    std::map<std::string, std::string> options;  // last value wins
    std::map<std::string, std::vector<std::string>> multi;  // all values, for repeatable flags
    bool help = false;
};

Result<GlobalOptions> parse_global_args(int argc, char** argv);
Result<ParsedArgs> parse_command_args(const std::vector<std::string>& args,
                                      const CommandSpec& spec);

bool flag_set(const ParsedArgs& a, std::string_view name);
std::optional<std::string> flag_value(const ParsedArgs& a, std::string_view name);
std::vector<std::string> flag_values(const ParsedArgs& a, std::string_view name);

class Printer {
public:
    Printer();
    explicit Printer(const GlobalOptions& opt);

    bool color() const { return color_; }
    OutputFormat format() const { return format_; }

    void print(std::string_view text);
    void println(std::string_view text = {});
    void err(std::string_view text);
    void errln(std::string_view text);
    void kv(std::string_view key, std::string_view value, int key_width = 20);
    void heading(std::string_view text);
    void json(const Json& j);
    void yaml(const Json& j);
    void porcelain(std::string_view key, std::string_view value);

    int usage_error(std::string_view message, std::string_view usage);

private:
    bool color_ = false;
    bool quiet_ = false;
    OutputFormat format_ = OutputFormat::Text;
    FILE* out_ = stdout;
    FILE* err_ = stderr;
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> owned_{nullptr, &std::fclose};
};

std::string format_count(uint64_t n);
std::string format_bytes(uint64_t n);
std::string format_si(double n, std::string_view unit = "");
std::string yes_no(bool v);

const std::vector<CommandSpec>& all_commands();
const CommandSpec* find_command(std::string_view name);
int run_cli(int argc, char** argv);

void print_top_help(Printer& p);
void print_command_help(Printer& p, const CommandSpec& spec);

std::map<std::string, std::string> load_config();
Status save_user_config(const std::map<std::string, std::string>& values);

}  // namespace nn

#endif
