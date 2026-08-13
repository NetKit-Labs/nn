#include "nn/cli.h"

#include "util/format_text.h"
#include "util/terminal.h"

#include <cstring>
#include <iostream>

namespace nn {

Printer::Printer() = default;

Printer::Printer(const GlobalOptions& opt) {
    quiet_ = opt.quiet;
    format_ = opt.output_format;
    color_ = want_color(opt.color == ColorMode::Always, opt.color == ColorMode::Never, true);
    if (opt.output_file) {
        FILE* f = std::fopen(opt.output_file->string().c_str(), "w");
        if (f) {
            owned_.reset(f);
            out_ = f;
            color_ = false;
        }
    }
}

void Printer::print(std::string_view text) {
    if (quiet_ && format_ == OutputFormat::Text) {
        return;
    }
    std::fwrite(text.data(), 1, text.size(), out_);
}

void Printer::println(std::string_view text) {
    print(text);
    print("\n");
}

void Printer::err(std::string_view text) { std::fwrite(text.data(), 1, text.size(), err_); }

void Printer::errln(std::string_view text) {
    err(text);
    err("\n");
}

void Printer::kv(std::string_view key, std::string_view value, int key_width) {
    if (format_ == OutputFormat::Porcelain) {
        porcelain(key, value);
        return;
    }
    std::string line;
    line.append(key);
    if (static_cast<int>(key.size()) < key_width) {
        line.append(static_cast<std::size_t>(key_width) - key.size(), ' ');
    }
    line += "  ";
    line += value;
    println(line);
}

void Printer::heading(std::string_view text) {
    if (format_ != OutputFormat::Text) {
        return;
    }
    println();
    println(text);
}

void Printer::json(const Json& j) { print(j.dump(true)); }

void Printer::yaml(const Json& j) { print(to_yaml(j)); }

void Printer::porcelain(std::string_view key, std::string_view value) {
    std::string line;
    line.append(key);
    line.push_back('\t');
    line.append(value);
    println(line);
}

int Printer::usage_error(std::string_view message, std::string_view usage) {
    errln(message);
    errln("");
    err("usage: ");
    errln(usage);
    errln("");
    errln("Try 'nn --help' for more information.");
    return kExitUsage;
}

std::string format_count(uint64_t n) { return human_count(n); }
std::string format_bytes(uint64_t n) { return human_bytes(n); }
std::string format_si(double n, std::string_view unit) { return human_si(n, unit); }
std::string yes_no(bool v) { return v ? "yes" : "no"; }

}  // namespace nn
