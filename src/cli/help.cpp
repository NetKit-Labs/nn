#include "nn/cli.h"

namespace nn {

void print_top_help(Printer& p) {
    p.println("nn - neural-network engineering toolkit");
    p.println();
    p.println("usage: nn [--version] [--help] [-v | -q] [--json | --yaml | --porcelain]");
    p.println("          [--color=auto|always|never] [--output FILE] [--threads N]");
    p.println("          <command> [<args>]");
    p.println();
    p.println("These are common nn commands:");
    p.println();
    p.println("inspect     summarize a model");
    p.println("ops         list operators");
    p.println("tensors     list tensors");
    p.println("io          show input/output contracts");
    p.println("metadata    show model metadata");
    p.println("hash        hash artifact, graph, or weights");
    p.println("diff        semantic comparison of two models");
    p.println("graph       export graph as text/dot/mermaid/json");
    p.println("memory      analyze memory and optional arena plan");
    p.println("compute     estimate MACs/FLOPs");
    p.println("quant       analyze quantization");
    p.println("sparsity    analyze weight sparsity");
    p.println("lint        static correctness checks");
    p.println("run         execute with an available backend");
    p.println("compare     compare models or backends on inputs");
    p.println("test        run test-vector manifests");
    p.println("validate    evaluate against a dataset/metrics");
    p.println("regression  policy check between two models");
    p.println("benchmark   measure inference latency");
    p.println("profile     per-operator runtime profile");
    p.println("compat      operator compatibility vs a runtime");
    p.println("target      deployability against a hardware profile");
    p.println("partition   accelerator / CPU partition");
    p.println("convert     conversion frontend");
    p.println("optimize    safe graph optimizations");
    p.println("extract     extract tensors or subgraphs");
    p.println("canonicalize");
    p.println("            stable canonical representation");
    p.println("bisect      regression bisect (artifacts or git)");
    p.println("doctor      installation and compile-time features");
    p.println("formats     registered format capabilities");
    p.println("backends    runtime backends");
    p.println("targets     built-in hardware profiles");
    p.println("config      git-like configuration");
    p.println("help        show this help or a command's help");
    p.println("version     version and build options");
    p.println();
    p.println("Try 'nn help <command>' or 'nn <command> --help'.");
}

void print_command_help(Printer& p, const CommandSpec& spec) {
    p.println("NAME");
    p.println("    nn-" + spec.name + " - " + spec.description);
    p.println();
    p.println("SYNOPSIS");
    p.println("    " + spec.synopsis);
    p.println();
    p.println("DESCRIPTION");
    p.println("    " + spec.description);
    p.println();
    if (!spec.arguments.empty()) {
        p.println("ARGUMENTS");
        for (const auto& a : spec.arguments) {
            p.println("    " + a);
        }
        p.println();
    }
    if (!spec.flags.empty()) {
        p.println("OPTIONS");
        for (const auto& f : spec.flags) {
            std::string line = "    ";
            if (f.short_name) {
                line += "-";
                line += f.short_name;
                line += ", ";
            }
            line += "--";
            line += f.long_name;
            if (f.takes_value) {
                line += " ";
                line += f.value_name.empty() ? "VALUE" : f.value_name;
            }
            line += "  ";
            line += f.help;
            p.println(line);
        }
        p.println();
    }
    if (!spec.examples.empty()) {
        p.println("EXAMPLES");
        for (const auto& e : spec.examples) {
            p.println("    " + e);
        }
        p.println();
    }
    p.println("EXIT STATUS");
    p.println("    " + (spec.exit_status.empty()
                            ? std::string("0 on success; non-zero on error. See nn(1).")
                            : spec.exit_status));
}

}  // namespace nn
