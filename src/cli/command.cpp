#include "nn/cli.h"

#include "cli/commands.h"
#include "nn/logging.h"
#include "nn/version.h"

#include <iostream>

namespace nn {
namespace {

CommandSpec make(std::string name, std::string synopsis, std::string description,
                 std::vector<std::string> arguments, std::vector<FlagSpec> flags,
                 std::vector<std::string> examples, std::string exit_status,
                 std::function<int(const GlobalOptions&)> run) {
    CommandSpec s;
    s.name = std::move(name);
    s.synopsis = std::move(synopsis);
    s.description = std::move(description);
    s.arguments = std::move(arguments);
    s.flags = std::move(flags);
    s.examples = std::move(examples);
    s.exit_status = std::move(exit_status);
    s.run = std::move(run);
    return s;
}

const std::vector<CommandSpec>& commands() {
    static const std::vector<CommandSpec> k = {
        make("inspect", "nn inspect [options] <model>",
             "Summarize a neural-network model artifact.", {"<model>  Path to a model file or directory"},
             {{"summary", '\0', false, "", "short summary"},
              {"all", '\0', false, "", "include all sections"},
              {"metadata", '\0', false, "", "show metadata"},
              {"inputs", '\0', false, "", "show inputs"},
              {"outputs", '\0', false, "", "show outputs"},
              {"ops", '\0', false, "", "show operators"},
              {"tensors", '\0', false, "", "show tensors"},
              {"weights", '\0', false, "", "show weights"},
              {"quantization", '\0', false, "", "show quantization"},
              {"subgraphs", '\0', false, "", "show subgraphs"},
              {"raw", '\0', false, "", "include raw format metadata"}},
             {"nn inspect model.onnx", "nn inspect model.onnx --json"},
             "0 success; 3 file error; 4 malformed; 5 unsupported format", cmd_inspect),
        make("ops", "nn ops [options] <model>", "List operators in a model.", {"<model>"},
             {{"by-type", '\0', false, "", "group by operator type"},
              {"by-cost", '\0', false, "", "sort by compute cost"},
              {"by-memory", '\0', false, "", "sort by memory"},
              {"details", '\0', false, "", "per-node details"},
              {"unsupported", '\0', false, "", "list unknown/unrecognized ops"},
              {"canonical", '\0', false, "", "use canonical names"},
              {"native", '\0', false, "", "use native names"},
              {"op", '\0', true, "TYPE", "filter by operator type"}},
             {"nn ops model.onnx", "nn ops model.onnx --op Conv"},
             "0 success", cmd_ops),
        make("tensors", "nn tensors [options] <model>", "List tensors.", {"<model>"},
             {{"weights", '\0', false, "", "constants only"},
              {"activations", '\0', false, "", "non-constants only"},
              {"inputs", '\0', false, "", "model inputs"},
              {"outputs", '\0', false, "", "model outputs"},
              {"largest", '\0', false, "", "sort by size descending"},
              {"dtype", '\0', true, "TYPE", "filter by dtype"},
              {"name", '\0', true, "PATTERN", "substring match on name"}},
             {"nn tensors model.onnx --largest"}, "0 success", cmd_tensors),
        make("io", "nn io [options] <model>", "Show input/output contracts.", {"<model>"}, {},
             {"nn io model.onnx"}, "0 success", cmd_io),
        make("metadata", "nn metadata [options] <model>", "Show model metadata only.", {"<model>"}, {},
             {"nn metadata model.gguf"}, "0 success", cmd_metadata),
        make("hash", "nn hash [options] <model>", "Hash artifact, graph, weights, or a tensor.",
             {"<model>"},
             {{"graph", '\0', false, "", "hash canonical graph"},
              {"weights", '\0', false, "", "hash weight payloads"},
              {"tensor", '\0', true, "NAME", "hash one tensor"},
              {"canonical", '\0', false, "", "hash canonical representation"}},
             {"nn hash model.onnx", "nn hash model.onnx --weights"}, "0 success", cmd_hash),
        make("diff", "nn diff [options] <old> <new>", "Semantic comparison of two models.",
             {"<old>", "<new>"},
             {{"summary", '\0', false, "", "summary only"},
              {"graph", '\0', false, "", "graph topology"},
              {"weights", '\0', false, "", "numeric weight comparison"},
              {"tensors", '\0', false, "", "tensor shapes/types"},
              {"ops", '\0', false, "", "operators"},
              {"quantization", '\0', false, "", "quantization"},
              {"metadata", '\0', false, "", "metadata"},
              {"numeric", '\0', false, "", "numeric weight stats"},
              {"structural", '\0', false, "", "structure only"},
              {"ignore-weights", '\0', false, "", "ignore weights"},
              {"ignore-metadata", '\0', false, "", "ignore metadata"},
              {"atol", '\0', true, "VALUE", "absolute tolerance"},
              {"rtol", '\0', true, "VALUE", "relative tolerance"}},
             {"nn diff old.onnx new.onnx", "nn diff a.onnx b.onnx --weights"},
             "0 no relevant difference; 1 differences found; 2+ error", cmd_diff),
        make("graph", "nn graph [options] <model>", "Export the compute graph (no GUI).", {"<model>"},
             {{"format", 'f', true, "dot|mermaid|json|text", "export format"},
              {"from", '\0', true, "NODE", "start node"},
              {"to", '\0', true, "NODE", "end node"},
              {"op", '\0', true, "TYPE", "filter op type"},
              {"depth", '\0', true, "N", "neighborhood depth"},
              {"collapse-activations", '\0', false, "", "collapse activations"},
              {"collapse-constants", '\0', false, "", "collapse constants"}},
             {"nn graph model.onnx --format dot"}, "0 success", cmd_graph),
        make("memory", "nn memory [options] <model>", "Analyze memory requirements.", {"<model>"},
             {{"timeline", '\0', false, "", "print lifetime timeline"},
              {"plan", '\0', false, "", "static arena plan"}},
             {"nn memory model.tflite", "nn memory model.onnx --plan"}, "0 success", cmd_memory),
        make("compute", "nn compute [options] <model>", "Estimate MACs/FLOPs.", {"<model>"},
             {{"per-node", '\0', false, "", "per-node costs"}},
             {"nn compute model.onnx"}, "0 success", cmd_compute),
        make("quant", "nn quant [options] <model>|compare <a> <b>", "Analyze quantization.",
             {"<model>"},
             {{"compare", '\0', false, "", "compare two models (see nn quant compare)"}},
             {"nn quant model.tflite", "nn quant compare float.onnx int8.onnx"}, "0 success",
             cmd_quant),
        make("sparsity", "nn sparsity [options] <model>", "Analyze weight sparsity.", {"<model>"},
             {{"threshold", '\0', true, "VALUE", "near-zero threshold"}},
             {"nn sparsity model.onnx --threshold 1e-6"}, "0 success", cmd_sparsity),
        make("lint", "nn lint [options] <model>", "Static correctness and quality checks.", {"<model>"},
             {}, {"nn lint model.onnx"}, "0 no errors; 9 if errors present", cmd_lint),
        make("run", "nn run [options] <model>", "Execute a model if a backend is available.",
             {"<model>"},
             {{"backend", '\0', true, "NAME", "runtime backend"},
              {"input", '\0', true, "NAME=FILE", "input tensor (npy/npz/csv/raw; repeatable)"},
              {"output", '\0', true, "FILE", "write primary output"},
              {"dump", '\0', true, "NAME", "dump intermediate"},
              {"dump-all", '\0', false, "", "dump all intermediates"},
              {"seed", '\0', true, "N", "RNG seed"},
              {"iterations", '\0', true, "N", "repeat count"}},
             {"nn run model.onnx --input input.npy", "nn run model.onnx --input tensors.npz"},
             "0 success; 7 backend unavailable; 8 execution failure", cmd_run),
        make("compare", "nn compare [options] <a> <b>", "Compare two models or backends on inputs.",
             {"<a>", "<b>"},
             {{"input", '\0', true, "NAME=FILE", "input tensor (npy/npz/csv/raw; repeatable)"},
              {"backend", '\0', true, "NAME", "backend for first"},
              {"backend2", '\0', true, "NAME", "backend for second"},
              {"activations", '\0', false, "", "intermediate activation comparison"},
              {"threshold", '\0', true, "VALUE", "divergence threshold"},
              {"atol", '\0', true, "VALUE", "absolute tolerance"},
              {"rtol", '\0', true, "VALUE", "relative tolerance"}},
             {"nn compare float.onnx quant.tflite --input test.npy",
              "nn compare a.onnx b.tflite --input tensors.npz --activations"}, "0 similar; 1 different",
             cmd_compare),
        make("test", "nn test [options] <model> <tests>", "Run deterministic test vectors.",
             {"<model>", "<tests>"}, {}, {"nn test model.onnx tests/"},
             "0 all pass; 9 failure", cmd_test),
        make("validate", "nn validate [options] <model> <dataset>",
             "Validate a model against a dataset or test cases.", {"<model>", "<dataset>"},
             {{"metric", '\0', true, "NAME", "accuracy|mse|mae|rmse"}},
             {"nn validate model.onnx tests/"}, "0 success; 9 validation failure", cmd_validate),
        make("regression", "nn regression [options] <old> <new> <tests>",
             "Compare old vs new against validation vectors with policy thresholds.",
             {"<old>", "<new>", "<tests>"},
             {{"max-accuracy-loss", '\0', true, "VALUE", "fail if accuracy drops more than VALUE"},
              {"max-memory-growth", '\0', true, "VALUE", "fail if RAM grows more than fraction"},
              {"max-latency-growth", '\0', true, "VALUE", "fail if latency grows more than fraction"},
              {"max-model-growth", '\0', true, "VALUE", "fail if file size grows more than fraction"}},
             {"nn regression old.onnx new.onnx tests/"}, "0 pass; 1 policy failure", cmd_regression),
        make("benchmark", "nn benchmark [options] <model>", "Benchmark inference latency.",
             {"<model>"},
             {{"warmup", '\0', true, "N", "warmup iterations"},
              {"iterations", '\0', true, "N", "timed iterations"},
              {"backend", '\0', true, "NAME", "runtime backend"},
              {"input", '\0', true, "FILE", "input tensor"}},
             {"nn benchmark model.onnx --iterations 50"},
             "0 success; 7 backend unavailable", cmd_benchmark),
        make("profile", "nn profile [options] <model>", "Per-operator runtime profile.", {"<model>"},
             {{"backend", '\0', true, "NAME", "runtime backend"},
              {"input", '\0', true, "FILE", "input tensor"}},
             {"nn profile model.onnx"}, "0 success; 7 if backend cannot profile", cmd_profile),
        make("compat", "nn compat [options] <model>", "Runtime/operator compatibility.", {"<model>"},
             {{"runtime", '\0', true, "NAME", "runtime name"},
              {"runtime-version", '\0', true, "VER", "capability table version"},
              {"target", '\0', true, "NAME", "hardware target (alias)"}},
             {"nn compat model.onnx --runtime onnxruntime"}, "0 compatible; 1 incompatible",
             cmd_compat),
        make("target", "nn target [options] <model>", "Deployability against a hardware profile.",
             {"<model>"},
             {{"target", '\0', true, "NAME", "built-in target name"},
              {"target-file", '\0', true, "FILE", "JSON target description"},
              {"accelerator", '\0', true, "FILE", "JSON accelerator description"}},
             {"nn target model.onnx --target cortex-m4f"}, "0 fits; 1 does not fit", cmd_target),
        make("partition", "nn partition [options] <model>",
             "Partition a graph onto an accelerator vs CPU.", {"<model>"},
             {{"accelerator", '\0', true, "FILE", "JSON accelerator description"}},
             {"nn partition model.onnx --accelerator my_npu.json"}, "0 success", cmd_partition),
        make("convert", "nn convert [options] <input>", "Conversion frontend (adapter-based).",
             {"<input>"},
             {{"to", '\0', true, "FORMAT", "destination format"},
              {"output", 'o', true, "FILE", "output path"},
              {"list", '\0', false, "", "list conversion routes"}},
             {"nn convert --list", "nn convert in.onnx --to tflite -o out.tflite"},
             "0 success; 7 conversion unavailable", cmd_convert),
        make("optimize", "nn optimize [options] <input>", "Safe graph optimizations.", {"<input>"},
             {{"output", 'o', true, "FILE", "write optimized model"},
              {"dry-run", '\0', false, "", "show proposed changes only"}},
             {"nn optimize model.onnx --dry-run"}, "0 success", cmd_optimize),
        make("extract", "nn extract [options] <model>", "Extract tensors, weights, or subgraphs.",
             {"<model>"},
             {{"tensor", '\0', true, "NAME", "tensor name"},
              {"from", '\0', true, "NODE", "subgraph start"},
              {"to", '\0', true, "NODE", "subgraph end"},
              {"output", 'o', true, "FILE", "output path"}},
             {"nn extract model.onnx --tensor conv1.weight -o conv1.npy"}, "0 success", cmd_extract),
        make("canonicalize", "nn canonicalize [options] <model>",
             "Write a stable canonical representation for hashing/diffing.", {"<model>"}, {},
             {"nn canonicalize model.onnx"}, "0 success", cmd_canonicalize),
        make("bisect", "nn bisect [options]", "Regression bisect over artifacts or git history.", {},
             {{"good", '\0', true, "PATH|COMMIT", "known good"},
              {"bad", '\0', true, "PATH|COMMIT", "known bad"},
              {"manifest", '\0', true, "FILE", "artifact sequence JSON"},
              {"git", '\0', false, "", "git mode"},
              {"model", '\0', true, "PATH", "model path in repo"},
              {"test", '\0', true, "CMD", "test command; {} is replaced with the artifact"}},
             {"nn bisect --good a.onnx --bad b.onnx --test './t.sh {}'"}, "0 found; 9 test error",
             cmd_bisect),
        make("doctor", "nn doctor", "Inspect installation and compiled features.", {}, {},
             {"nn doctor"}, "0 success", cmd_doctor),
        make("formats", "nn formats [name]", "Show registered format capabilities.",
             {"[name]  optional format name"}, {}, {"nn formats", "nn formats onnx"}, "0 success",
             cmd_formats),
        make("backends", "nn backends", "List runtime backends.", {}, {}, {"nn backends"}, "0 success",
             cmd_backends),
        make("targets", "nn targets", "List built-in hardware profiles.", {}, {}, {"nn targets"},
             "0 success", cmd_targets),
        make("config", "nn config [options] [key] [value]", "Git-like configuration.",
             {"[key]", "[value]"},
             {{"list", '\0', false, "", "list all"},
              {"get", '\0', true, "KEY", "get one"},
              {"unset", '\0', true, "KEY", "remove one"}},
             {"nn config --list", "nn config color.ui auto"}, "0 success", cmd_config),
        make("help", "nn help [command]", "Show help.", {"[command]"}, {}, {"nn help inspect"},
             "0 success", cmd_help),
        make("version", "nn version [--build-options]", "Show version.", {},
             {{"build-options", '\0', false, "", "compiler, commit, enabled modules"}},
             {"nn version --build-options"}, "0 success", cmd_version),
    };
    return k;
}

}  // namespace

const std::vector<CommandSpec>& all_commands() { return commands(); }

const CommandSpec* find_command(std::string_view name) {
    for (const auto& c : commands()) {
        if (c.name == name) {
            return &c;
        }
    }
    return nullptr;
}

int run_cli(int argc, char** argv) {
    auto parsed = parse_global_args(argc, argv);
    if (!parsed) {
        std::cerr << parsed.error().format() << "\n";
        return parsed.error().exit_status();
    }
    GlobalOptions g = std::move(parsed.value());
    if (g.verbose >= 2) {
        set_log_level(LogLevel::Trace);
    } else if (g.verbose == 1) {
        set_log_level(LogLevel::Info);
    }
    Printer p(g);

    if (g.version && g.command.empty()) {
        p.println("nn " + version_string());
        return kExitOk;
    }
    if (g.command.empty()) {
        print_top_help(p);
        return kExitOk;
    }
    if (g.command == "help" || (g.help && g.command.empty())) {
        return cmd_help(g);
    }
    const CommandSpec* spec = find_command(g.command);
    if (!spec) {
        p.errln("nn: unknown command '" + g.command + "'");
        p.errln("Try 'nn --help' for more information.");
        return kExitUsage;
    }
    auto carg = parse_command_args(g.args, *spec);
    if (!carg) {
        p.errln(carg.error().format());
        p.errln("");
        p.errln("usage: " + spec->synopsis);
        p.errln("");
        p.errln("Try 'nn " + spec->name + " --help' for more information.");
        return kExitUsage;
    }
    if (carg.value().help || g.help) {
        print_command_help(p, *spec);
        return kExitOk;
    }
    return spec->run(g);
}

}  // namespace nn
