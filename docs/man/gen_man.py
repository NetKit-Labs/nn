#!/usr/bin/env python3
"""Generate remaining nn man pages from command metadata."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

pages = {
    "ops": (
        "list operators in a model",
        "nn ops [options] <model>",
        "List operators, optionally grouped or filtered.",
        [
            ("--by-type", "Group by operator type."),
            ("--by-cost", "Sort by compute cost."),
            ("--by-memory", "Sort by memory."),
            ("--details", "Per-node details."),
            ("--unsupported", "List unknown/unrecognized ops."),
            ("--canonical", "Use canonical names."),
            ("--native", "Use native names."),
            ("--op TYPE", "Filter by operator type."),
        ],
        "0 success.",
        "nn ops model.onnx",
        "nn-inspect(1), nn-compute(1)",
    ),
    "tensors": (
        "list tensors in a model",
        "nn tensors [options] <model>",
        "List tensors with optional filters.",
        [
            ("--weights", "Constants only."),
            ("--activations", "Non-constants only."),
            ("--inputs", "Model inputs."),
            ("--outputs", "Model outputs."),
            ("--largest", "Sort by size descending."),
            ("--dtype TYPE", "Filter by dtype."),
            ("--name PATTERN", "Substring match on name."),
        ],
        "0 success.",
        "nn tensors model.onnx --largest",
        "nn-inspect(1), nn-io(1)",
    ),
    "io": (
        "show input/output contracts",
        "nn io [options] <model>",
        "Print model input and output names, shapes, and dtypes.",
        [],
        "0 success.",
        "nn io model.onnx",
        "nn-inspect(1), nn-run(1)",
    ),
    "metadata": (
        "show model metadata",
        "nn metadata [options] <model>",
        "Print format metadata only (producer, version, extra keys).",
        [],
        "0 success.",
        "nn metadata model.gguf",
        "nn-inspect(1)",
    ),
    "hash": (
        "hash a model artifact, graph, or tensor",
        "nn hash [options] <model>",
        "SHA-256 of the file, canonical graph, weights, or a named tensor.",
        [
            ("--graph", "Hash canonical graph."),
            ("--weights", "Hash weight payloads."),
            ("--tensor NAME", "Hash one tensor."),
            ("--canonical", "Hash canonical representation."),
        ],
        "0 success.",
        "nn hash model.onnx --weights",
        "nn-diff(1), nn-canonicalize(1)",
    ),
    "graph": (
        "export the compute graph",
        "nn graph [options] <model>",
        "Write the graph as text, DOT, mermaid, or JSON. No GUI.",
        [
            ("-f, --format FMT", "dot, mermaid, json, or text."),
            ("--from NODE", "Start node."),
            ("--to NODE", "End node."),
            ("--op TYPE", "Filter op type."),
            ("--depth N", "Neighborhood depth."),
            ("--collapse-activations", "Collapse activations."),
            ("--collapse-constants", "Collapse constants."),
        ],
        "0 success.",
        "nn graph model.onnx --format dot",
        "nn-inspect(1), nn-ops(1)",
    ),
    "compute": (
        "estimate MACs and FLOPs",
        "nn compute [options] <model>",
        "Static compute estimates from shapes and operators. Unknown ops are counted separately.",
        [("--per-node", "Per-node costs.")],
        "0 success.",
        "nn compute model.onnx",
        "nn-ops(1), nn-memory(1)",
    ),
    "quant": (
        "analyze quantization",
        "nn quant [options] <model>|compare <a> <b>",
        "Report quantized vs float tensors, Q/DQ nodes, and optional pairwise comparison.",
        [("--compare", "Compare two models (nn quant compare).")],
        "0 success.",
        "nn quant model.tflite",
        "nn-sparsity(1), nn-compare(1)",
    ),
    "sparsity": (
        "weight sparsity and prune-candidate hints",
        "nn sparsity [options] <model>",
        "Count zeros and near-zeros, join weights to layers, and rank conv/dense channels with small L1 against MAC share. Inspect only; does not rewrite the graph. Estimated savings are an upper bound.",
        [
            ("--threshold VALUE", "Absolute |w| near-zero threshold (default 0)."),
            ("--channel-frac VALUE", "Weak if channel L1 <= VALUE times the layer max (default 0.01)."),
        ],
        "0 success.",
        "nn sparsity model.onnx --threshold 1e-6",
        "nn-quant(1), nn-tensors(1), nn-compute(1)",
    ),
    "lint": (
        "static correctness and quality checks",
        "nn lint [options] <model>",
        "Graph and tensor sanity checks without executing the model.",
        [],
        "0 no errors; 9 if errors present.",
        "nn lint model.onnx",
        "nn-inspect(1), nn-validate(1)",
    ),
    "run": (
        "execute a model if a backend is available",
        "nn run [options] <model>",
        "Run inference using a compiled backend (onnxruntime, litert, or reference). "
        "Inputs may be npy, npz (ZIP-stored NumPy archive), csv, or raw. "
        "An npz with one array binds to a single input; multiple arrays match input names. "
        "--dump-all uses the reference backend when no --backend is given.",
        [
            ("--backend NAME", "Runtime backend."),
            ("--input NAME=FILE", "Input tensor; repeatable. FILE may be npy/npz/csv/raw."),
            ("--output FILE", "Write primary output."),
            ("--dump NAME", "Dump one intermediate."),
            ("--dump-all", "Dump all intermediates (reference backend)."),
            ("--seed N", "RNG seed for random fill."),
            ("--iterations N", "Repeat count."),
        ],
        "0 success; 7 backend unavailable; 8 execution failure.",
        "nn run model.onnx --input input.npy",
        "nn-compare(1), nn-backends(1)",
    ),
    "compare": (
        "compare two models or backends on inputs",
        "nn compare [options] <a> <b>",
        "Without --input, compare structure. With inputs, run both models (each bound independently "
        "so ONNX vs TFLite names may differ) and compare outputs by name, then by index. "
        "Rank-4 NCHW/NHWC pairs are permuted when shapes match that layout. "
        "--activations compares intermediates when a backend can dump them (reference); "
        "ORT and LiteRT typically cannot dump all activations.",
        [
            ("--input NAME=FILE", "Input tensor (npy/npz/csv/raw); repeatable."),
            ("--backend NAME", "Backend for the first model."),
            ("--backend2 NAME", "Backend for the second model."),
            ("--activations", "Compare intermediate activations when available."),
            ("--threshold VALUE", "Divergence threshold on max abs (default 1e-5)."),
            ("--atol VALUE", "Absolute tolerance."),
            ("--rtol VALUE", "Relative tolerance."),
        ],
        "0 similar; 1 different.",
        "nn compare float.onnx quant.tflite --input test.npy",
        "nn-diff(1), nn-run(1)",
    ),
    "test": (
        "run deterministic test vectors",
        "nn test [options] <model> <tests>",
        "Load a directory of npy files or a manifest.json of named inputs/expected outputs.",
        [],
        "0 all pass; 9 failure.",
        "nn test model.onnx tests/",
        "nn-validate(1), nn-run(1)",
    ),
    "validate": (
        "validate a model against a dataset",
        "nn validate [options] <model> <dataset>",
        "Score predictions against expected tensors using the chosen metric.",
        [("--metric NAME", "accuracy, mse, mae, or rmse.")],
        "0 success; 9 validation failure.",
        "nn validate model.onnx tests/",
        "nn-test(1), nn-regression(1)",
    ),
    "regression": (
        "compare old vs new against validation vectors",
        "nn regression [options] <old> <new> <tests>",
        "Fail if accuracy, memory, latency, or file size grow beyond policy thresholds.",
        [
            ("--max-accuracy-loss VALUE", "Fail if accuracy drops more than VALUE."),
            ("--max-memory-growth VALUE", "Fail if RAM grows more than this fraction."),
            ("--max-latency-growth VALUE", "Fail if latency grows more than this fraction."),
            ("--max-model-growth VALUE", "Fail if file size grows more than this fraction."),
        ],
        "0 pass; 1 policy failure.",
        "nn regression old.onnx new.onnx tests/",
        "nn-compare(1), nn-validate(1)",
    ),
    "benchmark": (
        "benchmark inference latency",
        "nn benchmark [options] <model>",
        "Warmup then timed iterations on a selected backend.",
        [
            ("--warmup N", "Warmup iterations."),
            ("--iterations N", "Timed iterations."),
            ("--backend NAME", "Runtime backend."),
            ("--input FILE", "Input tensor."),
        ],
        "0 success; 7 backend unavailable.",
        "nn benchmark model.onnx --iterations 50",
        "nn-profile(1), nn-run(1)",
    ),
    "profile": (
        "per-operator runtime profile",
        "nn profile [options] <model>",
        "Time operators when the backend can profile (reference). Other backends may report overall latency only.",
        [
            ("--backend NAME", "Runtime backend."),
            ("--input FILE", "Input tensor."),
        ],
        "0 success; 7 if backend cannot profile.",
        "nn profile model.onnx",
        "nn-benchmark(1), nn-ops(1)",
    ),
    "compat": (
        "runtime and operator compatibility",
        "nn compat [options] <model>",
        "Check operators against a runtime capability table.",
        [
            ("--runtime NAME", "Runtime name."),
            ("--runtime-version VER", "Capability table version."),
            ("--target NAME", "Hardware target (alias)."),
        ],
        "0 compatible; 1 incompatible.",
        "nn compat model.onnx --runtime onnxruntime",
        "nn-target(1), nn-backends(1)",
    ),
    "target": (
        "deployability against a hardware profile",
        "nn target [options] <model>",
        "Compare estimated RAM/compute against a built-in or JSON target profile.",
        [
            ("--target NAME", "Built-in target name."),
            ("--target-file FILE", "JSON target description."),
            ("--accelerator FILE", "JSON accelerator description."),
        ],
        "0 fits; 1 does not fit.",
        "nn target model.onnx --target cortex-m4f",
        "nn-targets(1), nn-memory(1)",
    ),
    "partition": (
        "partition a graph onto an accelerator vs CPU",
        "nn partition [options] <model>",
        "Assign nodes to accelerator or CPU from a JSON accelerator description.",
        [("--accelerator FILE", "JSON accelerator description.")],
        "0 success.",
        "nn partition model.onnx --accelerator my_npu.json",
        "nn-target(1), nn-compat(1)",
    ),
    "convert": (
        "conversion frontend",
        "nn convert [options] <input>",
        "Adapter-based conversion. Currently IR to ONNX is available; other routes report unavailable.",
        [
            ("--to FORMAT", "Destination format."),
            ("-o, --output FILE", "Output path."),
            ("--list", "List conversion routes."),
        ],
        "0 success; 7 conversion unavailable.",
        "nn convert --list",
        "nn-formats(1), nn-optimize(1)",
    ),
    "optimize": (
        "safe graph optimizations",
        "nn optimize [options] <input>",
        "Identity/Dropout rewiring, dead-node elimination, and simple constant folding.",
        [
            ("-o, --output FILE", "Write optimized model."),
            ("--dry-run", "Show proposed changes only."),
        ],
        "0 success.",
        "nn optimize model.onnx --dry-run",
        "nn-extract(1), nn-canonicalize(1)",
    ),
    "extract": (
        "extract tensors, weights, or subgraphs",
        "nn extract [options] <model>",
        "Write a tensor as npy or a subgraph as ONNX.",
        [
            ("--tensor NAME", "Tensor name."),
            ("--from NODE", "Subgraph start."),
            ("--to NODE", "Subgraph end."),
            ("-o, --output FILE", "Output path."),
        ],
        "0 success.",
        "nn extract model.onnx --tensor conv1.weight -o conv1.npy",
        "nn-tensors(1), nn-optimize(1)",
    ),
    "canonicalize": (
        "write a stable canonical representation",
        "nn canonicalize [options] <model>",
        "Print a weight-free canonical graph text suitable for hashing and diffing.",
        [],
        "0 success.",
        "nn canonicalize model.onnx",
        "nn-hash(1), nn-diff(1)",
    ),
    "bisect": (
        "regression bisect over artifacts or git history",
        "nn bisect [options]",
        "Find the first bad artifact or commit. {} in --test is replaced with the artifact path.",
        [
            ("--good PATH|COMMIT", "Known good."),
            ("--bad PATH|COMMIT", "Known bad."),
            ("--manifest FILE", "Artifact sequence JSON."),
            ("--git", "Git mode."),
            ("--model PATH", "Model path in repo."),
            ("--test CMD", "Test command."),
        ],
        "0 found; 9 test error.",
        "nn bisect --good a.onnx --bad b.onnx --test './t.sh {}'",
        "nn-regression(1), nn-diff(1)",
    ),
    "doctor": (
        "inspect installation and compiled features",
        "nn doctor",
        "Print enabled formats, runtimes, and build identity.",
        [],
        "0 success.",
        "nn doctor",
        "nn-version(1), nn-formats(1), nn-backends(1)",
    ),
    "formats": (
        "show registered format capabilities",
        "nn formats [name]",
        "List formats and whether they can read, expose a graph, weights, execute, or convert. "
        "Execute is yes only when a runtime was actually linked.",
        [],
        "0 success.",
        "nn formats onnx",
        "nn-backends(1), nn-doctor(1)",
    ),
    "backends": (
        "list runtime backends",
        "nn backends",
        "Print compiled backends and whether they are available at runtime.",
        [],
        "0 success.",
        "nn backends",
        "nn-run(1), nn-formats(1)",
    ),
    "targets": (
        "list built-in hardware profiles",
        "nn targets",
        "Names accepted by nn target --target.",
        [],
        "0 success.",
        "nn targets",
        "nn-target(1)",
    ),
    "config": (
        "git-like configuration",
        "nn config [options] [key] [value]",
        "Read or write user configuration keys such as color.ui.",
        [
            ("--list", "List all."),
            ("--get KEY", "Get one."),
            ("--unset KEY", "Remove one."),
        ],
        "0 success.",
        "nn config --list",
        "nn(1)",
    ),
    "help": (
        "show help",
        "nn help [command]",
        "Top-level help, or help for one command.",
        [],
        "0 success.",
        "nn help inspect",
        "nn(1)",
    ),
    "version": (
        "show version",
        "nn version [--build-options]",
        "Print the nn version string.",
        [("--build-options", "Compiler, commit, enabled modules.")],
        "0 success.",
        "nn version --build-options",
        "nn-doctor(1)",
    ),
}


def troff_escape(s: str) -> str:
    return s.replace("\\", "\\\\")


def render(cmd, meta):
    name, syn, desc, opts, exit_st, example, see = meta
    lines = [
        f'.TH NN-{cmd.upper()} 1 "2026" "nn 0.1.0" "NN Manual"',
        ".SH NAME",
        f"nn-{cmd} \\- {name}",
        ".SH SYNOPSIS",
        f".B {syn}",
        ".SH DESCRIPTION",
        troff_escape(desc),
    ]
    if opts:
        lines.append(".SH OPTIONS")
        for flag, help_ in opts:
            flag_esc = flag.replace("-", "\\-")
            lines.append(".TP")
            lines.append(f".B {flag_esc}")
            lines.append(troff_escape(help_))
    lines += [
        ".SH EXIT STATUS",
        exit_st,
        ".SH EXAMPLES",
        ".nf",
        example,
        ".fi",
        ".SH SEE ALSO",
        see,
        "",
    ]
    return "\n".join(lines)


def main():
    for cmd, meta in pages.items():
        path = ROOT / f"nn-{cmd}.1"
        path.write_text(render(cmd, meta))
        print(path.name)


if __name__ == "__main__":
    main()
