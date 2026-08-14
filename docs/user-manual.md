# nn user manual

Version 0.1.0. This manual describes the `nn` command-line toolkit as implemented. Flags and exit codes match `nn help <command>` and `src/cli/command.cpp`. Capabilities that are not compiled in are documented as unavailable, not as future promises.

Companion projects: [netkit](https://github.com/NetKit-Labs/netkit) (inference engine), [memkit](https://github.com/NetKit-Labs/memkit) (embedded containers). `nn` is the inspect / diff / hash / target / sparsity tool for model files, not a production runtime.

Captured command lines and output for `.onnx` and `.tflite`: [example-usage.md](example-usage.md).

- [1. What nn is](#1-what-nn-is)
- [2. Install and build](#2-install-and-build)
- [3. Invocation](#3-invocation)
- [4. Global options](#4-global-options)
- [5. Output formats](#5-output-formats)
- [6. Exit status](#6-exit-status)
- [7. Formats](#7-formats)
- [8. Execution backends](#8-execution-backends)
- [9. Tensor inputs and outputs](#9-tensor-inputs-and-outputs)
- [10. Commands](#10-commands)
- [11. Test manifests](#11-test-manifests)
- [12. Configuration](#12-configuration)
- [13. Hardware targets and accelerators](#13-hardware-targets-and-accelerators)
- [14. Scripting and CI](#14-scripting-and-ci)
- [15. Security](#15-security)
- [16. Troubleshooting](#16-troubleshooting)
- [17. Glossary](#17-glossary)

---

## 1. What nn is

`nn` is a single native executable with git-style subcommands. There is no GUI, web UI, or notebook interface.

Typical work:

```bash
nn inspect model.onnx
nn diff old.onnx new.onnx
nn memory kws.tflite --plan
nn sparsity kws.onnx --threshold 1e-6
nn target kws.onnx --target cortex-m4f
nn run model.onnx --input input.npy
nn compare float.onnx quant.tflite --input test.npy
```

A file is loaded into an in-memory intermediate representation (`ModelIR`): graphs, tensors, operators (native name plus a canonical kind), metadata, and hashes. Commands operate on that IR. Format-specific details that the reader did not extract are not invented.

`nn formats` is the capability table for **this build**. Do not assume a format can execute or convert unless that row says so.

---

## 2. Install and build

### Requirements

- C++20 compiler (GCC, Clang, or MSVC)
- CMake 3.20 or newer
- Network at configure time if ONNX Runtime and/or LiteRT are enabled (default)

### Build

```bash
git clone https://github.com/NetKit-Labs/nn.git
cd nn
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/nn doctor
```

On Windows, use a Release configuration for tests:

```bat
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

### Install

```bash
cmake --install build
```

This installs the `nn` binary and public headers under `include/nn`.

```bash
nn help inspect
nn inspect --help
```

### CMake options

| Option | Default | Meaning |
| --- | --- | --- |
| `NN_ENABLE_ONNX` | ON | ONNX reader |
| `NN_ENABLE_TFLITE` | ON | TFLite / LiteRT reader |
| `NN_ENABLE_GGUF` | ON | GGUF reader |
| `NN_ENABLE_SAFETENSORS` | ON | SafeTensors reader |
| `NN_ENABLE_PYTORCH` | ON | PyTorch zip/metadata inspection (no pickle) |
| `NN_ENABLE_EXECUTORCH` | ON | ExecuTorch `.pte` inspection |
| `NN_ENABLE_COREML` | ON | Core ML inspection |
| `NN_ENABLE_OPENVINO` | ON | OpenVINO IR inspection |
| `NN_ENABLE_TENSORRT` | ON | TensorRT engine metadata |
| `NN_ENABLE_NCNN` | ON | NCNN inspection |
| `NN_ENABLE_MNN` | ON | MNN inspection |
| `NN_ENABLE_TFJS` | ON | TensorFlow.js inspection |
| `NN_ENABLE_LEGACY` | ON | Caffe, Darknet, MXNet, Paddle |
| `NN_ENABLE_KERAS` | ON | Keras / HDF5 inspection |
| `NN_ENABLE_TENSORFLOW` | ON | SavedModel inspection |
| `NN_ENABLE_ONNXRUNTIME` | ON | Download/link ONNX Runtime |
| `NN_ENABLE_LITERT_RUNTIME` | ON | Download/link LiteRT |
| `NN_REQUIRE_RUNTIMES` | OFF | Fail configure if a requested runtime cannot be fetched |
| `NN_ENABLE_WERROR` | ON | Treat warnings as errors |
| `NN_BUILD_TESTS` | ON | Unit and integration tests |
| `NN_BUILD_FUZZERS` | OFF | libFuzzer targets (Clang) |
| `NN_SANITIZER` | (empty) | `address`, `undefined`, or `address,undefined` |

Example: analysis-only build with no downloaded runtimes:

```bash
cmake -S . -B build \
  -DNN_ENABLE_ONNXRUNTIME=OFF \
  -DNN_ENABLE_LITERT_RUNTIME=OFF
```

Sanitizer build:

```bash
cmake -S . -B build-asan -DNN_SANITIZER=address,undefined
cmake --build build-asan
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir build-asan --output-on-failure
```

Environment overrides:

| Variable | Meaning |
| --- | --- |
| `NN_ONNXRUNTIME_ROOT` | Existing ONNX Runtime SDK (must contain `include/onnxruntime_cxx_api.h`) |
| `NN_LITERT_ROOT` | Existing LiteRT C SDK root |
| `NN_LITERT_LIB` | Path to `libLiteRt` shared library |

---

## 3. Invocation

```text
nn [global options] <command> [command options] [args]
```

Global options may appear before the command. Command options follow the command name.

```bash
nn --json inspect model.onnx
nn inspect model.onnx --summary
nn --porcelain hash model.onnx --graph
nn -v run model.onnx --backend onnxruntime --input x.npy
```

Help:

```bash
nn
nn --help
nn help
nn help inspect
nn inspect --help
nn --version
nn version --build-options
```

Unknown commands and unknown options exit **2**.

---

## 4. Global options

| Flag | Meaning |
| --- | --- |
| `-h`, `--help` | Show help (top-level or, after a command, that command) |
| `--version` | Print `nn <version>` when no command is given |
| `-v`, `--verbose` | Increase verbosity (`-vv` is trace) |
| `-q`, `--quiet` | Suppress ordinary text output |
| `--json` | JSON on stdout |
| `--yaml` | YAML on stdout |
| `--porcelain` | Tab-separated `key<TAB>value` lines |
| `--color=auto\|always\|never` | Color (also `--color auto`) |
| `--output FILE`, `-o FILE` | Redirect printer output to a file |
| `--threads N` | Thread hint for runtimes that honor it |

Commands that own `-o` / `--output` (`convert`, `optimize`, `extract`, `run`) treat that flag as **their** output path, not a redirect of the human-readable report.

`--json`, `--yaml`, and `--porcelain` are mutually intended as one output mode. The last one parsed wins.

Not every command implements JSON. Commands that do include `schema_version` (currently `1`). See [json-schema.md](json-schema.md). Additive keys are allowed; breaking field changes bump `schema_version`.

---

## 5. Output formats

**Text (default).** Headings and `key:  value` lines. Suitable for terminals.

**JSON.** Pretty-printed object. Use with `jq`:

```bash
nn --json inspect model.onnx | jq -r .format
nn --json formats | jq '.formats[] | select(.execute==true)'
nn --json hash model.onnx | jq -r .artifact
nn --json sparsity model.onnx | jq '.layers[] | select(.score > 0)'
```

**YAML.** Same structure as JSON where the command supports structured output.

**Porcelain.** Stable `key<TAB>value` for scripts that should not parse aligned columns:

```bash
nn --porcelain hash model.onnx
# artifact:	<64 hex chars>
```

Redirect a text report:

```bash
nn --output report.txt inspect model.onnx
```

---

## 6. Exit status

Codes are stable. Scripts should branch on these integers, not on English text. Full table: [exit-status.md](exit-status.md).

| Code | Meaning | Typical commands |
| --- | --- | --- |
| 0 | Success | Most commands |
| 1 | Difference or policy failure | `diff`, `compare`, `compat`, `target`, `regression` |
| 2 | Usage error | Missing args, unknown option |
| 3 | File not found / unreadable | Any loader |
| 4 | Malformed model | Parsers |
| 5 | Unsupported format | Probe failed |
| 6 | Unsupported operator | Reference backend / some execute paths |
| 7 | Backend or conversion adapter unavailable | `run`, `benchmark`, `profile`, `convert` |
| 8 | Execution failure | `run` |
| 9 | Validation / lint errors | `lint`, `test`, `validate`, `bisect` test errors |
| 10 | Internal error | Should not happen |

`nn diff` and `nn compare` returning **1** is a successful comparison that found a difference, not a crash.

---

## 7. Formats

`nn formats` prints the compiled capability matrix. `nn formats <name>` prints one row.

| Format | Typical files | Graph | Weights | Execute | Convert |
| --- | --- | --- | --- | --- | --- |
| ONNX | `.onnx` | yes | yes | yes if ONNX Runtime linked | rewrite as ONNX |
| TFLite / LiteRT | `.tflite` | yes | yes | yes if LiteRT linked | no writer |
| GGUF | `.gguf` | tensors / metadata | yes | no | no |
| SafeTensors | `.safetensors` | no | yes | no | no |
| PyTorch | `.pt`, `.pth`, `.pt2` | zip member names | — | no | pickle refused |
| ExecuTorch | `.pte` | inspection | varies | no | no |
| Core ML | `.mlmodel`, `.mlpackage` | inspection | varies | no | no |
| OpenVINO | XML/bin IR | inspection | varies | no | no |
| TensorRT | engine metadata | inspection | — | no | no |
| NCNN / MNN / TFJS | format-specific | inspection | varies | no | no |
| Keras / HDF5 | `.h5`, `.keras` | inspection | varies | no | no |
| SavedModel | directory | inspection | varies | no | no |
| Caffe / Darknet / MXNet / Paddle | legacy | inspection | varies | no | no |

**Execute** is `yes` only when that runtime was actually linked at compile time (`NN_HAS_ONNXRUNTIME` / `NN_HAS_LITERT`). A format reader without a backend never claims Execute.

SafeTensors and GGUF do not grow a fake compute graph. Weight-only artifacts can still be hashed, listed as tensors, and have sparsity/quant stats where payloads exist.

PyTorch pickle payloads are **not** deserialized. Zip-based archives can be listed; unsafe load is refused.

---

## 8. Execution backends

`nn backends` lists compiled backends and whether they are available.

| Name | Selected when | Role |
| --- | --- | --- |
| `onnxruntime` | default for `.onnx` | Official ONNX Runtime |
| `litert` | default for `.tflite` | LiteRT C API, CPU only |
| `reference` | fallback; also when `--dump-all` / `--activations` and no `--backend` | Small in-tree float32 interpreter |

Selection: if `--backend NAME` is set, that backend is required. Otherwise the first registered backend that `available() && supports(model)` wins (ORT, then LiteRT, then reference). If dump-all or named dumps are requested and `--backend` is omitted, **reference** is preferred so intermediates exist.

### ONNX Runtime

Enabled with `NN_ENABLE_ONNXRUNTIME` (default ON). CMake downloads a platform SDK if needed. Default for ONNX graphs.

### LiteRT

Enabled with `NN_ENABLE_LITERT_RUNTIME` (default ON). Downloads the LiteRT C SDK and a prebuilt `libLiteRt` for `macos_arm64`, `linux_x86_64`, `linux_arm64`, or `windows_x86_64`. There is **no** macOS x86_64 prebuilt; configure warns and skips the backend.

CPU only. LiteRT may still print an XNNPACK delegate info line to stderr.

### Reference interpreter

Supports a slice of float32 ops, including: Add, Sub, Mul, Div, Max, Min, Pow, Relu, Relu6, Tanh, Sigmoid, Abs, Neg, Clip, Softmax, Identity, Dropout, Reshape, Squeeze, Unsqueeze, Flatten, Transpose, Concat, MatMul, Gemm, Conv/Conv2d, Constant.

It does **not** claim pooling, BatchNorm/LayerNorm, attention, Gather/Slice/Pad/Resize, recurrent ops, quantized/int8 graphs, or multiple outputs per node. Production models should use ORT or LiteRT. Reference exists for tiny graphs, tests, and activation dumps.

`--threads` is a hint. The reference backend does not parallelize kernels.

---

## 9. Tensor inputs and outputs

Commands that run a model (`run`, `compare`, `test`, `validate`, `benchmark`, `profile`, `regression`) bind inputs with `--input`.

### `--input` forms

```text
--input FILE
--input NAME=FILE
```

Repeat the flag for multiple tensors:

```bash
nn run add.onnx --input input0=a.npy --input input1=b.npy
```

If the model has a single input, a bare filename is bound to that input.

If names differ, matching uses (in order): exact name, file stem, first unbound input.

### File types

| Extension | Meaning |
| --- | --- |
| `.npy` | NumPy ndarray (native endian descrs as implemented) |
| `.npz` | Zip of `.npy` members, **stored** (uncompressed). This is `numpy.savez`, not `numpy.savez_compressed` |
| `.csv`, `.txt` | Numeric CSV; one row → 1-D, several rows → 2-D |
| `.raw`, `.bin` | Opaque bytes (uint8); reinterpreted as float32 if the byte size matches the model input |

A multi-array npz binds members to model inputs by name (then by order if counts match). A single-array npz binds to the remaining input. Extra arrays are ignored once every input is bound.

```python
import numpy as np
np.savez("tensors.npz", input0=a, input1=b)   # OK
# np.savez_compressed(...)                    # nn will reject deflate members
```

```bash
nn run model.onnx --input tensors.npz
nn run model.onnx --input x=serving.npz
```

If no `--input` is given and shapes are static, tensors are filled with **zeros**. `--seed N` on `nn run` fills float32 inputs with a deterministic random field instead.

### Writing tensors

`nn run --output out.npy` writes the **primary** output. Extension selects npy (default), csv, or raw/bin.

`--dump NAME` / `--dump-all` write intermediate `.npy` files next to `--output` (or the current directory). Only the **reference** backend dumps intermediates. ORT and LiteRT do not.

---

## 10. Commands

Each subsection lists synopsis, options, behavior, examples, and exit status. Online help: `nn help <command>` or `nn <command> --help`.

### inspect

```text
nn inspect [options] <model>
```

Summarize format, hashes, graph counts, parameters, and optional sections.

| Option | Meaning |
| --- | --- |
| `--summary` | Short summary only |
| `--all` | Include every section |
| `--metadata` | Metadata |
| `--inputs` / `--outputs` | I/O tensors |
| `--ops` | Operators |
| `--tensors` | Tensors |
| `--weights` | Weight tensors |
| `--quantization` | Quantization summary |
| `--subgraphs` | Subgraphs |
| `--raw` | Raw format metadata |

```bash
nn inspect model.onnx
nn inspect model.onnx --json
nn inspect model.onnx --all
```

Exit: 0 success; 3 file; 4 malformed; 5 unsupported format.

### ops

```text
nn ops [options] <model>
```

| Option | Meaning |
| --- | --- |
| `--by-type` | Group by operator type |
| `--by-cost` | Sort by estimated compute |
| `--by-memory` | Sort by memory |
| `--details` | Per-node details |
| `--unsupported` | Unknown / unrecognized ops |
| `--canonical` | Canonical names (Conv, MatMul, …) |
| `--native` | Native names (ONNX `Conv`, TFLite `CONV_2D`, …) |
| `--op TYPE` | Filter |

```bash
nn ops model.onnx --canonical
nn ops model.onnx --op Conv
```

### tensors

```text
nn tensors [options] <model>
```

| Option | Meaning |
| --- | --- |
| `--weights` | Constants only |
| `--activations` | Non-constants |
| `--inputs` / `--outputs` | Model I/O |
| `--largest` | Sort by size descending |
| `--dtype TYPE` | Filter (e.g. `float32`, `int8`) |
| `--name PATTERN` | Substring on the tensor name |

```bash
nn tensors model.onnx --largest
nn tensors model.onnx --weights --dtype float32
```

### io

```text
nn io <model>
```

Print input and output names, shapes, and dtypes. Shapes use `x` (for example `1x3x224x224`); dynamic dims appear as `?` or a symbol.

```bash
nn io model.onnx
nn --json io model.onnx
```

### metadata

```text
nn metadata <model>
```

Producer, versions, and extra metadata keys only.

```bash
nn metadata model.gguf
```

### hash

```text
nn hash [options] <model>
```

SHA-256 of the file and/or derived content.

| Option | Meaning |
| --- | --- |
| *(none)* | Artifact, graph, and weights |
| `--graph` | Canonical graph (no weights) |
| `--weights` | Weight payloads |
| `--tensor NAME` | One tensor payload |
| `--canonical` | Canonical graph text hash |

```bash
nn hash model.onnx
nn --json hash model.onnx
nn --porcelain hash model.onnx --graph
```

Graph hashes ignore weight bytes and sort structurally, so two files can share a graph hash with different weights.

### diff

```text
nn diff [options] <old> <new>
```

Semantic comparison: topology, tensors, ops, quantization, metadata, optional numeric weights.

| Option | Meaning |
| --- | --- |
| `--summary` | Summary only |
| `--graph` | Topology |
| `--weights` / `--numeric` | Numeric weight stats |
| `--tensors` | Shapes and types |
| `--ops` | Operators |
| `--quantization` | Quantization |
| `--metadata` | Metadata |
| `--structural` | Structure only |
| `--ignore-weights` | Skip weights |
| `--ignore-metadata` | Skip metadata |
| `--atol VALUE` | Absolute tolerance (default `1e-5`) |
| `--rtol VALUE` | Relative tolerance (default `1e-5`) |

Exit **0** if no relevant difference; **1** if differences exist.

```bash
nn diff v1.onnx v2.onnx
nn diff v1.onnx v2.onnx --weights --atol 1e-4
```

### graph

```text
nn graph [options] <model>
```

Export the compute graph. No GUI.

| Option | Meaning |
| --- | --- |
| `-f`, `--format FMT` | `dot`, `mermaid`, `json`, or `text` |
| `--from NODE` / `--to NODE` | Subgraph endpoints |
| `--op TYPE` | Filter |
| `--depth N` | Neighborhood depth |
| `--collapse-activations` | Collapse activations |
| `--collapse-constants` | Collapse constants |

`--format json` (or global `--json`) prints `{ "schema_version": 1, "nodes": [ { "id", "name", "op" } ] }`.

```bash
nn graph model.onnx --format dot > model.dot
nn graph model.onnx --format mermaid
nn --json graph model.onnx
```

### memory

```text
nn memory [options] <model>
```

Weight storage, activation lifetimes, peak RAM. Estimates are labeled as such.

| Option | Meaning |
| --- | --- |
| `--timeline` | Tensor lifetime timeline |
| `--plan` | Static arena plan (offsets) |

```bash
nn memory model.tflite
nn memory model.onnx --plan
```

### compute

```text
nn compute [options] <model>
```

Static MAC / FLOP estimates from shapes and operators. Unknown ops are counted separately.

```bash
nn compute model.onnx
nn compute model.onnx --per-node
```

### quant

```text
nn quant [options] <model>
nn quant compare <a> <b>
```

Counts quantized vs float tensors, per-channel vs per-tensor, Q/DQ nodes.

`nn quant compare` compares those counts between two models. Per-layer activation error is **not** computed unless you use `nn compare` with inputs.

```bash
nn quant model.tflite
nn quant compare float.onnx int8.onnx
nn --json quant model.tflite
```

### sparsity

```text
nn sparsity [options] <model>
```

Zeros, near-zeros, and prune-candidate hints for constant tensors. Inspect only: `nn` does not rewrite the graph.

Each constant is joined to the Conv / Gemm / MatMul that consumes it. For conv (`Co`) and dense (`out`) weights, channels whose L1 is at most `--channel-frac` (default **0.01**) times that layer’s max channel L1 are **weak**. Rows are sorted by `weak_channel_frac × mac_share`.

`--threshold` is the absolute `|w|` near-zero line and is independent of the channel cutoff.

Estimated saved bytes/MACs assume those weak channels are dropped. That figure is an **upper bound**: residual Add, Concat, and depthwise→pointwise couples make the real save smaller. Magnitude is a where-to-look hint, not an accuracy claim.

| Option | Meaning |
| --- | --- |
| `--threshold VALUE` | Count `|w| <= VALUE` as near-zero (default 0) |
| `--channel-frac VALUE` | Weak if channel L1 `<= VALUE * max` (default 0.01) |

`--json` prints `schema_version` 1 with a `layers` array. Field list: [json-schema.md](json-schema.md).

```bash
nn sparsity model.onnx --threshold 1e-6
nn --json sparsity model.onnx --channel-frac 0.05
```

### lint

```text
nn lint <model>
```

Static graph/tensor checks without executing. Exit **0** if no errors; **9** if errors are present. Warnings do not by themselves force 9 unless they are classified as errors.

```bash
nn lint model.onnx
```

### run

```text
nn run [options] <model>
```

Execute if a backend supports the model.

| Option | Meaning |
| --- | --- |
| `--backend NAME` | `onnxruntime`, `litert`, or `reference` |
| `--input NAME=FILE` | Repeatable; npy/npz/csv/raw |
| `--output FILE` | Write primary output tensor |
| `--dump NAME` | Dump one intermediate (reference) |
| `--dump-all` | Dump all intermediates (selects reference unless `--backend` is set) |
| `--seed N` | Random float32 fill |
| `--iterations N` | Repeat count (last result is reported) |

```bash
nn run model.onnx --input input.npy
nn run model.onnx --input tensors.npz --output out.npy
nn run model.tflite --backend litert
nn run model.onnx --dump-all --output /tmp/out.npy
nn --json run model.onnx --input x.npy
```

JSON includes `schema_version`, `backend`, `latency_ms`, and output name/dtype/shape/byte size (not the payload).

Exit: 0 success; 7 backend unavailable; 6 unsupported operator; 8 execution failure.

### compare

```text
nn compare [options] <a> <b>
```

Without `--input` and without `--activations`, this is a **structural** `diff` (identical or not).

With inputs (or `--activations`), both models are **run**. Inputs are bound **independently** on each graph so ONNX `Input3` and TFLite `serving_default_x:0` can still receive the same file. Outputs are matched by normalized name, then by index. Rank-4 NCHW vs NHWC is permuted when the dimension lists match that layout.

| Option | Meaning |
| --- | --- |
| `--input NAME=FILE` | Shared input files; bound per model |
| `--backend NAME` | Backend for the first model |
| `--backend2 NAME` | Backend for the second (default: same as `--backend`) |
| `--activations` | Compare intermediates when dumps exist |
| `--threshold VALUE` | Fail if `max_abs` exceeds this (default `1e-5`) |
| `--atol` / `--rtol` | Numeric tolerances (default `1e-5`) |

`--activations`: only backends that dump intermediates participate (reference). If ORT/LiteRT dump nothing, `nn` says so instead of fabricating tensors.

Exit **0** similar; **1** different.

```bash
nn compare a.onnx b.onnx
nn compare float.onnx quant.tflite --input test.npy
nn compare a.onnx a.onnx --backend onnxruntime --backend2 reference --input x.npy
nn compare a.onnx b.onnx --activations --input x.npy
```

Name normalization strips path components, `serving_default_` / `StatefulPartitionedCall_` prefixes, and `:0`-style suffixes, then compares alphanumerics case-insensitively.

### test

```text
nn test <model> <tests>
```

`<tests>` is a directory or a `manifest.json`. See [Test manifests](#11-test-manifests). Default numeric tolerance is `1e-5` absolute and relative.

Exit **0** all pass; **9** if any case fails.

```bash
nn test model.onnx tests/
nn test model.onnx tests/manifest.json
```

### validate

```text
nn validate [options] <model> <dataset>
```

Same manifest loader as `test`. Aggregates a metric over cases.

| Option | Meaning |
| --- | --- |
| `--metric NAME` | `accuracy` (argmax agree), `mse`, `mae`, or `rmse` |

Exit **0** success; **9** validation failure.

```bash
nn validate model.onnx tests/ --metric mse
```

### regression

```text
nn regression [options] <old> <new> <tests>
```

Compares estimated RAM and file size always. If the test path is executable, also compares MSE and latency.

| Option | Meaning |
| --- | --- |
| `--max-accuracy-loss VALUE` | Fail if new MSE − old MSE exceeds VALUE |
| `--max-memory-growth VALUE` | Fail if estimated RAM grows by more than this **fraction** (0.1 = +10%) |
| `--max-latency-growth VALUE` | Same for measured latency |
| `--max-model-growth VALUE` | Same for file size |

Exit **0** pass; **1** policy failure.

```bash
nn regression old.onnx new.onnx tests/ \
  --max-memory-growth 0.1 \
  --max-model-growth 0.05
```

### benchmark

```text
nn benchmark [options] <model>
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--warmup N` | 5 | Untimed iterations |
| `--iterations N` | 20 | Timed iterations |
| `--backend NAME` | auto | Backend |
| `--input FILE` | zeros | Input tensor |

Prints mean, min, p50, p90, p99, max latency in milliseconds.

```bash
nn benchmark model.onnx --iterations 50 --backend onnxruntime
```

Exit **7** if no backend can run the model.

### profile

```text
nn profile [options] <model>
```

Per-operator timings. The **reference** backend fills a profile. ORT/LiteRT typically cannot; then `nn` exits **7** (`backend cannot expose operator timings`).

```bash
nn profile model.onnx --backend reference --input x.npy
```

### compat

```text
nn compat [options] <model>
```

Check operators against a runtime capability table.

| Option | Default | Meaning |
| --- | --- | --- |
| `--runtime NAME` | `reference` | `reference`, `onnxruntime`, … |
| `--runtime-version VER` | (table default) | Table version |
| `--target NAME` | | Hardware target alias |

Exit **0** compatible; **1** incompatible.

```bash
nn compat model.onnx --runtime onnxruntime
nn compat model.onnx --runtime reference
```

### target

```text
nn target [options] <model>
```

Compare model size and estimated RAM against a hardware profile. Default target if omitted: `cortex-m4f`.

| Option | Meaning |
| --- | --- |
| `--target NAME` | Built-in name (`nn targets`) |
| `--target-file FILE` | JSON profile |
| `--accelerator FILE` | JSON accelerator (also used by `partition`) |

Exit **0** if the model fits; **1** if it does not.

```bash
nn target model.onnx --target cortex-m4f
nn target model.onnx --target esp32-s3
nn target model.onnx --target-file myboard.json
```

Built-in names: `cortex-m0+`, `cortex-m3`, `cortex-m4`, `cortex-m4f`, `cortex-m7`, `cortex-m33`, `cortex-m55`, `cortex-m85`, `cortex-a53`, `cortex-a55`, `cortex-a72`, `cortex-a76`, `riscv-mcu`, `riscv-linux`, `esp32`, `esp32-s3`, `apple-silicon`, `x86-64`.

These are **budget sketches** (RAM/flash/FPU), not cycle-accurate simulators.

### partition

```text
nn partition --accelerator FILE <model>
```

Assign nodes to `accelerator` vs `cpu` from a JSON accelerator description.

```bash
nn partition model.onnx --accelerator npu.json
```

### convert

```text
nn convert [options] <input>
nn convert --list
```

| Option | Meaning |
| --- | --- |
| `--to FORMAT` | Destination (default `onnx`) |
| `-o`, `--output FILE` | Output path (required when converting) |
| `--list` | Print routes |

**Available now:** any loaded graph IR → ONNX (`write_onnx_model`). Native TFLite op names are **not** rewritten to ONNX `Add`/`Conv`, so a TFLite graph dumped as ONNX may not run on ORT.

Unavailable (exit **7**): `onnx→tflite`, `onnx→coreml`, `pytorch→onnx`, `savedmodel→tflite`, `tflite→onnx` (as a dedicated adapter), `keras→onnx`.

```bash
nn convert --list
nn convert model.onnx --to onnx -o copy.onnx
```

### optimize

```text
nn optimize [options] <input>
```

Safe rewrites: Identity/Dropout rewiring, dead-node elimination, simple constant folding.

| Option | Meaning |
| --- | --- |
| `--dry-run` | Print proposed changes only |
| `-o`, `--output FILE` | Write optimized graph as ONNX |

Without `-o`, changes are listed and nothing is written.

```bash
nn optimize model.onnx --dry-run
nn optimize model.onnx -o opt.onnx
```

### extract

```text
nn extract [options] <model>
```

`-o` / `--output` is required.

| Option | Meaning |
| --- | --- |
| `--tensor NAME` | Write that tensor (`npy` / `csv` / `raw` by extension) |
| `--from NODE` / `--to NODE` | Subgraph; written as ONNX |

```bash
nn extract model.onnx --tensor conv1.weight -o conv1.npy
nn extract model.onnx --from conv1 --to relu1 -o slice.onnx
```

### canonicalize

```text
nn canonicalize <model>
```

Print a stable, weight-free graph text (sorted inputs/outputs/nodes). Useful as input to `diff` or external hash tools. Related: `nn hash --canonical`.

```bash
nn canonicalize model.onnx
```

### bisect

```text
nn bisect [options]
```

Find the first bad artifact or git commit. `nn` does **not** modify your working tree. Git mode uses a detached worktree under the temp directory and removes it afterward.

`--test` is required. `{}` is replaced with a shell-quoted artifact path.

| Option | Meaning |
| --- | --- |
| `--good PATH\|COMMIT` | Known good |
| `--bad PATH\|COMMIT` | Known bad |
| `--manifest FILE` | JSON `{ "artifacts": [ ... ] }` or `{ "sequence": [ ... ] }` |
| `--git` | Git mode |
| `--model PATH` | Model path inside the repo (git mode) |
| `--test CMD` | Command; exit 0 = good |

Artifact mode:

```bash
nn bisect --good a.onnx --bad b.onnx --test './check.sh {}'
nn bisect --manifest seq.json --test './check.sh {}'
```

Git mode (run from a git repository):

```bash
nn bisect --git --good abcdef0 --bad deadbee \
  --model models/net.onnx --test './check.sh {}'
```

Exit **0** when the first bad item is printed; **9** if the test command cannot run.

### doctor

```text
nn doctor
```

OS, compiler, git commit, enabled formats, and linked runtimes. Use this to see whether ORT/LiteRT actually compiled in.

```bash
nn doctor
```

### formats

```text
nn formats
nn formats <name>
```

```bash
nn formats
nn formats onnx
nn --json formats
```

JSON: `{ "schema_version": 1, "formats": [ { "name", "display", "read", "graph", "weights", "execute", "convert", "notes" } ] }`.

### backends

```text
nn backends
```

Name, availability, version.

### targets

```text
nn targets
```

Built-in names for `nn target --target`.

### config

```text
nn config --list
nn config <key>
nn config <key> <value>
```

User file: `$XDG_CONFIG_HOME/nn/config` or `~/.config/nn/config` (Windows: `%APPDATA%\nn\config`).

Repo overlay (read-only merge, higher precedence): `./.nnconfig` in the current directory.

Format: `key=value` lines. `#` and `;` comments. Setting a key writes the **user** file.

```bash
nn config --list
nn config color.ui
nn config color.ui auto
```

Do not store secrets in these files.

### help / version

```bash
nn help
nn help run
nn version
nn version --build-options
```

`--build-options` prints commit, compiler, OS, arch, enabled formats, and enabled runtimes.

---

## 11. Test manifests

Used by `nn test`, `nn validate`, and `nn regression`.

### Directory shorthand

If `<tests>` is a directory **without** `manifest.json`, `nn` looks for:

```text
tests/input.npy
tests/expected.npy
```

and binds them as `input` / `output`.

### `manifest.json`

```json
{
  "tests": [
    {
      "name": "add",
      "inputs": {
        "input0": "a.npy",
        "input1": "b.npy"
      },
      "expected": {
        "output": "exp.npy"
      }
    }
  ]
}
```

- Paths are relative to the manifest directory unless absolute.
- `"input": "file.npy"` is a shorthand for a single input named `input`.
- `"outputs"` is accepted as an alias of `"expected"`.
- If `expected` is omitted, `nn test` still runs inference and reports `PASS (executed; no expected tensors)`.

A directory that contains `manifest.json` is treated as that file.

---

## 12. Configuration

See [`config`](#config). Typical keys are tool preferences such as `color.ui`. Command-line flags always override config for that invocation when they apply.

---

## 13. Hardware targets and accelerators

### Target JSON (`--target-file`)

Required keys:

```json
{
  "name": "my-mcu",
  "cpu": "Cortex-M4F",
  "clock_hz": 168000000,
  "ram_bytes": 262144,
  "flash_bytes": 1048576,
  "simd": "DSP",
  "fpu": "FPv4"
}
```

`storage_bytes` may be used instead of `flash_bytes`. Empty `fpu` means no FPU; native types then stay integer-only.

`nn target` checks file size vs flash and estimated activation/scratch/overhead vs RAM. It is a packing check, not a guarantee that a vendor runtime will fit.

### Accelerator JSON (`--accelerator`)

```json
{
  "name": "my-npu",
  "supported_ops": ["Conv", "Add", "Relu"],
  "data_types": ["int8", "float32"],
  "macs_per_cycle": 64,
  "clock_hz": 400000000,
  "local_memory_bytes": 1048576,
  "alignment": 16
}
```

`nn partition` places nodes whose canonical/native names appear in `supported_ops` on `accelerator`, others on `cpu`.

---

## 14. Scripting and CI

Prefer `--json` or `--porcelain` in pipelines.

```bash
set -e
fmt=$(nn --json inspect model.onnx | jq -r .format)
hash=$(nn --porcelain hash model.onnx --graph | cut -f2)
nn lint model.onnx
nn diff baseline.onnx model.onnx
status=$?
if [ "$status" -eq 1 ]; then
  echo "models differ"
elif [ "$status" -ne 0 ]; then
  exit "$status"
fi
```

GitHub Actions does not run on push or pull requests. Start a run from the Actions tab or `gh workflow run ci`. All jobs are Linux (`ubuntu-24.04`): GCC and Clang builds with `NN_REQUIRE_RUNTIMES=ON`, Clang ASan/UBSan, and a libFuzzer smoke. macOS and Windows are not in CI.

Locally:

```bash
cmake -S . -B build -DNN_ENABLE_WERROR=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

---

## 15. Security

Model files are untrusted input.

- Parsers bounds-check offsets and lengths.
- Tensor dimensions and allocations are overflow-checked.
- PyTorch pickle is not deserialized.
- Embedded code in artifacts is never executed.
- `nn bisect --git` uses a throwaway worktree; it still runs **your** `--test` command, which is as powerful as a shell.
- Do not put credentials in `nn` config files.

See [SECURITY.md](../SECURITY.md).

---

## 16. Troubleshooting

| Symptom | What to do |
| --- | --- |
| `unsupported format` | `nn formats`; confirm extension and that the reader is compiled (`nn doctor`) |
| `backend unavailable` | `nn backends`; rebuild with `NN_ENABLE_ONNXRUNTIME` / `NN_ENABLE_LITERT_RUNTIME`; check `nn doctor` |
| `does not support this model` | Reference hit an unsupported op; use `--backend onnxruntime` or `litert`, or `nn ops --unsupported` |
| LiteRT missing on macOS Intel | No official `macos_x86_64` prebuilt; use arm64 or disable LiteRT |
| `compressed ZIP member (deflate)` | Re-save with `numpy.savez` (uncompressed) or pass `.npy` files |
| `--activations` empty | Only reference dumps intermediates; omit `--backend` or pass `--backend reference` |
| `nn compare` names look unrelated | Binding is per-model; check `nn io` on both files; alignment is by normalized name then index |
| Convert to TFLite says unavailable | Only IR→ONNX is implemented |
| Windows cannot load LiteRT | Confirm `libLiteRt.dll` was copied next to `nn.exe` (CMake post-build does this when LiteRT is found) |
| XNNPACK `INFO:` on stderr | LiteRT CPU delegate log; not an `nn` error |
| `lint` exit 9 | Read the issue list; 9 means errors, not warnings only |
| `weights not in memory` | Constant payload is a graph input (common in CNTK ONNX). `nn sparsity` still lists layers and MAC share; channel scores need in-file weights. |

```bash
nn doctor
nn formats
nn backends
nn io model.onnx
nn ops model.onnx --unsupported
```

---

## 17. Glossary

| Term | Meaning |
| --- | --- |
| Artifact | A model file or directory on disk |
| Backend | Execution engine (`onnxruntime`, `litert`, `reference`) |
| Canonical op | Format-neutral operator kind (`Convolution`, `MatMul`, `Elementwise`, …) |
| IR / ModelIR | In-memory model after a successful load |
| Native op | Name in the source format (`Conv`, `CONV_2D`, `ADD`) |
| Porcelain | Stable tab-separated output for scripts |
| Prune candidate | Inspect-only ranking from `nn sparsity`; `nn` does not rewrite the graph |
| Reference | In-tree interpreter, not a production runtime |
| schema_version | Integer in JSON objects; `1` in this release |
| Weak channel | Conv/dense output channel whose L1 is ≤ `--channel-frac` × that layer’s max channel L1 |

---

## See also

- [README.md](../README.md) — project overview
- [example-usage.md](example-usage.md) — captured CLI sessions (ONNX and TFLite)
- [exit-status.md](exit-status.md) — exit codes
- [json-schema.md](json-schema.md) — JSON compatibility rules
- [CONTRIBUTING.md](../CONTRIBUTING.md) — build and coding rules
- `nn help <command>` — flags for the binary you actually built
