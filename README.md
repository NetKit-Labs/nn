# nn

Command-line neural-network engineering toolkit.

`nn` is a native Unix-style tool: **one executable, subcommands, no GUI**. Inspect, hash, diff, lint, and analyze model artifacts. Execution is optional — when ONNX Runtime or LiteRT is linked, `nn run` and `nn compare` can actually infer.

Companion to [netkit](https://github.com/NetKit-Labs/netkit) (inference engine) and [memkit](https://github.com/NetKit-Labs/memkit) (embedded containers). `nn` does not replace a runtime; it is the `objdump` / `readelf` / `git` of neural-network files.

```bash
cmake -S . -B build && cmake --build build
./build/nn inspect model.onnx
./build/nn diff old.onnx new.onnx
./build/nn run model.onnx --input input.npy
```

**Status:** 0.1.0. C++20, CMake 3.20+, Apache-2.0.

Full usage: [docs/user-manual.md](docs/user-manual.md). Captured CLI sessions (ONNX and TFLite): [docs/example-usage.md](docs/example-usage.md). Command help: `nn help <command>` or `man nn-<command>`.

## Quick start

```bash
nn inspect mobilenet.onnx
nn ops mobilenet.onnx --canonical
nn tensors model.onnx --largest
nn memory kws.tflite --plan
nn diff model-v1.onnx model-v2.onnx --weights
nn compute model.onnx
nn lint model.onnx
nn hash model.onnx --graph
nn compare float.onnx quant.tflite --input test.npy
nn run model.onnx --input tensors.npz
nn formats
nn backends
nn doctor
```

Machine-readable output:

```bash
nn inspect model.onnx --json | jq .
nn ops model.onnx --porcelain
nn --json formats
```

## What it does

Load a file into a common in-memory IR (`ModelIR`), then inspect or (optionally) execute it. Format readers register **actual** capabilities; `nn formats` prints that table. Missing runtimes print `backend unavailable` instead of inventing results.

### Formats

| Format | Graph | Weights | Execute |
| --- | --- | --- | --- |
| ONNX | yes | yes | yes, if ONNX Runtime is linked |
| TFLite / LiteRT | yes | yes | yes, if LiteRT is linked |
| GGUF | tensors / metadata | yes | no (container, not a graph runtime) |
| SafeTensors | no | yes | no (weights only; no invented graph) |
| PyTorch `.pt` / `.pth` | zip listing | — | no (pickle deserialize is refused) |
| Core ML, OpenVINO, TensorRT, NCNN, MNN, TFJS, ExecuTorch, Keras, SavedModel, Caffe, Darknet, MXNet, Paddle | structural / metadata as implemented | varies | no |

Run `nn formats` or `nn formats onnx` for the compiled table on your build.

### Execution

| Backend | Default for | Notes |
| --- | --- | --- |
| `onnxruntime` | `.onnx` | Downloaded at configure time (`NN_ENABLE_ONNXRUNTIME`, on by default) |
| `litert` | `.tflite` | LiteRT 2.1.6 C SDK + prebuilt (`NN_ENABLE_LITERT_RUNTIME`, on by default). CPU only. No macOS x86_64 prebuilt. |
| `reference` | fallback | Small in-tree float32 interpreter (Add/Mul/Conv/MatMul/activations/reshape/…). Used for `--dump-all` / `--activations` when no `--backend` is set. |

`--input` accepts **npy**, uncompressed **npz** (`numpy.savez`), csv, or raw. Named arrays: `--input NAME=file.npy`. A multi-array npz matches model input names.

`nn convert --to onnx` re-serializes a loaded graph as ONNX. Other conversion routes print `unavailable` unless an adapter is compiled in.

## Build

```text
C++20, CMake 3.20+
GCC, Clang, or MSVC
```

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/nn doctor
```

Configure-time options (all format readers and both runtimes default **on**):

```bash
cmake -S . -B build \
  -DNN_ENABLE_ONNXRUNTIME=ON \
  -DNN_ENABLE_LITERT_RUNTIME=ON \
  -DNN_ENABLE_WERROR=ON
```

| Option | Default | Meaning |
| --- | --- | --- |
| `NN_ENABLE_ONNXRUNTIME` | ON | Download/link ONNX Runtime |
| `NN_ENABLE_LITERT_RUNTIME` | ON | Download/link LiteRT |
| `NN_REQUIRE_RUNTIMES` | OFF | Fail configure if a requested runtime cannot be fetched |
| `NN_SANITIZER` | (empty) | `address`, `undefined`, or `address,undefined` |
| `NN_BUILD_FUZZERS` | OFF | libFuzzer targets (Clang) |
| `NN_ENABLE_<FORMAT>` | ON | Per-format readers |

Sanitizers:

```bash
cmake -S . -B build-asan -DNN_SANITIZER=address,undefined
cmake --build build-asan
ctest --test-dir build-asan --output-on-failure
```

Install (binary + public headers + man pages):

```bash
cmake --install build
man nn
man nn-inspect
```

## Commands

`nn help <command>` and `nn <command> --help` are the source of truth for flags.

| Command | Purpose |
| --- | --- |
| `inspect` | Summarize a model |
| `ops` | List operators |
| `tensors` | List tensors |
| `io` | Input/output contracts |
| `metadata` | Format metadata only |
| `hash` | Artifact / graph / weight / tensor SHA-256 |
| `diff` | Semantic comparison of two models |
| `graph` | Export graph as text, DOT, mermaid, or JSON |
| `memory` | Weights, activation lifetimes, optional arena plan |
| `compute` | Estimate MACs/FLOPs |
| `quant` | Quantization analysis |
| `sparsity` | Weight sparsity and prune-candidate hints |
| `lint` | Static correctness checks |
| `run` | Execute with an available backend |
| `compare` | Compare two models or backends on the same inputs |
| `test` | Deterministic test-vector manifests |
| `validate` | Score against expected tensors |
| `regression` | Policy check between two models |
| `benchmark` | Inference latency |
| `profile` | Per-operator timings (reference backend) |
| `compat` | Operator compatibility vs a runtime |
| `target` | Deployability against a hardware profile |
| `partition` | Accelerator vs CPU partition |
| `convert` | Conversion frontend |
| `optimize` | Safe graph rewrites |
| `extract` | Tensors or subgraphs |
| `canonicalize` | Stable graph text for hashing/diffing |
| `bisect` | Artifact or git bisect |
| `doctor` | Installation and compile-time features |
| `formats` / `backends` / `targets` | Capability tables |
| `config` | Git-like user config |
| `help` / `version` | Help and build identity |

Global flags: `--json`, `--yaml`, `--porcelain`, `-v` / `-q`, `--color=auto|always|never`, `--threads N`, `--output FILE`.

### Exit status

Stable process codes. See [docs/exit-status.md](docs/exit-status.md).

| Code | Meaning |
| --- | --- |
| 0 | success |
| 1 | comparison difference / policy failure |
| 2 | usage error |
| 3 | file error |
| 4 | malformed model |
| 5 | unsupported format |
| 6 | unsupported operator |
| 7 | backend / conversion adapter unavailable |
| 8 | execution failure |
| 9 | validation / lint errors |
| 10 | internal error |

JSON objects include `"schema_version": 1`. Additive keys are allowed; bump the version for breaking changes. See [docs/json-schema.md](docs/json-schema.md).

## Design

- Analysis lives in `libnn_core.a`. The CLI (`src/main.cpp`) is a thin dispatcher.
- Artifacts are untrusted: parsers bounds-check; tensor sizes are overflow-checked; PyTorch pickle is not deserialized; embedded code is never executed.
- Do not claim Execute or Convert unless that SDK is actually linked.
- `nn compare` binds inputs independently on each model (ONNX vs TFLite names), matches outputs by normalized name then index, and permutes rank-4 NCHW/NHWC when shapes match that layout.

Layout:

```text
include/nn/     public headers
src/cli/        option parsing, help, dispatcher
src/commands/   one file per subcommand
src/formats/    readers (and ONNX writer)
src/runtime/    ORT, LiteRT, reference, tensor I/O
src/analysis/   inspect, diff, memory, lint, …
docs/man/       man pages
tests/          unit, integration, fuzz
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

ONNX Runtime and LiteRT are downloaded as optional SDKs at configure time and keep their own licenses.
