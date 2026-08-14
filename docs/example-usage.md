# nn example usage

Captured from `nn` 0.1.0 on macOS arm64 with ONNX Runtime 1.20.1 and LiteRT 2.1.6 linked. Commands were run in a directory that contained the model files. Latency numbers vary by machine; hashes are of these exact files.

Exit status is shown after each prompt (`# exit 0`). **1** from `diff` / `compare` / `compat` is a successful comparison that found a difference or a policy miss, not a crash. **2** is a usage / bind error. **3** is a file or payload error.

LiteRT often prints `INFO: Created TensorFlow Lite XNNPACK delegate for CPU.` on stderr. Those lines are included below when they appeared.

Full flag reference: [user-manual.md](user-manual.md).

## Contents

- [Models used](#models-used)
- [Discovery](#discovery)
- [ONNX — inspect, I/O, operators, tensors](#onnx--inspect-io-operators-tensors)
- [ONNX — hash, memory, compute, quantization, sparsity, lint](#onnx--hash-memory-compute-quantization-sparsity-lint)
- [ONNX — graph, canonicalize, compat, target](#onnx--graph-canonicalize-compat-target)
- [ONNX — run and benchmark (MNIST)](#onnx--run-and-benchmark-mnist)
- [ONNX — tiny Add / Mul graphs](#onnx--tiny-add--mul-graphs)
- [TFLite — inspect through execute](#tflite--inspect-through-execute)
- [TFLite — compare and diff](#tflite--compare-and-diff)
- [Cross-format ONNX vs TFLite](#cross-format-onnx-vs-tflite)
- [Shell one-liners](#shell-one-liners)

## Models used

| File | Source | Role |
| --- | --- | --- |
| `mnist-8.onnx` | [ONNX Model Zoo MNIST](https://github.com/onnx/models) (`mnist-8.onnx`, 26,454 bytes) | Real ONNX CNN (CNTK, opset 8) |
| `add.tflite` | TensorFlow `tensorflow/lite/testdata/add.bin` (544 bytes) | Real TFLite: two `ADD` nodes, input `1x8x8x3` |
| `add.onnx` | Tiny graph written by `nn` tests (`Add`, two `1x4` inputs) | Runnable ONNX with known npy/npz inputs |
| `mul.onnx` | Same shape as `add.onnx` but `Mul` | Diff / numeric compare contrast |
| `add_wide.onnx` | Same Add graph with `1x8` tensors | Shape-change diff |

Download the public fixtures (CMake also fetches them into `build/testdata/`):

```bash
curl -L -o mnist-8.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx
curl -L -o add.tflite \
  https://raw.githubusercontent.com/tensorflow/tensorflow/v2.18.0/tensorflow/lite/testdata/add.bin
```

Inputs for `add.onnx` (float32, shape `1x4`):

```bash
# a.npy = [[1, 2, 3, 4]]   b.npy = [[10, 20, 30, 40]]
# tensors.npz is numpy.savez (uncompressed ZIP stored) with members input0.npy and input1.npy
```

Do not commit model binaries. Generate tiny ONNX fixtures in tests, and download the public MNIST / TFLite files as above.

---

## Discovery

```text
$ nn --version    # exit 0
nn 0.1.0
```

```text
$ nn version --build-options    # exit 0
nn 0.1.0
commit:               unknown
compiler:             AppleClang 21.0.0.21000101
os:                   macOS
arch:                 arm64
formats:              onnx, tflite, gguf, safetensors, pytorch, executorch, coreml, openvino, tensorrt, ncnn, mnn, tfjs, legacy, keras, tensorflow
runtimes:             reference, onnxruntime, litert
```

```text
$ nn doctor    # exit 0
NN version            0.1.0

Core
  ONNX                yes
  TFLite/LiteRT       yes
  GGUF                yes
  SafeTensors         yes
  ExecuTorch          yes
  Core ML             yes
  OpenVINO IR         yes
  NCNN                yes
  MNN                 yes
  TensorFlow          yes
  Keras               yes
  PyTorch             yes
  TensorFlow.js       yes
  TensorRT Engine     yes
  Caffe               yes
  Darknet             yes
  MXNet               yes
  PaddlePaddle        yes

Optional runtimes
  onnxruntime         1.20.1
  litert              2.1.6
  reference           0.1.0

System
  OS                  macOS
  Architecture        arm64
  Compiler            AppleClang 21.0.0.21000101
  Git commit          unknown
  CPUs                10
```

This capture was from a binary whose git metadata was not embedded (`Git commit unknown`). A build from a git checkout may print a SHA.

```text
$ nn formats    # exit 0
Format             Read   Graph   Weights   Execute   Convert
---------------------------------------------------------------
ONNX               yes    yes     yes       yes       yes
TFLite/LiteRT      yes    yes     yes       yes       no
GGUF               yes    yes     yes       no        no
SafeTensors        yes    no      yes       no        no
ExecuTorch         yes    yes     yes       no        no
Core ML            yes    yes     yes       no        no
OpenVINO IR        yes    yes     yes       no        no
NCNN               yes    yes     yes       no        no
MNN                yes    yes     yes       no        no
TensorFlow         yes    yes     yes       no        no
Keras              yes    yes     yes       no        no
PyTorch            yes    no      no        no        no
TensorFlow.js      yes    yes     yes       no        no
TensorRT Engine    yes    no      no        no        no
Caffe              yes    yes     yes       no        no
Darknet            yes    yes     yes       no        no
MXNet              yes    yes     yes       no        no
PaddlePaddle       yes    yes     yes       no        no
```

```text
$ nn formats onnx    # exit 0
Name:                 ONNX
Read:                 yes
Graph:                yes
Weights:              yes
Execute:              yes
Convert:              yes
Notes:                graph and weights; ONNX rewrite via nn convert --to onnx; execution via ONNX Runtime
```

```text
$ nn formats tflite    # exit 0
Name:                 TFLite/LiteRT
Read:                 yes
Graph:                yes
Weights:              yes
Execute:              yes
Convert:              no
Notes:                graph and weights; execution via LiteRT
```

```text
$ nn backends    # exit 0
NAME            AVAILABLE   VERSION
-----------------------------------
onnxruntime    yes    1.20.1
litert    yes    2.1.6
reference    yes    0.1.0
```

```text
$ nn targets    # exit 0
NAME                 CPU              RAM         FLASH
-------------------------------------------------------
cortex-m0+           Cortex-M0+       16.0 KB     128.0 KB
cortex-m3            Cortex-M3        64.0 KB     256.0 KB
cortex-m4            Cortex-M4        128.0 KB    512.0 KB
cortex-m4f           Cortex-M4F       256.0 KB    1.00 MB
cortex-m7            Cortex-M7        512.0 KB    2.00 MB
cortex-m33           Cortex-M33       256.0 KB    1.00 MB
cortex-m55           Cortex-M55       512.0 KB    2.00 MB
cortex-m85           Cortex-M85       1.00 MB     4.00 MB
cortex-a53           Cortex-A53       1.00 GB     8.00 GB
cortex-a55           Cortex-A55       2.00 GB     16.0 GB
cortex-a72           Cortex-A72       4.00 GB     32.0 GB
cortex-a76           Cortex-A76       8.00 GB     64.0 GB
riscv-mcu            generic RISC-V M 64.0 KB     256.0 KB
riscv-linux          generic RISC-V L 1.00 GB     8.00 GB
esp32                ESP32            520.0 KB    4.00 MB
esp32-s3             ESP32-S3         512.0 KB    8.00 MB
apple-silicon        Apple Silicon ge 8.00 GB     256.0 GB
x86-64               x86-64 generic   8.00 GB     256.0 GB
```

```text
$ nn help inspect    # exit 0
NAME
    nn-inspect - Summarize a neural-network model artifact.

SYNOPSIS
    nn inspect [options] <model>

DESCRIPTION
    Summarize a neural-network model artifact.

ARGUMENTS
    <model>  Path to a model file or directory

OPTIONS
    --summary  short summary
    --all  include all sections
    --metadata  show metadata
    --inputs  show inputs
    --outputs  show outputs
    --ops  show operators
    --tensors  show tensors
    --weights  show weights
    --quantization  show quantization
    --subgraphs  show subgraphs
    --raw  include raw format metadata

EXAMPLES
    nn inspect model.onnx
    nn inspect model.onnx --json

EXIT STATUS
    0 success; 3 file error; 4 malformed; 5 unsupported format
```

---

## ONNX — inspect, I/O, operators, tensors

```text
$ nn inspect mnist-8.onnx    # exit 0
Model
File:                 mnist-8.onnx
Format:               onnx
Format version:       3
Framework:            2.5.1
Producer:             CNTK
File size:            25.8 KB
SHA-256:              2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf

Graph
Graphs:               1
Nodes:                12
Tensors:              21

Parameters
Count:                5,998
Storage:              23.4 KB

Compute
MACs:                 786.6 K
FLOPs:                1.59 M

Inputs
  Input3
    shape:            1x1x28x28
    dtype:            float32
  Parameter5
    shape:            8x1x5x5
    dtype:            float32
  Parameter6
    shape:            8x1x1
    dtype:            float32
  Parameter87
    shape:            16x8x5x5
    dtype:            float32
  Parameter88
    shape:            16x1x1
    dtype:            float32
  Pooling160_Output_0_reshape0_shape
    shape:            2
    dtype:            int64
  Parameter193
    shape:            16x4x4x10
    dtype:            float32
  Parameter193_reshape1_shape
    shape:            2
    dtype:            int64
  Parameter194
    shape:            1x10
    dtype:            float32

Outputs
  Plus214_Output_0
    shape:            1x10
    dtype:            float32
```

This CNTK export lists initializers on the graph input list as well as `Input3`. That is the file, not a bug in `nn`.

```text
$ nn inspect mnist-8.onnx --summary    # exit 0
Model
File:                 mnist-8.onnx
Format:               onnx
Format version:       3
Framework:            2.5.1
Producer:             CNTK
File size:            25.8 KB
SHA-256:              2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf

Parameters
Count:                5,998
Storage:              23.4 KB

Compute
MACs:                 786.6 K
FLOPs:                1.59 M

Inputs
  Input3
    shape:            1x1x28x28
    dtype:            float32
  Parameter5
    shape:            8x1x5x5
    dtype:            float32
  Parameter6
    shape:            8x1x1
    dtype:            float32
  Parameter87
    shape:            16x8x5x5
    dtype:            float32
  Parameter88
    shape:            16x1x1
    dtype:            float32
  Pooling160_Output_0_reshape0_shape
    shape:            2
    dtype:            int64
  Parameter193
    shape:            16x4x4x10
    dtype:            float32
  Parameter193_reshape1_shape
    shape:            2
    dtype:            int64
  Parameter194
    shape:            1x10
    dtype:            float32

Outputs
  Plus214_Output_0
    shape:            1x10
    dtype:            float32
```

```text
$ nn inspect mnist-8.onnx --all    # exit 0
Model
File:                 mnist-8.onnx
Format:               onnx
Format version:       3
Framework:            2.5.1
Producer:             CNTK
File size:            25.8 KB
SHA-256:              2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf

Graph
Graphs:               1
Nodes:                12
Tensors:              21

Parameters
Count:                5,998
Storage:              23.4 KB

Compute
MACs:                 786.6 K
FLOPs:                1.59 M

Inputs
  Input3
    shape:            1x1x28x28
    dtype:            float32
  Parameter5
    shape:            8x1x5x5
    dtype:            float32
  Parameter6
    shape:            8x1x1
    dtype:            float32
  Parameter87
    shape:            16x8x5x5
    dtype:            float32
  Parameter88
    shape:            16x1x1
    dtype:            float32
  Pooling160_Output_0_reshape0_shape
    shape:            2
    dtype:            int64
  Parameter193
    shape:            16x4x4x10
    dtype:            float32
  Parameter193_reshape1_shape
    shape:            2
    dtype:            int64
  Parameter194
    shape:            1x10
    dtype:            float32

Outputs
  Plus214_Output_0
    shape:            1x10
    dtype:            float32

Metadata
opset.0:              ai.onnx:8
```

```text
$ nn --json inspect mnist-8.onnx    # exit 0
{
  "file": "mnist-8.onnx",
  "file_size": 26454,
  "format": "onnx",
  "format_version": "3",
  "graphs": 1,
  "macs": 786560,
  "nodes": 12,
  "parameters": 5998,
  "producer": "CNTK",
  "schema_version": 1,
  "sha256": "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf",
  "tensors": 21
}
```

```text
$ nn io mnist-8.onnx    # exit 0

Inputs
Input3
  shape:              1x1x28x28
  dtype:              float32
  layout:             -
  quant:              none
Parameter5
  shape:              8x1x5x5
  dtype:              float32
  layout:             -
  quant:              none
Parameter6
  shape:              8x1x1
  dtype:              float32
  layout:             -
  quant:              none
Parameter87
  shape:              16x8x5x5
  dtype:              float32
  layout:             -
  quant:              none
Parameter88
  shape:              16x1x1
  dtype:              float32
  layout:             -
  quant:              none
Pooling160_Output_0_reshape0_shape
  shape:              2
  dtype:              int64
  layout:             -
  quant:              none
Parameter193
  shape:              16x4x4x10
  dtype:              float32
  layout:             -
  quant:              none
Parameter193_reshape1_shape
  shape:              2
  dtype:              int64
  layout:             -
  quant:              none
Parameter194
  shape:              1x10
  dtype:              float32
  layout:             -
  quant:              none

Outputs
Plus214_Output_0
  shape:              1x10
  dtype:              float32
  layout:             -
  quant:              none
```

```text
$ nn --json io mnist-8.onnx    # exit 0
{
  "inputs": [
    {
      "dtype": "float32",
      "name": "Input3",
      "shape": "1x1x28x28"
    },
    {
      "dtype": "float32",
      "name": "Parameter5",
      "shape": "8x1x5x5"
    },
    {
      "dtype": "float32",
      "name": "Parameter6",
      "shape": "8x1x1"
    },
    {
      "dtype": "float32",
      "name": "Parameter87",
      "shape": "16x8x5x5"
    },
    {
      "dtype": "float32",
      "name": "Parameter88",
      "shape": "16x1x1"
    },
    {
      "dtype": "int64",
      "name": "Pooling160_Output_0_reshape0_shape",
      "shape": "2"
    },
    {
      "dtype": "float32",
      "name": "Parameter193",
      "shape": "16x4x4x10"
    },
    {
      "dtype": "int64",
      "name": "Parameter193_reshape1_shape",
      "shape": "2"
    },
    {
      "dtype": "float32",
      "name": "Parameter194",
      "shape": "1x10"
    }
  ],
  "outputs": [
    {
      "dtype": "float32",
      "name": "Plus214_Output_0",
      "shape": "1x10"
    }
  ],
  "schema_version": 1
}
```

```text
$ nn ops mnist-8.onnx --canonical    # exit 0
OPERATOR                 COUNT          MACs
------------------------------------------------
Activation                    2              -
Convolution                   2        784.0 K
Elementwise                   3              -
MatMul                        1         2.56 K
Pooling                       2              -
Reshape                       2              -
```

`nn ops` with no flags is the same table as `--canonical` (and `--by-type` on this model).

```text
$ nn ops mnist-8.onnx --native    # exit 0
OPERATOR                 COUNT          MACs
------------------------------------------------
Add                           3              -
Conv                          2        784.0 K
MatMul                        1         2.56 K
MaxPool                       2              -
Relu                          2              -
Reshape                       2              -
```

```text
$ nn ops mnist-8.onnx --by-cost    # exit 0
OPERATOR                 COUNT          MACs
------------------------------------------------
Convolution                   2        784.0 K
MatMul                        1         2.56 K
Activation                    2              -
Elementwise                   3              -
Pooling                       2              -
Reshape                       2              -
```

```text
$ nn ops mnist-8.onnx --details    # exit 0
Times212_reshape1
  op:                 Reshape
  canonical:          Reshape
Convolution28
  op:                 Conv
  canonical:          Convolution
Plus30
  op:                 Add
  canonical:          Elementwise
ReLU32
  op:                 Relu
  canonical:          Activation
Pooling66
  op:                 MaxPool
  canonical:          Pooling
Convolution110
  op:                 Conv
  canonical:          Convolution
Plus112
  op:                 Add
  canonical:          Elementwise
ReLU114
  op:                 Relu
  canonical:          Activation
Pooling160
  op:                 MaxPool
  canonical:          Pooling
Times212_reshape0
  op:                 Reshape
  canonical:          Reshape
Times212
  op:                 MatMul
  canonical:          MatMul
Plus214
  op:                 Add
  canonical:          Elementwise
```

```text
$ nn ops mnist-8.onnx --op Conv    # exit 0
Convolution28
  op:                 Conv
  canonical:          Convolution
Convolution110
  op:                 Conv
  canonical:          Convolution
```

```text
$ nn tensors mnist-8.onnx --largest    # exit 0
NAME                             SHAPE            DTYPE      BYTES  CONST  QUANT
-------------------------------------------------------------------------------
Convolution28_Output_0           1x8x28x28        float32     24.5 KB    no  none
Plus30_Output_0                  1x8x28x28        float32     24.5 KB    no  none
ReLU32_Output_0                  1x8x28x28        float32     24.5 KB    no  none
Parameter87                      16x8x5x5         float32     12.5 KB   yes  none
Convolution110_Output_0          1x16x14x14       float32     12.2 KB    no  none
Plus112_Output_0                 1x16x14x14       float32     12.2 KB    no  none
ReLU114_Output_0                 1x16x14x14       float32     12.2 KB    no  none
Parameter193                     16x4x4x10        float32     10.0 KB   yes  none
Parameter193_reshape1            256x10           float32     10.0 KB    no  none
Pooling66_Output_0               1x8x14x14        float32     6.12 KB    no  none
Input3                           1x1x28x28        float32     3.06 KB    no  none
Pooling160_Output_0              1x16x4x4         float32     1.00 KB    no  none
Pooling160_Output_0_reshape0     1x256            float32     1.00 KB    no  none
Parameter5                       8x1x5x5          float32       800 B   yes  none
Parameter88                      16x1x1           float32        64 B   yes  none
Parameter194                     1x10             float32        40 B   yes  none
Plus214_Output_0                 1x10             float32        40 B    no  none
Times212_Output_0                1x10             float32        40 B    no  none
Parameter6                       8x1x1            float32        32 B   yes  none
Pooling160_Output_0_reshape0_sha 2                int64          16 B   yes  none
Parameter193_reshape1_shape      2                int64          16 B   yes  none
```

```text
$ nn tensors mnist-8.onnx --weights    # exit 0
NAME                             SHAPE            DTYPE      BYTES  CONST  QUANT
-------------------------------------------------------------------------------
Parameter193                     16x4x4x10        float32     10.0 KB   yes  none
Parameter87                      16x8x5x5         float32     12.5 KB   yes  none
Parameter5                       8x1x5x5          float32       800 B   yes  none
Parameter6                       8x1x1            float32        32 B   yes  none
Parameter88                      16x1x1           float32        64 B   yes  none
Pooling160_Output_0_reshape0_sha 2                int64          16 B   yes  none
Parameter193_reshape1_shape      2                int64          16 B   yes  none
Parameter194                     1x10             float32        40 B   yes  none
```

```text
$ nn tensors mnist-8.onnx --inputs    # exit 0
NAME                             SHAPE            DTYPE      BYTES  CONST  QUANT
-------------------------------------------------------------------------------
Parameter193                     16x4x4x10        float32     10.0 KB   yes  none
Parameter87                      16x8x5x5         float32     12.5 KB   yes  none
Parameter5                       8x1x5x5          float32       800 B   yes  none
Parameter6                       8x1x1            float32        32 B   yes  none
Parameter88                      16x1x1           float32        64 B   yes  none
Pooling160_Output_0_reshape0_sha 2                int64          16 B   yes  none
Parameter193_reshape1_shape      2                int64          16 B   yes  none
Parameter194                     1x10             float32        40 B   yes  none
Input3                           1x1x28x28        float32     3.06 KB    no  none
```

```text
$ nn tensors mnist-8.onnx --outputs    # exit 0
NAME                             SHAPE            DTYPE      BYTES  CONST  QUANT
-------------------------------------------------------------------------------
Plus214_Output_0                 1x10             float32        40 B    no  none
```

```text
$ nn tensors mnist-8.onnx --dtype float32    # exit 0
NAME                             SHAPE            DTYPE      BYTES  CONST  QUANT
-------------------------------------------------------------------------------
Parameter193                     16x4x4x10        float32     10.0 KB   yes  none
Parameter87                      16x8x5x5         float32     12.5 KB   yes  none
Parameter5                       8x1x5x5          float32       800 B   yes  none
Parameter6                       8x1x1            float32        32 B   yes  none
Parameter88                      16x1x1           float32        64 B   yes  none
Parameter194                     1x10             float32        40 B   yes  none
Input3                           1x1x28x28        float32     3.06 KB    no  none
Plus214_Output_0                 1x10             float32        40 B    no  none
Parameter193_reshape1            256x10           float32     10.0 KB    no  none
Convolution28_Output_0           1x8x28x28        float32     24.5 KB    no  none
Plus30_Output_0                  1x8x28x28        float32     24.5 KB    no  none
ReLU32_Output_0                  1x8x28x28        float32     24.5 KB    no  none
Pooling66_Output_0               1x8x14x14        float32     6.12 KB    no  none
Convolution110_Output_0          1x16x14x14       float32     12.2 KB    no  none
Plus112_Output_0                 1x16x14x14       float32     12.2 KB    no  none
ReLU114_Output_0                 1x16x14x14       float32     12.2 KB    no  none
Pooling160_Output_0              1x16x4x4         float32     1.00 KB    no  none
Pooling160_Output_0_reshape0     1x256            float32     1.00 KB    no  none
Times212_Output_0                1x10             float32        40 B    no  none
```

```text
$ nn tensors mnist-8.onnx --name Parameter    # exit 0
NAME                             SHAPE            DTYPE      BYTES  CONST  QUANT
-------------------------------------------------------------------------------
Parameter193                     16x4x4x10        float32     10.0 KB   yes  none
Parameter87                      16x8x5x5         float32     12.5 KB   yes  none
Parameter5                       8x1x5x5          float32       800 B   yes  none
Parameter6                       8x1x1            float32        32 B   yes  none
Parameter88                      16x1x1           float32        64 B   yes  none
Parameter193_reshape1_shape      2                int64          16 B   yes  none
Parameter194                     1x10             float32        40 B   yes  none
Parameter193_reshape1            256x10           float32     10.0 KB    no  none
```

```text
$ nn metadata mnist-8.onnx    # exit 0
Format:               onnx
Producer:             CNTK
Version:              3
opset.0:              ai.onnx:8
```

---

## ONNX — hash, memory, compute, quantization, sparsity, lint

```text
$ nn hash mnist-8.onnx    # exit 0
artifact:             2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf
graph:                423336d2bde9f3f4c0f08880f426e81112fef1c9c55d9d5b35c905752b066c35
weights:              48791bb71f12ccbcdaf6229e8b5a3bda5aaa5f32fa0b5fc0623c99583c50fdbe
```

```text
$ nn --json hash mnist-8.onnx    # exit 0
{
  "artifact": "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf",
  "graph": "423336d2bde9f3f4c0f08880f426e81112fef1c9c55d9d5b35c905752b066c35",
  "schema_version": 1,
  "weights": "48791bb71f12ccbcdaf6229e8b5a3bda5aaa5f32fa0b5fc0623c99583c50fdbe"
}
```

```text
$ nn --porcelain hash mnist-8.onnx    # exit 0
artifact:	2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf
graph:	423336d2bde9f3f4c0f08880f426e81112fef1c9c55d9d5b35c905752b066c35
weights:	48791bb71f12ccbcdaf6229e8b5a3bda5aaa5f32fa0b5fc0623c99583c50fdbe
```

```text
$ nn hash mnist-8.onnx --graph    # exit 0
graph:                423336d2bde9f3f4c0f08880f426e81112fef1c9c55d9d5b35c905752b066c35
```

```text
$ nn hash mnist-8.onnx --weights    # exit 0
weights:              48791bb71f12ccbcdaf6229e8b5a3bda5aaa5f32fa0b5fc0623c99583c50fdbe
```

```text
$ nn hash mnist-8.onnx --canonical    # exit 0
graph:                423336d2bde9f3f4c0f08880f426e81112fef1c9c55d9d5b35c905752b066c35
```

```text
$ nn memory mnist-8.onnx    # exit 0
Memory Analysis

Weights               23.4 KB
Persistent tensors    26.5 KB
Peak live activations  59.0 KB
Estimated scratch     0 B
--------------------------------------
Estimated RAM requirement  59.0 KB
Flash/model storage   25.8 KB
```

```text
$ nn memory mnist-8.onnx --plan    # exit 0
Memory Analysis

Weights               23.4 KB
Persistent tensors    26.5 KB
Peak live activations  59.0 KB
Estimated scratch     59.0 KB
--------------------------------------
Estimated RAM requirement  59.0 KB
Flash/model storage   25.8 KB

Tensor                   Start       Size        Lifetime
---------------------------------------------------------
Convolution28_Output_0          0      25088      1-2
Plus30_Output_0             25088      25088      2-3
ReLU32_Output_0                 0      25088      3-4
Convolution110_Output_0         0      12544      5-6
Plus112_Output_0            12544      12544      6-7
ReLU114_Output_0                0      12544      7-8
Parameter193_reshape1       50176      10240      0-10
Pooling66_Output_0          25088       6272      4-5
Pooling160_Output_0         12544       1024      8-9
Pooling160_Output_0_resh        0       1024      9-10
Times212_Output_0            1024         48      10-11
Plus214_Output_0                0         48      11-12
```

```text
$ nn memory mnist-8.onnx --timeline    # exit 0
Memory Analysis

Weights               23.4 KB
Persistent tensors    26.5 KB
Peak live activations  59.0 KB
Estimated scratch     0 B
--------------------------------------
Estimated RAM requirement  59.0 KB
Flash/model storage   25.8 KB

Timeline
Parameter193                       10.0 KB  life -1-0  persistent
Parameter87                        12.5 KB  life -1-5  persistent
Parameter5                           800 B  life -1-1  persistent
Parameter6                            32 B  life -1-2  persistent
Parameter88                           64 B  life -1-6  persistent
Pooling160_Output_0_reshape0_sha      16 B  life -1-9  persistent
Parameter193_reshape1_shape           16 B  life -1-0  persistent
Parameter194                          40 B  life -1-11  persistent
Input3                             3.06 KB  life -1-1  persistent
Plus214_Output_0                      40 B  life 11-12
Parameter193_reshape1              10.0 KB  life 0-10
Convolution28_Output_0             24.5 KB  life 1-2
Plus30_Output_0                    24.5 KB  life 2-3
ReLU32_Output_0                    24.5 KB  life 3-4
Pooling66_Output_0                 6.12 KB  life 4-5
Convolution110_Output_0            12.2 KB  life 5-6
Plus112_Output_0                   12.2 KB  life 6-7
ReLU114_Output_0                   12.2 KB  life 7-8
Pooling160_Output_0                1.00 KB  life 8-9
Pooling160_Output_0_reshape0       1.00 KB  life 9-10
Times212_Output_0                     40 B  life 10-11
```

```text
$ nn compute mnist-8.onnx    # exit 0
MACs:                 786.6 K
FLOPs:                1.59 M
Integer ops:          unknown
Float ops:            1.59 M
Unknown nodes:        0
```

```text
$ nn compute mnist-8.onnx --per-node    # exit 0
MACs:                 786.6 K
FLOPs:                1.59 M
Integer ops:          unknown
Float ops:            1.59 M
Unknown nodes:        0
Times212_reshape1  Reshape  unknown
Convolution28  Conv  156.8 K
Plus30  Add  unknown
ReLU32  Relu  unknown
Pooling66  MaxPool  unknown
Convolution110  Conv  627.2 K
Plus112  Add  unknown
ReLU114  Relu  unknown
Pooling160  MaxPool  unknown
Times212_reshape0  Reshape  unknown
Times212  MatMul  2.56 K
Plus214  Add  unknown
```

```text
$ nn quant mnist-8.onnx    # exit 0
Quantized tensors:    0
Float tensors:        19
Integer tensors:      2
Per-channel:          0
Per-tensor:           0
Quantize nodes:       0
Dequantize nodes:     0
```

```text
$ nn sparsity mnist-8.onnx --threshold 1e-6    # exit 0
Tensors considered:   8
Tensors computed:     0
Zero fraction:        0.000000
Near-zero fraction:   0.000000
Near-zero |w| <=:     0.000001
Weak-channel cut:     1.00% of max channel L1
Total MACs:           786.6 K
Est. saved bytes:     0 B (upper bound)
Est. saved MACs:      0.00 (upper bound)

LAYER                OP         SHAPE            ZEROS       NEAR        WEAK     MAC%    SCORE
-----------------------------------------------------------------------------------------------
Convolution110       Conv       16x8x5x5         -           -           -          79.7%        -
Convolution28        Conv       8x1x5x5          -           -           -          19.9%        -
Times212_reshape1    Reshape    16x4x4x10        -           -           -              -        -
Plus112              Add        16x1x1           -           -           -              -        -
Plus214              Add        1x10             -           -           -              -        -
Plus30               Add        8x1x1            -           -           -              -        -
Times212_reshape0    Reshape    2                -           -           -              -        -
Times212_reshape1    Reshape    2                -           -           -              -        -

Candidates only; nn does not prune. Magnitude is a where-to-look hint, not accuracy.

# stderr
nn: warning: weights not in memory for Parameter193; sparsity not computed
nn: warning: weights not in memory for Parameter87; sparsity not computed
nn: warning: weights not in memory for Parameter5; sparsity not computed
nn: warning: weights not in memory for Parameter6; sparsity not computed
nn: warning: weights not in memory for Parameter88; sparsity not computed
nn: warning: weights not in memory for Pooling160_Output_0_reshape0_shape; sparsity not computed
nn: warning: weights not in memory for Parameter193_reshape1_shape; sparsity not computed
nn: warning: weights not in memory for Parameter194; sparsity not computed
```

CNTK keeps some weights as graph inputs rather than in-memory initializers, so sparsity cannot be computed. Warnings go to stderr. Layers are still listed with MAC share.

```text
$ nn lint mnist-8.onnx    # exit 0
info: [reshape] reshape present; verify layout assumptions (Times212_reshape1)
info: [reshape] reshape present; verify layout assumptions (Times212_reshape0)
errors:               0
warnings:             0
```

---

## ONNX — graph, canonicalize, compat, target

```text
$ nn graph mnist-8.onnx --format text    # exit 0
0  Reshape  Times212_reshape1
1  Conv  Convolution28
2  Add  Plus30
3  Relu  ReLU32
4  MaxPool  Pooling66
5  Conv  Convolution110
6  Add  Plus112
7  Relu  ReLU114
8  MaxPool  Pooling160
9  Reshape  Times212_reshape0
10  MatMul  Times212
11  Add  Plus214
```

```text
$ nn graph mnist-8.onnx --format mermaid    # exit 0
graph TD
  n0[Times212_reshape1]
  n1[Convolution28]
  n2[Plus30]
  n3[ReLU32]
  n4[Pooling66]
  n5[Convolution110]
  n6[Plus112]
  n7[ReLU114]
  n8[Pooling160]
  n9[Times212_reshape0]
  n10[Times212]
  n11[Plus214]
```

```text
$ nn compat mnist-8.onnx --runtime onnxruntime    # exit 0
Runtime:              onnxruntime
Capability table:     1.16.0
Notes:                documented ONNX Runtime opset coverage; runtime is not linked unless NN_ENABLE_ONNXRUNTIME is on and the SDK is present
Nodes supported:      12 / 12
RESULT: compatible (vs onnxruntime table 1.16.0)
```

```text
$ nn compat mnist-8.onnx --runtime reference    # exit 1
Runtime:              reference
Capability table:     0.1.0
Notes:                built-in reference interpreter; compiled into this binary
Nodes supported:      10 / 12

Unsupported:
  Pooling66 (MaxPool)
  Pooling160 (MaxPool)
RESULT: incompatible
```

The in-tree interpreter does not implement `MaxPool`. ONNX Runtime reports 12/12 supported.

```text
$ nn target mnist-8.onnx --target cortex-m4f    # exit 0
Target: cortex-m4f
RAM:                  256.0 KB
Flash:                1.00 MB

Storage
    Model:            25.8 KB       PASS

RAM
    Activations:      59.0 KB
    Scratch:          0 B
    Runtime overhead:  24.0 KB
    Total:            83.0 KB       PASS

Result:
    MODEL FITS
    runtime overhead is estimated (24 KiB), not measured
```

```text
$ nn target mnist-8.onnx --target cortex-a76    # exit 0
Target: cortex-a76
RAM:                  8.00 GB
Flash:                64.0 GB

Storage
    Model:            25.8 KB       PASS

RAM
    Activations:      59.0 KB
    Scratch:          0 B
    Runtime overhead:  24.0 KB
    Total:            83.0 KB       PASS

Result:
    MODEL FITS
    runtime overhead is estimated (24 KiB), not measured
```

```text
$ nn target mnist-8.onnx --target esp32-s3    # exit 0
Target: esp32-s3
RAM:                  512.0 KB
Flash:                8.00 MB

Storage
    Model:            25.8 KB       PASS

RAM
    Activations:      59.0 KB
    Scratch:          0 B
    Runtime overhead:  24.0 KB
    Total:            83.0 KB       PASS

Result:
    MODEL FITS
    runtime overhead is estimated (24 KiB), not measured
```

```text
$ nn canonicalize mnist-8.onnx    # exit 0
format=onnx
graph CNTKGraph
inputs Input3:1x1x28x28:float32 Parameter193:16x4x4x10:float32 Parameter193_reshape1_shape:2:int64 Parameter194:1x10:float32 Parameter5:8x1x5x5:float32 Parameter6:8x1x1:float32 Parameter87:16x8x5x5:float32 Parameter88:16x1x1:float32 Pooling160_Output_0_reshape0_shape:2:int64
outputs Plus214_Output_0:1x10:float32
Activation|Relu in:1x16x14x14/float32 out:1x16x14x14/float32
Activation|Relu in:1x8x28x28/float32 out:1x8x28x28/float32
Convolution|Conv in:1x1x28x28/float32 in:8x1x5x5/float32 out:1x8x28x28/float32 auto_pad=SAME_UPPER dilations=[1, 1] group=1 kernel_shape=[5, 5] strides=[1, 1]
Convolution|Conv in:1x8x14x14/float32 in:16x8x5x5/float32 out:1x16x14x14/float32 auto_pad=SAME_UPPER dilations=[1, 1] group=1 kernel_shape=[5, 5] strides=[1, 1]
Elementwise|Add in:1x10/float32 in:1x10/float32 out:1x10/float32
Elementwise|Add in:1x16x14x14/float32 in:16x1x1/float32 out:1x16x14x14/float32
Elementwise|Add in:1x8x28x28/float32 in:8x1x1/float32 out:1x8x28x28/float32
MatMul|MatMul in:1x256/float32 in:256x10/float32 out:1x10/float32
Pooling|MaxPool in:1x16x14x14/float32 out:1x16x4x4/float32 auto_pad=NOTSET kernel_shape=[3, 3] pads=[0, 0, 0, 0] strides=[3, 3]
Pooling|MaxPool in:1x8x28x28/float32 out:1x8x14x14/float32 auto_pad=NOTSET kernel_shape=[2, 2] pads=[0, 0, 0, 0] strides=[2, 2]
Reshape|Reshape in:16x4x4x10/float32 in:2/int64 out:256x10/float32
Reshape|Reshape in:1x16x4x4/float32 in:2/int64 out:1x256/float32
```

---

## ONNX — run and benchmark (MNIST)

```text
$ nn run mnist-8.onnx    # exit 0
backend:              onnxruntime
latency_ms:           0.300917
outputs:              1
  Plus214_Output_0  float32  1x10
```

Zeros are used when `--input` is omitted and shapes are static. Default backend for `.onnx` is ONNX Runtime. Latency is from this capture; reruns differ.

```text
$ nn run mnist-8.onnx --backend onnxruntime    # exit 0
backend:              onnxruntime
latency_ms:           0.104209
outputs:              1
  Plus214_Output_0  float32  1x10
```

```text
$ nn --json run mnist-8.onnx --backend onnxruntime    # exit 0
{
  "backend": "onnxruntime",
  "latency_ms": 0.104375,
  "outputs": [
    {
      "bytes": 40,
      "dtype": "float32",
      "name": "Plus214_Output_0",
      "shape": [
        1,
        10
      ]
    }
  ],
  "schema_version": 1
}
```

```text
$ nn benchmark mnist-8.onnx --warmup 1 --iterations 5 --backend onnxruntime    # exit 0
backend:              onnxruntime
warmup:               1
iterations:           5
mean_ms:              0.017792
min_ms:               0.015667
p50_ms:               0.016167
p90_ms:               0.016958
p99_ms:               0.016958
max_ms:               0.024125
```

---

## ONNX — tiny Add / Mul graphs

The tiny Add graph is generated by the test helper (`Producer: nn-test`). It is not committed in the repo. `a.npy` / `b.npy` are float32 `1×4` arrays `{1,2,3,4}` and `{10,20,30,40}`. `tensors.npz` is uncompressed `numpy.savez` with members `input0.npy` and `input1.npy`.

```text
$ nn inspect add.onnx    # exit 0
Model
File:                 add.onnx
Format:               onnx
Format version:       8
Framework:            0.1.0
Producer:             nn-test
File size:            149 B
SHA-256:              1166b516844ef243cc8bc46543b0df9479720932b496eb4d339462a639b22a4e

Graph
Graphs:               1
Nodes:                1
Tensors:              3

Parameters
Count:                0
Storage:              0 B

Compute
MACs:                 unknown
FLOPs:                4.00

Inputs
  input0
    shape:            1x4
    dtype:            float32
  input1
    shape:            1x4
    dtype:            float32

Outputs
  output
    shape:            1x4
    dtype:            float32
```

```text
$ nn io add.onnx    # exit 0

Inputs
input0
  shape:              1x4
  dtype:              float32
  layout:             -
  quant:              none
input1
  shape:              1x4
  dtype:              float32
  layout:             -
  quant:              none

Outputs
output
  shape:              1x4
  dtype:              float32
  layout:             -
  quant:              none
```

```text
$ nn ops add.onnx --details    # exit 0
Add_0
  op:                 Add
  canonical:          Elementwise
```

```text
$ nn run add.onnx --input input0=a.npy --input input1=b.npy    # exit 0
backend:              onnxruntime
latency_ms:           0.059750
outputs:              1
  output  float32  1x4
```

```text
$ nn run add.onnx --input tensors.npz --backend reference    # exit 0
backend:              reference
latency_ms:           0.031417
outputs:              1
  output  float32  1x4
```

```text
$ nn run add.onnx --input tensors.npz --backend onnxruntime    # exit 0
backend:              onnxruntime
latency_ms:           0.063584
outputs:              1
  output  float32  1x4
```

```text
$ nn run add.onnx --dump-all --input tensors.npz --output out.npy    # exit 0
backend:              reference
latency_ms:           0.018708
outputs:              1
  output  float32  1x4
dump input0.npy
dump input1.npy
dump output.npy
```

`--dump-all` selects the **reference** backend when `--backend` is omitted, and writes `.npy` dumps next to `--output`.

```text
$ nn profile add.onnx --backend reference --input tensors.npz    # exit 0
latency_ms:           0.016542
node                         op              time_ms
Add_0  Add  0.008916
```

```text
$ nn compare add.onnx add.onnx --input tensors.npz    # exit 0
output
  max_abs:            0.000000
  mean_abs:           0.000000
  rmse:               0.000000
  cosine:             1.000000
  changed:            0 / 4
RESULT: similar
```

```text
$ nn compare add.onnx mul.onnx --input tensors.npz    # exit 1
output
  max_abs:            116.000000
  mean_abs:           48.000000
  rmse:               65.249521
  cosine:             0.970371
  changed:            4 / 4
RESULT: different
```

```text
$ nn compare add.onnx add.onnx --backend onnxruntime --backend2 reference --input tensors.npz    # exit 0
output
  max_abs:            0.000000
  mean_abs:           0.000000
  rmse:               0.000000
  cosine:             1.000000
  changed:            0 / 4
RESULT: similar
```

```text
$ nn compare add.onnx add.onnx --activations --input tensors.npz    # exit 0
output
  max_abs:            0.000000
  mean_abs:           0.000000
  rmse:               0.000000
  cosine:             1.000000
  changed:            0 / 4

Activations
Elementwise output
  max_abs:            0.000000
  mean_abs:           0.000000
  rmse:               0.000000
  cosine:             1.000000
  changed:            0 / 4
input0
  max_abs:            0.000000
  mean_abs:           0.000000
  rmse:               0.000000
  cosine:             1.000000
  changed:            0 / 4
input1
  max_abs:            0.000000
  mean_abs:           0.000000
  rmse:               0.000000
  cosine:             1.000000
  changed:            0 / 4
RESULT: similar
```

```text
$ nn diff add.onnx add.onnx    # exit 0
MODEL DIFFERENCE

Summary
                              OLD             NEW
---------------------------------------------------------
Nodes                          1               1
Parameters                  0.00            0.00
Weight storage               0 B             0 B

Inputs/Outputs
    unchanged
```

Identical files still print a summary table. Exit status 0 means no architecture or I/O change.

```text
$ nn diff add.onnx mul.onnx    # exit 1
MODEL DIFFERENCE

Summary
                              OLD             NEW
---------------------------------------------------------
Nodes                          1               1
Parameters                  0.00            0.00
Weight storage               0 B             0 B

Architecture changes
- Add_0  removed
- Mul_0  added

Inputs/Outputs
    unchanged
```

```text
$ nn diff add.onnx add_wide.onnx    # exit 1
MODEL DIFFERENCE

Summary
                              OLD             NEW
---------------------------------------------------------
Nodes                          1               1
Parameters                  0.00            0.00
Weight storage               0 B             0 B

Architecture changes
- Add_0  modified  output 1x4 float32 -> 1x8 float32

Inputs/Outputs
inputs: input contract changed
outputs: output contract changed
```

```text
$ nn diff add.onnx mul.onnx --weights    # exit 1
MODEL DIFFERENCE

Summary
                              OLD             NEW
---------------------------------------------------------
Nodes                          1               1
Parameters                  0.00            0.00
Weight storage               0 B             0 B

Architecture changes
- Add_0  removed
- Mul_0  added

Inputs/Outputs
    unchanged
```

`--weights` did not add extra rows here: these graphs have no constant payloads.

```text
$ nn optimize add.onnx --dry-run    # exit 0
Proposed changes: 0
```

```text
$ nn convert --list    # exit 0
onnx -> onnx               available
  re-serialize the loaded graph as ONNX
ir -> onnx               available
  write any loaded graph IR as ONNX
onnx -> tflite               unavailable
  requires an external conversion adapter
onnx -> coreml               unavailable
  requires an external conversion adapter
pytorch -> onnx               unavailable
  requires torch.onnx; pickle load is refused
savedmodel -> tflite               unavailable
  requires an external conversion adapter
tflite -> onnx               unavailable
  requires an external conversion adapter
keras -> onnx               unavailable
  requires an external conversion adapter
```

```text
$ nn convert add.onnx --to onnx -o copy.onnx    # exit 0
wrote copy.onnx (onnx)
```

```text
$ nn extract add.onnx --tensor input0 -o extracted.npy    # exit 3
nn: tensor has no payload: input0
```

A graph *input* has no stored payload. The values live in the npy/npz you pass at run time. Use `--tensor` on a constant weight, or `--from` / `--to` to write a subgraph.

---

## TFLite — inspect through execute

```text
$ nn inspect add.tflite    # exit 0
Model
File:                 add.tflite
Format:               tflite
Format version:       3
Framework:            -
Producer:             -
File size:            544 B
SHA-256:              038f8d317c07b094e3b7d0c7aba60a3d9e441eb4188c8515920108b963c3873d

Graph
Graphs:               1
Nodes:                2
Tensors:              3

Parameters
Count:                0
Storage:              0 B

Compute
MACs:                 unknown
FLOPs:                384.0

Inputs
  input
    shape:            1x8x8x3
    dtype:            float32

Outputs
  output
    shape:            1x8x8x3
    dtype:            float32
```

```text
$ nn inspect add.tflite --all    # exit 0
Model
File:                 add.tflite
Format:               tflite
Format version:       3
Framework:            -
Producer:             -
File size:            544 B
SHA-256:              038f8d317c07b094e3b7d0c7aba60a3d9e441eb4188c8515920108b963c3873d

Graph
Graphs:               1
Nodes:                2
Tensors:              3

Parameters
Count:                0
Storage:              0 B

Compute
MACs:                 unknown
FLOPs:                384.0

Inputs
  input
    shape:            1x8x8x3
    dtype:            float32

Outputs
  output
    shape:            1x8x8x3
    dtype:            float32

Metadata
```

```text
$ nn --json inspect add.tflite    # exit 0
{
  "file": "add.tflite",
  "file_size": 544,
  "format": "tflite",
  "format_version": "3",
  "graphs": 1,
  "nodes": 2,
  "parameters": 0,
  "producer": "",
  "schema_version": 1,
  "sha256": "038f8d317c07b094e3b7d0c7aba60a3d9e441eb4188c8515920108b963c3873d",
  "tensors": 3
}
```

```text
$ nn io add.tflite    # exit 0

Inputs
input
  shape:              1x8x8x3
  dtype:              float32
  layout:             -
  quant:              none

Outputs
output
  shape:              1x8x8x3
  dtype:              float32
  layout:             -
  quant:              none
```

```text
$ nn ops add.tflite --canonical    # exit 0
OPERATOR                 COUNT          MACs
------------------------------------------------
Elementwise                   2              -
```

```text
$ nn ops add.tflite --native    # exit 0
OPERATOR                 COUNT          MACs
------------------------------------------------
ADD                           2              -
```

```text
$ nn ops add.tflite --details    # exit 0
ADD_0
  op:                 ADD
  canonical:          Elementwise
ADD_1
  op:                 ADD
  canonical:          Elementwise
```

```text
$ nn tensors add.tflite    # exit 0
NAME                             SHAPE            DTYPE      BYTES  CONST  QUANT
-------------------------------------------------------------------------------
add                              1x8x8x3          float32       768 B    no  none
input                            1x8x8x3          float32       768 B    no  none
output                           1x8x8x3          float32       768 B    no  none
```

```text
$ nn tensors add.tflite --inputs    # exit 0
NAME                             SHAPE            DTYPE      BYTES  CONST  QUANT
-------------------------------------------------------------------------------
input                            1x8x8x3          float32       768 B    no  none
```

```text
$ nn tensors add.tflite --outputs    # exit 0
NAME                             SHAPE            DTYPE      BYTES  CONST  QUANT
-------------------------------------------------------------------------------
output                           1x8x8x3          float32       768 B    no  none
```

```text
$ nn metadata add.tflite    # exit 0
Format:               tflite
Producer:             
Version:              3
```

```text
$ nn hash add.tflite    # exit 0
artifact:             038f8d317c07b094e3b7d0c7aba60a3d9e441eb4188c8515920108b963c3873d
graph:                59c84adccea3a08eeaa605110324b34d524bb958081dc9c67ecf68d5fb07f035
weights:              e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

`weights` is SHA-256 of empty content (`e3b0c442…`). This fixture has no constant payloads.

```text
$ nn --porcelain hash add.tflite --graph    # exit 0
graph:	59c84adccea3a08eeaa605110324b34d524bb958081dc9c67ecf68d5fb07f035
```

```text
$ nn memory add.tflite    # exit 0
Memory Analysis

Weights               0 B
Persistent tensors    768 B
Peak live activations  1.50 KB
Estimated scratch     0 B
--------------------------------------
Estimated RAM requirement  1.50 KB
Flash/model storage   544 B
```

```text
$ nn memory add.tflite --plan    # exit 0
Memory Analysis

Weights               0 B
Persistent tensors    768 B
Peak live activations  1.50 KB
Estimated scratch     1.50 KB
--------------------------------------
Estimated RAM requirement  1.50 KB
Flash/model storage   544 B

Tensor                   Start       Size        Lifetime
---------------------------------------------------------
add                             0        768      0-1
output                        768        768      1-2
```

```text
$ nn compute add.tflite    # exit 0
MACs:                 unknown
FLOPs:                384.0
Integer ops:          unknown
Float ops:            384.0
Unknown nodes:        0
```

```text
$ nn quant add.tflite    # exit 0
Quantized tensors:    0
Float tensors:        3
Integer tensors:      0
Per-channel:          0
Per-tensor:           0
Quantize nodes:       0
Dequantize nodes:     0
```

```text
$ nn sparsity add.tflite    # exit 0
Tensors considered:   0
Tensors computed:     0
Zero fraction:        0.000000
Near-zero fraction:   0.000000
Near-zero |w| <=:     0.000000
Weak-channel cut:     1.00% of max channel L1
Total MACs:           unknown
Est. saved bytes:     0 B (upper bound)
Est. saved MACs:      unknown

LAYER                OP         SHAPE            ZEROS       NEAR        WEAK     MAC%    SCORE
-----------------------------------------------------------------------------------------------

Candidates only; nn does not prune. Magnitude is a where-to-look hint, not accuracy.
```

```text
$ nn lint add.tflite    # exit 0
errors:               0
warnings:             0
```

```text
$ nn graph add.tflite --format text    # exit 0
0  ADD  ADD_0
1  ADD  ADD_1
```

```text
$ nn compat add.tflite --runtime reference    # exit 0
Runtime:              reference
Capability table:     0.1.0
Notes:                built-in reference interpreter; compiled into this binary
Nodes supported:      2 / 2
RESULT: compatible (vs reference table 0.1.0)
```

```text
$ nn target add.tflite --target cortex-m4f    # exit 0
Target: cortex-m4f
RAM:                  256.0 KB
Flash:                1.00 MB

Storage
    Model:            544 B       PASS

RAM
    Activations:      1.50 KB
    Scratch:          0 B
    Runtime overhead:  24.0 KB
    Total:            25.5 KB       PASS

Result:
    MODEL FITS
    runtime overhead is estimated (24 KiB), not measured
```

```text
$ nn run add.tflite    # exit 0
backend:              litert
latency_ms:           0.256208
outputs:              1
  x  float32  1x8x8x3

# stderr
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
```

Default backend for `.tflite` is LiteRT. The runtime output name is `x`; `nn io` reported the IR name `output`. Compare matches by index when names differ.

```text
$ nn run add.tflite --backend litert    # exit 0
backend:              litert
latency_ms:           0.015292
outputs:              1
  x  float32  1x8x8x3

# stderr
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
```

```text
$ nn --json run add.tflite --backend litert    # exit 0
{
  "backend": "litert",
  "latency_ms": 0.0145,
  "outputs": [
    {
      "bytes": 768,
      "dtype": "float32",
      "name": "x",
      "shape": [
        1,
        8,
        8,
        3
      ]
    }
  ],
  "schema_version": 1
}

# stderr
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
```

```text
$ nn benchmark add.tflite --warmup 1 --iterations 5    # exit 0
backend:              litert
warmup:               1
iterations:           5
mean_ms:              0.000700
min_ms:               0.000500
p50_ms:               0.000583
p90_ms:               0.000666
p99_ms:               0.000666
max_ms:               0.001208

# stderr
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
```

---

## TFLite — compare and diff

```text
$ nn compare add.tflite add.tflite    # exit 0
Structural comparison (no inputs supplied)
identical:            yes
```

```text
$ nn compare add.tflite add.tflite --input a.npy    # exit 0
x
  max_abs:            0.000000
  mean_abs:           0.000000
  rmse:               0.000000
  cosine:             1.000000
  changed:            0 / 192
RESULT: similar

# stderr
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
```

Same file compared to itself; `changed: 0 / 192` is the output element count (`1×8×8×3`). `a.npy` is `1×4` and does not match the TFLite input shape; both sides received the same bind, so the numeric compare still reports similar.

```text
$ nn diff add.tflite add.tflite    # exit 0
MODEL DIFFERENCE

Summary
                              OLD             NEW
---------------------------------------------------------
Nodes                          2               2
Parameters                  0.00            0.00
Weight storage               0 B             0 B

Inputs/Outputs
    unchanged
```

---

## Cross-format ONNX vs TFLite

```text
$ nn compare add.onnx add.tflite    # exit 1
Structural comparison (no inputs supplied)
identical:            no
```

Without `--input`, `compare` is structural (`diff`), not numeric.

```text
$ nn compare add.onnx add.tflite --input tensors.npz    # exit 2
nn: could not match npz arrays to inputs in tensors.npz
```

A two-array `tensors.npz` (`input0` / `input1`) does not match TFLite’s single `input`.

```text
$ nn diff mnist-8.onnx add.tflite    # exit 1
MODEL DIFFERENCE

Summary
                              OLD             NEW
---------------------------------------------------------
Nodes                         12               2
Parameters                6.00 K            0.00
Weight storage           23.4 KB             0 B

Architecture changes
- Times212_reshape1  removed
- Convolution28  removed
- Plus30  removed
- ReLU32  removed
- Pooling66  removed
- Convolution110  removed
- Plus112  removed
- ReLU114  removed
- Pooling160  removed
- Times212_reshape0  removed
- Times212  removed
- Plus214  removed
- ADD_0  added
- ADD_1  added

Inputs/Outputs
inputs: input contract changed
outputs: output contract changed
```

```text
$ nn quant compare mnist-8.onnx add.tflite    # exit 0
Quantization comparison
Float tensors (A/B):  19 / 3
Quantized tensors (A/B):  0 / 0
Per-layer activation error: unavailable (no test vectors supplied)
```

---

## Shell one-liners

```bash
# Format of a file
nn --json inspect model.onnx | jq -r .format

# Artifact hash for a lockfile
nn --porcelain hash model.onnx | awk -F'\t' '/^artifact:/{print $2}'

# Fail CI if lint errors exist
nn lint model.onnx

# Fail CI if two ONNX files differ structurally
nn diff old.onnx new.onnx; echo $?   # 0 identical, 1 different

# Run MNIST zeros through ONNX Runtime
nn run mnist-8.onnx --backend onnxruntime

# Run TFLite add through LiteRT
nn run add.tflite --backend litert

# Check MCU budget
nn target model.tflite --target cortex-m4f

# Rank prune-candidate layers (inspect only)
nn sparsity model.onnx --threshold 1e-6
nn --json sparsity model.onnx | jq '.layers[] | select(.score > 0)'
```

## See also

- [user-manual.md](user-manual.md) — every flag
- [exit-status.md](exit-status.md) — process codes
- [json-schema.md](json-schema.md) — JSON output
- `nn help <command>`
