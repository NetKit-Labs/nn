# JSON output

Machine-readable commands include:

```json
{ "schema_version": 1 }
```

Do not change field names casually. Additive keys are allowed. Bump `schema_version` for breaking changes.

Commands that emit `schema_version` (currently `1`):

- `inspect`, `ops`, `tensors`, `io`, `metadata`, `hash`
- `diff`, `graph`, `memory`, `compute`, `quant`, `sparsity`
- `run`, `formats`

Not every command implements JSON. `lint`, `compare`, `target`, and others are text (or porcelain) unless listed above.

## sparsity

Top-level object:

| Key | Meaning |
| --- | --- |
| `tensors_considered` | Constants considered |
| `tensors_computed` | Constants whose payload was in memory |
| `overall_zero_fraction` / `overall_near_zero_fraction` | Across computed tensors |
| `threshold` | Absolute `|w|` near-zero line (`--threshold`) |
| `channel_l1_frac` | Weak-channel cutoff (`--channel-frac`) |
| `total_macs` / `total_macs_known` | Graph MAC estimate when shapes allow |
| `estimated_saved_bytes` / `estimated_saved_macs` | If weak channels were dropped |
| `savings_are_upper_bound` | Always `true` |
| `layers` | Per-constant rows, ranked like the text table |
| `notes` | Residual / depthwise coupling caveats |

Each `layers[]` object includes `tensor`, `layer`, `op`, `canonical`, `shape`, `bytes`, `elements`, `zeros`, `near_zeros`, `zero_fraction`, `near_zero_fraction`, `computed`, `layout`, `channels`, `weak_channels`, `weak_channel_frac`, `max_channel_l1`, `macs`, `macs_known`, `mac_share`, `score`, `estimated_saved_bytes`, `estimated_saved_macs`, `skip_coupled`, `depthwise`.

`layout` is `conv-onnx`, `conv-tflite`, `dense-out-rows`, `dense-out-cols`, or empty when the tensor is not scored as conv/dense. When `computed` is false (payload not in the file), zero and channel stats are not scored; the row may still carry MAC share.

```bash
nn --json sparsity model.onnx --threshold 1e-6 | jq '.layers[] | select(.score > 0)'
```

## graph

```json
{ "schema_version": 1, "nodes": [ { "id": 0, "name": "…", "op": "Conv" } ] }
```

Also produced by `nn graph --format json`.

## run

```json
{
  "schema_version": 1,
  "backend": "onnxruntime",
  "latency_ms": 1.23,
  "outputs": [ { "name": "…", "dtype": "float32", "shape": [1, 10], "bytes": 40 } ]
}
```

Output payloads are not included.

## formats

```json
{
  "schema_version": 1,
  "formats": [
    { "name": "onnx", "display": "ONNX", "read": true, "graph": true, "weights": true, "execute": true, "convert": true, "notes": "…" }
  ]
}
```
