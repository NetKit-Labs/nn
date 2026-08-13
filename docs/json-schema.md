# JSON output

Machine-readable commands include:

```json
{ "schema_version": 1 }
```

Do not change field names casually. Additive keys are allowed. Bump `schema_version` for breaking changes.

Commands with JSON:

- `inspect`, `ops`, `tensors`, `io`, `metadata`, `hash`
- `diff`, `memory`, `compute`, `quant`, `formats`
