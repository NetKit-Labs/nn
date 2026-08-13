# Contributing

Build with warnings enabled and keep `ctest` green.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

- C++20, RAII, no global mutable state
- Do not claim format/runtime capabilities that are not compiled in
- Prefer `Result<T>` / `Status` over thrown exceptions
- Parser code must bounds-check; treat artifacts as untrusted
- Tests: unit tests for math/IR, integration tests for each implemented reader
- Do not commit model binaries; generate tiny fixtures in tests

Command help is defined next to the command spec in `src/cli/command.cpp`. Keep help and flags in sync.
