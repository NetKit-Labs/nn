# Exit status

| Code | Meaning |
| --- | --- |
| 0 | success |
| 1 | comparison difference / policy failure (`diff`, `target`, `compat`, `regression`) |
| 2 | usage error |
| 3 | file error |
| 4 | malformed model |
| 5 | unsupported format |
| 6 | unsupported operator |
| 7 | backend / conversion adapter unavailable |
| 8 | execution failure |
| 9 | validation / lint errors |
| 10 | internal error |

`nn diff` returns 0 when no relevant difference is found and 1 when differences exist.
