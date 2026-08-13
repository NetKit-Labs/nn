# Security

Model files are untrusted input.

- Parsers must bounds-check every offset and length
- Tensor dimensions and allocation sizes are overflow-checked
- PyTorch pickle payloads are **not** deserialized by default
- Embedded code in artifacts is never executed
- Do not store secrets in `nn` config files

Report vulnerabilities privately to the maintainers. Do not open a public issue with exploit details.
