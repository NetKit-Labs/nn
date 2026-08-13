#!/usr/bin/env python3
"""Optional helper to emit extra fixtures. Core tests generate models in C++."""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Generate extra NN test models (optional).")
    p.add_argument("--out", default="tests/models")
    args = p.parse_args()
    print(f"No extra models generated; C++ tests write tiny ONNX/GGUF/SafeTensors under {args.out} temp paths.")


if __name__ == "__main__":
    main()
