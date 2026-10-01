"""Canonical repository audit helpers shared by the Makefile, CI and the test suite.

The forbidden-extension set lives here so local `make audit`, GitHub Actions and pytest
can never drift apart.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FORBIDDEN_TRACKED_SUFFIXES = (".bin", ".gguf", ".onnx", ".pt", ".pth", ".safetensors")
MARKER = "MODEL_WEIGHTS_TRACKED"


def is_forbidden(path: str) -> bool:
    return str(path).lower().endswith(FORBIDDEN_TRACKED_SUFFIXES)


def tracked_files(root: Path | None = None) -> list[str]:
    root = Path(root or Path.cwd())
    output = subprocess.check_output(["git", "ls-files"], cwd=root, text=True)
    return [line for line in output.splitlines() if line.strip()]


def forbidden_tracked_files(paths: list[str]) -> list[str]:
    return sorted(path for path in paths if is_forbidden(path))


def main() -> int:
    offenders = forbidden_tracked_files(tracked_files())
    if offenders:
        print("tracked model or binary weight artifact detected:", file=sys.stderr)
        for path in offenders:
            print(f"  {path}", file=sys.stderr)
        return 1
    print(f"{MARKER}=NO")
    print(f"AUDIT_FORBIDDEN_SUFFIXES={','.join(FORBIDDEN_TRACKED_SUFFIXES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())