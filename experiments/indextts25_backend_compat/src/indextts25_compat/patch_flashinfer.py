"""Patch a FlashInfer 0.6.16 annotation that is invalid on Python 3.11."""

from __future__ import annotations

import importlib.util
from pathlib import Path


OLD_ANNOTATION = "array.array[int]"
NEW_ANNOTATION = "array.array"


def patch_file(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    if OLD_ANNOTATION in source:
        path.write_text(source.replace(OLD_ANNOTATION, NEW_ANNOTATION), encoding="utf-8")
        return f"patched {path}"
    if NEW_ANNOTATION in source:
        return f"already compatible: {path}"
    raise RuntimeError(f"FlashInfer compatibility target changed; inspect {path} before continuing")


def main() -> None:
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("flashinfer is not installed in the active environment")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    target = package_root / "comm" / "fd_exchange.py"
    if not target.is_file():
        raise RuntimeError(f"FlashInfer compatibility target is missing: {target}")
    print(patch_file(target))


if __name__ == "__main__":
    main()
