"""Create a ZIP64 release-asset archive and adjacent SHA-256 checksum.

The archive is deliberately a release attachment/DOI deposit, not Git content.
It preserves the package-root layout expected by ``drosophila-repro``.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


EXCLUDED_DIRS = {"__pycache__", ".ipynb_checkpoints"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
UNRESOLVED_DEEPISA_PREFIX = Path("04_deepisa/scripts/Ep_ISA_NEW_src")


def archive_members(source: Path, include_unresolved_deepisa: bool):
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if not include_unresolved_deepisa and relative.is_relative_to(UNRESOLVED_DEEPISA_PREFIX):
            continue
        yield path, relative


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Audited reproducibility_package directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination .zip path")
    parser.add_argument("--compression", choices=["stored", "deflated"], default="stored")
    parser.add_argument(
        "--include-unresolved-deepisa",
        action="store_true",
        help="Internal audit only: include Ep_ISA_NEW_src despite unresolved upstream redistribution rights.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not (source / "MANIFEST.md").is_file():
        parser.error(f"{source} is not a release-asset root (MANIFEST.md missing)")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_STORED if args.compression == "stored" else zipfile.ZIP_DEFLATED

    count = 0
    with zipfile.ZipFile(output, "w", compression=compression, allowZip64=True) as archive:
        for path, relative in archive_members(source, args.include_unresolved_deepisa):
            archive.write(path, Path(source.name) / relative)
            count += 1
    checksum = sha256(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    print(f"archive: {output}")
    print(f"members: {count}")
    print(f"bytes: {output.stat().st_size}")
    print(f"sha256: {checksum}")
    print(f"unresolved-deepisa-included: {args.include_unresolved_deepisa}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
