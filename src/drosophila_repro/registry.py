"""Release-registry loading and path validation; intentionally stdlib-only."""

from __future__ import annotations

import json
import os
from pathlib import Path


# registry.py -> drosophila_repro -> src -> repository root
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_registry() -> dict:
    path = REPOSITORY_ROOT / "configs" / "figure_registry.json"
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_package_root() -> Path:
    configured = os.environ.get("DROSOPHILA_REPRO_PACKAGE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    published = REPOSITORY_ROOT / "release_assets"
    # A source checkout intentionally carries an empty, tracked mount point so
    # release data are never committed by accident.  Only prefer that mount
    # once a deposited archive has supplied its manifest.
    if published.exists() and (published / "MANIFEST.md").is_file():
        return published.resolve()
    # Source-archive development default; not used by a released checkout.
    return (REPOSITORY_ROOT.parents[1] / "reproducibility_package").resolve()


def check_registry(package_root: Path | None = None) -> list[str]:
    package_root = package_root or resolve_package_root()
    errors: list[str] = []
    for figure, entry in load_registry()["figures"].items():
        for target in entry["targets"]:
            for field in ("script", "required_root"):
                path = package_root / target[field]
                if not path.exists():
                    errors.append(f"{figure}: missing {field}: {path}")
    return errors
