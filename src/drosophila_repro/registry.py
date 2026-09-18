"""Release-registry loading and path validation; intentionally stdlib-only."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path


# registry.py -> drosophila_repro -> src -> repository root
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def registry_path() -> Path:
    """Use the checkout registry when available, otherwise the wheel payload."""
    checkout_path = REPOSITORY_ROOT / "configs" / "figure_registry.json"
    if checkout_path.is_file():
        return checkout_path
    return Path(__file__).parent / "configs" / "figure_registry.json"


def load_registry() -> dict:
    return json.loads(registry_path().read_text(encoding="utf-8"))


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
        if entry.get("manual_assets_required"):
            composite = package_root / "frozen_data" / "manuscript_visual_assets" / "main_figures_png" / f"{figure}.png"
            if not composite.is_file():
                errors.append(f"{figure}: missing final manuscript composite: {composite}")
    return errors


def verify_visual_assets(package_root: Path | None = None) -> list[str]:
    """Check final visual assets against the frozen size/SHA-256 inventory."""
    package_root = package_root or resolve_package_root()
    visual_root = package_root / "frozen_data" / "manuscript_visual_assets"
    manifest = visual_root / "MANIFEST.sha256"
    if not manifest.is_file():
        return [f"missing visual asset manifest: {manifest}"]

    errors: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            relative, byte_count, expected_hash = line.split("\t")
        except ValueError:
            errors.append(f"malformed visual manifest line: {line}")
            continue
        path = visual_root / relative
        if not path.is_file():
            errors.append(f"missing visual asset: {path}")
            continue
        if path.stat().st_size != int(byte_count):
            errors.append(f"size mismatch: {relative}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            errors.append(f"SHA-256 mismatch: {relative}")
    return errors
