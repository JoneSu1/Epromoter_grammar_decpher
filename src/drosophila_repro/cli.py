"""Small command-line interface for readers, reproductions, and developers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .registry import REPOSITORY_ROOT, check_registry, load_registry, resolve_package_root, verify_visual_assets


def _selected(name: str) -> list[tuple[str, dict]]:
    figures = load_registry()["figures"]
    if name == "all":
        return list(figures.items())
    return [(name, figures[name])]


def doctor() -> int:
    root = resolve_package_root()
    print(f"release assets: {root}")
    print(f"registry: {REPOSITORY_ROOT / 'configs' / 'figure_registry.json'}")
    errors = check_registry(root)
    print("doctor: PASS" if not errors else f"doctor: FAIL ({len(errors)} missing paths)")
    for error in errors:
        print(f"  {error}")
    return int(bool(errors))


def assets_verify() -> int:
    root = resolve_package_root()
    errors = check_registry(root) + verify_visual_assets(root)
    print("assets-verify: PASS" if not errors else f"assets-verify: FAIL ({len(errors)} issue(s))")
    for error in errors:
        print(f"  {error}")
    return int(bool(errors))


def reproduce(figure: str, dry_run: bool, continue_on_error: bool) -> int:
    root = resolve_package_root()
    failures = 0
    for figure_name, entry in _selected(figure):
        print(f"\n== {figure_name}: {entry['question']}")
        for target in entry["targets"]:
            script = root / target["script"]
            required = root / target["required_root"]
            if not script.exists() or not required.exists():
                print(f"MISSING: {script if not script.exists() else required}")
                failures += 1
                if not continue_on_error:
                    return 1
                continue
            print(f"RUN: {target['script']}")
            if not dry_run:
                result = subprocess.run([sys.executable, str(script)], cwd=script.parent)
                if result.returncode:
                    failures += 1
                    if not continue_on_error:
                        return 1
    return int(bool(failures))


def extension_init(name: str) -> int:
    if not name.replace("_", "").isalnum() or not name:
        raise ValueError("Extension name must contain letters, numbers, and underscores only.")
    target = Path.cwd() / name
    target.mkdir(exist_ok=False)
    (target / "README.md").write_text(
        f"# {name}\n\nQuestion:\n\nInput contract:\n\nOutput contract:\n\nValidation:\n",
        encoding="utf-8",
    )
    (target / "analysis.py").write_text('"""Extension entry point; do not mutate frozen release assets."""\n', encoding="utf-8")
    print(f"Created extension scaffold: {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drosophila-repro")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("assets-verify")
    sub.add_parser("list")
    sub.add_parser("registry-check")
    render = sub.add_parser("reproduce")
    render.add_argument("--figure", choices=["all", "figure1", "figure2", "figure3", "figure4", "figure5", "figure6"], default="all")
    render.add_argument("--dry-run", action="store_true")
    render.add_argument("--continue-on-error", action="store_true")
    init = sub.add_parser("extension-init")
    init.add_argument("--name", required=True)
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return doctor()
    if args.command == "assets-verify":
        return assets_verify()
    if args.command == "list":
        print(json.dumps(load_registry()["figures"], indent=2, ensure_ascii=False))
        return 0
    if args.command == "registry-check":
        errors = check_registry()
        for error in errors:
            print(error)
        return int(bool(errors))
    if args.command == "reproduce":
        return reproduce(args.figure, args.dry_run, args.continue_on_error)
    return extension_init(args.name)


if __name__ == "__main__":
    raise SystemExit(main())
