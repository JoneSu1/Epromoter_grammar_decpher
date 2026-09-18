import json
from pathlib import Path

import pytest

from drosophila_repro.registry import REPOSITORY_ROOT, check_registry, load_registry, registry_path, verify_visual_assets


def test_registry_covers_six_final_figures():
    assert set(load_registry()["figures"]) == {f"figure{i}" for i in range(1, 7)}


def test_every_target_has_script_and_input_contract():
    for figure in load_registry()["figures"].values():
        assert figure["question"]
        assert figure["targets"]
        for target in figure["targets"]:
            assert target["script"] and target["required_root"]


def test_source_archive_candidate_satisfies_registry():
    candidate = REPOSITORY_ROOT.parents[1] / "reproducibility_package"
    if not candidate.is_dir():
        pytest.skip("Frozen release assets are intentionally absent from a Git-only clone.")
    assert check_registry() == []


def test_final_visual_archive_is_intact():
    candidate = REPOSITORY_ROOT.parents[1] / "reproducibility_package"
    if not candidate.is_dir():
        pytest.skip("Frozen release assets are intentionally absent from a Git-only clone.")
    assert verify_visual_assets() == []


def test_packaged_registry_matches_checkout_registry():
    packaged = Path(__import__("drosophila_repro").__file__).parent / "configs" / "figure_registry.json"
    assert json.loads(packaged.read_text(encoding="utf-8")) == json.loads(
        (REPOSITORY_ROOT / "configs" / "figure_registry.json").read_text(encoding="utf-8")
    )


def test_missing_asset_root_is_reported_not_crashed(tmp_path):
    errors = check_registry(tmp_path)
    assert errors
    assert all("missing" in error for error in errors)
