import json

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
    assert check_registry() == []


def test_final_visual_archive_is_intact():
    assert verify_visual_assets() == []


def test_packaged_registry_matches_checkout_registry():
    packaged = registry_path().parent.parent / "src" / "drosophila_repro" / "configs" / "figure_registry.json"
    assert json.loads(packaged.read_text(encoding="utf-8")) == json.loads(
        (REPOSITORY_ROOT / "configs" / "figure_registry.json").read_text(encoding="utf-8")
    )
