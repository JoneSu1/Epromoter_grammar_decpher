from drosophila_repro.registry import check_registry, load_registry, verify_visual_assets


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
