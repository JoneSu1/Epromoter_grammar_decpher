"""Small, data-free guards for the Drive-backed sharing rerun workflow."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "hk_dev_shared_pipeline.py"
SPEC = spec_from_file_location("hk_dev_shared_pipeline", PIPELINE)
pipeline = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pipeline)


def test_coordinate_parser_keeps_model_window_prefix():
    assert pipeline.coordinates("chr2L_100_349_+_positive_peaks") == ("chr2L", 100, 349)


def test_track_contract_separates_s3_and_deepisa():
    cohort = pd.DataFrame(
        [
            {"canonical_id": "chr2L_1_250_+_a", "chrom": "chr2L", "start": 1, "end": 250, "coordinate": "chr2L:1-250", "fig1_promoter_group": "proximal_promoter", "cage_observed": True, "deepstarr_sequence": "A" * 249, "cage_sequence": "C" * 249},
            {"canonical_id": "chr2L_2_251_+_b", "chrom": "chr2L", "start": 2, "end": 251, "coordinate": "chr2L:2-251", "fig1_promoter_group": "distal_promoter", "cage_observed": False, "deepstarr_sequence": "G" * 249, "cage_sequence": ""},
        ]
    )
    assert len(pipeline.track_frame(cohort, "s3_hk")) == 2
    assert len(pipeline.track_frame(cohort, "s3_dev")) == 2
    assert len(pipeline.track_frame(cohort, "deepisa_hk")) == 1
    assert len(pipeline.track_frame(cohort, "deepisa_dev")) == 1
    cage = pipeline.track_frame(cohort, "deepisa_cage")
    assert len(cage) == 1
    assert cage.iloc[0].sequence == "C" * 249
