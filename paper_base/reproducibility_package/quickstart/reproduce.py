#!/usr/bin/env python3
"""Package-local orchestration for manuscript Figure 1-6 reproduction.

The registry deliberately uses the package candidate scripts, not formal_script.
It never changes scientific parameters; it merely invokes the registered
canonical renderer from that renderer's own directory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Target:
    manuscript_figure: str
    script: str
    required_root: str
    required_inputs: tuple[str, ...]
    note: str


# One registry, ordered by the final manuscript's six-figure narrative.
TARGETS = (
    Target("figure1", "01_model_build/deepstarr/scripts/run_all.py", "01_model_build/deepstarr", ("01_model_build/deepstarr/data/prediction_output/prediction_train.tsv", "01_model_build/deepstarr/data/overlap_annotation/deepstarr_overlap_development.tsv"), "Computational panels; manual schematic is a release asset."),
    Target("figure2", "01_model_build/deepcage/scripts/run_all.py", "01_model_build/deepcage", ("01_model_build/deepcage/data/DATA/Merged_All_Data_Predictions_Splits.tsv", "01_model_build/deepcage/data/DATA/PROMOTER_Dominant_True_Pred_Verify.tsv"), "Computational panels; manual schematic is a release asset."),
    Target("figure3", "quickstart/adapters/figure3_atlas.py", "02_sharing_motif/data", ("02_sharing_motif/data/clean/shared_motif_density_complexity.csv",), "Three frozen motif-atlas panels via package-local path adapter."),
    Target("figure3", "quickstart/adapters/figure3_promoter_logos.py", "frozen_data/sharing_motif_logos", ("frozen_data/sharing_motif_logos/attributions/HK/attributions.h5", "frozen_data/sharing_motif_logos/finemo_input/HK/finemo_input.npz"), "Promoter logos via localized input adapter."),
    Target("figure4", "03_motif_analysis/scripts/fig3b_tss_distance.py", "03_motif_analysis", ("03_motif_analysis/data/clean/shared_motif_hit_contribution_tss.csv",), "TSS deployment."),
    Target("figure4", "03_motif_analysis/scripts/fig3c_density_complexity.py", "03_motif_analysis", ("03_motif_analysis/data/S3_clean/motif_density_complexity_by_sequence.csv",), "Density and complexity."),
    Target("figure4", "03_motif_analysis/scripts/fig3d_contribution.py", "03_motif_analysis", ("03_motif_analysis/data/S3_clean/motif_contribution_hits.csv",), "Attribution contribution."),
    Target("figure5", "04_deepisa/scripts/fig2_screening_null_distribution.py", "frozen_data/deepisa_raw", ("frozen_data/deepisa_raw/results_hk_newisa/null_isa.csv",), "Frozen ISA null calibration."),
    Target("figure5", "04_deepisa/scripts/fig3de_discovery_summary.py", "04_deepisa", ("04_deepisa/data/position_aware_summary/selected_pair_interactions.csv",), "Pair discovery summary."),
    Target("figure5", "04_deepisa/scripts/fig3g_top_pairs.py", "04_deepisa", ("04_deepisa/data/position_aware_summary/selected_pair_interactions.csv",), "Top pair resolution."),
    Target("figure5", "04_deepisa/scripts/fig3h_complete_matrix.py", "04_deepisa", ("04_deepisa/data/position_aware_summary/selected_pair_interactions.csv",), "Complete pair matrix."),
    Target("figure5", "quickstart/adapters/figure5_distance.py", "04_deepisa/data/fig4_pos_neg", ("04_deepisa/data/fig4_pos_neg/fig4_positive_negative_distance_summary.csv",), "Distance/context result via package-local path adapter."),
    Target("figure6", "05_evolution/scripts/plot_fig5_polish2.py", "05_evolution", ("05_evolution/data/processed/primary_greedy_summary.tsv", "frozen_data/evolution_fig5d/attribution/HK/attribution_arrays.npz", "frozen_data/evolution_fig5d/summaries/finemo_hits_annotated_long.tsv"), "Frozen trajectory summary."),
    Target("figure6", "quickstart/adapters/figure6_branch_pairs.py", "frozen_data/evolution_fig5d", ("frozen_data/evolution_fig5d/attribution/HK/attribution_arrays.npz",), "Representative attribution plates via localized-data adapter."),
    Target("figure6", "quickstart/adapters/figure6_motif_bars.py", "05_evolution", ("05_evolution/data/processed/motif_new_scatter.tsv",), "Target-versus-CAGE motif acquisition with release-compatible series aliases."),
)


def selected(group: str) -> tuple[Target, ...]:
    if group == "all":
        return TARGETS
    return tuple(t for t in TARGETS if t.manuscript_figure == group)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["all", "figure1", "figure2", "figure3", "figure4", "figure5", "figure6"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Check registered scripts/inputs without rendering.")
    parser.add_argument("--output-root", type=Path, default=PACKAGE / "reproduced", help="Directory for newly rendered outputs; never defaults to frozen figure directories.")
    parser.add_argument("--continue-on-error", action="store_true", help="Attempt later targets after one renderer fails.")
    parser.add_argument("--list", action="store_true", help="List the release registry and exit.")
    args = parser.parse_args()

    targets = selected(args.group)
    if args.list:
        for target in TARGETS:
            print(f"{target.manuscript_figure}: {target.script} | requires {target.required_root}")
        return 0

    failed = 0
    for target in targets:
        script = PACKAGE / target.script
        required = PACKAGE / target.required_root
        missing_inputs = [item for item in target.required_inputs if not (PACKAGE / item).is_file()]
        ready = script.is_file() and required.is_dir() and not missing_inputs
        state = "READY" if ready else "MISSING"
        print(f"[{state}] {target.manuscript_figure}: {target.script}\n        {target.note}")
        if not ready:
            for item in missing_inputs:
                print(f"        missing input: {item}")
            failed += 1
            if not args.continue_on_error:
                break
            continue
        if args.dry_run:
            continue
        env = os.environ.copy()
        env["REPRO_OUTPUT_ROOT"] = str(args.output_root.resolve())
        result = subprocess.run([sys.executable, str(script)], cwd=script.parent, env=env)
        if result.returncode:
            failed += 1
            if not args.continue_on_error:
                break

    print("Manual conceptual panels and final composite assembly are release assets, not Python render targets.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
