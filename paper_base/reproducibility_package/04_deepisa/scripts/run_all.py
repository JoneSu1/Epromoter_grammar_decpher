from __future__ import annotations

import runpy
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "fig2_screening_null_distribution.py",
    "export_selected_verified_panels.py",
    "fig3de_discovery_summary.py",
    "fig3g_top_pairs.py",
    "fig3g_top_pairs_dotmatrix.py",
    "fig3h_complete_matrix.py",
    "fig4b_positive_negative_distance.py",
    "fig4_shared_tf_pair_distance.py",
    "audit_fig4_shared_tf_pair_distance.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n=== {script} ===")
        runpy.run_path(str(HERE / script), run_name="__main__")


if __name__ == "__main__":
    main()
