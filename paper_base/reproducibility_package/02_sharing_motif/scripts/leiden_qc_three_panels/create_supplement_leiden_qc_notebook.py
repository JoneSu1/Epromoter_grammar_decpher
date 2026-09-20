"""Create a Colab-ready supplement notebook for shared-motif Leiden QC.

The notebook reuses the stable meta-clustering functions from Motif_discover.ipynb
cells 63-68, but replaces notebook-only `plt.show()` outputs with reproducible
CSV/NPZ source data and SVG/PDF/TIFF/PNG exports.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_NB = HERE / "Motif_discover.ipynb"
OUT_NB = HERE / "Supplement_Leiden_QC_Colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def source_cell(nb: dict, idx: int) -> str:
    return "".join(nb["cells"][idx].get("source", []))


def main() -> None:
    nb = json.loads(SOURCE_NB.read_text(encoding="utf-8"))

    # Original dependency/function cells needed for section 9.
    cell_63 = source_cell(nb, 63)
    cell_64 = source_cell(nb, 64)
    cell_65 = source_cell(nb, 65)
    cell_66 = source_cell(nb, 66)
    cell_67 = source_cell(nb, 67)
    cell_68 = source_cell(nb, 68)

    # Remove notebook-local pip line from original config; install is handled in cell 1.
    cell_63 = cell_63.replace("!pip install leidenalg igraph -q\n\n", "")
    cell_63 = cell_63.replace(
        'TASK_COLORS = {"DEV": "#009E73", "HK": "#0072B2", "CAGE_NEW": "#D55E00"}',
        'TASK_COLORS = {"DEV": "#009E73", "HK": "#0072B2", "CAGE_NEW": "#D64F4F"}',
    )
    cell_63 = cell_63.replace(
        'print("✅ Paths configured. Ready for CTSS vs HK vs DEV.")',
        """
from pathlib import Path

OUT_ROOT = Path(BASE_DIR) / "Motif_cluster" / "supplement_leiden_qc"
DATA_OUT = OUT_ROOT / "data"
FIG_OUT = OUT_ROOT / "figures"
QA_OUT = OUT_ROOT / "qa"
for _p in [DATA_OUT, FIG_OUT, QA_OUT]:
    _p.mkdir(parents=True, exist_ok=True)

missing = [p for p in TASKS_H5_PATHS.values() if not os.path.exists(p)]
if missing:
    raise FileNotFoundError("Missing input H5 files:\\n" + "\\n".join(missing))

print("✅ Paths configured. Ready for CTSS/CAGE_NEW vs HK vs DEV.")
print("Output root:", OUT_ROOT)
""".strip(),
    )

    publication_plotting = r'''
# ==========================================
# Publication-style QC plotting and export
# ==========================================

import json
import textwrap
from datetime import datetime

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

def save_pub(fig, stem, dpi=600):
    paths = {}
    for ext in ["svg", "pdf", "png", "tiff"]:
        out = FIG_OUT / f"{stem}.{ext}"
        if ext == "tiff":
            fig.savefig(out, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        elif ext == "png":
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(out, bbox_inches="tight")
        paths[ext] = str(out)
    return paths

def save_matrix_table(mat, labels, stem):
    df = pd.DataFrame(mat, index=labels, columns=labels)
    out = DATA_OUT / f"{stem}.csv"
    df.to_csv(out)
    return out

def plot_leiden_trajectory_pub(quality_history, best_seed, best_quality):
    q = np.asarray(quality_history, dtype=float)
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    x = np.arange(1, len(q) + 1)
    ax.plot(x, q, marker="o", markersize=3.5, linewidth=1.1, color="#4C78A8")
    ax.scatter([best_seed], [best_quality], marker="*", s=90, color="#D64F4F", zorder=5)
    ax.axhline(best_quality, color="#D64F4F", linestyle="--", linewidth=0.8, alpha=0.65)
    ax.set_xlabel("Random seed iteration")
    ax.set_ylabel("Leiden modularity")
    ax.set_title("Leiden optimization stability", pad=6)
    ax.text(
        0.02, 0.04,
        f"best={best_quality:.4f}\nstd={np.std(q):.4f}",
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D9D9D9", linewidth=0.5),
    )
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)
    fig.tight_layout()
    return fig

def sorted_by_cluster(mat, meta_labels):
    order_df = pd.DataFrame({"idx": np.arange(len(meta_labels)), "cluster": meta_labels}).sort_values(["cluster", "idx"])
    order = order_df["idx"].to_numpy()
    sorted_mat = mat[order][:, order]
    sorted_labels = np.asarray(meta_labels)[order]
    return sorted_mat, sorted_labels, order

def draw_cluster_boundaries(ax, sorted_labels, color="white"):
    changes = np.where(np.diff(sorted_labels) != 0)[0] + 1
    for pos in changes:
        ax.axhline(pos, color=color, linewidth=0.45, alpha=0.75)
        ax.axvline(pos, color=color, linewidth=0.45, alpha=0.75)

def plot_density_adaptation_pub(raw_mat, prob_mat, meta_labels):
    raw_sorted, sorted_labels, order = sorted_by_cluster(raw_mat, meta_labels)
    prob_sorted = prob_mat[order][:, order]
    vmax_raw = np.percentile(raw_sorted, 99)
    vmax_prob = np.percentile(prob_sorted, 99)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), constrained_layout=True)
    im0 = axes[0].imshow(raw_sorted, cmap="viridis", vmin=0, vmax=vmax_raw, interpolation="nearest")
    axes[0].set_title("Raw CWM affinity")
    draw_cluster_boundaries(axes[0], sorted_labels)
    im1 = axes[1].imshow(prob_sorted, cmap="viridis", vmin=0, vmax=vmax_prob, interpolation="nearest")
    axes[1].set_title("Density-adapted affinity")
    draw_cluster_boundaries(axes[1], sorted_labels)

    for ax in axes:
        ax.set_xlabel("Subcluster motif")
        ax.set_ylabel("Subcluster motif")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    cb0.ax.tick_params(labelsize=6)
    cb1.ax.tick_params(labelsize=6)
    return fig, order, raw_sorted, prob_sorted, sorted_labels

def plot_composition_bar_pub(df_report):
    df = df_report.copy()
    weight_cols = [c for c in ["CAGE_NEW_Weight", "HK_Weight", "DEV_Weight"] if c in df.columns]
    df = df.sort_values("Total_Weight", ascending=True)
    y = np.arange(len(df))

    fig_h = max(2.4, 0.22 * len(df) + 0.7)
    fig, ax = plt.subplots(figsize=(3.6, fig_h))
    left = np.zeros(len(df))
    for col in weight_cols:
        task = col.replace("_Weight", "")
        ax.barh(y, df[col].to_numpy(), left=left, height=0.72, color=TASK_COLORS.get(task, "#999999"), label=task.replace("CAGE_NEW", "CAGE"))
        left += df[col].to_numpy()
    labels = [f"{mc}  {ft}" if "Family_Type" in df.columns else str(mc) for mc, ft in zip(df["MC_ID"], df.get("Family_Type", [""] * len(df)))]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_xlabel("Seqlet weight")
    ax.set_title("Meta-cluster task composition", pad=6)
    ax.legend(loc="lower right", fontsize=6)
    ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.45)
    fig.tight_layout()
    return fig

def plot_sequence_validation_pub(df_seq_val):
    df = df_seq_val.copy()
    df = df.sort_values("Mean_Seq_Similarity", ascending=True)
    fig_h = max(2.2, 0.23 * len(df) + 0.7)
    fig, ax = plt.subplots(figsize=(3.7, fig_h))
    size = np.clip(df["Motif_Count"].to_numpy(dtype=float), 3, None)
    ax.scatter(
        df["Mean_Seq_Similarity"],
        np.arange(len(df)),
        s=10 + 3.0 * size,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.88,
    )
    ax.axvline(df["Mean_Seq_Similarity"].median(), color="#D64F4F", linestyle="--", linewidth=0.8)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["MC_ID"], fontsize=6.2)
    ax.set_xlabel("Mean intra-cluster sequence similarity")
    ax.set_ylabel("Meta-cluster")
    ax.set_title("Sequence-level consistency of CWM clusters", pad=6)
    ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.45)
    fig.tight_layout()
    return fig

def write_manifest(payload):
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "base_dir": BASE_DIR,
        "output_root": str(OUT_ROOT),
        "task_h5_paths": TASKS_H5_PATHS,
        "target_len": TARGET_LEN,
        "perplexity": payload.get("perplexity"),
        "n_seeds": payload.get("n_seeds"),
        "source_mode": payload.get("source_mode"),
        "outputs": payload.get("outputs", {}),
        "source_data": payload.get("source_data", {}),
        "notes": [
            "Leiden QC supplement generated from the shared-motif meta-clustering pipeline.",
            "CWM matrices are median/L2 scaled before affinity computation.",
            "Sequence validation uses PWM minus 0.25 as pseudo-CWM before affinity computation.",
        ],
    }
    out = OUT_ROOT / "supplement_leiden_qc_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out

print("✅ Publication-style plotting helpers loaded.")
'''

    canonical_loader = r'''
# ==========================================
# Canonical downstream-MC loader
# ==========================================

CANONICAL_RESULTS_PKL = Path(BASE_DIR) / "Motif_cluster" / "ic_trimmed_results" / "results_complete.pkl"
CANONICAL_ENTITIES_PKL = Path(BASE_DIR) / "Motif_cluster" / "ic_trimmed_results" / "entities_trimmed.pkl"

# Default: use the exact saved meta_labels/entity order used by downstream MC_000-MC_009 panels.
# Set this to False only if you explicitly want to rerun Leiden and create a new clustering result.
USE_CANONICAL_RESULTS_PKL = True

def load_canonical_meta_clustering_results(perplexity=10):
    if not CANONICAL_RESULTS_PKL.exists() or not CANONICAL_ENTITIES_PKL.exists():
        raise FileNotFoundError(
            "Canonical pkl files are missing. Expected:\n"
            f"{CANONICAL_RESULTS_PKL}\n{CANONICAL_ENTITIES_PKL}\n"
            "Either copy them to Drive or set USE_CANONICAL_RESULTS_PKL=False to rerun Leiden."
        )

    with open(CANONICAL_RESULTS_PKL, "rb") as handle:
        saved = pickle.load(handle)
    with open(CANONICAL_ENTITIES_PKL, "rb") as handle:
        entities = pickle.load(handle)

    entity_names = list(saved["entity_names"])
    meta_labels = np.asarray(saved["meta_labels"], dtype=int)
    missing = [name for name in entity_names if name not in entities]
    if missing:
        raise KeyError(f"Canonical entities missing {len(missing)} names, e.g. {missing[:5]}")

    cwm_array = np.array([entities[name]["cwm"] for name in entity_names], dtype=np.float64)
    scaled_cwms, task_scale_recomputed = apply_median_norm_scaling(cwm_array, entities, entity_names)
    raw_sim_mat = compute_global_affinity_official(scaled_cwms, min_overlap_fraction=0.5)
    prob_mat = apply_density_adaptation(raw_sim_mat, perplexity=perplexity)

    df_report = aggregate_universal_cwms_with_breakdown(meta_labels, entities, entity_names)

    task_scale = saved.get("task_scale", task_scale_recomputed)
    quality_history = saved.get("quality_history", [])
    best_seed = saved.get("best_seed", None)

    return {
        "entities": entities,
        "entity_names": entity_names,
        "cwm_array": cwm_array,
        "scaled_cwms": scaled_cwms,
        "raw_sim_mat": raw_sim_mat,
        "prob_mat": prob_mat,
        "meta_labels": meta_labels,
        "df_report": df_report,
        "task_scale": task_scale,
        "quality_history": quality_history,
        "best_seed": best_seed,
        "source_mode": "canonical_results_complete_pkl",
    }

print("✅ Canonical downstream-MC loader loaded.")
print("Canonical results:", CANONICAL_RESULTS_PKL)
print("Canonical entities:", CANONICAL_ENTITIES_PKL)
'''

    execute_and_save = r'''
# ==========================================
# Execute pipeline, save source data, and export supplement figures
# ==========================================

PERPLEXITY = 10
N_SEEDS = 20

if USE_CANONICAL_RESULTS_PKL:
    print("🔥 Loading canonical downstream MC labels from saved pkl...")
    results = load_canonical_meta_clustering_results(perplexity=PERPLEXITY)
else:
    print("🔥 Rerunning shared-motif Leiden QC pipeline...")
    results = run_meta_clustering_pipeline(
        TASKS_H5_PATHS,
        perplexity=PERPLEXITY,
        n_seeds=N_SEEDS,
    )
    results["source_mode"] = "rerun_from_modisco_h5"

entity_manifest = pd.DataFrame({
    "entity_id": results["entity_names"],
    "task": [results["entities"][name]["task"] for name in results["entity_names"]],
    "weight": [results["entities"][name]["weight"] for name in results["entity_names"]],
    "meta_label": results["meta_labels"],
    "MC_ID": [f"MC_{int(x):03d}" for x in results["meta_labels"]],
})

quality_df = pd.DataFrame({
    "seed_iteration": np.arange(1, len(results["quality_history"]) + 1),
    "modularity": results["quality_history"],
})

results["df_report"].to_csv(DATA_OUT / "leiden_meta_cluster_report.csv", index=False)
quality_df.to_csv(DATA_OUT / "leiden_quality_history.csv", index=False)
entity_manifest.to_csv(DATA_OUT / "leiden_entity_manifest.csv", index=False)
pd.DataFrame([results["task_scale"]]).to_csv(DATA_OUT / "median_norm_task_scale.csv", index=False)

save_matrix_table(results["raw_sim_mat"], results["entity_names"], "raw_cwm_affinity_matrix")
save_matrix_table(results["prob_mat"], results["entity_names"], "density_adapted_affinity_matrix")
np.savez_compressed(
    DATA_OUT / "leiden_qc_arrays.npz",
    cwm_array=results["cwm_array"],
    scaled_cwms=results["scaled_cwms"],
    raw_sim_mat=results["raw_sim_mat"],
    prob_mat=results["prob_mat"],
    meta_labels=results["meta_labels"],
    quality_history=np.asarray(results["quality_history"], dtype=float),
)

print("\n" + "=" * 95)
print("🏆 Meta-clustering report")
print("=" * 95)
cols = [
    "MC_ID", "Total_Weight",
    "CAGE_NEW_Weight", "HK_Weight", "DEV_Weight",
    "CAGE_NEW_Frac_In_Family", "HK_Frac_In_Family", "DEV_Frac_In_Family",
    "Family_Type",
]
available_cols = [c for c in cols if c in results["df_report"].columns]
print(results["df_report"][available_cols].to_markdown(index=False))

df_seq_val, seq_sim_mat = validate_cwm_vs_seq_clustering(
    results["meta_labels"],
    results["entities"],
    results["entity_names"],
)
df_seq_val.to_csv(DATA_OUT / "sequence_level_cluster_validation.csv", index=False)
save_matrix_table(seq_sim_mat, results["entity_names"], "sequence_based_affinity_matrix")

outputs = {}
source_data = {
    "leiden_meta_cluster_report": str(DATA_OUT / "leiden_meta_cluster_report.csv"),
    "leiden_quality_history": str(DATA_OUT / "leiden_quality_history.csv"),
    "leiden_entity_manifest": str(DATA_OUT / "leiden_entity_manifest.csv"),
    "median_norm_task_scale": str(DATA_OUT / "median_norm_task_scale.csv"),
    "raw_cwm_affinity_matrix": str(DATA_OUT / "raw_cwm_affinity_matrix.csv"),
    "density_adapted_affinity_matrix": str(DATA_OUT / "density_adapted_affinity_matrix.csv"),
    "sequence_level_cluster_validation": str(DATA_OUT / "sequence_level_cluster_validation.csv"),
    "sequence_based_affinity_matrix": str(DATA_OUT / "sequence_based_affinity_matrix.csv"),
    "arrays_npz": str(DATA_OUT / "leiden_qc_arrays.npz"),
}

best_quality = float(max(results["quality_history"]))
best_seed = int(results["best_seed"]) if results["best_seed"] is not None else int(np.argmax(results["quality_history"]) + 1)

fig = plot_leiden_trajectory_pub(results["quality_history"], best_seed, best_quality)
outputs["leiden_trajectory"] = save_pub(fig, "supp_leiden_qc_a_leiden_trajectory")
plt.show()
plt.close(fig)

fig, order, raw_sorted, prob_sorted, sorted_labels = plot_density_adaptation_pub(
    results["raw_sim_mat"],
    results["prob_mat"],
    results["meta_labels"],
)
outputs["density_adaptation_heatmaps"] = save_pub(fig, "supp_leiden_qc_b_density_adaptation_heatmaps")
pd.DataFrame({"sorted_index": order, "meta_label": sorted_labels}).to_csv(DATA_OUT / "density_adaptation_heatmap_order.csv", index=False)
plt.show()
plt.close(fig)

fig = plot_composition_bar_pub(results["df_report"])
outputs["composition_bar"] = save_pub(fig, "supp_leiden_qc_c_composition_bar")
plt.show()
plt.close(fig)

fig = plot_sequence_validation_pub(df_seq_val)
outputs["sequence_validation"] = save_pub(fig, "supp_leiden_qc_d_sequence_validation")
plt.show()
plt.close(fig)

manifest_path = write_manifest({
    "perplexity": PERPLEXITY,
    "n_seeds": N_SEEDS,
    "outputs": outputs,
    "source_data": source_data,
    "source_mode": results.get("source_mode"),
})

print("\n✅ Supplement Leiden QC complete.")
print("Data:", DATA_OUT)
print("Figures:", FIG_OUT)
print("Manifest:", manifest_path)
'''

    normalization_qc_addendum = r'''
# ==========================================
# 10. Median/L2 normalization QC addendum
# ==========================================

def compute_median_l2_norm_qc(results):
    """Export per-entity L2 norms before and after the median/L2 scaling step."""
    cwm_array = np.asarray(results["cwm_array"])
    final_scaled = np.asarray(results["scaled_cwms"])
    entity_names = list(results["entity_names"])
    entities = results["entities"]
    task_scale = results["task_scale"]

    median_scaled = np.zeros_like(cwm_array)
    records = []
    for i, name in enumerate(entity_names):
        task = entities[name]["task"]
        scale = float(task_scale[task])
        if scale > 1e-8:
            median_scaled[i] = cwm_array[i] / scale
        else:
            median_scaled[i] = cwm_array[i]

        records.append({
            "entity_id": name,
            "task": task,
            "task_display": "CAGE" if task == "CAGE_NEW" else task,
            "weight": entities[name].get("weight", np.nan),
            "raw_l2_norm": float(np.linalg.norm(cwm_array[i].reshape(-1))),
            "task_median_scale": scale,
            "median_scaled_l2_norm": float(np.linalg.norm(median_scaled[i].reshape(-1))),
            "final_l2_norm": float(np.linalg.norm(final_scaled[i].reshape(-1))),
        })

    df = pd.DataFrame(records)
    return df


def plot_median_l2_norm_qc(df_norm):
    """Before/after norm distributions for the median scaling and final L2 step."""
    plot_df = df_norm.copy()
    order = ["CAGE", "HK", "DEV"]
    task_colors = {"CAGE": "#D64F4F", "HK": "#59A14F", "DEV": "#00A88A"}
    stages = [
        ("raw_l2_norm", "Raw CWM norm"),
        ("median_scaled_l2_norm", "After task-median scaling"),
        ("final_l2_norm", "After final L2 normalization"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15), sharex=False)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.24, top=0.82, wspace=0.32)

    rng = np.random.default_rng(123)
    for ax, (col, title) in zip(axes, stages):
        data = [plot_df.loc[plot_df["task_display"] == task, col].to_numpy() for task in order]
        bp = ax.boxplot(
            data,
            positions=np.arange(len(order)),
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=0.9),
            whiskerprops=dict(color="#777777", linewidth=0.8),
            capprops=dict(color="#777777", linewidth=0.8),
        )
        for patch, task in zip(bp["boxes"], order):
            patch.set_facecolor(task_colors[task])
            patch.set_alpha(0.55)
            patch.set_edgecolor("#555555")
            patch.set_linewidth(0.7)

        for xi, task in enumerate(order):
            vals = data[xi]
            jitter = rng.normal(0, 0.045, len(vals))
            ax.scatter(
                np.full(len(vals), xi) + jitter,
                vals,
                s=9,
                color=task_colors[task],
                alpha=0.55,
                edgecolor="white",
                linewidth=0.2,
                rasterized=True,
            )
            med = float(np.median(vals)) if len(vals) else np.nan
            ax.text(xi, ax.get_ylim()[1], f"med={med:.2f}", ha="center", va="bottom", fontsize=6.3)

        ax.set_title(title, fontsize=8, pad=5)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.85)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    axes[0].set_ylabel("L2 norm of CWM")
    fig.suptitle("Median/L2 normalization reduces task-scale differences before motif clustering", fontsize=9, y=0.98)
    return fig


df_norm_qc = compute_median_l2_norm_qc(results)
norm_qc_path = DATA_OUT / "median_l2_norm_per_entity_qc.csv"
df_norm_qc.to_csv(norm_qc_path, index=False)

summary_norm_qc = (
    df_norm_qc
    .groupby("task_display")[["raw_l2_norm", "median_scaled_l2_norm", "final_l2_norm"]]
    .agg(["count", "median", "mean", "std"])
)
summary_norm_qc.to_csv(DATA_OUT / "median_l2_norm_task_summary_qc.csv")

fig = plot_median_l2_norm_qc(df_norm_qc)
norm_fig_paths = save_pub(fig, "supp_leiden_qc_e_median_l2_norm_qc")
plt.show()
plt.close(fig)

manifest_path = OUT_ROOT / "supplement_leiden_qc_manifest.json"
if manifest_path.exists():
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
else:
    manifest = {}

manifest.setdefault("source_data", {})
manifest.setdefault("outputs", {})
manifest["source_data"]["median_l2_norm_per_entity_qc"] = str(norm_qc_path)
manifest["source_data"]["median_l2_norm_task_summary_qc"] = str(DATA_OUT / "median_l2_norm_task_summary_qc.csv")
manifest["outputs"]["median_l2_norm_qc"] = norm_fig_paths
manifest["normalization_qc_note"] = (
    "Per-entity CWM L2 norms were exported before scaling, after task-median "
    "scaling, and after final per-CWM L2 normalization."
)

with open(manifest_path, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, ensure_ascii=False)

print("\n✅ Median/L2 normalization QC addendum complete.")
print("Norm QC table:", norm_qc_path)
print("Norm QC summary:", DATA_OUT / "median_l2_norm_task_summary_qc.csv")
print("Norm QC figure:", norm_fig_paths)
print("Manifest updated:", manifest_path)
'''

    assignment_stability_addendum = r'''
# ==========================================
# 11. Leiden assignment-stability QC addendum
# ==========================================

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def collect_leiden_seed_assignments(prob_matrix, n_seeds=20, n_iterations=-1):
    """Run Leiden with fixed seed schedule and retain every seed assignment."""
    n_vertices = prob_matrix.shape[0]
    sources, targets = np.where(prob_matrix > 0)
    weights = prob_matrix[sources, targets]

    g = ig.Graph(directed=False)
    g.add_vertices(n_vertices)
    g.add_edges(list(zip(sources, targets)))
    g.es["weight"] = weights

    records = []
    label_rows = []
    for seed in range(1, n_seeds + 1):
        partition = leidenalg.find_partition(
            graph=g,
            partition_type=leidenalg.ModularityVertexPartition,
            weights=g.es["weight"],
            n_iterations=n_iterations,
            seed=seed * 100,
        )
        labels = np.asarray(partition.membership, dtype=int)
        records.append({
            "seed_iteration": seed,
            "leiden_seed": seed * 100,
            "modularity": float(partition.quality()),
            "n_clusters": int(len(np.unique(labels))),
        })
        label_rows.append(labels)
    return pd.DataFrame(records), np.vstack(label_rows)


def compute_pairwise_assignment_metrics(label_matrix):
    n = label_matrix.shape[0]
    ari = np.eye(n)
    nmi = np.eye(n)
    records = []
    for i in range(n):
        for j in range(i + 1, n):
            ari_ij = float(adjusted_rand_score(label_matrix[i], label_matrix[j]))
            nmi_ij = float(normalized_mutual_info_score(label_matrix[i], label_matrix[j]))
            ari[i, j] = ari[j, i] = ari_ij
            nmi[i, j] = nmi[j, i] = nmi_ij
            records.append({
                "seed_i": i + 1,
                "seed_j": j + 1,
                "ARI": ari_ij,
                "NMI": nmi_ij,
            })
    return pd.DataFrame(records), ari, nmi


def plot_assignment_stability_qc(pairwise_df):
    metrics = [("ARI", "Adjusted Rand index"), ("NMI", "Normalized mutual information")]
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.15), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.22, top=0.78, wspace=0.18)
    color = "#4E79A7"

    for ax, (col, title) in zip(axes, metrics):
        vals = pairwise_df[col].to_numpy(dtype=float)
        ax.boxplot(
            [vals],
            positions=[0],
            widths=0.32,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=0.9),
            boxprops=dict(facecolor=color, alpha=0.45, edgecolor="#555555", linewidth=0.7),
            whiskerprops=dict(color="#777777", linewidth=0.8),
            capprops=dict(color="#777777", linewidth=0.8),
        )
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(jitter, vals, s=11, color=color, alpha=0.55, edgecolor="white", linewidth=0.2)
        ax.set_title(title, fontsize=8, pad=5)
        ax.set_xticks([0])
        ax.set_xticklabels([f"pairwise seeds\nn={len(vals)}"])
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.85)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.text(
            0.02,
            0.08,
            f"median={np.median(vals):.3f}\nmin={np.min(vals):.3f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="#4D4D4D",
        )

    axes[0].set_ylabel("Pairwise assignment similarity")
    fig.suptitle("Leiden assignments are stable across random seeds", fontsize=9, y=0.98)
    return fig


assignment_history_df, seed_label_matrix = collect_leiden_seed_assignments(
    results["prob_mat"],
    n_seeds=N_SEEDS,
)
pairwise_assignment_df, ari_matrix, nmi_matrix = compute_pairwise_assignment_metrics(seed_label_matrix)

assignment_history_path = DATA_OUT / "leiden_assignment_seed_history.csv"
assignment_pairwise_path = DATA_OUT / "leiden_assignment_pairwise_ari_nmi.csv"
assignment_labels_path = DATA_OUT / "leiden_assignment_labels_by_seed.csv"
ari_matrix_path = DATA_OUT / "leiden_assignment_ari_matrix.csv"
nmi_matrix_path = DATA_OUT / "leiden_assignment_nmi_matrix.csv"
assignment_summary_path = DATA_OUT / "leiden_assignment_stability_summary.csv"

assignment_history_df.to_csv(assignment_history_path, index=False)
pairwise_assignment_df.to_csv(assignment_pairwise_path, index=False)
pd.DataFrame(
    seed_label_matrix,
    index=[f"seed_{i}" for i in range(1, seed_label_matrix.shape[0] + 1)],
    columns=results["entity_names"],
).to_csv(assignment_labels_path)
pd.DataFrame(
    ari_matrix,
    index=[f"seed_{i}" for i in range(1, ari_matrix.shape[0] + 1)],
    columns=[f"seed_{i}" for i in range(1, ari_matrix.shape[0] + 1)],
).to_csv(ari_matrix_path)
pd.DataFrame(
    nmi_matrix,
    index=[f"seed_{i}" for i in range(1, nmi_matrix.shape[0] + 1)],
    columns=[f"seed_{i}" for i in range(1, nmi_matrix.shape[0] + 1)],
).to_csv(nmi_matrix_path)

assignment_summary = pd.DataFrame({
    "metric": ["ARI", "NMI"],
    "n_pairwise_seed_comparisons": [len(pairwise_assignment_df), len(pairwise_assignment_df)],
    "median": [pairwise_assignment_df["ARI"].median(), pairwise_assignment_df["NMI"].median()],
    "mean": [pairwise_assignment_df["ARI"].mean(), pairwise_assignment_df["NMI"].mean()],
    "min": [pairwise_assignment_df["ARI"].min(), pairwise_assignment_df["NMI"].min()],
    "max": [pairwise_assignment_df["ARI"].max(), pairwise_assignment_df["NMI"].max()],
})
assignment_summary.to_csv(assignment_summary_path, index=False)

fig = plot_assignment_stability_qc(pairwise_assignment_df)
assignment_fig_paths = save_pub(fig, "supp_leiden_qc_f_assignment_stability_qc")
plt.show()
plt.close(fig)

manifest_path = OUT_ROOT / "supplement_leiden_qc_manifest.json"
if manifest_path.exists():
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
else:
    manifest = {}

manifest.setdefault("source_data", {})
manifest.setdefault("outputs", {})
manifest["source_data"]["leiden_assignment_seed_history"] = str(assignment_history_path)
manifest["source_data"]["leiden_assignment_pairwise_ari_nmi"] = str(assignment_pairwise_path)
manifest["source_data"]["leiden_assignment_labels_by_seed"] = str(assignment_labels_path)
manifest["source_data"]["leiden_assignment_ari_matrix"] = str(ari_matrix_path)
manifest["source_data"]["leiden_assignment_nmi_matrix"] = str(nmi_matrix_path)
manifest["source_data"]["leiden_assignment_stability_summary"] = str(assignment_summary_path)
manifest["outputs"]["assignment_stability_qc"] = assignment_fig_paths
manifest["assignment_stability_qc_note"] = (
    "Leiden was rerun over the same seed schedule and all pairwise seed "
    "assignments were compared using ARI and NMI."
)

with open(manifest_path, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, ensure_ascii=False)

print("\n✅ Leiden assignment-stability QC addendum complete.")
print("Assignment history:", assignment_history_path)
print("Pairwise ARI/NMI:", assignment_pairwise_path)
print("Assignment summary:", assignment_summary_path)
print("Assignment QC figure:", assignment_fig_paths)
print("Manifest updated:", manifest_path)
'''

    cells = [
        md(
            """# Supplement Leiden QC for shared-motif meta-clustering

Purpose: generate manuscript-ready supplementary QC panels for the shared-motif Leiden meta-clustering pipeline.

Scientific claim: the shared-motif meta-clusters are reproducible across random Leiden seeds, sharpened by density adaptation, and supported by sequence-level consistency rather than being only contribution-matrix artifacts.

Outputs are saved as source data plus SVG/PDF/PNG/TIFF under:

`/content/drive/MyDrive/DeepEpromote/Drosophila/Motif_cluster/supplement_leiden_qc/`
"""
        ),
        code(
            """# Colab setup
!pip install leidenalg igraph numba scikit-learn h5py seaborn pandas matplotlib tabulate -q

from google.colab import drive
drive.mount('/content/drive')
"""
        ),
        code(cell_63),
        code(cell_64),
        code(cell_65),
        code(cell_66),
        code(cell_67),
        code(cell_68),
        code(publication_plotting),
        code(canonical_loader),
        code(execute_and_save),
        code(normalization_qc_addendum),
        code(assignment_stability_addendum),
        md(
            """## Suggested figure legend draft

**Supplementary Fig. Sx | Quality control of shared-motif meta-clustering.**
**a**, Leiden modularity across 20 random seed initializations; the selected seed is marked by a red star.
**b**, Pairwise CWM affinity matrix before and after density adaptation, sorted by the final Leiden meta-cluster labels.
**c**, Task composition of each inferred meta-cluster, summarized by seqlet weight from CAGE, HK and DEV motif sources.
**d**, Sequence-level validation of CWM-based meta-clusters. Motif sequence matrices were converted to pseudo-CWMs by subtracting the background frequency of 0.25, and mean intra-cluster sequence similarity was computed for clusters with at least three motifs.
**e**, Median/L2 normalization QC. Per-entity CWM L2 norms are shown before task scaling, after task-wise median scaling, and after final per-CWM L2 normalization.
**f**, Leiden assignment-stability QC. All pairwise seed assignments are compared using adjusted Rand index (ARI) and normalized mutual information (NMI).

Update the final label `Sx` after deciding where this QC panel sits relative to the proximal-logo supplement.
"""
        ),
    ]

    out = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_NB.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT_NB)


if __name__ == "__main__":
    main()
