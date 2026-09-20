from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from position_aware_topk_interaction_architecture import (
    POSITION_BIN_SIZE,
    REGION_LEN,
    TASKS,
    TOP_K,
    TaskData,
    build_position_bin_summary,
    draw_example,
    find_newisa_root,
    load_task,
)


MIN_SELECTED = 4
MIN_PAIRS = 3
MIN_SPAN = 60
POSITION_BINS = [f"{i * POSITION_BIN_SIZE}-{(i + 1) * POSITION_BIN_SIZE}" for i in range(5)]
TASK_COLOR = {task: color for task, _, _, color in TASKS}


def output_root(input_root: Path) -> Path:
    out = input_root.parent / "position_aware_selected_motif_architecture_figure"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "source_data").mkdir(parents=True, exist_ok=True)
    return out


def region_task_summary(td: TaskData) -> pd.DataFrame:
    selected = (
        td.selected.groupby("region")
        .agg(
            n_selected=("motif_key", "size"),
            min_center=("center", "min"),
            max_center=("center", "max"),
            tf_set=("tf", lambda x: "|".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    selected["motif_span"] = selected["max_center"] - selected["min_center"]
    pairs = (
        td.selected_pairs.groupby("region")
        .agg(
            n_pairs=("interaction", "size"),
            mean_interaction=("interaction", "mean"),
            median_interaction=("interaction", "median"),
            mean_abs_interaction=("abs_interaction", "mean"),
            fraction_negative=("interaction", lambda x: float((x < 0).mean())),
            fraction_positive=("interaction", lambda x: float((x > 0).mean())),
            median_pair_distance=("distance", "median"),
        )
        .reset_index()
    )
    out = selected.merge(pairs, on="region", how="left")
    out["task"] = td.task
    out["n_pairs"] = out["n_pairs"].fillna(0).astype(int)
    for c in [
        "mean_interaction",
        "median_interaction",
        "mean_abs_interaction",
        "fraction_negative",
        "fraction_positive",
        "median_pair_distance",
    ]:
        out[c] = out[c].fillna(0.0)
    out["eligible"] = (
        (out["n_selected"] >= MIN_SELECTED)
        & (out["n_pairs"] >= MIN_PAIRS)
        & (out["motif_span"] >= MIN_SPAN)
    )
    return out


def region_bin_profiles(td: TaskData) -> pd.DataFrame:
    pairs = td.selected_pairs.copy()
    pairs["bin_low"] = np.minimum(pairs["bin1"], pairs["bin2"])
    pairs["bin_high"] = np.maximum(pairs["bin1"], pairs["bin2"])
    prof = (
        pairs.groupby(["region", "bin_low", "bin_high"], observed=True)
        .agg(median_interaction=("interaction", "median"))
        .reset_index()
    )
    prof["feature"] = prof["bin_low"].astype(str) + "_" + prof["bin_high"].astype(str)
    wide = prof.pivot_table(index="region", columns="feature", values="median_interaction", aggfunc="first")
    all_features = [f"{i}_{j}" for i in range(5) for j in range(i, 5)]
    wide = wide.reindex(columns=all_features)
    wide["task"] = td.task
    return wide.reset_index()


def jaccard_from_sets(a: str, b: str) -> float:
    sa = set(str(a).split("|")) if pd.notna(a) and str(a) else set()
    sb = set(str(b).split("|")) if pd.notna(b) and str(b) else set()
    if not sa and not sb:
        return np.nan
    return len(sa & sb) / len(sa | sb)


def profile_correlation(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 3:
        return np.nan
    av = a[mask].astype(float).to_numpy()
    bv = b[mask].astype(float).to_numpy()
    if np.std(av) == 0 or np.std(bv) == 0:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])


def sign_concordance(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 3:
        return np.nan
    av = np.sign(a[mask].astype(float).to_numpy())
    bv = np.sign(b[mask].astype(float).to_numpy())
    keep = (av != 0) & (bv != 0)
    if int(keep.sum()) < 3:
        return np.nan
    return float((av[keep] == bv[keep]).mean())


def cross_task_similarity(summaries: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [f"{i}_{j}" for i in range(5) for j in range(i, 5)]
    summary_by_task = {
        task: summaries[(summaries["task"] == task) & summaries["eligible"]].set_index("region")
        for task, _, _, _ in TASKS
    }
    profile_by_task = {
        task: profiles[profiles["task"] == task].set_index("region")
        for task, _, _, _ in TASKS
    }
    rows = []
    pairs = [("CAGE", "DEV"), ("CAGE", "HK"), ("DEV", "HK")]
    for a, b in pairs:
        common = sorted(set(summary_by_task[a].index) & set(summary_by_task[b].index))
        for region in common:
            row_a = summary_by_task[a].loc[region]
            row_b = summary_by_task[b].loc[region]
            pa = profile_by_task[a].loc[region, feature_cols] if region in profile_by_task[a].index else pd.Series(index=feature_cols, dtype=float)
            pb = profile_by_task[b].loc[region, feature_cols] if region in profile_by_task[b].index else pd.Series(index=feature_cols, dtype=float)
            rows.append(
                {
                    "region": region,
                    "task_pair": f"{a}-{b}",
                    "tf_set_jaccard": jaccard_from_sets(row_a["tf_set"], row_b["tf_set"]),
                    "position_profile_correlation": profile_correlation(pa, pb),
                    "interaction_sign_concordance": sign_concordance(pa, pb),
                    "mean_abs_difference": abs(row_a["mean_abs_interaction"] - row_b["mean_abs_interaction"]),
                }
            )
    return pd.DataFrame(rows)


def build_representative_candidate_table(summaries: pd.DataFrame, similarities: pd.DataFrame) -> pd.DataFrame:
    eligible = summaries[summaries["eligible"]].copy()
    pivot = eligible.pivot_table(index="region", columns="task", values="mean_abs_interaction", aggfunc="first")
    pivot = pivot.dropna(subset=["CAGE", "DEV", "HK"], how="any")
    sim_mean = similarities.groupby("region").agg(
        mean_tf_jaccard=("tf_set_jaccard", "mean"),
        mean_profile_corr=("position_profile_correlation", "mean"),
    )
    table = pivot.join(sim_mean, how="left").fillna({"mean_tf_jaccard": 0, "mean_profile_corr": 0})
    if table.empty:
        return table.reset_index()
    table["mean_abs_all"] = table[["CAGE", "DEV", "HK"]].mean(axis=1)
    table["hk_dominance"] = table["HK"] - table[["CAGE", "DEV"]].mean(axis=1)
    table["dev_dominance"] = table["DEV"] - table[["CAGE", "HK"]].mean(axis=1)
    table["shared_score"] = table["mean_tf_jaccard"] + table["mean_profile_corr"].clip(lower=0)
    table["divergence_score"] = (
        table["mean_abs_all"]
        + (1 - table["mean_tf_jaccard"].clip(lower=0, upper=1))
        + (1 - table["mean_profile_corr"].clip(lower=-1, upper=1)) / 2
    )
    table["shared_rank"] = (
        table.sort_values(["shared_score", "mean_abs_all"], ascending=False)
        .assign(shared_rank=lambda x: np.arange(1, len(x) + 1))
        ["shared_rank"]
    )
    table["divergent_rank"] = (
        table.sort_values(["mean_profile_corr", "mean_tf_jaccard", "mean_abs_all"], ascending=[True, True, False])
        .assign(divergent_rank=lambda x: np.arange(1, len(x) + 1))
        ["divergent_rank"]
    )
    table["hk_strong_rank"] = (
        table.sort_values(["hk_dominance", "HK"], ascending=False)
        .assign(hk_strong_rank=lambda x: np.arange(1, len(x) + 1))
        ["hk_strong_rank"]
    )
    return table.reset_index()


def choose_representatives(candidate_table: pd.DataFrame) -> pd.DataFrame:
    reps = []
    table = candidate_table.set_index("region").copy() if "region" in candidate_table.columns else candidate_table.copy()
    if not table.empty:
        shared = table.sort_values(["shared_score", "mean_abs_all"], ascending=False).index[0]
        reps.append({"category": "shared-like", "region": shared})
        divergent = table.drop(index=[shared], errors="ignore").sort_values(["mean_profile_corr", "mean_tf_jaccard", "mean_abs_all"], ascending=[True, True, False]).index[0]
        reps.append({"category": "task-divergent", "region": divergent})
        hk = table.drop(index=[shared, divergent], errors="ignore").sort_values(["hk_dominance", "HK"], ascending=False).index[0]
        reps.append({"category": "HK-strong", "region": hk})
    reps = pd.DataFrame(reps)
    if reps.empty:
        return reps
    metric_cols = [
        "CAGE",
        "DEV",
        "HK",
        "mean_tf_jaccard",
        "mean_profile_corr",
        "mean_abs_all",
        "hk_dominance",
        "dev_dominance",
        "shared_score",
        "divergence_score",
        "shared_rank",
        "divergent_rank",
        "hk_strong_rank",
    ]
    return reps.merge(candidate_table[["region"] + metric_cols], on="region", how="left")


def _null_thresholds(vals: pd.Series, percentile: float = 80.0) -> tuple[float, float]:
    arr = vals.dropna().to_numpy(dtype=float)
    pos = arr[arr > 0]
    neg = arr[arr < 0]
    pos_thr = float(np.nanpercentile(pos, percentile)) if len(pos) else np.nan
    neg_thr = float(np.nanpercentile(neg, 100 - percentile)) if len(neg) else np.nan
    return pos_thr, neg_thr


def build_representative_evidence(root: Path, task_data: list[TaskData], reps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep_regions = set(reps["region"]) if not reps.empty else set()
    rep_category = reps.set_index("region")["category"].to_dict() if not reps.empty else {}
    task_folder = {task: folder for task, folder, _, _ in TASKS}
    task_track = {task: track for task, _, track, _ in TASKS}
    single_thresholds = {}
    pair_thresholds = {}
    for task in task_folder:
        data_dir = root / task_folder[task] / "Data"
        track = task_track[task]
        isa_col = f"isa_t{track}"
        inter_col = f"interaction_t{track}"
        null_isa = pd.read_csv(data_dir / "null_isa.csv")
        null_inter = pd.read_csv(data_dir / "null_interaction.csv")
        single_thresholds[task] = _null_thresholds(null_isa[isa_col])[0]
        pair_thresholds[task] = _null_thresholds(null_inter[inter_col])

    motif_rows = []
    pair_rows = []
    for td in task_data:
        track = td.track
        isa_col = f"isa_t{track}"
        inter_col = f"interaction_t{track}"
        motifs = td.selected[td.selected["region"].isin(rep_regions)].copy()
        if not motifs.empty:
            motifs["category"] = motifs["region"].map(rep_category)
            motifs["target_isa"] = motifs[isa_col]
            motifs["single_null_positive_threshold"] = single_thresholds[td.task]
            motifs["passes_single_threshold"] = motifs["target_isa"] >= motifs["single_null_positive_threshold"]
            motif_rows.append(motifs)

        pairs = td.selected_pairs[td.selected_pairs["region"].isin(rep_regions)].copy()
        if not pairs.empty:
            pos_thr, neg_thr = pair_thresholds[td.task]
            pairs["category"] = pairs["region"].map(rep_category)
            pairs["target_interaction_col"] = inter_col
            pairs["target_interaction"] = pairs["interaction"]
            pairs["pair_null_positive_threshold"] = pos_thr
            pairs["pair_null_negative_threshold"] = neg_thr
            pairs["pair_null_class"] = np.select(
                [
                    pairs["target_interaction"] > pos_thr,
                    pairs["target_interaction"] < neg_thr,
                ],
                ["positive", "negative"],
                default="near-null",
            )
            pair_rows.append(pairs)

    motif_evidence = pd.concat(motif_rows, ignore_index=True) if motif_rows else pd.DataFrame()
    pair_evidence = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    return motif_evidence, pair_evidence


def panel_label(ax, label: str, x: float = -0.18, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")


def draw_method(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a")
    ax.set_title("Selected-motif workflow", fontsize=7, pad=10)
    y = 0.63
    ax.hlines(y, 0.08, 0.92, color="#CFCFCF", lw=1.6)
    xs = [0.16, 0.28, 0.42, 0.57, 0.72, 0.84]
    for i, x in enumerate(xs, 1):
        ax.add_patch(plt.Rectangle((x - 0.025, y - 0.055), 0.05, 0.11, facecolor="#777777", edgecolor="#333333", lw=0.4))
        ax.text(x, y + 0.08, f"M{i}", ha="center", fontsize=5.5)
    ax.annotate("", xy=(xs[4], y + 0.16), xytext=(xs[1], y + 0.16), arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.18", lw=1.3, color="#C9473D"))
    ax.annotate("", xy=(xs[5], y + 0.13), xytext=(xs[2], y + 0.13), arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=-0.18", lw=1.3, color="#4E79A7"))
    ax.text(0.5, 0.92, f"Top {TOP_K} non-overlapping motifs\nby single ISA", ha="center", va="top", fontsize=5.4)
    ax.text(0.5, 0.34, "retain position +\nnormalized interaction", ha="center", va="center", fontsize=5.4)
    for r in range(5):
        for c in range(5):
            face = "#DDEAF3" if (r + c) % 2 else "#F4D8D4"
            ax.add_patch(plt.Rectangle((0.36 + c * 0.045, 0.07 + r * 0.045), 0.045, 0.045, facecolor=face, edgecolor="white", lw=0.35))
    ax.text(0.5, 0.02, "aggregate by position bins\nand compare tasks", ha="center", va="bottom", fontsize=5.3)


def draw_global_heatmaps(fig, grid, summary: pd.DataFrame) -> None:
    vmax = float(np.nanpercentile(np.abs(summary["median_interaction"]), 95))
    vmax = max(vmax, 0.05)
    im = None
    for i, (task, _, _, _) in enumerate(TASKS):
        ax = fig.add_subplot(grid[i])
        sub = summary[summary["task"] == task]
        mat = np.full((5, 5), np.nan)
        nmat = np.zeros((5, 5), dtype=int)
        for row in sub.itertuples(index=False):
            mat[int(row.bin1), int(row.bin2)] = row.median_interaction
            nmat[int(row.bin1), int(row.bin2)] = int(row.n_pairs)
        im = ax.imshow(mat, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        task_pairs = int(sub[sub["bin1"] <= sub["bin2"]]["n_pairs"].sum())
        ax.set_title(f"{task}\nselected pairs n={task_pairs:,}", fontsize=6.6)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(POSITION_BINS, rotation=45, ha="right", fontsize=5.2)
        ax.set_yticklabels(POSITION_BINS, fontsize=5.2)
        ax.set_xlabel("Position bin B (bp)", fontsize=5.8)
        if i == 0:
            ax.set_ylabel("Position bin A (bp)", fontsize=5.8)
            panel_label(ax, "b")
        for r in range(5):
            for c in range(5):
                if nmat[r, c] > 0:
                    ax.text(c, r, str(nmat[r, c]), ha="center", va="center", fontsize=4.6, color="#202020")
    cax = fig.add_axes([0.93, 0.59, 0.012, 0.20])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Median normalized interaction", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.2, length=2)


def draw_region_distributions(ax, summaries: pd.DataFrame) -> None:
    panel_label(ax, "c", x=-0.20, y=1.20)
    data = summaries[summaries["eligible"]].copy()
    positions = np.arange(len(TASKS))
    for i, (task, _, _, color) in enumerate(TASKS):
        vals = data.loc[data["task"] == task, "mean_abs_interaction"].dropna().to_numpy()
        parts = ax.violinplot(vals, positions=[i], widths=0.7, showmeans=False, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor("#333333")
            pc.set_alpha(0.35)
        parts["cmedians"].set_color("#222222")
        rng = np.random.default_rng(42 + i)
        sample = vals if len(vals) <= 300 else rng.choice(vals, 300, replace=False)
        ax.scatter(np.full(len(sample), i) + rng.normal(0, 0.045, len(sample)), sample, s=3.2, color=color, alpha=0.28, linewidths=0)
    ax.set_xticks(positions)
    ax.set_xticklabels([t[0] for t in TASKS])
    ax.set_ylabel("Mean |interaction| per region")
    ns = data.groupby("task").size().to_dict()
    ax.set_title(
        "Region-level interaction strength\n"
        + ", ".join(f"{task} n={int(ns.get(task, 0))}" for task, _, _, _ in TASKS),
        fontsize=6.5,
    )


def draw_similarity(ax1, ax2, similarities: pd.DataFrame) -> None:
    panel_label(ax1, "d", x=-0.20, y=1.20)
    order = ["CAGE-DEV", "CAGE-HK", "DEV-HK"]
    metrics = [
        (ax1, "tf_set_jaccard", "Selected TF-set Jaccard", (0, 1)),
        (ax2, "position_profile_correlation", "Position-bin profile correlation", (-1, 1)),
    ]
    for ax, metric, ylabel, ylim in metrics:
        for i, pair in enumerate(order):
            vals = similarities.loc[similarities["task_pair"] == pair, metric].dropna().to_numpy()
            if len(vals) == 0:
                continue
            parts = ax.violinplot(vals, positions=[i], widths=0.65, showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("#CFCFCF")
                pc.set_edgecolor("#444444")
                pc.set_alpha(0.55)
            parts["cmedians"].set_color("#222222")
            rng = np.random.default_rng(123 + i)
            sample = vals if len(vals) <= 300 else rng.choice(vals, 300, replace=False)
            ax.scatter(np.full(len(sample), i) + rng.normal(0, 0.045, len(sample)), sample, s=3.0, color="#555555", alpha=0.25, linewidths=0)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=5.6)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
    ns = similarities.groupby("task_pair").size().to_dict()
    n_text = ", ".join(f"{pair} n={int(ns.get(pair, 0))}" for pair in order)
    ax1.set_title(f"Cross-task selected-TF similarity\n{n_text}", fontsize=6.4)
    ax2.set_title("Interaction profile similarity", fontsize=7)
    ax2.axhline(0, color="#999999", lw=0.7, ls=":")


def draw_representatives(fig, grids, task_data: list[TaskData], reps: pd.DataFrame, matrix_vmax: float | None = None):
    panel_written = False
    last_im = None
    task_map = {td.task: td for td in task_data}
    for c, row in enumerate(reps.itertuples(index=False)):
        col_grid = grids[c].subgridspec(3, 2, width_ratios=[0.92, 1.08], hspace=0.45, wspace=0.22)
        for r, (task, _, _, _) in enumerate(TASKS):
            sub = col_grid[r, :].subgridspec(1, 2, width_ratios=[0.95, 1.05], wspace=0.28)
            ax_track = fig.add_subplot(sub[0, 0])
            ax_mat = fig.add_subplot(sub[0, 1])
            last_im = draw_example(
                ax_track,
                ax_mat,
                task_map[task],
                row.region,
                label="e" if not panel_written else None,
                matrix_vmax=matrix_vmax,
            )
            panel_written = True
            ax_track.set_title(task, fontsize=6.1)
            if r == 0:
                ax_track.text(
                    0.5,
                    1.42,
                    f"{row.category}\n{row.region}",
                    transform=ax_track.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=5.4,
                )
    return last_im


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )
    root = find_newisa_root()
    out = output_root(root)
    task_data = [load_task(root, *spec) for spec in TASKS]

    selected = pd.concat([td.selected for td in task_data], ignore_index=True)
    selected_pairs = pd.concat([td.selected_pairs for td in task_data], ignore_index=True)
    matrix_vmax = max(float(np.nanpercentile(selected_pairs["abs_interaction"], 95)), 0.05)
    global_position = build_position_bin_summary(selected_pairs)
    summaries = pd.concat([region_task_summary(td) for td in task_data], ignore_index=True)
    profiles = pd.concat([region_bin_profiles(td) for td in task_data], ignore_index=True)
    similarities = cross_task_similarity(summaries, profiles)
    representative_candidates = build_representative_candidate_table(summaries, similarities)
    reps = choose_representatives(representative_candidates)
    motif_evidence, pair_evidence = build_representative_evidence(root, task_data, reps)

    selected.to_csv(out / "source_data" / "selected_motif_instances.csv", index=False)
    selected_pairs.to_csv(out / "source_data" / "selected_pair_interactions.csv", index=False)
    global_position.to_csv(out / "source_data" / "global_position_bin_interaction_summary.csv", index=False)
    summaries.to_csv(out / "source_data" / "region_task_architecture_summary.csv", index=False)
    profiles.to_csv(out / "source_data" / "region_position_bin_profiles.csv", index=False)
    similarities.to_csv(out / "source_data" / "cross_task_architecture_similarity.csv", index=False)
    representative_candidates.to_csv(out / "source_data" / "representative_region_candidate_ranking.csv", index=False)
    reps.to_csv(out / "source_data" / "representative_regions_for_panel_e.csv", index=False)
    motif_evidence.to_csv(out / "source_data" / "representative_selected_motifs_evidence.csv", index=False)
    pair_evidence.to_csv(out / "source_data" / "representative_selected_pairs_evidence.csv", index=False)

    fig = plt.figure(figsize=(7.6, 8.8))
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[1.0, 1.08, 1.0, 2.4],
        width_ratios=[1.08, 1, 1, 1],
        hspace=0.72,
        wspace=0.55,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    draw_method(ax_a)
    draw_global_heatmaps(fig, [gs[0, 1], gs[0, 2], gs[0, 3]], global_position)
    ax_c = fig.add_subplot(gs[1, 0])
    draw_region_distributions(ax_c, summaries)
    ax_d1 = fig.add_subplot(gs[1, 1:3])
    ax_d2 = fig.add_subplot(gs[1, 3])
    draw_similarity(ax_d1, ax_d2, similarities)
    ax_note = fig.add_subplot(gs[2, 0])
    ax_note.axis("off")
    ax_note.text(
        0,
        0.9,
        "Representative regions\nexplain the global\narchitecture classes.",
        ha="left",
        va="top",
        fontsize=6.2,
    )
    ax_note.text(
        0,
        0.38,
        "Exploratory selected-motif\nanalysis; not a global\nTF-pair class claim.",
        ha="left",
        va="top",
        fontsize=5.7,
        color="#555555",
    )
    rep_im = draw_representatives(fig, [gs[2:, 1], gs[2:, 2], gs[2:, 3]], task_data, reps, matrix_vmax=matrix_vmax)
    if rep_im is not None:
        cax = fig.add_axes([0.93, 0.12, 0.012, 0.20])
        cb = fig.colorbar(rep_im, cax=cax)
        cb.set_label("Normalized interaction", fontsize=5.8)
        cb.ax.tick_params(labelsize=5.2, length=2)

    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out / "figures" / f"position_aware_selected_motif_architecture.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Cleaner main figure: global evidence only.
    fig_summary = plt.figure(figsize=(7.4, 4.8))
    gs_sum = fig_summary.add_gridspec(
        2,
        4,
        height_ratios=[1.05, 1.0],
        width_ratios=[1.08, 1, 1, 1],
        hspace=0.78,
        wspace=0.58,
    )
    draw_method(fig_summary.add_subplot(gs_sum[0, 0]))
    draw_global_heatmaps(fig_summary, [gs_sum[0, 1], gs_sum[0, 2], gs_sum[0, 3]], global_position)
    draw_region_distributions(fig_summary.add_subplot(gs_sum[1, 0]), summaries)
    draw_similarity(fig_summary.add_subplot(gs_sum[1, 1:3]), fig_summary.add_subplot(gs_sum[1, 3]), similarities)
    for ext in ["png", "svg", "pdf"]:
        fig_summary.savefig(out / "figures" / f"position_aware_selected_motif_architecture_summary.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig_summary)

    # Separate readable representative examples.
    fig_rep = plt.figure(figsize=(8.4, 7.4))
    gs_rep = fig_rep.add_gridspec(1, 3, wspace=0.45)
    rep_im = draw_representatives(fig_rep, [gs_rep[0, 0], gs_rep[0, 1], gs_rep[0, 2]], task_data, reps, matrix_vmax=matrix_vmax)
    fig_rep.suptitle("Representative same-region task triptychs", fontsize=8, y=1.01)
    if rep_im is not None:
        cax = fig_rep.add_axes([0.92, 0.15, 0.012, 0.18])
        cb = fig_rep.colorbar(rep_im, cax=cax)
        cb.set_label("Normalized interaction", fontsize=5.8)
        cb.ax.tick_params(labelsize=5.2, length=2)
    for ext in ["png", "svg", "pdf"]:
        fig_rep.savefig(out / "figures" / f"position_aware_selected_motif_architecture_representatives.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig_rep)

    print("Input:", root)
    print("Output:", out)
    print("Selected instances:", len(selected))
    print("Selected pair interactions:", len(selected_pairs))
    print("Eligible region-task rows:", int(summaries["eligible"].sum()))
    print("Cross-task similarity rows:", len(similarities))
    print("Representative candidate regions:", len(representative_candidates))
    print("Representative motif evidence rows:", len(motif_evidence))
    print("Representative pair evidence rows:", len(pair_evidence))
    print("Representatives:")
    print(reps.to_string(index=False))


if __name__ == "__main__":
    main()
