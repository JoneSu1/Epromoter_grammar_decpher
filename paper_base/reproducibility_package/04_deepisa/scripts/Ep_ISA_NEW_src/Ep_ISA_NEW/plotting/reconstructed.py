from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
import urllib.request

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr


TASK_META = {
    "CAGE": {"folder": "results_cage", "track": 0, "color": "#1f77b4"},
    "D/DEV": {"folder": "results_dev", "track": 0, "color": "#d95f02"},
    "E/HK-like": {"folder": "results_hk", "track": 1, "color": "#2ca02c"},
}

TASK_ORDER = ("CAGE", "D/DEV", "E/HK-like")


@dataclass
class TaskResult:
    name: str
    folder: str
    track: int
    color: str
    data_dir: Path
    motif_locs: pd.DataFrame
    motif_single_isa: pd.DataFrame
    motif_combi_isa: pd.DataFrame
    null_isa: pd.DataFrame
    null_interaction: pd.DataFrame
    coop_pair: pd.DataFrame
    coop_tf: pd.DataFrame
    tf_importance: pd.DataFrame
    pred_orig: pd.DataFrame


def setup_nature_style(font_dir: Optional[Path] = None, download_arial: bool = True) -> Path:
    """Configure compact Nature-style plotting and register Arial."""
    if font_dir is None:
        font_dir = Path.home() / ".Ep_ISA_NEW_fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    arial_path = font_dir / "Arial.ttf"

    if download_arial and not arial_path.exists():
        url = "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf"
        urllib.request.urlretrieve(url, arial_path)

    if arial_path.exists():
        fm.fontManager.addfont(str(arial_path))

    sns.set_theme(style="ticks", rc={"font.family": "Arial"})
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.45,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    return arial_path


def savefig(fig: mpl.figure.Figure, outpath: Optional[Path]) -> None:
    if outpath is None:
        return
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    if outpath.suffix.lower() == ".pdf":
        fig.savefig(outpath.with_suffix(".png"), dpi=300)


def despine(ax: mpl.axes.Axes) -> None:
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)


def _format_pvalue(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 1e-300:
        return "<1e-300"
    return f"{p:.1e}"


def _read_csv(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_precomputed_results(result_dir: Path) -> Dict[str, TaskResult]:
    """Load already computed ISA outputs. This function performs no ISA computation."""
    result_dir = Path(result_dir)
    results: Dict[str, TaskResult] = {}
    for name, meta in TASK_META.items():
        folder = meta["folder"]
        track = int(meta["track"])
        data_dir = result_dir / folder / "Data"
        results[name] = TaskResult(
            name=name,
            folder=folder,
            track=track,
            color=str(meta["color"]),
            data_dir=data_dir,
            motif_locs=_read_csv(data_dir, "motif_locs.csv"),
            motif_single_isa=_read_csv(data_dir, "motif_single_isa.csv"),
            motif_combi_isa=_read_csv(data_dir, "motif_combi_isa.csv"),
            null_isa=_read_csv(data_dir, "null_isa.csv"),
            null_interaction=_read_csv(data_dir, "null_interaction.csv"),
            coop_pair=_read_csv(data_dir, f"coop_tf_pair_t{track}.csv"),
            coop_tf=_read_csv(data_dir, f"coop_tf_t{track}.csv"),
            tf_importance=_read_csv(data_dir, "tf_importance.csv"),
            pred_orig=_read_csv(data_dir, "pred_orig.csv"),
        )
    return results


def load_finemo_hits(snapshot_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load Fi-NeMo hit tables from the Google Drive snapshot."""
    snapshot_dir = Path(snapshot_dir)
    mapping = {"CAGE": "CAGE_NEW", "D/DEV": "DEV", "E/HK-like": "HK"}
    out = {}
    for task, folder in mapping.items():
        path = snapshot_dir / "finemo_scans" / folder / "hits.tsv"
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            df["task"] = task
            out[task] = df
        else:
            out[task] = pd.DataFrame()
    return out


def summarize_loaded_results(results: Mapping[str, TaskResult]) -> pd.DataFrame:
    rows = []
    for task in results.values():
        rows.append(
            {
                "task": task.name,
                "track": task.track,
                "motif_locs": len(task.motif_locs),
                "single_isa": len(task.motif_single_isa),
                "combi_isa": len(task.motif_combi_isa),
                "null_isa": len(task.null_isa),
                "null_interaction": len(task.null_interaction),
                "coop_pair": len(task.coop_pair),
                "coop_tf": len(task.coop_tf),
                "tf_importance": len(task.tf_importance),
            }
        )
    return pd.DataFrame(rows)


def plot_sequence_cage_with_ed_overview(
    hits: Mapping[str, pd.DataFrame],
    outpath: Optional[Path] = None,
    figsize: Tuple[float, float] = (7.1, 4.4),
) -> pd.DataFrame:
    """Redraw Figure 1: sequence-level CAGE-with-E/D classification."""
    seq_sets = {
        task: set(df["peak_id"].dropna().astype(int))
        for task, df in hits.items()
        if not df.empty and "peak_id" in df.columns
    }
    cage = seq_sets.get("CAGE", set())
    dev = seq_sets.get("D/DEV", set())
    hk = seq_sets.get("E/HK-like", set())
    all_ids = sorted(cage | dev | hk)

    rows = []
    for peak_id in all_ids:
        c = peak_id in cage
        d = peak_id in dev
        e = peak_id in hk
        if c and d and e:
            label = "CAGE-with-E-and-D"
        elif c and e:
            label = "CAGE-with-E"
        elif c and d:
            label = "CAGE-with-D"
        elif c:
            label = "CAGE-only"
        elif d:
            label = "D-only"
        elif e:
            label = "E-only"
        else:
            label = "None"
        rows.append({"peak_id": peak_id, "CAGE": c, "D/DEV": d, "E/HK-like": e, "class": label})
    class_df = pd.DataFrame(rows)

    def seqs_with_pos_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame) -> set:
        if df_a.empty or df_b.empty:
            return set()
        m = df_a[["peak_id", "start", "end"]].merge(
            df_b[["peak_id", "start", "end"]],
            on="peak_id",
            suffixes=("_a", "_b"),
        )
        if m.empty:
            return set()
        overlap = (m[["end_a", "end_b"]].min(axis=1) - m[["start_a", "start_b"]].max(axis=1)) > 0
        return set(m.loc[overlap, "peak_id"].astype(int))

    cage_df = hits.get("CAGE", pd.DataFrame())
    dev_df = hits.get("D/DEV", pd.DataFrame())
    hk_df = hits.get("E/HK-like", pd.DataFrame())
    e_same = seqs_with_pos_overlap(cage_df, hk_df)
    d_same = seqs_with_pos_overlap(cage_df, dev_df)
    context_rows = [
        {"group": "CAGE-with-E", "positional_context": "same_pos", "n": len((cage & hk) & e_same)},
        {"group": "CAGE-with-E", "positional_context": "diff_pos", "n": len((cage & hk) - e_same)},
        {"group": "CAGE-with-D", "positional_context": "same_pos", "n": len((cage & dev) & d_same)},
        {"group": "CAGE-with-D", "positional_context": "diff_pos", "n": len((cage & dev) - d_same)},
    ]
    context_df = pd.DataFrame(context_rows)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.05, 1.25], height_ratios=[1, 1.15], wspace=0.75, hspace=0.78)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    steps = [
        "Core-promoter\ncandidate sequences",
        "Fi-NeMo hits\n(shared CWMs)",
        "CAGE-with-E/D\nsequence classes",
        "Precomputed ISA\ntables",
        "Reconstructed\nfigures",
    ]
    y = 0.95
    for i, step in enumerate(steps):
        ax.text(0.5, y, step, fontsize=5.5, va="center", ha="center", bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.6", lw=0.45))
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.12), xytext=(0.5, y - 0.04), arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))
        y -= 0.2

    ax = fig.add_subplot(gs[0, 1])
    class_order = ["CAGE-only", "CAGE-with-E", "CAGE-with-D", "CAGE-with-E-and-D", "D-only", "E-only"]
    counts = class_df["class"].value_counts().reindex(class_order).fillna(0).astype(int)
    short_labels = ["C-only", "C+E", "C+D", "C+E+D", "D-only", "E-only"]
    ax.barh(range(len(counts)), counts.values, color=["#1f77b4", "#66a61e", "#d95f02", "#7570b3", "#e6ab02", "#2ca02c"])
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(short_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Sequences")
    ax.set_title("Sequence classes")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {v}", va="center", fontsize=5.5)
    despine(ax)

    ax = fig.add_subplot(gs[0, 2])
    sns.barplot(data=context_df, x="group", y="n", hue="positional_context", ax=ax, palette={"same_pos": "#4daf4a", "diff_pos": "#984ea3"})
    ax.set_xlabel("")
    ax.set_ylabel("Sequences")
    ax.set_title("Position reuse")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, title=None)
    despine(ax)

    ax = fig.add_subplot(gs[1, :])
    pos_frames = []
    for task, df in hits.items():
        if df.empty:
            continue
        tmp = df[["start", "end"]].copy()
        tmp["task"] = task
        tmp["motif_center"] = (tmp["start"] + tmp["end"]) / 2.0
        pos_frames.append(tmp)
    pos_df = pd.concat(pos_frames, ignore_index=True) if pos_frames else pd.DataFrame()
    for task in TASK_ORDER:
        sub = pos_df[pos_df["task"] == task]
        if not sub.empty:
            sns.kdeplot(sub["motif_center"], ax=ax, color=TASK_META[task]["color"], lw=0.9, label=f"{task} (n={len(sub):,})")
    ax.axvline(124.5, color="0.35", lw=0.5, ls=":", label="sequence center")
    ax.set_xlabel("Motif center in 249-bp sequence")
    ax.set_ylabel("Density")
    ax.set_title("Motif positional architecture", pad=3)
    ax.legend(frameon=False, loc="upper right")
    despine(ax)

    savefig(fig, outpath)
    return pd.concat([class_df, context_df.assign(peak_id=np.nan, **{"CAGE": np.nan, "D/DEV": np.nan, "E/HK-like": np.nan, "class": context_df["group"]})], ignore_index=True, sort=False)


def _non_independent(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "cooperativity" not in df.columns:
        return df.copy()
    return df[df["cooperativity"] != "Independent"].copy()


def _absolute_sign(score: float, eps: float = 1e-12) -> str:
    if pd.isna(score):
        return "NA"
    if score > eps:
        return "Positive"
    if score < -eps:
        return "Negative"
    return "Near-zero"


def build_coop_long(results: Mapping[str, TaskResult], include_independent: bool = False) -> pd.DataFrame:
    frames = []
    for task in results.values():
        df = task.coop_pair.copy()
        if df.empty:
            continue
        if not include_independent:
            df = _non_independent(df)
        df["task"] = task.name
        df["track"] = task.track
        df["sign_class"] = df["coop_score"].map(_absolute_sign)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def recurrent_pair_sets(results: Mapping[str, TaskResult]) -> Dict[str, set]:
    task_sets = {
        task.name: set(_non_independent(task.coop_pair)["tf_pair"])
        for task in results.values()
        if not task.coop_pair.empty
    }
    cage = task_sets.get("CAGE", set())
    dev = task_sets.get("D/DEV", set())
    hk = task_sets.get("E/HK-like", set())
    return {
        "CAGE": cage,
        "D/DEV": dev,
        "E/HK-like": hk,
        "CAGE-with-E": cage & hk,
        "CAGE-with-D": cage & dev,
        "CAGE-with-E-and-D": cage & dev & hk,
    }


def classify_recurrent_pair(tf_pair: str, pair_sets: Mapping[str, set]) -> str:
    in_e = tf_pair in pair_sets.get("CAGE-with-E", set())
    in_d = tf_pair in pair_sets.get("CAGE-with-D", set())
    if in_e and in_d:
        return "CAGE-with-E-and-D"
    if in_e:
        return "CAGE-with-E"
    if in_d:
        return "CAGE-with-D"
    return "Specific"


def plot_real_vs_null_composite(
    results: Mapping[str, TaskResult],
    outpath: Optional[Path] = None,
    figsize: Tuple[float, float] = (7.1, 3.8),
) -> pd.DataFrame:
    """Plot real motif ISA / interaction against matched null distributions."""
    fig, axes = plt.subplots(2, len(TASK_ORDER), figsize=figsize, sharey=False)
    stats_rows = []

    for col_idx, task_name in enumerate(TASK_ORDER):
        task = results[task_name]
        track = task.track
        color = task.color
        isa_col = f"isa_t{track}"
        inter_col = f"interaction_t{track}"

        ax = axes[0, col_idx]
        real_isa = task.motif_single_isa.get(isa_col, pd.Series(dtype=float)).dropna()
        null_isa = task.null_isa.get(isa_col, pd.Series(dtype=float)).dropna()
        if len(real_isa) and len(null_isa):
            sns.kdeplot(null_isa, ax=ax, color="0.55", lw=0.8, label="Null")
            sns.kdeplot(real_isa, ax=ax, color=color, lw=0.9, label="Real")
            p = mannwhitneyu(real_isa, null_isa, alternative="two-sided").pvalue
            stats_rows.append(
                {
                    "task": task_name,
                    "comparison": "single_isa",
                    "real_n": len(real_isa),
                    "null_n": len(null_isa),
                    "real_median": real_isa.median(),
                    "null_median": null_isa.median(),
                    "mw_p": p,
                }
            )
            ax.text(
                0.03,
                0.96,
                f"n={len(real_isa):,}/{len(null_isa):,}\nMed={real_isa.median():.3g}/{null_isa.median():.3g}\nP={_format_pvalue(p)}",
                transform=ax.transAxes,
                va="top",
                fontsize=5.2,
            )
        ax.axvline(0, color="0.2", lw=0.45, ls=":")
        ax.set_title(task_name)
        ax.set_xlabel("Single ISA")
        ax.set_ylabel("Density" if col_idx == 0 else "")
        despine(ax)

        ax = axes[1, col_idx]
        real_inter = task.motif_combi_isa.get(inter_col, pd.Series(dtype=float)).dropna()
        null_inter = task.null_interaction.get(inter_col, pd.Series(dtype=float)).dropna()
        if len(real_inter) and len(null_inter):
            sns.kdeplot(null_inter, ax=ax, color="0.55", lw=0.8, label="Null")
            sns.kdeplot(real_inter, ax=ax, color=color, lw=0.9, label="Real")
            p = mannwhitneyu(real_inter, null_inter, alternative="two-sided").pvalue
            stats_rows.append(
                {
                    "task": task_name,
                    "comparison": "pair_interaction",
                    "real_n": len(real_inter),
                    "null_n": len(null_inter),
                    "real_median": real_inter.median(),
                    "null_median": null_inter.median(),
                    "mw_p": p,
                }
            )
            ax.text(
                0.03,
                0.96,
                f"n={len(real_inter):,}/{len(null_inter):,}\nMed={real_inter.median():.3g}/{null_inter.median():.3g}\nP={_format_pvalue(p)}",
                transform=ax.transAxes,
                va="top",
                fontsize=5.2,
            )
        ax.axvline(0, color="0.2", lw=0.45, ls=":")
        ax.set_xlabel("Pair interaction")
        ax.set_ylabel("Density" if col_idx == 0 else "")
        despine(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper right", bbox_to_anchor=(0.99, 1.02))
    for ax in axes.ravel():
        leg = ax.get_legend()
        if leg:
            leg.remove()
    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    savefig(fig, outpath)
    return pd.DataFrame(stats_rows)


def plot_coop_ecdf_and_sign(
    results: Mapping[str, TaskResult],
    outpath: Optional[Path] = None,
    figsize: Tuple[float, float] = (6.8, 2.6),
) -> pd.DataFrame:
    coop = build_coop_long(results, include_independent=False)
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1.0]})

    ax = axes[0]
    for task_name in TASK_ORDER:
        sub = coop[coop["task"] == task_name]
        if sub.empty:
            continue
        sns.ecdfplot(sub["coop_score"], ax=ax, color=TASK_META[task_name]["color"], lw=1.0, label=f"{task_name} (n={len(sub)})")
    ax.axvline(0, color="0.2", lw=0.55, ls=":")
    ax.set_xlabel("Cooperativity score")
    ax.set_ylabel("Cumulative proportion")
    ax.legend(frameon=False, loc="lower right")
    despine(ax)

    sign_counts = (
        coop.groupby(["task", "sign_class"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = sign_counts.groupby("task")["n"].transform("sum")
    sign_counts["fraction"] = sign_counts["n"] / totals
    order = ["Negative", "Near-zero", "Positive"]
    palette = {"Negative": "#4575b4", "Near-zero": "0.7", "Positive": "#d73027"}

    ax = axes[1]
    bottom = np.zeros(len(TASK_ORDER))
    for cls in order:
        vals = []
        for task_name in TASK_ORDER:
            hit = sign_counts[(sign_counts["task"] == task_name) & (sign_counts["sign_class"] == cls)]
            vals.append(float(hit["fraction"].iloc[0]) if len(hit) else 0.0)
        ax.bar(TASK_ORDER, vals, bottom=bottom, color=palette[cls], edgecolor="white", linewidth=0.4, label=cls)
        bottom += np.array(vals)
    ax.set_ylabel("Fraction of non-independent pairs")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    despine(ax)

    fig.tight_layout(w_pad=1.2)
    savefig(fig, outpath)
    return sign_counts


def add_pair_context(
    combi_df: pd.DataFrame,
    motif_locs: pd.DataFrame,
    local_window: int = 50,
    local_threshold: int = 2,
) -> pd.DataFrame:
    """Annotate motif pairs with intervening and local motif context."""
    if combi_df.empty or motif_locs.empty:
        out = combi_df.copy()
        out["intervening_motif_count"] = np.nan
        out["local_motif_count"] = np.nan
        out["isolated_pair"] = False
        return out

    motif = motif_locs.copy()
    motif["motif_center"] = (motif["start_rel"] + motif["end_rel"]) / 2.0
    by_region = {}
    for region, df in motif.groupby("region", sort=False):
        by_region[region] = {
            "center": df["motif_center"].to_numpy(dtype=float),
            "start": df["start_rel"].to_numpy(dtype=float),
            "end": df["end_rel"].to_numpy(dtype=float),
            "tf": df["tf"].astype(str).to_numpy(),
        }

    rows = []
    for row in combi_df.itertuples(index=False):
        d = row._asdict()
        region_data = by_region.get(d["region"])
        if region_data is None:
            rows.append((np.nan, np.nan, False))
            continue

        left_end = min(d["end1_rel"], d["end2_rel"])
        right_start = max(d["start1_rel"], d["start2_rel"])
        pair_start = min(d["start1_rel"], d["start2_rel"])
        pair_end = max(d["end1_rel"], d["end2_rel"])
        centers = region_data["center"]
        intervening = int(((centers > left_end) & (centers < right_start)).sum())

        in_local = (centers >= pair_start - local_window) & (centers <= pair_end + local_window)
        is_first = (
            (region_data["start"] == d["start1_rel"])
            & (region_data["end"] == d["end1_rel"])
            & (region_data["tf"] == str(d["tf1"]))
        )
        is_second = (
            (region_data["start"] == d["start2_rel"])
            & (region_data["end"] == d["end2_rel"])
            & (region_data["tf"] == str(d["tf2"]))
        )
        local_count = int((in_local & ~(is_first | is_second)).sum())
        isolated = intervening == 0 and local_count <= local_threshold
        rows.append((intervening, local_count, isolated))

    out = combi_df.copy()
    context = pd.DataFrame(rows, columns=["intervening_motif_count", "local_motif_count", "isolated_pair"])
    return pd.concat([out.reset_index(drop=True), context], axis=1)


def build_context_long(results: Mapping[str, TaskResult], local_window: int = 50, local_threshold: int = 2) -> pd.DataFrame:
    frames = []
    for task in results.values():
        df = add_pair_context(task.motif_combi_isa, task.motif_locs, local_window=local_window, local_threshold=local_threshold)
        inter_col = f"interaction_t{task.track}"
        if inter_col not in df.columns:
            continue
        df["task"] = task.name
        df["track"] = task.track
        df["interaction"] = df[inter_col]
        df["context"] = np.where(df["isolated_pair"], "Isolated", "Embedded")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_distance_context(
    results: Mapping[str, TaskResult],
    outpath: Optional[Path] = None,
    local_window: int = 50,
    local_threshold: int = 2,
    bin_width: int = 10,
    min_bin_n: int = 30,
    figsize: Tuple[float, float] = (7.1, 2.4),
) -> pd.DataFrame:
    context = build_context_long(results, local_window=local_window, local_threshold=local_threshold)
    if context.empty:
        return pd.DataFrame()

    context["distance_bin"] = (np.floor(context["distance"] / bin_width) * bin_width).astype(int)
    summary = (
        context.dropna(subset=["interaction"])
        .groupby(["task", "context", "distance_bin"], observed=True)
        .agg(mean_interaction=("interaction", "mean"), median_abs_interaction=("interaction", lambda x: np.median(np.abs(x))), n=("interaction", "size"))
        .reset_index()
    )
    summary["passes_n_filter"] = summary["n"] >= min_bin_n

    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=figsize, sharey=True)
    if len(TASK_ORDER) == 1:
        axes = [axes]
    palette = {"Isolated": "#222222", "Embedded": "#c44e52"}
    for ax, task_name in zip(axes, TASK_ORDER):
        sub = summary[summary["task"] == task_name]
        for ctx, ctx_df in sub.groupby("context", sort=False):
            high = ctx_df[ctx_df["passes_n_filter"]]
            low = ctx_df[~ctx_df["passes_n_filter"]]
            if not high.empty:
                ax.plot(high["distance_bin"], high["mean_interaction"], marker="o", ms=2.2, lw=0.8, color=palette.get(ctx, "0.4"), label=ctx)
            if not low.empty:
                ax.plot(low["distance_bin"], low["mean_interaction"], marker="o", ms=2.0, lw=0.55, color=palette.get(ctx, "0.4"), alpha=0.25, label=None)
        ax.axhline(0, color="0.4", lw=0.45, ls=":")
        ax.set_title(task_name)
        ax.set_xlabel("Motif-pair distance (bp)")
        ax.set_ylabel("Mean interaction" if task_name == TASK_ORDER[0] else "")
        ax.text(0.02, 0.04, f"solid: n >= {min_bin_n}", transform=ax.transAxes, fontsize=5.2, color="0.35")
        despine(ax)
    axes[-1].legend(frameon=False, loc="best")
    fig.tight_layout(w_pad=1.0)
    savefig(fig, outpath)
    return summary


def plot_recurrent_heatmap_and_switch(
    results: Mapping[str, TaskResult],
    outpath: Optional[Path] = None,
    figsize: Tuple[float, float] = (7.1, 4.0),
) -> pd.DataFrame:
    coop = build_coop_long(results, include_independent=False)
    pair_sets = recurrent_pair_sets(results)
    coop["group"] = coop["tf_pair"].map(lambda x: classify_recurrent_pair(x, pair_sets))
    recurrent = coop[coop["group"] != "Specific"].copy()
    if recurrent.empty:
        return pd.DataFrame()

    pivot = recurrent.pivot_table(index="tf_pair", columns="task", values="coop_score", aggfunc="first")
    row_group = recurrent.drop_duplicates("tf_pair").set_index("tf_pair")["group"]
    order = row_group.sort_values().index
    pivot = pivot.reindex(order)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        pivot[[c for c in TASK_ORDER if c in pivot.columns]],
        ax=ax,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Coop score", "shrink": 0.75},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Recurrent CAGE-with-E/D pairs")
    ax.tick_params(axis="y", labelsize=4.8)

    ax = fig.add_subplot(gs[0, 1])
    if {"CAGE", "D/DEV"}.issubset(pivot.columns):
        ax.scatter(pivot["CAGE"], pivot["D/DEV"], s=16, color=TASK_META["D/DEV"]["color"], alpha=0.75, label="CAGE vs D")
    if {"CAGE", "E/HK-like"}.issubset(pivot.columns):
        ax.scatter(pivot["CAGE"], pivot["E/HK-like"], s=16, color=TASK_META["E/HK-like"]["color"], alpha=0.75, label="CAGE vs E")
    ax.axhline(0, color="0.45", lw=0.45, ls=":")
    ax.axvline(0, color="0.45", lw=0.45, ls=":")
    ax.plot([-1, 1], [-1, 1], color="0.7", lw=0.45, ls="--")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("CAGE coop score")
    ax.set_ylabel("Other task coop score")
    ax.set_title("Direction switching")
    ax.legend(frameon=False, loc="lower right")
    despine(ax)

    savefig(fig, outpath)
    return recurrent


def plot_recurrent_paired_differences(
    results: Mapping[str, TaskResult],
    outpath: Optional[Path] = None,
    figsize: Tuple[float, float] = (4.8, 2.4),
) -> pd.DataFrame:
    """Main/extended support: paired recurrent-pair score differences."""
    coop = build_coop_long(results, include_independent=False)
    pair_sets = recurrent_pair_sets(results)
    coop["group"] = coop["tf_pair"].map(lambda x: classify_recurrent_pair(x, pair_sets))
    pivot = coop.pivot_table(index="tf_pair", columns="task", values="coop_score", aggfunc="first")
    rows = []
    comparisons = [("CAGE", "D/DEV", "CAGE-with-D"), ("CAGE", "E/HK-like", "CAGE-with-E")]
    for a, b, group in comparisons:
        valid_pairs = pair_sets.get(group, set())
        for pair in sorted(valid_pairs):
            if pair in pivot.index and a in pivot.columns and b in pivot.columns and pd.notna(pivot.loc[pair, a]) and pd.notna(pivot.loc[pair, b]):
                rows.append(
                    {
                        "tf_pair": pair,
                        "comparison": f"{b} - {a}",
                        "group": group,
                        "score_a": pivot.loc[pair, a],
                        "score_b": pivot.loc[pair, b],
                        "delta": pivot.loc[pair, b] - pivot.loc[pair, a],
                    }
                )
    diff = pd.DataFrame(rows)
    if diff.empty:
        return diff

    stats = []
    for comp, sub in diff.groupby("comparison", sort=False):
        vals = sub["delta"].dropna()
        if len(vals) > 0:
            pos = int((vals > 0).sum())
            neg = int((vals < 0).sum())
            try:
                from scipy.stats import wilcoxon

                p = wilcoxon(vals, alternative="two-sided", zero_method="wilcox").pvalue if len(vals) > 1 else np.nan
            except Exception:
                p = np.nan
            stats.append({"comparison": comp, "n": len(vals), "median_delta": vals.median(), "positive": pos, "negative": neg, "wilcoxon_p": p})
    stats_df = pd.DataFrame(stats)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    palette = {"D/DEV - CAGE": "#d95f02", "E/HK-like - CAGE": "#2ca02c"}
    short_labels = {"D/DEV - CAGE": "D-CAGE", "E/HK-like - CAGE": "E-CAGE"}
    for ax, (comp, sub) in zip(axes, diff.groupby("comparison", sort=False)):
        sub = sub.copy()
        sub["comparison_short"] = short_labels.get(comp, comp)
        ax.axhline(0, color="0.5", lw=0.45, ls=":")
        sns.stripplot(data=sub, x="comparison_short", y="delta", ax=ax, color=palette.get(comp, "0.3"), size=3.2, jitter=0.18)
        sns.boxplot(data=sub, x="comparison_short", y="delta", ax=ax, width=0.35, showfliers=False, boxprops={"facecolor": "none", "edgecolor": "0.25", "linewidth": 0.6}, medianprops={"color": "0.1", "linewidth": 0.8}, whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6})
        st = stats_df[stats_df["comparison"] == comp].iloc[0]
        ax.set_xlabel("")
        ax.set_ylabel("Paired score difference" if ax is axes[0] else "")
        ax.set_title(comp)
        ax.text(0.02, 0.95, f"n={int(st['n'])}\nmedian={st['median_delta']:.2f}\np={st['wilcoxon_p']:.1e}", transform=ax.transAxes, va="top", fontsize=5.4)
        ax.tick_params(axis="x", labelrotation=0)
        despine(ax)
    fig.tight_layout(w_pad=0.8)
    savefig(fig, outpath)
    return diff.merge(stats_df, on="comparison", how="left")


def plot_recurrent_combined_figure(
    results: Mapping[str, TaskResult],
    outpath: Optional[Path] = None,
    figsize: Tuple[float, float] = (7.1, 5.2),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Manuscript-strength Fig5: recurrent scores, direction switching, and paired tests."""
    coop = build_coop_long(results, include_independent=False)
    pair_sets = recurrent_pair_sets(results)
    coop["group"] = coop["tf_pair"].map(lambda x: classify_recurrent_pair(x, pair_sets))
    recurrent = coop[coop["group"] != "Specific"].copy()
    if recurrent.empty:
        return recurrent, pd.DataFrame()

    pivot = recurrent.pivot_table(index="tf_pair", columns="task", values="coop_score", aggfunc="first")
    row_group = recurrent.drop_duplicates("tf_pair").set_index("tf_pair")["group"]
    order = row_group.sort_values().index
    pivot = pivot.reindex(order)

    diff_rows = []
    comparisons = [("CAGE", "D/DEV", "CAGE-with-D"), ("CAGE", "E/HK-like", "CAGE-with-E")]
    for a, b, group in comparisons:
        for pair in sorted(pair_sets.get(group, set())):
            if pair in pivot.index and a in pivot.columns and b in pivot.columns and pd.notna(pivot.loc[pair, a]) and pd.notna(pivot.loc[pair, b]):
                diff_rows.append(
                    {
                        "tf_pair": pair,
                        "comparison": f"{b} - {a}",
                        "comparison_short": "D-CAGE" if b == "D/DEV" else "E-CAGE",
                        "group": group,
                        "score_a": pivot.loc[pair, a],
                        "score_b": pivot.loc[pair, b],
                        "delta": pivot.loc[pair, b] - pivot.loc[pair, a],
                    }
                )
    diff = pd.DataFrame(diff_rows)
    stats = []
    for comp, sub in diff.groupby("comparison", sort=False):
        vals = sub["delta"].dropna()
        try:
            from scipy.stats import wilcoxon

            p = wilcoxon(vals, alternative="two-sided", zero_method="wilcox").pvalue if len(vals) > 1 else np.nan
        except Exception:
            p = np.nan
        stats.append(
            {
                "comparison": comp,
                "n": len(vals),
                "median_delta": vals.median(),
                "positive": int((vals > 0).sum()),
                "negative": int((vals < 0).sum()),
                "wilcoxon_p": p,
            }
        )
    stats_df = pd.DataFrame(stats)
    diff_annot = diff.merge(stats_df, on="comparison", how="left")

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 0.95], wspace=0.38, hspace=0.48)

    ax = fig.add_subplot(gs[:, 0])
    sns.heatmap(
        pivot[[c for c in TASK_ORDER if c in pivot.columns]],
        ax=ax,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Coop score", "shrink": 0.65},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Recurrent CAGE-with-E/D pairs")
    ax.tick_params(axis="y", labelsize=4.6)
    ax.text(-0.08, 1.03, "a", transform=ax.transAxes, fontweight="bold", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    if {"CAGE", "D/DEV"}.issubset(pivot.columns):
        ax.scatter(pivot["CAGE"], pivot["D/DEV"], s=14, color=TASK_META["D/DEV"]["color"], alpha=0.8, label="CAGE vs D")
    if {"CAGE", "E/HK-like"}.issubset(pivot.columns):
        ax.scatter(pivot["CAGE"], pivot["E/HK-like"], s=14, color=TASK_META["E/HK-like"]["color"], alpha=0.8, label="CAGE vs E")
    ax.axhline(0, color="0.45", lw=0.45, ls=":")
    ax.axvline(0, color="0.45", lw=0.45, ls=":")
    ax.plot([-1, 1], [-1, 1], color="0.7", lw=0.45, ls="--")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("CAGE coop score")
    ax.set_ylabel("Other-task coop score")
    ax.set_title("Direction switching")
    ax.legend(frameon=False, loc="lower right")
    ax.text(-0.16, 1.06, "b", transform=ax.transAxes, fontweight="bold", fontsize=8)
    despine(ax)

    ax = fig.add_subplot(gs[1, 1])
    if not diff_annot.empty:
        palette = {"D-CAGE": TASK_META["D/DEV"]["color"], "E-CAGE": TASK_META["E/HK-like"]["color"]}
        order_short = ["D-CAGE", "E-CAGE"]
        ax.axhline(0, color="0.5", lw=0.45, ls=":")
        sns.boxplot(
            data=diff_annot,
            x="comparison_short",
            y="delta",
            order=order_short,
            ax=ax,
            width=0.45,
            showfliers=False,
            boxprops={"facecolor": "none", "edgecolor": "0.25", "linewidth": 0.6},
            medianprops={"color": "0.1", "linewidth": 0.8},
            whiskerprops={"linewidth": 0.6},
            capprops={"linewidth": 0.6},
        )
        sns.stripplot(data=diff_annot, x="comparison_short", y="delta", order=order_short, ax=ax, hue="comparison_short", palette=palette, size=3.0, jitter=0.18, legend=False)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.22 * (ymax - ymin))
        for idx, label in enumerate(order_short):
            comp = "D/DEV - CAGE" if label == "D-CAGE" else "E/HK-like - CAGE"
            st = stats_df[stats_df["comparison"] == comp]
            if not st.empty:
                row = st.iloc[0]
                ax.text(0.27 + idx * 0.47, 0.98, f"n={int(row['n'])}\nP={_format_pvalue(row['wilcoxon_p'])}", transform=ax.transAxes, ha="center", va="top", fontsize=5.2)
    ax.set_xlabel("")
    ax.set_ylabel("Paired score difference")
    ax.set_title("Paired recurrent-pair shift")
    ax.text(-0.16, 1.06, "c", transform=ax.transAxes, fontweight="bold", fontsize=8)
    despine(ax)

    savefig(fig, outpath)
    return recurrent, diff_annot


def plot_tf_level_summary(
    results: Mapping[str, TaskResult],
    outpath: Optional[Path] = None,
    top_n: int = 12,
    figsize: Tuple[float, float] = (7.1, 3.0),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tf_rows = []
    spec_rows = []
    for task in results.values():
        tf_coop = _non_independent(task.coop_tf)
        imp = task.tf_importance.copy()
        if tf_coop.empty or imp.empty:
            continue
        mean_cols = [c for c in imp.columns if c.startswith("mean_isa")]
        if not mean_cols:
            continue
        mean_col = mean_cols[0]
        merged = tf_coop[["tf", "coop_score", "abs_i_sum"]].merge(imp[["tf", mean_col]], on="tf", how="inner")
        merged["importance_z"] = (merged[mean_col] - merged[mean_col].mean()) / (merged[mean_col].std(ddof=0) or 1)
        merged["task"] = task.name
        tf_rows.append(merged)

        cp = _non_independent(task.coop_pair)
        if not cp.empty:
            split = cp["tf_pair"].str.split("|", expand=True)
            long = pd.concat(
                [
                    cp[["abs_i_sum"]].assign(tf=split[0], partner=split[1]),
                    cp[["abs_i_sum"]].assign(tf=split[1], partner=split[0]),
                ],
                ignore_index=True,
            )
            def ratio(group: pd.DataFrame) -> float:
                total = group["abs_i_sum"].sum()
                if total <= 0:
                    return np.nan
                return group.nlargest(2, "abs_i_sum")["abs_i_sum"].sum() / total
            spec = long.groupby("tf", observed=True).apply(ratio).rename("top2_partner_ratio").reset_index()
            spec["task"] = task.name
            spec_rows.append(spec)

    tf_df = pd.concat(tf_rows, ignore_index=True) if tf_rows else pd.DataFrame()
    spec_df = pd.concat(spec_rows, ignore_index=True) if spec_rows else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[0]
    for task_name in TASK_ORDER:
        sub = tf_df[tf_df["task"] == task_name]
        if sub.empty:
            continue
        ax.scatter(sub["coop_score"], sub["importance_z"], s=18, alpha=0.75, color=TASK_META[task_name]["color"], label=task_name)
    ax.axvline(0, color="0.5", lw=0.45, ls=":")
    ax.axhline(0, color="0.5", lw=0.45, ls=":")
    ax.set_xlabel("TF coop score")
    ax.set_ylabel("Standardized TF importance")
    ax.legend(frameon=False, loc="best")
    despine(ax)

    ax = axes[1]
    if not spec_df.empty:
        order = (
            spec_df.groupby("tf")["top2_partner_ratio"]
            .median()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        plot_df = spec_df[spec_df["tf"].isin(order)].copy()
        sns.stripplot(data=plot_df, x="top2_partner_ratio", y="tf", hue="task", order=order, ax=ax, dodge=True, size=3, palette={k: v["color"] for k, v in TASK_META.items()})
        ax.legend(frameon=False, loc="lower right", title=None)
    ax.set_xlabel("Top-2 partner contribution ratio")
    ax.set_ylabel("")
    despine(ax)
    fig.tight_layout(w_pad=1.0)
    savefig(fig, outpath)
    return tf_df, spec_df


def plot_extended_full_heatmaps(
    results: Mapping[str, TaskResult],
    outpath: Optional[Path] = None,
    figsize: Tuple[float, float] = (7.1, 2.6),
) -> Dict[str, pd.DataFrame]:
    """Extended figure: full significant pair heatmaps for each task."""
    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=figsize)
    matrices: Dict[str, pd.DataFrame] = {}
    for ax, task_name in zip(axes, TASK_ORDER):
        task = results[task_name]
        df = _non_independent(task.coop_pair)
        if df.empty:
            ax.axis("off")
            continue
        split = df["tf_pair"].str.split("|", expand=True)
        df = df.assign(tf1=split[0], tf2=split[1])
        mirror = df.rename(columns={"tf1": "tf2", "tf2": "tf1"})
        full = pd.concat([df, mirror], ignore_index=True)
        matrix = full.pivot_table(index="tf1", columns="tf2", values="coop_score", aggfunc="mean")
        order = sorted(set(matrix.index) | set(matrix.columns))
        matrix = matrix.reindex(index=order, columns=order)
        matrices[task_name] = matrix
        sns.heatmap(matrix, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, center=0, cbar=task_name == TASK_ORDER[-1], xticklabels=True, yticklabels=True, linewidths=0.15)
        ax.set_title(f"{task_name} significant pairs")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=90, labelsize=4.8)
        ax.tick_params(axis="y", labelsize=4.8)
    fig.tight_layout(w_pad=0.8)
    savefig(fig, outpath)
    return matrices


def plot_extended_quality_metrics(
    results: Mapping[str, TaskResult],
    outpath: Optional[Path] = None,
    figsize: Tuple[float, float] = (7.1, 2.4),
) -> pd.DataFrame:
    """Extended figure: n_effective, median distance, and q-value distributions."""
    coop = build_coop_long(results, include_independent=False)
    if coop.empty:
        return pd.DataFrame()
    coop = coop.copy()
    min_q = 1e-50
    coop["q_clipped_at"] = min_q
    coop["q_was_clipped"] = coop["mw_q"] < min_q
    coop["neg_log10_q"] = -np.log10(coop["mw_q"].clip(lower=min_q))

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    metrics = [
        ("n_effective", "Effective pair instances"),
        ("median_distance", "Median motif-pair distance"),
        ("neg_log10_q", "-log10(q), capped at 50"),
    ]
    for ax, (col, label) in zip(axes, metrics):
        sns.boxplot(data=coop, x="task", y=col, ax=ax, order=TASK_ORDER, color="white", fliersize=2, linewidth=0.6)
        sns.stripplot(data=coop, x="task", y=col, ax=ax, order=TASK_ORDER, hue="task", palette={k: v["color"] for k, v in TASK_META.items()}, size=2.2, jitter=0.18, alpha=0.75, legend=False)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=25)
        despine(ax)
    fig.tight_layout(w_pad=1.0)
    savefig(fig, outpath)
    return coop


def plot_extended_category_audit(
    results: Mapping[str, TaskResult],
    outpath: Optional[Path] = None,
    figsize: Tuple[float, float] = (7.1, 2.6),
) -> pd.DataFrame:
    """Extended figure: old relative labels against absolute sign categories."""
    coop = build_coop_long(results, include_independent=False)
    if coop.empty:
        return pd.DataFrame()
    audit = (
        coop.groupby(["task", "cooperativity", "sign_class"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=figsize, sharey=True)
    palette = {"Negative": "#4575b4", "Near-zero": "0.7", "Positive": "#d73027"}
    for ax, task_name in zip(axes, TASK_ORDER):
        sub = audit[audit["task"] == task_name]
        pivot = sub.pivot_table(index="cooperativity", columns="sign_class", values="n", fill_value=0, aggfunc="sum")
        for col in ["Negative", "Near-zero", "Positive"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[["Negative", "Near-zero", "Positive"]]
        bottom = np.zeros(len(pivot))
        for cls in pivot.columns:
            vals = pivot[cls].to_numpy()
            ax.bar(pivot.index.astype(str), vals, bottom=bottom, color=palette[cls], edgecolor="white", linewidth=0.4, label=cls)
            bottom += vals
        ax.set_title(task_name)
        ax.set_xlabel("")
        ax.set_ylabel("TF pairs" if task_name == TASK_ORDER[0] else "")
        ax.tick_params(axis="x", rotation=35)
        despine(ax)
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), title=None)
    fig.tight_layout(w_pad=1.0)
    savefig(fig, outpath)
    return audit


def plot_extended_motif_position_by_tf(
    results: Mapping[str, TaskResult],
    outpath: Optional[Path] = None,
    top_n: int = 12,
    figsize: Tuple[float, float] = (7.1, 3.2),
) -> pd.DataFrame:
    """Extended figure: motif positions in sequence and TF-level median positions."""
    frames = []
    for task in results.values():
        df = task.motif_locs.copy()
        if df.empty:
            continue
        df["task"] = task.name
        df["motif_center"] = (df["start_rel"] + df["end_rel"]) / 2.0
        frames.append(df)
    pos = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if pos.empty:
        return pd.DataFrame()

    top_tfs = pos["tf"].value_counts().head(top_n).index
    heat = (
        pos[pos["tf"].isin(top_tfs)]
        .groupby(["tf", "task"], observed=True)["motif_center"]
        .median()
        .reset_index()
        .pivot(index="tf", columns="task", values="motif_center")
        .reindex(index=top_tfs)
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax = axes[0]
    for task_name in TASK_ORDER:
        sub = pos[pos["task"] == task_name]
        sns.kdeplot(sub["motif_center"], ax=ax, color=TASK_META[task_name]["color"], lw=0.9, label=f"{task_name}")
    ax.axvline(124.5, color="0.35", lw=0.5, ls=":")
    ax.set_xlabel("Motif center in 249-bp sequence")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    despine(ax)

    ax = axes[1]
    sns.heatmap(heat[[c for c in TASK_ORDER if c in heat.columns]], ax=ax, cmap="viridis", cbar_kws={"label": "Median center"}, linewidths=0.2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Top {top_n} TF motif positions")
    ax.tick_params(axis="y", labelsize=5.2)
    fig.tight_layout(w_pad=1.0)
    savefig(fig, outpath)
    return pos
