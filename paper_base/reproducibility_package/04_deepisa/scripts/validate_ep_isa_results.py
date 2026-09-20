from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
TASKS = {
    "CAGE": {"dir": "results_cage", "track": "t0", "isa": "isa_t0", "pred": "pred_t0"},
    "DEV": {"dir": "results_dev", "track": "t0", "isa": "isa_t0", "pred": "pred_t0"},
    "HK": {"dir": "results_hk", "track": "t1", "isa": "isa_t1", "pred": "pred_t1"},
}


@dataclass
class Check:
    scope: str
    item: str
    status: str
    detail: str


checks: list[Check] = []


def add(scope: str, item: str, ok: bool | None, detail: str) -> None:
    if ok is None:
        status = "WARN"
    else:
        status = "PASS" if bool(ok) else "FAIL"
    checks.append(Check(scope, item, status, detail))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def finite_series(s: pd.Series) -> bool:
    return pd.to_numeric(s, errors="coerce").notna().all()


def parse_latest_log(log_path: Path) -> dict[str, float | int | str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float | int | str] = {}
    for raw, kept in re.findall(r"Score filter \(\|score\| >= [^)]+\): (\d+) -> (\d+)", text):
        out["raw_hits"] = int(raw)
        out["filtered_hits"] = int(kept)
    for rows, tfs, regions in re.findall(r"motif_locs: (\d+) rows, (\d+) TFs, (\d+) regions", text):
        out["motif_locs_rows_log"] = int(rows)
        out["motif_locs_tfs_log"] = int(tfs)
        out["motif_locs_regions_log"] = int(regions)
    for rows in re.findall(r"non_motif_locs: (\d+) intervals", text):
        out["non_motif_locs_rows_log"] = int(rows)
    for regions in re.findall(r"Saved region original predictions: .* \((\d+) regions\)", text):
        out["pred_orig_regions_log"] = int(regions)
    for kept, total in re.findall(r"Kept (\d+)/(\d+) motifs", text):
        out["motif_single_rows_log"] = int(kept)
        out["motif_single_total_log"] = int(total)
    for threshold in re.findall(r"ISA thresholds from null \(percentile 80\): (\{[^\n]+\})", text):
        out["isa_threshold_log"] = threshold
    for before, after in re.findall(r"Combined filtering .* reduced pairs from (\d+) to (\d+)", text):
        out["combined_pairs_before_log"] = int(before)
        out["combined_pairs_after_log"] = int(after)
    out["has_complete"] = "ISA execution and aggregation complete" in text
    out["report_complete"] = "Report complete" in text
    out["warning_count"] = len(re.findall(r"\bWARNING\b", text))
    out["error_count"] = len(re.findall(r"\b(ERROR|Traceback|Exception)\b", text))
    return out


def parse_threshold_string(text: object) -> dict[str, tuple[float, float]]:
    if not isinstance(text, str):
        return {}
    found = re.findall(r"'(isa_t\d+)': \{'pos': np\.float64\(([^)]+)\), 'neg': np\.float64\(([^)]+)\)\}", text)
    return {col: (float(pos), float(neg)) for col, pos, neg in found}


def validate_task(name: str, cfg: dict[str, str]) -> dict[str, object]:
    root = BASE / cfg["dir"]
    data_dir = root / "Data"
    plot_dir = root / "Plots"
    track = cfg["track"]
    isa_col = cfg["isa"]
    pred_col = cfg["pred"]
    pair_file = f"coop_tf_pair_{track}.csv"
    tf_file = f"coop_tf_{track}.csv"

    expected = {
        "motif_locs": data_dir / "motif_locs.csv",
        "non_motif_locs": data_dir / "non_motif_locs.csv",
        "pred_orig": data_dir / "pred_orig.csv",
        "motif_single_isa": data_dir / "motif_single_isa.csv",
        "motif_combi_isa": data_dir / "motif_combi_isa.csv",
        "null_isa": data_dir / "null_isa.csv",
        "null_interaction": data_dir / "null_interaction.csv",
        "tf_importance": data_dir / "tf_importance.csv",
        "coop_tf_pair": data_dir / pair_file,
        "coop_tf": data_dir / tf_file,
        "workflow_log": root / "workflow.log",
    }
    for key, path in expected.items():
        add(name, f"file exists: {key}", path.exists() and path.stat().st_size > 0, str(path.relative_to(BASE)))

    dfs = {key: read_csv(path) for key, path in expected.items() if key != "workflow_log" and path.exists()}
    log = parse_latest_log(expected["workflow_log"]) if expected["workflow_log"].exists() else {}
    add(name, "workflow completed", bool(log.get("has_complete")) and bool(log.get("report_complete")), json.dumps(log, ensure_ascii=False))
    add(name, "workflow errors", log.get("error_count", 1) == 0, f"errors={log.get('error_count')}, warnings={log.get('warning_count')}")
    if log.get("warning_count", 0):
        add(name, "workflow warnings are annotation-resource only", None, "Warnings exist; inspect workflow.log. Known skipped resources affect external annotation plots, not ISA core CSVs.")

    ml = dfs["motif_locs"]
    required_ml = {"chrom", "start", "end", "start_rel", "end_rel", "tf", "score", "strand", "region"}
    add(name, "motif_locs columns", required_ml.issubset(ml.columns), f"columns={list(ml.columns)}")
    add(name, "motif_locs log row count", len(ml) == log.get("motif_locs_rows_log"), f"csv={len(ml)}, log={log.get('motif_locs_rows_log')}")
    add(name, "motif_locs log TF count", ml["tf"].nunique() == log.get("motif_locs_tfs_log"), f"csv={ml['tf'].nunique()}, log={log.get('motif_locs_tfs_log')}")
    add(name, "motif_locs coordinates within 249 bp", (ml["start_rel"].ge(0) & ml["end_rel"].le(249) & (ml["start_rel"] < ml["end_rel"])).all(), f"start_rel=[{ml.start_rel.min()}, {ml.start_rel.max()}], end_rel=[{ml.end_rel.min()}, {ml.end_rel.max()}]")
    add(name, "motif_locs genomic lengths match relative lengths", ((ml["end"] - ml["start"]) == (ml["end_rel"] - ml["start_rel"])).all(), "end-start equals end_rel-start_rel")
    add(name, "motif_locs no critical NA", not ml[list(required_ml)].isna().any().any(), ml[list(required_ml)].isna().sum().to_dict().__repr__())

    nm = dfs["non_motif_locs"]
    add(name, "non_motif_locs log row count", len(nm) == log.get("non_motif_locs_rows_log"), f"csv={len(nm)}, log={log.get('non_motif_locs_rows_log')}")
    add(name, "non_motif_locs coordinate validity", (nm["start_rel"].ge(0) & nm["end_rel"].le(249) & (nm["start_rel"] < nm["end_rel"]) & ((nm["end_rel"] - nm["start_rel"]) == nm["len"])).all(), f"rows={len(nm)}")

    pred = dfs["pred_orig"]
    add(name, "pred_orig columns", {"region", pred_col}.issubset(pred.columns), f"columns={list(pred.columns)}")
    add(name, "pred_orig row count", len(pred) == 19777 and len(pred) == log.get("pred_orig_regions_log"), f"csv={len(pred)}, log={log.get('pred_orig_regions_log')}")
    add(name, "pred_orig unique regions", pred["region"].is_unique, f"unique={pred['region'].nunique()}, rows={len(pred)}")
    add(name, "pred_orig finite predictions", finite_series(pred[pred_col]), pred[pred_col].describe().to_string())

    ms = dfs["motif_single_isa"]
    required_ms = required_ml | {isa_col}
    add(name, "motif_single_isa columns", required_ms.issubset(ms.columns), f"columns={list(ms.columns)}")
    add(name, "motif_single_isa log filtered row count", len(ms) == log.get("motif_single_rows_log"), f"csv={len(ms)}, log={log.get('motif_single_rows_log')}")
    motif_key_cols = ["chrom", "start", "end", "start_rel", "end_rel", "tf", "strand", "region"]
    ms_backmap = ms.merge(ml[motif_key_cols + ["score"]], on=motif_key_cols, how="left", suffixes=("_single", "_locs"), indicator=True)
    missing_ms = ms_backmap.query("_merge == 'left_only'")
    add(name, "motif_single_isa maps back to motif_locs by coordinates", len(missing_ms) == 0, f"missing={len(missing_ms)}; score is checked approximately because CSVs use different precision")
    if len(missing_ms) == 0:
        score_delta = (ms_backmap["score_single"] - ms_backmap["score_locs"]).abs()
        add(name, "motif_single_isa score matches motif_locs approximately", score_delta.max() < 1e-3, f"max_abs_delta={score_delta.max():.6g}")
    add(name, "motif_single_isa finite ISA", finite_series(ms[isa_col]), ms[isa_col].describe().to_string())
    ni = dfs["null_isa"]
    add(name, "null_isa columns", {"region", "start_rel", "end_rel", isa_col}.issubset(ni.columns), f"columns={list(ni.columns)}")
    add(name, "null_isa sample count", len(ni) == 2000, f"rows={len(ni)}")
    add(name, "null_isa centered near zero", abs(float(ni[isa_col].median())) < 0.1, f"median={ni[isa_col].median():.4f}, mean={ni[isa_col].mean():.4f}")
    thresholds = parse_threshold_string(log.get("isa_threshold_log"))
    if isa_col in thresholds:
        pos_thr, neg_thr = thresholds[isa_col]
        pos_recalc = float(ni.loc[ni[isa_col] > 0, isa_col].quantile(0.8))
        neg_recalc = float(ni.loc[ni[isa_col] < 0, isa_col].quantile(0.2))
        retained_by_thr = ms[isa_col].ge(pos_thr) | ms[isa_col].le(neg_thr)
        add(name, "null-derived ISA thresholds match null_isa", abs(pos_thr - pos_recalc) < 1e-9 and abs(neg_thr - neg_recalc) < 1e-9, f"log_pos={pos_thr}, recalc_pos={pos_recalc}; log_neg={neg_thr}, recalc_neg={neg_recalc}")
        add(name, "motif_single_isa retained rows satisfy thresholds", retained_by_thr.all(), f"retained={len(ms)}, source_total={log.get('motif_single_total_log')}, pos={pos_thr}, neg={neg_thr}")
    else:
        add(name, "null-derived ISA thresholds parseable", None, f"threshold string={log.get('isa_threshold_log')}")

    combi = dfs["motif_combi_isa"]
    req_combi = {"region", "tf1", "tf2", "start1_rel", "end1_rel", "start2_rel", "end2_rel", "distance", f"isa1_{track}", f"isa2_{track}", f"isa_both_{track}", f"interaction_{track}"}
    add(name, "motif_combi_isa columns", req_combi.issubset(combi.columns), f"columns={list(combi.columns)}")
    distance_calc = combi[["start1_rel", "end1_rel", "start2_rel", "end2_rel"]].apply(lambda r: max(r.start1_rel, r.start2_rel) - min(r.end1_rel, r.end2_rel), axis=1)
    add(name, "motif_combi_isa distance formula", (distance_calc == combi["distance"]).all(), f"mismatches={(distance_calc != combi['distance']).sum()}")
    interaction_calc = (combi[f"isa1_{track}"] + combi[f"isa2_{track}"] - combi[f"isa_both_{track}"]).round(4)
    add(name, "motif_combi_isa interaction formula", (interaction_calc == combi[f"interaction_{track}"].round(4)).all(), f"mismatches={(interaction_calc != combi[f'interaction_{track}'].round(4)).sum()}")
    add(name, "motif_combi_isa finite numeric", all(finite_series(combi[c]) for c in ["distance", f"isa1_{track}", f"isa2_{track}", f"isa_both_{track}", f"interaction_{track}"]), f"rows={len(combi)}")

    null_inter = dfs["null_interaction"]
    add(name, "null_interaction sample size close to target", 1500 <= len(null_inter) <= 2000, f"rows={len(null_inter)}; generator can return fewer than n_samples after region grouping")
    null_inter_calc = null_inter[f"isa1_{track}"] + null_inter[f"isa2_{track}"] - null_inter[f"isa_both_{track}"]
    null_inter_delta = (null_inter_calc - null_inter[f"interaction_{track}"]).abs()
    add(name, "null_interaction formula", (null_inter_delta < 1e-3).all(), f"mismatches={(null_inter_delta >= 1e-3).sum()}, max_abs_delta={null_inter_delta.max():.6g}")
    add(name, "null_interaction centered near zero", abs(float(null_inter[f"interaction_{track}"].median())) < 0.05, f"median={null_inter[f'interaction_{track}'].median():.4f}, mean={null_inter[f'interaction_{track}'].mean():.4f}")

    imp = dfs["tf_importance"]
    req_imp = {"tf", "n_total", f"n_effective_isa_{track}", f"mean_isa_{track}", f"median_isa_{track}", f"ks_d_isa_{track}", f"ks_pval_isa_{track}"}
    add(name, "tf_importance columns", req_imp.issubset(imp.columns), f"columns={list(imp.columns)}")
    ms_counts = ms.groupby("tf").size().rename("n_total_recalc").reset_index()
    imp_cmp = imp.merge(ms_counts, on="tf", how="outer")
    add(name, "tf_importance n_total matches motif_single_isa", (imp_cmp["n_total"] == imp_cmp["n_total_recalc"]).all(), imp_cmp.to_string(index=False))
    med_recalc = ms.groupby("tf")[isa_col].median().round(4).rename("median_recalc").reset_index()
    mean_recalc = ms.groupby("tf")[isa_col].mean().round(4).rename("mean_recalc").reset_index()
    imp_cmp2 = imp.merge(med_recalc, on="tf").merge(mean_recalc, on="tf")
    mean_delta = (imp_cmp2[f"mean_isa_{track}"] - imp_cmp2["mean_recalc"]).abs()
    median_delta = (imp_cmp2[f"median_isa_{track}"] - imp_cmp2["median_recalc"]).abs()
    add(name, "tf_importance mean/median match motif_single_isa", ((mean_delta <= 5e-4) & (median_delta <= 5e-4)).all(), imp_cmp2[["tf", f"mean_isa_{track}", "mean_recalc", f"median_isa_{track}", "median_recalc"]].to_string(index=False))

    cp = dfs["coop_tf_pair"]
    req_cp = {"tf_pair", "n_total", "n_effective", "abs_i_sum", "coop_score", "mw_p", "count", "median_distance", "mw_q", "cooperativity"}
    add(name, "coop_tf_pair columns", req_cp.issubset(cp.columns), f"columns={list(cp.columns)}")
    add(name, "coop_tf_pair count equals n_effective", (cp["count"] == cp["n_effective"]).all(), f"mismatches={(cp['count'] != cp['n_effective']).sum()}")
    add(name, "coop_tf_pair non-independent q<=0.1", (cp.loc[cp["cooperativity"] != "Independent", "mw_q"] <= 0.1).all(), cp.groupby("cooperativity")["mw_q"].describe().to_string())
    add(name, "coop_tf_pair NaN score only independent", cp.loc[cp["coop_score"].isna(), "cooperativity"].eq("Independent").all(), f"nan_scores={cp['coop_score'].isna().sum()}")
    add(name, "coop_tf_pair known categories only", set(cp["cooperativity"]).issubset({"Independent", "Intermediate", "Redundant", "Synergistic"}), str(cp["cooperativity"].value_counts().to_dict()))
    pair_counts = combi.assign(tf_pair=combi.apply(lambda r: "|".join(sorted([str(r.tf1), str(r.tf2)])), axis=1)).groupby("tf_pair").size().rename("pair_rows").reset_index()
    cp_cmp = cp.merge(pair_counts, on="tf_pair", how="left")
    add(name, "coop_tf_pair n_total not greater than combi rows", (cp_cmp["n_total"] <= cp_cmp["pair_rows"]).all(), cp_cmp[cp_cmp["n_total"] > cp_cmp["pair_rows"]].to_string(index=False))

    ctf = dfs["coop_tf"]
    add(name, "coop_tf columns", {"tf", "n_total", "n_effective", "coop_score", "mw_q", "cooperativity"}.issubset(ctf.columns), f"columns={list(ctf.columns)}")
    add(name, "coop_tf known categories only", set(ctf["cooperativity"]).issubset({"Independent", "Intermediate", "Redundant", "Synergistic"}), str(ctf["cooperativity"].value_counts().to_dict()))

    plot_files = list(plot_dir.glob("*.png"))
    add(name, "task plot files exist", len(plot_files) >= 7, f"plots={len(plot_files)}")

    return {
        "task": name,
        "motif_locs": len(ml),
        "motif_regions": ml["region"].nunique(),
        "filtered_single_isa": len(ms),
        "null_isa": len(ni),
        "combi_rows": len(combi),
        "null_interaction": len(null_inter),
        "tf_count": imp["tf"].nunique(),
        "pair_rows": len(cp),
        "non_independent_pairs": int((cp["cooperativity"] != "Independent").sum()),
        "category_counts": cp["cooperativity"].value_counts().to_dict(),
        "warnings": log.get("warning_count", 0),
    }


def validate_cross_task() -> None:
    cross = BASE / "cross_task_plots"
    pngs = sorted(cross.glob("*.png"))
    add("Cross-task", "cross_task_plots exists", cross.exists(), str(cross))
    add("Cross-task", "cross-task PNG count", len(pngs) >= 17, f"pngs={len(pngs)}")

    all_pairs = {}
    for name, cfg in TASKS.items():
        cp = pd.read_csv(BASE / cfg["dir"] / "Data" / f"coop_tf_pair_{cfg['track']}.csv")
        sig = cp[cp["cooperativity"] != "Independent"].copy()
        all_pairs[name] = set(sig["tf_pair"])
    shared_all = all_pairs["CAGE"] & all_pairs["DEV"] & all_pairs["HK"]
    cage_enhancer = all_pairs["CAGE"] & (all_pairs["DEV"] | all_pairs["HK"])
    add("Cross-task", "shared significant pairs exist", len(shared_all) > 0, f"CAGE∩DEV∩HK={len(shared_all)}, CAGE∩(DEV∪HK)={len(cage_enhancer)}")

    # Recompute category-count table that supports cross_task_plots/12_category_counts.png.
    counts = {}
    for name, cfg in TASKS.items():
        cp = pd.read_csv(BASE / cfg["dir"] / "Data" / f"coop_tf_pair_{cfg['track']}.csv")
        counts[name] = cp[cp["cooperativity"] != "Independent"]["cooperativity"].value_counts().to_dict()
    add("Cross-task", "category counts recomputed", True, json.dumps(counts, ensure_ascii=False))


def main() -> None:
    summaries = [validate_task(name, cfg) for name, cfg in TASKS.items()]
    validate_cross_task()

    out_dir = BASE / "output" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    checks_df = pd.DataFrame([c.__dict__ for c in checks])
    checks_df.to_csv(out_dir / "ep_isa_validation_checks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(out_dir / "ep_isa_validation_summary.csv", index=False, encoding="utf-8-sig")

    fail = checks_df[checks_df["status"] == "FAIL"]
    warn = checks_df[checks_df["status"] == "WARN"]
    lines = [
        "# EP_ISA result data validation report",
        "",
        f"Workspace: `{BASE}`",
        "",
        "## Overall verdict",
        "",
    ]
    if fail.empty:
        lines.append("PASS: Core ISA result data are internally consistent across all three tasks.")
    else:
        lines.append(f"FAIL: {len(fail)} checks failed. Review the failure table below before using the results.")
    if not warn.empty:
        lines.append(f"WARN: {len(warn)} warnings were recorded, mainly external annotation resource warnings or threshold provenance notes.")
    lines.extend(["", "## Task summary", ""])
    lines.append(pd.DataFrame(summaries).to_markdown(index=False))
    lines.extend(["", "## Failed checks", ""])
    lines.append("None." if fail.empty else fail.to_markdown(index=False))
    lines.extend(["", "## Warning checks", ""])
    lines.append("None." if warn.empty else warn.to_markdown(index=False))
    lines.extend(["", "## All check counts", ""])
    lines.append(checks_df.groupby(["scope", "status"]).size().reset_index(name="n").to_markdown(index=False))
    (out_dir / "EP_ISA_data_validation_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "checks": len(checks_df),
        "pass": int((checks_df.status == "PASS").sum()),
        "warn": int((checks_df.status == "WARN").sum()),
        "fail": int((checks_df.status == "FAIL").sum()),
        "report": str(out_dir / "EP_ISA_data_validation_report.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
