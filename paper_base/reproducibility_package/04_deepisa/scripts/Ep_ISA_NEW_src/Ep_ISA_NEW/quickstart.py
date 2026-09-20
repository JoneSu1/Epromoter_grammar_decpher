import os
import pandas as pd
from loguru import logger
from Ep_ISA_NEW.adapter.finemo_io import (
    load_finemo_hits,
    prepare_region_map,
    hits_to_motif_locs,
    compute_non_motif_regions,
    suggest_score_threshold,
)
from Ep_ISA_NEW.scoring.single_isa import run_single_isa, calc_tf_importance
from Ep_ISA_NEW.scoring.combi_isa import (
    run_combi_isa,
    run_null_interaction,
    add_interaction,
    calc_coop_score,
)
from Ep_ISA_NEW.scoring.preflight import run_preflight_pair_audit
from Ep_ISA_NEW.utils import setup_logger, find_available_gpu

from Ep_ISA_NEW.plotting.interaction import (
    plot_null_isa,
    plot_null_interaction,
    plot_interaction_decay,
)
from Ep_ISA_NEW.plotting.cooperativity import (
    hist_coop_score,
    heatmap_coop_score,
    plot_motif_distance_by_category,
)
from Ep_ISA_NEW.plotting.tf import (
    plot_motif_gc_by_coop,
    plot_coop_vs_importance,
    plot_partner_specificity,
)
from Ep_ISA_NEW.exploring.tf_family import (
    plot_coop_by_tf_pair_family,
    plot_coop_by_dbd,
    plot_intra_family_coop_score,
)
from Ep_ISA_NEW.exploring.tf_pair_ppi import (
    plot_ppi_enrichment,
    plot_cofactor_recruitment,
    plot_dna_mediated_ppi,
)
from Ep_ISA_NEW.exploring.tf_function import (
    plot_usf_pfs,
    plot_cell_specificity,
)


ISA_STAGES = [
    "preflight_audit",
    "single_isa",
    "combi_isa",
    "null_interaction",
    "aggregate_isa",
]


class EpQuickStart:
    """
    Orchestrator for ISA analysis on Fi-NeMo scan results.

    Unlike deepISA.QuickStart, this skips the map_motifs stage (Fi-NeMo
    already found motifs) and starts directly from single_isa.
    """

    def __init__(self, results_dir, fasta_path, df_regions, device=None):
        self.results_dir = results_dir
        self.data_dir = os.path.join(self.results_dir, "Data")
        self.plots_dir = os.path.join(self.results_dir, "Plots")
        self.model_dir = os.path.join(self.results_dir, "Models")
        setup_logger(self.results_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.files = {
            "motif_locs":     os.path.join(self.data_dir, "motif_locs.csv"),
            "non_motif_locs": os.path.join(self.data_dir, "non_motif_locs.csv"),
            "preflight_pair_audit": os.path.join(self.data_dir, "preflight_motif_pair_audit.csv"),
            "preflight_pair_audit_by_region": os.path.join(self.data_dir, "preflight_motif_pair_audit_by_region.csv"),
            "preflight_overlap_pairs": os.path.join(self.data_dir, "preflight_overlap_or_abutting_pairs.csv"),
            "null_isa":       os.path.join(self.data_dir, "null_isa.csv"),
            "pred_orig":      os.path.join(self.data_dir, "pred_orig.csv"),
            "isa_single":     os.path.join(self.data_dir, "motif_single_isa.csv"),
            "isa_combi":      os.path.join(self.data_dir, "motif_combi_isa.csv"),
            "null_interaction": os.path.join(self.data_dir, "null_interaction.csv"),
            "imp_tf":         os.path.join(self.data_dir, "tf_importance.csv"),
            "coop_tf_pair":   os.path.join(self.data_dir, "coop_tf_pair.csv"),
            "coop_tf":        os.path.join(self.data_dir, "coop_tf.csv"),
        }

        self.fasta_path = fasta_path
        self.df_regions = df_regions
        self.region_map = prepare_region_map(df_regions)
        self.df_train = None
        self.device = device if device is not None else find_available_gpu()
        self.model = None
        self.tracks = [0]

    def define_model(self, model_obj):
        self.model = model_obj
        rf = getattr(model_obj, 'rf', None)
        if rf:
            logger.info(f"Model receptive field: {rf}")
        else:
            logger.info("Model receptive field not set (default 255 for combi_isa)")
        logger.info("TF/Keras model internalized successfully.")

    def load_checkpoint(self, path=None):
        import tensorflow as tf
        if path is None:
            path = os.path.join(self.model_dir, "model_best")
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Loaded TF/Keras model from: {path}")

    def load_finemo(
        self,
        hits_tsv_path,
        finemo_h5_path=None,
        score_col='hit_coefficient',
        score_threshold=None,
        similarity_threshold=None,
        auto_threshold_percentile=None,
    ):
        """
        Load Fi-NeMo hits and produce motif_locs.csv + non_motif_locs.csv.
        This replaces deepISA's map_motifs stage.

        auto_threshold_percentile: if set (e.g. 50), automatically compute
            score_threshold from the score distribution at this percentile.
            Useful because Fi-NeMo scores are float, not JASPAR integers.
        """
        logger.info(f"Loading Fi-NeMo hits: {hits_tsv_path}")
        df_hits = load_finemo_hits(hits_tsv_path)
        logger.info(f"  {len(df_hits)} raw hits")

        if auto_threshold_percentile is not None and score_threshold is None:
            score_threshold = suggest_score_threshold(
                df_hits, score_col=score_col,
                percentile=auto_threshold_percentile)

        df_motif_locs = hits_to_motif_locs(
            df_hits,
            region_map=self.region_map,
            score_col=score_col,
            score_threshold=score_threshold,
            finemo_h5_path=finemo_h5_path,
            similarity_threshold=similarity_threshold,
        )
        df_motif_locs.to_csv(self.files["motif_locs"], index=False)
        logger.info(f"Saved motif_locs: {self.files['motif_locs']}")

        df_non_motif = compute_non_motif_regions(
            df_motif_locs, self.region_map)
        df_non_motif.to_csv(self.files["non_motif_locs"], index=False)
        logger.info(f"Saved non_motif_locs: {self.files['non_motif_locs']}")

    def _validate_start_from(self, start_from):
        if start_from not in ISA_STAGES:
            raise ValueError(
                f"Invalid start_from='{start_from}'. "
                f"Expected one of: {ISA_STAGES}")

    def _check_isa_dependencies(self, start_from):
        deps = {
            "single_isa": [
                self.files["motif_locs"], self.files["non_motif_locs"],
            ],
            "preflight_audit": [
                self.files["motif_locs"],
            ],
            "combi_isa": [
                self.files["isa_single"], self.files["pred_orig"],
                self.files["null_isa"], self.files["non_motif_locs"],
            ],
            "null_interaction": [
                self.files["isa_combi"], self.files["pred_orig"],
                self.files["isa_single"], self.files["null_isa"],
                self.files["non_motif_locs"],
            ],
            "aggregate_isa": [
                self.files["isa_single"], self.files["null_isa"],
                self.files["isa_combi"], self.files["null_interaction"],
            ],
        }
        required = deps.get(start_from, [])

        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            missing_str = "\n".join([f" - {p}" for p in missing])
            raise FileNotFoundError(
                f"Cannot start from stage '{start_from}' because "
                f"required files are missing:\n{missing_str}\n"
                f"Run load_finemo() and/or earlier ISA stages first.")

    def run_isa(
        self,
        isa_config,
        start_from="preflight_audit",
    ):
        if self.model is None:
            raise ValueError("Model not defined. Call define_model() first.")

        self._validate_start_from(start_from)
        self._check_isa_dependencies(start_from)

        self.tracks = isa_config.get('tracks', [0])
        start_idx = ISA_STAGES.index(start_from)
        receptive_field = isa_config.get(
            "receptive_field", getattr(self.model, "rf", 255))

        if start_idx <= ISA_STAGES.index("preflight_audit"):
            logger.info("Running stage: preflight_audit")
            run_preflight_pair_audit(
                motif_locs_path=self.files["motif_locs"],
                out_summary_path=self.files["preflight_pair_audit"],
                out_region_path=self.files["preflight_pair_audit_by_region"],
                out_overlap_path=self.files["preflight_overlap_pairs"],
                receptive_field=receptive_field,
            )
        else:
            logger.info(f"Skipping: preflight_audit (start_from='{start_from}')")

        if start_idx <= ISA_STAGES.index("single_isa"):
            logger.info("Running stage: single_isa")
            run_single_isa(
                model=self.model,
                fasta=self.fasta_path,
                motif_locs_path=self.files["motif_locs"],
                non_motif_locs_path=self.files["non_motif_locs"],
                single_isa_outpath=self.files["isa_single"],
                null_isa_outpath=self.files["null_isa"],
                null_percentile=isa_config.get("null_percentile", 80),
                pred_orig_outpath=self.files["pred_orig"],
                device=self.device,
                tracks=self.tracks,
                num_regions_per_batch=isa_config.get("num_regions_per_batch", 200),
                pred_batch_size=isa_config.get("pred_batch_size", 1024),
                null_n_samples=isa_config.get("single_null_n_samples", 8192),
                single_filter_mode=isa_config.get("single_filter_mode", "positive_all_tracks"),
            )
        else:
            logger.info(f"Skipping: single_isa (start_from='{start_from}')")

        if start_idx <= ISA_STAGES.index("combi_isa"):
            logger.info("Running stage: combi_isa")
            run_combi_isa(
                model=self.model,
                fasta=self.fasta_path,
                single_isa_path=self.files["isa_single"],
                outpath=self.files["isa_combi"],
                device=self.device,
                tracks=self.tracks,
                receptive_field=receptive_field,
                pred_orig_path=self.files["pred_orig"],
                num_regions_per_batch=isa_config.get("num_regions_per_batch", 200),
                pred_batch_size=isa_config.get("pred_batch_size", 1024),
            )
        else:
            logger.info(f"Skipping: combi_isa")

        if start_idx <= ISA_STAGES.index("null_interaction"):
            logger.info("Running stage: null_interaction")
            run_null_interaction(
                model=self.model,
                fasta=self.fasta_path,
                non_motif_locs_path=self.files["non_motif_locs"],
                combi_isa_path=self.files["isa_combi"],
                pred_orig_path=self.files["pred_orig"],
                tracks=self.tracks,
                outpath=self.files["null_interaction"],
                device=self.device,
                n_samples=isa_config.get("pair_null_n_samples", 8192),
                receptive_field=receptive_field,
                n_bins=isa_config.get("pair_null_n_bins", 20),
                num_regions_per_batch=isa_config.get("num_regions_per_batch", 200),
                pred_batch_size=isa_config.get("pred_batch_size", 1024),
            )
        else:
            logger.info(f"Skipping: null_interaction")

        if start_idx <= ISA_STAGES.index("aggregate_isa"):
            logger.info("Running stage: aggregate_isa")
            add_interaction(
                combi_isa_path=self.files["isa_combi"],
                null_interaction_path=self.files["null_interaction"],
                null_isa_path=self.files["null_isa"],
                tracks=self.tracks,
                null_percentile=isa_config.get("null_percentile", 80),
                tau_quantile=isa_config.get("tau_quantile", 50.0),
            )
            calc_tf_importance(
                self.files["isa_single"],
                self.files["null_isa"],
                self.files["imp_tf"],
                null_percentile=isa_config.get("null_percentile", 80),
                min_count=isa_config.get("min_count", 10),
            )

            for t in self.tracks:
                calc_coop_score(
                    self.files["isa_combi"],
                    self.files["null_interaction"],
                    outpath=self.files["coop_tf_pair"].replace(".csv", f"_t{t}.csv"),
                    level="tf_pair",
                    track_idx=t,
                    null_percentile=isa_config.get("null_percentile", 80),
                    min_count=isa_config.get("min_count", 10),
                    q_val_thresh=isa_config.get("q_val_thresh", 0.1),
                )
                calc_coop_score(
                    self.files["isa_combi"],
                    self.files["null_interaction"],
                    outpath=self.files["coop_tf"].replace(".csv", f"_t{t}.csv"),
                    level="tf",
                    track_idx=t,
                    null_percentile=isa_config.get("null_percentile", 80),
                    min_count=isa_config.get("min_count", 10),
                    q_val_thresh=isa_config.get("q_val_thresh", 0.1),
                )
        else:
            logger.info(f"Skipping: aggregate_isa")

        logger.info("ISA execution and aggregation complete.")

    def report(self):
        logger.info("Generating comprehensive reports and plots...")

        def safe(fn, *args, **kwargs):
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Skipped {fn.__name__}: {e}")

        if os.path.exists(self.files["isa_combi"]):
            df_isa_combi = pd.read_csv(self.files["isa_combi"])
        else:
            df_isa_combi = None

        if os.path.exists(self.files["null_isa"]):
            safe(plot_null_isa, self.files["null_isa"], tracks=self.tracks,
                 outpath=os.path.join(self.plots_dir, "null_isa.png"))
        if os.path.exists(self.files["null_interaction"]):
            safe(plot_null_interaction, self.files["null_interaction"],
                 tracks=self.tracks,
                 outpath=os.path.join(self.plots_dir, "null_interaction.png"))
        if df_isa_combi is not None:
            safe(plot_interaction_decay, df_isa_combi, self.tracks, mode='signed',
                 outpath=os.path.join(self.plots_dir, "interaction_decay_signed.png"))

        for t in self.tracks:
            t_suffix = f"_t{t}"

            def ppath(name):
                return os.path.join(self.plots_dir, f"{name}{t_suffix}.png")

            coop_pair_path = self.files["coop_tf_pair"].replace(".csv", f"_t{t}.csv")
            coop_tf_path = self.files["coop_tf"].replace(".csv", f"_t{t}.csv")
            imp_path = self.files["imp_tf"]

            if not os.path.exists(coop_pair_path) or not os.path.exists(coop_tf_path):
                logger.warning(f"Results for track {t} not found. Skipping.")
                continue

            df_coop_pair = pd.read_csv(coop_pair_path)
            df_coop_tf = pd.read_csv(coop_tf_path)
            df_imp = pd.read_csv(imp_path) if os.path.exists(imp_path) else None

            safe(hist_coop_score, df_coop_pair, outpath=ppath("coop_score_hist"))
            safe(heatmap_coop_score, df_coop_pair, outpath=ppath("coop_score_heatmap"))
            safe(plot_motif_distance_by_category, df_coop_pair, outpath=ppath("distance_by_category"))

            safe(plot_motif_gc_by_coop, df_coop_tf, outpath=ppath("motif_gc_by_coop"))
            if df_imp is not None:
                safe(plot_coop_vs_importance, df_coop_tf, df_imp,
                     x_col="coop_score", y_col=f"mean_isa_t{t}",
                     outpath=ppath("coop_vs_importance"))
            safe(plot_partner_specificity, df_coop_pair, df_coop_tf,
                 outpath=ppath("partner_specificity_ratio"))

            safe(plot_coop_by_tf_pair_family, df_coop_pair, outpath=ppath("family_coop_summary"))
            safe(plot_coop_by_dbd, df_coop_tf, outpath=ppath("dbd_coop_summary"))
            safe(plot_intra_family_coop_score, df_coop_pair, outpath=ppath("intra_family_distribution"))

            safe(plot_usf_pfs, df_coop_tf, outpath=ppath("usf_pioneer_ecdf"))
            safe(plot_cell_specificity, df_coop_tf, outpath=ppath("rolling_gini_specificity"))

            safe(plot_ppi_enrichment, df_coop_pair, rank_by="coop_score",
                 outpath=ppath("ppi_enrichment_by_coop_score"))
            safe(plot_ppi_enrichment, df_coop_pair, rank_by="p_val",
                 outpath=ppath("ppi_enrichment_by_p_val"))
            safe(plot_cofactor_recruitment, df_coop_pair, outpath=ppath("ppi_violin_validation"))
            safe(plot_dna_mediated_ppi, df_coop_pair, rank_by="coop_score",
                 outpath=ppath("dna_ppi_enrichment_by_score"))
            safe(plot_dna_mediated_ppi, df_coop_pair, rank_by="p_val",
                 outpath=ppath("dna_ppi_enrichment_by_pval"))

        logger.info(f"Report complete. Plots saved to {self.plots_dir}")
