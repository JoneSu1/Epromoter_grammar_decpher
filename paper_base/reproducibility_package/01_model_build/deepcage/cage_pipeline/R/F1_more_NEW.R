suppressPackageStartupMessages({
  library(optparse)
  library(CAGEfightR)
  library(GenomicRanges)
  library(rtracklayer)
  library(GenomicFeatures)
  library(SummarizedExperiment)
  library(dplyr)
  library(BiocParallel)
  library(S4Vectors)
})

# -----------------------------
# 1. 参数定义 (Arguments)
# -----------------------------
option_list <- list(
  make_option("--project_root", default = "/home/junhua/Documents/Drophila/promoter_model"),
  make_option("--cage_dir",     default = "data/cage/raw"),
  make_option("--gtf",          default = "/home/junhua/Documents/Drophila/refGene.txt"),
  make_option("--output",       default = "/home/junhua/Documents/Drophila/promoter_model/data/cage/process/N_dataset_neg_more_ctss_only_plus_random_mainchr.tsv"),

  # 样本数量设置
  make_option("--negA_n",       type="integer", default = 44435, help="Total number of random negative samples to generate."),

  # 窗口大小
  make_option("--up_bp",        type="integer", default = 200),
  make_option("--down_bp",      type="integer", default = 48),
  make_option("--seed",         type="integer", default = 123),

  # 过滤参数
  make_option("--rep_min",      type="integer", default = 2,
              help="Minimum number of samples required to support a TC (reproducibility)."),
  make_option("--tpm_th",       type="double", default = 1.0,
              help="Minimum TPM required to count a sample as supported (Default: 1.0)."),

  # 染色体白名单
  make_option("--main_chrs",    type="character", default = "chr2L,chr2R,chr3L,chr3R,chrX,chr4",
              help="Comma-separated whitelist of chromosomes to keep."),

  # 注释参数
  make_option("--tss_tol",      type="integer", default = 250,
              help="Distance tolerance to annotate a CAGE peak as RefSeq overlap."),
  make_option("--ignore_strand_anno", type="logical", default = FALSE,
              help="Ignore strand in peak-to-TSS annotation (recommended FALSE).")
)
opt <- parse_args(OptionParser(option_list = option_list))

# 全局变量
TOTAL_WIDTH <- opt$up_bp + 1 + opt$down_bp
msg <- function(...) cat(format(Sys.time(), "[%Y-%m-%d %H:%M:%S]"), ..., "\n")
set.seed(opt$seed)

# 路径处理
project_root <- normalizePath(opt$project_root)
cage_dir <- file.path(project_root, opt$cage_dir)
out_path <- opt$output
if (!dir.exists(dirname(out_path))) dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

main_chrs <- trimws(unlist(strsplit(opt$main_chrs, ",")))
if (length(main_chrs) == 0) stop("main_chrs is empty. Provide --main_chrs.")

tss_tol <- as.integer(opt$tss_tol)
tpm_threshold <- as.numeric(opt$tpm_th)

# -----------------------------
# 2. 辅助函数 (Helper Functions)
# -----------------------------
quantifyClustersOlap <- function(object, clusters, inputAssay) {
  revmap <- GenomicRanges::reduce(clusters, with.revmap = TRUE)$revmap
  max.nr <- max(sapply(revmap, length))
  ids <- lapply(seq_len(max.nr), function(level) setdiff(sapply(revmap, `[`, level), NA))

  res <- BiocParallel::bplapply(seq_len(max.nr), function(i) {
    clu <- clusters[ids[[i]]]
    obj <- object
    spec <- FALSE
    if (length(clu) == 1) {
      spec <- TRUE
      clu <- c(clu, GenomicRanges::shift(clu, width(clu)))
    }
    suppressMessages(m_ <- CAGEfightR::quantifyClusters(object=obj, clusters=clu, inputAssay=inputAssay))
    if (spec) m_ <- m_[1]
    SummarizedExperiment::assays(m_)[[inputAssay]]
  })

  mat <- do.call("rbind", res)[order(unlist(ids)), ]
  rownames(mat) <- names(clusters)

  o <- SummarizedExperiment::SummarizedExperiment(
    assays = S4Vectors::SimpleList(mat),
    rowRanges = clusters,
    colData = SummarizedExperiment::colData(object)
  )
  names(SummarizedExperiment::assays(o)) <- inputAssay
  o
}

counts_to_tpm <- function(count_mat) {
  libsizes <- colSums(count_mat)
  libsizes[libsizes == 0] <- NA_real_
  tpm <- sweep(count_mat, 2, libsizes, FUN="/") * 1e6
  tpm[is.na(tpm)] <- 0
  tpm
}

# -----------------------------
# 3. 加载 RefSeq 注释
# -----------------------------
msg("Loading RefSeq Data...")
if (grepl("\\.txt$", opt$gtf)) {
  ref <- read.table(opt$gtf, header=FALSE, sep="\t", stringsAsFactors=FALSE, quote="")
  ref <- ref[ref$V3 %in% main_chrs, ]

  tss_pos <- ifelse(ref$V4 == "+", ref$V5 + 1, ref$V6)
  ref_gr <- GRanges(ref$V3, IRanges(tss_pos, width=1), strand=ref$V4)
} else {
  txdb <- makeTxDbFromGFF(opt$gtf)
  ref_gr <- promoters(txdb, upstream=0, downstream=1)
}
ref_window <- promoters(ref_gr, upstream=opt$up_bp, downstream=opt$down_bp + 1)

# -----------------------------
# 4. 加载与处理 CAGE 数据
# -----------------------------
msg("Loading CAGE BigWigs...")
plus_files  <- sort(list.files(cage_dir, pattern="\\.plus\\.bw$", full.names=TRUE))
minus_files <- sort(list.files(cage_dir, pattern="\\.minus\\.bw$", full.names=TRUE))
stopifnot(length(plus_files) == length(minus_files), length(plus_files) > 0)

sample_names <- sub("\\.plus\\.bw$", "", basename(plus_files))
names(plus_files)  <- sample_names
names(minus_files) <- sample_names
design <- data.frame(sample=sample_names, row.names=sample_names)
bw_plus  <- BigWigFileList(plus_files)
bw_minus <- BigWigFileList(minus_files)

msg("Quantifying CTSSs...")
ctss <- quantifyCTSSs(plusStrand=bw_plus, minusStrand=bw_minus, design=design)

# 过滤染色体
ctss_chrs <- intersect(seqlevels(ctss), main_chrs)
if (length(ctss_chrs) == 0) stop("No valid chromosomes found in CTSS data.")
ctss <- keepSeqlevels(ctss, ctss_chrs, pruning.mode="coarse")

# pooled library size
total_library_size_all <- sum(colSums(assay(ctss, "counts")))
if (!is.finite(total_library_size_all) || total_library_size_all <= 0) stop("Invalid pooled library size.")

# 修正：同步 seqlevels/seqinfo（避免 for(obj in list(...)) 的无效回写）
ref_window <- keepSeqlevels(ref_window, intersect(seqlevels(ref_window), seqlevels(ctss)), pruning.mode="coarse")
ref_gr     <- keepSeqlevels(ref_gr,     intersect(seqlevels(ref_gr),     seqlevels(ctss)), pruning.mode="coarse")
seqlevels(ref_window) <- seqlevels(ctss)
seqlevels(ref_gr)     <- seqlevels(ctss)
seqinfo(ref_window)   <- seqinfo(ctss)
seqinfo(ref_gr)       <- seqinfo(ctss)

ref_window <- trim(ref_window[width(ref_window) == TOTAL_WIDTH])
ref_gr     <- trim(ref_gr[width(ref_gr) == 1])

# -----------------------------
# 5. 正样本: CAGE Dominant Windows
# -----------------------------
msg("Clustering and Defining Dominant Windows...")
ctss_pool <- calcPooled(ctss, inputAssay="counts")
TCs <- clusterUnidirectionally(ctss_pool, pooledCutoff=0, mergeDist=20)

dom_peaks_ir <- mcols(TCs)$thick
dom_peaks_gr <- GRanges(seqnames(TCs), dom_peaks_ir, strand(TCs))

ctss_window <- promoters(dom_peaks_gr, upstream=opt$up_bp, downstream=opt$down_bp + 1)
tc_peak     <- resize(dom_peaks_gr, width=1, fix="center")

# 对齐 seqinfo
ctss_window <- keepSeqlevels(ctss_window, seqlevels(ctss), pruning.mode="coarse")
tc_peak     <- keepSeqlevels(tc_peak,     seqlevels(ctss), pruning.mode="coarse")
seqlevels(ctss_window) <- seqlevels(ctss)
seqlevels(tc_peak)     <- seqlevels(ctss)
seqinfo(ctss_window)   <- seqinfo(ctss)
seqinfo(tc_peak)       <- seqinfo(ctss)

ctss_window <- trim(ctss_window)
valid_idx <- width(ctss_window) == TOTAL_WIDTH
ctss_window <- ctss_window[valid_idx]
tc_peak     <- tc_peak[valid_idx]

msg("Quantifying Positive Candidates...")
ctss_se <- quantifyClustersOlap(ctss, clusters=ctss_window, inputAssay="counts")
ctss_counts_mat <- assay(ctss_se, "counts")

# -----------------------------
# 6. 质量控制过滤
# -----------------------------
keep_rep <- rep(TRUE, nrow(ctss_counts_mat))
support_n <- rep(NA_integer_, nrow(ctss_counts_mat))

if (opt$rep_min > 0) {
  ctss_tpm_mat <- counts_to_tpm(ctss_counts_mat)
  support_n <- rowSums(ctss_tpm_mat >= tpm_threshold)
  keep_rep <- support_n >= opt$rep_min
  msg(sprintf("Filtering: Keeping %d / %d regions (TPM >= %.1f in >= %d samples).",
              sum(keep_rep), length(keep_rep), tpm_threshold, opt$rep_min))
}

ctss_window <- ctss_window[keep_rep]
tc_peak     <- tc_peak[keep_rep]
ctss_counts_mat <- ctss_counts_mat[keep_rep, , drop=FALSE]
if (opt$rep_min > 0) support_n <- support_n[keep_rep]

ctss_counts_sum <- rowSums(ctss_counts_mat)
ctss_tpm_pooled <- (ctss_counts_sum / total_library_size_all) * 1e6
y_ctss <- log2(ctss_tpm_pooled + 1)

# 正样本 strand 分布（用于负样本比例匹配）
pos_strand_chr <- as.character(strand(ctss_window))
msg("Positive strand distribution (after filtering):")
print(table(pos_strand_chr))
p_plus <- mean(pos_strand_chr == "+")
if (!is.finite(p_plus) || p_plus <= 0 || p_plus >= 1) {
  msg("WARNING: Positive strand proportion is extreme; defaulting to 0.5/0.5 for negatives.")
  p_plus <- 0.5
}

# -----------------------------
# 7. 正样本标注
# -----------------------------
msg(paste("Annotating TC peaks to Ref TSS (Tol:", opt$tss_tol, "bp)..."))
nn <- distanceToNearest(tc_peak, ref_gr, ignore.strand = opt$ignore_strand_anno)
is_ref_overlap <- rep(FALSE, length(tc_peak))
if (length(nn) > 0) {
  is_ref_overlap[queryHits(nn)] <- (mcols(nn)$distance <= tss_tol)
}

ctss_df <- data.frame(
  chrom  = as.character(seqnames(ctss_window)),
  start  = start(ctss_window),
  end    = end(ctss_window),
  strand = as.character(strand(ctss_window)),
  label  = "ctss",
  y      = y_ctss,
  source = ifelse(is_ref_overlap, "CTSS_Overlap_RefTSS", "CTSS_NoRefTSS"),
  peak_pos = start(tc_peak)
)
if (opt$rep_min > 0) ctss_df$support_n <- support_n

# -----------------------------
# 8. 负样本: Balanced & Spatial-Uniform (strand-matched negatives)
# -----------------------------
msg("Generating Balanced & Spatially Uniform Background (Promoter-Depleted)...")

mask_gr <- reduce(c(GRanges(ctss_window), GRanges(ref_window)), ignore.strand=TRUE)

seq_lengths <- seqlengths(ctss)
valid_chrs <- intersect(names(seq_lengths), main_chrs)
whole_genome <- GRanges(valid_chrs, IRanges(1, seq_lengths[valid_chrs]))

available_space <- GenomicRanges::setdiff(whole_genome, mask_gr, ignore.strand=TRUE)

pos_counts_by_chr <- table(seqnames(ctss_window))
total_pos_n <- length(ctss_window)
target_ratio <- opt$negA_n / total_pos_n
rand_list <- list()

N_BINS <- 10
msg(paste("Target Ratio (Neg/Pos):", round(target_ratio, 2)))
msg(paste("Spatial Binning: Splitting each chromosome into", N_BINS, "bins."))
msg(sprintf("Negative strand sampling will match positives: P(+)=%.3f, P(-)=%.3f", p_plus, 1 - p_plus))

for (chr in valid_chrs) {
  n_pos_on_chr <- pos_counts_by_chr[[chr]]
  if (is.na(n_pos_on_chr) || n_pos_on_chr == 0) next

  n_neg_total_chr <- ceiling(n_pos_on_chr * target_ratio)

  chr_tiles <- unlist(tileGenome(seqinfo(ctss)[chr], ntile = N_BINS))
  n_neg_per_bin <- ceiling(n_neg_total_chr / N_BINS)

  chr_samples <- GRanges()

  for (j in seq_along(chr_tiles)) {
    current_tile <- chr_tiles[j]

    bin_available <- GenomicRanges::intersect(available_space, current_tile, ignore.strand=TRUE)
    bin_available <- bin_available[width(bin_available) >= TOTAL_WIDTH]
    if (length(bin_available) == 0) next

    sampled_indices <- sample(seq_along(bin_available), n_neg_per_bin, replace=TRUE, prob=width(bin_available))
    chosen_ranges <- bin_available[sampled_indices]

    random_shifts <- floor(runif(n_neg_per_bin) * (width(chosen_ranges) - TOTAL_WIDTH))
    final_starts <- start(chosen_ranges) + random_shifts

    # 关键修正：在创建负样本时就赋予 +/- strand，且比例匹配正样本
    neg_strands <- sample(c("+", "-"), length(final_starts), replace=TRUE, prob=c(p_plus, 1 - p_plus))

    bin_gr <- GRanges(chr, IRanges(final_starts, width=TOTAL_WIDTH), strand=neg_strands)
    chr_samples <- c(chr_samples, bin_gr)
  }

  rand_list[[chr]] <- chr_samples
}

rand_gr <- unlist(GRangesList(rand_list))
rand_gr <- unique(rand_gr)

if (length(rand_gr) > opt$negA_n) rand_gr <- rand_gr[sample(length(rand_gr), opt$negA_n)]

seqlevels(rand_gr) <- seqlevels(ctss)
seqinfo(rand_gr)   <- seqinfo(ctss)

# sanity checks：负样本量化前不得含 *
msg("Negative strand distribution BEFORE quantification:")
print(table(as.character(strand(rand_gr))))
stopifnot(!any(as.character(strand(rand_gr)) == "*"))

msg("Quantifying Random Regions...")
rand_se <- quantifyClustersOlap(ctss, clusters=rand_gr, inputAssay="counts")
rand_counts_sum <- rowSums(assay(rand_se, "counts"))
rand_tpm_pooled <- (rand_counts_sum / total_library_size_all) * 1e6
y_rand <- log2(rand_tpm_pooled + 1)

rand_df <- data.frame(
  chrom  = as.character(seqnames(rand_gr)),
  start  = start(rand_gr),
  end    = end(rand_gr),
  strand = as.character(strand(rand_gr)),
  label  = "random",
  y      = y_rand,
  source = "Random_Background"
)

# -----------------------------
# 9. Merge & Output
# -----------------------------
dataset <- bind_rows(ctss_df, rand_df)

widths <- dataset$end - dataset$start + 1
if (any(widths != TOTAL_WIDTH)) stop("CRITICAL: Output width check failed!")

msg("Final Dataset Summary:")
print(table(dataset$source))
print(table(dataset$label))
msg("Chromosomes included: ", paste(sort(unique(dataset$chrom)), collapse=", "))

write.table(dataset, file=out_path, sep="\t", quote=FALSE, row.names=FALSE)
msg("Done! Saved to ", out_path)

# 调用示例：
# Rscript /home/junhua/Documents/Drophila/promoter_model/script/CAGE_PRO/TPM/Model_for_data/F1_more_NEW.R \
#   --main_chrs "chr2L,chr2R,chr3L,chr3R,chrX,chr4" \
#   --negA_n 44435 \
#   --rep_min 2 \
#   --tpm_th 1.0 \
#   --tss_tol 250
#   --up_bp 200 \
#   --down_bp 48 \
#   --seed 123 \