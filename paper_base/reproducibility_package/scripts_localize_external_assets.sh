#!/usr/bin/env bash
# Helper to (re)localize the remaining G-drive external assets into the package.
# Run when Google Drive is mounted (G:\). Idempotent — safe to re-run.
#
# Phase 3.5 status: Fig5d fully localized; Fig3e finemo_input + beds localized;
#   attributions.h5 (~2.2GB) was mid-transfer. This script completes/re-verifies it.
set -e
PKG="$(cd "$(dirname "$0")" && pwd)"
GDRIVE="G:/我的云端硬盘"

echo "=== Localizing Sharing Motif attributions.h5 (Fig3e/series2 input) ==="
PROM_BASE="$GDRIVE/DeepEpromote/Drosophila/Motif_cluster/prom_scan_results/prom_run_20260415_041448"
for task in HK DEV CAGE; do
  src="$PROM_BASE/$task/attributions.h5"
  dst="$PKG/frozen_data/sharing_motif_logos/attributions/$task/attributions.h5"
  mkdir -p "$(dirname "$dst")"
  if [ ! -f "$src" ]; then echo "  SOURCE MISSING (G-drive not mounted?): $src"; continue; fi
  # skip if already present and hash-matches
  if [ -f "$dst" ]; then
    sh=$(md5sum "$src" | cut -d' ' -f1); ph=$(md5sum "$dst" | cut -d' ' -f1)
    if [ "$sh" = "$ph" ]; then echo "  ✓ $task already localized (hash match)"; continue; fi
  fi
  echo "  copying $task (~$(stat -c %s "$src" | awk '{print int($1/1000000)}')MB)..."
  cp "$src" "$dst"
  sh=$(md5sum "$src" | cut -d' ' -f1); ph=$(md5sum "$dst" | cut -d' ' -f1)
  [ "$sh" = "$ph" ] && echo "    ✓ $task hash match" || echo "    ✗ $task MISMATCH"
done
echo ""
echo "=== Verifying all localized external assets ==="
echo "Fig5d (evolution_fig5d):"
ATTR_BASE="$GDRIVE/DeepEpromote/Drosophila/DeepSTARR/promoter_mut/results_context_dependent/reviewer_grade_runs/reviewer_greedy_20260724_115905/attribution_finemo_24bp_annotated"
for track in HK DEV CAGE; do
  for f in attribution_arrays.npz metadata.tsv; do
    [ -f "$PKG/frozen_data/evolution_fig5d/attribution/$track/$f" ] && echo "  ✓ $track/$f" || echo "  ✗ $track/$f MISSING"
  done
done
[ -f "$PKG/frozen_data/evolution_fig5d/summaries/finemo_hits_annotated_long.tsv" ] && echo "  ✓ finemo_hits_long" || echo "  ✗ finemo_hits_long MISSING"
echo "Done."
