#!/usr/bin/env bash
# Geometry extraction + outcome aggregation + partial-correlation report for both datasets.
#
# Plain POSIX bash; no scheduler directives.
#
# NOTE (parallelization): `repsteer geometry` loops over layers sequentially,
# but each layer is independent — production ran each layer as a separate
# scheduler job (LAYERS 0 3 6 9 ... 57). To parallelize, run one process per
# layer with `--layers L` and concatenate the per-layer CSVs. GPU memory is
# the binding constraint (bfloat16 31B model, one process per GPU). The
# aggregation and correlation stages are CPU-only and cheap.
#
# Usage:
#   bash scripts/replication/geometry_and_correlations.sh [--dataset catdog|toxicity]
#   (default: run both sections)
#
# Inputs (produced by catdog_full.sh / toxicity_full.sh):
#   results/replication/catdog/judged_{general,conditioned}.csv
#   results/replication/toxicity/judged_toxicity_{general,conditioned}.csv
# Outputs per dataset: geometry/geometry_rows.csv, geometry_audit/matched_metrics.csv,
#   outcomes/ (production outcome tables), correlations/correlation_report.csv|md.
#
# Env overrides: ROOT, CATDOG_OUT, TOX_OUT, TOX_SPLITS_DIR, XLMR_SCORES.
# Setting GEOMETRY_ALREADY_DONE=1 skips the geometry stages and only runs the
# aggregation + correlation steps (useful when catdog_full.sh / toxicity_full.sh
# already produced them, or the geometry CSVs came from a parallel shard run).
set -eu

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
CATDOG_OUT="${CATDOG_OUT:-results/replication/catdog}"
TOX_OUT="${TOX_OUT:-results/replication/toxicity}"
TOX_SPLITS_DIR="${TOX_SPLITS_DIR:-${TOX_OUT}/splits}"

DATASETS=""
for arg in "$@"; do
  case "$arg" in
    --dataset) : ;;         # value consumed by the next argument
    catdog|toxicity) DATASETS="$arg" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if [ -z "$DATASETS" ]; then DATASETS="both"; fi

if [ -f "${ROOT}/src/repsteer/cli.py" ] && ! command -v repsteer >/dev/null 2>&1; then
  REPSSTEER=(env PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 -m repsteer.cli)
else
  REPSSTEER=(repsteer)
fi
repsteer_cli() { "${REPSSTEER[@]}" "$@"; }

CATDOG_CFG=configs/replication/catdog_style_general.json
TOX_CFG=configs/replication/toxicity_general.json

run_catdog() {
  if [ "${GEOMETRY_ALREADY_DONE:-0}" != "1" ]; then
    echo "==> catdog geometry (production protocol) + eval-disjoint matched audit"
    repsteer_cli geometry --config "$CATDOG_CFG" \
      --dataset datasets/catdog/corpus.csv --output-dir "${CATDOG_OUT}/geometry"
    repsteer_cli eval-disjoint --config "$CATDOG_CFG" \
      --output "${CATDOG_OUT}/geometry_audit/geometry_dataset_excluding_steering_eval.csv"
    repsteer_cli geometry --config "$CATDOG_CFG" --protocol matched_audit \
      --dataset "${CATDOG_OUT}/geometry_audit/geometry_dataset_excluding_steering_eval.csv" \
      --output-dir "${CATDOG_OUT}/geometry_audit"
  else
    echo "==> GEOMETRY_ALREADY_DONE: skipping catdog geometry extraction"
  fi
  # Production outcomes: analyze_gemma4_eval200_story.py layer summary
  # (n-weighted mean over alpha + best alpha per setup/task/layer).
  echo "==> aggregate: style/catdog"
  repsteer_cli aggregate --dataset catdog \
    --judged "${CATDOG_OUT}/judged_general.csv" --judged "${CATDOG_OUT}/judged_conditioned.csv" \
    --output-dir "${CATDOG_OUT}/outcomes"
  echo "==> correlations: style/catdog"
  repsteer_cli correlations --dataset catdog \
    --geometry "${CATDOG_OUT}/geometry/geometry_rows.csv" \
    --judged "${CATDOG_OUT}/outcomes/confirmatory_layer_summary.csv" \
    --audit-metrics "${CATDOG_OUT}/geometry_audit/matched_metrics.csv" \
    --output-dir "${CATDOG_OUT}/correlations"
}

run_toxicity() {
  if [ "${GEOMETRY_ALREADY_DONE:-0}" != "1" ]; then
    echo "==> toxicity geometry (production protocol) + matched decodability audit"
    repsteer_cli geometry --config "$TOX_CFG" \
      --dataset "${TOX_SPLITS_DIR}/geometry_400.csv" --output-dir "${TOX_OUT}/geometry"
    repsteer_cli geometry --config "$TOX_CFG" --protocol matched_audit \
      --dataset "${TOX_SPLITS_DIR}/geometry_400.csv" --output-dir "${TOX_OUT}/geometry_audit"
  else
    echo "==> GEOMETRY_ALREADY_DONE: skipping toxicity geometry extraction"
  fi
  # Production outcomes: analyze_gemma4_toxicity_replication.py best alpha per
  # (setup, layer) joined with PS/CCGP/SD and matched decodability.
  echo "==> aggregate: toxicity"
  repsteer_cli aggregate --dataset toxicity \
    --judged "${TOX_OUT}/judged_toxicity_general.csv" \
    --judged "${TOX_OUT}/judged_toxicity_conditioned.csv" \
    ${XLMR_SCORES:+--classifier-scores "$XLMR_SCORES"} \
    --geometry-rows "${TOX_OUT}/geometry/geometry_rows.csv" \
    --matched-metrics "${TOX_OUT}/geometry_audit/matched_metrics.csv" \
    --output-dir "${TOX_OUT}/outcomes"
  echo "==> correlations: toxicity"
  repsteer_cli correlations --dataset toxicity \
    --geometry "${TOX_OUT}/geometry/geometry_rows.csv" \
    --judged "${TOX_OUT}/outcomes/toxicity_best_alpha_geometry.csv" \
    --output-dir "${TOX_OUT}/correlations"
}

case "$DATASETS" in
  catdog) run_catdog ;;
  toxicity) run_toxicity ;;
  both) run_catdog; run_toxicity ;;
esac

echo "geometry_and_correlations complete."
echo "  catdog: ${CATDOG_OUT}/correlations/correlation_report.csv|md"
echo "  toxicity: ${TOX_OUT}/correlations/correlation_report.csv|md"
