#!/usr/bin/env bash
# Dataset B (toxicity) full replication stages.
#
# Plain POSIX bash; no scheduler directives. The candidate texts are NOT
# shipped (third-party social-media posts; see datasets/toxicity/DATASET.md),
# so this script starts by rebuilding the split from a user-supplied
# TOXICITY_DATA_DIR (the directory containing {en,ar}_labeled.csv produced by
# the documented textdetox download + gpt-4.1-mini annotation pipeline) and
# verifying it against the shipped fingerprints in datasets/toxicity/manifest.json.
#
# Usage:
#   bash scripts/replication/toxicity_full.sh                 # full pipeline
#   bash scripts/replication/toxicity_full.sh --validate-only # CPU-only stages:
#                                                             # split rebuild,
#                                                             # verification,
#                                                             # config validation
# Env overrides: MODEL_ID, JUDGE_MODEL_ID, ROOT, DATA_DIR (defaults
# results/replication/toxicity/splits), OUT (defaults results/replication/toxicity),
# LABELS (default $DATA_DIR's sources; see below).
set -eu

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

MODEL_ID="${MODEL_ID:-google/gemma-4-31B-it}"
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-Qwen/Qwen3.6-27B}"
OUT="${OUT:-results/replication/toxicity}"
SPLITS_DIR="${SPLITS_DIR:-${OUT}/splits}"

VALIDATE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --validate-only) VALIDATE_ONLY=1 ;;
    --help|-h) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# Resolve the repsteer CLI: installed entrypoint if present, else
# PYTHONPATH-based module fallback (no install required).
if command -v repsteer >/dev/null 2>&1; then
  REPSSTEER=(repsteer)
elif python3 -c "import repsteer" >/dev/null 2>&1; then
  REPSSTEER=(python3 -m repsteer.cli)
elif [ -f "${ROOT}/src/repsteer/cli.py" ]; then
  REPSSTEER=(env PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 -m repsteer.cli)
else
  echo "repsteer CLI not found; install with 'pip install -e .' or set PYTHONPATH" >&2
  exit 3
fi
repsteer_cli() { "${REPSSTEER[@]}" "$@"; }

# ---- Stage 0: rebuild the annotated splits from user-supplied labeled CSVs ----
if [ -z "${TOXICITY_DATA_DIR:-}" ] || [ ! -d "$TOXICITY_DATA_DIR" ]; then
  cat >&2 <<'EOF'
ERROR: TOXICITY_DATA_DIR is not set (or not a directory). This must point at a
directory containing {en,ar}_labeled.csv produced by the documented pipeline:
  1. download textdetox/multilingual_toxicity_dataset (en, ar) from Hugging Face
  2. re-annotate with gpt-4.1-mini using the exact developer prompt and schema
     in datasets/toxicity/DATASET.md (runner: scripts/toxicity/batch_label_toxicity.py)
  3. assemble the per-language labeled CSVs (columns text, toxic,
     mentions_gender, mentions_location, ...)

See datasets/toxicity/DATASET.md ("Rebuilding the data locally") for details.
EOF
  exit 2
fi

echo "==> prepare_toxicity_splits.py -> ${SPLITS_DIR}"
python3 datasets/toxicity/prepare_toxicity_splits.py \
  --input-dir "$TOXICITY_DATA_DIR" \
  --output-dir "$SPLITS_DIR" \
  --geometry-per-cell 50 --vectors-per-cell 32 --evaluation-per-cell 25 --seed 42

echo "==> verify_split.py (fingerprints vs datasets/toxicity/manifest.json)"
python3 datasets/toxicity/verify_split.py \
  --splits-dir "$SPLITS_DIR" \
  --manifest datasets/toxicity/manifest.json

# ---- Config validation (CPU-only; also runs in --validate-only mode) ----
for setup in general conditioned; do
  echo "==> validate configs/replication/toxicity_${setup}.json"
  repsteer_cli validate --config "configs/replication/toxicity_${setup}.json" \
    --dataset "${SPLITS_DIR}/geometry_400.csv"
done

if [ "$VALIDATE_ONLY" = 1 ]; then
  echo "validate-only: skipping GPU stages (vectors/steer/judge/geometry/aggregate)."
  exit 0
fi

for setup in general conditioned; do
  cfg="configs/replication/toxicity_${setup}.json"
  # ---- Stage 3: per-layer sampled toxic-minus-clean vectors (32 per class) ----
  echo "==> build-vectors toxicity ${setup}"
  repsteer_cli build-vectors --config "$cfg" \
    --dataset "${SPLITS_DIR}/vector_construction_256.csv" \
    --output "${OUT}/vectors_toxicity_${setup}.npz"
  # ---- Stage 4: baseline + steered continuations on the eval-200 split ----
  # Alphas 8..128 are magnitudes signed toward the opposite pole; rows are
  # ordered layer -> alpha -> evaluation row like the per-layer production files.
  echo "==> steer toxicity ${setup}"
  repsteer_cli steer --config "$cfg" \
    --dataset "${SPLITS_DIR}/evaluation_200.csv" \
    --vectors "${OUT}/vectors_toxicity_${setup}.npz" \
    --output "${OUT}/steered_toxicity_${setup}.csv"
  # ---- Stage 5: Qwen3.6-27B judge; A/B alternates within each layer (num_jobs 1) ----
  echo "==> judge toxicity ${setup}"
  repsteer_cli judge --config "$cfg" \
    --input "${OUT}/steered_toxicity_${setup}.csv" \
    --output "${OUT}/judged_toxicity_${setup}.csv"
done

# ---- Stage 6: geometry on the 400-row geometry split (production protocol
# -> geometry_rows.csv) and the matched decodability audit (-> matched_metrics.csv) ----
echo "==> geometry (production protocol)"
repsteer_cli geometry --config configs/replication/toxicity_general.json \
  --dataset "${SPLITS_DIR}/geometry_400.csv" \
  --output-dir "${OUT}/geometry"
echo "==> matched decodability audit"
repsteer_cli geometry --config configs/replication/toxicity_general.json --protocol matched_audit \
  --dataset "${SPLITS_DIR}/geometry_400.csv" \
  --output-dir "${OUT}/geometry_audit"

# ---- Stage 7: production outcome tables (summary by layer/alpha, best alpha
# joined with geometry). XLMR_SCORES optionally points at a CSV of
# text_hash,toxicity_probability (textdetox/xlmr-large-toxicity-classifier-v2)
# to add classifier_target_delta.
echo "==> aggregate -> ${OUT}/outcomes"
repsteer_cli aggregate --dataset toxicity \
  --judged "${OUT}/judged_toxicity_general.csv" --judged "${OUT}/judged_toxicity_conditioned.csv" \
  ${XLMR_SCORES:+--classifier-scores "$XLMR_SCORES"} \
  --geometry-rows "${OUT}/geometry/geometry_rows.csv" \
  --matched-metrics "${OUT}/geometry_audit/matched_metrics.csv" \
  --output-dir "${OUT}/outcomes"

echo "toxicity_full stages complete; outputs in ${OUT}."
echo "Correlation report: bash scripts/replication/geometry_and_correlations.sh --dataset toxicity"
