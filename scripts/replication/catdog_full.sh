#!/usr/bin/env bash
# Dataset A (style/cat-dog) full replication stages.
#
# Plain POSIX bash; no scheduler directives. Every stage is a `repsteer` CLI
# invocation reading configs/replication/catdog_{style,language,concept}_{general,conditioned}.json.
# GPU-bearing stages (build-vectors/steer/judge/geometry) use the production
# settings: google/gemma-4-31B-it (bfloat16, chat template, model.language_model.layers
# n=60 via configs/model_registry.yaml), 32 sampled contrast texts per class,
# alphas 8..128 signed toward the opposite pole, judge Qwen/Qwen3.6-27B greedy
# max_new_tokens 1024 with the production 64-shard A/B assignment and repair pass.
#
# Usage:
#   bash scripts/replication/catdog_full.sh                # full pipeline
#   bash scripts/replication/catdog_full.sh --validate-only # CPU-only stages:
#                                                          # validate, split
# Env overrides: ROOT (repo root), DATASET (defaults datasets/catdog/corpus.csv),
# OUT (defaults results/replication/catdog).
set -eu

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

DATASET="${DATASET:-datasets/catdog/corpus.csv}"
OUT="${OUT:-results/replication/catdog}"
SETUPS="general conditioned"
# Production task order inside every per-layer generations file (TASKS in
# caa_conditioned_steering.py); merge-steered relies on it.
TASKS="style language concept"

VALIDATE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --validate-only) VALIDATE_ONLY=1 ;;
    --help|-h)
      sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# Resolve the repsteer CLI: installed entrypoint if present, else
# PYTHONPATH-based module fallback (no install required).
if command -v repsteer >/dev/null 2>&1; then
  REPSSTEER=(repsteer)
elif [ -n "${PYTHONPATH:-}" ] && python3 -c "import repsteer" >/dev/null 2>&1; then
  REPSSTEER=(python3 -m repsteer.cli)
elif [ -f "${ROOT}/src/repsteer/cli.py" ]; then
  REPSSTEER=(env PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 -m repsteer.cli)
else
  echo "repsteer CLI not found; install with 'pip install -e .' or set PYTHONPATH" >&2
  exit 3
fi
repsteer_cli() { "${REPSSTEER[@]}" "$@"; }

cfg_path() { echo "configs/replication/catdog_$1_$2.json"; }

# Stage 1: validate every task/setup config against the corpus (CPU-only).
for setup in $SETUPS; do
  for task in $TASKS; do
    echo "==> validate $(cfg_path "$task" "$setup") (dataset ${DATASET})"
    repsteer_cli validate --config "$(cfg_path "$task" "$setup")" --dataset "$DATASET"
  done
done

# Stage 2: carve the deterministic 32/25-per-cell splits. The vector and
# evaluation CSVs reproduce production results-gemma4-steering-eval200-split
# row for row (contrast sampling and per-row seeds depend on that order).
echo "==> split -> ${OUT}/splits"
repsteer_cli split --config "$(cfg_path style general)" \
  --dataset "$DATASET" --output-dir "${OUT}/splits"

if [ "$VALIDATE_ONLY" = 1 ]; then
  echo "validate-only: skipping GPU stages (vectors/steer/judge/aggregate/geometry)."
  exit 0
fi

for setup in $SETUPS; do
  steered=()
  for task in $TASKS; do
    cfg="$(cfg_path "$task" "$setup")"
    # Stage 3: per-layer sampled difference-of-means vectors from the vector split.
    echo "==> build-vectors ${task} ${setup}"
    repsteer_cli build-vectors --config "$cfg" \
      --dataset "${OUT}/splits/vector.csv" \
      --output "${OUT}/vectors_${task}_${setup}.npz"
    # Stage 4: baseline + steered continuations on the eval-200 split.
    echo "==> steer ${task} ${setup}"
    repsteer_cli steer --config "$cfg" \
      --dataset "${OUT}/splits/evaluation.csv" \
      --vectors "${OUT}/vectors_${task}_${setup}.npz" \
      --output "${OUT}/steered_${task}_${setup}.csv"
    steered+=(--input "${OUT}/steered_${task}_${setup}.csv")
  done
  # Stage 5: merge the three task tables in production judged-input order and
  # judge them as one table (64-shard round-robin A/B assignment + repair).
  echo "==> merge-steered ${setup}"
  repsteer_cli merge-steered "${steered[@]}" --output "${OUT}/steered_${setup}.csv"
  echo "==> judge ${setup}"
  repsteer_cli judge --config "$(cfg_path style "$setup")" \
    --input "${OUT}/steered_${setup}.csv" \
    --output "${OUT}/judged_${setup}.csv"
done

# Stage 6: production outcome table (confirmatory_layer_summary.csv layout).
echo "==> aggregate -> ${OUT}/outcomes"
repsteer_cli aggregate --dataset catdog \
  --judged "${OUT}/judged_general.csv" --judged "${OUT}/judged_conditioned.csv" \
  --output-dir "${OUT}/outcomes"

# Stage 7: geometry on the full 1041-row pool (production protocol -> geometry_rows.csv),
# then the eval-disjoint matched audit (-> matched_metrics.csv).
echo "==> geometry (production protocol)"
repsteer_cli geometry --config "$(cfg_path style general)" \
  --dataset "$DATASET" --output-dir "${OUT}/geometry"
echo "==> eval-disjoint matched audit"
repsteer_cli eval-disjoint --config "$(cfg_path style general)" \
  --output "${OUT}/geometry_audit/geometry_dataset_excluding_steering_eval.csv"
repsteer_cli geometry --config "$(cfg_path style general)" --protocol matched_audit \
  --dataset "${OUT}/geometry_audit/geometry_dataset_excluding_steering_eval.csv" \
  --output-dir "${OUT}/geometry_audit"

echo "catdog_full stages complete; outputs in ${OUT}."
echo "Correlation report: bash scripts/replication/geometry_and_correlations.sh --dataset catdog"
