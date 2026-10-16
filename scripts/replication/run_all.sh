#!/usr/bin/env bash
# Full paper replication chain: Dataset A (style/catdog), Dataset B (toxicity),
# then geometry + correlation reports for both.
#
# Plain POSIX bash; no scheduler directives. Chains:
#   catdog_full.sh          (validate, split, build-vectors, steer, merge-steered,
#                            judge, aggregate, geometry + matched audit)
#   toxicity_full.sh        (split rebuild + verify from TOXICITY_DATA_DIR, then
#                            build-vectors, steer, judge, geometry, aggregate)
#   geometry_and_correlations.sh  (aggregation + full/partial correlations with
#                            production stars; reuses the geometry computed above)
#
# Usage:
#   bash scripts/replication/run_all.sh                  # full chain (GPUs)
#   bash scripts/replication/run_all.sh --validate-only  # CPU-only stages:
#                                                        # config/dataset validation,
#                                                        # deterministic splits,
#                                                        # split-verification
#
# Environment:
#   TOXICITY_DATA_DIR  required for the toxicity section (see toxicity_full.sh
#                      and datasets/toxicity/DATASET.md). In --validate-only
#                      mode it is still required so the split can be rebuilt
#                      and verified.
#   ROOT, MODEL_ID, JUDGE_MODEL_ID, DATASET, OUT, CATDOG_OUT, TOX_OUT,
#   SPLITS_DIR   forwarded to the stage scripts (see each script's header).
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"

VALIDATE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --validate-only) VALIDATE_ONLY=1 ;;
    --help|-h) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [ "$VALIDATE_ONLY" = 1 ]; then
  bash "${HERE}/catdog_full.sh" --validate-only
  bash "${HERE}/toxicity_full.sh" --validate-only
  echo "run_all --validate-only complete (CPU-only checks passed)."
  exit 0
fi

bash "${HERE}/catdog_full.sh"
bash "${HERE}/toxicity_full.sh"
GEOMETRY_ALREADY_DONE=1 bash "${HERE}/geometry_and_correlations.sh"
echo "run_all: full replication chain complete."
