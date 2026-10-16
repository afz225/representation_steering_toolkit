# Replication scripts (plain POSIX bash, no scheduler directives)

These scripts drive the full production pipeline of the paper through the
`repsteer` CLI. Defaults reproduce the production settings exactly; paths can be
overridden by environment variables.

| script | what it does |
|---|---|
| `catdog_full.sh` | Dataset A (style/cat-dog): validate the 6 task×setup configs, deterministic 32/25-per-cell splits, sampled steering vectors, steered+baseline generations per task, `merge-steered` per setup, Qwen judge (64-shard A/B protocol + repair), `aggregate`, full-pool geometry + eval-disjoint matched audit |
| `toxicity_full.sh` | Dataset B: rebuild + verify the annotated split from `TOXICITY_DATA_DIR` (texts are NOT shipped; see `datasets/toxicity/DATASET.md`), then vectors/steer/judge per setup, geometry + matched audit, `aggregate` |
| `geometry_and_correlations.sh` | Geometry (unless `GEOMETRY_ALREADY_DONE=1`), `repsteer aggregate`, and full + partial geometry-vs-outcome correlations (`repsteer correlations`) for either or both datasets |
| `run_all.sh` | Chains the three scripts (correlations reuse the geometry computed by the dataset scripts) |

## Common knob set

| env var | default | stage |
|---|---|---|
| `ROOT` | repo root (auto-detected) | all |
| `DATASET` | `datasets/catdog/corpus.csv` | catdog stages |
| `OUT` | `results/replication/<dataset>` | all outputs |
| `TOXICITY_DATA_DIR` | required (toxicity only) | split rebuild |
| `XLMR_SCORES` | unset | toxicity aggregate (`text_hash,toxicity_probability` CSV for `classifier_target_delta`) |
| `GEOMETRY_ALREADY_DONE` | `0` | geometry_and_correlations.sh |

Models, layers, alphas and every decoding/judge switch come from
`configs/replication/*.json`, which mirror the production PBS runs (layers
0,3,…,57; alphas 8/16/32/64/128 magnitudes applied signed toward the opposite
pole; 32 sampled contrast texts per class; batch_size 4; max_length 512;
max_new_tokens 35; greedy decode with production whitespace normalization and
empty-output retries; chat template ON; bfloat16; unit-normalized `mean`
vectors; hook on all positions). Each config's `provenance` field cites the
exact production script + line numbers.

## Outputs

- `steered_<task>_<setup>.csv` / `steered_toxicity_<setup>.csv`: rows ordered
  layer → alpha → evaluation row, with `task`, `setup`, `target_attribute`,
  `signed_coefficient`, `prompt`, `baseline_continuation`, `steered_continuation`.
- `steered_<setup>.csv` (catdog): the three task tables merged in production
  judged-input order (layer, then style/language/concept).
- `judged_*.csv`: production judge columns (`option_a_is`, `string_preferred`,
  `steering_preferred`, `tie`, `judge_broken_*`, `valid_decision`, `raw_judgement`).
- `outcomes/`: `confirmatory_layer_summary.csv` (catdog) or
  `toxicity_summary_by_layer_alpha.csv`, `toxicity_best_alpha.csv`,
  `toxicity_best_alpha_geometry.csv` (toxicity), in the production layouts.

## Validate-only mode (CPU-only)

`bash scripts/replication/run_all.sh --validate-only` runs, without touching a
GPU:

1. `repsteer validate` on every replication config against its dataset;
2. `repsteer split` — the deterministic 32/25-per-cell steering splits; these
   reproduce production `results-gemma4-steering-eval200-split` row for row
   for the catdog corpus and `datasets/toxicity/manifest.json` (via
   `verify_split.py`) for toxicity.

GPU-bearing stages (build-vectors/steer/judge/geometry) are skipped.

## Reproduction notes

- Vector setups mirror production: `general` (style/concept within language,
  language unconditioned; toxicity within language) vs `conditioned` (within
  the other two attributes; toxicity within language × gender). See
  `configs/replication/README.md`.
- Judge A/B placement reproduces production `judge_mode`: catdog setups were
  judged as one merged table split round-robin into 64 jobs; toxicity judged each
  (setup, layer) file as one job. Invalid catdog judgements get the production
  repair pass.
- Geometry: Decodanda engine, `hs_mean` pooling, `zscore=true`, ndata 26, SD
  ndata 8 / training fraction 0.75 / 25 cross validations, SD threshold 0.60,
  CCGP resamplings 25 (`geometry_rows.csv`); the matched audit writes
  `matched_metrics.csv`.
- Layer resolution: `google/gemma*` in `configs/model_registry.yaml` pins
  `layer_path=model.language_model.layers` (production n=60).

## Parallelizing geometry

`repsteer geometry` computes layers sequentially. Production ran each layer as
an independent scheduler job; on a multi-GPU box shard by layer with
`repsteer geometry --layers L` and concatenate the per-layer CSVs. See the
header comment in `geometry_and_correlations.sh`.
