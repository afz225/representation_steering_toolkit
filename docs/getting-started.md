# Getting Started

This guide runs a complete vector, steering, judging, and geometry experiment from a
CSV dataset. RepSteer executes as an ordinary local process and does not manage
cluster jobs or model downloads.

## 1. Install

For the full Hugging Face and Decodanda workflow:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[hf,geometry,test]'
```

For development with the native geometry backend and fake-model tests:

```bash
python -m pip install -e '.[test]'
pytest -q
```

## 2. Prepare a dataset

The input is a CSV with:

- one unique ID column;
- one text column;
- one column for every binary experimental variable;
- no null values in required columns.

Example:

```csv
id,text,language,style,is_dog
1,The cat waited by the door.,en,preschool,0
2,Hark, the hound approaches.,en,shakespeare,1
3,جلس القط قرب الباب.,ar,preschool,0
4,ها قد أقبل الكلب.,ar,shakespeare,1
```

All Cartesian-product cells must contain enough rows for the requested balanced
splits. Dataset values are read as strings, so the config should use `"0"` and `"1"`
for numeric-looking binary labels.

## 3. Create a config

Copy `configs/example_catdog.json` and update:

- dataset path and column names;
- binary variable values;
- target variable and optional conditioning variables;
- target and judge model IDs;
- layer and coefficient sweeps;
- split sizes and geometry compute limits.

Validate before loading a model:

```bash
repsteer validate --config config.json
```

## 4. Create disjoint splits

```bash
repsteer split --config config.json --output-dir results/splits
```

This writes `vector.csv`, `evaluation.csv`, and `geometry.csv`. IDs are disjoint
across splits and each split is balanced over the full condition grid.

## 5. Build vectors

```bash
repsteer build-vectors --config config.json \
  --dataset results/splits/vector.csv \
  --output results/vectors.npz
```

The archive stores one positive-minus-negative vector for each configured layer and,
when `condition_on` is nonempty, each nuisance condition.

## 6. Generate and judge continuations

```bash
repsteer steer --config config.json \
  --dataset results/splits/evaluation.csv \
  --vectors results/vectors.npz \
  --output results/generations.csv

repsteer judge --config config.json \
  --input results/generations.csv \
  --output results/judged.csv
```

Steering always targets the opposite class of each held-out source example. The
judge writes row-level labels and `judged.summary.csv` with preference, tie,
broken-output, and valid-decision rates.

## 7. Compute geometry

```bash
repsteer geometry --config config.json \
  --dataset results/splits/geometry.csv \
  --output-dir results/geometry
```

With the Decodanda engine (default `protocol: production`) the output directory
holds `geometry_rows.csv` (one row per layer, pooling, and target: `PS`,
`CCGP_nontargeted`, `SD`, `SD_mean_acc` plus per-variable statistics, the
production column layout), `dichotomy_rows.csv` (per-dichotomy SD accuracies), and
`counts.csv`. `--protocol matched_audit` writes `matched_metrics.csv`
(`decodability_mean`, `ccgp_mean`, ... per layer and target). The native engine
writes a long-form `geometry_metrics.csv` (layer, pooling, metric, variable or
dichotomy identity, resampling, value).

## Common checks

- Use held-out vector and evaluation rows; never evaluate on vector-construction text.
- Use `condition_on: []` for general vectors and explicit nuisance columns for
  condition-specific vectors.
- Confirm the model's decoder blocks are located correctly before a large sweep.
- Keep `geometry.zscore: true` when comparing with the documented Decodanda setup.
- Inspect generated text and parse-error rates before interpreting aggregate success.
