# RepSteer

RepSteer is a small, standalone toolkit for controlled activation-steering and
representation-geometry experiments.

## What it does

- validates arbitrary CSV datasets with any number of binary variables;
- creates deterministic, cell-balanced, disjoint vector/evaluation/geometry splits;
- extracts mean, final-token, minimum, or maximum pooled hidden representations;
- constructs general or condition-specific positive-minus-negative mean vectors;
- applies opposite-target interventions at chosen layers, coefficients, and positions;
- generates paired baseline/steered continuations;
- evaluates pairs with an independent generated-string LLM judge;
- computes named decodability, CCGP, parallelism score, and space-level SD with the production-compatible Decodanda engine or a lightweight native engine.

## Install

```bash
cd representation_steering_toolkit
python -m pip install -e '.[hf,geometry,test]'
```

The native geometry engine requires NumPy, pandas, and scikit-learn. Exact production geometry uses the optional Decodanda dependency. The Hugging Face adapter additionally requires PyTorch, Transformers, and Accelerate.

## Workflow

Edit a copy of `configs/example_catdog.json`, then run:

```bash
repsteer validate --config config.json
repsteer split --config config.json --output-dir results/splits
repsteer build-vectors --config config.json \
  --dataset results/splits/vector.csv --output results/vectors.npz
repsteer steer --config config.json --dataset results/splits/evaluation.csv \
  --vectors results/vectors.npz --output results/generations.csv
repsteer judge --config config.json --input results/generations.csv \
  --output results/judged.csv
repsteer geometry --config config.json --dataset results/splits/geometry.csv \
  --output-dir results/geometry
```

Model loading occurs only in `build-vectors`, `steer`, `judge`, and `geometry`.
The validation and split commands are CPU-only.

## Experimental guarantees

The vector and evaluation splits are disjoint. For a configured target, the stored
direction is always

```text
v(layer, condition) = mean(h_positive) - mean(h_negative)
```

and opposite-target steering applies `+alpha * v` for the positive class and
`-alpha * v` for the negative class. The vector may be general
(`condition_on: []`) or specific to nuisance-variable values.

This is difference-of-means activation steering. It is **not** canonical CAA unless
the input texts were themselves constructed with CAA's contrastive instruction
template. The toolkit records the method precisely instead of using those names
interchangeably.

The default Decodanda geometry backend reproduces the project procedure: Pearson PS, one-out CCGP, all balanced dichotomies for SD, and separate SD classifier settings. Z-scoring is enabled by default and follows Decodanda: train-split statistics for decoding and pair-local statistics for PS.

## Reproducing the paper

`configs/replication/` holds one JSON config per dataset × task × vector setup
(`catdog_{style,concept,language}_{general,conditioned}.json`,
`toxicity_{general,conditioned}.json`), with every hyperparameter the paper used:
split sizes, steering layers and coefficients, contrast-sample counts, prompts,
generation settings, Decodanda geometry settings, the matched decodability/CCGP
audit budget, and the judge. Each value is traced to the production script that
produced it in [configs/replication/README.md](configs/replication/README.md).

```bash
# CPU-only checks of data, splits, and configs (no model downloads)
bash scripts/replication/catdog_full.sh --validate-only
TOXICITY_DATA_DIR=/path/to/labeled bash scripts/replication/toxicity_full.sh --validate-only

# Full replication (GPU; runs every stage through the repsteer CLI)
TOXICITY_DATA_DIR=/path/to/labeled bash scripts/replication/run_all.sh
```

The scripts are plain bash with no scheduler directives; each stage is
independent and per-layer geometry can be parallelized (see
[scripts/replication/README.md](scripts/replication/README.md)).

## Datasets

- `datasets/catdog/corpus.csv`: the synthetic style × concept × language corpus
  (1,041 sentences, MIT). Generation details: [datasets/catdog/DATASET.md](datasets/catdog/DATASET.md).
- Toxicity: the texts come from `textdetox/multilingual_toxicity_dataset` (en, ar),
  whose upstream sources do not permit redistribution, so no texts are shipped.
  `datasets/toxicity/` contains the re-annotation prompt, a split builder, and a
  fingerprint manifest that checks a locally rebuilt split matches the paper's
  split exactly. See [datasets/toxicity/DATASET.md](datasets/toxicity/DATASET.md).

## Adding a model

Models are resolved through `configs/model_registry.yaml`. Each entry maps a model
id or a glob pattern to the module to steer and read from, given either as an
attribute path (`layer_path: model.layers`) or as a regex over
`named_modules()` (`layer_regex: 'model\.layers\.\d+\.mlp$'`), plus tokenizer,
dtype, and chat-template defaults. New architectures need a registry entry, not
code changes. See [Extending RepSteer](docs/extending.md#add-a-model-via-yaml).

## Documentation

- [Getting started](docs/getting-started.md)
- [Configuration reference](docs/configuration.md)
- [Methodology and metric definitions](docs/methodology.md)
- [Evaluation and collateral effects](docs/evaluation.md)
- [Extending RepSteer](docs/extending.md)
