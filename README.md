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

## Documentation

- [Getting started](docs/getting-started.md)
- [Configuration reference](docs/configuration.md)
- [Methodology and metric definitions](docs/methodology.md)
- [Evaluation and collateral effects](docs/evaluation.md)
- [Extending RepSteer](docs/extending.md)
