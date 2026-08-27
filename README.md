# RepSteer

RepSteer is a small, standalone toolkit for controlled activation-steering and
representation-geometry experiments. It was extracted conceptually from the cat/dog,
style, language, and toxicity investigations, but it does not import or modify the
parent repository.

The package deliberately contains no PBS, Slurm, queue, cluster, or job-submission
logic. Every command is an ordinary local process; execution policy belongs outside
the research code.

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

## Collateral and off-target evaluation

The toolkit also ports the established collateral analyses:

- token-weighted causal perplexity under intervention;
- MMLU/MMMLU-style next-token multiple-choice accuracy;
- M-IFEval-compatible paired instruction response generation and scorer integration;
- immediate and downstream off-target concept-axis projection drift;
- paired LLM labels for off-target concept flips and broken-output rate.

Example commands:

```bash
repsteer perplexity --config style-general.json --benchmark text.csv \
  --vectors vectors.npz --output-dir results/perplexity
repsteer multiple-choice --config style-general.json --benchmark mmlu.csv \
  --vectors vectors.npz --output-dir results/mmlu
repsteer instruction-generate --config style-general.json --benchmark mifeval.csv \
  --vectors vectors.npz --output results/mifeval_responses.csv
repsteer offtarget-project --config style.json --dataset probe.csv \
  --vectors style_vectors.npz --intervention-layer 12 --coefficient 8 \
  --capture-layers 12,15,18,21,24 \
  --axis-config language=language.json --axis-vectors language=language_vectors.npz \
  --axis-config is_dog=is_dog.json --axis-vectors is_dog=is_dog_vectors.npz \
  --output-dir results/projection
repsteer offtarget-judge --config style.json --input generations.csv \
  --labels configs/example_offtarget_labels.json --output-dir results/offtarget_judge
```

General capability benchmarks require general vectors (`condition_on: []`). Online
projection drift supports conditioned steering and conditioned readout axes because
the probe dataset supplies the necessary condition columns.

## Configuration and metric semantics

Datasets are CSV files with a unique ID, a text column, and one column per binary
variable. The ordered pair in `dataset.variables` defines the negative and positive
classes. `target.condition_on` is empty for a general vector or lists nuisance
variables for condition-specific vectors. Layer indices refer to decoder blocks,
with layer zero denoting the output of the first block.

Representations support attention-mask-aware `mean`, `last`, `min`, and `max`
pooling. Decodability measures held-out classification of a named variable. CCGP
trains on nuisance contexts and tests generalization to a held-out context. PS
measures alignment among context-specific class-difference directions. SD evaluates
all balanced dichotomies of the complete condition space and is therefore a
space-level metric rather than a metric for one named variable.

The generated-string judge randomizes baseline/steered A/B ordering and parses the
final `Decision: A|B|T` line. It labels brokenness separately. A fluent continuation
that misses the target is not broken; broken output means unusable prose due to
incoherence, semantic nonsense, severe grammar failure, empty text, pathological
repetition, truncation, boilerplate, or formatting artifacts.

Perplexity is token weighted: `exp(total_nll / total_scored_tokens)`. Multiple-choice
accuracy uses next-token answer-letter probabilities. Online projection drift uses a
separately constructed concept axis at each downstream readout layer and reports both
absolute drift and drift normalized by that layer's unnormalized class-mean gap.

## Development and verification

```bash
python -m pip install -e '.[test]'
pytest -q
```

Optional tests skip when Decodanda, PyTorch, or the original production artifacts are
not available. `scripts/verify_prior_subset.py` replays postprocessing against those
artifacts when they are present. The checked-in `parity_artifacts/parity_report.csv`
records a deterministic subset audit whose maximum discrepancy was below `1e-12`.

To support a new model family, implement the `ModelBackend` protocol or configure
`model.layer_path` for a Hugging Face decoder layout. New vector estimators should
return `SteeringVector` records and state their construction method explicitly.

## License

RepSteer is available under the [MIT License](LICENSE).
