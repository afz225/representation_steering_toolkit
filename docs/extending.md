# Extending RepSteer

## Add a dataset

Create a CSV with a unique ID, text, and binary variable columns, then declare those
columns in a config. No Python changes are required. For a multiclass attribute,
define each planned binary contrast explicitly rather than relying on an ambiguous
sign convention.

## Add a model backend

Implement `repsteer.models.base.ModelBackend`:

1. expose `n_layers`;
2. return `samples x hidden_size` sentence representations;
3. generate text with an optional `Intervention`;
4. return per-text causal NLL sums and token counts;
5. score next-token choice probabilities;
6. generate while capturing selected layer representations.

The vector, steering, geometry, and evaluation modules depend on this protocol rather
than directly on Transformers. For a Hugging Face architecture with an unfamiliar
module tree, first configure `model.layer_path`, such as `transformer.blocks`.

## Add a model via YAML

Instead of hand-writing `model.layer_path` in every experiment config, register the
architecture once in `configs/model_registry.yaml`. The registry maps a Hugging Face
model id (exact string) or an fnmatch pattern (`*`, `?`, `[`) to a profile:

```yaml
profiles:
  "google/gemma*":
    layer_path: model.language_model.layers
    use_chat_template: true
    dtype: bfloat16
```

Available profile fields (all optional): `layer_path`, `layer_regex`,
`tokenizer_id`, `dtype`, `use_chat_template`, `trust_remote_code`.

Resolution order:

1. exact `model_id` match;
2. first pattern match in file order (so put narrower patterns above wider ones);
3. the built-in attribute walk in `HuggingFaceBackend._LAYER_PATHS`.

Profiles are applied *before* the model is loaded, and only fill values the
experiment config leaves unset: explicit config fields always win. Two ways to
name steerable modules:

- `layer_path` — an attribute path walked from the loaded model object, e.g.
  `model.language_model.layers` for Gemma-4 or `transformer.h` for GPT-2;
- `layer_regex` — a regex searched against `model.named_modules()` names, e.g.
  `model\.layers\.\d+\.mlp$` to steer every MLP block. Matches are collected in
  deterministic natural order (`layers.2` before `layers.10`); zero matches is a
  hard error.

To point an experiment at a different registry file, set the optional
`model.model_registry` path field; when omitted, `HuggingFaceBackend` uses
`configs/model_registry.yaml` if it exists and otherwise falls back to the
built-in attribute walk. Registry errors (unknown profile fields, malformed YAML,
zero regex matches) raise `ValueError` with the offending key or pattern.

Add a new architecture by appending one profile; no Python changes are needed.
Keep the file ordered from most specific to most general pattern, and cite the
production script a profile mirrors in a YAML comment so future readers can
trace where the naming came from.

## Add a vector estimator

Return a list of `SteeringVector` records containing layer, variable, condition key,
vector values, class counts, and the unnormalized class gap. Existing consumers do
not require difference-of-means construction, so paired-difference, probe-normal,
covariance-scaled, and CAA estimators can be added without changing intervention
code. Name and document the estimator precisely.

## Add a geometry backend

Return a long-form DataFrame compatible with the existing columns: `metric`,
`variable`, `dichotomy_id`, `resampling`, and `value` where applicable. Do not reuse
an existing engine name when classifier, standardization, resampling, or dichotomy
semantics differ.

## Add a judge or collateral metric

Keep row-level outputs separate from aggregation. Generated-string judges should use
a constrained final answer and flag invalid responses (`valid_decision`) so they can
be repaired or reported. Brokenness and target match must remain independent.

For new collateral metrics, return per-example records first and add an explicit
summary function. Preserve setup, layer, coefficient, vector condition, sample count,
and missing-value semantics in every aggregate.

## Testing expectations

Run:

```bash
pytest -q
```

Add focused unit tests for formulas and grouping behavior. If replacing an existing
experimental script, add a deterministic parity test against trusted saved outputs.
Optional model and production-artifact tests should skip clearly when their external
resources are absent, so a clean GitHub clone remains testable.
