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
a constrained final answer and expose parse errors rather than silently converting
invalid responses to ties. Brokenness and target match must remain independent.

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
