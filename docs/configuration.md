# Configuration Reference

RepSteer uses one JSON file as the reproducibility record for an experiment. See
`configs/example_catdog.json` for a complete example.

## `dataset`

| Field | Meaning |
| --- | --- |
| `path` | Default CSV path. A CLI `--dataset` path overrides it. |
| `text_column` | Column containing input text. |
| `id_column` | Unique sample identifier. |
| `variables` | Mapping from variable name to `[negative, positive]` string values. |

Every declared variable must be binary. The order determines vector direction and
steering sign.

## `model`

| Field | Meaning |
| --- | --- |
| `model_id` | Hugging Face causal language model ID or local path. |
| `tokenizer_id` | Optional tokenizer override. |
| `dtype` | Torch dtype name such as `bfloat16`, `float16`, or `float32`. |
| `device_map` | Transformers device-map setting. |
| `trust_remote_code` | Whether Hugging Face may execute repository model code. |
| `use_chat_template` | Format prompts with the tokenizer's chat template. |
| `layer_path` | Optional dotted path to the decoder block collection. |

Review third-party code before enabling `trust_remote_code`.

## `target`

| Field | Meaning |
| --- | --- |
| `name` | Dataset variable to steer. |
| `negative` | Negative value, matching `dataset.variables`. |
| `positive` | Positive value, matching `dataset.variables`. |
| `condition_on` | Nuisance columns held fixed when constructing vectors. |

Examples:

```json
{"name": "style", "negative": "preschool", "positive": "shakespeare", "condition_on": []}
```

constructs one general style vector per layer. Using
`["language", "is_dog"]` constructs a separate style vector for every language and
concept condition.

## `split`

`vector_per_cell`, `eval_per_cell`, and optional `geometry_per_cell` specify rows per
full condition cell. `seed` controls deterministic selection. A clear error is raised
when a cell is underfilled.

## `generation`

`max_length` truncates the input, while `max_new_tokens` limits the continuation.
`temperature`, `top_p`, and `do_sample` control decoding. `seed` makes paired
baseline and intervention calls reproducible where the backend permits it.

## `steering`

| Field | Meaning |
| --- | --- |
| `layers` | Zero-based decoder blocks to intervene at. |
| `coefficients` | Non-negative intervention magnitudes. |
| `pooling` | `mean`, `last`, `min`, or `max` representation pooling. |
| `normalize_vector` | Unit-normalize class-mean differences before steering. |
| `positions` | `all`, `last`, or `generated`. |
| `batch_size` | Representation extraction batch size. |
| `max_length` | Vector-extraction input limit. |
| `prompt_template` | Python format string for generation prompts. |

A prompt template may use `{text}`, `{source}`, `{target}`, and any dataset column.
`all` intervenes on prompt and generated-token passes. `last` changes the final
position of every forward call. `generated` changes cached one-token decoding calls
and leaves the prompt prefill unchanged.

## `geometry`

`engine` is `decodanda` for production-compatible metrics or `native` for the
lightweight approximation. `layers`, `poolings`, `batch_size`, and `max_length`
control representation extraction.

`zscore` should normally remain true. `n_resamplings` and `ndata` control named
metric sampling. SD has separate `sd_ndata`, `sd_training_fraction`,
`sd_cross_validations`, and `sd_accuracy_threshold` settings.
`max_samples_per_condition` and `max_sd_dichotomies` cap compute; null evaluates all
available samples or balanced dichotomies.

## `judge`

The judge uses an independent `model` block, an `attribute_descriptions` mapping, an
optional custom `prompt_template`, and `max_new_tokens`. Every steerable target must
have a description. Descriptions may use `{target}`.

The default parser expects final `Broken_A`, `Broken_B`, and `Decision` fields. Keep
those fields when supplying a custom prompt.
