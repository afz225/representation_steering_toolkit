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

| Field | Meaning |
| --- | --- |
| `normalize_whitespace` | Collapse every whitespace run to one space (production `safe_text`) in the prompt before the chat template and in the decoded continuation. |
| `retry_empty` | On an empty continuation retry with `(seed + 1, do_sample)`, then `(seed, greedy)`, under the same hook registration. |
| `pad_with_eos` | Pass `pad_token_id = eos_token_id` to `generate`. |
| `baseline_seed_offset` | Baseline seed is `seed + baseline_seed_offset + row`. |
| `steered_seed_material` | When set, the steered seed is `seed + int(sha256(material)[:8], 16) % 1_000_000`, where the material is this string formatted with `{setup}`, `{model_id}`, `{task}`, `{layer}`, `{alpha}` (float) and `{id}`; otherwise `seed + row`. |

With the three switches on, `generate` is production `SearchCaaModel.generate`: one
prompt at a time, tokenized without padding, greedy unless `do_sample`.

## `steering`

| Field | Meaning |
| --- | --- |
| `layers` | Zero-based decoder blocks to intervene at. |
| `coefficients` | Non-negative intervention magnitudes. |
| `pooling` | `mean`, `hs_mean`, `last`, `min`, or `max` representation pooling (see `geometry`). |
| `normalize_vector` | Unit-normalize class-mean differences before steering. |
| `positions` | `all`, `last`, or `generated`. |
| `batch_size` | Representation extraction batch size. |
| `max_length` | Vector-extraction input limit. |
| `prompt_template` | Python format string for generation prompts. |
| `prompt_template_column`, `prompt_templates` | Per-row templates: `prompt_templates[row[prompt_template_column]]` replaces `prompt_template`. |
| `task`, `setup` | Labels written to the `task`/`setup` output columns (defaults: the target name; `general` without conditioning, else `conditioned`). The judge renders its target line from `task`. |
| `target_labels` | Maps target values to the `target_attribute` column (e.g. `{"0": "cat", "1": "dog"}`). |
| `sampling` | Optional contrast sampling: `per_class`, `seed`, `positive_seed_offset`, `negative_seed_offset`, `replace_if_short`. Each class of each condition group is drawn per layer with `DataFrame.sample(per_class, random_state=seed + layer + offset)` in file order; without it every row is used. |
| `relative_depths` | Optional, instead of `layers`: depths in (0, 1] resolved to `round(depth × n_layers)` (clipped, deduplicated). |
| `alpha_mode` | `absolute` (default; `coefficients` used as-is) or `relative_norm` (each coefficient is an `alpha_rel` multiplier of the layer's median residual norm stored in the vectors archive as `median_norm`). |
| `norm_calibration_texts`, `norm_calibration_dataset` | `relative_norm` calibration: number of texts (default 128) drawn deterministically from the vector dataset, or from `norm_calibration_dataset` when set (keeps coefficients identical across runs that differ only in their vector texts). |
| `batched_grid` | `true`: one batched generation per eval prompt covering the baseline and every layer × coefficient row (default `false`, the production per-row path). |

A prompt template may use `{text}`, `{safe_text}` (whitespace-collapsed text),
`{source}`, `{target}`, and any dataset column. Vectors are `mean(positive) −
mean(negative)` with positive texts extracted before negative ones; the hook adds
`sign × coefficient × v` (sign +1 toward the positive class) after casting `v` to the
hidden dtype. `all` intervenes on prompt and generated-token passes. `last` changes
the final position of every forward call. `generated` skips the first multi-token
(prefill) pass of each hook registration and changes every later pass.

`repsteer steer` writes rows ordered layer → coefficient → evaluation row after
computing one baseline per row. `repsteer merge-steered --input A --input B ...`
concatenates per-task steer tables ordered by (layer, input order), the judged-input
order production used.

## `geometry`

`engine` is `decodanda` for production-exact metrics or `native` for the
lightweight approximation. `layers`, `poolings`, `batch_size`, and `max_length`
control representation extraction: each text is whitespace-normalized, tokenized
without a chat template using the tokenizer's own default padding side, and the
output of the resolved decoder block (`model.layer_path`/`layer_regex` or the
registry) is captured with a forward hook. Poolings: `hs_mean` is the attention-mask
mean accumulated in the model dtype then cast to float32 (production geometry);
`mean` casts to float32 first and then averages (production steering vectors);
`last` takes position `attention_mask.sum - 1`; `min`/`max` reduce over real tokens.

Decodanda engine fields:

| Field | Meaning |
| --- | --- |
| `protocol` | `production` (SD over all unique dichotomies, then PS and CCGP per target; writes `geometry_rows.csv`, `dichotomy_rows.csv`, `counts.csv`) or `matched_audit` (named decoding and CCGP on the same data budget; writes `matched_metrics.csv`). `repsteer geometry --protocol` overrides it. |
| `variables` | 0/1 condition columns (default: `dataset.variables`; string columns are encoded by the declared `[negative, positive]` order). |
| `target_fields` | PS/CCGP targets in run order (default: `variables`). |
| `n_per_cell` | Examples kept per condition cell after a seeded `random.Random(seed)` shuffle of each cell in sorted cell order. |
| `auto_reduce_n_per_cell` | Production protocol only: shrink `n_per_cell` to the smallest cell instead of failing. |
| `audit_n_per_cell` | Matched-audit cell size (default `n_per_cell`); cells must be exactly this size. |
| `audit_training_fraction`, `audit_cross_validations` | Matched-audit named decoding settings. |
| `relative_depths` | Optional, instead of `layers`: depths in (0, 1] resolved against the model's layer count exactly like `steering.relative_depths`; geometry rows then carry the same `relative_depth` (`round(layer / n_layers, 2)`) as the steering rows. `repsteer geometry --layers` overrides it. |

`zscore` should normally remain true. `n_resamplings` (CCGP resamplings) and
`ndata` control named metric sampling. SD has separate `sd_ndata`,
`sd_training_fraction`, `sd_cross_validations`, and `sd_accuracy_threshold`
settings. `max_samples_per_condition` and `max_sd_dichotomies` cap compute for the
native engine; null evaluates all available samples or balanced dichotomies.

`repsteer eval-disjoint --config C --output F` writes the dataset minus the rows of
its steering-evaluation split (original row order), the input of the
eval-disjoint matched audit.

## `judge`

The judge uses an independent `model` block and an `attribute_descriptions` mapping
from each row's `task` to its target line: either a string formatted with
`{target}` or a mapping from `target_attribute` to the full line. The default prompt
is the production pairwise prompt with `{target_line}`, `{a}` and `{b}` (A/B
whitespace-collapsed); a custom `prompt_template` may also use `{prompt}`.

| Field | Meaning |
| --- | --- |
| `max_new_tokens`, `max_length` | Judge decoding budget (1024) and input truncation (2048). |
| `seed` | Judge seed base; row seed is `seed + k` (shard position). |
| `num_shards`, `shard_group_by` | A/B assignment: within each `shard_group_by` group, row i goes to shard `i % num_shards` at position `k = i // num_shards`; the steered text is option A iff `(k + shard)` is odd. |
| `repair_invalid`, `repair_max_new_tokens`, `repair_seed` | Re-judge rows without a valid final decision once with the strict repair suffix (seed `repair_seed + ordinal`). |

The parser takes the final `Decision: A|B|T` at the very end of the response and
the last `Broken_A`/`Broken_B` labels; responses without a valid decision become
`T` with `valid_decision = false`. Keep those fields when supplying a custom
prompt.

`repsteer aggregate --dataset catdog|toxicity --judged J [--judged J2]` writes the
production outcome tables (see `docs/evaluation.md`).
