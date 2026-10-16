# Replication configs

These eight JSON files (one per dataset task × vector setup) mirror the paper's
production experiment hyperparameters exactly. They load through
`repsteer.config.load_config()`; `tests/test_steering_parity.py` checks the
steering, judge and aggregation semantics they select against the production code
and artifacts. Each file carries a top-level `"provenance"` object citing the
production scripts and line numbers that produced the run; the loader ignores
unknown top-level keys, and the field is a document, not an input.

| file | task | vector setup (`target.condition_on`) | sampling seed offsets (+, −) |
| --- | --- | --- | --- |
| `catdog_style_general.json` | style | within [language] | 101, 102 |
| `catdog_style_conditioned.json` | style | within [language, is_dog] | 101, 102 |
| `catdog_language_general.json` | language | unconditioned | 201, 202 |
| `catdog_language_conditioned.json` | language | within [style, is_dog] | 201, 202 |
| `catdog_concept_general.json` | concept (`is_dog`) | within [language] | 301, 302 |
| `catdog_concept_conditioned.json` | concept (`is_dog`) | within [language, style] | 301, 302 |
| `toxicity_general.json` | toxicity (`toxicity_bin`) | within [language] | 101, 102 |
| `toxicity_conditioned.json` | toxicity (`toxicity_bin`) | within [language, gender_bin] | 101, 102 |

Steering prompts dispatch per row on `language` (`steering.prompt_template_column`
/ `prompt_templates`), exactly as production branches EN/AR; generation
collapses whitespace in the whole prompt before the chat template
(`generation.normalize_whitespace`), as `SearchCaaModel.generate` did.

## Dataset A — style/cat-dog (fully synthetic, ours)

Files: `catdog_{style,language,concept}_{general,conditioned}.json`. All six
share one dataset file, one split, one geometry/judge setting; only the steering
target, its conditioning keys, sampling offsets and `steering.task`/`setup` labels
differ. Production generated all three tasks per (setup, layer) in one run of
`gemma4_eval200_steering.py`; the toolkit steers one task per config and
`repsteer merge-steered` restores production's judged-input order.

| hyperparameter | value | production source |
| --- | --- | --- |
| dataset | `datasets/catdog/corpus.csv` (1041 rows, identical content and column order to `synthetic_catdog_full/geometry_dataset.csv`) | `pbs_gemma4_geometry_hsmean_stride3_zscore.sh:39`; `prepare_steering_eval200_split.py:17` |
| variables (order) | language [ar→en? see note], style [preschool, shakespeare], is_dog [0, 1] | `prepare_steering_eval200_split.py:11` (`CELLS = language, style, concept`); `caa_model_search.py:236-244` (vector polarity: style shakespeare−preschool, language en−ar, concept dog−cat) |
| target (per config) | style: negative preschool / positive shakespeare; concept (`is_dog`): negative 0 / positive 1 (`target_labels` 0→cat, 1→dog); language: negative ar / positive en. `condition_on`: general = [language] for style/concept and [] for language; conditioned = style [language, is_dog], language [style, is_dog], concept [language, style] | `gemma4_eval200_steering.py:108-167` (general vectors), `caa_conditioned_steering.py:91-117` (conditioned vectors) |
| model | google/gemma-4-31B-it, bfloat16, device_map auto, trust_remote_code true, chat template on | `pbs_gemma4_eval200_generate.sh:39-40`; `gemma4_eval200_steering.py:178,187,226-227`; layer resolution `model.language_model.layers` n=60 via the attribute walk (caa_model_search.py:33-56; mirrored in `configs/model_registry.yaml` and `src/repsteer/models/huggingface.py`) |
| generation | max_length 512, max_new_tokens 35, do_sample false, temperature 0.7, top_p 0.9, seed 42; `normalize_whitespace` (safe_text on the prompt before the chat template and on the decoded text), `retry_empty` (ladder (seed, sample), (seed+1, sample), (seed, greedy)), `pad_with_eos`; baseline seed 42+1000+row (`baseline_seed_offset 1000`); steered seed from `steered_seed_material` `eval200\|{setup}\|{model_id}\|{task}\|{layer}\|{alpha}\|{id}` | `caa_model_search.py:185-216` (format_generation_prompt/generate); `gemma4_eval200_steering.py:76-84,269-283`; temperature/top_p are inert under greedy decoding |
| steering layers | 0,3,6,…,57 (20 layers, stride 3) | `pbs_gemma4_eval200_generate.sh:23` (`LAYERS=(0 3 … 57)`) |
| coefficients (magnitudes) | 8, 16, 32, 64, 128 — applied signed toward the opposite pole at runtime (`direction_sign(task, target) * alpha`) | `pbs_gemma4_eval200_generate.sh:42`; `caa_model_search.py:236-244`; `gemma4_eval200_steering.py:267` |
| contrast sampling | `steering.sampling`: per condition group and layer, 32 texts per class drawn from the vector split in file order with `DataFrame.sample(32, random_state=42 + layer + offset)`; conditioned configs set `replace_if_short` (caa_conditioned_steering falls back to replacement; never triggered with 32/cell) | `gemma4_eval200_steering.py:23-29,108-167`; `caa_conditioned_steering.py:26-34` |
| vector construction | positive texts extracted in batches of 4, then negative texts; `mean` pooling = float32 masked mean of the decoder-block output; `vec = pos.mean(0) − neg.mean(0)` divided by `float(np.linalg.norm(vec))` | `caa_model_search.py:106-159` (`sentence_reps`, `corpus_difference_vector`); `gemma4_eval200_steering.py:226` (`normalize_vectors = True`) |
| hook positions | all (prompt + generated passes) | `pbs_gemma4_eval200_generate.sh:44` (`--hook_position all`); `gemma4_eval200_steering.py:186,268` |
| batch_size | 4 | `pbs_gemma4_eval200_generate.sh:43` |
| steering max_length | 512 | `gemma4_eval200_steering.py:184` |
| prompt_templates (EN) | `Continue the story in exactly one sentence. Keep the same subject matter. Write only the continuation sentence, with no explanation or notes:\n{safe_text}\nContinuation:` (`{safe_text}` = whitespace-collapsed text) | `caa_investigation.py:164,170` (style_generation_prompt / language_generation_prompt EN branch; routing `caa_conditioned_steering.py:50-62`) |
| prompt_templates (AR) | `أكمل القصة بجملة واحدة فقط. حافظ على نفس الموضوع. اكتب جملة المتابعة فقط دون شرح أو ملاحظات:\n{safe_text}\nمتابعة:` | `caa_investigation.py:163,169` |
| split | vector_per_cell 32, eval_per_cell 25, seed 42 | `pbs_gemma4_eval200_generate.sh` consumes `results-gemma4-steering-eval200-split/`; built by `prepare_steering_eval200_split.py:18-20` (`--vector_per_cell 32 --eval_per_cell 25 --seed 42`); recipe: group by (language, style, concept), `sample(frac=1, random_state=seed+cell_index)`, take first 32 as vector, next 25 as evaluation. Verified byte-deterministic against production (`evaluation_200.csv` / `vector_construction_256.csv`) this session |
| geometry engine | decodanda | `pbs_gemma4_geometry_hsmean_stride3_zscore.sh:49` (`PYTHONPATH=vendor_decodanda`, `geometry_olmo_reduced_full.py`) |
| geometry layers | 0,3,6,…,57 | `pbs_gemma4_geometry_hsmean_stride3_zscore.sh:27` |
| geometry pooling | `hs_mean` (masked mean accumulated in the model dtype, then float32; forward hook on the resolved decoder block, whitespace-normalized text, tokenizer default padding, no chat template) | `pbs_gemma4_geometry_hsmean_stride3_zscore.sh:46`; `geometry_olmo_reduced_full.py:35,257-309` |
| geometry protocol / variables | `protocol: production`, `variables` = `target_fields` = `[language_bin, style_bin, is_dog]` (the corpus's 0/1 columns, target order as passed) | `pbs_gemma4_geometry_hsmean_stride3_zscore.sh` (`--target_fields/--condition_fields language_bin style_bin is_dog`); `geometry_olmo_reduced_full.py:564-612` |
| zscore | true (feature-wise z-scoring of representations) | `pbs_gemma4_geometry_hsmean_stride3_zscore.sh` (`--zscore`) |
| n_resamplings (CCGP) | 25 | `pbs_gemma4_geometry_hsmean_stride3_zscore.sh` (`--ccgp_resamplings 25`); `geometry_olmo_reduced_full.py:544` |
| PS | nshuffles 0, pearson, max_semantic_dist 1 | `pbs_gemma4_geometry_hsmean_stride3_zscore.sh` (`--ps_nshuffles 0`); `geometry_olmo_reduced_full.py:543`; toolkit `geometry_decodanda.py::_run_ps_ccgp` |
| ndata (Decodanda, CCGP) | 26 | `pbs_gemma4_geometry_hsmean_stride3_zscore.sh` (`--decodanda_ndata 26`); `geometry_olmo_reduced_full.py:546` |
| SD decode_dichotomy | training_fraction 0.75, cross_validations 25, ndata 8, acc threshold 0.60, all balanced dichotomies | `pbs_gemma4_geometry_hsmean_stride3_zscore.sh` (`--sd_cross_validations 25 --sd_ndata 8`); `geometry_olmo_reduced_full.py:549-552`; `sd_acc_threshold 0.60` at `geometry_olmo_reduced_full.py:551` |
| geometry seed | 42; RNGs reseeded at each layer (production ran one layer per process after `set_all_seeds`); SD first with per-dichotomy seed `42 + layer×100 + pooling_index + i`, then PS/CCGP per target continuing the stream | `geometry_olmo_reduced_full.py:30,376-458,564,603`; `src/repsteer/geometry_decodanda.py::production_layer_geometry` |
| geometry batch_size / max_length | 4 / 512 | `pbs_gemma4_geometry_hsmean_stride3_zscore.sh` (`--batch_size 4 --max_length 512`) |
| geometry dataset | the FULL 1041-row pool with `n_per_cell: 50`, `auto_reduce_n_per_cell: true` (min cell 70 ≥ 50, so no reduction occurs); selection = sorted cells, one shared `random.Random(42)` shuffling each cell, first 50 | `pbs_gemma4_geometry_hsmean_stride3_zscore.sh:39` (`--dataset_csv synthetic_catdog_full/geometry_dataset.csv --n_per_cell 50 --auto_reduce_n_per_cell`); `geometry_olmo_reduced_full.py:170-236` |
| matched audit | `--protocol matched_audit` on the eval-disjoint pool (`repsteer eval-disjoint`: corpus minus the 200 evaluation ids, 841 rows, byte-identical to production `geometry_dataset_excluding_steering_eval.csv`), `audit_n_per_cell: 40`, svc, zscore, decoding 25 CV × 0.75, CCGP 25 resamplings, ndata 26, NumPy seed `42 + layer×100 + k` before each metric | `pbs_gemma4_matched_decodability_ccgp_audit.sh` (`N_PER_CELL=40`, `DATASET_CSV=…eval-disjoint/…csv`); `audit_matched_decodability_ccgp.py` |
| judge model | Qwen/Qwen3.6-27B, bfloat16, trust_remote_code, chat template (`enable_thinking=False` with fallbacks), greedy, max_new_tokens 1024, max_length 2048, pad=eos, output whitespace-collapsed | `pbs_gemma4_eval200_judge.sh` (`--judge_model_id Qwen/Qwen3.6-27B --judge_trust_remote_code`); `caa_model_search.py:437-475` (`StringJudge`) |
| judge prompt | `DEFAULT_JUDGE_PROMPT` in `src/repsteer/judge.py` = `pairwise_string_prompt` verbatim; the target line comes from `judge.attribute_descriptions[task][target_attribute]` (`Target style: …`, `Target language: English/Arabic`, `Target animal concept: …`) | `caa_model_search.py:393-434`; `caa_investigation.py:80-109` |
| judge A/B assignment | the setup's merged table (layer → task → alpha → row) dealt round-robin to `num_shards: 64` jobs: row i → job i mod 64 at position k = i div 64, steered is option A iff (k + job) is odd, seed 42 + k | `merge_gemma4_eval200_generations.py`; `pbs_gemma4_eval200_judge.sh` (`--num_jobs 64`); `caa_model_search.py:523-534` |
| judge parsing | last `Decision: [ABT]` anchored at the end, `Broken_A/B` regexes with "a/b is broken" fallbacks, JSON/keyed/bare-letter fallbacks, otherwise `T` with `valid_decision` false | `caa_model_search.py:477-520` |
| judge repair | `repair_invalid: true`: rows without a valid decision are re-judged once with the strict suffix, 2048 new tokens, seed 910000 + ordinal (production repaired 89 rows, 0 left invalid). Production counted the ordinal over all shard files of both setups in sorted-path order, the toolkit counts within each judged table; the seed is inert under greedy decoding | `repair_invalid_judgements.py`; `pbs_gemma4_eval200_repair.sh`; `results-gemma4-steering-eval200/judge_repair_summary.md` |

Note on variable order: `dataset.variables` declares
`[negative, positive]` pairs. The pairs above follow the toolkit convention
(negative first) while matching production's vector construction exactly:
production builds `positive − negative` differences in the order
shakespeare−preschool, en−ar, dog−cat (caa_model_search.py:237 comment), and the
toolkit builds `positive − negative` with positive = shakespeare / en / 1, so
the resulting vectors agree. Steering is applied as a magnitude in the direction
of the opposite pole of the row's source value.

### Outcome aggregation (Dataset A)

`repsteer aggregate --dataset catdog` writes `confirmatory_layer_summary.csv`
with the production layout, reproducing `caa_model_search.report_mode` followed
by `analyze_gemma4_eval200_story.py:46-56` (checked to 1e-12 against the
production file):

- headline success = `steering_preferred_rate_mean_alpha` — the n-weighted mean
  of the steered-preference rate over α ∈ {8,16,32,64,128} (equal n per α, so a
  plain mean; the weighting exists for robustness to ragged groups).
- headline broken outcome = `steered_broken_rate_mean_alpha` (same mean-over-α).
- also reported, and cross-checked: `best_success` / `best_broken` at the
  best-α per (setup, task, layer), where best is chosen by
  `sort_values(["steering_preferred_rate", "steered_broken_rate", "alpha"],
  ascending=[False, True, True])` (analyze_gemma4_eval200_story.py:55).
  Both aggregations were verified to support the same qualitative conclusions
  (the audit report lists all four outcomes side by side).
- language groups are saturated (geometry metric constant 1.0) and are excluded
  from the headline style/catdog correlations (correlation_regression_report.py:17-18).

## Dataset B — toxicity (real tweets; NOT redistributable)

Files: `toxicity_general.json`, `toxicity_conditioned.json`. The dataset file these configs point at
(`datasets/toxicity/geometry_400.csv`) is **not shipped** with the repo — the
texts are third-party social-media posts (textdetox/multilingual_toxicity_dataset,
openrail++; original sources carry noncommercial/academic-style terms). Rebuild
it with `datasets/toxicity/prepare_toxicity_splits.py` from user-supplied
`{en,ar}_labeled.csv` files, then verify against
`datasets/toxicity/manifest.json` (sha256 fingerprints of every candidate).
`scripts/replication/toxicity_full.sh` automates rebuild + verify + run.
The shipped manifest reproduces the production split byte-identically (re-verified
this session: all three split CSVs reproduce exactly from the production labeled
CSVs with seed 42).

| hyperparameter | value | production source |
| --- | --- | --- |
| dataset | geometry pool `geometry_400.csv` (400 rows, 50/cell × 8) — rebuild, not ship | `pbs_gemma4_toxicity_geometry.sh:27`; `prepare_toxicity_gemma4_splits.py:16` |
| variables (order) | language [en, ar], toxicity_bin [0, 1], gender_bin [0, 1] | `prepare_toxicity_gemma4_splits.py:49` (`groupby(["language","toxicity_bin","gender_bin"], sort=True)`); language_bin en=0/ar=1 at `prepare_toxicity_gemma4_splits.py:24-25` |
| target | toxicity_bin: negative 0 / positive 1 (`target_labels` 0→non-toxic, 1→toxic); `condition_on` general [language], conditioned [language, gender_bin] | `gemma4_toxicity_steering.py:42-56,117-120` |
| model | google/gemma-4-31B-it, bfloat16, chat template on, normalize_vectors true | `gemma4_toxicity_steering.py:89,91,110-111` |
| generation | max_length 512, max_new_tokens 35, do_sample false, temperature 0.7, top_p 0.9, seed 42; `normalize_whitespace`, `retry_empty`, `pad_with_eos` as Dataset A; baseline seed 42+row (`baseline_seed_offset 0`); steered seed material `toxicity\|{setup}\|{layer}\|{alpha}\|{id}` | `gemma4_toxicity_steering.py:72,121-123`; `caa_model_search.py:185-216` |
| steering layers | 0,3,6,…,57 | `pbs_gemma4_toxicity_generate.sh:23` |
| coefficients (magnitudes) | 8, 16, 32, 64, 128; sign = +1 toward "toxic", −1 toward "non-toxic" | `pbs_gemma4_toxicity_generate.sh:37`; `gemma4_toxicity_steering.py:100-101,107` |
| contrast sampling | 32 per class, `DataFrame.sample(32, random_state=42 + layer + 101)` toxic / `+102` clean, from `vector_construction_256.csv` in file order, no replacement | `gemma4_toxicity_steering.py:16-22,42-56` |
| vector construction | as Dataset A (mean pooling, positive then negative extraction, unit norm) | `gemma4_toxicity_steering.py:50-55`; `caa_model_search.py:106-159` |
| hook positions | all | `gemma4_toxicity_steering.py:106` (`register_caa_hook(..., "all")`), `:133-134` |
| batch_size / steering max_length | 4 / 512 | `gemma4_toxicity_steering.py:94-95` |
| prompt_templates (EN) | `Continue the following passage in English with one coherent sentence. Output only the continuation.\nPassage: {text}` (raw text) | `gemma4_toxicity_steering.py:39` |
| prompt_templates (AR) | `تابع النص التالي بجملة عربية واحدة مترابطة. أخرج التكملة فقط.\nالنص: {text}` | `gemma4_toxicity_steering.py:38` |
| split | geometry 50 / vector 32 / eval 25 per cell × 8 cells, seed 42; slicing order within each shuffled cell is geometry, then vector, then evaluation (fixed offsets 0-49, 50-81, 82-106) | `prepare_toxicity_gemma4_splits.py:15-17,43-58`; recipe: filter `toxic∈{0,1} ∧ mentions_gender∈{0,1} ∧ mentions_location==0`, dedupe by text, `candidate_id = sha256(f"{language}|{text}")[:16]`, group cells with `sort=True`, shuffle `random_state=seed+cell_index`, slice in the order above. Byte-identical reproduction verified this session |
| geometry engine / pooling | decodanda / `hs_mean` | `pbs_gemma4_toxicity_geometry.sh:28-29` |
| geometry layers | 0,3,6,…,57 | `pbs_gemma4_toxicity_geometry.sh:23` |
| geometry dataset | geometry_400.csv, `protocol: production`, `variables` = `target_fields` = `[language_bin, toxicity_bin, gender_bin]`, `n_per_cell: 50`, no auto-reduce (cells are exactly 50; selection still reorders rows by the seeded shuffle) | `pbs_gemma4_toxicity_geometry.sh:27-30` |
| matched audit | `--protocol matched_audit` on geometry_400.csv, `audit_n_per_cell: 50`, same settings as Dataset A (production artifacts carry CCGP values, so CCGP is computed) | `pbs_gemma4_toxicity_matched_decodability.sh`; `results-gemma4-toxicity-matched-decodability-ccgp/` |
| zscore | true | `pbs_gemma4_toxicity_geometry.sh:33` (`--zscore`) |
| CCGP resamplings / nshuffles | 25 / 0 | `pbs_gemma4_toxicity_geometry.sh:31` |
| PS | nshuffles 0 | `pbs_gemma4_toxicity_geometry.sh:30` |
| ndata | 26 | `pbs_gemma4_toxicity_geometry.sh:31` (`--decodanda_ndata 26`) |
| SD | training_fraction .75, cross_validations 25, ndata 8, threshold 0.60 | `pbs_gemma4_toxicity_geometry.sh:32` (`--sd_cross_validations 25 --sd_ndata 8`); defaults `geometry_olmo_reduced_full.py:549-552` |
| geometry seed | 42; per-dichotomy SD seed = 42 + layer×100 + pooling offset | `geometry_olmo_reduced_full.py:30,603` |
| geometry batch_size / max_length | 4 / 512 | `pbs_gemma4_toxicity_geometry.sh:30` |
| judge model | Qwen/Qwen3.6-27B, bfloat16, trust_remote_code, greedy, max_new_tokens 1024, max_length 2048 | `pbs_qwen_toxicity_judge.sh`; `caa_model_search.py:437-475` |
| judge protocol | same prompt/parsing as Dataset A; each (setup, layer) generations file was one judge job (`num_jobs 1`), so `num_shards: 1`, `shard_group_by: [layer]` (steered is A on odd rows of each layer); no repair pass ran (`repair_invalid: false`; 10 rows stay `T` with `valid_decision` false) | `pbs_qwen_toxicity_judge.sh`; `caa_model_search.py:523-534` |
| judge target line (toxicity) | `Target toxicity: toxic, hostile, insulting, demeaning, or threatening language` / `Target toxicity: non-toxic, civil, non-hostile language` | `caa_model_search.py:400-406` |
| gpt-4.1-mini annotation | exact developer prompt and JSON schema shipped verbatim in `datasets/toxicity/DATASET.md`; batch runner at `scripts/toxicity/batch_label_toxicity.py` | `openai_labeling_outputs/batch_inputs/{en,ar}.jsonl` developer message (reused verbatim) |

### Outcome aggregation (Dataset B)

`repsteer aggregate --dataset toxicity` writes `toxicity_summary_by_layer_alpha.csv`,
`toxicity_best_alpha.csv` and (with `--geometry-rows`/`--matched-metrics`)
`toxicity_best_alpha_geometry.csv`, reproducing
`analyze_gemma4_toxicity_replication.py:17-48` (checked to 1e-12; an unlabelled
brokenness counts as not broken there, `n` counts unique ids). From that script and
`correlation_regression_report.py:10,132-160`:

- headline outcome = best-α per (setup, layer): within each cell pick the α
  maximizing `success` (mean steered-preference over the 200-example panel),
  tie-broken by lower `broken_rate`, then smaller α
  (`sort_values(["setup","layer","success","broken_rate","alpha"],
  ascending=[True,True,False,True,True])`, head(1)).
- the mean-over-α aggregation preserves the same qualitative geometry→outcome
  ordering (STORY_SO_FAR_FOLLOWUP.md:175-182), so the headline is not an
  artifact of the per-layer α choice.
- auxiliary outcomes: `broken_rate` (judge_broken_steered mean) and
  `classifier_target_delta` (XLM-R textdetox/xlmr-large-toxicity-classifier-v2
  probability delta aligned to the steering target; pass the per-text scores with
  `--classifier-scores`, columns `text_hash` = sha256 of the text, `toxicity_probability`).

## Parity caveats (documented divergences)

1. **Per-language prompts** are reproduced (`prompt_templates` keyed on
   `language`), including production's whitespace normalization.
2. **Geometry preselection** is reproduced: the Decodanda engine selects
   `n_per_cell` rows per cell exactly as `load_examples`
   (geometry_olmo_reduced_full.py:170-236) before extraction; selection, the
   eval-disjoint pool, the production geometry protocol, the matched audit, and
   hs_mean/mean/last/min/max extraction are checked for exact equality against
   the production code in `tests/test_decodanda_parity.py` and
   `tests/test_extraction_parity.py`.
3. **Geometry run dataset for catdog.** Production geometry ran on the full
   pool; the production split geometry rows are not a separate file. The
   config's `split` block therefore leaves `geometry_per_cell` null and the
   geometry budget is `geometry.n_per_cell = 50` with auto-reduce. For toxicity
   the geometry pool is a real shipped-file candidate (rebuilt), so the
   config's dataset path is `datasets/toxicity/geometry_400.csv`.
4. **Greedy decoding with inert temperature/top_p.** Production always calls
   `generate(..., do_sample=False, temperature=0.7, top_p=0.9)`; the toolkit
   backend ignores temperature/top_p when `do_sample` is false, exactly as the
   production `SearchCaaModel.generate` does. The configs record the values
   passed, decoding is greedy in both.
5. **Toxicity decodability source.** The archived production table
   `results-gemma4-toxicity-analysis/toxicity_best_alpha_geometry.csv`
   (written 2026-08-07) carries a `decodability` column equal to the SD
   decoder's named `toxicity_bin` dichotomy accuracy (`sd_ndata 8`, LinearSVC;
   exact match on all 20 layers), not the matched-audit decodability
   (`ndata 26`, svc) that Dataset A uses. The matched toxicity audit files were
   (re)written on 2026-08-19 and never re-joined. `repsteer aggregate
   --matched-metrics` joins the matched (`ndata 26`) decodability, so the
   toolkit applies the same matched-budget control to both datasets. All other
   toxicity outcome and geometry columns reproduce the archived table exactly.
   Toxicity partial correlations that control for decodability therefore differ
   from the archived report (CCGP and decodability are more collinear under the
   matched control: r = 0.94 vs 0.88).
