# Cat/Dog synthetic corpus (`corpus.csv`)

Fully synthetic corpus used in the paper's Dataset A (style/cat-dog) experiments.
**Generated entirely by us — no human-authored content, no third-party text.**
The corpus file itself is released under MIT (see repository LICENSE). Please
credit the generator models below if you redistribute or build on it.

## Provenance

Produced by `synthetic_catdog_pilot.py` (full run) from the parent study:

- **Generator:** `mistralai/Mistral-7B-Instruct-v0.2` — generated the English
  source sentences (temperature 0.8, top-p 0.95, max_new_tokens 700).
- **Translator:** `tencent/Hy-MT2-1.8B` (HyMT) — Arabic versions are translations
  of the English sentences (direct Arabic generation was unreliable;
  max_new_tokens 180).
- **QA judge:** a Gemma-4 instruct model (`google/gemma-4-31B-it` text model)
  scoring each candidate against expected metadata with a strict JSON rubric
  (`synthetic_catdog_pilot.py::build_judge_prompt`): concept-as-noun, no
  opposite-concept mention, expected language, expected style, expected topic,
  exactly one sentence, no narrower subtypes.

Generation date: 2026-05-28T14:23:47 (per `dataset_stats` in the production
workspace).

## Pipeline

1. 16 conditions × 320 prompts = **5,120 raw candidates**
   (concept × style × topic × language, `full_rounds 32`).
2. All 5,120 candidates QA-judged; **1,274 pass**.
3. Balanced selection to a 125-per-cell target with
   `--allow_underfilled_full_dataset`: most cells underfilled → **1,041 accepted
   rows**.

## Contents and codebook

Columns of `corpus.csv` (1,041 rows):

| column | type | meaning |
| --- | --- | --- |
| `text` | str | the final (translated, if Arabic) candidate sentence |
| `concept` | {cat, dog} | nominal concept |
| `is_dog` | 0/1 | binary code: 1 = dog, 0 = cat |
| `style` | {shakespeare, preschool} | nominal style |
| `style_bin` | 0/1 | binary code: 0 = shakespeare, 1 = preschool |
| `topic` | {horror, sci_fi} | nominal topic |
| `topic_bin` | 0/1 | binary code: 0 = horror, 1 = sci_fi |
| `language` | {en, ar} | language |
| `language_bin` | 0/1 | binary code: 0 = en, 1 = ar |
| `candidate_id` | str | `stable_id` from `synthetic_catdog_pilot.py`: sha256 over `\n`-joined `[\"full\", batch_id, round_index, sentence_index, text]` (+ `source_text`, `translator_model_id` for ar), truncated to 16 hex chars — not a plain hash of `text` |
| `source_text` | str | English source sentence (equals `text` for en; the untranslated original for ar) |

## Condition counts

Full condition table (raw → QA pass → accepted; pass rate):

| concept | style | topic | language | raw | pass | accepted | pass_rate | status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| cat | shakespeare | horror | en | 320 | 51 | 51 | 0.159 | underfilled |
| cat | shakespeare | horror | ar | 320 | 53 | 53 | 0.166 | underfilled |
| cat | shakespeare | sci_fi | en | 320 | 19 | 19 | 0.059 | underfilled |
| cat | shakespeare | sci_fi | ar | 320 | 43 | 43 | 0.134 | underfilled |
| cat | preschool | horror | en | 320 | 16 | 16 | 0.050 | underfilled |
| cat | preschool | horror | ar | 320 | 28 | 28 | 0.087 | underfilled |
| cat | preschool | sci_fi | en | 320 | 168 | 125 | 0.525 | ok |
| cat | preschool | sci_fi | ar | 320 | 182 | 125 | 0.569 | ok |
| dog | shakespeare | horror | en | 320 | 76 | 76 | 0.237 | underfilled |
| dog | shakespeare | horror | ar | 320 | 84 | 84 | 0.263 | underfilled |
| dog | shakespeare | sci_fi | en | 320 | 36 | 36 | 0.113 | underfilled |
| dog | shakespeare | sci_fi | ar | 320 | 90 | 90 | 0.281 | underfilled |
| dog | preschool | horror | en | 320 | 18 | 18 | 0.056 | underfilled |
| dog | preschool | horror | ar | 320 | 27 | 27 | 0.084 | underfilled |
| dog | preschool | sci_fi | en | 320 | 178 | 125 | 0.556 | ok |
| dog | preschool | sci_fi | ar | 320 | 205 | 125 | 0.641 | ok |

Marginal counts over the 1,041 accepted rows (dominant-cell imbalance is expected
and acknowledged in the paper — the geometry analysis reduces the condition space
and `n_per_cell` is capped by the smallest cell):

- concept: 581 dog / 460 cat
- style: 589 preschool / 452 shakespeare
- topic: 688 sci_fi / 353 horror
- language: 575 ar / 466 en

Dominant failure reasons among rejected candidates:
`judge_pass=false` (3,836), style labeled other (3,256), topic labeled other
(733), specific-subtype mentions (214).

## Usage in the paper

- **Steering eval-200 split** (25/cell × 8 cells) and **vector-construction-256**
  (32/cell × 8 cells) are carved out of this pool by
  `scripts/replication/prepare_steering_eval200_split.py` (seed 42,
  `--vector_per_cell 32 --eval_per_cell 25`); the remaining 585 rows are unused
  for steering. Those split CSVs live with the results artifacts, not here.
- **Geometry analysis** (production headline) runs on the full 1,041-row pool
  with `n_per_cell 50` and auto-reduction; the eval-disjoint CCGP audit uses the
  pool minus the 200 steering-eval ids (841 rows), seeded-selecting 40/cell × 8.
- Topic is dropped from the reduced geometry condition tuple (language × style ×
  concept); it remains a recorded nuisance variable here.

## License

MIT (with the repository). The corpus is synthetic; its sentences are model
outputs of `mistralai/Mistral-7B-Instruct-v0.2` (and `tencent/Hy-MT2-1.8B`
translations), verified by a `google/gemma-4-31B-it` QA judge. The underlying
models carry their own licenses (Apache-2.0 for Mistral-7B-Instruct-v0.2; see
the Hy-MT2 and Gemma model cards) and are not distributed here.
