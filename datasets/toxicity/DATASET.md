# Toxicity corpus (texts NOT shipped)

Dataset B (toxicity) of the paper: real social-media posts in English (en) and
Arabic (ar), re-annotated for our binary condition grid
`language × toxicity × gender-mention` (8 cells).

**The candidate texts are intentionally NOT included in this repository.** They
are real social-media posts compiled by third parties, and redistribution of the
annotated CSVs is not clearly permitted. Instead this directory ships:

- `manifest.json` — per-candidate fingerprints (sha256 of the whitespace-stripped
  text), cell, and split membership for all 856 candidates, exactly as used in
  the production split.
- `verify_split.py` — proves a locally rebuilt split matches the manifest
  (fingerprints, cells, split sizes).
- `prepare_toxicity_splits.py` — the deterministic split builder, byte-for-byte
  compatible with the production run given the labeled inputs described below.

## Provenance

1. **Underlying texts:** [`textdetox/multilingual_toxicity_dataset`](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset)
   (license: **openrail++**), `en` and `ar` splits, 5,000 texts each.
   - English portion originally from the Jigsaw/Unitary toxic-comment corpora;
   - Arabic portion originally from OSACT4 shared-task data.
   These source corpora carry their own noncommercial/academic-use-style terms;
   respect them when you download.
2. **Re-annotation (our pipeline):** each text was labeled with
   OpenAI `gpt-4.1-mini` Batch API calls (Responses API, JSON-schema output
   `identity_location_labels`) for the five binary fields
   `mentions_gender`, `mentions_ethnicity`, `mentions_religion`,
   `mentions_nationality`, `mentions_location`, plus short justifications.
   Labeled outputs live in `openai_labeling_outputs/labeled/{en,ar}_labeled.csv`
   (production workspace; not shipped).
3. **Split:** `prepare_toxicity_splits.py` filters to
   `toxic ∈ {0,1}` ∧ `mentions_gender ∈ {0,1}` ∧ `mentions_location == 0`,
   dedupes texts, computes candidate ids, forms the 8 cells
   (language_bin en=0/ar=1, toxicity_bin = `toxic`, gender_bin =
   `mentions_gender`), and within each cell shuffles with
   `pandas.sample(frac=1, random_state=seed + cell_index)` (seed 42, cell order
   from `groupby(sort=True)`) before slicing, in fixed order:
   - `geometry_400.csv` — 50 per cell (400 rows)
   - `vector_construction_256.csv` — 32 per cell (256 rows)
   - `evaluation_200.csv` — 25 per cell (200 rows)

   Disjointness across splits is asserted (no leaked candidate ids).
   Production command:
   `python prepare_toxicity_gemma4_splits.py --input_dir openai_labeling_outputs/labeled --output_dir results-gemma4-toxicity-split --geometry_per_cell 50 --vectors_per_cell 32 --evaluation_per_cell 25 --seed 42`
   → 856 distinct candidates, 50/32/25 per cell × 8 cells.

## Exact gpt-4.1-mini labeling prompt (developer message, verbatim)

This is the `developer` message embedded in every batch request of
`openai_labeling_outputs/batch_inputs/{en,ar}.jsonl` (`/v1/responses`, model
`gpt-4.1-mini`); the `user` message is simply `Text: {text}`:

```text
You are labeling short social media texts.

Task:
For the given text, output binary labels for whether the text explicitly mentions:
1. gender
2. ethnicity
3. religion
4. nationality
5. location

Rules:
- Use 1 only for explicit mention, not implication.
- Base the decision only on the text provided.
- Do not infer author identity or target identity unless explicitly stated.
- Nationality and ethnicity are separate.
- Location means a geographic place name or region explicitly mentioned, mentioning a people or community does not count as the location, it must be an explicit mention of a place.
- Justifications must be extremely brief and cite the trigger word or say no explicit mention.
- Return only schema-compliant JSON.
```

Structured output schema (`body.text.format`, json_schema, name
`identity_location_labels`): object with required integer 0/1 properties
`mentions_gender`, `mentions_ethnicity`, `mentions_religion`,
`mentions_nationality`, `mentions_location`, plus string properties
`justification_gender`, `justification_ethnicity`, `justification_religion`,
`justification_nationality`, `justification_location`.
An optional batch-labeling runner is provided at
`scripts/toxicity/batch_label_toxicity.py`.

## Identifiers and fingerprints

- `candidate_id` = first 16 hex chars of `sha256(f"{language}|{text}")` over the
  UTF-8 whitespace-stripped text.
- `manifest.json` additionally stores `text_sha256` = sha256 hex digest of the
  whitespace-stripped text (so texts can be matched without ever being shipped).
- `cell` in the manifest = `[language, toxicity_bin, gender_bin]`.
- `split` ∈ {`geometry_400`, `vector_construction_256`, `evaluation_200`}.

## Rebuilding the data locally

1. Download the `en` and `ar` splits of
   `textdetox/multilingual_toxicity_dataset` from Hugging Face.
2. Run the gpt-4.1-mini annotation with the exact developer prompt above (see
   `scripts/toxicity/batch_label_toxicity.py` for an OpenAI Batch API runner
   stub; the production run used the 2026-03-25 batch snapshot
   `openai_data.zip`).
3. Assemble `{en,ar}_labeled.csv` with columns `text, toxic, mentions_gender,
   mentions_location` (plus the remaining schema columns; the splitter only
   reads those four).
4. `python datasets/toxicity/prepare_toxicity_splits.py --input-dir <labeled_dir> --output-dir <out>`
5. `python datasets/toxicity/verify_split.py --splits-dir <out>`
   → must print `OK: 856 rows ...` (verified against the production CSVs before
   release; the port reproduced all three split CSVs byte-identically).

## License

- The annotated texts: NOT redistributed (third-party social-media content,
  openrail++ source corpora with further noncommercial/academic terms).
- The annotation **prompts, identifiers, fingerprints, manifest, split logic and
  scripts**: ours, MIT (with the repository).
