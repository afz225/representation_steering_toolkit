# Evaluation and Collateral Effects

## Pairwise steering judge

`repsteer judge` compares a baseline continuation with its steered counterpart. A/B
placement is deterministic but randomized by example ID and seed. The generated
judge response is parsed from its final decision rather than answer-token logits.

The summary reports:

- `steering_preference_rate`;
- `baseline_preference_rate`;
- `tie_rate`;
- baseline and steered broken-output rates;
- judge parse-error rate.

Broken output means unusable prose due to emptiness, incoherence, semantic nonsense,
severe grammar failure, hallucinated labels or boilerplate, pathological repetition,
truncation, or artifact corruption. Missing the target attribute is not brokenness.

## Perplexity

Perplexity evaluation requires an unconditioned vector archive because benchmark
texts do not carry nuisance metadata:

```bash
repsteer perplexity --config style-general.json --benchmark text.csv \
  --text-column text --vectors vectors.npz --output-dir results/perplexity
```

The summary computes token-weighted causal perplexity:

```text
PPL = exp(sum_document_nll / sum_scored_tokens)
```

It reports baseline and steered PPL, absolute delta, and ratio for every
layer/coefficient setup.

## Multiple-choice capability

```bash
repsteer multiple-choice --config style-general.json --benchmark mmlu.csv \
  --vectors vectors.npz --output-dir results/mmlu
```

Rows need `question`, `answer`, and either a JSON `choices` list or `A`, `B`, `C`,
`D` columns. The evaluator applies a softmax over next-token answer-letter logits and
reports baseline/steered predictions, probabilities, accuracy, and accuracy delta.
Each answer letter with a leading space must tokenize to one token.

## Instruction following

```bash
repsteer instruction-generate --config style-general.json \
  --benchmark mifeval.csv --vectors vectors.npz \
  --output results/mifeval_responses.csv
```

The benchmark needs `key` and `prompt` columns by default. RepSteer writes paired
baseline and steered responses. Apply the official M-IFEval scorer to these outputs,
or use `score_instruction_responses` with a deterministic callback returning
`follow_*` metrics. Benchmark-specific instruction rules intentionally remain
outside the core package.

## Online off-target projection drift

```bash
repsteer offtarget-project --config style.json --dataset probe.csv \
  --vectors style_vectors.npz --intervention-layer 12 --coefficient 8 \
  --capture-layers 12,15,18,21,24 \
  --axis-config language=language.json --axis-vectors language=language_vectors.npz \
  --axis-config is_dog=is_dog.json --axis-vectors is_dog=is_dog_vectors.npz \
  --output-dir results/projection
```

During generation, the evaluator captures the final-position hidden state on each
forward call and averages those states per layer. At each readout layer it uses the
concept direction constructed for that same layer. For off-target direction `u_y`:

```text
projection_delta = (h_steered - h_baseline) dot unit(u_y)
absolute_drift = abs(projection_delta)
normalized_absolute_drift = absolute_drift / class_mean_gap_y
```

`source_opposed_drift` additionally orients the signed delta away from the sample's
source class. Reading downstream axes at downstream layers avoids incorrectly
reusing the intervention-layer coordinate system.

## Off-target LLM effects

```bash
repsteer offtarget-judge --config style.json --input generations.csv \
  --labels configs/example_offtarget_labels.json \
  --output-dir results/offtarget_judge
```

The label JSON defines allowed labels, descriptions, and mappings from dataset values
to judge labels. The judge labels baseline and steered text together for every
attribute. `changed_vs_baseline_rate` is the percentage whose steered off-target
label differs from baseline. `drift_from_source_rate` compares the steered label with
the source dataset class. Both remain separate from broken-output rate.

## Interpreting results

Always report the number of examples, parse-error rate, broken-output rate, direction,
coefficient, intervention position, layer, and whether vectors are general or
condition-specific. A higher preference rate without fluent outputs is not successful
steering. Likewise, an off-target label change may be a real collateral effect even
when the output remains fluent.
