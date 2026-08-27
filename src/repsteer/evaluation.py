"""Collateral capability evaluations for steering interventions."""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import GenerationConfig
from .models.base import ModelBackend
from .types import Intervention, PositionMode
from .vectors import SteeringVector, vector_index


def _general_intervention(vectors: Sequence[SteeringVector], layer: int, coefficient: float,
                          direction: str, positions: PositionMode) -> Intervention:
    index = vector_index(vectors)
    key = (layer, ())
    if key not in index:
        raise ValueError(f"Collateral evaluation requires an unconditioned vector at layer {layer}")
    if direction not in {"positive", "negative"}:
        raise ValueError("direction must be 'positive' or 'negative'")
    sign = 1.0 if direction == "positive" else -1.0
    return Intervention(layer, sign * index[key].values, coefficient, positions)


def evaluate_perplexity(backend: ModelBackend, texts: Sequence[str],
                        vectors: Sequence[SteeringVector], layers: Iterable[int],
                        coefficients: Iterable[float], *, direction: str = "positive",
                        positions: PositionMode = "all", max_length: int = 512,
                        batch_size: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure token-weighted causal perplexity before and after steering."""
    texts = [str(text) for text in texts]
    baseline = backend.sequence_nll(texts, max_length=max_length, batch_size=batch_size,
                                    intervention=None)
    rows = []
    for layer in layers:
        for coefficient in coefficients:
            intervention = _general_intervention(vectors, layer, coefficient, direction, positions)
            steered = backend.sequence_nll(texts, max_length=max_length, batch_size=batch_size,
                                           intervention=intervention)
            for sample_id, ((base_nll, base_tokens), (nll, tokens)) in enumerate(
                    zip(baseline, steered)):
                rows.append({"sample_id": sample_id, "layer": layer,
                             "coefficient": coefficient, "direction": direction,
                             "baseline_nll": base_nll, "baseline_tokens": base_tokens,
                             "steered_nll": nll, "steered_tokens": tokens})
    details = pd.DataFrame(rows)
    summaries = []
    for keys, group in details.groupby(["layer", "coefficient", "direction"], sort=True):
        base_ppl = math.exp(group.baseline_nll.sum() / group.baseline_tokens.sum())
        steer_ppl = math.exp(group.steered_nll.sum() / group.steered_tokens.sum())
        summaries.append({"layer": keys[0], "coefficient": keys[1], "direction": keys[2],
                          "baseline_perplexity": base_ppl, "steered_perplexity": steer_ppl,
                          "perplexity_delta": steer_ppl - base_ppl,
                          "perplexity_ratio": steer_ppl / base_ppl,
                          "n_examples": len(group),
                          "n_tokens": int(group.steered_tokens.sum())})
    return details, pd.DataFrame(summaries)


def _parse_choices(row: pd.Series) -> list[str]:
    if "choices" in row and pd.notna(row["choices"]):
        value = row["choices"]
        value = json.loads(value) if isinstance(value, str) else value
        return [str(item) for item in value]
    columns = [name for name in ("A", "B", "C", "D") if name in row]
    if len(columns) < 2:
        raise ValueError("Each row needs choices JSON or A/B/C/D columns")
    return [str(row[name]) for name in columns]


def _answer_index(row: pd.Series, labels: Sequence[str]) -> int:
    if "answer_idx" in row and pd.notna(row["answer_idx"]):
        return int(row["answer_idx"])
    answer = str(row["answer"]).strip().upper()
    return labels.index(answer) if answer in labels else int(answer)


def format_multiple_choice_prompt(question: str, choices: Sequence[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(f"{labels[i]}. {choice}" for i, choice in enumerate(choices))
    return f"{question}\n{options}\nAnswer with only the letter of the correct option.\nAnswer:"


def evaluate_multiple_choice(backend: ModelBackend, benchmark: pd.DataFrame,
                             vectors: Sequence[SteeringVector], layers: Iterable[int],
                             coefficients: Iterable[float], *, question_column: str = "question",
                             id_column: str | None = None, direction: str = "positive",
                             positions: PositionMode = "all", max_length: int = 2048,
                             batch_size: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate A/B/C/D-style tasks from next-token answer probabilities."""
    choice_rows = [_parse_choices(row) for _, row in benchmark.iterrows()]
    counts = {len(items) for items in choice_rows}
    if len(counts) != 1:
        raise ValueError("All rows must have the same number of choices")
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:counts.pop()])
    prompts = [format_multiple_choice_prompt(str(row[question_column]), choices)
               for (_, row), choices in zip(benchmark.iterrows(), choice_rows)]
    answers = np.asarray([_answer_index(row, labels) for _, row in benchmark.iterrows()])
    baseline = backend.next_token_choice_probabilities(
        prompts, labels, max_length=max_length, batch_size=batch_size, intervention=None)
    rows = []
    for layer in layers:
        for coefficient in coefficients:
            intervention = _general_intervention(vectors, layer, coefficient, direction, positions)
            steered = backend.next_token_choice_probabilities(
                prompts, labels, max_length=max_length, batch_size=batch_size,
                intervention=intervention)
            for i, (base_prob, probability) in enumerate(zip(baseline, steered)):
                item = {"sample_id": benchmark.iloc[i][id_column] if id_column else i,
                        "layer": layer, "coefficient": coefficient, "direction": direction,
                        "answer": labels[int(answers[i])],
                        "baseline_prediction": labels[int(np.argmax(base_prob))],
                        "steered_prediction": labels[int(np.argmax(probability))],
                        "baseline_correct": int(np.argmax(base_prob)) == answers[i],
                        "steered_correct": int(np.argmax(probability)) == answers[i]}
                for j, label in enumerate(labels):
                    item[f"baseline_prob_{label}"] = float(base_prob[j])
                    item[f"steered_prob_{label}"] = float(probability[j])
                rows.append(item)
    details = pd.DataFrame(rows)
    summary = details.groupby(["layer", "coefficient", "direction"], as_index=False).agg(
        baseline_accuracy=("baseline_correct", "mean"),
        steered_accuracy=("steered_correct", "mean"), n_examples=("sample_id", "size"))
    summary["accuracy_delta"] = summary.steered_accuracy - summary.baseline_accuracy
    return details, summary


def generate_instruction_responses(backend: ModelBackend, benchmark: pd.DataFrame,
                                   vectors: Sequence[SteeringVector], layers: Iterable[int],
                                   coefficients: Iterable[float], generation: GenerationConfig, *,
                                   prompt_column: str = "prompt", id_column: str = "key",
                                   direction: str = "positive",
                                   positions: PositionMode = "all") -> pd.DataFrame:
    """Generate paired response data compatible with M-IFEval-like scorers."""
    args = vars(generation)
    prompts = benchmark[prompt_column].astype(str).tolist()
    baseline = backend.generate(prompts, intervention=None, **args)
    rows = []
    for layer in layers:
        for coefficient in coefficients:
            intervention = _general_intervention(vectors, layer, coefficient, direction, positions)
            steered = backend.generate(prompts, intervention=intervention, **args)
            for i, (base_text, steer_text) in enumerate(zip(baseline, steered)):
                rows.append({"key": benchmark.iloc[i][id_column], "prompt": prompts[i],
                             "layer": layer, "coefficient": coefficient, "direction": direction,
                             "baseline_response": base_text, "steered_response": steer_text})
    return pd.DataFrame(rows)


def score_instruction_responses(responses: pd.DataFrame,
                                scorer: Callable[[str, str, pd.Series], dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a deterministic scorer callback and summarize follow_* metrics."""
    scored = []
    for _, row in responses.iterrows():
        for variant in ("baseline", "steered"):
            result = scorer(str(row.prompt), str(row[f"{variant}_response"]), row)
            output = row.to_dict()
            output.update(variant=variant, **result)
            scored.append(output)
    details = pd.DataFrame(scored)
    metrics = [column for column in details if column.startswith("follow_")
               and column != "follow_instruction_list"]
    if not metrics:
        raise ValueError("Scorer did not return a follow_* aggregate metric")
    summary = details.groupby(
        ["layer", "coefficient", "direction", "variant"], as_index=False)[metrics].mean()
    return details, summary
