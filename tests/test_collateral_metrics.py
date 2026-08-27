from __future__ import annotations

import json

import numpy as np
import pandas as pd

from repsteer.config import GenerationConfig
from repsteer.evaluation import (evaluate_multiple_choice, evaluate_perplexity,
                                 generate_instruction_responses, score_instruction_responses)
from repsteer.offtarget import (AttributeLabels, judge_offtarget_effects,
                                measure_projection_drift, summarize_offtarget_effects)
from repsteer.types import VariableSpec
from repsteer.vectors import SteeringVector


def vector(layer=1, variable="style", values=(1.0, 0.0), condition=(), class_gap=2.0):
    return SteeringVector(layer, variable, condition, np.asarray(values, dtype=np.float32),
                          20, 20, class_gap)


class CapabilityBackend:
    def sequence_nll(self, texts, *, intervention=None, **kwargs):
        extra = 0.0 if intervention is None else intervention.coefficient
        return [(float(i + 1 + extra), i + 2) for i, _ in enumerate(texts)]

    def next_token_choice_probabilities(self, prompts, choices, *, intervention=None, **kwargs):
        if intervention is None:
            return np.asarray([[0.8, 0.2], [0.7, 0.3]])
        return np.asarray([[0.1, 0.9], [0.2, 0.8]])

    def generate(self, prompts, *, intervention=None, **kwargs):
        suffix = "base" if intervention is None else "steered"
        return [f"{prompt}:{suffix}" for prompt in prompts]


def test_perplexity_is_token_weighted():
    details, summary = evaluate_perplexity(
        CapabilityBackend(), ["a", "b"], [vector()], [1], [1.0])
    row = summary.iloc[0]
    assert len(details) == 2
    np.testing.assert_allclose(row.baseline_perplexity, np.exp(3 / 5))
    np.testing.assert_allclose(row.steered_perplexity, np.exp(5 / 5))


def test_multiple_choice_uses_next_token_probabilities():
    benchmark = pd.DataFrame({"question": ["q1", "q2"],
                              "choices": ['["x", "y"]', '["x", "y"]'],
                              "answer": ["A", "B"]})
    details, summary = evaluate_multiple_choice(
        CapabilityBackend(), benchmark, [vector()], [1], [2.0])
    assert details.baseline_correct.tolist() == [True, False]
    assert details.steered_correct.tolist() == [False, True]
    assert summary.iloc[0].baseline_accuracy == 0.5
    assert summary.iloc[0].steered_accuracy == 0.5


def test_instruction_generation_and_external_scorer_boundary():
    benchmark = pd.DataFrame({"key": ["one", "two"], "prompt": ["p1", "p2"]})
    responses = generate_instruction_responses(
        CapabilityBackend(), benchmark, [vector()], [1], [1.0], GenerationConfig())
    details, summary = score_instruction_responses(
        responses, lambda prompt, response, row: {
            "follow_all_instructions": response.endswith("steered")})
    assert len(details) == 4
    means = summary.set_index("variant").follow_all_instructions.to_dict()
    assert means == {"baseline": 0.0, "steered": 1.0}


class ProjectionBackend:
    def generate_with_representations(self, prompt, *, capture_layers, intervention=None, **kwargs):
        states = {layer: np.asarray([1.0, 2.0]) for layer in capture_layers}
        if intervention is not None:
            # Represents model propagation, intentionally including off-target movement.
            states = {layer: value + np.asarray([2.0, -1.0]) for layer, value in states.items()}
        return ("base" if intervention is None else "steered"), states


def test_projection_drift_recalculates_axis_and_normalizes_by_class_gap():
    frame = pd.DataFrame([{"id": "x", "text": "hello", "style": "simple",
                           "language": "en"}])
    style = VariableSpec("style", "simple", "literary")
    language = VariableSpec("language", "en", "ar")
    result = measure_projection_drift(
        ProjectionBackend(), frame, text_column="text", id_column="id",
        steering_spec=style, steering_vectors=[vector(variable="style")],
        measured_specs={"style": style, "language": language},
        measured_vectors={"style": [vector(variable="style", class_gap=2.0)],
                          "language": [vector(variable="language", values=(0, 1), class_gap=4.0)]},
        intervention_layer=1, coefficient=2.0, capture_layers=[1],
        generation=GenerationConfig())
    style_row = result[result.measured_variable == "style"].iloc[0]
    language_row = result[result.measured_variable == "language"].iloc[0]
    assert style_row.projection_delta == 2.0
    assert style_row.normalized_absolute_drift == 1.0
    assert language_row.projection_delta == -1.0
    assert language_row.normalized_absolute_drift == 0.25
    assert language_row.source_opposed_drift == -1.0


class JudgeBackend:
    def generate(self, prompts, **kwargs):
        return [json.dumps({
            "baseline_broken": False, "steered_broken": False,
            "baseline_labels": {"style": "simple", "language": "en"},
            "steered_labels": {"style": "literary", "language": "ar"},
            "reason": "both changed"})]


def test_llm_offtarget_flip_and_brokenness_are_separate():
    frame = pd.DataFrame([{"id": "x", "style": "simple", "language": "en",
                           "layer": 3, "coefficient": 4.0, "vector_condition": "general",
                           "prompt": "continue", "baseline_continuation": "plain",
                           "steered_continuation": "ornate"}])
    attributes = {
        "style": AttributeLabels("register", ("simple", "literary", "mixed"),
                                  {"simple": "simple", "literary": "literary"}),
        "language": AttributeLabels("language", ("en", "ar", "mixed"),
                                     {"en": "en", "ar": "ar"}),
    }
    judged = judge_offtarget_effects(
        JudgeBackend(), frame, attributes, GenerationConfig(), steered_variable="style")
    off_target = judged[judged.measured_variable == "language"].iloc[0]
    assert bool(off_target.offtarget_changed_vs_baseline)
    assert not bool(off_target.steered_broken)
    summary = summarize_offtarget_effects(judged)
    language_summary = summary[summary.measured_variable == "language"].iloc[0]
    assert language_summary.changed_vs_baseline_rate == 1.0
    assert language_summary.steered_broken_rate == 0.0
