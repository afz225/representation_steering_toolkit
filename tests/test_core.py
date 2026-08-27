from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from repsteer.config import DatasetConfig, GenerationConfig, SplitConfig, SteeringConfig
from repsteer.data import split_dataset, validate_dataset
from repsteer.geometry import geometry_metrics
from repsteer.judge import parse_judgement
from repsteer.steering import run_steering
from repsteer.types import VariableSpec
from repsteer.vectors import build_vectors


class FakeBackend:
    n_layers = 3

    def sentence_representations(self, texts, layer, pooling, batch_size=8, max_length=512):
        rows = []
        for text in texts:
            # Encodes labels embedded in the synthetic text: target, nuisance, layer.
            target = 1.0 if "pos" in text else -1.0
            nuisance = 1.0 if "right" in text else -1.0
            rows.append([target * (layer + 1), nuisance, 1.0])
        return np.asarray(rows, dtype=np.float32)

    def generate(self, prompts, *, intervention=None, **kwargs):
        suffix = " baseline" if intervention is None else f" steered:{intervention.coefficient}"
        return [prompt + suffix for prompt in prompts]


@pytest.fixture
def dataset_config():
    return DatasetConfig(
        path="unused.csv",
        text_column="text",
        id_column="id",
        variables={"target": ("neg", "pos"), "side": ("left", "right")},
    )


@pytest.fixture
def frame():
    rows = []
    for target, side in itertools.product(("neg", "pos"), ("left", "right")):
        for index in range(8):
            rows.append({"id": f"{target}-{side}-{index}", "text": f"{target} {side}", "target": target, "side": side})
    return pd.DataFrame(rows)


def test_split_is_balanced_and_disjoint(frame, dataset_config):
    validate_dataset(frame, dataset_config)
    splits = split_dataset(
        frame,
        dataset_config,
        SplitConfig(vector_per_cell=2, eval_per_cell=2, geometry_per_cell=3, seed=7),
    )
    assert {name: len(part) for name, part in splits.items()} == {
        "vector": 8,
        "evaluation": 8,
        "geometry": 12,
    }
    id_sets = [set(part.id) for part in splits.values()]
    assert all(not left & right for left, right in itertools.combinations(id_sets, 2))


def test_conditioned_vectors_and_opposite_steering(frame):
    backend = FakeBackend()
    spec = VariableSpec("target", "neg", "pos", ("side",))
    vectors = build_vectors(backend, frame, "text", spec, [0, 2], normalize=True)
    assert len(vectors) == 4
    assert all(np.isclose(np.linalg.norm(vector.values), 1.0) for vector in vectors)
    evaluation = frame.iloc[[0, 16]].copy()
    steering = SteeringConfig(
        layers=(0,),
        coefficients=(2.0,),
        prompt_template="Continue: {text} toward {target}",
    )
    result = run_steering(
        backend,
        evaluation,
        "text",
        "id",
        spec,
        vectors,
        steering,
        GenerationConfig(),
    )
    assert len(result) == 2
    assert set(result.target_value) == {"neg", "pos"}
    assert result.steered_continuation.str.contains("steered:2.0").all()


def test_geometry_recovers_parallel_binary_structure(frame):
    backend = FakeBackend()
    x = backend.sentence_representations(frame.text.tolist(), layer=1, pooling="mean")
    metrics = geometry_metrics(
        x,
        frame,
        ["target", "side"],
        n_resamplings=3,
        zscore=True,
        max_per_condition=6,
        seed=5,
    )
    named = metrics[(metrics.metric == "decodability") & (metrics.variable == "target")]
    assert named.value.mean() > 0.95
    ccgp = metrics[(metrics.metric == "ccgp") & (metrics.variable == "target")]
    assert ccgp.value.mean() > 0.95
    assert set(metrics[metrics.metric == "sd_mean_accuracy"].variable) == {"condition_space"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reason.\nDecision: A\nBroken: NONE", ("A", "NONE")),
        ("Decision: B\nBroken: BOTH", ("B", "BOTH")),
        ("Decision: A\nDecision: TIE\nBroken: NONE", ("TIE", "NONE")),
    ],
)
def test_parse_judgement_uses_final_decision(text, expected):
    assert parse_judgement(text) == expected


def test_parse_judgement_rejects_unstructured_output():
    with pytest.raises(ValueError):
        parse_judgement("I prefer A")


def test_judgement_summary_excludes_parse_errors():
    from repsteer.judge import summarize_judgements

    judged = pd.DataFrame(
        {
            "layer": [3, 3, 3],
            "coefficient": [4.0, 4.0, 4.0],
            "vector_condition": ["general"] * 3,
            "judge_preference": ["STEERED", "TIE", "PARSE_ERROR"],
            "broken_steered": [False, True, False],
            "broken_baseline": [False, False, False],
        }
    )
    summary = summarize_judgements(judged).iloc[0]
    assert summary.n_valid == 2
    assert summary.steering_preference_rate == 0.5
    assert summary.steered_broken_rate == 0.5


def test_parse_judgement_preserves_decision_when_brokenness_is_missing():
    assert parse_judgement("Reason: clear target match. Decision: A") == ("A", "UNKNOWN")


def test_judgement_summary_all_unknown_brokenness_is_nan():
    from repsteer.judge import summarize_judgements

    judged = pd.DataFrame(
        {
            "layer": [0, 0],
            "coefficient": [1.0, 1.0],
            "vector_condition": ["general", "general"],
            "judge_preference": ["STEERED", "BASELINE"],
            "broken_steered": [pd.NA, pd.NA],
            "broken_baseline": [pd.NA, pd.NA],
        }
    )
    summary = summarize_judgements(judged).iloc[0]
    assert np.isnan(summary.steered_broken_rate)
    assert np.isnan(summary.baseline_broken_rate)
