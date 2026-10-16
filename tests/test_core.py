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
    for row in result.itertuples():
        signed = 2.0 if row.target_value == "pos" else -2.0
        assert row.signed_coefficient == signed
        assert row.steered_continuation.endswith(f"steered:{signed}")


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
        ("Reason.\nDecision: A", ("A", None, None, True)),
        ("Broken_A: true\nBroken_B: false\nReason: x\nDecision: B", ("B", True, False, True)),
        ("Decision: A\nDecision: T", ("T", None, None, True)),
        ("A is broken, so Decision: B", ("B", True, None, True)),
        ('{"preferred": "B", "broken_a": "yes", "broken_b": 0}', ("B", True, False, True)),
        ("answer: A", ("A", None, None, True)),
        ("Decision: A, I think", ("T", None, None, False)),
        ("I prefer A", ("T", None, None, False)),
    ],
)
def test_parse_judgement_production_semantics(text, expected):
    parsed = parse_judgement(text)
    assert (parsed["preferred"], parsed["broken_a"], parsed["broken_b"],
            parsed["valid_decision"]) == expected


def test_summary_skips_unlabelled_brokenness():
    from repsteer.aggregate import summarize_by_alpha

    judged = pd.DataFrame(
        {
            "setup": ["general"] * 3,
            "task": ["style"] * 3,
            "layer": [3, 3, 3],
            "coefficient": [4.0, 4.0, 4.0],
            "steering_preferred": [True, False, False],
            "tie": [False, True, True],
            "judge_broken_steered": [False, True, np.nan],
            "judge_broken_baseline": [np.nan, np.nan, np.nan],
            "valid_decision": [True, True, False],
        }
    )
    summary = summarize_by_alpha(judged).iloc[0]
    assert summary.n == 3
    assert summary.steering_preferred_rate == pytest.approx(1 / 3)
    assert summary.steered_broken_rate == 0.5
    assert np.isnan(summary.baseline_broken_rate)
    assert summary.valid_decision_rate == pytest.approx(2 / 3)


def test_assign_options_round_robin_matches_production_shards():
    from repsteer.judge import assign_options

    frame = pd.DataFrame({"layer": [0] * 6 + [3] * 4})
    steered_is_a, positions = assign_options(frame, num_shards=4)
    # Production: job = i % 4, k = position inside the job, flip = (k + job) % 2.
    expected = [((i // 4) + (i % 4)) % 2 == 1 for i in range(10)]
    assert steered_is_a.tolist() == expected
    assert positions.tolist() == [i // 4 for i in range(10)]
    per_layer, _ = assign_options(frame, num_shards=1, group_by=("layer",))
    assert per_layer.tolist() == [k % 2 == 1 for k in range(6)] + [k % 2 == 1 for k in range(4)]
