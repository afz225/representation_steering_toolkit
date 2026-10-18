"""Scaling-study steering pipeline: relative depths, relative-norm alphas,
batched grid, resume, and row sharding (fake backend, CPU only)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace as dataclasses_replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from repsteer.cli import filter_completed
from repsteer.config import (DatasetConfig, GenerationConfig, SplitConfig, SteeringConfig,
                             VectorSamplingConfig, load_config)
from repsteer.steering import (_norm_calibration_texts, layers_for_run,
                               relative_coefficients, resolve_relative_depths, run_steering)
from repsteer.types import VariableSpec
from repsteer.vectors import build_vectors, save_vectors


class FakeBatchBackend:
    """Deterministic fake: representations encode (target, layer) and generation
    encodes (prompt id, intervention layer, signed coefficient) so production and
    batched paths can be compared exactly."""

    n_layers = 4

    def sentence_representations(self, texts, layer, pooling, batch_size=8, max_length=512):
        rows = []
        for text in texts:
            target = 1.0 if "pos" in text else -1.0
            rows.append([target * (layer + 1), 1.0, 0.5 * layer])
        return np.asarray(rows, dtype=np.float32)

    def generate(self, prompts, *, intervention=None, **kwargs):
        suffix = " baseline" if intervention is None else (
            f" steered:{intervention.layer}:{intervention.coefficient:.6f}")
        return [prompt + suffix for prompt in prompts]

    def generate_steered_batch(self, prompt, interventions, *, max_length, max_new_tokens,
                               seed, normalize_whitespace=True, pad_with_eos=True):
        assert max_length == 2048 and max_new_tokens == 96 and seed == 42
        assert normalize_whitespace is False and pad_with_eos is False
        outputs = []
        for intervention in interventions:
            if intervention is None:
                outputs.append(prompt + " baseline")
            else:
                outputs.append(prompt + f" steered:{intervention.layer}:"
                               f"{intervention.coefficient:.6f}")
        return outputs

    def residual_norms(self, texts, layers, batch_size=8, max_length=512):
        return {layer: 10.0 + layer for layer in layers}


@pytest.fixture
def dataset_config():
    return DatasetConfig(
        path="unused.csv", text_column="text", id_column="id",
        variables={"target": ("neg", "pos")},
    )


@pytest.fixture
def frame():
    # Varying text lengths so per-layer draws (which pick different rows) produce
    # different mean vectors in the control test.
    rows = []
    for target in ("neg", "pos"):
        for index in range(8):
            padding = "X" * (index * 12)
            rows.append({"id": f"{target}-{index}", "text": f"{target} text {index} {padding}",
                         "target": target})
    return pd.DataFrame(rows)


def _run(backend, frame, steering, vectors, generation=None):
    spec = VariableSpec("target", "neg", "pos")
    return run_steering(
        backend, frame.reset_index(drop=True), "text", "id", spec, vectors,
        steering, generation or GenerationConfig(),
    )


def _with_norms(backend, vectors, layers):
    norms = backend.residual_norms(["x"], layers)
    return [dataclasses_replace(v, median_norm=norms[v.layer]) for v in vectors]


# ---------------------------------------------------------------- relative depths


def test_relative_depths_resolve_to_deduped_layer_indices():
    assert resolve_relative_depths([0.08, 0.16, 0.24, 0.92], 12) == (1, 2, 3, 11)
    # Rounding collisions collapse: 0.08*25 = 2, 0.09*25 = 2.25 -> 2.
    assert resolve_relative_depths([0.08, 0.09, 0.5, 0.5], 25) == (2, 12)
    assert resolve_relative_depths([0.999999], 3) == (2,)


def test_relative_depths_validation():
    with pytest.raises(ValueError):
        resolve_relative_depths([0.0], 12)
    with pytest.raises(ValueError):
        resolve_relative_depths([1.5], 12)


def test_layers_for_run_prefers_relative_depths():
    steering = SteeringConfig(layers=(0, 1), relative_depths=(0.5,), coefficients=(1.0,))
    assert layers_for_run(steering, FakeBatchBackend()) == (2,)
    steering = SteeringConfig(layers=(0, 1), coefficients=(1.0,))
    assert layers_for_run(steering, FakeBatchBackend()) == (0, 1)


def test_config_rejects_relative_depths_with_layers(tmp_path):
    raw = {
        "dataset": {"path": "unused.csv", "text_column": "text", "id_column": "id",
                    "variables": {"target": ["neg", "pos"]}},
        "model": {"model_id": "m"},
        "target": {"name": "target", "negative": "neg", "positive": "pos"},
        "split": {"vector_per_cell": 1, "eval_per_cell": 1},
        "generation": {},
        "steering": {"layers": [0], "relative_depths": [0.5], "coefficients": [1.0]},
        "geometry": {"layers": [0]},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(path)


# ------------------------------------------------------------------ relative alpha


def test_relative_coefficients_multiply_median_norm():
    backend = FakeBatchBackend()
    frame = pd.DataFrame({
        "id": ["a", "b"], "text": ["pos text", "neg text"], "target": ["pos", "neg"],
    })
    vectors = _with_norms(backend, build_vectors(
        backend, frame, "text", VariableSpec("target", "neg", "pos"), [1, 3], normalize=True),
        [1, 3])
    steering = SteeringConfig(layers=(1, 3), coefficients=(0.25, 2.0), alpha_mode="relative_norm")
    grid = relative_coefficients(steering, vectors, [1, 3])
    assert grid == {1: ((0.25, 0.25 * (10.0 + 1)), (2.0, 2.0 * (10.0 + 1))),
                    3: ((0.25, 0.25 * (10.0 + 3)), (2.0, 2.0 * (10.0 + 3)))}


def test_relative_coefficients_require_median_norm():
    backend = FakeBatchBackend()
    frame = pd.DataFrame({
        "id": ["a", "b"], "text": ["pos text", "neg text"], "target": ["pos", "neg"],
    })
    vectors = build_vectors(backend, frame, "text", VariableSpec("target", "neg", "pos"),
                            [0], normalize=True)
    steering = SteeringConfig(layers=(0,), coefficients=(1.0,), alpha_mode="relative_norm")
    with pytest.raises(ValueError, match="median_norm"):
        relative_coefficients(steering, vectors, [0])


def test_relative_norm_run_writes_both_columns(frame):
    backend = FakeBatchBackend()
    vectors = _with_norms(backend, build_vectors(
        backend, frame, "text", VariableSpec("target", "neg", "pos"), [1], normalize=True), [1])
    steering = SteeringConfig(layers=(1,), coefficients=(0.5,), alpha_mode="relative_norm")
    result = _run(backend, frame.iloc[[0, 6]], steering, vectors)
    assert result.alpha_rel.tolist() == [0.5, 0.5]
    assert result.coefficient.tolist() == [pytest.approx(0.5 * (10.0 + 1))] * 2


# ------------------------------------------------------------------- batched grid


def test_batched_grid_rows_equal_production_path(frame):
    backend = FakeBatchBackend()
    spec = VariableSpec("target", "neg", "pos")
    steering = SteeringConfig(layers=(0, 2), coefficients=(1.5, 3.0),
                              prompt_template="Continue: {text} toward {target}",
                              batched_grid=False)
    vectors = build_vectors(backend, frame, "text", spec, [0, 2])
    production = _run(backend, frame, steering, vectors)
    batched = _run(backend, frame, dataclasses_replace(steering, batched_grid=True), vectors)
    assert batched.steered_continuation.tolist() == production.steered_continuation.tolist()
    assert batched.baseline_continuation.tolist() == production.baseline_continuation.tolist()
    assert batched.layer.tolist() == production.layer.tolist()
    assert batched.coefficient.tolist() == production.coefficient.tolist()
    assert batched.id.tolist() == production.id.tolist()
    # Batched must lay out rows production-style: layer -> coefficient -> row.
    assert batched.layer.tolist() == [0] * 2 * len(frame) + [2] * 2 * len(frame)


def test_batched_grid_calls_generate_steered_batch_once_per_prompt(frame):
    backend = FakeBatchBackend()
    vectors = build_vectors(backend, frame, "text", VariableSpec("target", "neg", "pos"), [0, 2])
    steering = SteeringConfig(layers=(0, 2), coefficients=(1.0,), batched_grid=True)
    with patch.object(backend, "generate_steered_batch",
                      wraps=backend.generate_steered_batch) as spy:
        _run(backend, frame, steering, vectors)
    assert spy.call_count == len(frame)
    # Baseline + layers x alphas per prompt.
    interventions = spy.call_args_list[0].args[1]
    assert len(interventions) == 1 + 2 * 1
    assert interventions[0] is None


def test_batched_grid_requires_batch_capable_backend(frame):
    class PlainBackend:
        n_layers = 4

        def generate(self, prompts, **kwargs):
            return [prompt + " baseline" for prompt in prompts]

    steering = SteeringConfig(layers=(0,), coefficients=(1.0,), batched_grid=True)
    with pytest.raises(ValueError, match="generate_steered_batch"):
        _run(PlainBackend(), frame.iloc[[0]], steering, build_vectors(
            FakeBatchBackend(), frame, "text", VariableSpec("target", "neg", "pos"), [0]))


def test_new_columns_present(frame):
    backend = FakeBatchBackend()
    steering = SteeringConfig(layers=(1,), coefficients=(2.0,),
                              prompt_template="Continue: {text} toward {target}")
    result = _run(backend, frame.iloc[[0]], steering, build_vectors(
        backend, frame, "text", VariableSpec("target", "neg", "pos"), [1]))
    assert {"model_id", "concept", "relative_depth", "alpha_rel"} <= set(result.columns)
    assert result.model_id.tolist() == [""]
    assert result.concept.tolist() == ["target"]
    assert result.relative_depth.tolist() == [round(1 / 4, 2)]
    assert result.alpha_rel.tolist() == [2.0]


def test_model_id_and_concept_populated(frame):
    backend = FakeBatchBackend()
    steering = SteeringConfig(layers=(0,), coefficients=(1.0,),
                              prompt_template="Continue: {text} toward {target}")
    spec = VariableSpec("target", "neg", "pos")
    result = run_steering(backend, frame.iloc[[0]].reset_index(drop=True), "text", "id", spec,
                          build_vectors(backend, frame, "text", spec, [0]),
                          steering, GenerationConfig(), model_id="test/model")
    assert result.model_id.tolist() == ["test/model"]
    assert result.concept.tolist() == ["target"]


# ------------------------------------------------------------------ resume/shard


def test_filter_completed_skips_existing_keys():
    done = {(0, 1.0, "neg-0"), (0, 2.0, "neg-0")}
    output = pd.DataFrame({"layer": [0, 0, 1], "coefficient": [1.0, 2.0, 1.0],
                           "id": ["neg-0", "neg-0", "neg-0"]})
    kept = filter_completed(output, done, "id")
    assert kept.layer.tolist() == [1]


def test_resume_and_shard_round_trip(tmp_path, frame, dataset_config):
    """--rows slices the eval rows and --resume appends without redoing keys."""

    import repsteer.cli as cli
    from repsteer.cli import command_steer
    from repsteer.config import ExperimentConfig, GeometryConfig, ModelConfig
    from repsteer.data import split_dataset

    splits = split_dataset(frame, dataset_config,
                           SplitConfig(vector_per_cell=2, eval_per_cell=2, seed=3))
    pool = tmp_path / "pool.csv"
    splits["evaluation"].to_csv(pool, index=False)
    vectors_path = tmp_path / "vectors.npz"
    output_path = tmp_path / "steer.csv"

    experiment = ExperimentConfig(
        dataset=dataset_config, model=ModelConfig(model_id="fake"),
        target=VariableSpec("target", "neg", "pos"), split=SplitConfig(),
        generation=GenerationConfig(),
        steering=SteeringConfig(layers=(0,), coefficients=(1.0, 2.0),
                                prompt_template="Continue: {text} toward {target}"),
        geometry=GeometryConfig(layers=(0,)),
    )
    backend = FakeBatchBackend()
    vectors = build_vectors(backend, frame, "text", VariableSpec("target", "neg", "pos"), [0])
    save_vectors(vectors, vectors_path)

    def call(rows, resume):
        with patch.object(cli, "load_config", return_value=experiment), \
                patch.object(cli, "HuggingFaceBackend", return_value=backend), \
                patch.object(cli, "_frame", return_value=pd.read_csv(pool, keep_default_na=False)):
            command_steer(argparse.Namespace(
                config="cfg.json", dataset=pool, vectors=str(vectors_path),
                output=str(output_path), layers=None, coefficients=None,
                rows=rows, resume=resume, baseline_cache=None))
        return pd.read_csv(output_path, keep_default_na=False)

    first = call("0:2", False)
    assert len(first) == 2 * 1 * 2  # 2 rows x 1 layer x 2 alphas
    second = call("0:4", True)
    # The two resumed rows are skipped; the two new rows are appended.
    assert len(second) == len(first) + 2 * 1 * 2
    keys_first = set(zip(first.layer, first.coefficient, first.id))
    keys_second = set(zip(second.layer, second.coefficient, second.id))
    assert keys_first <= keys_second


def test_baseline_cache_round_trip(tmp_path, frame):
    from repsteer.cli import write_baseline_cache
    cache_path = tmp_path / "baselines.csv"
    output = pd.DataFrame({
        "id": ["a", "a", "b"], "baseline_continuation": ["one", "one", "two"]})
    write_baseline_cache(output, str(cache_path), "id")
    cached = pd.read_csv(cache_path)
    assert sorted(cached.id.tolist()) == ["a", "b"]
    write_baseline_cache(pd.DataFrame({
        "id": ["a", "c"], "baseline_continuation": ["new", "three"]}), str(cache_path), "id")
    cached = pd.read_csv(cache_path)
    assert dict(zip(cached.id, cached.baseline_continuation)) == {
        "a": "new", "b": "two", "c": "three"}


def test_norm_calibration_texts_are_deterministic_prefix(frame):
    texts = _norm_calibration_texts(frame, "text", 5)
    assert texts == frame.text.astype(str).tolist()[:5]
    assert _norm_calibration_texts(frame, "text", 100) == frame.text.astype(str).tolist()


# ----------------------------------------------------- per_layer_resample flag


def test_per_layer_resample_single_pass_matches_per_layer_build(frame):
    backend = FakeBatchBackend()
    spec = VariableSpec("target", "neg", "pos")
    sampling = VectorSamplingConfig(per_class=4, seed=42, positive_seed_offset=101,
                                    negative_seed_offset=102, per_layer_resample=False)
    single = build_vectors(backend, frame, "text", spec, [0, 2], sampling=sampling)
    per_layer = build_vectors(backend, frame, "text", spec, [0, 2], sampling=sampling)
    assert len(single) == len(per_layer) == 2
    for a, b in zip(single, per_layer):
        assert np.array_equal(a.values, b.values), a.layer


def test_per_layer_resample_default_changes_texts_per_layer(frame):
    """With per_layer_resample=True (production default), the fake's layer-indexed
    sampling makes class texts differ per layer, so vectors differ (control)."""

    class LengthMarkingBackend:
        n_layers = 4

        def sentence_representations(self, texts, layer, pooling, batch_size=8, max_length=512):
            # First component encodes text length only, so a changed draw changes
            # the mean vector. Long/short texts are mixed, so pos and neg means
            # stay distinct (nonzero norm).
            rows = [[float(len(t)) * (1.0 if "pos" in t else -1.0), 1.0, 0.0]
                    for t in texts]
            return np.asarray(rows, dtype=np.float32)

    marked = LengthMarkingBackend()
    spec = VariableSpec("target", "neg", "pos")
    sampling = VectorSamplingConfig(per_class=4, seed=42, positive_seed_offset=101,
                                    negative_seed_offset=102)
    vectors = build_vectors(marked, frame, "text", spec, [0, 2], sampling=sampling,
                            normalize=False)
    assert len(vectors) == 2
    assert not np.array_equal(vectors[0].values, vectors[1].values)


def test_per_layer_resample_false_falls_back_without_layers_method(frame):
    class NoLayersBackend(FakeBatchBackend):
        pass
    backend = NoLayersBackend()
    from repsteer.vectors import build_vectors as bv
    assert hasattr(backend, "sentence_representations_layers") is False
    sampling = VectorSamplingConfig(per_class=4, seed=42, positive_seed_offset=101,
                                    negative_seed_offset=102, per_layer_resample=False)
    vectors = build_vectors(backend, frame, "text", VariableSpec("target", "neg", "pos"),
                            [0, 2], sampling=sampling)
    assert len(vectors) == 2


def test_vectors_round_trip_preserves_median_norm(tmp_path, frame):
    from repsteer.vectors import load_vectors
    backend = FakeBatchBackend()
    spec = VariableSpec("target", "neg", "pos")
    vectors = _with_norms(backend, build_vectors(
        backend, frame, "text", spec, [1], normalize=True), [1])
    path = tmp_path / "vectors.npz"
    save_vectors(vectors, path)
    loaded = load_vectors(path)
    assert loaded[0].median_norm == pytest.approx(10.0 + 1)
    assert np.array_equal(loaded[0].values, vectors[0].values)


def test_vectors_round_trip_production_archives_have_null_median_norm(tmp_path, frame):
    from repsteer.vectors import load_vectors
    backend = FakeBatchBackend()
    spec = VariableSpec("target", "neg", "pos")
    vectors = build_vectors(backend, frame, "text", spec, [1], normalize=True)
    path = tmp_path / "vectors.npz"
    save_vectors(vectors, path)
    assert load_vectors(path)[0].median_norm is None
