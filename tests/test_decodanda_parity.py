from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("decodanda")

from repsteer.config import GeometryConfig
from repsteer.geometry_decodanda import decodanda_geometry_metrics


def synthetic_condition_space(seed=12):
    rng = np.random.default_rng(seed)
    metadata = []
    representations = []
    for a, b, c in itertools.product(("0", "1"), repeat=3):
        signs = np.asarray([2 * int(a) - 1, 2 * int(b) - 1, 2 * int(c) - 1])
        for sample in range(12):
            metadata.append({"id": f"{a}{b}{c}-{sample}", "a": a, "b": b, "c": c})
            # Three parallel named axes, two interactions, and low noise.
            representations.append(
                np.asarray(
                    [
                        3.0 * signs[0],
                        2.5 * signs[1],
                        2.0 * signs[2],
                        signs[0] * signs[1],
                        signs[1] * signs[2],
                        0.0,
                    ]
                )
                + rng.normal(scale=0.15, size=6)
            )
    return np.asarray(representations, dtype=np.float32), pd.DataFrame(metadata)


def config():
    return GeometryConfig(
        layers=(0,),
        engine="decodanda",
        zscore=True,
        n_resamplings=3,
        ndata=8,
        sd_ndata=6,
        sd_training_fraction=0.75,
        sd_cross_validations=5,
        sd_accuracy_threshold=0.6,
        seed=42,
    )


def test_decodanda_backend_has_production_metric_shapes_and_space_sd():
    x, metadata = synthetic_condition_space()
    metrics = decodanda_geometry_metrics(x, metadata, ["a", "b", "c"], config())
    assert set(metrics.engine) == {"decodanda"}
    assert len(metrics[metrics.metric == "parallelism_score"]) == 3 * 6
    assert len(metrics[metrics.metric == "ccgp"]) == 3 * 4
    assert len(metrics[metrics.metric == "decodability"]) == 3 * 5
    assert metrics[metrics.metric == "sd_dichotomy"].dichotomy_id.nunique() == 32
    assert len(metrics[metrics.metric == "sd_dichotomy"]) == 32 * 5
    assert len(metrics[metrics.metric == "sd_mean_accuracy"]) == 1
    assert set(metrics[metrics.metric == "sd_mean_accuracy"].variable) == {"condition_space"}
    assert metrics[metrics.metric == "decodability"].value.mean() > 0.95
    assert metrics[metrics.metric == "ccgp"].value.mean() > 0.95


def test_decodanda_backend_is_reproducible_at_fixed_seed():
    x, metadata = synthetic_condition_space()
    first = decodanda_geometry_metrics(x, metadata, ["a", "b", "c"], config())
    second = decodanda_geometry_metrics(x, metadata, ["a", "b", "c"], config())
    key = ["metric", "variable", "dichotomy_id", "resampling"]
    first = first.sort_values(key, na_position="first").reset_index(drop=True)
    second = second.sort_values(key, na_position="first").reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_fast_sd_cap_always_preserves_named_dichotomies():
    x, metadata = synthetic_condition_space()
    fast = GeometryConfig(
        layers=(0,),
        engine="decodanda",
        zscore=True,
        n_resamplings=2,
        ndata=6,
        sd_ndata=4,
        sd_training_fraction=0.75,
        sd_cross_validations=2,
        max_sd_dichotomies=5,
        seed=42,
    )
    metrics = decodanda_geometry_metrics(x, metadata, ["a", "b", "c"], fast)
    assert set(metrics[metrics.metric == "decodability"].variable) == {"a", "b", "c"}
    assert metrics[metrics.metric == "sd_dichotomy"].dichotomy_id.nunique() == 2
