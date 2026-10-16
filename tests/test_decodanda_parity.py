"""Production-exact Decodanda geometry: selection, geometry protocol, matched audit.

Production references live one directory above the toolkit
(geometry_olmo_reduced_full.py, audit_matched_decodability_ccgp.py and their
result folders); every production comparison skips when they are absent.
"""
from __future__ import annotations

import hashlib
import importlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("decodanda")

from repsteer.config import DatasetConfig, GeometryConfig, SplitConfig
from repsteer.data import exclude_evaluation_rows
from repsteer.geometry_decodanda import (matched_audit_layer, production_layer_geometry,
                                         select_examples)

TOOLKIT = Path(__file__).resolve().parents[1]
PRODUCTION = TOOLKIT.parent
CATDOG = TOOLKIT / "datasets" / "catdog" / "corpus.csv"
EVAL_DISJOINT = (PRODUCTION / "results-gemma4-matched-decodability-ccgp-audit-eval-disjoint"
                 / "geometry_dataset_excluding_steering_eval.csv")
TOXICITY = PRODUCTION / "results-gemma4-toxicity-split" / "geometry_400.csv"
CATDOG_FIELDS = ["language_bin", "style_bin", "is_dog"]
TOXICITY_FIELDS = ["language_bin", "toxicity_bin", "gender_bin"]
TIMING = {"SD_total_time_sec", "elapsed_sec"}


def production_module(name: str):
    if not (PRODUCTION / f"{name}.py").exists():
        pytest.skip("production scripts are not available in this checkout")
    if str(PRODUCTION) not in sys.path:
        sys.path.insert(0, str(PRODUCTION))
    return importlib.import_module(name)


def require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"production artifact missing: {path}")
    return path


def synthetic_reps(metadata: pd.DataFrame, fields, layer: int, dim: int = 64) -> np.ndarray:
    """Deterministic per-example features with layer-dependent named structure."""
    axes = np.random.default_rng(0).normal(size=(len(fields), dim))
    scales = np.asarray([1.5, 0.6, 0.35]) * (1 + layer / 10)
    out = []
    for _, row in metadata.iterrows():
        digest = hashlib.sha256(f"{row['candidate_id']}|{layer}".encode()).digest()
        noise = np.random.default_rng(int.from_bytes(digest[:8], "little")).normal(size=dim)
        signs = np.asarray([2 * int(row[field]) - 1 for field in fields])
        out.append(noise + (signs * scales[: len(fields)]) @ axes)
    return np.asarray(out, dtype=np.float32)


class _FakeStore:
    """Stands in for production PromptRepStore: no model, synthetic reps."""

    fields: list[str] = []

    def __init__(self, *args, **kwargs):
        self.tokenizer = None

    def extract_all_points(self, examples, layers, poolings, batch_size, max_length):
        rows = [{"idx": index, **example.meta} for index, example in enumerate(examples)]
        meta = pd.DataFrame(rows)
        reps = {pool: {layer: synthetic_reps(meta, self.fields, layer) for layer in layers}
                for pool in poolings}
        return reps, meta, 64


# --------------------------------------------------------------------------- selection

@pytest.mark.parametrize("dataset,fields,n,auto", [
    (CATDOG, CATDOG_FIELDS, 50, True),
    (EVAL_DISJOINT, CATDOG_FIELDS, 40, False),
    (TOXICITY, TOXICITY_FIELDS, 50, False),
])
def test_selection_matches_production_load_examples(dataset, fields, n, auto):
    production = production_module("geometry_olmo_reduced_full")
    require(dataset)
    examples, effective, counts = production.load_examples(
        str(dataset), fields, n, 42, None, 512, "all", auto)
    selected, toolkit_effective, toolkit_counts = select_examples(
        pd.read_csv(dataset), fields, n, 42, auto_reduce=auto)
    assert toolkit_effective == effective
    pd.testing.assert_frame_equal(toolkit_counts, counts, check_exact=True)
    assert selected["candidate_id"].astype(str).tolist() == [ex.meta["candidate_id"] for ex in examples]
    assert selected["text"].tolist() == [ex.prompt for ex in examples]
    for field in fields:
        assert selected[field].tolist() == [ex.meta[field] for ex in examples]


def test_selection_counts_match_production_artifact():
    counts_csv = require(PRODUCTION / "results-gemma4-geometry-hsmean-stride3-zscore" / "counts_job0.csv")
    _, _, counts = select_examples(pd.read_csv(CATDOG), CATDOG_FIELDS, 50, 42, auto_reduce=True)
    pd.testing.assert_frame_equal(counts, pd.read_csv(counts_csv), check_exact=True)


def test_eval_disjoint_dataset_is_byte_identical_to_production():
    require(EVAL_DISJOINT)
    dataset = DatasetConfig(
        path=str(CATDOG), text_column="text", id_column="candidate_id",
        variables={"language": ("ar", "en"), "style": ("preschool", "shakespeare"),
                   "is_dog": ("0", "1")})
    rebuilt = exclude_evaluation_rows(
        pd.read_csv(CATDOG), dataset, SplitConfig(vector_per_cell=32, eval_per_cell=25, seed=42))
    assert rebuilt.to_csv(index=False) == EVAL_DISJOINT.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- geometry protocol

def _csv_roundtrip(rows) -> pd.DataFrame:
    from io import StringIO
    return pd.read_csv(StringIO(pd.DataFrame(rows).to_csv(index=False)))


def _drop_timing(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[column for column in frame.columns if column in TIMING])


@pytest.mark.parametrize("dataset,fields,auto", [
    (CATDOG, CATDOG_FIELDS, True),
    (TOXICITY, TOXICITY_FIELDS, False),
])
def test_production_geometry_protocol_matches_production_main(tmp_path, monkeypatch, dataset, fields, auto):
    production = production_module("geometry_olmo_reduced_full")
    require(dataset)
    _FakeStore.fields = fields
    monkeypatch.setattr(production, "PromptRepStore", _FakeStore)
    config = GeometryConfig(layers=(0, 3), poolings=("hs_mean",), zscore=True, n_resamplings=5,
                            ndata=26, sd_ndata=8, sd_training_fraction=0.75, sd_cross_validations=4,
                            seed=42, n_per_cell=50, auto_reduce_n_per_cell=auto)
    selected, effective, _ = select_examples(pd.read_csv(dataset), fields, 50, 42, auto_reduce=auto)
    for layer in config.layers:
        out = tmp_path / f"geometry_job{layer}.csv"
        dich = tmp_path / f"geometry_job{layer}_dichotomies.csv"
        argv = ["geometry_olmo_reduced_full.py", "--dataset_csv", str(dataset), "--model_id", "m",
                "--layers", str(layer), "--poolings", "hs_mean", "--target_fields", *fields,
                "--condition_fields", *fields, "--n_per_cell", "50", "--ps_nshuffles", "0",
                "--ccgp_resamplings", "5", "--ccgp_nshuffles", "0", "--decodanda_ndata", "26",
                "--sd_cross_validations", "4", "--sd_ndata", "8", "--zscore",
                "--out_csv", str(out), "--dichotomy_out_csv", str(dich)]
        if auto:
            argv.append("--auto_reduce_n_per_cell")
        monkeypatch.setattr(sys, "argv", argv)
        production.main()  # set_all_seeds(42) at entry == one fresh process per layer

        rows, dichotomies = production_layer_geometry(
            synthetic_reps(selected, fields, layer), selected, config, condition_fields=fields,
            target_fields=fields, layer=layer, pooling="hs_mean", pooling_index=0,
            effective_n=effective, model_id="m")
        pd.testing.assert_frame_equal(_drop_timing(_csv_roundtrip(rows)),
                                      _drop_timing(pd.read_csv(out)), check_exact=True)
        pd.testing.assert_frame_equal(_drop_timing(_csv_roundtrip(dichotomies)),
                                      _drop_timing(pd.read_csv(dich)), check_exact=True)


@pytest.mark.parametrize("dataset,fields,n", [
    (EVAL_DISJOINT, CATDOG_FIELDS, 40),
    (TOXICITY, TOXICITY_FIELDS, 50),
])
def test_matched_audit_matches_production_main(tmp_path, monkeypatch, dataset, fields, n):
    production = production_module("audit_matched_decodability_ccgp")
    require(dataset)
    _FakeStore.fields = fields
    monkeypatch.setattr(production, "PromptRepStore", _FakeStore)
    config = GeometryConfig(layers=(3,), poolings=("hs_mean",), zscore=True, n_resamplings=5,
                            ndata=26, seed=42, protocol="matched_audit", n_per_cell=n,
                            audit_training_fraction=0.75, audit_cross_validations=6)
    selected, effective, _ = select_examples(pd.read_csv(dataset), fields, n, 42)
    monkeypatch.setattr(sys, "argv", [
        "audit_matched_decodability_ccgp.py", "--layer", "3", "--dataset_csv", str(dataset),
        "--model_id", "m", "--results_dir", str(tmp_path), "--n_per_cell", str(n),
        "--ndata", "26", "--cross_validations", "6", "--training_fraction", ".75",
        "--ccgp_resamplings", "5", "--condition_fields", *fields])
    production.main()
    rows = matched_audit_layer(synthetic_reps(selected, fields, 3), selected, config,
                               condition_fields=fields, layer=3, n_per_cell=effective,
                               model_id="m", dataset_csv=str(dataset))
    pd.testing.assert_frame_equal(_csv_roundtrip(rows),
                                  pd.read_csv(tmp_path / "matched_metrics_layer_3.csv"),
                                  check_exact=True)


# --------------------------------------------------------------------------- self-contained

def synthetic_condition_space(seed=12):
    rng = np.random.default_rng(seed)
    metadata, representations = [], []
    for a, b, c in itertools.product((0, 1), repeat=3):
        signs = np.asarray([2 * a - 1, 2 * b - 1, 2 * c - 1])
        for sample in range(12):
            metadata.append({"candidate_id": f"{a}{b}{c}-{sample}", "text": "x", "a": a, "b": b, "c": c})
            representations.append(
                np.asarray([3.0 * signs[0], 2.5 * signs[1], 2.0 * signs[2],
                            signs[0] * signs[1], signs[1] * signs[2], 0.0])
                + rng.normal(scale=0.15, size=6))
    return np.asarray(representations, dtype=np.float32), pd.DataFrame(metadata)


def small_config(**overrides):
    values = dict(layers=(0,), zscore=True, n_resamplings=3, ndata=8, sd_ndata=6,
                  sd_training_fraction=0.75, sd_cross_validations=5, seed=42, n_per_cell=12,
                  audit_cross_validations=5)
    values.update(overrides)
    return GeometryConfig(**values)


def test_production_protocol_shapes_and_reproducibility():
    x, metadata = synthetic_condition_space()
    kwargs = dict(condition_fields=["a", "b", "c"], target_fields=["a", "b", "c"], layer=0,
                  pooling="hs_mean", pooling_index=0, effective_n=12)
    rows, dichotomies = production_layer_geometry(x, metadata, small_config(), **kwargs)
    again, _ = production_layer_geometry(x, metadata, small_config(), **kwargs)
    assert [row["target_field"] for row in rows] == ["a", "b", "c"]
    assert len(dichotomies) == 35 and sum(item["is_named"] for item in dichotomies) == 3
    assert all(row["CCGP_nontargeted"] > 0.95 for row in rows)
    assert _drop_timing(pd.DataFrame(rows)).equals(_drop_timing(pd.DataFrame(again)))


def test_matched_audit_requires_balanced_cells():
    x, metadata = synthetic_condition_space()
    rows = matched_audit_layer(x, metadata, small_config(), condition_fields=["a", "b", "c"],
                               layer=0, n_per_cell=12)
    assert [row["target_field"] for row in rows] == ["a", "b", "c"]
    assert all(row["decodability_mean"] > 0.95 for row in rows)
    with pytest.raises(RuntimeError, match="balanced"):
        matched_audit_layer(x[:-1], metadata.iloc[:-1], small_config(),
                            condition_fields=["a", "b", "c"], layer=0, n_per_cell=12)


def test_selection_encodes_declared_string_variables():
    frame = pd.DataFrame({"text": [" a  b ", "c", "d", "e", ""],
                          "lang": ["en", "ar", "en", "ar", "en"],
                          "y": [0, 0, 1, 1, 1]})
    selected, effective, _ = select_examples(frame, ["lang", "y"], 1, 0,
                                             value_order={"lang": ("ar", "en")})
    assert effective == 1
    assert selected[["lang", "y"]].values.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert "a b" in selected.text.tolist()
