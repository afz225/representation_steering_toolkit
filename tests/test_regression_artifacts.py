from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from repsteer.judge import parse_judgement, summarize_judgements

REPO = Path(__file__).resolve().parents[2]
SPLIT_DIR = REPO / "results-gemma4-steering-eval200-split"
RESULT_DIR = REPO / "results-gemma4-steering-eval200" / "general"
GEOMETRY_DIR = REPO / "results-gemma4-geometry-hsmean-stride3-zscore"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"Prior regression artifact not available: {path}")
    return path


def test_production_eval_and_vector_splits_are_balanced_and_disjoint():
    evaluation = pd.read_csv(_require(SPLIT_DIR / "evaluation_200.csv"))
    vectors = pd.read_csv(_require(SPLIT_DIR / "vector_construction_256.csv"))
    assert len(evaluation) == 200
    assert len(vectors) == 256
    assert not set(evaluation.candidate_id.astype(str)) & set(vectors.candidate_id.astype(str))
    assert evaluation.groupby(["language", "style", "concept"]).size().eq(25).all()
    assert vectors.groupby(["language", "style", "concept"]).size().eq(32).all()
    assert len(evaluation.groupby(["language", "style", "concept"])) == 8
    assert len(vectors.groupby(["language", "style", "concept"])) == 8


def test_new_parser_reproduces_production_string_judge_labels():
    scores = pd.read_csv(_require(RESULT_DIR / "string_judge_scores.csv"))
    for row in scores.itertuples(index=False):
        decision, broken = parse_judgement(row.raw_judgement)
        legacy_decision = "TIE" if str(row.string_preferred) == "T" else str(row.string_preferred)
        assert decision == legacy_decision
        broken_a = broken in {"A", "BOTH"}
        broken_b = broken in {"B", "BOTH"}
        if str(row.option_a_is).lower() == "steered":
            assert broken_a == bool(row.judge_broken_steered)
            assert broken_b == bool(row.judge_broken_baseline)
        else:
            assert broken_b == bool(row.judge_broken_steered)
            assert broken_a == bool(row.judge_broken_baseline)


def test_aggregate_preference_and_broken_rates_reproduce_prior_summary():
    scores = pd.read_csv(_require(RESULT_DIR / "string_judge_scores.csv"))
    expected = pd.read_csv(_require(RESULT_DIR / "summary_by_model_task_layer_alpha.csv"))
    normalized = scores.copy()
    normalized["judge_preference"] = np.select(
        [normalized.steering_preferred.astype(bool), normalized.baseline_preferred.astype(bool)],
        ["STEERED", "BASELINE"],
        default="TIE",
    )
    normalized["broken_steered"] = normalized.judge_broken_steered.astype(bool)
    normalized["broken_baseline"] = normalized.judge_broken_baseline.astype(bool)
    normalized["coefficient"] = normalized["alpha"]
    actual = summarize_judgements(
        normalized,
        group_columns=("model_id", "task", "layer", "coefficient"),
    ).rename(
        columns={
            "coefficient": "alpha",
            "n_examples": "n",
            "steering_preference_rate": "steering_preferred_rate",
            "steered_broken_rate": "steered_broken_rate",
            "baseline_broken_rate": "baseline_broken_rate",
        }
    )
    columns = [
        "model_id",
        "task",
        "layer",
        "alpha",
        "n",
        "steering_preferred_rate",
        "tie_rate",
        "steered_broken_rate",
        "baseline_broken_rate",
    ]
    merged = expected[columns].merge(actual[columns], on=["model_id", "task", "layer", "alpha"], suffixes=("_old", "_new"))
    assert len(merged) == len(expected)
    for column in ["n", "steering_preferred_rate", "tie_rate", "steered_broken_rate", "baseline_broken_rate"]:
        np.testing.assert_allclose(merged[f"{column}_new"], merged[f"{column}_old"], atol=1e-12)


def test_production_sd_is_one_space_metric_and_matches_dichotomy_rows():
    geometry = pd.read_csv(_require(GEOMETRY_DIR / "geometry_rows.csv"))
    dichotomies = pd.read_csv(_require(GEOMETRY_DIR / "dichotomy_rows.csv"))
    for layer, layer_rows in geometry.groupby("layer"):
        assert layer_rows.SD_mean_acc.nunique() == 1
        assert layer_rows.SD.nunique() == 1
        layer_dichotomies = dichotomies[dichotomies.layer == layer]
        assert len(layer_dichotomies) == 35
        assert int(layer_dichotomies.is_named.sum()) == 3
        np.testing.assert_allclose(
            layer_rows.SD_mean_acc.iloc[0],
            layer_dichotomies.acc_mean.mean(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            layer_rows.SD.iloc[0],
            layer_dichotomies.above_acc_threshold.mean(),
            atol=1e-12,
        )

PROJECTION_DIR = REPO / "results-gemma4-online-offtarget-projection-analysis"
OFFTARGET_DIR = REPO / "results-gemma4-targeted-alpha-offtarget-llm"
MMMLU_DIR = REPO / "results-gemma4-mmmlu-if-100" / "figures"
MIFEVAL_DIR = REPO / "results-gemma4-mifeval-if-100" / "figures"


def test_projection_formulas_and_summary_reproduce_prior_artifacts():
    from repsteer.offtarget import summarize_projection_drift

    old_rows = pd.read_csv(_require(PROJECTION_DIR / "online_projection_rows.csv"))
    np.testing.assert_allclose(old_rows.projection_delta,
                               old_rows.steered_projection - old_rows.baseline_projection)
    np.testing.assert_allclose(old_rows.absolute_projection_drift,
                               old_rows.projection_delta.abs())
    np.testing.assert_allclose(old_rows.normalized_absolute_drift,
                               old_rows.absolute_projection_drift / old_rows.class_gap)
    adapted = old_rows.rename(columns={
        "steering_task": "steering_variable", "measured_task": "measured_variable",
        "candidate_id": "sample_id"}).drop(columns=["alpha"])
    actual = summarize_projection_drift(adapted)
    expected = pd.read_csv(_require(PROJECTION_DIR / "online_projection_summary.csv"))
    expected = expected.rename(columns={
        "steering_task": "steering_variable", "measured_task": "measured_variable",
        "alpha": "coefficient", "n": "n_examples",
        "mean_abs_drift": "mean_absolute_projection_drift",
        "mean_normalized_abs_drift": "mean_normalized_absolute_drift"})
    keys = ["setup", "steering_variable", "measured_variable", "is_off_target",
            "intervention_layer", "readout_layer", "coefficient"]
    # The historical projection summary intentionally collapsed alpha.
    keys.remove("coefficient")
    merged = expected.merge(actual, on=keys, suffixes=("_old", "_new"))
    assert len(merged) == len(expected)
    for metric in ["n_examples", "mean_absolute_projection_drift",
                   "mean_normalized_absolute_drift", "mean_source_opposed_drift"]:
        np.testing.assert_allclose(merged[f"{metric}_new"], merged[f"{metric}_old"], atol=1e-12)


def test_offtarget_llm_summary_reproduces_prior_artifact():
    from repsteer.offtarget import summarize_offtarget_effects

    rows = pd.read_csv(_require(OFFTARGET_DIR / "offtarget_llm_effect_rows.csv"))
    adapted = rows.rename(columns={
        "steered_task": "steered_variable", "offtarget_variable": "measured_variable",
        "alpha": "coefficient", "llm_broken_steered": "steered_broken",
        "llm_broken_baseline": "baseline_broken"})
    adapted["is_off_target"] = True
    adapted["judge_parse_error"] = np.where(adapted.valid_offtarget_parse, "", "legacy parse error")
    actual = summarize_offtarget_effects(adapted)
    expected = pd.read_csv(_require(OFFTARGET_DIR / "offtarget_llm_effect_summary.csv"))
    expected = expected.rename(columns={
        "steered_task": "steered_variable", "offtarget_variable": "measured_variable",
        "alpha": "coefficient", "n": "n_examples",
        "off_target_change_rate": "changed_vs_baseline_rate",
        "off_target_drift_rate": "drift_from_source_rate",
        "broken_rate": "steered_broken_rate"})
    keys = ["steering_setup", "steered_variable", "measured_variable", "layer", "coefficient"]
    merged = expected.merge(actual, on=keys, suffixes=("_old", "_new"))
    assert len(merged) == len(expected)
    np.testing.assert_allclose(merged.parse_error_rate, 1 - merged.valid_parse_rate, atol=1e-12)
    for metric in ["n_examples", "changed_vs_baseline_rate", "drift_from_source_rate",
                   "steered_broken_rate"]:
        np.testing.assert_allclose(merged[f"{metric}_new"], merged[f"{metric}_old"], atol=1e-12)


def test_prior_mmmlu_and_mifeval_deltas_are_baseline_relative():
    mmmlu = pd.read_csv(_require(MMMLU_DIR / "mmmlu_scores_by_intervention.csv"))
    baseline_accuracy = float(mmmlu.loc[mmmlu.run_type == "baseline", "accuracy"].iloc[0])
    steered = mmmlu[mmmlu.run_type == "steered"]
    np.testing.assert_allclose(steered.delta_accuracy,
                               steered.accuracy - baseline_accuracy, atol=1e-12)

    mifeval = pd.read_csv(_require(MIFEVAL_DIR / "mifeval_scores_by_intervention.csv"))
    baseline = mifeval[mifeval.run_type == "baseline"].iloc[0]
    steered_if = mifeval[mifeval.run_type == "steered"]
    for metric in ["strict_prompt_accuracy", "strict_instruction_accuracy",
                   "loose_prompt_accuracy", "loose_instruction_accuracy"]:
        np.testing.assert_allclose(steered_if[f"delta_{metric}"],
                                   steered_if[metric] - baseline[metric], atol=1e-12)
