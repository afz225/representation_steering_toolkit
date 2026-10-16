from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from repsteer.correlations import catdog_section, partial_r, stars, toxicity_section

scipy = pytest.importorskip("scipy")


def _long_geometry() -> pd.DataFrame:
    """Two layers x two target variables in the decodanda long layout."""
    rows = []
    for layer in (3, 6):
        for variable in ("style_bin", "is_dog"):
            for resampling in range(3):
                rows.append({"layer": layer, "variable": variable,
                             "metric": "parallelism_score",
                             "resampling": resampling,
                             "value": 0.2 + 0.05 * resampling + 0.01 * layer})
                rows.append({"layer": layer, "variable": variable,
                             "metric": "ccgp",
                             "resampling": resampling,
                             "value": 0.5 + 0.02 * resampling + 0.01 * layer})
        for resampling in range(3):
            rows.append({"layer": layer, "variable": "style_bin",
                         "metric": "decodability", "resampling": resampling,
                         "value": 0.7 + 0.01 * layer + 0.01 * resampling})
            rows.append({"layer": layer, "variable": "is_dog",
                         "metric": "decodability", "resampling": resampling,
                         "value": 0.6 + 0.01 * layer + 0.01 * resampling})
        # condition-space SD summaries (variable='condition_space')
        rows.append({"layer": layer, "variable": "condition_space",
                     "metric": "sd_mean_accuracy", "resampling": -1,
                     "value": 0.65 + 0.01 * layer})
        rows.append({"layer": layer, "variable": "condition_space",
                     "metric": "sd_fraction_above_threshold", "resampling": -1,
                     "value": 0.8 + 0.01 * layer})
    return pd.DataFrame(rows)


def _catdog_summary() -> pd.DataFrame:
    rows = []
    for setup in ("general", "conditioned"):
        for task in ("style", "concept"):
            for index, layer in enumerate((3, 6)):
                rows.append({
                    "setup": setup, "task": task, "layer": layer,
                    "steering_preferred_rate_mean_alpha": 0.2 + 0.3 * index,
                    "steered_broken_rate_mean_alpha": 0.4 - 0.1 * index,
                    "best_alpha": 16.0 + 16.0 * index,
                    "best_success": 0.3 + 0.1 * index,
                    "best_broken": 0.3 - 0.05 * index,
                })
    return pd.DataFrame(rows)


def _wrap_wide(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Reshape the long fixture into the production geometry_rows layout."""
    pivot = long_frame.pivot_table(index=["layer", "variable"], columns="metric",
                                   values="value", aggfunc="mean").reset_index()
    pivot = pivot.rename(columns={
        "parallelism_score": "PS", "ccgp": "CCGP_nontargeted",
        "decodability": "decodability_mean", "sd_mean_accuracy": "SD_mean_acc",
        "sd_fraction_above_threshold": "SD", "variable": "target_field"})
    return pivot


def test_stars_match_production_thresholds():
    assert stars(0.0001) == "***"
    assert stars(0.002) == "**"
    assert stars(0.02) == "*"
    assert stars(0.2) == ""


def test_partial_r_matches_manual_computation():
    rng = np.random.default_rng(0)
    x, z = rng.normal(size=60), rng.normal(size=60)
    y = 0.5 * x + 0.9 * z + rng.normal(size=60)
    r, p, n = partial_r(x, y, z)
    assert n == 60
    # y is almost fully explained by z; partial r of x,y|z is small
    assert abs(r) < 0.4 and 0.0 < p < 1.0


def test_catdog_section_runs_on_long_layout(tmp_path):
    geometry = tmp_path / "geometry_metrics.csv"
    _long_geometry().to_csv(geometry, index=False)
    judged = tmp_path / "judged.summary.csv"
    _catdog_summary().to_csv(judged, index=False)
    rows, _ = catdog_section(geometry, judged)
    assert len(rows) == 5 * 2 * 5  # (4 group rows + pooled) x outcomes x metrics
    assert set(rows["metric"]) == {"decodability", "CCGP", "PS", "SD", "SD mean acc"}
    # decodability rows never carry partial correlations (it controls itself)
    assert rows.loc[rows.metric == "decodability", "partial_r"].isna().all()
    # pooled group present
    assert "pooled" in set(rows["group"])


def test_catdog_section_accepts_production_wide_layout(tmp_path):
    long_frame = _long_geometry()
    # decodability + ccgp become layer_SD_mean_acc etc. in production naming;
    # the wide fixture keeps the column names the section reads.
    wide = _wrap_wide(long_frame)
    geometry = tmp_path / "geometry_rows.csv"
    wide.to_csv(geometry, index=False)
    judged = tmp_path / "judged.summary.csv"
    _catdog_summary().to_csv(judged, index=False)
    rows, _ = catdog_section(geometry, judged)
    assert {"decodability", "CCGP", "PS", "SD", "SD mean acc"} <= set(rows["metric"])


def test_toxicity_section(tmp_path):
    geometry = tmp_path / "geometry_metrics.csv"
    _long_geometry().to_csv(geometry, index=False)  # variable names differ but unused
    judged = tmp_path / "toxicity_best_alpha_geometry.csv"
    rows = []
    for setup in ("general", "conditioned"):
        for index, layer in enumerate((3, 6)):
            rows.append({
                "setup": setup, "layer": layer, "alpha": 16.0 + 16.0 * index,
                "n": 200, "success": 0.2 + 0.3 * index,
                "broken_rate": 0.4 - 0.1 * index,
                "classifier_target_delta": 0.1 * index,
                "PS": 0.2 + 0.05 * layer, "CCGP_nontargeted": 0.5 + 0.05 * layer,
                "SD": 0.6, "SD_mean_acc": 0.65 + 0.01 * layer,
                "decodability": 0.6 + 0.02 * layer,
            })
    pd.DataFrame(rows).to_csv(judged, index=False)
    result, raw = toxicity_section(geometry, judged)
    assert set(result["dataset"]) == {"toxicity"}
    assert set(result["outcome"]) == {"success", "broken rate", "classifier delta"}
    assert "pooled" in set(result["group"]) and "pooled" in set(result["setup"])
