"""Judge-row aggregation to the production outcome tables.

* ``catdog`` reproduces ``caa_model_search.report_mode`` (per task/layer/alpha
  rates; brokenness is averaged over rows whose judge returned a label) followed
  by ``analyze_gemma4_eval200_story.layer_summary`` -> the
  ``confirmatory_layer_summary.csv`` layout (n-weighted mean over alpha plus the
  best alpha by success desc, broken asc, alpha asc).
* ``toxicity`` reproduces ``analyze_gemma4_toxicity_replication.py``: string
  truth-casting (an unlabelled brokenness counts as not broken), ``n`` = unique
  ids, and the best alpha per (setup, layer) by success desc, broken asc, alpha
  asc -> ``toxicity_summary_by_layer_alpha.csv`` / ``toxicity_best_alpha``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

RATE_METRICS = ("steering_preferred_rate", "tie_rate", "steered_broken_rate",
                "baseline_broken_rate", "valid_decision_rate")


def _truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _labelled(series: pd.Series) -> pd.Series:
    """True/False labels as 1.0/0.0, anything else (missing label) as NaN."""

    text = series.astype(str).str.lower()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[text.isin(["true", "1", "yes", "1.0"])] = 1.0
    out[text.isin(["false", "0", "no", "0.0"])] = 0.0
    return out


def summarize_by_alpha(frame: pd.DataFrame, group_columns=("setup", "task", "layer", "coefficient"),
                       ) -> pd.DataFrame:
    """``report_mode`` summary: one row per group with n and the five rates."""

    data = pd.DataFrame({column: frame[column] for column in group_columns})
    data["steering_preferred"] = _truth(frame["steering_preferred"]).astype(float)
    data["tie"] = _truth(frame["tie"]).astype(float)
    data["judge_broken_steered"] = _labelled(frame["judge_broken_steered"])
    data["judge_broken_baseline"] = _labelled(frame["judge_broken_baseline"])
    data["valid_decision"] = _truth(frame["valid_decision"]).astype(float)
    return data.groupby(list(group_columns), dropna=False, sort=True).agg(
        n=("steering_preferred", "size"),
        steering_preferred_rate=("steering_preferred", "mean"),
        tie_rate=("tie", "mean"),
        steered_broken_rate=("judge_broken_steered", "mean"),
        baseline_broken_rate=("judge_broken_baseline", "mean"),
        valid_decision_rate=("valid_decision", "mean"),
    ).reset_index()


def _weighted_mean(group: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
    weights = pd.to_numeric(group["n"], errors="coerce").to_numpy(float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[keep], weights=weights[keep])) if keep.any() else np.nan


def layer_summary(summary: pd.DataFrame, alpha_column: str = "coefficient") -> pd.DataFrame:
    """``analyze_gemma4_eval200_story.layer_summary`` over every setup in ``summary``."""

    rows = []
    # Production concatenates the general table before the conditioned one.
    for setup in sorted(pd.unique(summary["setup"]), key=lambda name: (name != "general", name)):
        frame = summary[summary["setup"] == setup]
        for (task, layer), group in frame.groupby(["task", "layer"]):
            row = {"setup": setup, "task": task, "layer": int(layer),
                   "n_config_observations": int(group["n"].sum())}
            for metric in RATE_METRICS:
                row[f"{metric}_mean_alpha"] = _weighted_mean(group, metric)
            best = group.sort_values(["steering_preferred_rate", "steered_broken_rate", alpha_column],
                                     ascending=[False, True, True]).iloc[0]
            row.update({"best_alpha": float(best[alpha_column]),
                        "best_success": float(best["steering_preferred_rate"]),
                        "best_broken": float(best["steered_broken_rate"])})
            rows.append(row)
    return pd.DataFrame(rows)


def _digest(text) -> str:
    return hashlib.sha256(str(text).encode()).hexdigest()


def toxicity_summary(frame: pd.DataFrame, alpha_column: str = "coefficient",
                     id_column: str = "candidate_id",
                     classifier_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """``toxicity_summary_by_layer_alpha.csv``: setup, layer, alpha, n, success,
    broken_rate, tie_rate[, classifier_target_delta].

    ``classifier_scores`` (columns ``text_hash`` = sha256 of the continuation,
    ``toxicity_probability``) adds the XLM-R target-aligned delta.
    """

    data = pd.DataFrame({"setup": frame["setup"], "layer": frame["layer"],
                         "alpha": frame[alpha_column], "id": frame[id_column]})
    data["steering_preferred"] = _truth(frame["steering_preferred"])
    data["judge_broken_steered"] = _truth(frame["judge_broken_steered"])
    data["tie"] = _truth(frame["tie"])
    aggregations = dict(n=("id", "nunique"), success=("steering_preferred", "mean"),
                        broken_rate=("judge_broken_steered", "mean"), tie_rate=("tie", "mean"))
    if classifier_scores is not None:
        score_map = dict(zip(classifier_scores.text_hash, classifier_scores.toxicity_probability))
        baseline = frame["baseline_continuation"].map(lambda text: score_map[_digest(text)])
        steered = frame["steered_continuation"].map(lambda text: score_map[_digest(text)])
        raw_delta = steered - baseline
        data["delta"] = np.where(frame["target_attribute"].eq("toxic"), raw_delta, -raw_delta)
        aggregations["classifier_target_delta"] = ("delta", "mean")
    return data.groupby(["setup", "layer", "alpha"], as_index=False).agg(**aggregations)


def toxicity_best_alpha(summary: pd.DataFrame) -> pd.DataFrame:
    """Best alpha per (setup, layer): success desc, broken_rate asc, alpha asc."""

    ordered = summary.sort_values(["setup", "layer", "success", "broken_rate", "alpha"],
                                  ascending=[True, True, False, True, True])
    return ordered.groupby(["setup", "layer"]).head(1)


def toxicity_best_alpha_geometry(best: pd.DataFrame, geometry_rows: pd.DataFrame,
                                  matched_metrics: pd.DataFrame,
                                  target_field: str = "toxicity_bin") -> pd.DataFrame:
    """``toxicity_best_alpha_geometry.csv``: best-alpha outcomes joined with the
    target's PS/CCGP/SD (geometry rows) and matched decodability, as production."""

    geometry = geometry_rows[geometry_rows.target_field.eq(target_field)][
        ["layer", "PS", "CCGP_nontargeted", "SD", "SD_mean_acc"]].drop_duplicates()
    decodability = matched_metrics[matched_metrics.target_field.eq(target_field)][
        ["layer", "decodability_mean"]].rename(columns={"decodability_mean": "decodability"})
    return best.merge(geometry, on="layer").merge(decodability, on="layer")


def generic_summary(frame: pd.DataFrame, group_columns=("model_id", "concept", "setup", "layer"),
                    alpha_column: str = "coefficient", id_column: str | None = None,
                    passthrough: tuple[str, ...] = ("relative_depth", "alpha_rel")) -> pd.DataFrame:
    """Dataset-agnostic per-alpha outcome table for the scaling study.

    One row per (``group_columns`` + present passthrough columns + ``alpha_column``)
    with ``n`` and the production rates: ``steering_preferred_rate`` (success),
    ``tie_rate``, ``steered_broken_rate`` / ``baseline_broken_rate`` (over rows
    carrying a brokenness label), ``valid_decision_rate``. ``id_column`` (e.g.
    ``candidate_id``) makes ``n`` count unique ids instead of rows.
    """

    missing = [column for column in (*group_columns, alpha_column, "steering_preferred",
                                     "tie", "judge_broken_steered", "judge_broken_baseline",
                                     "valid_decision") if column not in frame.columns]
    if missing:
        raise KeyError(f"Judged table lacks required columns: {missing}")
    data = pd.DataFrame({column: frame[column] for column in (*group_columns, alpha_column)})
    for name in passthrough:
        if name in frame.columns and name != alpha_column:
            data[name] = frame[name]
    for column in (*group_columns, *passthrough):
        if column in data and data[column].dtype == object:
            data[column] = data[column].astype(str)
    data["valid_decision"] = _truth(frame["valid_decision"]).astype(float)
    # Unparsed judgements are missing, not losses: success/tie rates are over valid
    # decisions only (valid_decision_rate reports the missing share).
    valid = data["valid_decision"] == 1.0
    data["steering_preferred"] = _truth(frame["steering_preferred"]).astype(float).where(valid)
    data["tie"] = _truth(frame["tie"]).astype(float).where(valid)
    data["judge_broken_steered"] = _labelled(frame["judge_broken_steered"])
    data["judge_broken_baseline"] = _labelled(frame["judge_broken_baseline"])
    aggregations = dict(
        steering_preferred_rate=("steering_preferred", "mean"),
        tie_rate=("tie", "mean"),
        steered_broken_rate=("judge_broken_steered", "mean"),
        baseline_broken_rate=("judge_broken_baseline", "mean"),
        valid_decision_rate=("valid_decision", "mean"),
    )
    if id_column is not None:
        data[id_column] = frame[id_column].astype(str)
        aggregations = {"n": (id_column, "nunique"), **aggregations}
    else:
        aggregations = {"n": ("steering_preferred", "size"), **aggregations}
    keys = [column for column in data.columns
            if column not in ("steering_preferred", "tie", "judge_broken_steered",
                              "judge_broken_baseline", "valid_decision")]
    return data.groupby(keys, dropna=False, sort=True).agg(**aggregations).reset_index()


def generic_layer_summary(summary: pd.DataFrame,
                          group_columns=("model_id", "concept", "setup", "layer"),
                          alpha_column: str = "coefficient",
                          passthrough: tuple[str, ...] = ("relative_depth",)) -> pd.DataFrame:
    """Mean-over-alpha plus best-alpha outcomes per (``group_columns``, passthrough).

    ``summary`` is a :func:`generic_summary` table. For every group this reports the
    n-weighted mean over the alpha levels (``*_mean_alpha``) and the best alpha by
    success desc, broken rate asc, alpha asc (``best_alpha`` / ``best_success`` /
    ``best_broken``). ``relative_depth`` is part of the group key (the steering grid
    samples one relative depth per layer), while other passthrough columns such as
    ``alpha_rel`` describe the best alpha and are attached to it.
    """

    group_keys = list(group_columns)
    depth_key = "relative_depth" if "relative_depth" in summary.columns else None
    if depth_key:
        group_keys.append(depth_key)
    carried = [column for column in passthrough if column in summary.columns
               and column not in group_keys]
    rows = []
    for key_values, group in summary.groupby(group_keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(group_keys, key_values))
        row["n_config_observations"] = int(pd.to_numeric(group["n"]).sum())
        for metric in RATE_METRICS:
            row[f"{metric}_mean_alpha"] = _weighted_mean(group, metric)
        best = group.sort_values(["steering_preferred_rate", "steered_broken_rate", alpha_column],
                                 ascending=[False, True, True]).iloc[0]
        row.update({"best_alpha": float(best[alpha_column]),
                    "best_success": float(best["steering_preferred_rate"]),
                    "best_broken": float(best["steered_broken_rate"])})
        for column in carried:
            if column == "alpha_rel":
                # alpha_rel describes the best alpha itself when present.
                row[column] = best.get(column, np.nan)
            else:
                row[column] = best[column]
        rows.append(row)
    return pd.DataFrame(rows)


def run_generic_aggregate(frame: pd.DataFrame, output_dir: str | Path,
                          group_columns: Sequence[str] = ("model_id", "concept", "setup", "layer"),
                          alpha_column: str = "coefficient", id_column: str | None = None) -> dict[str, pd.DataFrame]:
    """Write the generic scaling-study outcome tables and return them."""

    passthrough = tuple(name for name in ("relative_depth", "alpha_rel") if name in frame.columns)
    summary = generic_summary(frame, group_columns, alpha_column, id_column, passthrough)
    layers = generic_layer_summary(summary, group_columns, alpha_column, passthrough)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "generic_summary_by_alpha.csv", index=False)
    layers.to_csv(output / "generic_layer_summary.csv", index=False)
    return {"summary": summary, "layers": layers}


def run_aggregate(dataset: str, judged_paths: list[str | Path], output_dir: str | Path,
                  alpha_column: str = "coefficient",
                  classifier_scores_path: str | Path | None = None,
                  geometry_rows_path: str | Path | None = None,
                  matched_metrics_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
    frame = pd.concat([pd.read_csv(path, keep_default_na=False) for path in judged_paths],
                      ignore_index=True)
    frame["layer"] = pd.to_numeric(frame["layer"])
    frame[alpha_column] = pd.to_numeric(frame[alpha_column])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if dataset == "toxicity":
        scores = pd.read_csv(classifier_scores_path) if classifier_scores_path else None
        summary = toxicity_summary(frame, alpha_column, classifier_scores=scores)
        best = toxicity_best_alpha(summary)
        summary.to_csv(output / "toxicity_summary_by_layer_alpha.csv", index=False)
        best.to_csv(output / "toxicity_best_alpha.csv", index=False)
        tables = {"summary": summary, "best_alpha": best}
        if geometry_rows_path and matched_metrics_path:
            merged = toxicity_best_alpha_geometry(
                best, pd.read_csv(geometry_rows_path), pd.read_csv(matched_metrics_path))
            merged.to_csv(output / "toxicity_best_alpha_geometry.csv", index=False)
            tables["best_alpha_geometry"] = merged
        return tables
    summary = summarize_by_alpha(frame, ("setup", "task", "layer", alpha_column))
    layers = layer_summary(summary, alpha_column)
    summary.to_csv(output / "summary_by_setup_task_layer_alpha.csv", index=False)
    layers.to_csv(output / "confirmatory_layer_summary.csv", index=False)
    return {"summary": summary, "layers": layers}
