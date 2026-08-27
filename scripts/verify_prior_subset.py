#!/usr/bin/env python3
"""Audit RepSteer postprocessing against a deterministic subset of prior results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from repsteer.offtarget import summarize_offtarget_effects, summarize_projection_drift


def _sample_groups(frame: pd.DataFrame, keys: list[str], n: int, seed: int) -> pd.DataFrame:
    unique = frame[keys].drop_duplicates().reset_index(drop=True)
    return unique.sample(n=min(n, len(unique)), random_state=seed).sort_values(keys)


def _metric_comparison(old: pd.DataFrame, new: pd.DataFrame, keys: list[str],
                       metrics: list[str], family: str) -> pd.DataFrame:
    merged = old[keys + metrics].merge(new[keys + metrics], on=keys,
                                        suffixes=("_old", "_new"), validate="one_to_one")
    rows = []
    for metric in metrics:
        difference = (merged[f"{metric}_new"] - merged[f"{metric}_old"]).abs()
        rows.append({"family": family, "metric": metric, "n_compared": len(merged),
                     "max_absolute_error": float(difference.max()),
                     "mean_absolute_error": float(difference.mean()),
                     "exact_or_1e-12": bool((difference <= 1e-12).all())})
    return pd.DataFrame(rows)


def audit(repo: Path, output: Path, groups: int = 24, seed: int = 42) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    comparisons = []

    # Projection: old summaries intentionally aggregate over alpha.
    projection_dir = repo / "results-gemma4-online-offtarget-projection-analysis"
    raw_projection = pd.read_csv(projection_dir / "online_projection_rows.csv")
    old_projection = pd.read_csv(projection_dir / "online_projection_summary.csv").rename(columns={
        "steering_task": "steering_variable", "measured_task": "measured_variable",
        "n": "n_examples", "mean_abs_drift": "mean_absolute_projection_drift",
        "mean_normalized_abs_drift": "mean_normalized_absolute_drift"})
    projection_keys = ["setup", "steering_variable", "measured_variable", "is_off_target",
                       "intervention_layer", "readout_layer"]
    chosen = _sample_groups(old_projection, projection_keys, groups, seed)
    old_subset = old_projection.merge(chosen, on=projection_keys, validate="one_to_one")
    adapted = raw_projection.rename(columns={
        "steering_task": "steering_variable", "measured_task": "measured_variable",
        "candidate_id": "sample_id"}).drop(columns=["alpha"])
    projection_rows = adapted.merge(chosen, on=projection_keys, validate="many_to_one")
    new_projection = summarize_projection_drift(projection_rows)
    projection_metrics = ["n_examples", "mean_absolute_projection_drift",
                          "mean_normalized_absolute_drift", "mean_source_opposed_drift"]
    comparisons.append(_metric_comparison(old_subset, new_projection, projection_keys,
                                           projection_metrics, "projection_summary"))
    row_checks = pd.DataFrame({
        "projection_delta_error": (projection_rows.projection_delta -
            (projection_rows.steered_projection - projection_rows.baseline_projection)).abs(),
        "absolute_drift_error": (projection_rows.absolute_projection_drift -
            projection_rows.projection_delta.abs()).abs(),
        "normalized_drift_error": (projection_rows.normalized_absolute_drift -
            projection_rows.absolute_projection_drift / projection_rows.class_gap).abs()})
    for metric in row_checks:
        comparisons.append(pd.DataFrame([{
            "family": "projection_rows", "metric": metric,
            "n_compared": len(row_checks), "max_absolute_error": float(row_checks[metric].max()),
            "mean_absolute_error": float(row_checks[metric].mean()),
            "exact_or_1e-12": bool((row_checks[metric] <= 1e-12).all())}]))
    projection_rows.to_csv(output / "projection_subset_rows.csv", index=False)
    old_subset.to_csv(output / "projection_old_summary_subset.csv", index=False)
    new_projection.to_csv(output / "projection_new_summary_subset.csv", index=False)

    # Off-target LLM effects: sample complete setup/task/variable/layer/alpha groups.
    off_dir = repo / "results-gemma4-targeted-alpha-offtarget-llm"
    raw_off = pd.read_csv(off_dir / "offtarget_llm_effect_rows.csv")
    old_off = pd.read_csv(off_dir / "offtarget_llm_effect_summary.csv").rename(columns={
        "steered_task": "steered_variable", "offtarget_variable": "measured_variable",
        "alpha": "coefficient", "n": "n_examples",
        "off_target_change_rate": "changed_vs_baseline_rate",
        "off_target_drift_rate": "drift_from_source_rate",
        "broken_rate": "steered_broken_rate"})
    off_keys = ["steering_setup", "steered_variable", "measured_variable", "layer", "coefficient"]
    chosen_off = _sample_groups(old_off, off_keys, groups, seed + 1)
    old_off_subset = old_off.merge(chosen_off, on=off_keys, validate="one_to_one")
    adapted_off = raw_off.rename(columns={
        "steered_task": "steered_variable", "offtarget_variable": "measured_variable",
        "alpha": "coefficient", "llm_broken_steered": "steered_broken",
        "llm_broken_baseline": "baseline_broken"})
    adapted_off["is_off_target"] = True
    adapted_off["judge_parse_error"] = np.where(
        adapted_off.valid_offtarget_parse, "", "legacy parse error")
    off_rows = adapted_off.merge(chosen_off, on=off_keys, validate="many_to_one")
    new_off = summarize_offtarget_effects(off_rows)
    off_metrics = ["n_examples", "changed_vs_baseline_rate", "drift_from_source_rate",
                   "steered_broken_rate", "parse_error_rate"]
    expected_off = old_off_subset.copy()
    expected_off["parse_error_rate"] = 1.0 - expected_off.valid_parse_rate
    comparisons.append(_metric_comparison(expected_off, new_off, off_keys,
                                           off_metrics, "offtarget_judge_summary"))
    off_rows.to_csv(output / "offtarget_subset_rows.csv", index=False)
    old_off_subset.to_csv(output / "offtarget_old_summary_subset.csv", index=False)
    new_off.to_csv(output / "offtarget_new_summary_subset.csv", index=False)

    # Capability deltas are recomputed from the stored baseline on sampled interventions.
    mmmlu = pd.read_csv(repo / "results-gemma4-mmmlu-if-100/figures/mmmlu_scores_by_intervention.csv")
    baseline_accuracy = float(mmmlu.loc[mmmlu.run_type == "baseline", "accuracy"].iloc[0])
    mmmlu_subset = mmmlu[mmmlu.run_type == "steered"].sample(
        n=min(groups, int((mmmlu.run_type == "steered").sum())), random_state=seed)
    mmmlu_subset = mmmlu_subset.assign(
        recomputed_delta_accuracy=mmmlu_subset.accuracy - baseline_accuracy,
        absolute_error=lambda x: (x.recomputed_delta_accuracy - x.delta_accuracy).abs())
    comparisons.append(pd.DataFrame([{
        "family": "mmmlu", "metric": "delta_accuracy", "n_compared": len(mmmlu_subset),
        "max_absolute_error": float(mmmlu_subset.absolute_error.max()),
        "mean_absolute_error": float(mmmlu_subset.absolute_error.mean()),
        "exact_or_1e-12": bool((mmmlu_subset.absolute_error <= 1e-12).all())}]))
    mmmlu_subset.to_csv(output / "mmmlu_subset_comparison.csv", index=False)

    mifeval = pd.read_csv(repo / "results-gemma4-mifeval-if-100/figures/mifeval_scores_by_intervention.csv")
    baseline_if = mifeval[mifeval.run_type == "baseline"].iloc[0]
    mifeval_subset = mifeval[mifeval.run_type == "steered"].sample(
        n=min(groups, int((mifeval.run_type == "steered").sum())), random_state=seed)
    if_metrics = ["strict_prompt_accuracy", "strict_instruction_accuracy",
                  "loose_prompt_accuracy", "loose_instruction_accuracy"]
    for metric in if_metrics:
        recalculated = mifeval_subset[metric] - baseline_if[metric]
        error = (recalculated - mifeval_subset[f"delta_{metric}"]).abs()
        mifeval_subset[f"recomputed_delta_{metric}"] = recalculated
        mifeval_subset[f"error_{metric}"] = error
        comparisons.append(pd.DataFrame([{
            "family": "mifeval", "metric": f"delta_{metric}",
            "n_compared": len(mifeval_subset), "max_absolute_error": float(error.max()),
            "mean_absolute_error": float(error.mean()),
            "exact_or_1e-12": bool((error <= 1e-12).all())}]))
    mifeval_subset.to_csv(output / "mifeval_subset_comparison.csv", index=False)

    report = pd.concat(comparisons, ignore_index=True)
    report.to_csv(output / "parity_report.csv", index=False)
    summary = {"seed": seed, "requested_groups_per_family": groups,
               "checks": len(report), "all_passed": bool(report["exact_or_1e-12"].all()),
               "maximum_absolute_error": float(report.max_absolute_error.max()),
               "families": report.groupby("family").n_compared.max().astype(int).to_dict()}
    (output / "parity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("parity_artifacts"))
    parser.add_argument("--groups", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = audit(args.repo.resolve(), args.output.resolve(), args.groups, args.seed)
    print(report.to_string(index=False))
    if not report["exact_or_1e-12"].all():
        raise SystemExit("Subset parity audit failed")


if __name__ == "__main__":
    main()
