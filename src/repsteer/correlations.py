"""Full and partial correlations: geometry metrics vs steering outcomes.

Port of the production ``correlation_regression_report.py`` sections A
(style/catdog) and B (toxicity). Computes, for each (group, outcome, metric):

- ``full_r`` / ``full_p`` — Pearson r of metric vs outcome;
- ``partial_r`` / ``partial_p`` — partial r controlling the decodability/
  CCGP decode accuracy (the "gap" analysis);
- ``mirror_partial_r`` / ``mirror_partial_p`` — partial r of the control vs
  the outcome, controlling the metric;
- ``decodability_full_r`` / ``decodability_full_p`` — full r of the control
  vs the outcome;
- significance stars exactly like production: ``***`` p < 0.001, ``**``
  p < 0.01, ``*`` p < 0.05.

Inputs (produced by the other ``repsteer`` stages):
- ``geometry_metrics.csv`` — one row per (layer, variable/metric) from
  ``repsteer geometry`` (long format: ``metric``/``variable``/``value``
  rows from the decodanda engine, or per-variable columns).
- ``judged.csv`` — the judged steering table from ``repsteer judge`` (per-row
  records with judge preferences) and/or its ``judged.summary.csv``
  aggregate (one row per layer/alpha/condition).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError as exc:  # pragma: no cover - scipy ships with test extra
    raise RuntimeError("repsteer correlations requires scipy (pip install scipy)") from exc


# (label, column) outcome pairs, exactly as production.
STYLE_OUTCOMES = [
    ("success (mean over alpha)", "steering_preferred_rate_mean_alpha"),
    ("broken rate (mean over alpha)", "steered_broken_rate_mean_alpha"),
]
TOX_OUTCOMES = [
    ("success", "success"),
    ("broken rate", "broken_rate"),
    ("classifier delta", "classifier_target_delta"),
]
TASK_TO_FIELD = {"style": "style_bin", "concept": "is_dog"}
STYLE_METRICS = [
    ("decodability", "decodability_mean"),
    ("CCGP", "ccgp_mean"),
    ("PS", "PS"),
    ("SD", "SD"),
    ("SD mean acc", "SD_mean_acc"),
]
TOX_METRICS = [
    ("decodability", "decodability"),
    ("CCGP", "CCGP_nontargeted"),
    ("PS", "PS"),
    ("SD", "SD"),
    ("SD mean acc", "SD_mean_acc"),
]


def stars(p: float) -> str:
    """Production significance stars."""
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def full_r(x, y) -> tuple[float, float, int]:
    """Pearson r with production's finite/zero-variance guards."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan"), float("nan"), int(m.sum())
    r, p = stats.pearsonr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def partial_r(x, y, z) -> tuple[float, float, int]:
    """Partial correlation of x and y controlling z (Pearson)."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n = int(m.sum())
    if n < 4 or np.std(x[m]) == 0 or np.std(y[m]) == 0 or np.std(z[m]) == 0:
        return float("nan"), float("nan"), n
    rxz = np.corrcoef(x[m], z[m])[0, 1]
    ryz = np.corrcoef(y[m], z[m])[0, 1]
    den = (1 - rxz**2) * (1 - ryz**2)
    if den <= 0:
        return float("nan"), float("nan"), n
    r = (np.corrcoef(x[m], y[m])[0, 1] - rxz * ryz) / np.sqrt(den)
    df = n - 3
    t = abs(r) * np.sqrt(df / max(1 - r * r, 1e-12))
    p = float(2 * stats.t.sf(t, df))
    return float(r), p, n


def _row(**kw) -> dict:
    base = {
        "decodability_full_r": np.nan, "decodability_full_p": np.nan,
        "mirror_partial_r": np.nan, "mirror_partial_p": np.nan,
    }
    base.update(kw)
    return base


def _pivot_long_geometry(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce a geometry table to one row per (layer, variable) with
    PS/SD/SD_mean_acc/decodability_mean/ccgp_mean columns, mirroring the
    production geometry_rows.csv names.

    Accepts either layout:
    - long (the ``repsteer geometry`` decodanda output: rows with columns
      metric/variable/resampling/value; PS applies to every named variable,
      decodability to named dichotomies, sd_* summaries carry
      variable='condition_space');
    - wide (the production geometry_rows.csv: columns
      target_field/PS/CCGP_nontargeted/SD/SD_mean_acc).
    """
    raw = raw.copy()
    if "target_field" in raw.columns and "PS" in raw.columns:
        wide = raw.rename(columns={"target_field": "variable"})
        if "CCGP_nontargeted" in wide.columns and "ccgp_mean" not in wide.columns:
            wide["ccgp_mean"] = wide["CCGP_nontargeted"]
        keep = ["layer", "variable"]
        for column in ("PS", "CCGP_nontargeted", "SD", "SD_mean_acc",
                       "decodability_mean", "ccgp_mean"):
            if column in wide.columns:
                keep.append(column)
        wide = wide[keep].copy()
        wide["variable"] = wide["variable"].astype(str)
        wide["layer"] = wide["layer"].astype(int)
        return wide.drop_duplicates()
    if "target_field" in raw.columns and "ccgp_mean" in raw.columns:
        # Audit layout (matched_metrics_with_steering): no PS/SD columns;
        # named decoding/CCGP live in ccgp_mean/decodability_mean.
        wide = raw.rename(columns={"target_field": "variable"})
        keep = ["layer", "variable"]
        for column in ("decodability_mean", "ccgp_mean"):
            if column in wide.columns:
                keep.append(column)
        wide = wide[keep].copy()
        wide["variable"] = wide["variable"].astype(str)
        wide["layer"] = wide["layer"].astype(int)
        return wide.drop_duplicates()

    raw["variable"] = raw["variable"].astype(str)
    named = raw[raw["variable"] != "condition_space"]

    def agg(metric: str) -> pd.DataFrame:
        part = named[named["metric"] == metric]
        out = part.groupby(["layer", "variable"], as_index=False)["value"].mean()
        return out.rename(columns={"value": metric})

    blocks = []
    seen_layers = set(raw["layer"].unique())
    for metric in ("parallelism_score", "ccgp", "decodability", "sd_dichotomy"):
        part = agg(metric)
        if not part.empty:
            blocks.append(part)
    if blocks:
        merged = blocks[0]
        for block in blocks[1:]:
            merged = merged.merge(block, on=["layer", "variable"], how="outer")
    else:
        merged = pd.DataFrame(columns=["layer", "variable"])

    ps_col = "parallelism_score" if "parallelism_score" in merged.columns else "PS"
    merged = merged.rename(columns={ps_col: "PS", "ccgp": "ccgp_mean",
                                    "decodability": "decodability_mean"})

    sd_space = raw[raw["metric"] == "sd_mean_accuracy"]
    sd = sd_space.groupby("layer", as_index=False)["value"].mean().rename(
        columns={"value": "SD_mean_acc"})
    sd_frac = raw[raw["metric"] == "sd_fraction_above_threshold"]
    if not sd_frac.empty:
        sd = sd.merge(
            sd_frac.groupby("layer", as_index=False)["value"].mean().rename(
                columns={"value": "SD"}),
            on="layer", how="outer")
    if merged.empty:
        merged = sd.copy()
        merged["variable"] = np.nan
    else:
        # SD/SD_mean_acc are whole-condition-space quantities: broadcast to
        # every (layer, variable) group like the production geometry rows.
        merged = merged.merge(sd, on="layer", how="left")
    return merged


def catdog_section(metrics_path: str | Path, judged_summary_path: str | Path,
                   audit_metrics_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Production section A. ``judged_summary`` must carry, per
    (setup, task, layer): steering_preferred_rate_mean_alpha /
    steered_broken_rate_mean_alpha / best_alpha / best_success / best_broken
    plus decodability_mean/ccgp_mean columns (the audit's
    matched_metrics_with_steering layout). ``audit_metrics_path`` optionally
    supplies the matched-decoding table (decodability_mean/ccgp_mean per
    layer/target_field), mixed into the geometry table exactly as the
    production script does."""
    df = pd.read_csv(judged_summary_path)
    df = df[df.task.isin(TASK_TO_FIELD)].copy()  # language saturates several metrics
    df["group"] = df.setup + " " + df.task.map({"style": "style", "concept": "cat/dog"})

    geom = _pivot_long_geometry(pd.read_csv(metrics_path))
    if audit_metrics_path is not None:
        audit_piv = _pivot_long_geometry(pd.read_csv(audit_metrics_path))
        decode_cols = [column for column in ("decodability_mean", "ccgp_mean")
                       if column in audit_piv.columns and column not in geom.columns]
        if decode_cols:
            geom = geom.merge(audit_piv[["layer", "variable", *decode_cols]],
                              on=["layer", "variable"], how="left", validate="many_to_one")
    field_to_task = {value: key for key, value in TASK_TO_FIELD.items()}
    geom["task"] = geom["variable"].map(field_to_task)
    geom = geom.dropna(subset=["task"]).drop(columns=["variable"])
    geom = geom.drop_duplicates(["task", "layer"])
    overlap = [column for column in geom.columns if column in df.columns
               and column not in ("task", "layer")]
    geom = geom.drop(columns=overlap)  # judged input carries its own copy
    df = df.merge(geom, on=["task", "layer"], how="left", validate="many_to_one")
    # Decodability/CCGP decode metrics: the audit's matched-decoding file
    # (decodability_mean/ccgp_mean per layer/target_field) is the canonical
    # source in production. Accept --audit for it; fall back to whatever the
    # geometry table itself carries (also the case for the toolkit's long
    # decodanda output, which reports both).

    available = {(metric, mcol) for metric, mcol in STYLE_METRICS if mcol in df.columns}
    rows = []
    for group, g in list(df.groupby("group", sort=True)) + [("pooled", df)]:
        for outcome, ocol in STYLE_OUTCOMES:
            for metric, mcol in STYLE_METRICS:
                if (metric, mcol) not in available:
                    continue
                fr, fp, n = full_r(g[mcol], g[ocol])
                if metric == "decodability" or "decodability_mean" not in g.columns:
                    pr = pp = mr = mp = float("nan")
                    if metric == "decodability":
                        # metric == decodability implies the column is present
                        dfr, dfp, _ = full_r(g["decodability_mean"], g[ocol])
                    else:
                        dfr = dfp = float("nan")
                else:
                    pr, pp, _ = partial_r(g[mcol], g[ocol], g["decodability_mean"])
                    mr, mp, _ = partial_r(g["decodability_mean"], g[ocol], g[mcol])
                    dfr, dfp, _ = full_r(g["decodability_mean"], g[ocol])
                rows.append(_row(dataset="style/catdog", setup=g["setup"].iloc[0],
                                 task=g["task"].iloc[0], group=group,
                                 outcome=outcome, metric=metric, n=n,
                                 full_r=fr, full_p=fp, partial_r=pr, partial_p=pp,
                                 mirror_partial_r=mr, mirror_partial_p=mp,
                                 decodability_full_r=dfr, decodability_full_p=dfp))
    return pd.DataFrame(rows), df


def toxicity_section(metrics_path: str | Path, judged_summary_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Production section B. ``judged_summary`` must carry, per
    (setup, layer): success / broken_rate / classifier_target_delta plus the
    PS/CCGP_nontargeted/SD/SD_mean_acc/decodability columns (the
    toxicity_best_alpha_geometry layout)."""
    df = pd.read_csv(judged_summary_path)
    available = {(metric, mcol) for metric, mcol in TOX_METRICS if mcol in df.columns}
    rows = []
    for setup, g in list(df.groupby("setup", sort=True)) + [("pooled", df)]:
        for outcome, ocol in TOX_OUTCOMES:
            for metric, mcol in TOX_METRICS:
                if (metric, mcol) not in available:
                    continue
                fr, fp, n = full_r(g[mcol], g[ocol])
                if metric == "decodability" or "decodability" not in g.columns:
                    pr = pp = mr = mp = float("nan")
                    if metric == "decodability":
                        dfr, dfp, _ = full_r(g["decodability"], g[ocol])
                    else:
                        dfr = dfp = float("nan")
                else:
                    pr, pp, _ = partial_r(g[mcol], g[ocol], g["decodability"])
                    mr, mp, _ = partial_r(g["decodability"], g[ocol], g[mcol])
                    dfr, dfp, _ = full_r(g["decodability"], g[ocol])
                rows.append(_row(dataset="toxicity", setup=setup, task=setup,
                                 group=setup, outcome=outcome, metric=metric, n=n,
                                 full_r=fr, full_p=fp, partial_r=pr, partial_p=pp,
                                 mirror_partial_r=mr, mirror_partial_p=mp,
                                 decodability_full_r=dfr, decodability_full_p=dfp))
    return pd.DataFrame(rows), df[df.setup != "pooled"].copy()


def _fmt(r: float, p: float) -> str:
    if not np.isfinite(r):
        return "—"
    if np.isfinite(p):
        return f"{r:+.3f}{stars(p)} (p={p:.3f})"
    return f"{r:+.3f}"


def _markdown(kind: str, rows: pd.DataFrame) -> str:
    lines = [f"## {kind}", "",
             "| dataset | group | outcome | metric | n | full r | partial r | mirror partial r |",
             "|---|---|---|---|---:|---|---|---|"]
    for _, row in rows.iterrows():
        lines.append(
            "| {dataset} | {group} | {outcome} | {metric} | {n} | {full} | {partial} | {mirror} |".format(
                dataset=row["dataset"], group=row["group"], outcome=row["outcome"],
                metric=row["metric"], n=row["n"],
                full=_fmt(row["full_r"], row["full_p"]),
                partial=_fmt(row["partial_r"], row["partial_p"]),
                mirror=_fmt(row["mirror_partial_r"], row["mirror_partial_p"]),
            ))
    return "\n".join(lines)


def run_correlations(datASET: str, metrics_path: str | Path, judged_path: str | Path,
                     output_dir: str | Path,
                     audit_metrics_path: str | Path | None = None) -> pd.DataFrame:
    if datASET == "toxicity":
        rows, _ = toxicity_section(metrics_path, judged_path)
    else:
        rows, _ = catdog_section(metrics_path, judged_path, audit_metrics_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "correlation_report.csv", index=False)
    kind = "Toxicity: geometry vs steering outcomes" if datASET == "toxicity" \
        else "Style/catdog: geometry vs steering outcomes"
    (output / "correlation_report.md").write_text(
        _markdown(kind, rows) + "\n", encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="repsteer correlations",
        description="Full + partial geometry-vs-outcome correlations with production stars")
    parser.add_argument("--dataset", required=True, choices=("catdog", "toxicity"),
                        help="Which production section to run")
    parser.add_argument("--geometry", required=True, help="geometry_metrics.csv path")
    parser.add_argument("--judged", required=True,
                        help="Per-(setup,layer) outcomes CSV: judged.summary.csv "
                             "aggregated with geometry matched, or "
                             "toxicity_best_alpha_geometry.csv-style table")
    parser.add_argument("--audit-metrics", default=None,
                        help="Optional matched-decoding table "
                             "(decodability_mean/ccgp_mean per layer/target_field) mixed in "
                             "exactly as the production style/cat-dog audit does")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    rows = run_correlations(args.dataset, args.geometry, args.judged, args.output_dir,
                            audit_metrics_path=args.audit_metrics)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2))
