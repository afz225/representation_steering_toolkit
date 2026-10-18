"""Pre-registered scaling-study analysis (``scaling_study/PREREGISTRATION.md``).

Unit: one row per (model_id, concept, layer). :func:`build_cells` joins the judged
outcomes (:func:`repsteer.aggregate.generic_layer_summary`) with the geometry of the
concept's target field: production rows (PS, SD, SD mean acc) and matched-audit rows
(decodability, CCGP). :func:`run_analysis` then fits

primary
    for each outcome Y in {success, broken} (mean over alpha) and each metric
    M in {CCGP, PS, SD, SD mean acc}::

        base: Y ~ decod_w + d + d^2 + (1|model) + (1|concept)
        full: Y ~ decod_w + M_w + d + d^2 + (1 + M_w|model) + (1 + M_w|concept)

    ``_w`` = z-scored within each (model, concept) cell across layers; d is the
    realized relative depth (centred). ML fits (:mod:`repsteer.mixed`, crossed random
    effects with unstructured per-term covariance); likelihood-ratio test with the
    naive chi-square p and the boundary-mixture p; Holm over the 8 tests. A full model
    that fails to converge drops the concept slope, then the model slope.
secondary
    1. within-cell partial r (controls decod, d, d^2) pooled by REML random effects;
    2. between-cell OLS of cell means with family and concept fixed effects;
    3. all metrics together (drop-one LRTs) with VIFs;
    4. moderator interactions (family, log10 size, MoE, abstractness);
    5. the primary analysis on best-alpha outcomes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .mixed import MixedFit, RandomTerm, fit_mixed, marginal_r2

try:
    import patsy
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:  # pragma: no cover - exercised through _require_statsmodels
    patsy = None

METRICS = {"CCGP": "ccgp", "PS": "ps", "SD": "sd", "SD mean acc": "sd_mean_acc"}
DECODABILITY = "decodability"
OUTCOMES = {"success": "success", "broken": "broken"}
BEST_ALPHA_OUTCOMES = {"success": "success_best", "broken": "broken_best"}
CELL = ["model_id", "concept"]
OUTCOME_SD_MIN = 0.02
COLLINEAR_R = 0.95
MISSING_FLAG = 0.05
MODERATORS = {
    "family": "C(family)",
    "log10_size": "log10_size_c",
    "moe": "moe",
    "abstractness": "C(abstractness)",
}


def _require_statsmodels() -> None:
    if patsy is None:
        raise RuntimeError("repsteer analyze needs statsmodels (pip install 'repsteer[analysis]')")


# --------------------------------------------------------------------------- #
# Cells table
# --------------------------------------------------------------------------- #

def _target_rows(frame: pd.DataFrame, target_field: str, columns: dict[str, str],
                 source: str) -> pd.DataFrame:
    rows = frame[frame["target_field"].astype(str) == target_field]
    if rows.empty:
        raise ValueError(f"{source}: no rows for target_field {target_field!r}")
    keep = ["model_id", "layer", *columns] + (["relative_depth"] if "relative_depth" in rows else [])
    rows = rows[keep].rename(columns=columns).copy()
    rows["model_id"] = rows["model_id"].astype(str)
    rows["layer"] = rows["layer"].astype(int)
    duplicated = rows.duplicated(["model_id", "layer"])
    if duplicated.any():
        raise ValueError(f"{source}: several rows per (model_id, layer) for {target_field!r} "
                         f"(e.g. several poolings): {rows.loc[duplicated, ['model_id', 'layer']].values[:5].tolist()}")
    return rows


def build_cells(layer_summary: pd.DataFrame, geometry_rows: pd.DataFrame,
                matched_metrics: pd.DataFrame, concept: str, target_field: str) -> pd.DataFrame:
    """Join one concept's outcomes with its target field's geometry per (model_id, layer).

    ``layer_summary``: :func:`generic_layer_summary` rows (any concepts; filtered to
    ``concept``). ``geometry_rows`` / ``matched_metrics``: ``repsteer geometry``
    outputs (production / matched_audit protocol) of the concept's condition space,
    possibly concatenated over models. Raises when any outcome layer lacks geometry
    or when realized relative depths disagree.
    """

    outcomes = layer_summary[layer_summary["concept"].astype(str) == concept]
    if outcomes.empty:
        raise ValueError(f"layer summary has no rows for concept {concept!r}")
    if outcomes.duplicated(["model_id", "layer"]).any():
        raise ValueError(f"{concept}: several layer-summary rows per (model_id, layer) (several setups?)")
    cells = pd.DataFrame({
        "model_id": outcomes["model_id"].astype(str).to_numpy(),
        "concept": concept,
        "target_field": target_field,
        "layer": outcomes["layer"].astype(int).to_numpy(),
        "relative_depth": pd.to_numeric(outcomes["relative_depth"]).to_numpy(),
        "success": pd.to_numeric(outcomes["steering_preferred_rate_mean_alpha"]).to_numpy(),
        "broken": pd.to_numeric(outcomes["steered_broken_rate_mean_alpha"]).to_numpy(),
        "success_best": pd.to_numeric(outcomes["best_success"]).to_numpy(),
        "broken_best": pd.to_numeric(outcomes["best_broken"]).to_numpy(),
        "valid_rate": pd.to_numeric(outcomes["valid_decision_rate_mean_alpha"]).to_numpy(),
    })
    production = _target_rows(geometry_rows, target_field,
                              {"PS": "ps", "SD": "sd", "SD_mean_acc": "sd_mean_acc"}, "geometry rows")
    audit = _target_rows(matched_metrics, target_field,
                         {"decodability_mean": DECODABILITY, "ccgp_mean": "ccgp"}, "matched metrics")
    for name, geometry in (("geometry rows", production), ("matched metrics", audit)):
        geometry_depth = geometry.pop("relative_depth") if "relative_depth" in geometry else None
        cells = cells.merge(geometry, on=["model_id", "layer"], how="left", validate="one_to_one")
        if geometry_depth is not None:
            depth = cells[["model_id", "layer"]].merge(
                geometry[["model_id", "layer"]].assign(depth=geometry_depth.to_numpy()),
                on=["model_id", "layer"], how="left")["depth"].to_numpy()
            mismatch = np.isfinite(depth) & (np.abs(depth - cells["relative_depth"].to_numpy()) > 1e-9)
            if mismatch.any():
                raise ValueError(f"{concept}: {name} relative_depth disagrees with steering rows at "
                                 f"{cells.loc[mismatch, ['model_id', 'layer']].values[:5].tolist()}")
    metric_columns = [DECODABILITY, *METRICS.values()]
    missing = cells[metric_columns].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{concept}: no geometry for (model_id, layer) "
                         f"{cells.loc[missing, ['model_id', 'layer']].values[:10].tolist()}")
    return cells


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def within_cell_z(cells: pd.DataFrame, column: str) -> pd.Series:
    """z-score ``column`` within each (model_id, concept) cell (NaN when constant)."""

    groups = cells.groupby(CELL, sort=False)[column]
    std = groups.transform(lambda values: values.std(ddof=0))
    return (cells[column] - groups.transform("mean")) / std.where(std > 0)


def outcome_exclusions(cells: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Cells whose outcome SD across layers is below 0.02 (pre-registered exclusion)."""

    sd = cells.groupby(CELL)[outcome].std(ddof=1).rename("outcome_sd").reset_index()
    excluded = sd[~(sd["outcome_sd"] >= OUTCOME_SD_MIN)].copy()
    excluded["outcome"] = outcome
    return excluded


def missing_flags(cells: pd.DataFrame) -> pd.DataFrame:
    """Cells with more than 5% missing (unparsed) judgements (flagged, not excluded)."""

    missing = (1.0 - cells.groupby(CELL)["valid_rate"].mean()).rename("missing_rate").reset_index()
    return missing[missing["missing_rate"] > MISSING_FLAG]


def _prepare(cells: pd.DataFrame, outcome: str) -> pd.DataFrame:
    data = cells.copy()
    excluded = outcome_exclusions(data, outcome)
    if len(excluded):
        drop = data.set_index(CELL).index.isin(excluded.set_index(CELL).index)
        data = data[~drop]
    data["y"] = data[outcome]
    data["decod_w"] = within_cell_z(data, DECODABILITY)
    for column in METRICS.values():
        data[f"{column}_w"] = within_cell_z(data, column)
    data["d"] = data["relative_depth"] - 0.5
    data["d2"] = data["d"] ** 2
    return data


def mixture_lrt_pvalue(statistic: float, df_min: int, n_boundary: int) -> float:
    """Boundary-mixture LRT p: sum_j Binom(n_boundary, j)/2^n_boundary * chi2_{df_min + j}.

    Adding one fixed effect plus k correlated random slopes (each to a term that already
    has an intercept) gives df_min = 1 + k and n_boundary = k (Stram & Lee 1994: each
    slope contributes 0.5 chi2_1 + 0.5 chi2_2); the naive test uses chi2_{1 + 2k}."""

    statistic = max(float(statistic), 0.0)
    return float(sum(comb(n_boundary, j) / 2 ** n_boundary * stats.chi2.sf(statistic, df_min + j)
                     for j in range(n_boundary + 1)))


def holm(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    if finite.any():
        out[finite] = multipletests(values[finite], method="holm")[1]
    return out


@dataclass(frozen=True)
class _Fitted:
    fit: MixedFit
    terms: list[RandomTerm]
    names: list[str]

    def coef(self, name: str) -> tuple[float, float]:
        index = self.names.index(name)
        return float(self.fit.beta[index]), float(self.fit.se[index])

    def r2m(self) -> float:
        return marginal_r2(self.fit, self.terms)


def _fit(rhs: str, frame: pd.DataFrame, slope: str | None = None, model_slope: bool = False,
         concept_slope: bool = False) -> _Fitted | None:
    """ML fit of ``y ~ rhs`` with ``(1 [+ slope] | model) + (1 [+ slope] | concept)``;
    ``None`` when the optimizer does not converge."""

    x = patsy.dmatrix(rhs, frame, return_type="dataframe", NA_action="raise")
    ones = np.ones((len(frame), 1))
    slope_column = frame[slope].to_numpy(float)[:, None] if slope else None
    terms = [
        RandomTerm("model", pd.factorize(frame["model_id"])[0],
                   np.hstack([ones, slope_column]) if model_slope else ones),
        RandomTerm("concept", pd.factorize(frame["concept"])[0],
                   np.hstack([ones, slope_column]) if concept_slope else ones),
    ]
    fit = fit_mixed(frame["y"].to_numpy(float), x.to_numpy(float), terms)
    return _Fitted(fit, terms, list(x.columns)) if fit.converged else None


# --------------------------------------------------------------------------- #
# Primary analysis
# --------------------------------------------------------------------------- #

BASE_RHS = "decod_w + d + d2"
STRUCTURES = (("model+concept slopes", True, True), ("model slope", True, False),
              ("intercepts only", False, False))


def primary_test(data: pd.DataFrame, metric_column: str) -> dict:
    slope = f"{metric_column}_w"
    frame = data.dropna(subset=["y", "decod_w", slope, "d"])
    row = {"n_rows": int(len(frame)), "n_cells": int(frame.groupby(CELL).ngroups)}
    base = _fit(BASE_RHS, frame)
    if base is None:
        return {**row, "status": "base model did not converge"}
    for label, model_slope, concept_slope in STRUCTURES:
        full = _fit(f"{BASE_RHS} + {slope}", frame, slope, model_slope, concept_slope)
        if full is None:
            continue
        k = int(model_slope) + int(concept_slope)
        statistic = max(2.0 * (full.fit.llf - base.fit.llf), 0.0)
        coef, se = full.coef(slope)
        return {
            **row, "status": "ok", "random_structure": label,
            "coef": coef, "ci_low": coef - 1.96 * se, "ci_high": coef + 1.96 * se,
            "lrt_stat": float(statistic), "lrt_df_naive": 1 + 2 * k,
            "p_naive": float(stats.chi2.sf(statistic, 1 + 2 * k)),
            "p_mixture": mixture_lrt_pvalue(statistic, 1 + k, k),
            "r2m_base": base.r2m(), "r2m_full": full.r2m(),
        }
    return {**row, "status": "full model did not converge under any random structure"}


def primary_table(cells: pd.DataFrame, outcomes: dict[str, str] = OUTCOMES) -> pd.DataFrame:
    rows = []
    for outcome, outcome_column in outcomes.items():
        data = _prepare(cells, outcome_column)
        for metric, metric_column in METRICS.items():
            rows.append({"outcome": outcome, "metric": metric, **primary_test(data, metric_column)})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table.get("p_mixture", pd.Series(np.nan, index=table.index)))
    if "r2m_full" in table:
        table["r2m_increment"] = table["r2m_full"] - table["r2m_base"]
    return table


# --------------------------------------------------------------------------- #
# Secondary analyses
# --------------------------------------------------------------------------- #

def _residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def partial_r_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Per (cell, outcome, metric): partial r of M and Y controlling decod, d, d^2."""

    rows = []
    for outcome, outcome_column in OUTCOMES.items():
        excluded = outcome_exclusions(cells, outcome_column).set_index(CELL).index
        for (model_id, concept), cell in cells.groupby(CELL, sort=True):
            depth = cell["relative_depth"].to_numpy(float)
            controls = np.column_stack([np.ones(len(cell)), cell[DECODABILITY].to_numpy(float), depth, depth ** 2])
            y = cell[outcome_column].to_numpy(float)
            for metric, metric_column in METRICS.items():
                m = cell[metric_column].to_numpy(float)
                r_decod = (float(np.corrcoef(m, cell[DECODABILITY].to_numpy(float))[0, 1])
                           if np.std(m) > 0 and np.std(cell[DECODABILITY]) > 0 else np.nan)
                rm, ry = _residualize(m, controls), _residualize(y, controls)
                r = float(np.corrcoef(rm, ry)[0, 1]) if np.std(rm) > 1e-12 and np.std(ry) > 1e-12 else np.nan
                if (model_id, concept) in excluded:
                    status = "excluded: outcome SD < 0.02"
                elif not np.isfinite(r_decod) or abs(r_decod) > COLLINEAR_R:
                    status = "excluded: |r(M, decod)| > .95"
                elif not np.isfinite(r):
                    status = "excluded: degenerate residuals"
                else:
                    status = "included"
                rows.append({"model_id": model_id, "concept": concept, "outcome": outcome,
                             "metric": metric, "n": int(len(cell)), "partial_r": r,
                             "r_metric_decod": r_decod, "status": status})
    return pd.DataFrame(rows)


def reml_tau2(effects: np.ndarray, variances: np.ndarray, tol: float = 1e-10,
              max_iter: int = 500) -> float:
    """REML between-study variance (fixed-point iteration, Viechtbauer 2005)."""

    tau2 = max(float(np.var(effects, ddof=1) - np.mean(variances)), 0.0) if len(effects) > 1 else 0.0
    for _ in range(max_iter):
        weights = 1.0 / (variances + tau2)
        mu = float(np.sum(weights * effects) / np.sum(weights))
        updated = float(np.sum(weights ** 2 * ((effects - mu) ** 2 - variances)) / np.sum(weights ** 2)
                        + 1.0 / np.sum(weights))
        updated = max(updated, 0.0)
        if abs(updated - tau2) < tol:
            return updated
        tau2 = updated
    return tau2


def partial_r_meta(partials: pd.DataFrame, n_controls: int = 3) -> pd.DataFrame:
    """REML random-effects pooling of within-cell Fisher-z partial r."""

    rows = []
    for (outcome, metric), group in partials.groupby(["outcome", "metric"], sort=False):
        included = group[group["status"] == "included"]
        row = {"outcome": outcome, "metric": metric, "k_cells": int(len(included)),
               "n_excluded_collinear": int((group["status"] == "excluded: |r(M, decod)| > .95").sum()),
               "n_excluded_outcome_sd": int((group["status"] == "excluded: outcome SD < 0.02").sum())}
        if len(included) >= 2:
            z = np.arctanh(np.clip(included["partial_r"].to_numpy(float), -0.999999, 0.999999))
            variances = 1.0 / (included["n"].to_numpy(float) - 3 - n_controls)
            tau2 = reml_tau2(z, variances)
            weights = 1.0 / (variances + tau2)
            mu = float(np.sum(weights * z) / np.sum(weights))
            se = float(np.sqrt(1.0 / np.sum(weights)))
            row.update({"partial_r_pooled": float(np.tanh(mu)), "ci_low": float(np.tanh(mu - 1.96 * se)),
                        "ci_high": float(np.tanh(mu + 1.96 * se)), "z": mu / se,
                        "p": float(2 * stats.norm.sf(abs(mu / se))), "tau2": tau2})
        rows.append(row)
    return pd.DataFrame(rows)


def between_cell_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Cell means of Y on cell means of M and decodability + family + concept fixed effects."""

    means = cells.groupby(CELL).agg(
        family=("family", "first"), y_success=("success", "mean"), y_broken=("broken", "mean"),
        decod=(DECODABILITY, "mean"), **{column: (column, "mean") for column in METRICS.values()}).reset_index()
    rows = []
    for outcome in OUTCOMES:
        for metric, column in METRICS.items():
            fit = smf.ols(f"y_{outcome} ~ {column} + decod + C(family) + C(concept)", data=means).fit()
            low, high = fit.conf_int().loc[column].tolist()
            rows.append({"outcome": outcome, "metric": metric, "n_cells": int(fit.nobs),
                         "coef": float(fit.params[column]), "ci_low": low, "ci_high": high,
                         "p": float(fit.pvalues[column])})
    return pd.DataFrame(rows)


def metric_vs_metric_table(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All five metrics in one intercepts-only model; drop-one LRTs and fixed-design VIFs."""

    slopes = [f"{column}_w" for column in METRICS.values()]
    tests, vifs = [], []
    for outcome, outcome_column in OUTCOMES.items():
        data = _prepare(cells, outcome_column).dropna(subset=["y", "decod_w", *slopes])
        full = _fit(f"{BASE_RHS} + {' + '.join(slopes)}", data)
        for metric, slope in zip(METRICS, slopes):
            reduced = _fit(f"{BASE_RHS} + {' + '.join(s for s in slopes if s != slope)}", data)
            if full is None or reduced is None:
                tests.append({"outcome": outcome, "metric": metric, "status": "did not converge"})
                continue
            statistic = max(2.0 * (full.fit.llf - reduced.fit.llf), 0.0)
            coef, se = full.coef(slope)
            tests.append({"outcome": outcome, "metric": metric, "status": "ok", "n_rows": int(len(data)),
                          "coef": coef, "ci_low": coef - 1.96 * se, "ci_high": coef + 1.96 * se,
                          "lrt_stat": statistic, "p": float(stats.chi2.sf(statistic, 1))})
        design = data[["decod_w", *slopes, "d", "d2"]].to_numpy(float)
        design = np.column_stack([np.ones(len(design)), design])
        for index, term in enumerate(["decod_w", *slopes, "d", "d2"], start=1):
            vifs.append({"outcome": outcome, "term": term,
                         "vif": float(variance_inflation_factor(design, index))})
    return pd.DataFrame(tests), pd.DataFrame(vifs)


def moderator_table(cells: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    """M_w x moderator interactions added to each primary full model (Holm per moderator)."""

    rows = []
    for outcome, outcome_column in OUTCOMES.items():
        data = _prepare(cells, outcome_column)
        data["log10_size_c"] = data["log10_size"] - data["log10_size"].mean()
        for metric, column in METRICS.items():
            slope = f"{column}_w"
            frame = data.dropna(subset=["y", "decod_w", slope])
            chosen = structures[(structures["outcome"] == outcome) & (structures["metric"] == metric)]
            label = chosen["random_structure"].iloc[0] if len(chosen) else None
            flags = {name: (model_slope, concept_slope) for name, model_slope, concept_slope in STRUCTURES}
            if label not in flags:
                continue
            structure = flags[label]
            for moderator, term in MODERATORS.items():
                if frame[moderator].nunique() < 2:
                    rows.append({"outcome": outcome, "metric": metric, "moderator": moderator,
                                 "status": "moderator constant"})
                    continue
                extended = _fit(f"{BASE_RHS} + {slope} + {term} + {slope}:{term}", frame, slope, *structure)
                main = _fit(f"{BASE_RHS} + {slope} + {term}", frame, slope, *structure)
                if extended is None or main is None:
                    rows.append({"outcome": outcome, "metric": metric, "moderator": moderator,
                                 "status": "did not converge"})
                    continue
                df = len(extended.names) - len(main.names)
                statistic = max(2.0 * (extended.fit.llf - main.fit.llf), 0.0)
                rows.append({"outcome": outcome, "metric": metric, "moderator": moderator,
                             "status": "ok", "lrt_stat": statistic, "df": int(df),
                             "p": float(stats.chi2.sf(statistic, df))})
    table = pd.DataFrame(rows)
    if "p" in table:
        table["p_holm"] = np.nan
        for moderator, group in table.groupby("moderator"):
            table.loc[group.index, "p_holm"] = holm(group["p"])
    return table


def interpretation(primary: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """'Predicts beyond decodability' iff Holm-significant and same-sign pooled partial r."""

    merged = primary.merge(meta[["outcome", "metric", "partial_r_pooled"]], on=["outcome", "metric"], how="left")
    same_sign = np.sign(merged["coef"]) == np.sign(merged["partial_r_pooled"])
    merged["predicts_beyond_decodability"] = (merged["p_holm"] < 0.05) & same_sign
    return merged[["outcome", "metric", "coef", "ci_low", "ci_high", "p_mixture", "p_holm",
                   "partial_r_pooled", "predicts_beyond_decodability"]]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def attach_metadata(cells: pd.DataFrame, model_meta: pd.DataFrame | None,
                    concept_meta: pd.DataFrame | None) -> pd.DataFrame:
    out = cells.copy()
    if model_meta is not None:
        meta = model_meta.assign(model_id=model_meta["model_id"].astype(str))
        out = out.merge(meta[["model_id", "family", "params_b", "moe"]], on="model_id", how="left",
                        validate="many_to_one")
        if out["family"].isna().any():
            raise ValueError(f"model metadata lacks {sorted(out.loc[out['family'].isna(), 'model_id'].unique())}")
        out["log10_size"] = np.log10(out["params_b"].astype(float))
        out["moe"] = out["moe"].astype(int)
    if concept_meta is not None:
        out = out.merge(concept_meta[["concept", "abstractness"]], on="concept", how="left", validate="many_to_one")
        if out["abstractness"].isna().any():
            raise ValueError(f"concept metadata lacks {sorted(out.loc[out['abstractness'].isna(), 'concept'].unique())}")
    return out


def run_analysis(cells_paths: Sequence[str], output_dir: str, model_meta_path: str | None = None,
                 concept_meta_path: str | None = None) -> dict[str, pd.DataFrame]:
    _require_statsmodels()
    cells = pd.concat([pd.read_csv(path) for path in cells_paths], ignore_index=True)
    if cells.duplicated(["model_id", "concept", "layer"]).any():
        raise ValueError("duplicate (model_id, concept, layer) rows across cells files")
    cells = attach_metadata(
        cells,
        pd.read_csv(model_meta_path) if model_meta_path else None,
        pd.read_csv(concept_meta_path) if concept_meta_path else None)
    primary = primary_table(cells)
    partials = partial_r_cells(cells)
    meta = partial_r_meta(partials)
    tables = {
        "primary": primary,
        "primary_best_alpha": primary_table(cells, BEST_ALPHA_OUTCOMES),
        "partial_r_cells": partials,
        "partial_r_meta": meta,
        "interpretation": interpretation(primary, meta) if "coef" in primary else pd.DataFrame(),
        "exclusions": pd.concat([outcome_exclusions(cells, column) for column in OUTCOMES.values()],
                                ignore_index=True),
        "missing_flags": missing_flags(cells),
    }
    tables["metric_vs_metric"], tables["vif"] = metric_vs_metric_table(cells)
    if "family" in cells:
        tables["between_cell"] = between_cell_table(cells)
    if {"family", "log10_size", "moe", "abstractness"} <= set(cells.columns):
        tables["moderators"] = moderator_table(cells, primary)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "summary.json").write_text(json.dumps({
        "n_rows": int(len(cells)), "n_cells": int(cells.groupby(CELL).ngroups),
        "models": sorted(cells["model_id"].unique()), "concepts": sorted(cells["concept"].unique()),
        "tables": sorted(tables)}, indent=2), encoding="utf-8")
    return tables
