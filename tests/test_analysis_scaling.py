import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

pytest.importorskip("statsmodels")

from repsteer import analysis  # noqa: E402
from repsteer.aggregate import generic_layer_summary, generic_summary  # noqa: E402

DEPTHS = np.round(np.linspace(0.08, 0.92, 12), 2)


def _layer_summary(concept, models=("m1", "m2")):
    rows = []
    for model in models:
        for layer, depth in enumerate(DEPTHS):
            rows.append({"model_id": model, "concept": concept, "setup": "general", "layer": layer,
                         "relative_depth": depth, "steering_preferred_rate_mean_alpha": 0.1 * layer,
                         "steered_broken_rate_mean_alpha": 0.01 * layer, "best_success": 0.2,
                         "best_broken": 0.1, "valid_decision_rate_mean_alpha": 1.0})
    return pd.DataFrame(rows)


def _geometry(models=("m1", "m2"), fields=("target", "nuisance")):
    production, audit = [], []
    for model in models:
        for layer, depth in enumerate(DEPTHS):
            for offset, field in enumerate(fields):
                value = 100 * offset + layer
                production.append({"model_id": model, "layer": layer, "target_field": field,
                                   "relative_depth": depth, "PS": value, "SD": value + 0.5,
                                   "SD_mean_acc": value + 0.25})
                audit.append({"model_id": model, "layer": layer, "target_field": field,
                              "relative_depth": depth, "decodability_mean": value + 0.1,
                              "ccgp_mean": value + 0.2})
    return pd.DataFrame(production), pd.DataFrame(audit)


def test_build_cells_uses_only_the_concepts_target_field():
    production, audit = _geometry()
    cells = analysis.build_cells(_layer_summary("concept_a"), production, audit, "concept_a", "nuisance")
    assert len(cells) == 24
    # nuisance rows carry 100 + layer; target rows carry layer.
    np.testing.assert_allclose(cells["ps"], 100 + cells["layer"])
    np.testing.assert_allclose(cells["decodability"], 100.1 + cells["layer"])
    np.testing.assert_allclose(cells["ccgp"], 100.2 + cells["layer"])


def test_build_cells_ignores_other_concepts_in_layer_summary():
    production, audit = _geometry()
    summary = pd.concat([_layer_summary("concept_a"), _layer_summary("concept_b")])
    cells = analysis.build_cells(summary, production, audit, "concept_b", "target")
    assert set(cells["concept"]) == {"concept_b"} and len(cells) == 24


def test_build_cells_fails_when_a_layer_has_no_geometry():
    production, audit = _geometry()
    audit = audit[~((audit["model_id"] == "m2") & (audit["layer"] == 5))]
    with pytest.raises(ValueError, match="no geometry"):
        analysis.build_cells(_layer_summary("c"), production, audit, "c", "target")


def test_build_cells_fails_on_relative_depth_disagreement():
    production, audit = _geometry()
    production.loc[production["layer"] == 3, "relative_depth"] += 0.01
    with pytest.raises(ValueError, match="relative_depth disagrees"):
        analysis.build_cells(_layer_summary("c"), production, audit, "c", "target")


def test_build_cells_rejects_several_rows_per_layer():
    production, audit = _geometry()
    production = pd.concat([production, production])  # e.g. two poolings
    with pytest.raises(ValueError, match="several rows"):
        analysis.build_cells(_layer_summary("c"), production, audit, "c", "target")


def test_within_cell_z_standardizes_each_cell_separately():
    cells = pd.DataFrame({"model_id": ["a"] * 4 + ["b"] * 4, "concept": "c",
                          "x": [1.0, 2.0, 3.0, 4.0, 100.0, 300.0, 500.0, 700.0]})
    z = analysis.within_cell_z(cells, "x")
    for _, group in z.groupby(cells["model_id"]):
        assert abs(group.mean()) < 1e-12 and abs(group.std(ddof=0) - 1) < 1e-12
    np.testing.assert_allclose(z[:4].to_numpy(), z[4:].to_numpy())


def test_mixture_pvalue_matches_chi_square_mixtures():
    statistic = 5.0
    assert analysis.mixture_lrt_pvalue(statistic, 1, 0) == pytest.approx(stats.chi2.sf(statistic, 1))
    assert analysis.mixture_lrt_pvalue(statistic, 1, 1) == pytest.approx(
        0.5 * stats.chi2.sf(statistic, 1) + 0.5 * stats.chi2.sf(statistic, 2))
    assert analysis.mixture_lrt_pvalue(statistic, 1, 2) == pytest.approx(
        0.25 * stats.chi2.sf(statistic, 1) + 0.5 * stats.chi2.sf(statistic, 2)
        + 0.25 * stats.chi2.sf(statistic, 3))
    # The boundary mixture is never more conservative than the naive chi2(1 + k) test.
    assert analysis.mixture_lrt_pvalue(statistic, 1, 2) < stats.chi2.sf(statistic, 3)


def test_reml_tau2_matches_direct_restricted_likelihood_maximum():
    rng = np.random.default_rng(3)
    variances = rng.uniform(0.05, 0.3, size=25)
    effects = rng.normal(0.2, np.sqrt(variances + 0.08))

    def negative_restricted_loglik(tau2):
        weights = 1.0 / (variances + tau2)
        mu = np.sum(weights * effects) / np.sum(weights)
        return 0.5 * (np.sum(np.log(variances + tau2)) + np.log(np.sum(weights))
                      + np.sum(weights * (effects - mu) ** 2))

    direct = optimize.minimize_scalar(negative_restricted_loglik, bounds=(0, 5), method="bounded",
                                      options={"xatol": 1e-10}).x
    assert analysis.reml_tau2(effects, variances) == pytest.approx(direct, abs=1e-5)
    homogeneous = np.full(10, 0.3)
    assert analysis.reml_tau2(homogeneous, np.full(10, 0.1)) == 0.0


def test_mixed_matches_statsmodels_for_correlated_random_slope():
    from statsmodels.regression.mixed_linear_model import MixedLM
    from repsteer.mixed import RandomTerm, fit_mixed

    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(12), 20)
    x = rng.normal(size=len(groups))
    b0 = rng.normal(0, 0.5, 12)
    b1 = 0.6 * b0 + rng.normal(0, 0.3, 12)
    y = 1 + 0.4 * x + b0[groups] + b1[groups] * x + rng.normal(0, 0.5, len(groups))
    reference = MixedLM.from_formula("y ~ x", groups="g", re_formula="1 + x",
                                     data=pd.DataFrame({"y": y, "x": x, "g": groups})).fit(reml=False)
    design = np.column_stack([np.ones_like(x), x])
    fit = fit_mixed(y, design, [RandomTerm("g", groups, design)])
    assert fit.llf == pytest.approx(reference.llf, abs=1e-5)
    np.testing.assert_allclose(fit.beta, reference.fe_params.to_numpy(), atol=1e-4)
    np.testing.assert_allclose(fit.covariances["g"], np.asarray(reference.cov_re), atol=1e-3)


def test_mixed_matches_statsmodels_for_crossed_intercepts():
    from statsmodels.regression.mixed_linear_model import MixedLM
    from repsteer.mixed import RandomTerm, fit_mixed

    rng = np.random.default_rng(2)
    a = np.repeat(np.arange(8), 27)
    b = np.tile(np.repeat(np.arange(9), 3), 8)
    x = rng.normal(size=len(a))
    y = 0.5 * x + rng.normal(0, 0.4, 8)[a] + rng.normal(0, 0.3, 9)[b] + rng.normal(0, 0.5, len(a))
    frame = pd.DataFrame({"y": y, "x": x, "a": a, "b": b, "group": 1})
    reference = MixedLM.from_formula("y ~ x", groups="group", re_formula="0",
                                     vc_formula={"a": "0 + C(a)", "b": "0 + C(b)"}, data=frame).fit(reml=False)
    ones = np.ones((len(a), 1))
    fit = fit_mixed(y, np.column_stack([ones[:, 0], x]), [RandomTerm("a", a, ones), RandomTerm("b", b, ones)])
    assert fit.llf == pytest.approx(reference.llf, abs=1e-5)
    np.testing.assert_allclose([fit.covariances["a"][0, 0], fit.covariances["b"][0, 0]], reference.vcomp, atol=1e-4)


def _synthetic_cells(seed=0, n_models=6, n_concepts=6):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_models):
        model_offset = rng.normal(0, 0.1)
        for c in range(n_concepts):
            concept_offset = rng.normal(0, 0.1)
            decod = rng.normal(size=12)
            good = rng.normal(size=12)
            noise = rng.normal(size=(3, 12))
            for layer, depth in enumerate(DEPTHS):
                success = (0.5 + model_offset + concept_offset + 0.05 * decod[layer]
                           + 0.08 * good[layer] + 0.1 * depth + rng.normal(0, 0.03))
                broken = 0.2 + 0.05 * decod[layer] + rng.normal(0, 0.03)
                rows.append({"model_id": f"m{m}", "concept": f"c{c}", "layer": layer,
                             "relative_depth": depth, "decodability": decod[layer],
                             "ccgp": good[layer], "ps": noise[0, layer], "sd": noise[1, layer],
                             "sd_mean_acc": noise[2, layer], "success": success, "broken": broken,
                             "success_best": success, "broken_best": broken, "valid_rate": 1.0})
    return pd.DataFrame(rows)


def test_primary_detects_incremental_metric_and_not_noise():
    data = analysis._prepare(_synthetic_cells(n_models=4, n_concepts=4), "success")
    signal = analysis.primary_test(data, "ccgp")
    noise = analysis.primary_test(data, "ps")
    assert signal["status"] == "ok" and noise["status"] == "ok"
    assert signal["coef"] > 0 and signal["p_mixture"] < 1e-6
    assert noise["p_mixture"] > 0.05
    assert signal["r2m_full"] > signal["r2m_base"]


def test_flat_outcome_cells_are_excluded_from_tests():
    cells = _synthetic_cells(n_models=4, n_concepts=4)
    flat = (cells["model_id"] == "m0") & (cells["concept"] == "c0")
    cells.loc[flat, "success"] = 0.5
    excluded = analysis.outcome_exclusions(cells, "success")
    assert excluded[["model_id", "concept"]].values.tolist() == [["m0", "c0"]]
    data = analysis._prepare(cells, "success")
    assert data.groupby(analysis.CELL).ngroups == 15
    assert not ((data["model_id"] == "m0") & (data["concept"] == "c0")).any()


def test_partial_r_meta_excludes_collinear_cells_and_pools_the_rest():
    cells = _synthetic_cells(n_models=4, n_concepts=4)
    collinear = (cells["model_id"] == "m1") & (cells["concept"] == "c2")
    cells.loc[collinear, "ccgp"] = cells.loc[collinear, "decodability"] * 2 + 1
    partials = analysis.partial_r_cells(cells)
    ccgp = partials[(partials["metric"] == "CCGP") & (partials["outcome"] == "success")]
    assert (ccgp["status"] == "excluded: |r(M, decod)| > .95").sum() == 1
    meta = analysis.partial_r_meta(partials)
    pooled = meta[(meta["metric"] == "CCGP") & (meta["outcome"] == "success")].iloc[0]
    assert pooled["k_cells"] == 15 and pooled["n_excluded_collinear"] == 1
    assert pooled["partial_r_pooled"] > 0.5 and pooled["p"] < 0.001


def test_generic_success_rate_ignores_unparsed_judgements():
    judged = pd.DataFrame({
        "model_id": "m", "concept": "c", "setup": "general", "layer": 3, "relative_depth": 0.25,
        "coefficient": 1.0,
        "steering_preferred": ["True", "False", "False", "False"],
        "tie": ["False", "True", "False", "False"],
        "valid_decision": ["True", "True", "False", "False"],
        "judge_broken_steered": ["False", "False", "", ""],
        "judge_broken_baseline": ["False", "False", "", ""],
    })
    summary = generic_summary(judged)
    row = summary.iloc[0]
    assert row["steering_preferred_rate"] == 0.5 and row["tie_rate"] == 0.5
    assert row["valid_decision_rate"] == 0.5 and row["n"] == 4
    layers = generic_layer_summary(summary)
    assert layers.iloc[0]["steering_preferred_rate_mean_alpha"] == 0.5
