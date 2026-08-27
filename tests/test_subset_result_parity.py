from pathlib import Path

import pytest

from scripts.verify_prior_subset import audit


def test_direct_prior_result_subset_parity(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    required = [
        repo / "results-gemma4-online-offtarget-projection-analysis/online_projection_rows.csv",
        repo / "results-gemma4-targeted-alpha-offtarget-llm/offtarget_llm_effect_rows.csv",
        repo / "results-gemma4-mmmlu-if-100/figures/mmmlu_scores_by_intervention.csv",
        repo / "results-gemma4-mifeval-if-100/figures/mifeval_scores_by_intervention.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip("Prior production artifacts are not available in this checkout")
    report = audit(repo, tmp_path, groups=6, seed=42)
    assert report["exact_or_1e-12"].all()
    assert set(report.family) == {
        "projection_summary", "projection_rows", "offtarget_judge_summary",
        "mmmlu", "mifeval"}
    assert (tmp_path / "parity_summary.json").exists()
