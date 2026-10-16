"""Production-exact steering vectors, prompts, hook, generation, judge and aggregation.

Production references live one directory above the toolkit (caa_model_search.py,
caa_conditioned_steering.py, gemma4_eval200_steering.py,
gemma4_toxicity_steering.py, repair_invalid_judgements.py and their result
folders). Every comparison skips when they are absent, so a clean clone stays
green. Model-backed checks use the offline Qwen2.5-0.5B-Instruct snapshot on CPU
in float32.
"""
from __future__ import annotations

import argparse
import ast
import glob
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from repsteer.aggregate import (layer_summary, run_aggregate, summarize_by_alpha,
                                toxicity_best_alpha, toxicity_summary)
from repsteer.config import load_config
from repsteer.data import load_dataset, split_dataset
from repsteer.judge import (REPAIR_SUFFIX, assign_options, judge_steering, judgement_fields,
                            parse_judgement, render_judge_prompt, repair_invalid_judgements)
from repsteer.steering import format_prompt, order_for_judging, run_steering, template_for_row
from repsteer.text import safe_text
from repsteer.types import Intervention

TOOLKIT = Path(__file__).resolve().parents[1]
PRODUCTION = TOOLKIT.parent
CONFIGS = TOOLKIT / "configs" / "replication"
CATDOG_SPLIT = PRODUCTION / "results-gemma4-steering-eval200-split"
TOX_SPLIT = PRODUCTION / "results-gemma4-toxicity-split"
EVAL200 = PRODUCTION / "results-gemma4-steering-eval200"
TOX_JUDGE = PRODUCTION / "results-gemma4-toxicity-judge"
TOX_ANALYSIS = PRODUCTION / "results-gemma4-toxicity-analysis"
SETUPS = ("general", "conditioned")
CATDOG_TASKS = (("style", "catdog_style"), ("language", "catdog_language"),
                ("concept", "catdog_concept"))
QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"production artifact not available: {path}")
    return path


def production_module(name: str):
    if not (PRODUCTION / f"{name}.py").exists():
        pytest.skip("production scripts are not available in this checkout")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if str(PRODUCTION) not in sys.path:
        sys.path.insert(0, str(PRODUCTION))
    return importlib.import_module(name)


def _config(name: str):
    return load_config(CONFIGS / f"{name}.json")


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(_require(path), keep_default_na=False)


def _toxicity_files(setup: str) -> list[Path]:
    files = sorted((TOX_JUDGE / setup).glob("layer_*/string_judge_scores.csv"),
                   key=lambda path: int(path.parent.name.split("_")[1]))
    if len(files) != 20:
        pytest.skip(f"toxicity judge files incomplete for {setup}")
    return files


# --------------------------------------------------------------------------- splits


def test_catdog_split_row_order_matches_production():
    config = _config("catdog_style_general")
    splits = split_dataset(load_dataset(config.dataset), config.dataset, config.split)
    for name, file in (("vector", "vector_construction_256.csv"), ("evaluation", "evaluation_200.csv")):
        production = pd.read_csv(_require(CATDOG_SPLIT / file))
        assert splits[name].candidate_id.tolist() == production.candidate_id.tolist()
        assert splits[name].text.tolist() == production.text.tolist()


def test_toxicity_split_row_order_matches_production(tmp_path):
    labeled = _require(PRODUCTION / "openai_labeling_outputs" / "labeled")
    subprocess.run([sys.executable, str(TOOLKIT / "datasets" / "toxicity" / "prepare_toxicity_splits.py"),
                    "--input-dir", str(labeled), "--output-dir", str(tmp_path)], check=True)
    for file in ("vector_construction_256.csv", "evaluation_200.csv"):
        toolkit = pd.read_csv(tmp_path / file)
        production = pd.read_csv(_require(TOX_SPLIT / file))
        assert toolkit.candidate_id.tolist() == production.candidate_id.tolist()
        assert toolkit.text.tolist() == production.text.tolist()


# -------------------------------------------------------------------------- prompts


def _toolkit_prompt(config, row):
    source = str(row[config.target.name])
    target = config.target.opposite(source)
    prompt = format_prompt(template_for_row(config.steering, row), row,
                           config.dataset.text_column, source, target)
    return prompt, target


def test_catdog_steering_prompts_match_production_pipeline():
    css = production_module("caa_conditioned_steering")
    cms = production_module("caa_model_search")
    raw_model = cms.SearchCaaModel.__new__(cms.SearchCaaModel)
    raw_model.use_chat_template = False  # format_generation_prompt == safe_text(prompt)
    evaluation = pd.read_csv(_require(CATDOG_SPLIT / "evaluation_200.csv"))
    for setup in SETUPS:
        for task, stem in CATDOG_TASKS:
            config = _config(f"{stem}_{setup}")
            assert config.generation.normalize_whitespace
            for _, row in evaluation.iterrows():
                prompt, target = _toolkit_prompt(config, row)
                expected, expected_target = css.task_prompt_and_target(task, row, "instruction")
                assert prompt == expected
                assert safe_text(prompt) == raw_model.format_generation_prompt(expected)
                label = config.steering.target_labels.get(target, target)
                assert label == expected_target
                assert config.target.sign_for_target(target) == cms.direction_sign(task, expected_target)


def test_toxicity_steering_prompts_match_production_pipeline():
    tox = production_module("gemma4_toxicity_steering")
    cms = production_module("caa_model_search")
    raw_model = cms.SearchCaaModel.__new__(cms.SearchCaaModel)
    raw_model.use_chat_template = False
    evaluation = pd.read_csv(_require(TOX_SPLIT / "evaluation_200.csv"))
    for setup in SETUPS:
        config = _config(f"toxicity_{setup}")
        for _, row in evaluation.iterrows():
            prompt, target = _toolkit_prompt(config, row)
            assert prompt == tox.prompt_for(row)
            assert safe_text(prompt) == raw_model.format_generation_prompt(tox.prompt_for(row))
            expected_target = "toxic" if int(row["toxicity_bin"]) == 0 else "non-toxic"
            assert config.steering.target_labels[target] == expected_target
            assert config.target.sign_for_target(target) == (1 if expected_target == "toxic" else -1)


def test_recorded_generation_prompts_equal_toolkit_prompts():
    evaluation = pd.read_csv(_require(CATDOG_SPLIT / "evaluation_200.csv")).set_index("candidate_id")
    for setup in SETUPS:
        generations = _csv(EVAL200 / f"{setup}_generations.csv")
        unique = generations.drop_duplicates(["task", "candidate_id"])
        assert len(unique) == 600
        for record in unique.itertuples():
            stem = dict(CATDOG_TASKS)[record.task]
            row = evaluation.loc[record.candidate_id].copy()
            row["candidate_id"] = record.candidate_id
            prompt, target = _toolkit_prompt(_config(f"{stem}_{setup}"), row)
            assert prompt == record.prompt
    tox_eval = pd.read_csv(_require(TOX_SPLIT / "evaluation_200.csv")).set_index("candidate_id")
    for setup in SETUPS:
        config = _config(f"toxicity_{setup}")
        recorded = _csv(_toxicity_files(setup)[0]).drop_duplicates("candidate_id")
        for record in recorded.itertuples():
            row = tox_eval.loc[record.candidate_id].copy()
            prompt, target = _toolkit_prompt(config, row)
            assert prompt == record.prompt
            assert config.steering.target_labels[target] == record.target_attribute


# --------------------------------------------------------------------- judge order


def test_merge_steered_reproduces_production_judged_input_order():
    evaluation = pd.read_csv(_require(CATDOG_SPLIT / "evaluation_200.csv"))
    rank = {cid: i for i, cid in enumerate(evaluation.candidate_id)}
    for setup in SETUPS:
        generations = _csv(EVAL200 / f"{setup}_generations.csv")
        generations["_row"] = np.arange(len(generations))
        # run_steering emits one table per task ordered layer -> coefficient -> row.
        per_task = []
        for task, _ in CATDOG_TASKS:
            part = generations[generations.task == task].copy()
            part["_eval"] = part.candidate_id.map(rank)
            per_task.append(part.sort_values(["layer", "alpha", "_eval"]).drop(columns="_eval"))
        merged = order_for_judging(per_task)
        assert merged["_row"].tolist() == list(range(len(generations)))


def test_toxicity_judge_files_follow_run_steering_order():
    evaluation = pd.read_csv(_require(TOX_SPLIT / "evaluation_200.csv"))
    for setup in SETUPS:
        for path in _toxicity_files(setup):
            frame = _csv(path)
            alphas = sorted(set(frame.alpha.astype(float)))
            expected_ids = list(evaluation.candidate_id) * len(alphas)
            assert frame.candidate_id.tolist() == expected_ids
            assert frame.alpha.astype(float).tolist() == [a for a in alphas for _ in range(200)]


def _prompt_pair(row, steered_is_a):
    if steered_is_a:
        return row.steered_continuation, row.baseline_continuation
    return row.baseline_continuation, row.steered_continuation


def test_catdog_option_assignment_and_judge_prompt_match_every_production_row():
    cms = production_module("caa_model_search")
    for setup in SETUPS:
        generations = _csv(EVAL200 / f"{setup}_generations.csv")
        judge = _config(f"catdog_style_{setup}").judge
        steered_is_a, positions = assign_options(generations, judge.num_shards, judge.shard_group_by)
        for job in range(judge.num_shards):
            shard = _csv(EVAL200 / setup / f"string_judge_scores_job{job}.csv")
            index = np.arange(job, len(generations), judge.num_shards)
            source = generations.iloc[index]
            assert len(shard) == len(index)
            for column in ("task", "candidate_id", "steered_continuation", "baseline_continuation"):
                assert shard[column].tolist() == source[column].tolist()
            assert shard.layer.astype(int).tolist() == source.layer.astype(int).tolist()
            assert positions[index].tolist() == list(range(len(index)))
            assert (shard.option_a_is.eq("steered").to_numpy() == steered_is_a[index]).all()
            for row, flip in zip(shard.itertuples(), steered_is_a[index]):
                a, b = _prompt_pair(row, flip)
                assert render_judge_prompt(judge, row.task, row.target_attribute, a, b) == \
                    cms.pairwise_string_prompt(row.task, row.target_attribute, a, b)


def test_toxicity_option_assignment_and_judge_prompt_match_every_production_row():
    cms = production_module("caa_model_search")
    for setup in SETUPS:
        judge = _config(f"toxicity_{setup}").judge
        frame = pd.concat([_csv(path) for path in _toxicity_files(setup)], ignore_index=True)
        steered_is_a, _ = assign_options(frame, judge.num_shards, judge.shard_group_by)
        assert (frame.option_a_is.eq("steered").to_numpy() == steered_is_a).all()
        for row, flip in zip(frame.itertuples(), steered_is_a):
            a, b = _prompt_pair(row, flip)
            assert render_judge_prompt(judge, row.task, row.target_attribute, a, b) == \
                cms.pairwise_string_prompt(row.task, row.target_attribute, a, b)


# --------------------------------------------------------------------------- parse


def _recorded(value) -> str:
    return "" if isinstance(value, float) and np.isnan(value) else str(value)


PARSED_COLUMNS = ("string_preferred", "steering_preferred", "baseline_preferred", "tie",
                  "judge_broken_steered", "judge_broken_baseline", "valid_decision",
                  "llm_broken_label_steered_available", "llm_broken_label_baseline_available")


def _assert_parse_matches(frame: pd.DataFrame, cms) -> None:
    expected = frame[list(PARSED_COLUMNS)].astype(str).to_numpy()
    for row_index, (raw, option_a) in enumerate(zip(frame.raw_judgement, frame.option_a_is)):
        parsed = parse_judgement(raw)
        assert parsed == cms.parse_judgement(raw)
        fields = judgement_fields(parsed, option_a == "steered")
        assert [_recorded(fields[column]) for column in PARSED_COLUMNS] == list(expected[row_index])


def test_parse_reproduces_every_production_judgement():
    cms = production_module("caa_model_search")
    for setup in SETUPS:
        _assert_parse_matches(_csv(EVAL200 / setup / "string_judge_scores.csv"), cms)
    for setup in SETUPS:
        _assert_parse_matches(
            pd.concat([_csv(path) for path in _toxicity_files(setup)], ignore_index=True), cms)


# ------------------------------------------------------------ judge end to end


class LookupJudge:
    """Fake judge backend returning production's recorded raw judgements per prompt,
    in production row order (identical prompts recur with different recorded raws)."""

    def __init__(self, responses: dict[str, list[str]]):
        self.responses = {prompt: list(reversed(raws)) for prompt, raws in responses.items()}
        self.calls = []

    def generate(self, prompts, **kwargs):
        self.calls.append(kwargs)
        return [self.responses[prompt].pop() for prompt in prompts]


def _lookup(frame: pd.DataFrame, cms) -> dict[str, list[str]]:
    responses: dict[str, list[str]] = {}
    for row in frame.itertuples():
        a, b = _prompt_pair(row, row.option_a_is == "steered")
        prompt = cms.pairwise_string_prompt(row.task, row.target_attribute, a, b)
        responses.setdefault(prompt, []).append(row.raw_judgement)
    return responses


JUDGE_COLUMNS = ("option_a_is", "option_b_is", "judge_reason", "raw_judgement") + PARSED_COLUMNS


def test_judge_steering_reproduces_production_judge_rows():
    cms = production_module("caa_model_search")
    cases = []
    for setup in SETUPS:
        recorded = pd.concat([_csv(path) for path in _toxicity_files(setup)], ignore_index=True)
        cases.append((_config(f"toxicity_{setup}").judge, recorded, recorded))
    generations = _csv(EVAL200 / "general_generations.csv")
    judge = _config("catdog_style_general").judge
    shards = [_csv(EVAL200 / "general" / f"string_judge_scores_job{job}.csv")
              for job in range(judge.num_shards)]
    recorded = pd.concat(shards, keys=range(judge.num_shards), names=["job", "k"]).reset_index()
    recorded["_i"] = recorded.job + judge.num_shards * recorded.k
    recorded = recorded.sort_values("_i").reset_index(drop=True)
    cases.append((judge, generations, recorded))
    for config, inputs, expected in cases:
        backend = LookupJudge(_lookup(expected, cms))
        steer_columns = [c for c in inputs.columns if c not in JUDGE_COLUMNS + ("judge_model_id",)]
        judged = judge_steering(backend, inputs[steer_columns], config, judge_model_id="Qwen/Qwen3.6-27B")
        assert all(call["max_new_tokens"] == 1024 and call["max_length"] == 2048
                   and call["do_sample"] is False and call["pad_with_eos"] for call in backend.calls)
        for column in JUDGE_COLUMNS:
            actual = [_recorded(value) for value in judged[column]]
            assert actual == expected[column].astype(str).tolist(), column


def test_repair_suffix_is_production_verbatim():
    source = _require(PRODUCTION / "repair_invalid_judgements.py").read_text(encoding="utf-8")
    constants = [node.value.value for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "strict_suffix"
                                                         for t in node.targets)]
    assert constants == [REPAIR_SUFFIX]


def test_repair_rejudges_only_invalid_rows_with_strict_suffix():
    config = _config("catdog_style_general").judge
    judged = pd.DataFrame({
        "task": ["style", "concept"], "target_attribute": ["shakespeare", "dog"],
        "steered_continuation": ["S one", "S two"], "baseline_continuation": ["B one", "B two"],
        "option_a_is": ["steered", "baseline"], "option_b_is": ["baseline", "steered"],
        "raw_judgement": ["Broken_A: false Broken_B: false Reason: x Decision: A", "Decision: B maybe"],
    })
    prompt = render_judge_prompt(config, "concept", "dog", "B two", "S two") + REPAIR_SUFFIX
    backend = LookupJudge({prompt: ["Broken_A: true Broken_B: false Reason: y Decision: B"]})
    repaired = repair_invalid_judgements(backend, judged, config)
    assert [call["seed"] for call in backend.calls] == [910000]
    assert backend.calls[0]["max_new_tokens"] == 2048
    assert repaired.raw_judgement.iloc[0] == judged.raw_judgement.iloc[0]
    assert repaired.string_preferred.iloc[1] == "B"
    assert bool(repaired.steering_preferred.iloc[1]) is True
    assert bool(repaired.judge_broken_baseline.iloc[1]) is True
    assert bool(repaired.valid_decision.iloc[1]) is True


# ------------------------------------------------------------------- aggregation


def test_catdog_aggregation_reproduces_confirmatory_layer_summary():
    frames = []
    for setup in SETUPS:
        frame = _csv(EVAL200 / setup / "string_judge_scores.csv")
        frame["setup"] = setup
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["layer"] = pd.to_numeric(data.layer)
    data["alpha"] = pd.to_numeric(data.alpha)
    summary = summarize_by_alpha(data, ("setup", "task", "layer", "alpha"))
    for setup in SETUPS:
        reference = pd.read_csv(_require(EVAL200 / setup / "summary_by_model_task_layer_alpha.csv"))
        merged = reference.merge(summary[summary.setup == setup], on=["task", "layer", "alpha"],
                                 suffixes=("_old", "_new"))
        assert len(merged) == len(reference) == 300
        for column in ("n", "steering_preferred_rate", "tie_rate", "steered_broken_rate",
                       "baseline_broken_rate", "valid_decision_rate"):
            np.testing.assert_allclose(merged[f"{column}_new"], merged[f"{column}_old"], rtol=0, atol=1e-12)
    actual = layer_summary(summary, "alpha")
    expected = pd.read_csv(_require(PRODUCTION / "results-gemma4-steering-eval200-analysis"
                                    / "confirmatory_layer_summary.csv"))
    assert list(actual.columns) == list(expected.columns)
    assert actual[["setup", "task", "layer"]].astype(str).values.tolist() == \
        expected[["setup", "task", "layer"]].astype(str).values.tolist()
    for column in expected.columns[3:]:
        np.testing.assert_allclose(actual[column].to_numpy(float), expected[column].to_numpy(float),
                                   rtol=0, atol=1e-12, err_msg=column)


def test_toxicity_aggregation_reproduces_summary_and_best_alpha(tmp_path):
    files = [path for setup in SETUPS for path in _toxicity_files(setup)]
    scores = sorted(glob.glob(str(PRODUCTION / "results-gemma4-toxicity-xlmr" / "xlmr_scores_job_*.csv")))
    geometry = [p for p in (PRODUCTION / "results-gemma4-toxicity-geometry-zscore").glob("geometry_job*.csv")
                if "dichotomies" not in p.name]
    matched = sorted((PRODUCTION / "results-gemma4-toxicity-matched-decodability-ccgp")
                     .glob("matched_metrics_layer_*.csv"))
    if not scores or not geometry or len(matched) != 20:
        pytest.skip("toxicity classifier/geometry artifacts not available")
    pd.concat([pd.read_csv(p) for p in scores]).to_csv(tmp_path / "scores.csv", index=False)
    pd.concat([pd.read_csv(p) for p in geometry]).to_csv(tmp_path / "geometry_rows.csv", index=False)
    pd.concat([pd.read_csv(p) for p in matched]).to_csv(tmp_path / "matched.csv", index=False)
    tables = run_aggregate("toxicity", files, tmp_path / "out", alpha_column="alpha",
                           classifier_scores_path=tmp_path / "scores.csv",
                           geometry_rows_path=tmp_path / "geometry_rows.csv",
                           matched_metrics_path=tmp_path / "matched.csv")
    for name, file in (("summary", "toxicity_summary_by_layer_alpha.csv"),
                       ("best_alpha_geometry", "toxicity_best_alpha_geometry.csv")):
        actual = tables[name].reset_index(drop=True)
        expected = pd.read_csv(_require(TOX_ANALYSIS / file))
        assert list(actual.columns) == list(expected.columns)
        assert actual[["setup", "layer", "alpha", "n"]].values.tolist() == \
            expected[["setup", "layer", "alpha", "n"]].values.tolist()
        for column in expected.columns[4:]:
            if column == "decodability":
                # The production analysis CSV (written 2026-08-07) predates the
                # current matched_metrics_layer_*.csv rerun (2026-08-19); the join
                # itself is checked against the current matched metrics below.
                continue
            np.testing.assert_allclose(actual[column].to_numpy(float), expected[column].to_numpy(float),
                                       rtol=0, atol=1e-12, err_msg=f"{file}:{column}")
    matched_frame = pd.read_csv(tmp_path / "matched.csv")
    current = matched_frame[matched_frame.target_field == "toxicity_bin"].set_index("layer").decodability_mean
    joined = tables["best_alpha_geometry"]
    assert joined.decodability.tolist() == current.loc[joined.layer].tolist()
    # Without the classifier the outcome columns are unchanged.
    data = pd.concat([_csv(path) for path in files], ignore_index=True)
    data["layer"] = pd.to_numeric(data.layer)
    data["alpha"] = pd.to_numeric(data.alpha)
    best = toxicity_best_alpha(toxicity_summary(data, "alpha")).reset_index(drop=True)
    reference = tables["best_alpha"].reset_index(drop=True)
    assert best[["setup", "layer", "alpha", "n", "success", "broken_rate", "tie_rate"]].equals(
        reference[["setup", "layer", "alpha", "n", "success", "broken_rate", "tie_rate"]])


# ------------------------------------------------------------------------- hook


class _TupleBlock:
    def __init__(self, torch):
        self.module = torch.nn.Module()
        self.module.forward = lambda x: (x * 1, None)


def test_hook_arithmetic_matches_register_caa_hook_in_bfloat16():
    torch = pytest.importorskip("torch")
    cms = production_module("caa_model_search")
    from repsteer.models.huggingface import HuggingFaceBackend

    generator = torch.Generator().manual_seed(0)
    vector = torch.randn(16, generator=generator, dtype=torch.float32)
    vector = vector / float(torch.linalg.norm(vector))
    for module_kind in ("tensor", "tuple"):
        for position in ("all", "last", "generated"):
            for alpha in (8.0, -32.0, 128.0):
                outputs = {}
                for implementation in ("production", "toolkit"):
                    block = torch.nn.Identity() if module_kind == "tensor" else _TupleBlock(torch).module
                    inputs = [torch.randn(2, 5, 16, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16),
                              torch.randn(2, 1, 16, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)]
                    if implementation == "production":
                        model = cms.SearchCaaModel.__new__(cms.SearchCaaModel)
                        model.layers, model.hooks = [block], []
                        model.register_caa_hook(0, vector, alpha, position)
                        result = [block(x) for x in inputs]
                        model.clear_hooks()
                    else:
                        backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
                        backend.torch, backend.layers = torch, [block]
                        with backend._steering_hook(Intervention(0, vector.numpy(), alpha, position)):
                            result = [block(x) for x in inputs]
                    outputs[implementation] = [r[0] if isinstance(r, tuple) else r for r in result]
                for left, right in zip(outputs["production"], outputs["toolkit"]):
                    assert left.dtype == right.dtype == torch.bfloat16
                    assert torch.equal(left, right), (module_kind, position, alpha)


# ------------------------------------------------------------------ tiny model


@pytest.fixture(scope="module")
def qwen():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(QWEN_ID, local_files_only=True)
    except Exception:
        pytest.skip(f"{QWEN_ID} is not in the local Hugging Face cache")
    cms = production_module("caa_model_search")
    from repsteer.config import ModelConfig
    from repsteer.models.huggingface import HuggingFaceBackend

    torch.manual_seed(0)
    backend = HuggingFaceBackend(ModelConfig(model_id=path, dtype="float32", device_map="cpu",
                                             use_chat_template=True))
    production = cms.SearchCaaModel.__new__(cms.SearchCaaModel)
    production.model_id = path
    production.tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    if production.tokenizer.pad_token_id is None:
        production.tokenizer.pad_token = production.tokenizer.eos_token
    production.model = backend.model
    production.layers = cms.get_decoder_layers(backend.model)
    production.hooks = []
    production.normalize_vectors = True
    production.use_chat_template = True
    return backend, production


LAYER = 6
VECTOR_ARGS = argparse.Namespace(num_contrast_pairs=32, seed=42, batch_size=4, max_length=512,
                                 vector_source="conditioned_corpus_mean")


def _toolkit_vectors(backend, config, pool):
    from repsteer.vectors import build_vectors

    steering = config.steering
    return {vector.condition: vector.values for vector in build_vectors(
        backend, pool, config.dataset.text_column, config.target, [LAYER], steering.pooling,
        steering.normalize_vector, steering.batch_size, steering.max_length, sampling=steering.sampling)}


def test_catdog_vectors_equal_production(qwen):
    backend, production = qwen
    e200 = production_module("gemma4_eval200_steering")
    css = production_module("caa_conditioned_steering")
    pool = pd.read_csv(_require(CATDOG_SPLIT / "vector_construction_256.csv"))
    general = _config("catdog_style_general")
    expected = e200.build_general_vectors(production, pool, "style", LAYER, VECTOR_ARGS)
    actual = _toolkit_vectors(backend, general, load_dataset(general.dataset, CATDOG_SPLIT / "vector_construction_256.csv"))
    assert set(actual) == {(("language", language),) for language in expected}
    for language, vector in expected.items():
        assert np.array_equal(actual[(("language", language),)], vector.numpy())
    conditioned = _config("catdog_concept_conditioned")
    expected = css.build_conditioned_vectors(production, pool, "concept", LAYER, VECTOR_ARGS)
    actual = _toolkit_vectors(backend, conditioned, load_dataset(conditioned.dataset, CATDOG_SPLIT / "vector_construction_256.csv"))
    assert len(actual) == len(expected) == 4
    for (language, style), vector in expected.items():
        assert np.array_equal(actual[(("language", language), ("style", style))], vector.numpy())


def test_toxicity_vectors_equal_production(qwen):
    backend, production = qwen
    tox = production_module("gemma4_toxicity_steering")
    pool_path = _require(TOX_SPLIT / "vector_construction_256.csv")
    pool = pd.read_csv(pool_path)
    for setup in SETUPS:
        config = _config(f"toxicity_{setup}")
        expected = tox.build_vectors(production, pool, setup, LAYER, VECTOR_ARGS)
        actual = _toolkit_vectors(backend, config, load_dataset(config.dataset, pool_path))
        assert len(actual) == len(expected)
        for key, vector in expected.items():
            condition = (("language", key[0]),) if len(key) == 1 else \
                (("gender_bin", str(key[1])), ("language", key[0]))
            assert np.array_equal(actual[condition], vector.numpy())


def test_chat_formatting_matches_production(qwen):
    backend, production = qwen
    cms = production_module("caa_model_search")
    judge = cms.StringJudge.__new__(cms.StringJudge)
    judge.tokenizer = production.tokenizer
    evaluation = pd.read_csv(_require(CATDOG_SPLIT / "evaluation_200.csv"))
    config = _config("catdog_style_general")
    for _, row in evaluation.iterrows():
        prompt, _ = _toolkit_prompt(config, row)
        assert backend._format_prompts([safe_text(prompt)]) == [production.format_generation_prompt(prompt)]
        assert backend._format_prompts([prompt]) == [judge.format_judge_prompt(prompt)]


def _production_generations(production, cms, rows, prompts_targets, task, vectors, alpha, seeds):
    outputs = []
    for (prompt, target), row, (baseline_seed, steered_seed) in zip(prompts_targets, rows, seeds):
        baseline = production.generate(prompt, baseline_seed, 35, False, 0.7, 0.9, 512)
        signed = (cms.direction_sign(task, target) if task != "toxicity"
                  else (1 if target == "toxic" else -1)) * float(alpha)
        production.register_caa_hook(LAYER, vectors(row), signed, "all")
        steered = production.generate(prompt, steered_seed, 35, False, 0.7, 0.9, 512)
        production.clear_hooks()
        outputs.append((baseline, steered))
    return outputs


def test_steered_and_baseline_generations_equal_production(qwen):
    from dataclasses import replace

    from repsteer.steering import steered_seed
    from repsteer.vectors import build_vectors

    backend, production = qwen
    cms = production_module("caa_model_search")
    css = production_module("caa_conditioned_steering")
    e200 = production_module("gemma4_eval200_steering")
    tox = production_module("gemma4_toxicity_steering")
    alpha = 8.0
    cases = []
    catdog = _config("catdog_style_general")
    catdog_eval = pd.read_csv(_require(CATDOG_SPLIT / "evaluation_200.csv")).iloc[[0, 60, 120, 180]]
    catdog_pool = pd.read_csv(_require(CATDOG_SPLIT / "vector_construction_256.csv"))
    prod_catdog = e200.build_general_vectors(production, catdog_pool, "style", LAYER, VECTOR_ARGS)
    cases.append((catdog, catdog_eval, catdog_pool, "style",
                  [css.task_prompt_and_target("style", row, "instruction") for _, row in catdog_eval.iterrows()],
                  lambda row: prod_catdog[row["language"]]))
    toxicity = _config("toxicity_general")
    tox_eval = pd.read_csv(_require(TOX_SPLIT / "evaluation_200.csv")).iloc[[10, 150]]
    tox_pool = pd.read_csv(_require(TOX_SPLIT / "vector_construction_256.csv"))
    prod_tox = tox.build_vectors(production, tox_pool, "general", LAYER, VECTOR_ARGS)
    cases.append((toxicity, tox_eval, tox_pool, "toxicity",
                  [(tox.prompt_for(row), "toxic" if int(row["toxicity_bin"]) == 0 else "non-toxic")
                   for _, row in tox_eval.iterrows()],
                  lambda row: prod_tox[(row["language"],)]))
    for config, evaluation, pool, task, prompts_targets, production_vector in cases:
        steering = replace(config.steering, layers=(LAYER,), coefficients=(alpha,))
        vectors = build_vectors(backend, pool, config.dataset.text_column, config.target, [LAYER],
                                steering.pooling, steering.normalize_vector, steering.batch_size,
                                steering.max_length, sampling=steering.sampling)
        result = run_steering(backend, evaluation.reset_index(drop=True), config.dataset.text_column,
                              config.dataset.id_column, config.target, vectors, steering,
                              config.generation, model_id=config.model.model_id)
        seeds = [(config.generation.seed + config.generation.baseline_seed_offset + number,
                  steered_seed(config.generation, number, setup=steering.setup,
                               model_id=config.model.model_id, task=steering.task, layer=LAYER,
                               alpha=alpha, id=row[config.dataset.id_column]))
                 for number, (_, row) in enumerate(evaluation.iterrows())]
        expected = _production_generations(production, cms, [row for _, row in evaluation.iterrows()],
                                           prompts_targets, task, production_vector, alpha, seeds)
        assert result.prompt.tolist() == [prompt for prompt, _ in prompts_targets]
        assert result.baseline_continuation.tolist() == [baseline for baseline, _ in expected]
        assert result.steered_continuation.tolist() == [steered for _, steered in expected]
        assert (result.baseline_continuation != result.steered_continuation).any()
