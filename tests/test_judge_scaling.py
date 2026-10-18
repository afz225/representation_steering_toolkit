"""Judge scaling-study extensions: generic prompt preset, execution shards,
merge-judged coverage, --resume, vLLM backend glue, refusal heuristic."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

from repsteer.config import JudgeConfig, ModelConfig
from repsteer.judge import (GENERIC_JUDGE_PROMPT, DEFAULT_JUDGE_PROMPT, REFUSAL_SUBSTRINGS,
                            assign_options, judge_steering, parse_judgement,
                            refusal_heuristic, render_judge_prompt, resume_key,
                            shard_positions)
from repsteer.judge_vllm import VLLMJudgeBackend, judge_steering_vllm


CONFIG = JudgeConfig(
    model=ModelConfig(model_id="Qwen/Qwen3.6-27B"),
    attribute_descriptions={
        "concept": {"cat": "Target animal concept: cat", "dog": "Target animal concept: dog"},
    },
)


def _frame(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(n)],
        "layer": [i % 3 for i in range(n)],
        "coefficient": [1.0] * n,
        "task": ["concept"] * n,
        "target_attribute": ["cat" if i % 2 else "dog" for i in range(n)],
        "prompt": [f"prompt {i}" for i in range(n)],
        "steered_continuation": [f"S{i}" for i in range(n)],
        "baseline_continuation": [f"B{i}" for i in range(n)],
    })


class FakeJudge:
    """Deterministic fake judge backend echoing the Decision it was seeded with."""

    def __init__(self):
        self.prompts: list[str] = []
        self.seeds: list[int] = []
        self.kwarg_list: list[dict] = []

    def generate(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        self.seeds.append(kwargs.get("seed"))
        self.kwarg_list.append(kwargs)
        return ["Broken_A: false Broken_B: false Reason: x Decision: A" for _ in prompts]


# --------------------------------------------------------------- generic prompt


def test_generic_prompt_renders_target_line_from_attribute_descriptions():
    config = JudgeConfig(
        model=CONFIG.model, attribute_descriptions=CONFIG.attribute_descriptions,
        prompt_preset="generic")
    text = render_judge_prompt(config, "concept", "cat", "A text", "B text")
    expected = GENERIC_JUDGE_PROMPT.format(
        target_line="Target animal concept: cat", a="A text", b="B text")
    assert text == expected
    assert "requested style, language, or animal concept" not in text  # domain-neutral wording
    assert "requested target attribute" in text
    # The production preset stays default and verbatim.
    default = JudgeConfig(
        model=CONFIG.model, attribute_descriptions=CONFIG.attribute_descriptions)
    assert render_judge_prompt(default, "concept", "cat", "A", "B") == \
        DEFAULT_JUDGE_PROMPT.format(target_line="Target animal concept: cat", a="A", b="B")


# ------------------------------------------------------------ execution shards


def test_shard_union_equals_unsharded_assignment_and_seeds():
    frame = _frame(24)
    unsharded_fake = FakeJudge()
    unsharded = judge_steering(unsharded_fake, frame, CONFIG)
    shard_rows: list[pd.DataFrame] = []
    shard_seeds: list[int] = []
    for shard_index in range(4):
        fake = FakeJudge()
        judged = judge_steering(fake, frame, CONFIG, num_exec_shards=4,
                                shard_index=shard_index)
        shard_seeds.extend(fake.seeds)
        shard_rows.append(judged)
    # The union of the shards covers every row exactly once, each with the
    # unsharded A/B assignment, judge position, and seed.
    merged = pd.concat(shard_rows, ignore_index=True).sort_values(
        "position", kind="stable").reset_index(drop=True)
    unsorted = unsharded.sort_values("position", kind="stable").reset_index(drop=True)
    assert len(merged) == len(unsorted) == 24
    assert merged.position.tolist() == list(range(24))
    for column in ("candidate_id", "raw_judgement", "string_preferred", "option_a_is",
                   "judge_backend", "position"):
        assert merged[column].tolist() == unsorted[column].tolist(), column
    # Shard seeds are the unsharded seeds of the rows each shard owns.
    steered_is_a_full, k_full = assign_options(frame, 1, ())
    assert sorted(shard_seeds) == sorted(CONFIG.seed + int(k) for k in k_full)
    assert sorted(shard_seeds) == sorted(unsharded_fake.seeds)


def _shard(frame: pd.DataFrame, shard_index: int, num_exec_shards: int) -> pd.DataFrame:
    keep = shard_positions(frame, num_exec_shards) == shard_index
    return frame[keep].reset_index(drop=True)


def test_shard_positions_assign_on_full_table_positions():
    frame = _frame(10)
    positions = shard_positions(frame, 3)
    # Shard membership is row index %% num_exec_shards; union covers everything.
    assert positions.tolist() == [i % 3 for i in range(len(frame))]
    # CLI path: judge_steering receives the FULL frame and filters rows itself,
    # so the A/B flip and judge position k (seed = seed + k) are the unsharded
    # deal: row i keeps flip_i and k = i (num_shards=1 deal).
    steered_is_a_full, k_full = assign_options(frame, 1, ())
    fake = FakeJudge()
    judged_part = judge_steering(fake, frame, CONFIG, num_exec_shards=3, shard_index=0)
    assert judged_part.candidate_id.tolist() == [f"c{i}" for i in (0, 3, 6, 9)]
    assert judged_part.option_a_is.tolist() == [
        "steered" if flag else "baseline" for flag in steered_is_a_full[[0, 3, 6, 9]]]
    assert fake.seeds == [CONFIG.seed + int(k_full[i]) for i in (0, 3, 6, 9)]


def test_seeds_match_production_catdog_shard_protocol():
    """Production: row i -> job i %% num_shards (A/B), judge position
    k = i // num_shards, seed 42 + k. With num_exec_shards=4 the union of the
    four shards covers every row and keeps the unsharded (seed, flip) pair
    per row."""

    frame = _frame(64 * 4 + 8)
    num_shards = 64  # production A/B deal (judge.num_shards)
    config = replace(CONFIG, num_shards=num_shards)
    per_shard_fakes: list[FakeJudge] = []
    per_shard_rows: list[pd.DataFrame] = []
    for shard_index in range(4):
        fake = FakeJudge()
        judged = judge_steering(fake, frame, config, num_exec_shards=4,
                                shard_index=shard_index)
        per_shard_fakes.append(fake)
        per_shard_rows.append(judged)
    unsharded_fake = FakeJudge()
    unsharded = judge_steering(unsharded_fake, frame, config)
    merged = pd.concat(per_shard_rows, ignore_index=True)
    unsharded_sorted = unsharded.copy()
    # The production position column is not unique across jobs here (each k
    # repeats every 64 rows), so compare keyed on the row identity instead.
    key_merged = merged.set_index(["candidate_id"])
    key_unsharded = unsharded_sorted.set_index(["candidate_id"])
    key_merged = key_merged.sort_index()
    key_unsharded = key_unsharded.sort_index()
    assert list(key_merged.index) == list(key_unsharded.index)
    assert key_merged.option_a_is.tolist() == key_unsharded.option_a_is.tolist()
    assert key_merged.raw_judgement.tolist() == key_unsharded.raw_judgement.tolist()
    # Shard 0 seeds are 42 + k with the production k = i // num_shards.
    shard0 = merged[merged.candidate_id.isin(
        {f"c{i}" for i in range(len(frame)) if i % 4 == 0})]
    assert per_shard_fakes[0].seeds == [
        CONFIG.seed + i // num_shards for i in range(len(frame)) if i % 4 == 0]


# ------------------------------------------------------- merge-judged coverage


def _shard_table(positions: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "position": positions,
        "candidate_id": [f"c{p}" for p in positions],
        "raw_judgement": [f"Decision: A {p}" for p in positions],
    })


def test_merge_judged_detects_duplicates_and_gaps():
    from repsteer.cli import merge_judged_shards

    merged = merge_judged_shards([_shard_table([0, 4]), _shard_table([1, 5]),
                                  _shard_table([2, 6]), _shard_table([3, 7])])
    assert merged.position.tolist() == list(range(8))
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        merge_judged_shards([_shard_table([0, 1]), _shard_table([1, 3])])
    with pytest.raises(ValueError, match="gap"):
        merge_judged_shards([_shard_table([0, 2]), _shard_table([3])])
    with pytest.raises(ValueError, match="position"):
        merge_judged_shards([pd.DataFrame({"candidate_id": ["c0"]})])


# ------------------------------------------------------------------ --resume


def test_resume_skips_already_judged_rows_and_appends(tmp_path: Path):
    existing = tmp_path / "judged.csv"
    partial = _frame(4).iloc[:2]
    partial_out = judge_steering(FakeJudge(), partial, CONFIG)
    partial_out.to_csv(existing, index=False)
    before = len(partial_out)
    fake = FakeJudge()
    resumed = judge_steering(fake, _frame(4), CONFIG, resume_path=str(existing))
    assert len(resumed) == before + 2  # two rows appended, two kept
    assert len(fake.prompts) == 2  # only the missing rows were judged
    assert resumed.raw_judgement.iloc[0] == partial_out.raw_judgement.iloc[0]


# --------------------------------------------------------- vLLM backend glue


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.calls: list[tuple] = []

    def generate(self, prompts, params):
        self.calls.append((list(prompts), params))
        return self.responses(prompts, params)


class FakeCompletion:
    def __init__(self, text):
        self.text = text


class FakeOutput:
    def __init__(self, text):
        self.outputs = [FakeCompletion(text)]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=None):
        assert enable_thinking is False
        return f"CHAT[{messages[0]['content']}]"

    def __call__(self, text, **kwargs):
        ids = list(range(len(text.split())))
        if kwargs.get("truncation") and kwargs.get("max_length", 10**9) < len(ids):
            ids = ids[: kwargs["max_length"]]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{i}" for i in ids)


def _unchat(p: str) -> str:
    return p[len("CHAT["):-len("]")] if p.startswith("CHAT[") else p


def _fake_vllm_backend(monkeypatch, responses):
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = lambda model, **kwargs: FakeLLM(responses)
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    backend = VLLMJudgeBackend.__new__(VLLMJudgeBackend)
    backend.llm = FakeLLM(responses)
    backend.tokenizer = FakeTokenizer()
    return backend


def test_vllm_backend_matches_hf_parsed_rows_given_same_raw_texts(monkeypatch):
    """Same raw judge texts → identical parsed rows on the HF and vLLM paths."""

    frame = _frame(6)
    # Deterministic raw text per unique judge prompt, shared by both paths
    # (the HF path issues one prompt per call, so a per-call counter would
    # restart at r0 on every row).
    raw_by_prompt: dict[str, str] = {}
    counter = iter(range(1000))

    def raw_for(prompt: str) -> str:
        if prompt not in raw_by_prompt:
            raw_by_prompt[prompt] = (
                f"Broken_A: false Broken_B: false Reason: r{next(counter)} Decision: A")
        return raw_by_prompt[prompt]

    class HFStub:
        def generate(self, prompts, **kwargs):
            return [raw_for(p) for p in prompts]

    hf_judged = judge_steering(HFStub(), frame, CONFIG)

    def responses(prompts, params):
        # prompts are chat-templated ("CHAT[...]"); strip the wrapper to look
        # the raw judgement up by the original judge prompt.
        return [FakeOutput(raw_for(_unchat(p))) for p in prompts]

    backend = _fake_vllm_backend(monkeypatch, responses)
    vllm_judged = judge_steering_vllm(backend, frame, CONFIG)
    columns = ("option_a_is", "option_b_is", "string_preferred", "steering_preferred",
               "baseline_preferred", "tie", "judge_broken_steered", "judge_broken_baseline",
               "judge_reason", "valid_decision", "raw_judgement", "position")
    for column in columns:
        assert hf_judged[column].tolist() == vllm_judged[column].tolist(), column
    assert vllm_judged.judge_backend.tolist() == ["vllm"] * len(vllm_judged)
    assert hf_judged.judge_backend.tolist() == ["hf"] * len(hf_judged)
    # Greedy batched params; every prompt went through one batched call.
    assert len(backend.llm.calls) == 1
    prompts, params = backend.llm.calls[0]
    assert len(prompts) == len(frame)
    assert params.temperature == 0.0 and params.max_tokens == CONFIG.max_new_tokens
    assert all(p.startswith("CHAT[") for p in prompts)


# --------------------------------------------------------- refusal heuristic


def test_refusal_heuristic_column_only_when_configured():
    frame = _frame(4).copy()
    frame["steered_continuation"] = ["I'm sorry, I cannot help.",
                                     "totally fine", "As an AI", "ok"]
    substrings = list(REFUSAL_SUBSTRINGS)
    with_substr = judge_steering(FakeJudge(), frame, CONFIG,
                                 backend_config={"refusal_substrings": substrings})
    assert with_substr.refusal_heuristic.tolist() == [True, False, True, False]
    without = judge_steering(FakeJudge(), frame, CONFIG)
    assert "refusal_heuristic" not in without.columns


def test_refusal_heuristic_matches_arditi_substring_semantics():
    assert refusal_heuristic("I'M SORRY, I cannot do that") is True
    assert refusal_heuristic("As a language model, I am unable to") is True
    assert refusal_heuristic("I'm unable to help with that") is True
    assert refusal_heuristic("Here is how you do it") is False
    assert refusal_heuristic("") is False


def test_resume_key_identity():
    row = pd.Series({"candidate_id": "c3", "layer": 5, "coefficient": 1.5})
    assert resume_key(row) == ("c3", "5", "1.5")


def test_parse_judgement_validity_flag_still_production_semantics():
    assert parse_judgement("Decision: A")["valid_decision"] is True
    assert parse_judgement("nothing useful")["preferred"] == "T"
    assert parse_judgement("nothing useful")["valid_decision"] is False
