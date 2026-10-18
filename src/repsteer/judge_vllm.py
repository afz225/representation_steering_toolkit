"""vLLM judge backend: batched greedy judging with the same prompts, parsing,
and output columns as the production HF path (``repsteer judge`` with
``judge.backend = "vllm"``).

vLLM lives in a separate environment, so the import is lazy; everything in
here is opt-in. The chat-template handling mirrors
``HuggingFaceBackend._apply_chat_template`` (enable_thinking=False, retry
without it on TypeError, fall back to the raw prompt) so identical raw judge
prompts produce identical parsed rows on both backends.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import JudgeConfig
from .judge import (REPAIR_SUFFIX, _ab, _task_and_target, assign_options,
                    judgement_fields, parse_judgement, refusal_heuristic,
                    render_judge_prompt, resume_key)
from .text import safe_text


def _apply_chat_template(tokenizer, prompt: str) -> str:
    """Same fallback chain as the production HF judge prompt formatting."""

    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return prompt
    except Exception:
        return prompt


def _truncate(tokenizer, text: str, max_length: int) -> str:
    """Token-level truncation to ``max_length`` tokens, then decode (same
    truncation budget as the HF path tokenizing with ``max_length``)."""

    ids = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
    if len(ids) < len(tokenizer(text, add_special_tokens=False)["input_ids"]):
        return tokenizer.decode(ids, skip_special_tokens=True)
    return text


class VLLMJudgeBackend:
    """Minimal ``ModelBackend``-shaped wrapper around ``vllm.LLM`` for judging.

    ``vllm`` is imported lazily on first use (it lives in a separate env).
    Generation is greedy (temperature 0) with the same ``max_new_tokens`` /
    ``max_length`` budget as the HF path; the rendered chat-template prompt is
    tokenized with truncation to ``max_length``. ``generate`` keeps the
    production ``ModelBackend.generate`` signature so ``judge_steering`` treats
    both backends identically.
    """

    def __init__(self, config: JudgeConfig, **llm_kwargs):
        import torch
        import vllm  # lazy: separate env

        model_kwargs = {"tensor_parallel_size": max(torch.cuda.device_count(), 1)}
        if config.model.dtype:
            model_kwargs["dtype"] = config.model.dtype
        if config.model.trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        model_kwargs.update(llm_kwargs)
        self.llm = vllm.LLM(model=config.model.model_id, **model_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_length: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        seed: int = 0,
        intervention=None,
        normalize_whitespace: bool = False,
        retry_empty: bool = False,
        pad_with_eos: bool = False,
    ) -> list[str]:
        """Batched greedy decode of every prompt through one LLM call.

        The rendered chat-template prompt is NOT whitespace-normalized before
        templating (the HF judge path passes ``normalize_whitespace=False`` and
        applies ``safe_text`` to the decoded response only, in
        ``judge._judge_once``); mirror that here.
        Greedy only; ``do_sample`` must be False.
        """

        import vllm  # lazy: separate env

        if do_sample:
            raise ValueError("The judge backend is greedy; do_sample must be False")
        texts = list(prompts)
        formatted = [_apply_chat_template(self.tokenizer, text) for text in texts]
        formatted = [_truncate(self.tokenizer, text, max_length) for text in formatted]
        params = vllm.SamplingParams(
            temperature=0.0, top_p=top_p, max_tokens=max_new_tokens, seed=None,
        )
        outputs = self.llm.generate(formatted, params)
        results = [output.outputs[0].text if output.outputs else "" for output in outputs]
        # The HF path decodes with skip_special_tokens then safe_texts the
        # response (judge._judge_once); mirror that here.
        return [safe_text(text) for text in results]


def judge_steering_vllm(backend: VLLMJudgeBackend, frame: pd.DataFrame, config: JudgeConfig,
                        *, judge_model_id: str = "", shard_index: int = 0,
                        num_exec_shards: int = 1,
                        resume_path: str | None = None,
                        prompt_preset: str = "production") -> pd.DataFrame:
    """Judge every steered/baseline pair through the vLLM backend.

    Identical prompt rendering, A/B assignment, seeds, parse, and output
    columns as ``judge.judge_steering`` (including ``judge_backend="vllm"`` and
    the optional ``refusal_heuristic`` column), but all prompts are issued in
    ONE batched ``llm.generate`` call per judge pass (main + optional repair)
    instead of one generation per row.
    """

    from . import judge as _judge

    frame = frame.reset_index(drop=True)
    template = config.prompt_template
    if template is None and prompt_preset == "generic":
        template = _judge.GENERIC_JUDGE_PROMPT
    if template is not None:
        config = _replace(config, prompt_template=template)
    substrings = config.refusal_substrings
    steered_is_a, positions = assign_options(frame, config.num_shards, config.shard_group_by)
    resume_keys: set[tuple] | None = None
    previous: pd.DataFrame | None = None
    if resume_path and Path(resume_path).exists():
        previous = pd.read_csv(resume_path, keep_default_na=False)
        resume_keys = {resume_key(row) for _, row in previous.iterrows()}
    selected: list[int] = []
    prompts: list[str] = []
    seeds: list[int] = []
    row_order = np.arange(len(frame))
    shard_of = row_order % num_exec_shards
    for index, row in frame.iterrows():
        if num_exec_shards > 1 and shard_of[index] != shard_index:
            continue
        key = resume_key(row)
        if resume_keys is not None and key in resume_keys:
            continue
        task, target = _task_and_target(row)
        a, b = _ab(row, bool(steered_is_a[index]))
        prompt = render_judge_prompt(config, task, target, a, b, str(row.get("prompt", "")))
        selected.append(index)
        prompts.append(prompt)
        seeds.append(config.seed + int(positions[index]))
    raws = backend.generate(prompts, max_length=config.max_length,
                            max_new_tokens=config.max_new_tokens, temperature=0.0,
                            top_p=1.0, do_sample=False, seed=0,
                            pad_with_eos=True)
    rows: list[dict] = []
    for index, prompt, raw in zip(selected, prompts, raws):
        row = frame.loc[index]
        parsed = parse_judgement(raw)
        parsed["raw_judgement"] = raw
        fields = judgement_fields(parsed, bool(steered_is_a[index]))
        if substrings is not None:
            fields["refusal_heuristic"] = refusal_heuristic(row["steered_continuation"], substrings)
        rows.append({**row.to_dict(), **fields,
                     "position": int(positions[index]),
                     "raw_judgement": parsed["raw_judgement"], "judge_model_id": judge_model_id,
                     "judge_backend": "vllm"})
    judged = _concat_judged(frame, rows, previous)
    if config.repair_invalid:
        judged = _repair_invalid_vllm(backend, judged, config)
    return judged


def _repair_invalid_vllm(backend: VLLMJudgeBackend, judged: pd.DataFrame,
                         config: JudgeConfig) -> pd.DataFrame:
    """One batched repair pass mirroring ``judge.repair_invalid_judgements``."""

    from . import judge as _judge

    judged = judged.copy()
    indices: list[int] = []
    prompts: list[str] = []
    for index, row in judged.iterrows():
        if _judge.parse_judgement(row["raw_judgement"])["valid_decision"]:
            continue
        steered_is_a = str(row["option_a_is"]).strip().lower() == "steered"
        task, target = _task_and_target(row)
        a, b = _ab(row, steered_is_a)
        prompt = render_judge_prompt(config, task, target, a, b, str(row.get("prompt", "")))
        indices.append(index)
        prompts.append(prompt + REPAIR_SUFFIX)
    if not prompts:
        return judged
    raws = backend.generate(prompts, max_length=config.max_length,
                            max_new_tokens=config.repair_max_new_tokens, temperature=0.0,
                            top_p=1.0, do_sample=False, seed=0,
                            pad_with_eos=True)
    for ordinal, (index, raw) in enumerate(zip(indices, raws)):
        row = judged.loc[index]
        parsed = parse_judgement(raw)
        steered_is_a = str(row["option_a_is"]).strip().lower() == "steered"
        for column, value in judgement_fields(parsed, steered_is_a).items():
            judged.at[index, column] = value
        judged.at[index, "raw_judgement"] = raw
    return judged


def _concat_judged(frame: pd.DataFrame, rows: list[dict],
                   previous: pd.DataFrame | None) -> pd.DataFrame:
    parts = []
    if previous is not None:
        parts.append(previous)
    if rows:
        parts.append(pd.DataFrame(rows))
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()
