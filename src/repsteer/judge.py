from __future__ import annotations

import hashlib
import re

import pandas as pd

from .config import GenerationConfig, JudgeConfig
from .models.base import ModelBackend

DEFAULT_JUDGE_PROMPT = """You are a conservative pairwise judge for generated continuations. Evaluate ONLY A and B for the target below.

First decide whether each continuation is broken as generated text. Broken means the text is not usable as coherent fluent continuation text because it is empty, incoherent, semantically nonsensical, grammatically broken, hallucinated boilerplate/labels, repetitive, truncated, or contains severe formatting/artifact corruption. A continuation is NOT broken merely because it fails to match the target attribute. Wrong target attribute is a target-match failure, not a broken-output failure.

Then compare target match among outputs that are not broken. A tie is correct when both continuations are acceptable but neither has a clear, sustained target advantage. Do not pick a winner for one adjective, one synonym, length alone, or small wording differences. If one output is fluent but off-target and the other is fluent and on-target, prefer the on-target output. If both are fluent but equally off-target, choose tie.

Criteria, in order:
1. Brokenness/fluency: broken continuations cannot win, but off-target fluent continuations are not broken.
2. Target match: the continuation consistently displays the requested attribute.
3. Meaningful margin: the winning continuation must be clearly closer to the target than the other.

Return:
- Decision: A if A clearly satisfies the target better and A is not broken.
- Decision: B if B clearly satisfies the target better and B is not broken.
- Decision: T if both are broken, neither clearly wins, or the only differences are minor.

Output exactly this structure, with the final Decision line at the very end:
Broken_A: true or false
Broken_B: true or false
Reason: one short explanation
Decision: <A, B, or T>

Replace the placeholder on the final line with exactly one of these three strings: Decision: A, Decision: B, or Decision: T

Target:
{target_description}

Original prompt:
{prompt}

A: {a}
B: {b}"""


_DECISION = re.compile(r"(?i)Decision\s*:\s*(A|B|TIE|T)\b")
_BROKEN_COMBINED = re.compile(r"(?i)Broken\s*:\s*(NONE|A|B|BOTH)\b")
_BROKEN_A = re.compile(r"(?i)Broken[_ -]?A\s*:\s*(true|false|yes|no|1|0)\b")
_BROKEN_B = re.compile(r"(?i)Broken[_ -]?B\s*:\s*(true|false|yes|no|1|0)\b")


def _bool_label(value: str) -> bool:
    return value.strip().lower() in {"true", "yes", "1"}


def parse_judgement(text: str) -> tuple[str, str]:
    """Parse final generated-string decisions in current and legacy formats."""

    decisions = _DECISION.findall(str(text))
    if not decisions:
        raise ValueError("Judge response lacks a final Decision line")
    decision = decisions[-1].upper()
    if decision == "T":
        decision = "TIE"

    combined = _BROKEN_COMBINED.findall(str(text))
    if combined:
        return decision, combined[-1].upper()

    broken_a = _BROKEN_A.findall(str(text))
    broken_b = _BROKEN_B.findall(str(text))
    if not broken_a or not broken_b:
        return decision, "UNKNOWN"
    a_value = _bool_label(broken_a[-1])
    b_value = _bool_label(broken_b[-1])
    if a_value and b_value:
        broken = "BOTH"
    elif a_value:
        broken = "A"
    elif b_value:
        broken = "B"
    else:
        broken = "NONE"
    return decision, broken


def _steered_is_a(example_id: str, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    return bool(digest[0] & 1)


def judge_steering(
    backend: ModelBackend,
    frame: pd.DataFrame,
    config: JudgeConfig,
    generation: GenerationConfig,
    *,
    id_column: str,
    target_name: str,
) -> pd.DataFrame:
    template = config.prompt_template or DEFAULT_JUDGE_PROMPT
    target_description = config.attribute_descriptions[target_name]
    rows: list[dict] = []
    for index, row in frame.iterrows():
        example_id = str(row[id_column])
        steered_is_a = _steered_is_a(example_id, generation.seed + int(index))
        a = row["steered_continuation"] if steered_is_a else row["baseline_continuation"]
        b = row["baseline_continuation"] if steered_is_a else row["steered_continuation"]
        prompt = template.format(
            target_description=target_description.format(target=row["target_value"]),
            prompt=row["prompt"],
            a=a,
            b=b,
        )
        response = backend.generate(
            [prompt],
            max_length=generation.max_length,
            max_new_tokens=config.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            seed=generation.seed + int(index),
        )[0]
        try:
            decision, broken = parse_judgement(response)
            parse_error = ""
        except ValueError as exc:
            decision, broken, parse_error = "PARSE_ERROR", "PARSE_ERROR", str(exc)
        if decision in ("TIE", "PARSE_ERROR"):
            preference = decision
        elif (decision == "A") == steered_is_a:
            preference = "STEERED"
        else:
            preference = "BASELINE"
        if broken == "UNKNOWN":
            broken_steered = pd.NA
            broken_baseline = pd.NA
        else:
            broken_steered = broken in (("A", "BOTH") if steered_is_a else ("B", "BOTH"))
            broken_baseline = broken in (("B", "BOTH") if steered_is_a else ("A", "BOTH"))
        result = row.to_dict()
        result.update(
            judge_response=response,
            judge_decision=decision,
            judge_preference=preference,
            broken_steered=broken_steered,
            broken_baseline=broken_baseline,
            judge_parse_error=parse_error,
            steered_position="A" if steered_is_a else "B",
        )
        rows.append(result)
    return pd.DataFrame(rows)


def _nullable_bool_mean(series: pd.Series) -> float:
    value = series.astype("boolean").mean()
    return float("nan") if pd.isna(value) else float(value)


def summarize_judgements(
    frame: pd.DataFrame,
    group_columns: tuple[str, ...] = ("layer", "coefficient", "vector_condition"),
) -> pd.DataFrame:
    """Aggregate parsed pairwise preferences without treating parse errors as ties."""

    required = {*group_columns, "judge_preference", "broken_steered", "broken_baseline"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Judgement table is missing columns: {sorted(missing)}")
    rows = []
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for key, group in frame.groupby(grouper, dropna=False, sort=True):
        keys = (key,) if len(group_columns) == 1 else key
        valid = group[group.judge_preference != "PARSE_ERROR"]
        record = dict(zip(group_columns, keys))
        record.update(
            n_examples=len(group),
            n_valid=len(valid),
            parse_error_rate=1.0 - len(valid) / len(group),
            steering_preference_rate=float((valid.judge_preference == "STEERED").mean()),
            baseline_preference_rate=float((valid.judge_preference == "BASELINE").mean()),
            tie_rate=float((valid.judge_preference == "TIE").mean()),
            steered_broken_rate=_nullable_bool_mean(valid.broken_steered),
            baseline_broken_rate=_nullable_bool_mean(valid.broken_baseline),
        )
        rows.append(record)
    return pd.DataFrame(rows)
