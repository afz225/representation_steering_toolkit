from __future__ import annotations

import hashlib
from collections.abc import Iterable

import pandas as pd

from .config import GenerationConfig, SteeringConfig
from .models.base import ModelBackend
from .text import safe_text
from .types import Intervention, VariableSpec
from .vectors import SteeringVector, condition_for_row, vector_index


class _FormatDict(dict):
    def __missing__(self, key):
        raise KeyError(f"Prompt template references missing column {key!r}")


def format_prompt(template: str, row: pd.Series, text_column: str, source: str, target: str) -> str:
    """Render a prompt; ``{text}`` is the raw text, ``{safe_text}`` its whitespace-collapsed form."""

    values = _FormatDict({str(key): value for key, value in row.items()})
    text = str(row[text_column])
    values.update(text=text, safe_text=safe_text(text), source=source, target=target)
    return template.format_map(values)


def template_for_row(steering: SteeringConfig, row: pd.Series) -> str:
    if steering.prompt_templates is None:
        return steering.prompt_template
    key = str(row[steering.prompt_template_column])
    if key not in steering.prompt_templates:
        raise KeyError(f"No prompt template for {steering.prompt_template_column}={key!r}")
    return steering.prompt_templates[key]


def steered_seed(generation: GenerationConfig, row_number: int, **fields) -> int:
    """Production per-row steered seed (sha256 of the seed material) or ``seed + row``."""

    if generation.steered_seed_material is None:
        return generation.seed + row_number
    material = generation.steered_seed_material.format(**fields)
    return generation.seed + int(hashlib.sha256(material.encode()).hexdigest()[:8], 16) % 1_000_000


def run_steering(
    backend: ModelBackend,
    frame: pd.DataFrame,
    text_column: str,
    id_column: str,
    spec: VariableSpec,
    vectors: list[SteeringVector],
    steering: SteeringConfig,
    generation: GenerationConfig,
    *,
    model_id: str = "",
) -> pd.DataFrame:
    """Generate paired baseline and opposite-target steered continuations.

    Production order and semantics (gemma4_eval200_steering.py /
    gemma4_toxicity_steering.py): one unsteered baseline per evaluation row
    (seed ``seed + baseline_seed_offset + row``), then rows ordered layer ->
    coefficient -> evaluation row. The hook adds ``sign * coefficient`` times the
    (unsigned) vector, where sign is +1 when the target is the positive pole.
    """

    index = vector_index(vectors)
    task = steering.task or spec.name
    setup = steering.setup or ("conditioned" if spec.condition_on else "general")
    decoding = generation.generate_kwargs()
    prepared = []
    for row_number, (_, row) in enumerate(frame.iterrows()):
        source = str(row[spec.name])
        target = spec.opposite(source)
        prompt = format_prompt(template_for_row(steering, row), row, text_column, source, target)
        baseline = backend.generate(
            [prompt],
            intervention=None,
            **{**decoding, "seed": generation.seed + generation.baseline_seed_offset + row_number},
        )[0]
        prepared.append((row_number, row, source, target, prompt, baseline))

    rows: list[dict] = []
    for layer in steering.layers:
        for coefficient in steering.coefficients:
            for row_number, row, source, target, prompt, baseline in prepared:
                condition = condition_for_row(row, spec.condition_on)
                key = (layer, condition)
                if key not in index:
                    raise KeyError(f"No vector for layer={layer}, condition={dict(condition)}")
                signed = spec.sign_for_target(target) * float(coefficient)
                intervention = Intervention(
                    layer=layer,
                    vector=index[key].values,
                    coefficient=signed,
                    positions=steering.positions,
                )
                seed = steered_seed(
                    generation, row_number, setup=setup, model_id=model_id, task=task,
                    layer=layer, alpha=float(coefficient), id=row[id_column],
                )
                steered = backend.generate(
                    [prompt], intervention=intervention, **{**decoding, "seed": seed}
                )[0]
                result = row.to_dict()
                result.update(
                    task=task,
                    setup=setup,
                    source_value=source,
                    target_value=target,
                    target_attribute=steering.target_labels.get(target, target),
                    layer=layer,
                    coefficient=coefficient,
                    signed_coefficient=signed,
                    vector_condition="|".join(f"{k}={v}" for k, v in condition) or "general",
                    prompt=prompt,
                    baseline_continuation=baseline,
                    steered_continuation=steered,
                )
                result[id_column] = row[id_column]
                rows.append(result)
    return pd.DataFrame(rows)


def order_for_judging(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-task steering tables in production judged-input order.

    Production merged every layer's generations file, whose rows run task (in
    the given order) -> alpha -> evaluation row; ``run_steering`` tables run
    layer -> coefficient -> row, so a stable sort on (layer, input rank)
    reproduces ``results-gemma4-steering-eval200/{setup}_generations.csv``.
    """

    parts = [frame.assign(_input_rank=rank) for rank, frame in enumerate(frames)]
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.sort_values(["layer", "_input_rank"], kind="stable")
    return merged.drop(columns="_input_rank").reset_index(drop=True)
