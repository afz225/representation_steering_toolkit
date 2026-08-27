from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .config import GenerationConfig, SteeringConfig
from .models.base import ModelBackend
from .types import Intervention, VariableSpec
from .vectors import SteeringVector, condition_for_row, vector_index


class _FormatDict(dict):
    def __missing__(self, key):
        raise KeyError(f"Prompt template references missing column {key!r}")


def format_prompt(template: str, row: pd.Series, text_column: str, source: str, target: str) -> str:
    values = _FormatDict({str(key): value for key, value in row.items()})
    values.update(text=str(row[text_column]), source=source, target=target)
    return template.format_map(values)


def run_steering(
    backend: ModelBackend,
    frame: pd.DataFrame,
    text_column: str,
    id_column: str,
    spec: VariableSpec,
    vectors: list[SteeringVector],
    steering: SteeringConfig,
    generation: GenerationConfig,
) -> pd.DataFrame:
    """Generate paired baseline and opposite-target steered continuations."""

    index = vector_index(vectors)
    rows: list[dict] = []
    generation_args = asdict(generation)
    for row_number, (_, row) in enumerate(frame.iterrows()):
        source = str(row[spec.name])
        target = spec.opposite(source)
        prompt = format_prompt(steering.prompt_template, row, text_column, source, target)
        paired_args = dict(generation_args)
        paired_args["seed"] = generation.seed + row_number
        baseline = backend.generate([prompt], intervention=None, **paired_args)[0]
        condition = condition_for_row(row, spec.condition_on)
        for layer in steering.layers:
            key = (layer, condition)
            if key not in index:
                raise KeyError(f"No vector for layer={layer}, condition={dict(condition)}")
            signed_vector = spec.sign_for_target(target) * index[key].values
            for coefficient in steering.coefficients:
                intervention = Intervention(
                    layer=layer,
                    vector=signed_vector,
                    coefficient=coefficient,
                    positions=steering.positions,
                )
                steered_args = dict(paired_args)
                steered = backend.generate(
                    [prompt], intervention=intervention, **steered_args
                )[0]
                result = row.to_dict()
                result.update(
                    source_value=source,
                    target_value=target,
                    layer=layer,
                    coefficient=coefficient,
                    vector_condition="|".join(f"{k}={v}" for k, v in condition) or "general",
                    prompt=prompt,
                    baseline_continuation=baseline,
                    steered_continuation=steered,
                )
                result[id_column] = row[id_column]
                rows.append(result)
    return pd.DataFrame(rows)
