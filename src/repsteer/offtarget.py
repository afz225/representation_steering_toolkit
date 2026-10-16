"""Representational and judged collateral effects of steering."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import GenerationConfig
from .models.base import ModelBackend
from .steering import format_prompt
from .types import Intervention, PositionMode, VariableSpec
from .vectors import SteeringVector, condition_for_row, vector_index


@dataclass(frozen=True)
class AttributeLabels:
    """Allowed judge labels and dataset-value-to-label mapping for an attribute."""
    description: str
    labels: tuple[str, ...]
    source_map: Mapping[str, str]

    def source_label(self, value: object) -> str:
        key = str(value)
        if key not in self.source_map:
            raise ValueError(f"No judge-label mapping for dataset value {key!r}")
        return self.source_map[key]


OFFTARGET_JUDGE_PROMPT = """You are evaluating collateral changes between two generated continuations.
Judge only the continuation text, not the original prompt. Label every requested attribute in both outputs.

Broken means unusable generated prose: empty output, incoherence, semantic nonsense, severe grammar failure, hallucinated labels or boilerplate, pathological repetition, truncation, or formatting corruption. An output is NOT broken merely because it omits or fails to match a requested concept, language, or style.

Attributes and allowed labels:
{attribute_instructions}

Return exactly one JSON object with this schema and no markdown:
{{
  "baseline_broken": true or false,
  "steered_broken": true or false,
  "baseline_labels": {{"ATTRIBUTE": "LABEL"}},
  "steered_labels": {{"ATTRIBUTE": "LABEL"}},
  "reason": "brief explanation"
}}
Include one label for every attribute and use only its allowed labels.

Original prompt (context only):
{prompt}

Baseline continuation:
{baseline}

Steered continuation:
{steered}
"""


def measure_projection_drift(
    backend: ModelBackend,
    frame: pd.DataFrame,
    *,
    text_column: str,
    id_column: str,
    steering_spec: VariableSpec,
    steering_vectors: Sequence[SteeringVector],
    measured_specs: Mapping[str, VariableSpec],
    measured_vectors: Mapping[str, Sequence[SteeringVector]],
    intervention_layer: int,
    coefficient: float,
    capture_layers: Sequence[int],
    generation: GenerationConfig,
    prompt_template: str = "Continue the following text naturally:\n{text}",
    positions: PositionMode = "all",
) -> pd.DataFrame:
    """Measure immediate and downstream projection drift along concept axes.

    The concept axis is recalculated at each readout layer. Captured states are
    last-token states averaged over generation forward calls, matching the prior
    online off-target experiment.
    """
    steering_index = vector_index(steering_vectors)
    measured_indices = {name: vector_index(vectors) for name, vectors in measured_vectors.items()}
    generation_args = generation.decoding_kwargs()
    rows = []
    for row_number, (_, row) in enumerate(frame.iterrows()):
        source = str(row[steering_spec.name])
        target = steering_spec.opposite(source)
        prompt = format_prompt(prompt_template, row, text_column, source, target)
        condition = condition_for_row(row, steering_spec.condition_on)
        key = (intervention_layer, condition)
        if key not in steering_index:
            raise KeyError(f"No steering vector for layer={intervention_layer}, condition={dict(condition)}")
        signed = steering_spec.sign_for_target(target) * steering_index[key].values
        intervention = Intervention(intervention_layer, signed, coefficient, positions)
        args = dict(generation_args)
        args["seed"] = generation.seed + row_number
        baseline_text, baseline_states = backend.generate_with_representations(
            prompt, capture_layers=capture_layers, intervention=None, **args)
        steered_text, steered_states = backend.generate_with_representations(
            prompt, capture_layers=capture_layers, intervention=intervention, **args)
        for measured_name, measured_spec in measured_specs.items():
            axis_condition = condition_for_row(row, measured_spec.condition_on)
            for readout_layer in capture_layers:
                axis_key = (readout_layer, axis_condition)
                if axis_key not in measured_indices[measured_name]:
                    raise KeyError(f"No {measured_name} axis for layer={readout_layer}, condition={dict(axis_condition)}")
                axis_record = measured_indices[measured_name][axis_key]
                axis_norm = float(np.linalg.norm(axis_record.values))
                if axis_norm <= 1e-12 or axis_record.class_gap <= 1e-12:
                    raise ValueError(f"Degenerate {measured_name} axis at layer {readout_layer}")
                axis = axis_record.values / axis_norm
                base_projection = float(baseline_states[readout_layer] @ axis)
                steer_projection = float(steered_states[readout_layer] @ axis)
                delta = steer_projection - base_projection
                source_sign = measured_spec.sign_for_target(str(row[measured_name]))
                item = row.to_dict()
                item.update({
                    "sample_id": row[id_column], "steering_variable": steering_spec.name,
                    "source_value": source, "target_value": target,
                    "intervention_layer": intervention_layer, "readout_layer": readout_layer,
                    "coefficient": coefficient, "positions": positions,
                    "vector_condition": "|".join(f"{k}={v}" for k, v in condition) or "general",
                    "measured_variable": measured_name,
                    "is_off_target": measured_name != steering_spec.name,
                    "baseline_projection": base_projection, "steered_projection": steer_projection,
                    "projection_delta": delta, "absolute_projection_drift": abs(delta),
                    "normalized_absolute_drift": abs(delta) / axis_record.class_gap,
                    "source_opposed_drift": -source_sign * delta,
                    "baseline_continuation": baseline_text,
                    "steered_continuation": steered_text,
                })
                if measured_name == steering_spec.name:
                    item["target_aligned_drift"] = steering_spec.sign_for_target(target) * delta
                else:
                    item["target_aligned_drift"] = np.nan
                rows.append(item)
    return pd.DataFrame(rows)


def summarize_projection_drift(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = ["setup", "steering_setup", "vector_condition", "steering_variable",
                  "measured_variable", "is_off_target", "intervention_layer",
                  "readout_layer", "coefficient"]
    columns = [name for name in candidates if name in frame]
    return frame.groupby(columns, as_index=False, dropna=False).agg(
        mean_absolute_projection_drift=("absolute_projection_drift", "mean"),
        mean_normalized_absolute_drift=("normalized_absolute_drift", "mean"),
        mean_source_opposed_drift=("source_opposed_drift", "mean"),
        n_examples=("sample_id", "size"))


def _extract_json(text: str) -> dict:
    text = str(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        matches = list(re.finditer(r"\{", text))
        for match in reversed(matches):
            try:
                return json.JSONDecoder().raw_decode(text[match.start():])[0]
            except json.JSONDecodeError:
                pass
    raise ValueError("Judge response contains no parseable JSON object")


def _json_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"Expected a JSON boolean, got {value!r}")


def judge_offtarget_effects(
    backend: ModelBackend,
    frame: pd.DataFrame,
    attributes: Mapping[str, AttributeLabels],
    generation: GenerationConfig,
    *,
    steered_variable: str,
    max_new_tokens: int = 384,
    prompt_template: str = OFFTARGET_JUDGE_PROMPT,
) -> pd.DataFrame:
    """Label baseline/steered attributes jointly and derive off-target flips."""
    instructions = "\n".join(
        f"- {name}: {spec.description}; labels: {', '.join(spec.labels)}"
        for name, spec in attributes.items())
    rows = []
    for row_number, (_, row) in enumerate(frame.iterrows()):
        judge_prompt = prompt_template.format(
            attribute_instructions=instructions, prompt=row["prompt"],
            baseline=row["baseline_continuation"], steered=row["steered_continuation"])
        response = backend.generate(
            [judge_prompt], max_length=generation.max_length, max_new_tokens=max_new_tokens,
            temperature=0.0, top_p=1.0, do_sample=False, seed=generation.seed + row_number)[0]
        try:
            parsed = _extract_json(response)
            base_labels = parsed["baseline_labels"]
            steer_labels = parsed["steered_labels"]
            base_broken = _json_bool(parsed["baseline_broken"])
            steer_broken = _json_bool(parsed["steered_broken"])
            for name, spec in attributes.items():
                if base_labels[name] not in spec.labels or steer_labels[name] not in spec.labels:
                    raise ValueError(f"Invalid label for {name}")
            error = ""
        except (KeyError, TypeError, ValueError) as exc:
            base_labels = steer_labels = {}
            base_broken = steer_broken = pd.NA
            error = str(exc)
        common = row.to_dict()
        common.update({"steered_variable": steered_variable, "judge_response": response,
                       "judge_parse_error": error, "baseline_broken": base_broken,
                       "steered_broken": steer_broken})
        for name, spec in attributes.items():
            item = dict(common)
            source_label = spec.source_label(row[name])
            baseline_label = base_labels.get(name, pd.NA)
            steered_label = steer_labels.get(name, pd.NA)
            valid = not error
            item.update({"measured_variable": name, "is_off_target": name != steered_variable,
                         "source_label": source_label, "baseline_label": baseline_label,
                         "steered_label": steered_label,
                         "offtarget_changed_vs_baseline": (steered_label != baseline_label) if valid else pd.NA,
                         "offtarget_drift_from_source": (steered_label != source_label) if valid else pd.NA})
            rows.append(item)
    return pd.DataFrame(rows)


def _nullable_mean(series: pd.Series) -> float:
    value = series.astype("boolean").mean()
    return float("nan") if pd.isna(value) else float(value)


def summarize_offtarget_effects(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = ["setup", "steering_setup", "vector_condition", "steered_variable",
                  "measured_variable", "is_off_target", "layer", "coefficient"]
    groups = [name for name in candidates if name in frame]
    rows = []
    for key, group in frame.groupby(groups, dropna=False, sort=True):
        keys = (key,) if len(groups) == 1 else key
        valid = group[group.judge_parse_error == ""]
        item = dict(zip(groups, keys))
        item.update({"n_examples": len(group), "n_valid": len(valid),
                     "parse_error_rate": 1.0 - len(valid) / len(group),
                     "steered_broken_rate": _nullable_mean(group.steered_broken),
                     "baseline_broken_rate": _nullable_mean(group.baseline_broken),
                     "changed_vs_baseline_rate": _nullable_mean(group.offtarget_changed_vs_baseline),
                     "drift_from_source_rate": _nullable_mean(group.offtarget_drift_from_source)})
        rows.append(item)
    return pd.DataFrame(rows)
