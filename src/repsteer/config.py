from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import Pooling, PositionMode, VariableSpec


@dataclass(frozen=True)
class DatasetConfig:
    path: str
    text_column: str
    id_column: str
    variables: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    tokenizer_id: str | None = None
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    use_chat_template: bool = True
    layer_path: str | None = None


@dataclass(frozen=True)
class SplitConfig:
    vector_per_cell: int = 20
    eval_per_cell: int = 20
    geometry_per_cell: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class GenerationConfig:
    max_length: int = 2048
    max_new_tokens: int = 96
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    seed: int = 42


@dataclass(frozen=True)
class SteeringConfig:
    layers: tuple[int, ...]
    coefficients: tuple[float, ...]
    pooling: Pooling = "mean"
    normalize_vector: bool = True
    positions: PositionMode = "all"
    batch_size: int = 8
    max_length: int = 512
    prompt_template: str = "Continue the following text naturally:\n{text}"


@dataclass(frozen=True)
class GeometryConfig:
    layers: tuple[int, ...]
    engine: str = "decodanda"
    poolings: tuple[Pooling, ...] = ("mean",)
    zscore: bool = True
    n_resamplings: int = 50
    ndata: int = 26
    sd_ndata: int = 8
    sd_training_fraction: float = 0.75
    sd_cross_validations: int = 25
    sd_accuracy_threshold: float = 0.60
    max_samples_per_condition: int | None = None
    max_sd_dichotomies: int | None = None
    seed: int = 42
    batch_size: int = 8
    max_length: int = 512


@dataclass(frozen=True)
class JudgeConfig:
    model: ModelConfig
    attribute_descriptions: dict[str, str]
    prompt_template: str | None = None
    max_new_tokens: int = 256


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig
    target: VariableSpec
    split: SplitConfig
    generation: GenerationConfig
    steering: SteeringConfig
    geometry: GeometryConfig
    judge: JudgeConfig | None = None


def _tuple_values(values: dict[str, list[str]]) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for name, pair in values.items():
        if len(pair) != 2:
            raise ValueError(f"Variable {name!r} must have exactly two values")
        result[name] = (str(pair[0]), str(pair[1]))
    return result


def _model(raw: dict[str, Any]) -> ModelConfig:
    return ModelConfig(**raw)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment JSON file."""

    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    dataset_raw = dict(raw["dataset"])
    dataset_raw["variables"] = _tuple_values(dataset_raw["variables"])
    target_raw = dict(raw["target"])
    target_raw["condition_on"] = tuple(target_raw.get("condition_on", ()))
    steering_raw = dict(raw["steering"])
    steering_raw["layers"] = tuple(steering_raw["layers"])
    steering_raw["coefficients"] = tuple(steering_raw["coefficients"])
    geometry_raw = dict(raw["geometry"])
    geometry_raw["layers"] = tuple(geometry_raw["layers"])
    geometry_raw["poolings"] = tuple(geometry_raw.get("poolings", ("mean",)))
    judge = None
    if raw.get("judge"):
        judge_raw = dict(raw["judge"])
        judge_raw["model"] = _model(judge_raw["model"])
        judge = JudgeConfig(**judge_raw)
    config = ExperimentConfig(
        dataset=DatasetConfig(**dataset_raw),
        model=_model(raw["model"]),
        target=VariableSpec(**target_raw),
        split=SplitConfig(**raw.get("split", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        steering=SteeringConfig(**steering_raw),
        geometry=GeometryConfig(**geometry_raw),
        judge=judge,
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    variables = config.dataset.variables
    if config.target.name not in variables:
        raise ValueError(f"Target {config.target.name!r} is not declared in dataset.variables")
    expected = variables[config.target.name]
    if (config.target.negative, config.target.positive) != expected:
        raise ValueError("Target negative/positive values must match dataset variable order")
    unknown = set(config.target.condition_on) - set(variables)
    if unknown:
        raise ValueError(f"Unknown conditioning variables: {sorted(unknown)}")
    if config.target.name in config.target.condition_on:
        raise ValueError("A target cannot condition on itself")
    if config.geometry.engine not in {"decodanda", "native"}:
        raise ValueError("geometry.engine must be decodanda or native")
    if not config.steering.layers or not config.geometry.layers:
        raise ValueError("Steering and geometry layer lists cannot be empty")
    if any(layer < 0 for layer in (*config.steering.layers, *config.geometry.layers)):
        raise ValueError("Layer indices must be non-negative")
