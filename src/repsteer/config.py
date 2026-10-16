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
    layer_regex: str | None = None
    model_registry: str | None = None


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
    # Production ``SearchCaaModel.generate`` semantics: collapse whitespace in the
    # prompt before the chat template and in the decoded continuation.
    normalize_whitespace: bool = False
    # Retry ladder [(seed, do_sample), (seed + 1, do_sample), (seed, greedy)] on empty output.
    retry_empty: bool = False
    # Pass ``pad_token_id = eos_token_id`` to ``model.generate`` (production).
    pad_with_eos: bool = False
    # Baseline seed = seed + baseline_seed_offset + evaluation row number.
    baseline_seed_offset: int = 0
    # Steered seed = seed + int(sha256(material)[:8], 16) % 1_000_000 where material is
    # this template formatted with setup/model_id/task/layer/alpha/id; None -> seed + row.
    steered_seed_material: str | None = None

    def decoding_kwargs(self) -> dict[str, Any]:
        """Keyword arguments accepted by every backend decoding entry point."""

        return {
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "seed": self.seed,
        }

    def generate_kwargs(self) -> dict[str, Any]:
        """``decoding_kwargs`` plus the production post-processing switches of ``generate``."""

        return {
            **self.decoding_kwargs(),
            "normalize_whitespace": self.normalize_whitespace,
            "retry_empty": self.retry_empty,
            "pad_with_eos": self.pad_with_eos,
        }


@dataclass(frozen=True)
class VectorSamplingConfig:
    """Production contrast sampling: ``DataFrame.sample(per_class, random_state=seed + layer + offset)``."""

    per_class: int
    seed: int = 42
    positive_seed_offset: int = 0
    negative_seed_offset: int = 0
    # caa_conditioned_steering.sample_texts falls back to replace=True when a class is short.
    replace_if_short: bool = False


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
    # Per-row template dispatch: prompt_templates[row[prompt_template_column]].
    prompt_template_column: str | None = None
    prompt_templates: dict[str, str] | None = None
    # Labels recorded in output ``task``/``setup``/``target_attribute`` columns; the
    # judge renders its target line from (task, target_attribute).
    task: str | None = None
    setup: str | None = None
    target_labels: dict[str, str] = field(default_factory=dict)
    sampling: VectorSamplingConfig | None = None


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
    # Decodanda engine (production geometry_olmo_reduced_full.py / audit) only.
    protocol: str = "production"
    variables: tuple[str, ...] | None = None
    target_fields: tuple[str, ...] | None = None
    n_per_cell: int = 50
    auto_reduce_n_per_cell: bool = False
    audit_n_per_cell: int | None = None
    audit_training_fraction: float = 0.75
    audit_cross_validations: int = 25


@dataclass(frozen=True)
class JudgeConfig:
    model: ModelConfig
    # task -> "line with {target}" or {target_attribute: line}; rendered as the
    # judge prompt's target line (production pairwise_string_prompt).
    attribute_descriptions: dict[str, str | dict[str, str]]
    prompt_template: str | None = None
    max_new_tokens: int = 1024
    max_length: int = 2048
    seed: int = 42
    # Production A/B assignment: the judged table is dealt round-robin to
    # ``num_shards`` jobs (row i -> job i % n, position k = i // n) within each
    # ``shard_group_by`` group; the steered output is option A when (k + job) is odd.
    num_shards: int = 1
    shard_group_by: tuple[str, ...] = ()
    # Production repair_invalid_judgements.py: re-judge rows without a valid
    # final decision once with the strict suffix appended.
    repair_invalid: bool = False
    repair_max_new_tokens: int = 2048
    repair_seed: int = 910000


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
    raw = dict(raw)
    for optional in ("layer_path", "layer_regex", "model_registry", "tokenizer_id"):
        if raw.get(optional) is None:
            raw.pop(optional, None)
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
    for optional in ("variables", "target_fields"):
        if geometry_raw.get(optional) is not None:
            geometry_raw[optional] = tuple(geometry_raw[optional])
    judge = None
    if raw.get("judge"):
        judge_raw = dict(raw["judge"])
        judge_raw["model"] = _model(judge_raw["model"])
        judge_raw["shard_group_by"] = tuple(judge_raw.get("shard_group_by", ()))
        judge = JudgeConfig(**judge_raw)
    if steering_raw.get("sampling") is not None:
        steering_raw["sampling"] = VectorSamplingConfig(**steering_raw["sampling"])
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
    steering = config.steering
    if (steering.prompt_templates is None) != (steering.prompt_template_column is None):
        raise ValueError("steering.prompt_templates and prompt_template_column must be set together")
    if config.judge is not None and config.judge.num_shards < 1:
        raise ValueError("judge.num_shards must be positive")
    if config.geometry.engine not in {"decodanda", "native"}:
        raise ValueError("geometry.engine must be decodanda or native")
    if config.geometry.protocol not in {"production", "matched_audit"}:
        raise ValueError("geometry.protocol must be production or matched_audit")
    geometry_variables = config.geometry.variables or tuple(variables)
    unknown_targets = set(config.geometry.target_fields or ()) - set(geometry_variables)
    if unknown_targets:
        raise ValueError(f"geometry.target_fields not in geometry variables: {sorted(unknown_targets)}")
    if not config.steering.layers or not config.geometry.layers:
        raise ValueError("Steering and geometry layer lists cannot be empty")
    if any(layer < 0 for layer in (*config.steering.layers, *config.geometry.layers)):
        raise ValueError("Layer indices must be non-negative")
