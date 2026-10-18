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
    # --- scaling-study extensions (all default = current behavior) ---
    loader: str | None = None
    max_memory: dict[str, Any] | None = None
    attn_implementation: str | None = None
    add_special_tokens: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    system_prompt: str | None = None
    generation_prefix: str | None = None
    single_device: bool = False


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
    # Production draws each class per layer (random_state=seed+layer+offset), so the
    # texts differ across layers and union extraction is not bit-identical. Set
    # ``per_layer_resample=False`` (opt-in) to draw the texts once with
    # ``random_state=seed+offset`` — identical across layers — and extract them in a
    # single pass via ``sentence_representations_layers`` when the backend provides it.
    per_layer_resample: bool = True


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
    # Layers expressed as relative depths in (0, 1] of backend.n_layers; resolved
    # to layer indices (mutually exclusive with ``layers``).
    relative_depths: tuple[float, ...] | None = None
    # "absolute": coefficients are used as-is (production). "relative_norm":
    # coefficients are alpha_rel multipliers of the layer's median residual norm
    # (stored in the vectors npz as ``median_norm``).
    alpha_mode: str = "absolute"
    # Number of calibration texts for the median norm, drawn deterministically
    # from the vector split (front of each cell's vector block, cell-major).
    norm_calibration_texts: int = 128
    # False (production): one generate call per (layer, alpha, row).
    # True: generate_steered_batch with one batch per eval prompt covering the
    # baseline plus all layer×alpha rows (row layout identical to production).
    batched_grid: bool = False
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
    # Layers as relative depths of backend.n_layers, resolved exactly like
    # steering.relative_depths (mutually exclusive with ``layers``).
    relative_depths: tuple[float, ...] | None = None


@dataclass(frozen=True)
class JudgeConfig:
    model: ModelConfig
    # task -> "line with {target}" or {target_attribute: line}; rendered as the
    # judge prompt's target line (production pairwise_string_prompt).
    attribute_descriptions: dict[str, str | dict[str, str]]
    prompt_template: str | None = None
    # "production" keeps DEFAULT_JUDGE_PROMPT (prompt_template wins if set);
    # "generic" selects the shipped domain-neutral GENERIC_JUDGE_PROMPT.
    prompt_preset: str = "production"
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
    # Scaling-study extensions (all default = current behavior):
    # backend "hf" = production HuggingFaceBackend; "vllm" = judge_vllm.VLLMJudgeBackend
    # (lazy import; vllm lives in a separate env). refusal_substrings enables the
    # deterministic Arditi-style `refusal_heuristic` column (None = off).
    backend: str = "hf"
    refusal_substrings: tuple[str, ...] | None = None


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
    for optional in (
        "layer_path", "layer_regex", "model_registry", "tokenizer_id",
        "loader", "max_memory", "attn_implementation", "add_special_tokens",
        "chat_template_kwargs", "system_prompt", "generation_prefix",
    ):
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
    for optional in ("relative_depths",):
        if steering_raw.get(optional) is not None:
            steering_raw[optional] = tuple(steering_raw[optional])
    geometry_raw = dict(raw["geometry"])
    geometry_raw["layers"] = tuple(geometry_raw["layers"])
    geometry_raw["poolings"] = tuple(geometry_raw.get("poolings", ("mean",)))
    for optional in ("variables", "target_fields", "relative_depths"):
        if geometry_raw.get(optional) is not None:
            geometry_raw[optional] = tuple(geometry_raw[optional])
    judge = None
    if raw.get("judge"):
        judge_raw = dict(raw["judge"])
        judge_raw["model"] = _model(judge_raw["model"])
        judge_raw["shard_group_by"] = tuple(judge_raw.get("shard_group_by", ()))
        if judge_raw.get("refusal_substrings") is not None:
            judge_raw["refusal_substrings"] = tuple(judge_raw["refusal_substrings"])
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
    if steering.relative_depths is not None and steering.layers:
        raise ValueError("steering.layers and steering.relative_depths are mutually exclusive")
    if steering.alpha_mode not in ("absolute", "relative_norm"):
        raise ValueError("steering.alpha_mode must be absolute or relative_norm")
    if steering.norm_calibration_texts < 1:
        raise ValueError("steering.norm_calibration_texts must be positive")
    if (steering.prompt_templates is None) != (steering.prompt_template_column is None):
        raise ValueError("steering.prompt_templates and prompt_template_column must be set together")
    if config.judge is not None and config.judge.num_shards < 1:
        raise ValueError("judge.num_shards must be positive")
    if config.judge is not None:
        if config.judge.prompt_preset not in {"production", "generic"}:
            raise ValueError("judge.prompt_preset must be production or generic")
        if config.judge.backend not in {"hf", "vllm"}:
            raise ValueError("judge.backend must be hf or vllm")
    judge_models = [config.judge.model] if config.judge is not None else []
    for model_config in [config.model, *judge_models]:
        if model_config.loader not in (None, "causal_lm", "image_text_to_text"):
            raise ValueError(f"model.loader must be causal_lm or image_text_to_text, got {model_config.loader!r}")
    if config.geometry.engine not in {"decodanda", "native"}:
        raise ValueError("geometry.engine must be decodanda or native")
    if config.geometry.protocol not in {"production", "matched_audit"}:
        raise ValueError("geometry.protocol must be production or matched_audit")
    geometry_variables = config.geometry.variables or tuple(variables)
    unknown_targets = set(config.geometry.target_fields or ()) - set(geometry_variables)
    if unknown_targets:
        raise ValueError(f"geometry.target_fields not in geometry variables: {sorted(unknown_targets)}")
    if config.geometry.relative_depths is not None and config.geometry.layers:
        raise ValueError("geometry.layers and geometry.relative_depths are mutually exclusive")
    if (not (config.steering.layers or config.steering.relative_depths)
            or not (config.geometry.layers or config.geometry.relative_depths)):
        raise ValueError("Steering and geometry layer lists cannot be empty")
    geometry_and_steering_layers = (*config.steering.layers, *config.geometry.layers)
    if any(layer < 0 for layer in geometry_and_steering_layers):
        raise ValueError("Layer indices must be non-negative")
