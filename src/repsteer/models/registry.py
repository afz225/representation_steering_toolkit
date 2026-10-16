"""YAML model registry: map Hugging Face model ids to backend profiles.

The registry lets a config omit architecture-specific details such as the
decoder-layer attribute path. Resolution order is:

1. exact ``model_id`` match (a registry key without ``*``, ``?`` or ``[``);
2. first pattern match in file order (fnmatch-style, i.e. ``*`` globs);
3. the built-in attribute walk in
   :class:`repsteer.models.huggingface.HuggingFaceBackend`.

Profiles may override ``tokenizer_id``, ``dtype``, ``use_chat_template`` and
``trust_remote_code`` before the model is loaded, and may select steerable
modules either via an attribute ``layer_path`` or a regular expression
``layer_regex`` matched against ``model.named_modules()``.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..config import ModelConfig

DEFAULT_REGISTRY_PATH = "configs/model_registry.yaml"

_WILDCARDS = ("*", "?", "[")


@dataclass(frozen=True)
class ModelProfile:
    """Architecture-specific settings applied on top of a :class:`ModelConfig`.

    Unset fields (``None``) leave the config value untouched. Config values that
    differ from the dataclass defaults are treated as explicit user choices and
    always win over the profile.
    """

    layer_path: str | None = None
    layer_regex: str | None = None
    tokenizer_id: str | None = None
    dtype: str | None = None
    use_chat_template: bool | None = None
    trust_remote_code: bool | None = None


_PROFILE_FIELDS = {field.name for field in fields(ModelProfile)}


@dataclass(frozen=True)
class RegistryEntry:
    key: str
    profile: ModelProfile
    is_pattern: bool


def _coerce_bool(value: Any, context: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{context}: expected a boolean, got {value!r}")
    return value


def _profile(raw: Any, context: str) -> ModelProfile:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{context}: profile must be a mapping")
    unknown = set(raw) - _PROFILE_FIELDS
    if unknown:
        raise ValueError(f"{context}: unknown profile field(s) {sorted(unknown)}")
    return ModelProfile(
        layer_path=None if raw.get("layer_path") is None else str(raw["layer_path"]),
        layer_regex=None if raw.get("layer_regex") is None else str(raw["layer_regex"]),
        tokenizer_id=None if raw.get("tokenizer_id") is None else str(raw["tokenizer_id"]),
        dtype=None if raw.get("dtype") is None else str(raw["dtype"]),
        use_chat_template=_coerce_bool(raw.get("use_chat_template"), context),
        trust_remote_code=_coerce_bool(raw.get("trust_remote_code"), context),
    )


class ModelRegistry:
    """Ordered collection of model-id/pattern profiles loaded from YAML."""

    def __init__(self, entries: Iterable[RegistryEntry], source: str | None = None):
        self.entries = list(entries)
        self.source = source

    @classmethod
    def from_mapping(cls, raw: Any, source: str | None = None) -> "ModelRegistry":
        if not isinstance(raw, dict):
            raise ValueError("Registry file must be a mapping with a 'profiles' section")
        profiles = raw.get("profiles")
        if profiles is None:
            raise ValueError("Registry file must define a 'profiles' mapping")
        if not isinstance(profiles, dict):
            raise ValueError("Registry 'profiles' section must be a mapping of key -> profile")
        unknown_top = set(raw) - {"profiles"}
        if unknown_top:
            raise ValueError(f"Registry file has unknown top-level key(s) {sorted(unknown_top)}")
        entries: list[RegistryEntry] = []
        for key, profile_raw in profiles.items():
            key = str(key)
            entries.append(
                RegistryEntry(
                    key=key,
                    profile=_profile(profile_raw, f"profile {key!r}"),
                    is_pattern=any(char in key for char in _WILDCARDS),
                )
            )
        return cls(entries, source=source)

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelRegistry":
        path = Path(path)
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.from_mapping(raw, source=str(path))

    def resolve(self, model_id: str) -> ModelProfile | None:
        """Exact id match first, then the first matching pattern in file order."""
        for entry in self.entries:
            if not entry.is_pattern and entry.key == model_id:
                return entry.profile
        for entry in self.entries:
            if entry.is_pattern and fnmatch.fnmatchcase(model_id, entry.key):
                return entry.profile
        return None


def load_registry(path: str | Path) -> ModelRegistry:
    """Load a YAML model registry from ``path``."""
    return ModelRegistry.from_file(path)


def default_registry_path() -> Path | None:
    """Return the conventional registry path when it exists in the working directory."""
    candidate = Path(DEFAULT_REGISTRY_PATH)
    return candidate if candidate.is_file() else None


def _natural_key(name: str) -> tuple[tuple[bool, int, str], ...]:
    parts = re.split(r"(\d+)", name)
    return tuple(
        (part.isdigit(), int(part) if part.isdigit() else 0, part)
        for part in parts
        if part != ""
    )


def select_modules_by_regex(model: Any, pattern: str) -> list[Any]:
    """Return modules whose ``named_modules()`` name matches ``pattern``.

    The regex is searched anywhere in the module name (anchor with ``$`` to
    restrict it). Matches are ordered by a deterministic natural sort so numeric
    layer indices stay in ascending order (``layers.2`` before ``layers.10``).
    Raises ``ValueError`` when nothing matches.
    """
    regex = re.compile(pattern)
    matches = {name: module for name, module in model.named_modules() if regex.search(name)}
    if not matches:
        raise ValueError(f"layer_regex {pattern!r} matched 0 modules in the model")
    return [matches[name] for name in sorted(matches, key=_natural_key)]


def apply_profile(config: ModelConfig, profile: ModelProfile | None) -> ModelConfig:
    """Return a config with profile overrides applied to unset fields.

    Rules:
    - explicit config values always win over the profile;
    - ``tokenizer_id`` applies when the config leaves it unset (``None``);
    - ``dtype``, ``use_chat_template`` and ``trust_remote_code`` apply when the
      config is left at its dataclass default;
    - ``layer_path``/``layer_regex`` apply only when ``config.layer_path`` is
      unset; ``layer_path`` takes precedence over ``layer_regex``.
    """
    if profile is None:
        return config
    updates: dict[str, Any] = {}
    if profile.tokenizer_id is not None and config.tokenizer_id is None:
        updates["tokenizer_id"] = profile.tokenizer_id
    if profile.dtype is not None and config.dtype == "bfloat16":
        updates["dtype"] = profile.dtype
    if profile.use_chat_template is not None and config.use_chat_template is True:
        updates["use_chat_template"] = profile.use_chat_template
    if profile.trust_remote_code is not None and config.trust_remote_code is False:
        updates["trust_remote_code"] = profile.trust_remote_code
    if config.layer_path is None:
        if profile.layer_path is not None:
            updates["layer_path"] = profile.layer_path
        elif profile.layer_regex is not None:
            updates["layer_regex"] = profile.layer_regex
    if not updates:
        return config
    return _replaced(config, updates)


def _replaced(config: ModelConfig, updates: dict[str, Any]) -> ModelConfig:
    return replace(config, **updates)


def registry_for(config: ModelConfig, registry: ModelRegistry | str | Path | None) -> ModelRegistry | None:
    """Resolve the registry to use for ``config``.

    ``registry`` may be a :class:`ModelRegistry`, a YAML path, or ``None`` to
    fall back to ``config.model_registry`` and then the conventional
    ``configs/model_registry.yaml`` when that file exists.
    """
    if isinstance(registry, ModelRegistry):
        return registry
    source: str | Path | None = registry
    if source is None:
        source = getattr(config, "model_registry", None)
    if source is None:
        source = default_registry_path()
    if source is None:
        return None
    return load_registry(source)


def resolve_profile(config: ModelConfig, registry: ModelRegistry | str | Path | None) -> ModelProfile | None:
    """Convenience: resolve the registry and match ``config.model_id``."""
    registry_obj = registry_for(config, registry)
    if registry_obj is None:
        return None
    return registry_obj.resolve(config.model_id)
