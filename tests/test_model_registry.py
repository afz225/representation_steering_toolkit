from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from repsteer.config import ModelConfig, load_config
from repsteer.models.registry import (
    ModelProfile,
    ModelRegistry,
    apply_profile,
    load_registry,
    select_modules_by_regex,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeModule:
    """Minimal module with a ``named_modules`` generator, no model load."""

    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return f"FakeModule({self._name})"


def fake_model() -> FakeModule:
    root = FakeModule("")

    def named_modules(prefix: str = ""):
        yield "", root
        yield "model.layers.0.self_attn", FakeModule("model.layers.0.self_attn")
        yield "model.layers.2.mlp", FakeModule("model.layers.2.mlp")
        yield "model.layers.2.mlp.act_fn", FakeModule("model.layers.2.mlp.act_fn")
        yield "model.layers.10.mlp", FakeModule("model.layers.10.mlp")
        yield "model.layers.2.self_attn", FakeModule("model.layers.2.self_attn")
        yield "lm_head", FakeModule("lm_head")

    root.named_modules = named_modules
    return root


def registry(tmp_path: Path, text: str) -> ModelRegistry:
    path = tmp_path / "registry.yaml"
    path.write_text(text, encoding="utf-8")
    return load_registry(path)


# ---------------------------------------------------------------------------
# YAML loading and validation
# ---------------------------------------------------------------------------


def test_production_registry_file_loads_with_expected_profiles():
    reg = load_registry(REPO_ROOT / "configs" / "model_registry.yaml")
    gemma = reg.resolve("google/gemma-4-31B-it")
    assert gemma is not None
    assert gemma.layer_path == "model.language_model.layers"
    assert gemma.use_chat_template is True
    assert gemma.dtype == "bfloat16"
    llama = reg.resolve("meta-llama/Llama-2-7b-hf")
    assert llama is not None and llama.layer_path == "model.layers"
    gpt2 = reg.resolve("gpt2-large")
    assert gpt2 is not None and gpt2.layer_path == "transformer.h"
    chatlm = reg.resolve("chatlm/mlp-demo")
    assert chatlm is not None and chatlm.layer_regex == r"model\.layers\.\d+\.mlp$"


def test_unknown_profile_field_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown profile field"):
        registry(
            tmp_path,
            "profiles:\n  'a/model':\n    not_a_field: true\n",
        )


def test_non_bool_flags_rejected(tmp_path):
    with pytest.raises(ValueError, match="expected a boolean"):
        registry(
            tmp_path,
            "profiles:\n  'a/model':\n    use_chat_template: yes-please\n",
        )


def test_malformed_registry_rejected(tmp_path):
    with pytest.raises(ValueError, match="profiles"):
        registry(tmp_path, "something_else: {}\n")
    with pytest.raises(ValueError, match="mapping"):
        registry(tmp_path, "profiles: [1, 2]\n")


# ---------------------------------------------------------------------------
# Resolution precedence: exact > pattern (file order) > built-in fallback
# ---------------------------------------------------------------------------


def test_resolution_precedence_exact_beats_pattern(tmp_path):
    reg = registry(
        tmp_path,
        "profiles:\n"
        "  'vendor/model-v1':\n"
        "    layer_path: exact.path\n"
        "  'vendor/*':\n"
        "    layer_path: pattern.path\n",
    )
    assert reg.resolve("vendor/model-v1").layer_path == "exact.path"


def test_resolution_first_pattern_in_file_order_wins(tmp_path):
    reg = registry(
        tmp_path,
        "profiles:\n"
        "  'vendor/*':\n"
        "    layer_path: first.pattern\n"
        "  'vendor/model*':\n"
        "    layer_path: second.pattern\n",
    )
    assert reg.resolve("vendor/model-x").layer_path == "first.pattern"
    assert reg.resolve("vendor/other-x").layer_path == "first.pattern"


def test_resolution_no_match_returns_none(tmp_path):
    reg = registry(tmp_path, "profiles:\n  'vendor/*':\n    layer_path: p\n")
    assert reg.resolve("other/model") is None


def test_exact_key_with_regex_chars_is_still_exact(tmp_path):
    # An exact key containing '(' or other non-wildcard regex characters must
    # not be treated as a pattern; only *, ? and [ mark patterns.
    reg = registry(
        tmp_path,
        "profiles:\n  'vendor/model-(v1)':\n    layer_path: exact.regex.key\n",
    )
    assert reg.resolve("vendor/model-(v1)").layer_path == "exact.regex.key"
    assert reg.resolve("vendor/model-v1") is None


def test_built_in_fallback_applies_when_registry_has_no_match(tmp_path):
    reg = registry(tmp_path, "profiles:\n  'vendor/*':\n    layer_path: vendor.path\n")
    profile = reg.resolve("unknown-org/unknown-model")
    assert profile is None  # caller falls back to HuggingFaceBackend._LAYER_PATHS


# ---------------------------------------------------------------------------
# apply_profile semantics
# ---------------------------------------------------------------------------


def test_apply_profile_fills_unset_fields():
    config = ModelConfig(model_id="google/gemma-4-31B-it")
    profile = ModelProfile(
        layer_path="model.language_model.layers",
        use_chat_template=True,
        dtype="bfloat16",
    )
    merged = apply_profile(config, profile)
    assert merged.layer_path == "model.language_model.layers"
    assert merged.use_chat_template is True
    assert merged.dtype == "bfloat16"
    assert merged.model_id == "google/gemma-4-31B-it"


def test_apply_profile_config_values_win():
    config = ModelConfig(
        model_id="google/gemma-4-31B-it",
        layer_path="model.layers",
        dtype="float16",
        use_chat_template=False,
        trust_remote_code=True,
    )
    profile = ModelProfile(
        layer_path="model.language_model.layers",
        use_chat_template=True,
        dtype="bfloat16",
        trust_remote_code=False,
    )
    merged = apply_profile(config, profile)
    assert merged.layer_path == "model.layers"
    assert merged.dtype == "float16"
    assert merged.use_chat_template is False
    assert merged.trust_remote_code is True


def test_apply_profile_layer_regex_used_when_no_layer_path():
    config = ModelConfig(model_id="chatlm/mlp-demo")
    profile = ModelProfile(layer_regex=r"model\.layers\.\d+\.mlp$")
    merged = apply_profile(config, profile)
    assert merged.layer_path is None
    assert merged.layer_regex == r"model\.layers\.\d+\.mlp$"


def test_apply_profile_layer_path_beats_layer_regex():
    config = ModelConfig(model_id="a/model")
    profile = ModelProfile(layer_path="a.layers", layer_regex=r"mlp$")
    merged = apply_profile(config, profile)
    assert merged.layer_path == "a.layers"
    assert merged.layer_regex is None


def test_apply_profile_none_is_identity():
    config = ModelConfig(model_id="a/model")
    assert apply_profile(config, None) is config


def test_apply_profile_tokenizer_id_applies_when_unset():
    config = ModelConfig(model_id="a/model")
    profile = ModelProfile(tokenizer_id="a/tokenizer")
    assert apply_profile(config, profile).tokenizer_id == "a/tokenizer"
    explicit = ModelConfig(model_id="a/model", tokenizer_id="explicit/tokenizer")
    assert apply_profile(explicit, profile).tokenizer_id == "explicit/tokenizer"


# ---------------------------------------------------------------------------
# Regex module matching on a fake module tree
# ---------------------------------------------------------------------------


def test_regex_module_match_deterministic_natural_order():
    modules = select_modules_by_regex(fake_model(), r"model\.layers\.\d+\.mlp$")
    names = [repr(module) for module in modules]
    # layer 2 sorts before layer 10; submodules inside an mlp are not matched
    assert names == [
        "FakeModule(model.layers.2.mlp)",
        "FakeModule(model.layers.10.mlp)",
    ]
    again = select_modules_by_regex(fake_model(), r"model\.layers\.\d+\.mlp$")
    assert [repr(module) for module in again] == names


def test_regex_module_match_searches_anywhere_and_can_be_anchored():
    modules = select_modules_by_regex(fake_model(), r"self_attn")
    assert [repr(module) for module in modules] == [
        "FakeModule(model.layers.0.self_attn)",
        "FakeModule(model.layers.2.self_attn)",
    ]


def test_regex_module_match_no_hit_raises():
    with pytest.raises(ValueError, match="matched 0 modules"):
        select_modules_by_regex(fake_model(), r"encoder\.blocks\.\d+$")


# ---------------------------------------------------------------------------
# HuggingFaceBackend integration (no model download)
# ---------------------------------------------------------------------------


def test_backend_regex_resolution_and_profile_capture():
    from repsteer.models.huggingface import HuggingFaceBackend

    backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
    model = fake_model()
    backend.model = model
    config = ModelConfig(model_id="chatlm/mlp-demo")
    profile = ModelProfile(layer_regex=r"model\.layers\.\d+\.mlp$")
    backend.layers = backend._resolve_layers(config.layer_path, profile.layer_regex)
    assert [repr(module) for module in backend.layers] == [
        "FakeModule(model.layers.2.mlp)",
        "FakeModule(model.layers.10.mlp)",
    ]
    assert backend.n_layers == 2


# ---------------------------------------------------------------------------
# config.py integration
# ---------------------------------------------------------------------------


def test_load_config_accepts_optional_model_registry_and_regex_fields(tmp_path):
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "path": "data.csv",
                    "text_column": "text",
                    "id_column": "id",
                    "variables": {"label": ["cold", "warm"]},
                },
                "model": {
                    "model_id": "chatlm/mlp-demo",
                    "model_registry": "configs/model_registry.yaml",
                    "layer_regex": r"model\.layers\.\d+\.mlp$",
                },
                "target": {"name": "label", "negative": "cold", "positive": "warm"},
                "steering": {"layers": [0, 1], "coefficients": [1.0]},
                "geometry": {"layers": [0, 1]},
            }
        )
    )
    config = load_config(config_path)
    assert config.model.model_registry == "configs/model_registry.yaml"
    assert config.model.layer_regex == r"model\.layers\.\d+\.mlp$"
    assert config.model.layer_path is None


def test_load_config_strips_null_optional_model_fields(tmp_path):
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "path": "data.csv",
                    "text_column": "text",
                    "id_column": "id",
                    "variables": {"label": ["cold", "warm"]},
                },
                "model": {
                    "model_id": "a/model",
                    "layer_path": None,
                    "model_registry": None,
                },
                "target": {"name": "label", "negative": "cold", "positive": "warm"},
                "steering": {"layers": [0], "coefficients": [1.0]},
                "geometry": {"layers": [0]},
            }
        )
    )
    config = load_config(config_path)
    assert config.model.layer_path is None
    assert config.model.model_registry is None


def test_existing_configs_still_load():
    example = REPO_ROOT / "configs" / "example_catdog.json"
    config = load_config(example)
    assert config.model.model_registry is None
    assert config.model.layer_regex is None
