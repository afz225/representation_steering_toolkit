"""CPU-only tests for the scaling-study backend extensions.

Uses Qwen/Qwen2.5-0.5B-Instruct from the local HF cache (local_files_only);
skips cleanly when the model is unavailable. No GPU work.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch_mod = pytest.importorskip("torch")

from repsteer.config import ModelConfig
from repsteer.models.huggingface import HuggingFaceBackend
from repsteer.models.registry import default_registry_path, load_registry
from repsteer.types import Intervention

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REPO_ROOT = Path(__file__).resolve().parents[1]


def local_model_path() -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(MODEL_ID, local_files_only=True)
    except Exception:
        pytest.skip(f"{MODEL_ID} is not in the local Hugging Face cache")


def cpu_config(**kwargs) -> ModelConfig:
    return ModelConfig(model_id=local_model_path(), dtype="float32",
                       device_map="cpu", **kwargs)


@pytest.fixture(scope="module")
def backend():
    return HuggingFaceBackend(cpu_config())


TEXTS = [
    "The cat sat on the mat.",
    "Dogs bark loudly at night, and the neighbours complain about it.",
    "",
    "A much longer sentence about representation geometry, steering vectors "
    "and residual streams is toolkit work that we should smoke-test here.",
]

POOOLINGS = ("hs_mean", "mean", "last")
LAYERS = (0, 5, 23)


def test_sentence_representations_layers_bit_identical(backend):
    for pooling in POOOLINGS:
        single = {
            layer: backend.sentence_representations(TEXTS, layer, pooling,
                                                    batch_size=2, max_length=32)
            for layer in LAYERS
        }
        multi = backend.sentence_representations_layers(TEXTS, LAYERS, pooling,
                                                        batch_size=2, max_length=32)
        assert set(multi) == set(LAYERS)
        for layer in LAYERS:
            assert np.array_equal(multi[layer], single[layer]), pooling


def test_residual_norms_values_and_masking(backend):
    norms = backend.residual_norms(TEXTS, (0, 12, 23), batch_size=2, max_length=32)
    for layer, value in norms.items():
        assert isinstance(value, float)
        assert value > 0
        assert np.isfinite(value)
    assert norms[23] > norms[0] * 2


def _vector(hidden_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(hidden_size).astype(np.float32)


def test_generate_steered_batch_rows_equal_single_prompt(backend):
    hidden = backend.model.config.hidden_size
    prompt = "Write one short sentence about the sea."
    interventions = [
        None,
        Intervention(layer=0, vector=_vector(hidden, 1), coefficient=0.4,
                     positions="all"),
        Intervention(layer=23, vector=_vector(hidden, 2), coefficient=-0.2,
                     positions="last"),
        Intervention(layer=12, vector=_vector(hidden, 3), coefficient=1.0,
                     positions="generated"),
        Intervention(layer=0, vector=_vector(hidden, 4), coefficient=0.0,
                     positions="all"),
    ]
    batched = backend.generate_steered_batch(
        prompt, interventions, max_length=64, max_new_tokens=12, seed=7)
    assert len(batched) == len(interventions)
    for row, intervention in enumerate(interventions):
        kwargs = {
            "max_length": 64,
            "max_new_tokens": 12,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "seed": 7,
            "intervention": intervention,
            "normalize_whitespace": True,
            "pad_with_eos": True,
        }
        if intervention is not None:
            kwargs["retry_empty"] = True
        single = backend.generate([prompt], **kwargs)[0]
        assert batched[row] == single, row


def test_generate_steered_batch_layer_mix_in_one_batch(backend):
    hidden = backend.model.config.hidden_size
    prompt = "Name a colour."
    # Two rows steer the same layer with different vectors: the per-layer hook
    # must keep them apart.
    interventions = [
        Intervention(layer=5, vector=_vector(hidden, 11), coefficient=0.5,
                     positions="all"),
        Intervention(layer=5, vector=_vector(hidden, 12), coefficient=-0.5,
                     positions="all"),
    ]
    batched = backend.generate_steered_batch(
        prompt, interventions, max_length=64, max_new_tokens=8, seed=3)
    singles = [
        backend.generate([prompt], max_length=64, max_new_tokens=8,
                         temperature=0.0, top_p=1.0, do_sample=False, seed=3,
                         intervention=intervention, normalize_whitespace=True,
                         pad_with_eos=True, retry_empty=True)[0]
        for intervention in interventions
    ]
    assert batched == singles
    assert batched[0] != batched[1]


def test_generate_steered_batch_retries_empty_rows(backend):
    hidden = backend.model.config.hidden_size
    prompt = "Hello."
    # A huge coefficient at the last layer on "all" positions reliably empties
    # rows in the batch; the retry path must still return a string.
    interventions = [
        Intervention(layer=23, vector=_vector(hidden, 5), coefficient=1e4,
                     positions="all"),
        None,
    ]
    outputs = backend.generate_steered_batch(
        prompt, interventions, max_length=32, max_new_tokens=8, seed=1)
    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------


def test_registry_resolves_new_model_fields():
    reg = load_registry(REPO_ROOT / "configs" / "model_registry.yaml")
    profile = reg.resolve("google/gemma-4-E2B-it")
    assert profile.layer_path == "model.language_model.layers"
    assert profile.single_device is True
    gptoss = reg.resolve("openai/gpt-oss-120b")
    assert gptoss.layer_path == "model.layers"
    assert gptoss.generation_prefix == "<|channel|>final<|message|>"
    assert gptoss.chat_template_kwargs == {"reasoning_effort": "low"}
    granite = reg.resolve("ibm-granite/granite-4.2-8b")
    assert granite.layer_path == "model.layers"
    judge = reg.resolve("Qwen/Qwen3.6-27B")
    assert judge.layer_path == "model.layers"
    glm = reg.resolve("zai-org/GLM-4.5-Air")
    assert glm.layer_path == "model.layers"
    for model_id in (
        "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-122B-A10B", "ibm-granite/granite-4.2-3b",
        "ibm-granite/granite-4.2-30b", "allenai/Olmo-3-7B-Instruct",
        "allenai/Olmo-3.1-32B-Instruct", "google/gemma-4-E4B-it",
        "google/gemma-4-31B-it",
    ):
        assert reg.resolve(model_id) is not None, model_id


def test_registry_resolves_new_fields_onto_config():
    reg = load_registry(REPO_ROOT / "configs" / "model_registry.yaml")
    from repsteer.models.registry import apply_profile

    config = ModelConfig(model_id="google/gemma-4-E2B-it", dtype="float32")
    merged = apply_profile(config, reg.resolve(config.model_id))
    assert merged.layer_path == "model.language_model.layers"
    assert merged.single_device is True
    assert merged.dtype == "float32"  # explicit config value wins
    # Unset fields default to current behavior.
    plain = ModelConfig(model_id="some/other-model")
    assert plain.loader is None and plain.max_memory is None
    assert plain.system_prompt is None and plain.generation_prefix is None


def test_registry_rejects_invalid_loader():
    path = REPO_ROOT / "configs" / "model_registry.yaml"
    import yaml

    from repsteer.models.registry import ModelRegistry

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["profiles"]["bad/loader"] = {"loader": "magic"}
    with pytest.raises(ValueError):
        ModelRegistry.from_mapping(raw)


def test_default_registry_path_resolves_from_repo_root(monkeypatch, tmp_path):
    # Even when the CWD is elsewhere, the packaged path resolves to the repo.
    monkeypatch.chdir(tmp_path)
    path = default_registry_path()
    assert path is not None
    assert path.name == "model_registry.yaml"
    assert path.parent == REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# Loader/model-config plumbing
# ---------------------------------------------------------------------------


def test_model_config_defaults_and_providence():
    config = ModelConfig(model_id="x/y")
    assert config.loader is None
    assert config.max_memory is None
    assert config.attn_implementation is None
    assert config.add_special_tokens is None
    assert config.chat_template_kwargs is None
    assert config.system_prompt is None
    assert config.generation_prefix is None
    assert config.single_device is False


def test_chat_template_extras_applied(backend):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(local_model_path())
    text = "HOST sentinel"
    if not hasattr(tokenizer, "apply_chat_template"):
        pytest.skip("tokenizer has no chat template")
    try:
        tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                      tokenize=False, add_generation_prompt=True)
    except Exception:
        pytest.skip("chat template raised before extras test")

    class _T:
        pass

    holder = _T()
    holder.tokenizer = tokenizer
    holder.use_chat_template = True
    holder.config = replace(backend.config, system_prompt="Be terse.",
                            chat_template_kwargs={"system": None})
    holder.config = replace(holder.config, chat_template_kwargs={"max_tokens": -1})
    holder.config = replace(holder.config, generation_prefix="PREFIX>>")
    from repsteer.models.huggingface import HuggingFaceBackend as _B

    formatted = _B._template_messages(holder, text)
    assert formatted[0]["content"] == "Be terse."
    assert formatted[1]["content"] == text


def test_generation_prefix_appended_to_rendered_template(backend):
    config = replace(backend.config, generation_prefix="<<GO>>")
    clone = HuggingFaceBackend.__new__(HuggingFaceBackend)
    clone.config = config
    clone.torch = backend.torch
    clone.use_chat_template = True
    clone.tokenizer = backend.tokenizer
    clone.add_special_tokens = config.add_special_tokens
    clone.extraction_padding_side = backend.extraction_padding_side
    formatted = clone._format_prompts(["Hi there"])
    assert formatted[0].endswith("<<GO>>")


# ---------------------------------------------------------------------------
# smoke-model CLI output shape
# ---------------------------------------------------------------------------


def test_smoke_model_json_shape(capsys):
    from repsteer.cli import build_parser, main

    parser = build_parser()
    args = parser.parse_args([
        "smoke-model", "--model-id", local_model_path(),
        "--dtype", "float32", "--device", "cpu",
    ])
    main_args = args
    from repsteer.cli import command_smoke_model

    command_smoke_model(main_args)
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["n_layers"] == 24
    assert "rendered_prompt" in payload and "bos_count" in payload
    assert payload["template_fallback"] is False
    assert payload["hook_effect_nonzero"] is True
    assert len(payload["residual_norms"]) == 3
    assert set(payload["residual_norms"]) == {"0", "12", "23"}
