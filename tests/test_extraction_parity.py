"""Tiny-model CPU parity: toolkit extraction == production extraction code paths.

Compares ``HuggingFaceBackend.sentence_representations`` with production
``geometry_olmo_reduced_full.PromptRepStore.extract_all_points`` (hs_mean /
hs_last / hs_min / hs_max) and ``caa_model_search.SearchCaaModel.sentence_reps``
(mean / last) on Qwen2.5-0.5B-Instruct from the local Hugging Face cache.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from repsteer.config import ModelConfig
from repsteer.models.huggingface import HuggingFaceBackend

TOOLKIT = Path(__file__).resolve().parents[1]
PRODUCTION = TOOLKIT.parent
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS = (0, 5, 23)


def production_module(name: str):
    if not (PRODUCTION / f"{name}.py").exists():
        pytest.skip("production scripts are not available in this checkout")
    if str(PRODUCTION) not in sys.path:
        sys.path.insert(0, str(PRODUCTION))
    return importlib.import_module(name)


def local_model_path() -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(MODEL_ID, local_files_only=True)
    except Exception:
        pytest.skip(f"{MODEL_ID} is not in the local Hugging Face cache")


@pytest.fixture(scope="module")
def backend():
    return HuggingFaceBackend(ModelConfig(model_id=local_model_path(), dtype="float32",
                                          device_map="cpu", use_chat_template=True))


def fresh_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(local_model_path())
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def corpus_texts() -> list[str]:
    frame = pd.read_csv(TOOLKIT / "datasets" / "catdog" / "corpus.csv")
    english = frame[frame.language == "en"].text.head(7).tolist()
    arabic = frame[frame.language == "ar"].text.head(4).tolist()
    # Raw whitespace exercises the production normalization (safe_text/normalize_text).
    return [*english, *arabic, "  a   short\n\ttext  "]


def production_store(backend, model=None):
    geometry = production_module("geometry_olmo_reduced_full")
    store = geometry.PromptRepStore.__new__(geometry.PromptRepStore)
    store.tokenizer = fresh_tokenizer()
    store.model = model if model is not None else backend.model
    store.layers = geometry.get_decoder_layers(store.model)
    return geometry, store


def test_extraction_keeps_tokenizer_default_padding(backend):
    assert backend.extraction_padding_side == fresh_tokenizer().padding_side == "right"
    assert backend.tokenizer.padding_side == "left"
    backend.sentence_representations(["a b", "c"], 0, "mean", batch_size=2)
    assert backend.tokenizer.padding_side == "left"


def test_geometry_poolings_match_prompt_rep_store(backend):
    geometry, store = production_store(backend)
    texts = corpus_texts()
    examples = [geometry.Example(prompt=geometry.normalize_text(text), meta={}) for text in texts]
    poolings = ["hs_last", "hs_mean", "hs_min", "hs_max"]
    reps, _, _ = store.extract_all_points(examples, list(LAYERS), poolings, 4, 512)
    toolkit_name = {"hs_last": "last", "hs_mean": "hs_mean", "hs_min": "min", "hs_max": "max"}
    for layer in LAYERS:
        for pooling in poolings:
            ours = backend.sentence_representations(texts, layer, toolkit_name[pooling], 4, 512)
            assert ours.dtype == np.float32
            assert np.array_equal(ours, reps[pooling][layer]), (layer, pooling)


def test_mean_and_last_match_caa_sentence_reps(backend):
    search = production_module("caa_model_search")
    geometry = production_module("geometry_olmo_reduced_full")
    model = search.SearchCaaModel.__new__(search.SearchCaaModel)
    model.tokenizer = fresh_tokenizer()
    model.model = backend.model
    model.layers = geometry.get_decoder_layers(backend.model)
    texts = corpus_texts()
    for layer in LAYERS:
        for pooling in ("mean", "last"):
            expected = model.sentence_reps(texts, layer, 4, 512, pooling)
            ours = backend.sentence_representations(texts, layer, pooling, 4, 512)
            assert np.array_equal(ours, expected), (layer, pooling)


def test_bfloat16_hs_mean_and_mean_follow_their_production_dtypes(backend):
    """In bf16 the two means differ; each matches its own production recipe."""
    import copy

    bf16 = copy.deepcopy(backend.model).to(torch.bfloat16)
    geometry, store = production_store(backend, bf16)
    search = production_module("caa_model_search")
    caa = search.SearchCaaModel.__new__(search.SearchCaaModel)
    caa.tokenizer, caa.model, caa.layers = fresh_tokenizer(), bf16, store.layers
    texts = corpus_texts()
    examples = [geometry.Example(prompt=geometry.normalize_text(text), meta={}) for text in texts]
    reps, _, _ = store.extract_all_points(examples, [5], ["hs_mean"], 4, 512)
    original = backend.model
    backend.model = bf16
    backend.layers = store.layers
    try:
        hs_mean = backend.sentence_representations(texts, 5, "hs_mean", 4, 512)
        mean = backend.sentence_representations(texts, 5, "mean", 4, 512)
    finally:
        backend.model = original
        backend.layers = geometry.get_decoder_layers(original)
    assert np.array_equal(hs_mean, reps["hs_mean"][5])
    assert np.array_equal(mean, caa.sentence_reps(texts, 5, 4, 512, "mean"))
    assert not np.array_equal(hs_mean, mean)
