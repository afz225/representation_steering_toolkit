from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from repsteer.models.huggingface import HuggingFaceBackend
from repsteer.types import Intervention
from repsteer.vectors import SteeringVector, load_vectors, save_vectors


class FakeHandle:
    def __init__(self, layer):
        self.layer = layer

    def remove(self):
        self.layer.hook = None


class FakeLayer:
    def __init__(self):
        self.hook = None

    def register_forward_hook(self, hook):
        self.hook = hook
        return FakeHandle(self)


def backend_without_loading():
    backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
    backend.torch = torch
    backend.layers = [FakeLayer(), FakeLayer()]
    return backend


def test_attention_mask_aware_pooling_with_left_padding():
    backend = backend_without_loading()
    hidden = torch.tensor(
        [
            [[99.0, 99.0], [1.0, 3.0], [5.0, 7.0]],
            [[99.0, 99.0], [99.0, 99.0], [2.0, 4.0]],
        ]
    )
    mask = torch.tensor([[0, 1, 1], [0, 0, 1]])
    np.testing.assert_allclose(backend._pool(hidden, mask, "last"), [[5, 7], [2, 4]])
    np.testing.assert_allclose(backend._pool(hidden, mask, "mean"), [[3, 5], [2, 4]])
    np.testing.assert_allclose(backend._pool(hidden, mask, "min"), [[1, 3], [2, 4]])
    np.testing.assert_allclose(backend._pool(hidden, mask, "max"), [[5, 7], [2, 4]])


@pytest.mark.parametrize("position", ["all", "last", "generated"])
def test_intervention_position_semantics_and_hook_cleanup(position):
    backend = backend_without_loading()
    intervention = Intervention(
        layer=1,
        vector=np.asarray([1.0, -2.0], dtype=np.float32),
        coefficient=3.0,
        positions=position,
    )
    original = torch.zeros((1, 3, 2))
    with backend._steering_hook(intervention):
        hook = backend.layers[1].hook
        assert hook is not None
        modified = hook(None, None, original)
        if position == "all":
            np.testing.assert_allclose(modified.numpy(), [[[3, -6], [3, -6], [3, -6]]])
        elif position == "last":
            np.testing.assert_allclose(modified.numpy(), [[[0, 0], [0, 0], [3, -6]]])
        else:
            np.testing.assert_allclose(modified.numpy(), original.numpy())
            one_token = hook(None, None, torch.zeros((1, 1, 2)))
            np.testing.assert_allclose(one_token.numpy(), [[[3, -6]]])
    assert backend.layers[1].hook is None


def test_tuple_layer_output_is_preserved():
    backend = backend_without_loading()
    intervention = Intervention(0, np.asarray([1.0, 1.0]), 2.0, "all")
    cache = object()
    with backend._steering_hook(intervention):
        output = backend.layers[0].hook(None, None, (torch.zeros((1, 1, 2)), cache))
    np.testing.assert_allclose(output[0].numpy(), [[[2, 2]]])
    assert output[1] is cache


def test_nested_gemma_layer_path_is_resolved():
    backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
    expected = [object(), object(), object()]
    backend.model = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(layers=expected))
    )
    assert backend._resolve_layers(None) is expected


def test_chat_template_disables_thinking_with_legacy_fallback():
    backend = backend_without_loading()

    class Tokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(kwargs)
            if "enable_thinking" in kwargs and len(self.calls) == 2:
                raise TypeError("legacy tokenizer")
            return "formatted"

    backend.tokenizer = Tokenizer()
    assert backend._apply_chat_template("hello") == "formatted"
    assert backend.tokenizer.calls[0]["enable_thinking"] is False
    # Trigger fallback on this call.
    assert backend._apply_chat_template("hello") == "formatted"
    assert "enable_thinking" not in backend.tokenizer.calls[-1]


def test_vector_archive_round_trip(tmp_path):
    vectors = [
        SteeringVector(
            layer=3,
            variable="style",
            condition=(("language", "ar"), ("is_dog", "0")),
            values=np.asarray([1.5, -2.0], dtype=np.float32),
            negative_count=20,
            positive_count=20,
        )
    ]
    path = tmp_path / "vectors.npz"
    save_vectors(vectors, path)
    loaded = load_vectors(path)
    assert len(loaded) == 1
    assert loaded[0].key == vectors[0].key
    assert loaded[0].variable == "style"
    np.testing.assert_array_equal(loaded[0].values, vectors[0].values)
