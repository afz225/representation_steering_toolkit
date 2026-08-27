from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..types import Intervention, Pooling


class ModelBackend(Protocol):
    """Minimal boundary required by steering, judging, and geometry."""

    @property
    def n_layers(self) -> int: ...

    def sentence_representations(
        self,
        texts: Sequence[str],
        layer: int,
        pooling: Pooling,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> np.ndarray: ...

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        seed: int,
        intervention: Intervention | None = None,
    ) -> list[str]: ...



    def sequence_nll(
        self,
        texts: Sequence[str],
        *,
        max_length: int,
        batch_size: int = 1,
        intervention: Intervention | None = None,
    ) -> list[tuple[float, int]]: ...

    def next_token_choice_probabilities(
        self,
        prompts: Sequence[str],
        choices: Sequence[str],
        *,
        max_length: int,
        batch_size: int = 8,
        intervention: Intervention | None = None,
    ) -> np.ndarray: ...

    def generate_with_representations(
        self,
        prompt: str,
        *,
        capture_layers: Sequence[int],
        max_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        seed: int,
        intervention: Intervention | None = None,
    ) -> tuple[str, dict[int, np.ndarray]]: ...
