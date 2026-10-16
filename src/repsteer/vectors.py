from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import VectorSamplingConfig
from .models.base import ModelBackend
from .types import Pooling, VariableSpec


@dataclass(frozen=True)
class SteeringVector:
    layer: int
    variable: str
    condition: tuple[tuple[str, str], ...]
    values: np.ndarray
    negative_count: int
    positive_count: int
    class_gap: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition", tuple(sorted(self.condition)))

    @property
    def key(self) -> tuple[int, tuple[tuple[str, str], ...]]:
        return self.layer, self.condition


def _condition_groups(frame: pd.DataFrame, names: tuple[str, ...]):
    if not names:
        yield (), frame
        return
    grouper = names[0] if len(names) == 1 else list(names)
    for values, group in frame.groupby(grouper, sort=True, dropna=False):
        if len(names) == 1:
            values = (values,)
        yield tuple(sorted((name, str(value)) for name, value in zip(names, values))), group


def _contrast_texts(
    group: pd.DataFrame,
    text_column: str,
    mask: np.ndarray,
    layer: int,
    offset: int,
    sampling: VectorSamplingConfig | None,
) -> list[str]:
    """Class texts in file order, or the production ``DataFrame.sample`` draw."""

    part = group[mask]
    if sampling is not None:
        n = sampling.per_class
        replace = len(part) < n
        if replace and not sampling.replace_if_short:
            raise ValueError(f"Need {n} contrast rows; found {len(part)}")
        part = part.sample(n=n, random_state=sampling.seed + layer + offset, replace=replace)
    return part[text_column].astype(str).tolist()


def build_vectors(
    backend: ModelBackend,
    frame: pd.DataFrame,
    text_column: str,
    spec: VariableSpec,
    layers: Iterable[int],
    pooling: Pooling = "mean",
    normalize: bool = True,
    batch_size: int = 8,
    max_length: int = 512,
    sampling: VectorSamplingConfig | None = None,
) -> list[SteeringVector]:
    """Build positive-minus-negative difference-of-means vectors.

    Mirrors ``SearchCaaModel.corpus_difference_vector``: positive texts are
    extracted (in batches) before negative texts, ``vec = pos.mean(0) -
    neg.mean(0)`` in float32, and normalization divides by
    ``float(np.linalg.norm(vec))``. With ``sampling`` each class is drawn per
    layer by ``DataFrame.sample(per_class, random_state=seed + layer + offset)``.
    """

    layers = list(layers)
    vectors: list[SteeringVector] = []
    for condition, group in _condition_groups(frame, spec.condition_on):
        labels = group[spec.name].astype(str).to_numpy()
        neg_mask = labels == spec.negative
        pos_mask = labels == spec.positive
        if not neg_mask.any() or not pos_mask.any():
            raise ValueError(f"Condition {dict(condition)} does not contain both target classes")
        for layer in layers:
            offsets = (0, 0) if sampling is None else (
                sampling.positive_seed_offset, sampling.negative_seed_offset)
            pos_texts = _contrast_texts(group, text_column, pos_mask, layer, offsets[0], sampling)
            neg_texts = _contrast_texts(group, text_column, neg_mask, layer, offsets[1], sampling)
            pos = backend.sentence_representations(
                pos_texts, layer, pooling, batch_size=batch_size, max_length=max_length
            ).astype(np.float32)
            neg = backend.sentence_representations(
                neg_texts, layer, pooling, batch_size=batch_size, max_length=max_length
            ).astype(np.float32)
            values = pos.mean(axis=0) - neg.mean(axis=0)
            norm = float(np.linalg.norm(values))
            if norm == 0 or not np.isfinite(norm):
                raise ValueError(f"Bad vector norm {norm} at layer {layer}, condition {dict(condition)}")
            if normalize:
                values = values / norm
            vectors.append(
                SteeringVector(
                    layer=layer,
                    variable=spec.name,
                    condition=tuple(sorted(condition)),
                    values=values.astype(np.float32),
                    negative_count=len(neg_texts),
                    positive_count=len(pos_texts),
                    class_gap=norm,
                )
            )
    return vectors


def vector_index(vectors: Iterable[SteeringVector]):
    return {vector.key: vector for vector in vectors}


def save_vectors(vectors: Iterable[SteeringVector], path: str | Path) -> None:
    vectors = list(vectors)
    arrays = {f"vector_{i}": vector.values for i, vector in enumerate(vectors)}
    metadata = [
        {
            "array": f"vector_{i}",
            "layer": vector.layer,
            "variable": vector.variable,
            "condition": dict(vector.condition),
            "negative_count": vector.negative_count,
            "positive_count": vector.positive_count,
            "class_gap": vector.class_gap,
        }
        for i, vector in enumerate(vectors)
    ]
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)


def load_vectors(path: str | Path) -> list[SteeringVector]:
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata"]))
    return [
        SteeringVector(
            layer=int(item["layer"]),
            variable=item["variable"],
            condition=tuple(sorted((str(k), str(v)) for k, v in item["condition"].items())),
            values=archive[item["array"]],
            negative_count=int(item["negative_count"]),
            positive_count=int(item["positive_count"]),
            class_gap=float(item.get("class_gap", np.linalg.norm(archive[item["array"]]))),
        )
        for item in metadata
    ]


def condition_for_row(row: pd.Series, names: tuple[str, ...]):
    return tuple(sorted((name, str(row[name])) for name in names))
