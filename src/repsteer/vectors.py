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
    # Median residual-stream norm at this layer (alpha_mode="relative_norm"); production archives have None.
    median_norm: float | None = None

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
    resample: bool = True,
) -> list[str]:
    """Class texts in file order, or the production ``DataFrame.sample`` draw.

    ``resample=False`` draws once (``random_state=sampling.seed + offset``),
    identical across layers — used with ``per_layer_resample=False``.
    """

    part = group[mask]
    if sampling is not None and resample:
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

    With ``sampling.per_layer_resample=False`` the class texts are drawn once
    (``random_state=seed + offset``, identical across layers) and — when the
    backend implements ``sentence_representations_layers`` — extracted in one
    pass over the union layer set; each layer's values stay bit-identical to the
    per-layer ``sentence_representations`` call.
    """

    layers = list(layers)
    resample = sampling is None or sampling.per_layer_resample
    layer_extractor = getattr(backend, "sentence_representations_layers", None)
    single_pass = not resample and layer_extractor is not None
    vectors: list[SteeringVector] = []
    for condition, group in _condition_groups(frame, spec.condition_on):
        labels = group[spec.name].astype(str).to_numpy()
        neg_mask = labels == spec.negative
        pos_mask = labels == spec.positive
        if not neg_mask.any() or not pos_mask.any():
            raise ValueError(f"Condition {dict(condition)} does not contain both target classes")
        if single_pass:
            offsets = (sampling.positive_seed_offset, sampling.negative_seed_offset)
            pos_texts = _contrast_texts(group, text_column, pos_mask, layers[0], offsets[0], sampling,
                                        resample=False)
            neg_texts = _contrast_texts(group, text_column, neg_mask, layers[0], offsets[1], sampling,
                                        resample=False)
            pos_by_layer = layer_extractor(pos_texts, layers, pooling, batch_size=batch_size,
                                           max_length=max_length)
            neg_by_layer = layer_extractor(neg_texts, layers, pooling, batch_size=batch_size,
                                           max_length=max_length)
            for layer in layers:
                vectors.append(_vector_for_layer(
                    layer, condition, spec, pos_by_layer[layer], neg_by_layer[layer],
                    normalize, len(neg_texts), len(pos_texts)))
            continue
        for layer in layers:
            offsets = (0, 0) if sampling is None else (
                sampling.positive_seed_offset, sampling.negative_seed_offset)
            pos_texts = _contrast_texts(group, text_column, pos_mask, layer, offsets[0], sampling,
                                        resample=resample)
            neg_texts = _contrast_texts(group, text_column, neg_mask, layer, offsets[1], sampling,
                                        resample=resample)
            pos = backend.sentence_representations(
                pos_texts, layer, pooling, batch_size=batch_size, max_length=max_length
            ).astype(np.float32)
            neg = backend.sentence_representations(
                neg_texts, layer, pooling, batch_size=batch_size, max_length=max_length
            ).astype(np.float32)
            vectors.append(_vector_for_layer(
                layer, condition, spec, pos, neg, normalize, len(neg_texts), len(pos_texts)))
    return vectors


def _vector_for_layer(
    layer: int,
    condition: tuple[tuple[str, str], ...],
    spec: VariableSpec,
    pos: np.ndarray,
    neg: np.ndarray,
    normalize: bool,
    negative_count: int,
    positive_count: int,
) -> SteeringVector:
    values = pos.astype(np.float32).mean(axis=0) - neg.astype(np.float32).mean(axis=0)
    norm = float(np.linalg.norm(values))
    if norm == 0 or not np.isfinite(norm):
        raise ValueError(f"Bad vector norm {norm} at layer {layer}, condition {dict(condition)}")
    if normalize:
        values = values / norm
    return SteeringVector(
        layer=layer,
        variable=spec.name,
        condition=tuple(sorted(condition)),
        values=values.astype(np.float32),
        negative_count=negative_count,
        positive_count=positive_count,
        class_gap=norm,
    )


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
            "median_norm": vector.median_norm,
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
            median_norm=(float(item["median_norm"]) if item.get("median_norm") is not None else None),
        )
        for item in metadata
    ]


def condition_for_row(row: pd.Series, names: tuple[str, ...]):
    return tuple(sorted((name, str(row[name])) for name in names))
