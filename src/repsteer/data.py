from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from .config import DatasetConfig, SplitConfig


def load_dataset(config: DatasetConfig, path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path or config.path)
    frame = pd.read_csv(source)
    validate_dataset(frame, config)
    return frame


def validate_dataset(frame: pd.DataFrame, config: DatasetConfig) -> None:
    required = {config.id_column, config.text_column, *config.variables}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if frame[list(required)].isnull().any().any():
        raise ValueError("Required dataset columns contain null values")
    if frame[config.id_column].duplicated().any():
        raise ValueError(f"IDs in {config.id_column!r} must be unique")
    for variable, allowed in config.variables.items():
        observed = set(frame[variable].astype(str).unique())
        unexpected = observed - set(allowed)
        if unexpected:
            raise ValueError(f"Unexpected values for {variable!r}: {sorted(unexpected)}")


def condition_cells(config: DatasetConfig) -> list[tuple[str, ...]]:
    return list(itertools.product(*(config.variables[name] for name in config.variables)))


def split_dataset(
    frame: pd.DataFrame,
    dataset: DatasetConfig,
    split: SplitConfig,
) -> dict[str, pd.DataFrame]:
    """Create disjoint, deterministic splits while preserving every condition cell."""

    names = list(dataset.variables)
    parts: dict[str, list[pd.DataFrame]] = {"vector": [], "evaluation": [], "geometry": []}
    for cell_index, cell in enumerate(condition_cells(dataset)):
        mask = pd.Series(True, index=frame.index)
        for name, value in zip(names, cell):
            mask &= frame[name].astype(str).eq(value)
        group = frame.loc[mask].sample(frac=1.0, random_state=split.seed + cell_index)
        requested = split.vector_per_cell + split.eval_per_cell
        if len(group) < requested:
            raise ValueError(f"Cell {dict(zip(names, cell))} has {len(group)} rows; needs {requested}")
        v_end = split.vector_per_cell
        e_end = v_end + split.eval_per_cell
        parts["vector"].append(group.iloc[:v_end])
        parts["evaluation"].append(group.iloc[v_end:e_end])
        geometry = group.iloc[e_end:]
        if split.geometry_per_cell is not None:
            if len(geometry) < split.geometry_per_cell:
                raise ValueError(
                    f"Cell {dict(zip(names, cell))} lacks {split.geometry_per_cell} geometry rows"
                )
            geometry = geometry.iloc[: split.geometry_per_cell]
        parts["geometry"].append(geometry)
    return {
        name: pd.concat(groups, ignore_index=True) if groups else frame.iloc[:0].copy()
        for name, groups in parts.items()
    }


def write_splits(splits: dict[str, pd.DataFrame], output_dir: str | Path, dataset: DatasetConfig) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ids: dict[str, set[str]] = {}
    for name, frame in splits.items():
        frame.to_csv(output / f"{name}.csv", index=False)
        ids[name] = set(frame[dataset.id_column].astype(str))
    overlap = {
        f"{left}:{right}": len(ids[left] & ids[right])
        for left, right in itertools.combinations(ids, 2)
    }
    manifest = {
        "rows": {name: len(frame) for name, frame in splits.items()},
        "overlap": overlap,
        "variables": {name: list(values) for name, values in dataset.variables.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
