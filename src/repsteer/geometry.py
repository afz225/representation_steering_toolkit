from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _classifier(zscore: bool, seed: int):
    model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
    return make_pipeline(StandardScaler(), model) if zscore else model


def _fit_score(x_train, y_train, x_test, y_test, zscore: bool, seed: int) -> float:
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        return float("nan")
    classifier = _classifier(zscore, seed)
    classifier.fit(x_train, y_train)
    return float(accuracy_score(y_test, classifier.predict(x_test)))


def _condition_ids(frame: pd.DataFrame, variables: list[str]) -> np.ndarray:
    tuples = list(map(tuple, frame[variables].astype(str).to_numpy()))
    unique = {value: index for index, value in enumerate(sorted(set(tuples)))}
    return np.asarray([unique[value] for value in tuples], dtype=int)


def _balanced_split(
    condition_ids: np.ndarray,
    rng: np.random.Generator,
    max_per_condition: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    train: list[int] = []
    test: list[int] = []
    counts = [np.sum(condition_ids == condition) for condition in np.unique(condition_ids)]
    usable = min(counts)
    if max_per_condition is not None:
        usable = min(usable, max_per_condition)
    if usable < 2:
        raise ValueError("Every condition needs at least two samples for geometry")
    for condition in np.unique(condition_ids):
        indices = rng.permutation(np.flatnonzero(condition_ids == condition))[:usable]
        cut = max(1, usable // 2)
        train.extend(indices[:cut])
        test.extend(indices[cut:])
    return np.asarray(train), np.asarray(test)


def decodability(
    x: np.ndarray,
    frame: pd.DataFrame,
    variables: list[str],
    *,
    n_resamplings: int,
    zscore: bool,
    max_per_condition: int | None,
    seed: int,
) -> list[dict]:
    conditions = _condition_ids(frame, variables)
    rows: list[dict] = []
    for variable in variables:
        classes = {value: index for index, value in enumerate(sorted(frame[variable].astype(str).unique()))}
        y = frame[variable].astype(str).map(classes).to_numpy()
        for resampling in range(n_resamplings):
            rng = np.random.default_rng(seed + resampling)
            train, test = _balanced_split(conditions, rng, max_per_condition)
            rows.append(
                {
                    "metric": "decodability",
                    "variable": variable,
                    "resampling": resampling,
                    "value": _fit_score(x[train], y[train], x[test], y[test], zscore, seed),
                }
            )
    return rows


def parallelism_score(x: np.ndarray, frame: pd.DataFrame, target: str, variables: list[str]) -> float:
    nuisance = [name for name in variables if name != target]
    target_values = sorted(frame[target].astype(str).unique())
    if len(target_values) != 2:
        raise ValueError("Parallelism score requires a binary target")
    directions: list[np.ndarray] = []
    grouper = nuisance[0] if len(nuisance) == 1 else nuisance
    groups = [((), frame)] if not nuisance else frame.groupby(grouper, sort=True)
    for _, group in groups:
        negative = x[group.index[group[target].astype(str) == target_values[0]]]
        positive = x[group.index[group[target].astype(str) == target_values[1]]]
        if not len(negative) or not len(positive):
            continue
        direction = positive.mean(axis=0) - negative.mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            directions.append(direction / norm)
    if len(directions) < 2:
        return float("nan")
    similarities = [
        float(np.dot(left, right))
        for left, right in itertools.combinations(directions, 2)
    ]
    return float(np.mean(similarities))


def ccgp(
    x: np.ndarray,
    frame: pd.DataFrame,
    target: str,
    variables: list[str],
    *,
    n_resamplings: int,
    zscore: bool,
    max_per_condition: int | None,
    seed: int,
) -> list[dict]:
    nuisance = [name for name in variables if name != target]
    target_values = sorted(frame[target].astype(str).unique())
    y = frame[target].astype(str).eq(target_values[1]).astype(int).to_numpy()
    contexts = list(map(tuple, frame[nuisance].astype(str).to_numpy())) if nuisance else [()] * len(frame)
    unique_contexts = sorted(set(contexts))
    context_array = np.asarray(contexts, dtype=object)
    rows: list[dict] = []
    for held_index, held in enumerate(unique_contexts):
        held_mask = np.asarray([value == held for value in contexts])
        train_mask = ~held_mask
        for resampling in range(n_resamplings):
            rng = np.random.default_rng(seed + 1000 * held_index + resampling)
            train_indices: list[int] = []
            test_indices: list[int] = []
            limits = []
            for mask, destination in ((train_mask, train_indices), (held_mask, test_indices)):
                class_counts = [np.sum(mask & (y == label)) for label in (0, 1)]
                limit = min(class_counts)
                if max_per_condition is not None:
                    limit = min(limit, max_per_condition)
                limits.append(limit)
                for label in (0, 1):
                    candidates = np.flatnonzero(mask & (y == label))
                    destination.extend(rng.permutation(candidates)[:limit])
            if min(limits) < 1:
                value = float("nan")
            else:
                value = _fit_score(
                    x[train_indices], y[train_indices], x[test_indices], y[test_indices], zscore, seed
                )
            rows.append(
                {
                    "metric": "ccgp",
                    "variable": target,
                    "held_out_context": "|".join(held),
                    "resampling": resampling,
                    "value": value,
                }
            )
    return rows


def _balanced_dichotomies(n_conditions: int) -> Iterable[tuple[int, ...]]:
    if n_conditions % 2:
        raise ValueError("SD requires an even number of conditions")
    half = n_conditions // 2
    # Fix condition zero in the positive half to remove sign-complement duplicates.
    for remainder in itertools.combinations(range(1, n_conditions), half - 1):
        yield (0, *remainder)


def shattering_dimensionality(
    x: np.ndarray,
    frame: pd.DataFrame,
    variables: list[str],
    *,
    n_resamplings: int,
    zscore: bool,
    max_per_condition: int | None,
    max_dichotomies: int | None,
    seed: int,
) -> tuple[list[dict], float]:
    condition_ids = _condition_ids(frame, variables)
    n_conditions = len(np.unique(condition_ids))
    dichotomies = list(_balanced_dichotomies(n_conditions))
    if max_dichotomies is not None and len(dichotomies) > max_dichotomies:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(dichotomies), max_dichotomies, replace=False)
        dichotomies = [dichotomies[index] for index in sorted(selected)]
    rows: list[dict] = []
    for dichotomy_index, positive in enumerate(dichotomies):
        y = np.isin(condition_ids, positive).astype(int)
        scores = []
        for resampling in range(n_resamplings):
            rng = np.random.default_rng(seed + 10000 * dichotomy_index + resampling)
            train, test = _balanced_split(condition_ids, rng, max_per_condition)
            score = _fit_score(x[train], y[train], x[test], y[test], zscore, seed)
            scores.append(score)
            rows.append(
                {
                    "metric": "sd_dichotomy",
                    "variable": f"dichotomy_{dichotomy_index}",
                    "resampling": resampling,
                    "value": score,
                }
            )
    return rows, float(np.nanmean([row["value"] for row in rows]))


def geometry_metrics(
    x: np.ndarray,
    frame: pd.DataFrame,
    variables: list[str],
    *,
    n_resamplings: int = 50,
    zscore: bool = True,
    max_per_condition: int | None = None,
    max_sd_dichotomies: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute named decodability, CCGP, PS, and space-level SD."""

    work = frame.reset_index(drop=True)
    if len(work) != len(x):
        raise ValueError("Representation and metadata row counts differ")
    rows = decodability(
        x, work, variables, n_resamplings=n_resamplings, zscore=zscore,
        max_per_condition=max_per_condition, seed=seed
    )
    for variable in variables:
        rows.extend(
            ccgp(
                x, work, variable, variables, n_resamplings=n_resamplings, zscore=zscore,
                max_per_condition=max_per_condition, seed=seed
            )
        )
        rows.append(
            {
                "metric": "parallelism_score",
                "variable": variable,
                "resampling": -1,
                "value": parallelism_score(x, work, variable, variables),
            }
        )
    sd_rows, sd_mean = shattering_dimensionality(
        x, work, variables, n_resamplings=n_resamplings, zscore=zscore,
        max_per_condition=max_per_condition, max_dichotomies=max_sd_dichotomies, seed=seed
    )
    rows.extend(sd_rows)
    rows.append(
        {"metric": "sd_mean_accuracy", "variable": "condition_space", "resampling": -1, "value": sd_mean}
    )
    return pd.DataFrame(rows)
