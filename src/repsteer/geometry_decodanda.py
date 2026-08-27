from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

from .config import GeometryConfig


def _import_decodanda():
    try:
        from decodanda import Decodanda
    except ImportError as exc:
        raise RuntimeError(
            "The Decodanda geometry engine requires the optional decodanda package. "
            "Install it or select geometry.engine='native'."
        ) from exc
    return Decodanda


def _binary_metadata(frame: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    metadata = pd.DataFrame(index=frame.index)
    for variable in variables:
        values = sorted(frame[variable].astype(str).unique())
        if len(values) != 2:
            raise ValueError(f"Decodanda requires binary variable {variable!r}; found {values}")
        mapping = {values[0]: 0, values[1]: 1}
        metadata[variable] = frame[variable].astype(str).map(mapping).astype(int)
    return metadata


def _make_decodanda(
    x: np.ndarray,
    metadata: pd.DataFrame,
    variables: list[str],
    *,
    zscore: bool,
    sd_classifier: bool,
):
    Decodanda = _import_decodanda()
    data = {
        "raster": np.asarray(x, dtype=np.float32),
        "trial": np.arange(len(metadata), dtype=np.int64),
    }
    for variable in variables:
        data[variable] = metadata[variable].to_numpy(dtype=int)
    classifier = (
        LinearSVC(
            dual=False,
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            tol=0.001,
        )
        if sd_classifier
        else "svc"
    )
    return Decodanda(
        data=data,
        conditions={variable: [0, 1] for variable in variables},
        neural_attr="raster",
        trial_attr="trial",
        min_data_per_condition=2,
        min_trials_per_condition=2,
        classifier=classifier,
        verbose=False,
        zscore=zscore,
    )


def _dichotomy_key(dichotomy: list[list[str]]) -> str:
    return "_".join(dichotomy[0]) + "_v_" + "_".join(dichotomy[1])


def _all_unique_dichotomies(dec) -> list[dict]:
    semantic_dichotomies, semantic_keys = dec._find_semantic_dichotomies()
    items: list[dict] = []
    seen: set[str] = set()
    for name, dichotomy in zip(semantic_keys, semantic_dichotomies):
        key = _dichotomy_key(dichotomy)
        seen.add(key)
        items.append(
            {
                "id": str(name),
                "name": str(name),
                "named_variable": str(name),
                "is_named": True,
                "dichotomy": dichotomy,
            }
        )
    for dichotomy in dec._find_nonsemantic_dichotomies():
        key = _dichotomy_key(dichotomy)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "id": key,
                "name": key,
                "named_variable": "",
                "is_named": False,
                "dichotomy": dichotomy,
            }
        )
    return items


def decodanda_geometry_metrics(
    x: np.ndarray,
    frame: pd.DataFrame,
    variables: list[str],
    config: GeometryConfig,
) -> pd.DataFrame:
    """Run the same Decodanda PS/CCGP/SD procedure as the prior production scripts."""

    work = frame.reset_index(drop=True)
    if len(work) != len(x):
        raise ValueError("Representation and metadata row counts differ")
    metadata = _binary_metadata(work, variables)
    counts = metadata.groupby(variables).size()
    minimum = int(counts.min())
    if config.ndata > minimum:
        raise ValueError(f"geometry.ndata={config.ndata} exceeds minimum cell count {minimum}")
    if config.sd_ndata > minimum:
        raise ValueError(f"geometry.sd_ndata={config.sd_ndata} exceeds minimum cell count {minimum}")

    dec = _make_decodanda(x, metadata, variables, zscore=config.zscore, sd_classifier=False)
    sd_dec = _make_decodanda(x, metadata, variables, zscore=config.zscore, sd_classifier=True)
    rows: list[dict] = []

    np.random.seed(config.seed)
    random.seed(config.seed)
    for variable in variables:
        dichotomy = dec._dichotomy_from_key(variable)
        ps_values, _ = dec.PS_with_nullmodel(
            dichotomy=dichotomy,
            nshuffles=0,
            max_semantic_dist=1,
            method="pearson",
            return_combinations=True,
        )
        for combination, value in enumerate(np.asarray(ps_values, dtype=float).ravel()):
            rows.append(
                {
                    "metric": "parallelism_score",
                    "variable": variable,
                    "resampling": combination,
                    "value": float(value),
                    "engine": "decodanda",
                }
            )

        ccgp_values, _ = dec.CCGP_with_nullmodel(
            dichotomy=dichotomy,
            resamplings=config.n_resamplings,
            nshuffles=0,
            ndata=config.ndata,
            max_semantic_dist=1,
            return_combinations=True,
        )
        for combination, value in enumerate(np.asarray(ccgp_values, dtype=float).ravel()):
            rows.append(
                {
                    "metric": "ccgp",
                    "variable": variable,
                    "resampling": combination,
                    "value": float(value),
                    "engine": "decodanda",
                }
            )

    dichotomy_means: list[float] = []
    dichotomies = _all_unique_dichotomies(sd_dec)
    if config.max_sd_dichotomies is not None and len(dichotomies) > config.max_sd_dichotomies:
        named = [item for item in dichotomies if item["is_named"]]
        unnamed = [item for item in dichotomies if not item["is_named"]]
        remaining = config.max_sd_dichotomies - len(named)
        if remaining < 0:
            raise ValueError("max_sd_dichotomies cannot be smaller than the number of named variables")
        rng = np.random.default_rng(config.seed)
        chosen = sorted(rng.choice(len(unnamed), remaining, replace=False).tolist())
        dichotomies = named + [unnamed[index] for index in chosen]

    for index, item in enumerate(dichotomies, start=1):
        np.random.seed(config.seed + index)
        random.seed(config.seed + index)
        values = np.asarray(
            sd_dec.decode_dichotomy(
                item["dichotomy"],
                training_fraction=config.sd_training_fraction,
                cross_validations=config.sd_cross_validations,
                ndata=config.sd_ndata,
            ),
            dtype=float,
        ).ravel()
        mean_value = float(np.nanmean(values))
        dichotomy_means.append(mean_value)
        metric = "decodability" if item["is_named"] else "sd_dichotomy"
        variable = item["named_variable"] if item["is_named"] else item["id"]
        for resampling, value in enumerate(values):
            rows.append(
                {
                    "metric": metric,
                    "variable": variable,
                    "dichotomy_id": item["id"],
                    "is_named": item["is_named"],
                    "resampling": resampling,
                    "value": float(value),
                    "engine": "decodanda",
                }
            )

    rows.append(
        {
            "metric": "sd_mean_accuracy",
            "variable": "condition_space",
            "resampling": -1,
            "value": float(np.nanmean(dichotomy_means)),
            "engine": "decodanda",
        }
    )
    rows.append(
        {
            "metric": "sd_fraction_above_threshold",
            "variable": "condition_space",
            "resampling": -1,
            "value": float(
                np.mean(np.asarray(dichotomy_means) >= config.sd_accuracy_threshold)
            ),
            "engine": "decodanda",
        }
    )
    return pd.DataFrame(rows)
