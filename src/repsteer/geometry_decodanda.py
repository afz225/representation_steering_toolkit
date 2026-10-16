"""Production-exact Decodanda geometry.

Ports of ``geometry_olmo_reduced_full.py`` (example selection, the per-layer
SD -> PS/CCGP protocol, ``geometry_rows.csv`` summary columns) and of
``audit_matched_decodability_ccgp.py`` (matched named decoding + CCGP). Given
the same representations, the RNG call sequence and every Decodanda argument
match production, so summary values are bit-identical.
"""
from __future__ import annotations

import random
import time
from itertools import product
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

from .config import GeometryConfig
from .text import safe_text

BACKEND_NAME = "decodanda_olmo_reduced_all_unique_sd"


def _import_decodanda():
    try:
        from decodanda import Decodanda
        from decodanda.utilities import semantic_score
    except ImportError as exc:
        raise RuntimeError(
            "The Decodanda geometry engine requires the optional decodanda package "
            "(pip install 'repsteer[geometry]'), or select geometry.engine='native'."
        ) from exc
    return Decodanda, semantic_score


def select_examples(
    frame: pd.DataFrame,
    condition_fields: Sequence[str],
    n_per_cell: int,
    seed: int,
    *,
    text_column: str = "text",
    auto_reduce: bool = False,
    value_order: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """``load_examples`` with ``length_bin='all'``.

    Returns the selected rows (cell order, then shuffled order within each
    cell), the effective per-cell count, and the available-count table.
    Non-numeric condition columns are first encoded 0/1 by ``value_order``
    (declared ``[negative, positive]`` pairs); production only used 0/1 columns.
    """
    fields = list(condition_fields)
    missing = [column for column in dict.fromkeys([text_column, *fields]) if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required geometry columns: {missing}")
    df = frame.copy()
    df[text_column] = df[text_column].fillna("").map(safe_text)
    df = df[df[text_column].str.len() > 0].reset_index(drop=True)
    for column in fields:
        if not pd.api.types.is_numeric_dtype(df[column]):
            order = (value_order or {}).get(column)
            if order is None:
                raise ValueError(f"Non-numeric geometry variable {column!r} needs a declared value order")
            mapping = {str(order[0]): 0, str(order[1]): 1}
            unknown = set(df[column].astype(str)) - set(mapping)
            if unknown:
                raise ValueError(f"Unexpected values for {column!r}: {sorted(unknown)}")
            df[column] = df[column].astype(str).map(mapping)
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

    counts = df.groupby(fields, dropna=False).size().reset_index(name="available")
    expected = pd.MultiIndex.from_tuples(
        list(product([0, 1], repeat=len(fields))), names=fields).to_frame(index=False)
    counts = expected.merge(counts, on=fields, how="left").fillna({"available": 0})
    counts["available"] = counts["available"].astype(int)
    min_count = int(counts["available"].min()) if len(counts) else 0
    effective_n = int(n_per_cell)
    if min_count < n_per_cell:
        if auto_reduce and min_count > 0:
            effective_n = min_count
        else:
            raise RuntimeError(
                f"Underfilled condition space for {fields}: min={min_count} < n_per_cell={n_per_cell}\n"
                f"{counts.to_string(index=False)}")

    rng = random.Random(seed)
    keep: list[int] = []
    for _, sub in df.groupby(fields, dropna=False):
        indices = list(sub.index)
        rng.shuffle(indices)
        keep.extend(indices[:effective_n])
    return df.loc[keep].reset_index(drop=True), effective_n, counts


def condition_counts(metadata: pd.DataFrame, condition_fields: Sequence[str]) -> pd.DataFrame:
    fields = list(condition_fields)
    index = pd.MultiIndex.from_tuples(list(product([0, 1], repeat=len(fields))), names=fields)
    return metadata.groupby(fields).size().reindex(index, fill_value=0).reset_index(name="count")


def summarize_values(values) -> dict:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "n": 0}
    mean = float(np.nanmean(arr))
    if n == 1:
        return {"mean": mean, "std": float("nan"), "sem": float("nan"), "n": n}
    std = float(np.nanstd(arr, ddof=1))
    return {"mean": mean, "std": std, "sem": float(std / np.sqrt(n)), "n": n}


def _add_stats(out: dict, prefix: str, values, alias: bool = True) -> None:
    stats = summarize_values(values)
    if alias:
        out[prefix] = stats["mean"]
    out[f"{prefix}_mean"] = stats["mean"]
    out[f"{prefix}_std"] = stats["std"]
    out[f"{prefix}_sem"] = stats["sem"]
    out[f"{prefix}_n"] = stats["n"]


def make_decodanda(x: np.ndarray, metadata: pd.DataFrame, condition_fields: Sequence[str],
                   *, zscore: bool, sd_classifier: bool = False):
    Decodanda, _ = _import_decodanda()
    data = {"raster": np.asarray(x).astype(np.float32),
            "trial": np.arange(len(metadata), dtype=np.int64)}
    for column in condition_fields:
        data[column] = metadata[column].astype(int).to_numpy()
    classifier = "svc"
    if sd_classifier:
        classifier = LinearSVC(dual=False, C=1.0, class_weight="balanced", max_iter=1000, tol=0.001)
    return Decodanda(
        data=data,
        conditions={column: [0, 1] for column in condition_fields},
        neural_attr="raster",
        trial_attr="trial",
        min_data_per_condition=2,
        min_trials_per_condition=2,
        classifier=classifier,
        verbose=False,
        zscore=bool(zscore),
    )


def _dichotomy_key(dichotomy) -> str:
    return "_".join(dichotomy[0]) + "_v_" + "_".join(dichotomy[1])


def all_unique_dichotomies(dec) -> list[dict]:
    semantic, keys = dec._find_semantic_dichotomies()
    items: list[dict] = []
    seen: set[str] = set()
    for key, dichotomy in zip(keys, semantic):
        seen.add(_dichotomy_key(dichotomy))
        items.append({"dichotomy_id": str(key), "dichotomy_name": str(key), "is_named": 1,
                      "named_variable": str(key), "dic": dichotomy})
    for dichotomy in dec._find_nonsemantic_dichotomies():
        key = _dichotomy_key(dichotomy)
        if key in seen:
            continue
        seen.add(key)
        items.append({"dichotomy_id": key, "dichotomy_name": key, "is_named": 0,
                      "named_variable": "", "dic": dichotomy})
    return items


def _run_sd(dec, config: GeometryConfig, seed: int) -> tuple[dict, pd.DataFrame]:
    _, semantic_score = _import_decodanda()
    rows = []
    t0_total = time.perf_counter()
    for index, item in enumerate(all_unique_dichotomies(dec), start=1):
        np.random.seed(seed + index)
        random.seed(seed + index)
        t0 = time.perf_counter()
        perfs = dec.decode_dichotomy(
            item["dic"],
            training_fraction=config.sd_training_fraction,
            cross_validations=config.sd_cross_validations,
            ndata=config.sd_ndata,
        )
        stats = summarize_values(perfs)
        rows.append({
            "dichotomy_id": item["dichotomy_id"],
            "dichotomy_name": item["dichotomy_name"],
            "is_named": item["is_named"],
            "named_variable": item["named_variable"],
            "semantic_score": float(semantic_score(item["dic"])),
            "acc_mean": stats["mean"],
            "acc_std": stats["std"],
            "acc_sem": stats["sem"],
            "acc_n": stats["n"],
            "above_acc_threshold": (int(stats["mean"] >= config.sd_accuracy_threshold)
                                    if np.isfinite(stats["mean"]) else 0),
            "elapsed_sec": time.perf_counter() - t0,
        })
    df = pd.DataFrame(rows)
    acc = df["acc_mean"].to_numpy(dtype=float) if len(df) else np.asarray([])
    flags = df["above_acc_threshold"].to_numpy(dtype=float) if len(df) else np.asarray([])
    metrics = {
        "SD": summarize_values(flags)["mean"],
        "SD_mean_acc": summarize_values(acc)["mean"],
        "SD_frac_above_acc_threshold": summarize_values(flags)["mean"],
        "SD_frac_significant": float("nan"),
        "SD_null_mean_acc": float("nan"),
        "SD_total_time_sec": float(time.perf_counter() - t0_total),
        "SD_num_dichotomies": int(len(df)),
        "SD_num_named_dichotomies": int(df["is_named"].sum()) if len(df) else 0,
        "SD_num_random_dichotomies": 0,
        "SD_mode": "all_unique_dichotomies",
        "SD_num_null_shuffles": 0,
        "SD_acc_threshold": float(config.sd_accuracy_threshold),
        "SD_ndata": int(config.sd_ndata),
        "SD_training_fraction": float(config.sd_training_fraction),
        "SD_cross_validations": int(config.sd_cross_validations),
    }
    _add_stats(metrics, "SD", flags, alias=False)
    _add_stats(metrics, "SD_mean_acc", acc, alias=False)
    _add_stats(metrics, "SD_frac_above_acc_threshold", flags, alias=False)
    return metrics, df


def _run_ps_ccgp(dec, target: str, config: GeometryConfig) -> dict:
    dichotomy = dec._dichotomy_from_key(target)
    ps_scores, ps_null = dec.PS_with_nullmodel(
        dichotomy=dichotomy, nshuffles=0, max_semantic_dist=1,
        method="pearson", return_combinations=True)
    ccgp_scores, ccgp_null = dec.CCGP_with_nullmodel(
        dichotomy=dichotomy, resamplings=config.n_resamplings, nshuffles=0,
        ndata=config.ndata, max_semantic_dist=1, return_combinations=True)
    metrics: dict = {}
    _add_stats(metrics, f"PS_{target}", ps_scores)
    _add_stats(metrics, f"CCGP_{target}", ccgp_scores)
    metrics[f"PS_{target}_null_mean"] = summarize_values(ps_null)["mean"]
    metrics[f"CCGP_{target}_null_mean"] = summarize_values(ccgp_null)["mean"]
    return metrics


def production_layer_geometry(
    x: np.ndarray,
    metadata: pd.DataFrame,
    config: GeometryConfig,
    *,
    condition_fields: Sequence[str],
    target_fields: Sequence[str],
    layer: int,
    pooling: str,
    pooling_index: int,
    effective_n: int,
    model_id: str = "",
) -> tuple[list[dict], list[dict]]:
    """One (layer, pooling) of ``geometry_olmo_reduced_full.main``.

    Production ran one layer per process right after ``set_all_seeds(seed)``;
    the global RNGs are reseeded here to emulate that fresh process. SD runs
    first (per-dichotomy seeds ``sd_seed + i``), then PS/CCGP for each target
    continue the RNG stream without reseeding. Returns
    ``(geometry_rows, dichotomy_rows)`` with production column names.
    """
    fields = list(condition_fields)
    minimum = int(condition_counts(metadata, fields)["count"].min())
    if minimum < 2:
        raise RuntimeError(f"Minimum count over {fields} is {minimum}; need 2")
    if config.ndata > minimum:
        raise RuntimeError(f"geometry.ndata={config.ndata} > inferred cell count {minimum}")
    if config.sd_ndata > minimum:
        raise RuntimeError(f"geometry.sd_ndata={config.sd_ndata} > inferred cell count {minimum}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    dec = make_decodanda(x, metadata, fields, zscore=config.zscore, sd_classifier=False)
    sd_dec = make_decodanda(x, metadata, fields, zscore=config.zscore, sd_classifier=True)
    sd_seed = config.seed + int(layer) * 100 + int(pooling_index)
    sd_metrics, dichotomies = _run_sd(sd_dec, config, sd_seed)
    space = ",".join(fields)
    dichotomy_rows = []
    for _, item in dichotomies.iterrows():
        row = {"pooling": pooling, "rep_type": pooling, "layer": int(layer), "job_id": 0,
               "model_id": model_id, "condition_fields": space, "condition_space": space,
               "length_bin": "all", "sd_mode": "all_unique_dichotomies", "sd_seed": int(sd_seed)}
        row.update(item.to_dict())
        dichotomy_rows.append(row)

    rows = []
    for target in target_fields:
        if target not in fields:
            raise ValueError(f"target_field {target!r} is not in condition_fields {fields}")
        metrics = _run_ps_ccgp(dec, target, config)
        row = {
            "pooling": pooling, "rep_type": pooling, "layer": int(layer),
            "target_field": target, "metric_variable": target,
            "PS": metrics[f"PS_{target}"],
            "CCGP_nontargeted": metrics[f"CCGP_{target}"],
            "SD": sd_metrics["SD"],
            "SD_mean_acc": sd_metrics["SD_mean_acc"],
            "SD_frac_above_acc_threshold": sd_metrics["SD_frac_above_acc_threshold"],
            "condition_fields": space, "condition_space": space, "length_bin": "all",
            "n_per_cell_requested": int(config.n_per_cell),
            "n_per_cell_effective": int(effective_n),
            "decodanda_ndata": int(config.ndata),
            "n_examples": int(len(metadata)),
            "hidden_dim": int(np.asarray(x).shape[1]),
            "backend": BACKEND_NAME,
            "zscore": bool(config.zscore),
            "job_id": 0,
            "model_id": model_id,
        }
        for field in fields:
            for suffix in ("", "_mean", "_std", "_sem", "_n"):
                row.setdefault(f"PS_{field}{suffix}", metrics.get(f"PS_{field}{suffix}", float("nan")))
            for suffix in ("", "_mean", "_std", "_sem", "_n"):
                row.setdefault(f"CCGP_{field}{suffix}", metrics.get(f"CCGP_{field}{suffix}", float("nan")))
            row.setdefault(f"PS_{field}_null_mean", metrics.get(f"PS_{field}_null_mean", float("nan")))
            row.setdefault(f"CCGP_{field}_null_mean", metrics.get(f"CCGP_{field}_null_mean", float("nan")))
        row.update(metrics)
        row.update(sd_metrics)
        rows.append(row)
    return rows, dichotomy_rows


def _audit_summary(values) -> dict:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "sem": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
        "n": int(len(x)),
    }


def matched_audit_layer(
    x: np.ndarray,
    metadata: pd.DataFrame,
    config: GeometryConfig,
    *,
    condition_fields: Sequence[str],
    layer: int,
    n_per_cell: int,
    pooling: str = "hs_mean",
    model_id: str = "",
    dataset_csv: str = "",
) -> list[dict]:
    """One layer of ``audit_matched_decodability_ccgp.main`` (matched_metrics_layer_*.csv rows)."""
    fields = list(condition_fields)
    counts = condition_counts(metadata, fields)
    expected_cells = 2 ** len(fields)
    if len(metadata) != expected_cells * n_per_cell or not counts["count"].eq(n_per_cell).all():
        raise RuntimeError(f"Expected {n_per_cell} x {expected_cells} balanced points; got\n{counts}")
    dec = make_decodanda(x, metadata, fields, zscore=config.zscore, sd_classifier=False)
    rows = []
    for offset, field in enumerate(fields):
        np.random.seed(config.seed + int(layer) * 100 + offset)
        dichotomy = dec._dichotomy_from_key(field)
        decoding = dec.decode_dichotomy(
            dichotomy, training_fraction=config.audit_training_fraction,
            cross_validations=config.audit_cross_validations, ndata=config.ndata)
        np.random.seed(config.seed + int(layer) * 100 + offset)
        ccgp, _ = dec.CCGP_with_nullmodel(
            dichotomy, resamplings=config.n_resamplings, nshuffles=0,
            ndata=config.ndata, max_semantic_dist=1, return_combinations=True)
        cs = _audit_summary(ccgp)
        ds = _audit_summary(decoding)
        rows.append({
            "layer": int(layer), "target_field": field, "pooling": pooling,
            "zscore": bool(config.zscore), "n_examples": len(metadata), "n_per_cell": int(n_per_cell),
            "hidden_dim": int(np.asarray(x).shape[1]), "ndata": int(config.ndata),
            "decoding_training_fraction": config.audit_training_fraction,
            "decoding_cross_validations": config.audit_cross_validations,
            "ccgp_resamplings": config.n_resamplings,
            "decodability_mean": ds["mean"], "decodability_std": ds["std"],
            "decodability_sem": ds["sem"], "decodability_n": ds["n"],
            "ccgp_mean": cs["mean"], "ccgp_std": cs["std"],
            "ccgp_sem": cs["sem"], "ccgp_n_condition_splits": cs["n"],
            "model_id": model_id, "dataset_csv": dataset_csv,
        })
    return rows
