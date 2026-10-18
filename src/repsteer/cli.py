from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace as dataclasses_replace
from pathlib import Path

import pandas as pd

import numpy as np

from .config import ModelConfig, load_config
from .types import Intervention
from .correlations import run_correlations
from .data import exclude_evaluation_rows, load_dataset, split_dataset, write_splits
from .evaluation import (evaluate_multiple_choice, evaluate_perplexity,
                         generate_instruction_responses)
from .geometry import geometry_metrics
from .geometry_decodanda import geometry_layer_task, select_examples
from .aggregate import run_aggregate, run_generic_aggregate, summarize_by_alpha
from .judge import judge_steering
from .models.huggingface import HuggingFaceBackend
from .offtarget import (AttributeLabels, judge_offtarget_effects,
                        measure_projection_drift, summarize_offtarget_effects,
                        summarize_projection_drift)
from .steering import (_norm_calibration_texts, layers_for_run, order_for_judging,
                       relative_coefficients, resolve_relative_depths, run_steering)
from .vectors import build_vectors, load_vectors, save_vectors


def _resume_keys(path: Path, id_column: str) -> set[tuple[int, float, str]] | None:
    """Completed resume keys (layer, coefficient, row id) from an existing steer
    output CSV; ``None`` when the file is absent or not a steer table."""

    if not path.exists() or path.stat().st_size == 0:
        return None
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.empty or not {"layer", "coefficient", id_column} <= set(frame.columns):
        return None
    return {
        (int(layer), float(coefficient), str(row_id))
        for layer, coefficient, row_id in zip(frame["layer"], frame["coefficient"], frame[id_column])
    }


def filter_completed(
    output: pd.DataFrame,
    keys: set[tuple[int, float, str]],
    id_column: str,
) -> pd.DataFrame:
    """Drop output rows already present in the resume CSV (layer, coefficient, id)."""

    if output.empty:
        return output
    keep = [
        (int(row.layer), float(row.coefficient), str(getattr(row, id_column))) not in keys
        for row in output.itertuples()
    ]
    return output[pd.Series(keep, index=output.index)].reset_index(drop=True)


def write_baseline_cache(output: pd.DataFrame, path: str, id_column: str) -> pd.DataFrame:
    """Persist baseline rows to a sidecar CSV and return the output unchanged.

    Baselines are unsteered and identical across alphas/layers within a run, so
    reusing them across runs only needs the id -> continuation mapping.
    """

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    baselines = output[[id_column, "baseline_continuation"]].drop_duplicates(id_column)
    if path_obj.exists() and path_obj.stat().st_size > 0:
        cached = pd.read_csv(path_obj, keep_default_na=False)
        updated = cached.set_index(id_column)
        for _, row in baselines.iterrows():
            updated.loc[row[id_column], "baseline_continuation"] = row["baseline_continuation"]
        fresh = baselines[~baselines[id_column].isin(updated.index)]
        combined = pd.concat([updated.reset_index(), fresh], ignore_index=True)
        combined.to_csv(path_obj, index=False)
    else:
        baselines.to_csv(path_obj, index=False)
    return output


def _frame(config, path: str | None):
    return load_dataset(config.dataset, path)


def _output_dir(path: str) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def command_validate(args) -> None:
    config = load_config(args.config)
    frame = _frame(config, args.dataset)
    summary = {"rows": len(frame), "variables": {
        name: frame[name].value_counts().sort_index().to_dict()
        for name in config.dataset.variables}}
    print(json.dumps(summary, indent=2))


def command_split(args) -> None:
    config = load_config(args.config)
    write_splits(split_dataset(_frame(config, args.dataset), config.dataset, config.split),
                 args.output_dir, config.dataset)


def command_vectors(args) -> None:
    config = load_config(args.config)
    steering = config.steering
    if args.layers:
        steering = dataclasses_replace(steering, layers=tuple(int(l) for l in args.layers.split(",")),
                                       relative_depths=None)
        config = dataclasses_replace(config, steering=steering)
    backend = HuggingFaceBackend(config.model)
    frame = _frame(config, args.dataset)
    layers = layers_for_run(steering, backend)
    vectors = build_vectors(backend, frame,
                            config.dataset.text_column, config.target,
                            layers, steering.pooling,
                            steering.normalize_vector, steering.batch_size,
                            steering.max_length, sampling=steering.sampling)
    if steering.alpha_mode == "relative_norm":
        norms = getattr(backend, "residual_norms", None)
        if norms is None:
            raise ValueError("alpha_mode=relative_norm needs a backend with residual_norms")
        calibrator = steering.norm_calibration_texts
        texts = _norm_calibration_texts(frame, config.dataset.text_column, calibrator)
        medians = norms(texts, layers, batch_size=steering.batch_size,
                        max_length=steering.max_length)
        vectors = [dataclasses_replace(vector, median_norm=medians[vector.layer])
                   for vector in vectors]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_vectors(vectors, args.output)


def command_steer(args) -> None:
    config = load_config(args.config)
    steering = config.steering
    overrides: dict[str, object] = {}
    if args.layers:
        overrides["layers"] = tuple(int(layer) for layer in args.layers.split(","))
        overrides["relative_depths"] = None
    if args.coefficients:
        overrides["coefficients"] = tuple(float(a) for a in args.coefficients.split(","))
    if overrides:
        steering = dataclasses_replace(steering, **overrides)
    config = dataclasses_replace(config, steering=steering)
    backend = HuggingFaceBackend(config.model)
    frame = _frame(config, args.dataset)
    if args.rows:
        start_s, _, end_s = args.rows.partition(":")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else len(frame)
        if start < 0 or end > len(frame) or start > end:
            raise ValueError(f"--rows {args.rows} is outside the dataset ({len(frame)} rows)")
        frame = frame.iloc[start:end].reset_index(drop=True)
    vectors = load_vectors(args.vectors)
    id_column = config.dataset.id_column
    resume_keys = _resume_keys(Path(args.output), id_column) if args.resume else None
    if resume_keys:
        # Skip eval rows whose whole layer x coefficient grid is already on disk, so a
        # resumed run does not regenerate them; partial rows are filtered after the run.
        per_row = pd.Series([key[2] for key in resume_keys]).value_counts()
        grid_size = len(layers_for_run(steering, backend)) * len(steering.coefficients)
        done = set(per_row[per_row >= grid_size].index)
        frame = frame[~frame[id_column].astype(str).isin(done)].reset_index(drop=True)
        if frame.empty:
            print(json.dumps({"resume": "all rows complete", "output": args.output}))
            return
    output = run_steering(backend, frame,
                          config.dataset.text_column, config.dataset.id_column, config.target,
                          vectors, steering, config.generation,
                          model_id=config.model.model_id)
    if resume_keys is not None:
        output = filter_completed(output, resume_keys, config.dataset.id_column)
    if args.baseline_cache:
        output = write_baseline_cache(output, args.baseline_cache, config.dataset.id_column)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    existing = Path(args.output)
    append = bool(args.resume) and existing.exists() and existing.stat().st_size > 0
    output.to_csv(existing, mode="a" if append else "w", header=not append, index=False)


def command_merge_steered(args) -> None:
    merged = order_for_judging(pd.read_csv(path, keep_default_na=False) for path in args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)


def command_judge(args) -> None:
    config = load_config(args.config)
    if config.judge is None:
        raise ValueError("The config has no judge section")
    judge = config.judge
    if getattr(args, "backend", None):
        judge = dataclasses_replace(judge, backend=args.backend)
    num_exec_shards = args.num_exec_shards if getattr(args, "num_exec_shards", None) else 1
    shard_index = getattr(args, "shard_index", 0) or 0
    if not 0 <= shard_index < num_exec_shards:
        raise ValueError(f"shard-index must be in [0, {num_exec_shards})")
    frame = pd.read_csv(args.input, keep_default_na=False)
    if judge.backend == "vllm":
        from .judge_vllm import VLLMJudgeBackend, judge_steering_vllm

        backend = VLLMJudgeBackend(judge)
        output = judge_steering_vllm(
            backend, frame, judge, judge_model_id=judge.model.model_id,
            shard_index=shard_index, num_exec_shards=num_exec_shards,
            resume_path=args.resume, prompt_preset=judge.prompt_preset)
    else:
        output = judge_steering(
            HuggingFaceBackend(judge.model), frame, judge,
            judge_model_id=judge.model.model_id, shard_index=shard_index,
            num_exec_shards=num_exec_shards, resume_path=args.resume,
            prompt_preset=judge.prompt_preset, judge_backend="hf",
            backend_config={"refusal_substrings": judge.refusal_substrings})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summarize_by_alpha(output).to_csv(Path(args.output).with_suffix(".summary.csv"), index=False)


def command_merge_judged(args) -> None:
    parts = [pd.read_csv(path, keep_default_na=False) for path in args.input]
    merged = merge_judged_shards(parts)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)


def merge_judged_shards(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate judged shards and verify full coverage of the input table.

    Every shard row carries its production judge ``position`` (``k``); the
    merged table must contain each (position, candidate_id) exactly once with
    no gaps relative to the sharded row positions of the source shards. Raises
    ``ValueError`` on gaps (missing shard rows) or duplicates.
    """

    if not parts:
        raise ValueError("merge-judged needs at least one --input shard")
    merged = pd.concat(parts, ignore_index=True)
    if "position" not in merged.columns:
        raise ValueError("Judged shards must carry the 'position' column for coverage checking")
    merged = merged.sort_values("position", kind="stable").reset_index(drop=True)
    positions = merged["position"].astype(int)
    duplicates = positions[positions.duplicated()]
    if len(duplicates):
        raise ValueError(f"Duplicate judged positions: {sorted(set(duplicates.tolist()))[:10]}")
    expected = np.arange(positions.min(), positions.min() + len(merged))
    if not (positions.to_numpy() == expected).all():
        missing = sorted(set(expected) - set(positions.tolist()))
        raise ValueError(f"Judged shard coverage gap: missing positions {missing[:10]}")
    return merged


def command_aggregate(args) -> None:
    if args.dataset == "generic":
        frame = pd.concat([pd.read_csv(path) for path in args.judged], ignore_index=True)
        tables = run_generic_aggregate(frame, args.output_dir, alpha_column=args.alpha_column)
    else:
        tables = run_aggregate(args.dataset, args.judged, args.output_dir,
                               alpha_column=args.alpha_column,
                               classifier_scores_path=args.classifier_scores,
                               geometry_rows_path=args.geometry_rows,
                               matched_metrics_path=args.matched_metrics)
    print(json.dumps({name: len(table) for name, table in tables.items()}, indent=2))


def command_build_cells(args) -> None:
    from .analysis import build_cells

    def read_all(paths):
        return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    cells = build_cells(read_all(args.layer_summary), read_all(args.geometry_rows),
                        read_all(args.matched_metrics), args.concept, args.target_field)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output, index=False)
    print(json.dumps({"rows": len(cells), "models": int(cells["model_id"].nunique()),
                      "output": args.output}, indent=2))


def command_analyze(args) -> None:
    from .analysis import run_analysis

    tables = run_analysis(args.cells, args.output_dir, args.model_meta, args.concept_meta)
    print(json.dumps({name: len(table) for name, table in tables.items()}, indent=2))


def _decodanda_geometry(config, backend, frame, args, output, relative_depth_of=None) -> None:
    geometry = config.geometry
    fields = list(geometry.variables or config.dataset.variables)
    targets = list(geometry.target_fields or fields)
    audit = geometry.protocol == "matched_audit"
    n_per_cell = (geometry.audit_n_per_cell or geometry.n_per_cell) if audit else geometry.n_per_cell
    selected, effective_n, counts = select_examples(
        frame, fields, n_per_cell, geometry.seed,
        text_column=config.dataset.text_column,
        auto_reduce=geometry.auto_reduce_n_per_cell and not audit,
        value_order=config.dataset.variables)
    counts.to_csv(output / "counts.csv", index=False)
    texts = selected[config.dataset.text_column].tolist()
    rows: list[dict] = []
    dichotomy_rows: list[dict] = []
    layers = list(geometry.layers)
    # One forward pass per pooling for all layers when the backend supports it
    # (bit-identical to per-layer extraction); row order stays layer-major.
    if hasattr(backend, "sentence_representations_layers"):
        reps_by_pooling = {pooling: backend.sentence_representations_layers(
            texts, layers, pooling, geometry.batch_size, geometry.max_length)
            for pooling in geometry.poolings}
    else:
        reps_by_pooling = None
    tasks = []
    for layer in layers:
        for pooling_index, pooling in enumerate(geometry.poolings):
            reps = (reps_by_pooling[pooling][layer] if reps_by_pooling is not None
                    else backend.sentence_representations(
                        texts, layer, pooling, geometry.batch_size, geometry.max_length))
            tasks.append(dict(
                audit=audit, x=reps, metadata=selected, config=geometry, condition_fields=fields,
                target_fields=targets, layer=layer, pooling=pooling, pooling_index=pooling_index,
                effective_n=effective_n, model_id=config.model.model_id, dataset_csv=str(args.dataset)))
    workers = int(getattr(args, "layer_workers", None) or 1)
    if workers > 1 and len(tasks) > 1:
        # Every layer reseeds NumPy's global RNG (the only RNG decodanda and sklearn draw
        # from) before its computations, so per-layer processes are bit-identical to the
        # sequential loop. spawn: children must not inherit the parent's CUDA state.
        import multiprocessing
        import os
        from concurrent.futures import ProcessPoolExecutor

        threads = str(max(1, (os.cpu_count() or workers) // workers))
        saved = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        os.environ.update({name: threads for name in saved})
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                results = list(pool.map(geometry_layer_task, tasks))
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    else:
        results = [geometry_layer_task(task) for task in tasks]
    for task, (layer_rows, layer_dichotomies) in zip(tasks, results):
        if relative_depth_of is not None:
            for row in (*layer_rows, *layer_dichotomies):
                row["relative_depth"] = relative_depth_of(task["layer"])
        rows.extend(layer_rows)
        dichotomy_rows.extend(layer_dichotomies)
    if audit:
        pd.DataFrame(rows).to_csv(output / "matched_metrics.csv", index=False)
    else:
        pd.DataFrame(rows).to_csv(output / "geometry_rows.csv", index=False)
        pd.DataFrame(dichotomy_rows).to_csv(output / "dichotomy_rows.csv", index=False)


def command_geometry(args) -> None:
    config = load_config(args.config)
    overrides = {}
    if args.protocol:
        overrides["protocol"] = args.protocol
    if args.layers:
        overrides["layers"] = tuple(int(layer) for layer in args.layers.split(","))
        overrides["relative_depths"] = None
    if overrides:
        config = dataclasses_replace(config, geometry=dataclasses_replace(config.geometry, **overrides))
    backend = HuggingFaceBackend(config.model)
    relative_depth_of = None
    if config.geometry.relative_depths is not None:
        layers = resolve_relative_depths(config.geometry.relative_depths, backend.n_layers)
        config = dataclasses_replace(config, geometry=dataclasses_replace(
            config.geometry, layers=layers, relative_depths=None))
        # Same realized-depth value the steering rows carry, so the two join exactly.
        relative_depth_of = lambda layer: round(layer / backend.n_layers, 2)  # noqa: E731
    frame = _frame(config, args.dataset).reset_index(drop=True)
    output = _output_dir(args.output_dir)
    if config.geometry.engine == "decodanda":
        _decodanda_geometry(config, backend, frame, args, output, relative_depth_of)
        return
    all_metrics = []
    texts = frame[config.dataset.text_column].astype(str).tolist()
    variables = list(config.dataset.variables)
    for pooling in config.geometry.poolings:
        for layer in config.geometry.layers:
            reps = backend.sentence_representations(
                texts, layer, pooling, config.geometry.batch_size, config.geometry.max_length)
            metrics = geometry_metrics(
                reps, frame, variables, n_resamplings=config.geometry.n_resamplings,
                zscore=config.geometry.zscore,
                max_per_condition=config.geometry.max_samples_per_condition,
                max_sd_dichotomies=config.geometry.max_sd_dichotomies,
                seed=config.geometry.seed)
            metrics.insert(0, "pooling", pooling)
            metrics.insert(0, "layer", layer)
            all_metrics.append(metrics)
    pd.concat(all_metrics, ignore_index=True).to_csv(output / "geometry_metrics.csv", index=False)


def command_eval_disjoint(args) -> None:
    config = load_config(args.config)
    remainder = exclude_evaluation_rows(_frame(config, args.dataset), config.dataset, config.split)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    remainder.to_csv(args.output, index=False)
    print(json.dumps({"rows": len(remainder), "output": args.output}, indent=2))


def command_perplexity(args) -> None:
    config = load_config(args.config)
    benchmark = pd.read_csv(args.benchmark)
    text_column = args.text_column or config.dataset.text_column
    details, summary = evaluate_perplexity(
        HuggingFaceBackend(config.model), benchmark[text_column].astype(str).tolist(),
        load_vectors(args.vectors), config.steering.layers, config.steering.coefficients,
        direction=args.direction, positions=config.steering.positions,
        max_length=config.steering.max_length, batch_size=config.steering.batch_size)
    output = _output_dir(args.output_dir)
    details.to_csv(output / "perplexity_rows.csv", index=False)
    summary.to_csv(output / "perplexity_summary.csv", index=False)


def command_multiple_choice(args) -> None:
    config = load_config(args.config)
    details, summary = evaluate_multiple_choice(
        HuggingFaceBackend(config.model), pd.read_csv(args.benchmark),
        load_vectors(args.vectors), config.steering.layers, config.steering.coefficients,
        question_column=args.question_column, id_column=args.id_column,
        direction=args.direction, positions=config.steering.positions,
        max_length=config.generation.max_length, batch_size=config.steering.batch_size)
    output = _output_dir(args.output_dir)
    details.to_csv(output / "multiple_choice_rows.csv", index=False)
    summary.to_csv(output / "multiple_choice_summary.csv", index=False)


def command_instruction_generate(args) -> None:
    config = load_config(args.config)
    output = generate_instruction_responses(
        HuggingFaceBackend(config.model), pd.read_csv(args.benchmark), load_vectors(args.vectors),
        config.steering.layers, config.steering.coefficients, config.generation,
        prompt_column=args.prompt_column, id_column=args.id_column,
        direction=args.direction, positions=config.steering.positions)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


def command_smoke_model(args) -> None:
    from .models.huggingface import HuggingFaceBackend

    config = ModelConfig(model_id=args.model_id, model_registry=args.registry)
    if args.dtype:
        config = dataclasses_replace(config, dtype=args.dtype)
    if args.device:
        config = dataclasses_replace(config, device_map=args.device)
    backend = HuggingFaceBackend(config)
    prompt = args.prompt
    rendered = backend._format_prompts([prompt])[0]
    bos_token = getattr(backend.tokenizer, "bos_token", None)
    bos_id = getattr(backend.tokenizer, "bos_token_id", None)
    bos_count = 0
    if bos_id is not None:
        encoded = backend.tokenizer(rendered, add_special_tokens=False).input_ids
        bos_count = sum(1 for token in encoded if token == bos_id)
    template_fallback = not backend.use_chat_template and rendered == prompt
    # Multimodal configs (Gemma 4) nest the decoder's hidden size in the text config.
    vec = np.ones(backend.model.config.get_text_config().hidden_size, dtype=np.float32)
    intervention = Intervention(layer=0, vector=vec, coefficient=1.0,
                                positions="all")
    tops = backend.residual_norms([prompt], (0, backend.n_layers // 2, backend.n_layers - 1),
                                  batch_size=1, max_length=512)
    baseline = backend.generate(
        [prompt], max_length=64, max_new_tokens=8, temperature=0.0, top_p=1.0,
        do_sample=False, seed=0)
    steered = backend.generate(
        [prompt], max_length=64, max_new_tokens=8, temperature=0.0, top_p=1.0,
        do_sample=False, seed=0, intervention=intervention)
    hook_effect = "".join(steered) != "".join(baseline)
    summary = {
        "model_id": config.model_id,
        "n_layers": backend.n_layers,
        "layer_path": backend.layer_path,
        "rendered_prompt": rendered,
        "bos_count": bos_count,
        "template_fallback": template_fallback,
        "hook_effect_nonzero": hook_effect,
        "residual_norms": {str(layer): norm for layer, norm in tops.items()},
    }
    print(json.dumps(summary, indent=2))


def command_correlations(args) -> None:
    rows = run_correlations(args.dataset, args.geometry, args.judged, args.output_dir,
                            audit_metrics_path=getattr(args, "audit_metrics", None))
    print(json.dumps({"rows": len(rows)}, indent=2))


def _named_paths(values: list[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        parsed[name] = path
    return parsed


def command_offtarget_project(args) -> None:
    config = load_config(args.config)
    axis_configs = _named_paths(args.axis_config)
    axis_vectors = _named_paths(args.axis_vectors)
    if set(axis_configs) != set(axis_vectors):
        raise ValueError("--axis-config and --axis-vectors names must match")
    measured_specs = {name: load_config(path).target for name, path in axis_configs.items()}
    measured_vectors = {name: load_vectors(path) for name, path in axis_vectors.items()}
    capture_layers = tuple(int(value) for value in args.capture_layers.split(","))
    details = measure_projection_drift(
        HuggingFaceBackend(config.model), _frame(config, args.dataset),
        text_column=config.dataset.text_column, id_column=config.dataset.id_column,
        steering_spec=config.target, steering_vectors=load_vectors(args.vectors),
        measured_specs=measured_specs, measured_vectors=measured_vectors,
        intervention_layer=args.intervention_layer, coefficient=args.coefficient,
        capture_layers=capture_layers, generation=config.generation,
        prompt_template=config.steering.prompt_template, positions=config.steering.positions)
    output = _output_dir(args.output_dir)
    details.to_csv(output / "projection_drift_rows.csv", index=False)
    summarize_projection_drift(details).to_csv(output / "projection_drift_summary.csv", index=False)


def _load_attribute_labels(path: str) -> dict[str, AttributeLabels]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {name: AttributeLabels(item["description"], tuple(item["labels"]),
                                  {str(k): str(v) for k, v in item["source_map"].items()})
            for name, item in raw.items()}


def command_offtarget_judge(args) -> None:
    config = load_config(args.config)
    if config.judge is None:
        raise ValueError("The config has no judge section")
    details = judge_offtarget_effects(
        HuggingFaceBackend(config.judge.model), pd.read_csv(args.input),
        _load_attribute_labels(args.labels), config.generation,
        steered_variable=config.target.name, max_new_tokens=config.judge.max_new_tokens)
    output = _output_dir(args.output_dir)
    details.to_csv(output / "offtarget_judge_rows.csv", index=False)
    summarize_offtarget_effects(details).to_csv(output / "offtarget_judge_summary.csv", index=False)


def _common_benchmark(parser):
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--direction", choices=("positive", "negative"), default="positive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repsteer")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate configuration and dataset")
    validate.add_argument("--config", required=True); validate.add_argument("--dataset")
    validate.set_defaults(func=command_validate)
    split = sub.add_parser("split", help="Create disjoint balanced data splits")
    split.add_argument("--config", required=True); split.add_argument("--dataset")
    split.add_argument("--output-dir", required=True); split.set_defaults(func=command_split)
    vectors = sub.add_parser("build-vectors", help="Build steering vectors")
    vectors.add_argument("--config", required=True); vectors.add_argument("--dataset", required=True)
    vectors.add_argument("--output", required=True)
    vectors.add_argument("--layers", help="Comma-separated override of steering.layers/depths")
    vectors.set_defaults(func=command_vectors)
    steer = sub.add_parser("steer", help="Generate paired baseline and steered outputs")
    steer.add_argument("--config", required=True); steer.add_argument("--dataset", required=True)
    steer.add_argument("--vectors", required=True); steer.add_argument("--output", required=True)
    steer.add_argument("--layers", help="Comma-separated override of steering.layers")
    steer.add_argument("--coefficients", help="Comma-separated override of steering.coefficients")
    steer.add_argument("--rows", help="Shard the eval rows as START:END (Python slice bounds)")
    steer.add_argument("--resume", action="store_true",
                       help="Append to --output and skip completed (layer, coefficient, id) keys")
    steer.add_argument("--baseline-cache", default=None,
                       help="Sidecar CSV mapping id -> baseline continuation, reused across runs")
    steer.set_defaults(func=command_steer)
    judge = sub.add_parser("judge", help="Judge paired steering outputs")
    judge.add_argument("--config", required=True); judge.add_argument("--input", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--shard-index", type=int, default=0,
                       help="0-based index of this execution shard (default 0)")
    judge.add_argument("--num-exec-shards", type=int, default=1,
                       help="total number of execution shards (default 1 = unsharded)")
    judge.add_argument("--resume", default=None,
                       help="existing judged CSV; rows already present are kept, "
                            "the rest are judged and appended")
    judge.add_argument("--backend", choices=("hf", "vllm"), default=None,
                       help="override judge.backend (e.g. to judge one table with both backends)")
    judge.set_defaults(func=command_judge)
    merge_judged = sub.add_parser(
        "merge-judged",
        help="Concatenate judged shards and verify full coverage (position column)")
    merge_judged.add_argument("--input", action="append", required=True,
                              help="judged shard CSV; repeat once per shard")
    merge_judged.add_argument("--output", required=True)
    merge_judged.set_defaults(func=command_merge_judged)
    merge = sub.add_parser(
        "merge-steered",
        help="Concatenate per-task steer tables in production judged-input order (layer, input)")
    merge.add_argument("--input", action="append", required=True,
                       help="steer output CSV; repeat in production task order")
    merge.add_argument("--output", required=True); merge.set_defaults(func=command_merge_steered)
    aggregate = sub.add_parser(
        "aggregate", help="Production outcome tables from judged rows (mean-over-alpha / best alpha)")
    aggregate.add_argument("--dataset", required=True, choices=("catdog", "toxicity", "generic"),
                           help="generic: scaling-study per-(model, concept, layer) outcome tables")
    aggregate.add_argument("--judged", action="append", required=True)
    aggregate.add_argument("--alpha-column", default="coefficient")
    aggregate.add_argument("--classifier-scores", default=None,
                           help="toxicity: CSV with text_hash,toxicity_probability (XLM-R)")
    aggregate.add_argument("--geometry-rows", default=None,
                           help="toxicity: geometry_rows.csv to join PS/CCGP/SD at the best alpha")
    aggregate.add_argument("--matched-metrics", default=None,
                           help="toxicity: matched_metrics.csv to join decodability")
    aggregate.add_argument("--output-dir", required=True); aggregate.set_defaults(func=command_aggregate)
    geometry = sub.add_parser("geometry", help="Extract representations and compute geometry")
    geometry.add_argument("--config", required=True); geometry.add_argument("--dataset", required=True)
    geometry.add_argument("--output-dir", required=True)
    geometry.add_argument("--protocol", choices=("production", "matched_audit"),
                          help="Override geometry.protocol (decodanda engine)")
    geometry.add_argument("--layers", help="Comma-separated override of geometry.layers")
    geometry.add_argument("--layer-workers", type=int, default=1,
                          help="decodanda engine: compute layers in this many processes (exact; default 1)")
    geometry.set_defaults(func=command_geometry)
    disjoint = sub.add_parser(
        "eval-disjoint", help="Write the dataset minus its steering-evaluation split rows")
    disjoint.add_argument("--config", required=True); disjoint.add_argument("--dataset")
    disjoint.add_argument("--output", required=True); disjoint.set_defaults(func=command_eval_disjoint)
    perplexity = sub.add_parser("perplexity", help="Measure token-weighted perplexity after steering")
    _common_benchmark(perplexity); perplexity.add_argument("--text-column")
    perplexity.set_defaults(func=command_perplexity)
    choice = sub.add_parser("multiple-choice", help="Evaluate next-token multiple-choice accuracy")
    _common_benchmark(choice); choice.add_argument("--question-column", default="question")
    choice.add_argument("--id-column"); choice.set_defaults(func=command_multiple_choice)
    instruction = sub.add_parser("instruction-generate", help="Generate M-IFEval-compatible responses")
    instruction.add_argument("--config", required=True); instruction.add_argument("--benchmark", required=True)
    instruction.add_argument("--vectors", required=True); instruction.add_argument("--output", required=True)
    instruction.add_argument("--prompt-column", default="prompt"); instruction.add_argument("--id-column", default="key")
    instruction.add_argument("--direction", choices=("positive", "negative"), default="positive")
    instruction.set_defaults(func=command_instruction_generate)
    project = sub.add_parser("offtarget-project", help="Measure online projection drift")
    project.add_argument("--config", required=True); project.add_argument("--dataset", required=True)
    project.add_argument("--vectors", required=True); project.add_argument("--axis-config", action="append", required=True)
    project.add_argument("--axis-vectors", action="append", required=True)
    project.add_argument("--intervention-layer", type=int, required=True)
    project.add_argument("--coefficient", type=float, required=True)
    project.add_argument("--capture-layers", required=True, help="Comma-separated readout layers")
    project.add_argument("--output-dir", required=True); project.set_defaults(func=command_offtarget_project)
    offjudge = sub.add_parser("offtarget-judge", help="Judge off-target attribute flips and brokenness")
    offjudge.add_argument("--config", required=True); offjudge.add_argument("--input", required=True)
    offjudge.add_argument("--labels", required=True); offjudge.add_argument("--output-dir", required=True)
    offjudge.set_defaults(func=command_offtarget_judge)
    correlations = sub.add_parser(
        "correlations",
        help="Full + partial geometry-vs-outcome correlations with production stars")
    correlations.add_argument("--dataset", required=True, choices=("catdog", "toxicity"))
    correlations.add_argument("--geometry", required=True)
    correlations.add_argument("--judged", required=True)
    correlations.add_argument("--audit-metrics", default=None)
    correlations.add_argument("--output-dir", required=True)
    correlations.set_defaults(func=command_correlations)
    cells = sub.add_parser(
        "build-cells",
        help="Join one concept's layer summary with its target field's geometry (analysis cells)")
    cells.add_argument("--layer-summary", action="append", required=True,
                       help="generic_layer_summary.csv (repeatable; concatenated)")
    cells.add_argument("--geometry-rows", action="append", required=True,
                       help="production geometry_rows.csv of the concept's space (repeatable)")
    cells.add_argument("--matched-metrics", action="append", required=True,
                       help="matched_audit matched_metrics.csv of the concept's space (repeatable)")
    cells.add_argument("--concept", required=True, help="steering task name (layer-summary concept)")
    cells.add_argument("--target-field", required=True, help="geometry target_field of the concept")
    cells.add_argument("--output", required=True); cells.set_defaults(func=command_build_cells)
    analyze = sub.add_parser(
        "analyze", help="Pre-registered mixed-model and secondary analyses over cells tables")
    analyze.add_argument("--cells", action="append", required=True)
    analyze.add_argument("--model-meta", default=None, help="CSV model_id,family,params_b,moe")
    analyze.add_argument("--concept-meta", default=None, help="CSV concept,abstractness")
    analyze.add_argument("--output-dir", required=True); analyze.set_defaults(func=command_analyze)
    smoke = sub.add_parser(
        "smoke-model",
        help="Load a model and report layers/template/hooks/norms as JSON")
    smoke.add_argument("--model-id", required=True)
    smoke.add_argument("--registry", default=None)
    smoke.add_argument("--dtype", default=None)
    smoke.add_argument("--device", default=None)
    smoke.add_argument("--prompt", default="What is the capital of France?")
    smoke.set_defaults(func=command_smoke_model)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
