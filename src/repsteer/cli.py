from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import load_config
from .correlations import run_correlations
from .data import exclude_evaluation_rows, load_dataset, split_dataset, write_splits
from .evaluation import (evaluate_multiple_choice, evaluate_perplexity,
                         generate_instruction_responses)
from .geometry import geometry_metrics
from .geometry_decodanda import matched_audit_layer, production_layer_geometry, select_examples
from .aggregate import run_aggregate, summarize_by_alpha
from .judge import judge_steering
from .models.huggingface import HuggingFaceBackend
from .offtarget import (AttributeLabels, judge_offtarget_effects,
                        measure_projection_drift, summarize_offtarget_effects,
                        summarize_projection_drift)
from .steering import order_for_judging, run_steering
from .vectors import build_vectors, load_vectors, save_vectors


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
    vectors = build_vectors(HuggingFaceBackend(config.model), _frame(config, args.dataset),
                            config.dataset.text_column, config.target,
                            config.steering.layers, config.steering.pooling,
                            config.steering.normalize_vector, config.steering.batch_size,
                            config.steering.max_length, sampling=config.steering.sampling)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_vectors(vectors, args.output)


def command_steer(args) -> None:
    config = load_config(args.config)
    output = run_steering(HuggingFaceBackend(config.model), _frame(config, args.dataset),
                          config.dataset.text_column, config.dataset.id_column, config.target,
                          load_vectors(args.vectors), config.steering, config.generation,
                          model_id=config.model.model_id)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


def command_merge_steered(args) -> None:
    merged = order_for_judging(pd.read_csv(path, keep_default_na=False) for path in args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)


def command_judge(args) -> None:
    config = load_config(args.config)
    if config.judge is None:
        raise ValueError("The config has no judge section")
    output = judge_steering(HuggingFaceBackend(config.judge.model),
                            pd.read_csv(args.input, keep_default_na=False), config.judge,
                            judge_model_id=config.judge.model.model_id)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summarize_by_alpha(output).to_csv(Path(args.output).with_suffix(".summary.csv"), index=False)


def command_aggregate(args) -> None:
    tables = run_aggregate(args.dataset, args.judged, args.output_dir,
                           alpha_column=args.alpha_column,
                           classifier_scores_path=args.classifier_scores,
                           geometry_rows_path=args.geometry_rows,
                           matched_metrics_path=args.matched_metrics)
    print(json.dumps({name: len(table) for name, table in tables.items()}, indent=2))


def _decodanda_geometry(config, backend, frame, args, output) -> None:
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
    for layer in geometry.layers:
        for pooling_index, pooling in enumerate(geometry.poolings):
            reps = backend.sentence_representations(
                texts, layer, pooling, geometry.batch_size, geometry.max_length)
            if audit:
                rows.extend(matched_audit_layer(
                    reps, selected, geometry, condition_fields=fields, layer=layer,
                    n_per_cell=effective_n, pooling=pooling, model_id=config.model.model_id,
                    dataset_csv=str(args.dataset)))
            else:
                summary, dichotomies = production_layer_geometry(
                    reps, selected, geometry, condition_fields=fields, target_fields=targets,
                    layer=layer, pooling=pooling, pooling_index=pooling_index,
                    effective_n=effective_n, model_id=config.model.model_id)
                rows.extend(summary)
                dichotomy_rows.extend(dichotomies)
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
    if overrides:
        config = replace(config, geometry=replace(config.geometry, **overrides))
    backend = HuggingFaceBackend(config.model)
    frame = _frame(config, args.dataset).reset_index(drop=True)
    output = _output_dir(args.output_dir)
    if config.geometry.engine == "decodanda":
        _decodanda_geometry(config, backend, frame, args, output)
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
    vectors.add_argument("--output", required=True); vectors.set_defaults(func=command_vectors)
    steer = sub.add_parser("steer", help="Generate paired baseline and steered outputs")
    steer.add_argument("--config", required=True); steer.add_argument("--dataset", required=True)
    steer.add_argument("--vectors", required=True); steer.add_argument("--output", required=True)
    steer.set_defaults(func=command_steer)
    judge = sub.add_parser("judge", help="Judge paired steering outputs")
    judge.add_argument("--config", required=True); judge.add_argument("--input", required=True)
    judge.add_argument("--output", required=True); judge.set_defaults(func=command_judge)
    merge = sub.add_parser(
        "merge-steered",
        help="Concatenate per-task steer tables in production judged-input order (layer, input)")
    merge.add_argument("--input", action="append", required=True,
                       help="steer output CSV; repeat in production task order")
    merge.add_argument("--output", required=True); merge.set_defaults(func=command_merge_steered)
    aggregate = sub.add_parser(
        "aggregate", help="Production outcome tables from judged rows (mean-over-alpha / best alpha)")
    aggregate.add_argument("--dataset", required=True, choices=("catdog", "toxicity"))
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
