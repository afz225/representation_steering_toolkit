from __future__ import annotations

import hashlib
from collections.abc import Iterable

import pandas as pd

from .config import GenerationConfig, SteeringConfig
from .models.base import ModelBackend
from .text import safe_text
from .types import Intervention, VariableSpec
from .vectors import SteeringVector, condition_for_row, vector_index


class _FormatDict(dict):
    def __missing__(self, key):
        raise KeyError(f"Prompt template references missing column {key!r}")


def format_prompt(template: str, row: pd.Series, text_column: str, source: str, target: str) -> str:
    """Render a prompt; ``{text}`` is the raw text, ``{safe_text}`` its whitespace-collapsed form."""

    values = _FormatDict({str(key): value for key, value in row.items()})
    text = str(row[text_column])
    values.update(text=text, safe_text=safe_text(text), source=source, target=target)
    return template.format_map(values)


def template_for_row(steering: SteeringConfig, row: pd.Series) -> str:
    if steering.prompt_templates is None:
        return steering.prompt_template
    key = str(row[steering.prompt_template_column])
    if key not in steering.prompt_templates:
        raise KeyError(f"No prompt template for {steering.prompt_template_column}={key!r}")
    return steering.prompt_templates[key]


def steered_seed(generation: GenerationConfig, row_number: int, **fields) -> int:
    """Production per-row steered seed (sha256 of the seed material) or ``seed + row``."""

    if generation.steered_seed_material is None:
        return generation.seed + row_number
    material = generation.steered_seed_material.format(**fields)
    return generation.seed + int(hashlib.sha256(material.encode()).hexdigest()[:8], 16) % 1_000_000


def resolve_relative_depths(depths: Iterable[float], n_layers: int) -> tuple[int, ...]:
    """Relative depths {0.08, ..., 0.92} -> layer indices against ``n_layers``.

    Each depth is ``round(depth * n_layers)`` clipped to ``[0, n_layers - 1]``,
    then consecutive duplicates are removed (rounding can collide).
    """

    resolved: list[int] = []
    for depth in depths:
        if not 0.0 < depth <= 1.0:
            raise ValueError(f"Relative depth {depth} is outside (0, 1]")
        layer = min(max(int(round(depth * n_layers)), 0), n_layers - 1)
        if not resolved or resolved[-1] != layer:
            resolved.append(layer)
    return tuple(resolved)


def layers_for_run(steering: SteeringConfig, backend: ModelBackend) -> tuple[int, ...]:
    """Steering layer indices from ``layers`` or resolved ``relative_depths``."""

    if steering.relative_depths is not None:
        return resolve_relative_depths(steering.relative_depths, backend.n_layers)
    return tuple(steering.layers)


def relative_coefficients(
    steering: SteeringConfig,
    vectors: list[SteeringVector],
    layers: Iterable[int],
) -> dict[int, tuple[tuple[float, float], ...]]:
    """Per steering layer: ``(alpha_rel, absolute coefficient)`` pairs.

    ``alpha_mode="relative_norm"``: absolute coefficient = alpha_rel x the
    layer's median residual norm (from the vectors npz). ``absolute``: the
    coefficient is used as-is for every layer.
    """

    if steering.alpha_mode != "relative_norm":
        return {layer: tuple((float(c), float(c)) for c in steering.coefficients)
                for layer in layers}
    index = {(vector.layer, vector.condition): vector for vector in vectors}
    grid: dict[int, tuple[tuple[float, float], ...]] = {}
    for layer in layers:
        conditions = [key[1] for key in index if key[0] == layer]
        if not conditions:
            raise KeyError(f"No vector for layer={layer}")
        norms = {index[(layer, condition)].median_norm for condition in conditions}
        if len(norms) != 1:
            raise ValueError(f"Layer {layer} vectors disagree on median_norm: {sorted(norms)}")
        median_norm = norms.pop()
        if median_norm is None:
            raise ValueError(
                "alpha_mode=relative_norm needs median_norm in the vectors npz; "
                "rebuild vectors with alpha_mode='relative_norm'"
            )
        grid[layer] = tuple((float(alpha_rel), float(alpha_rel) * float(median_norm))
                            for alpha_rel in steering.coefficients)
    return grid


def _norm_calibration_texts(frame: pd.DataFrame, text_column: str, count: int) -> list[str]:
    """Deterministic calibration texts drawn from the vector split.

    The vector split pools ``vector_per_cell`` rows per condition cell in
    ``split_dataset`` order (cell-major); take the first ``count`` texts from it.
    """

    texts = frame[text_column].astype(str).tolist()
    return texts[:count]


def run_steering(
    backend: ModelBackend,
    frame: pd.DataFrame,
    text_column: str,
    id_column: str,
    spec: VariableSpec,
    vectors: list[SteeringVector],
    steering: SteeringConfig,
    generation: GenerationConfig,
    *,
    model_id: str = "",
) -> pd.DataFrame:
    """Generate paired baseline and opposite-target steered continuations.

    Production order and semantics (gemma4_eval200_steering.py /
    gemma4_toxicity_steering.py): one unsteered baseline per evaluation row
    (seed ``seed + baseline_seed_offset + row``), then rows ordered layer ->
    coefficient -> evaluation row. The hook adds ``sign * coefficient`` times the
    (unsigned) vector, where sign is +1 when the target is the positive pole.

    Scaling-study extensions (all default off/production-exact):
    ``relative_depths`` resolves the layer set against ``backend.n_layers``;
    ``alpha_mode="relative_norm"`` scales each ``alpha_rel`` by the layer's
    ``median_norm`` from the vectors npz; ``batched_grid`` covers the baseline
    plus every layer×alpha row with one ``generate_steered_batch`` call per eval
    prompt (row layout identical to the production path).
    """

    index = vector_index(vectors)
    task = steering.task or spec.name
    setup = steering.setup or ("conditioned" if spec.condition_on else "general")
    layers = layers_for_run(steering, backend)
    grid = relative_coefficients(steering, vectors, layers)
    decoding = generation.generate_kwargs()
    prepared = []
    for row_number, (_, row) in enumerate(frame.iterrows()):
        source = str(row[spec.name])
        target = spec.opposite(source)
        prompt = format_prompt(template_for_row(steering, row), row, text_column, source, target)
        baseline = backend.generate(
            [prompt],
            intervention=None,
            **{**decoding, "seed": generation.seed + generation.baseline_seed_offset + row_number},
        )[0]
        prepared.append((row_number, row, source, target, prompt, baseline))

    def result_row(
        row: pd.Series, source: str, target: str, prompt: str, baseline: str, steered: str,
        layer: int, alpha_rel: float, coefficient: float,
    ) -> dict:
        condition = condition_for_row(row, spec.condition_on)
        signed = spec.sign_for_target(target) * float(coefficient)
        result = row.to_dict()
        result.update(
            task=task,
            setup=setup,
            source_value=source,
            target_value=target,
            target_attribute=steering.target_labels.get(target, target),
            layer=layer,
            relative_depth=round(layer / backend.n_layers, 2),
            coefficient=coefficient,
            alpha_rel=alpha_rel,
            signed_coefficient=signed,
            vector_condition="|".join(f"{k}={v}" for k, v in condition) or "general",
            prompt=prompt,
            baseline_continuation=baseline,
            steered_continuation=steered,
            model_id=model_id,
            concept=task,
        )
        result[id_column] = row[id_column]
        return result

    rows: list[dict] = []
    if steering.batched_grid:
        batch_fn = getattr(backend, "generate_steered_batch", None)
        if batch_fn is None:
            raise ValueError("batched_grid=True needs a backend with generate_steered_batch")
        # One batch per eval prompt: baseline row first, then layer-major
        # (layer -> alpha) rows. Production output rows are row-major within
        # each (layer, alpha): layer -> coefficient -> row, so the batch's rows
        # are regrouped below.
        per_prompt: list[tuple[int, pd.Series, str, str, str, str, list[str]]] = []
        for row_number, row, source, target, prompt, baseline in prepared:
            prompt_interventions: list[Intervention | None] = [None]
            for layer in layers:
                condition = condition_for_row(row, spec.condition_on)
                key = (layer, condition)
                if key not in index:
                    raise KeyError(f"No vector for layer={layer}, condition={dict(condition)}")
                for alpha_rel, coefficient in grid[layer]:
                    prompt_interventions.append(Intervention(
                        layer=layer,
                        vector=index[key].values,
                        coefficient=spec.sign_for_target(target) * float(coefficient),
                        positions=steering.positions,
                    ))
            texts = batch_fn(
                prompt, prompt_interventions,
                max_length=generation.max_length, max_new_tokens=generation.max_new_tokens,
                seed=generation.seed, normalize_whitespace=generation.normalize_whitespace,
                pad_with_eos=generation.pad_with_eos,
            )
            if len(texts) != len(prompt_interventions):
                raise ValueError(
                    f"generate_steered_batch returned {len(texts)} rows for "
                    f"{len(prompt_interventions)} interventions"
                )
            per_prompt.append((row_number, row, source, target, prompt, baseline, texts))
        by_grid: dict[tuple[int, float], list[dict]] = {}
        for row_number, row, source, target, prompt, baseline, texts in per_prompt:
            position = 1
            for layer in layers:
                for alpha_rel, coefficient in grid[layer]:
                    by_grid.setdefault((layer, alpha_rel), []).append(result_row(
                        row, source, target, prompt, baseline, texts[position],
                        layer, alpha_rel, coefficient,
                    ))
                    position += 1
        for layer in layers:
            for alpha_rel, _coefficient in grid[layer]:
                rows.extend(by_grid[(layer, alpha_rel)])
        return pd.DataFrame(rows)

    for layer in layers:
        for alpha_rel, coefficient in grid[layer]:
            for row_number, row, source, target, prompt, baseline in prepared:
                condition = condition_for_row(row, spec.condition_on)
                key = (layer, condition)
                if key not in index:
                    raise KeyError(f"No vector for layer={layer}, condition={dict(condition)}")
                signed = spec.sign_for_target(target) * float(coefficient)
                intervention = Intervention(
                    layer=layer,
                    vector=index[key].values,
                    coefficient=signed,
                    positions=steering.positions,
                )
                seed = steered_seed(
                    generation, row_number, setup=setup, model_id=model_id, task=task,
                    layer=layer, alpha=float(coefficient), id=row[id_column],
                )
                steered = backend.generate(
                    [prompt], intervention=intervention, **{**decoding, "seed": seed}
                )[0]
                rows.append(result_row(
                    row, source, target, prompt, baseline, steered,
                    layer, alpha_rel, coefficient,
                ))
    return pd.DataFrame(rows)


def order_for_judging(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-task steering tables in production judged-input order.

    Production merged every layer's generations file, whose rows run task (in
    the given order) -> alpha -> evaluation row; ``run_steering`` tables run
    layer -> coefficient -> row, so a stable sort on (layer, input rank)
    reproduces ``results-gemma4-steering-eval200/{setup}_generations.csv``.
    """

    parts = [frame.assign(_input_rank=rank) for rank, frame in enumerate(frames)]
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.sort_values(["layer", "_input_rank"], kind="stable")
    return merged.drop(columns="_input_rank").reset_index(drop=True)
