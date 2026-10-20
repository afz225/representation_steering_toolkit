#!/usr/bin/env python3
"""Create balanced, disjoint toxicity geometry/vector/evaluation splits.

Port of production `prepare_toxicity_gemma4_splits.py`, kept logic-identical so
that the split assignment reproduces the shipped `manifest.json` exactly:
- input: per-language labeled CSVs from the gpt-4.1-mini annotation pipeline
  (columns: text, toxic, mentions_gender, mentions_location, ...)
- filters: toxic/mentions_gender must be 0/1; mentions_location must be 0;
  non-empty, deduplicated texts
- candidate_id: first 16 hex chars of sha256(f"{language}|{text}") over the
  UTF-8 whitespace-stripped text
- cells: (language, toxicity_bin, gender_bin); within each cell the rows are
  shuffled with pandas `sample(frac=1, random_state=seed + cell_index)` and
  sliced in fixed order geometry (50) -> vectors (32) -> evaluation (25)
- cells are enumerated with `groupby(..., sort=True)` so cell indices are stable

Provenance: production run was
  python prepare_toxicity_gemma4_splits.py \
      --input_dir openai_labeling_outputs/labeled \
      --output_dir results-gemma4-toxicity-split \
      --geometry_per_cell 50 --vectors_per_cell 32 --evaluation_per_cell 25 --seed 42

After generating, verify against the shipped manifest:
  python datasets/toxicity/verify_split.py --splits-dir <output_dir> --manifest datasets/toxicity/manifest.json
(Labels from a fresh gpt-4.1-mini run can differ from the production ones; for an exact
rebuild from the original texts use datasets/toxicity/rebuild_from_manifest.py instead.)
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

SPLIT_SIZES = {
    "geometry_400": 50,
    "vector_construction_256": 32,
    "evaluation_200": 25,
}

COLUMNS = [
    "text",
    "toxicity",
    "toxicity_bin",
    "gender",
    "gender_bin",
    "language",
    "language_bin",
    "candidate_id",
]


def build_splits(
    input_dir,
    output_dir,
    geometry_per_cell=50,
    vectors_per_cell=32,
    evaluation_per_cell=25,
    seed=42,
):
    """Deterministically build the three split CSVs (see module docstring)."""
    pieces = []
    for language, language_bin in (("en", 0), ("ar", 1)):
        frame = pd.read_csv(Path(input_dir) / f"{language}_labeled.csv")
        frame = frame[
            frame["toxic"].isin([0, 1])
            & frame["mentions_gender"].isin([0, 1])
            & frame["mentions_location"].eq(0)
        ].copy()
        frame["text"] = frame["text"].astype(str).str.strip()
        frame = frame[frame["text"].ne("")].drop_duplicates("text")
        frame["language"] = language
        frame["language_bin"] = language_bin
        frame["toxicity"] = frame["toxic"].astype(int)
        frame["toxicity_bin"] = frame["toxic"].astype(int)
        frame["gender"] = frame["mentions_gender"].astype(int)
        frame["gender_bin"] = frame["mentions_gender"].astype(int)
        frame["candidate_id"] = frame["text"].map(
            lambda value: hashlib.sha256(f"{language}|{value}".encode()).hexdigest()[:16]
        )
        pieces.append(frame)
    data = pd.concat(pieces, ignore_index=True)
    split_sizes = {
        "geometry_400": geometry_per_cell,
        "vector_construction_256": vectors_per_cell,
        "evaluation_200": evaluation_per_cell,
    }
    outputs = {name: [] for name in split_sizes}
    for cell_index, (_, cell) in enumerate(
        data.groupby(["language", "toxicity_bin", "gender_bin"], sort=True)
    ):
        required = sum(split_sizes.values())
        if len(cell) < required:
            raise RuntimeError(f"Cell has {len(cell)} rows; need {required}")
        shuffled = cell.sample(frac=1, random_state=seed + cell_index).reset_index(drop=True)
        offset = 0
        for name, size in split_sizes.items():
            outputs[name].append(shuffled.iloc[offset : offset + size])
            offset += size
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    id_sets = {}
    for name, frames in outputs.items():
        result = pd.concat(frames, ignore_index=True)[COLUMNS]
        result.to_csv(out / f"{name}.csv", index=False)
        id_sets[name] = set(result["candidate_id"])
        print(
            name,
            len(result),
            result.groupby(["language", "toxicity_bin", "gender_bin"]).size().to_dict(),
        )
    names = list(id_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = id_sets[left] & id_sets[right]
            if overlap:
                raise RuntimeError(f"Leakage between {left} and {right}: {len(overlap)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing {en,ar}_labeled.csv from the gpt-4.1-mini annotation pipeline",
    )
    ap.add_argument(
        "--output-dir",
        default="toxicity_splits",
        help="Where to write geometry_400.csv, vector_construction_256.csv, evaluation_200.csv",
    )
    ap.add_argument("--geometry-per-cell", type=int, default=50)
    ap.add_argument("--vectors-per-cell", type=int, default=32)
    ap.add_argument("--evaluation-per-cell", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_splits(
        args.input_dir,
        args.output_dir,
        geometry_per_cell=args.geometry_per_cell,
        vectors_per_cell=args.vectors_per_cell,
        evaluation_per_cell=args.evaluation_per_cell,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
