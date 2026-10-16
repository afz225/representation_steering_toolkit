#!/usr/bin/env python3
"""Verify a locally rebuilt toxicity split against the shipped manifest.

The toxicity texts themselves are real social-media posts (textdetox;
openrail++ source data) and are NOT shipped with this repository. Users who
regenerate the labeled CSVs (via the documented textdetox download +
gpt-4.1-mini annotation pipeline, see DATASET.md) can prove their rebuild
matches the production split used in the paper with this script.

Checks, per candidate_id in the manifest:
  - the id appears in exactly one of the three split CSVs
  - the sha256 of the (whitespace-stripped) text matches manifest text_sha256
  - the (language, toxicity_bin, gender_bin) cell matches the manifest cell
  - split membership matches the manifest split
  - per-split and per-cell sizes match, and there are no extra ids
No candidate texts are read from the manifest; only fingerprints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

SPLIT_FILES = ["geometry_400.csv", "vector_construction_256.csv", "evaluation_200.csv"]
SPLIT_NAMES = [Path(name).stem for name in SPLIT_FILES]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def verify_splits(splits_dir, manifest_path=None) -> int:
    """Programmatic entrypoint; returns 0 when the split matches, else 1."""
    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parent / "manifest.json"
    manifest_path_obj = Path(manifest_path)
    manifest = json.loads(manifest_path_obj.read_text(encoding="utf-8"))
    expected = manifest["candidates"]

    splits_dir = Path(splits_dir)
    observed = {}
    row_count = 0
    for filename, split_name in zip(SPLIT_FILES, SPLIT_NAMES):
        path = splits_dir / filename
        if not path.exists():
            print(f"FAIL: missing {path}")
            return 1
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row_count += 1
                cid = row["candidate_id"]
                entry = {
                    "split": split_name,
                    "text_sha256": sha256_text(row["text"]),
                    "cell": [row["language"], row["toxicity_bin"], row["gender_bin"]],
                }
                if cid in observed:
                    print(f"FAIL: duplicate candidate_id {cid} across split CSVs")
                    return 1
                observed[cid] = entry

    failures = []
    for cid, want in sorted(expected.items()):
        got = observed.pop(cid, None)
        if got is None:
            failures.append(f"{cid}: missing from split CSVs")
            continue
        if got["text_sha256"] != want["text_sha256"]:
            failures.append(f"{cid}: text fingerprint mismatch")
        elif got["cell"] != want["cell"]:
            failures.append(f"{cid}: cell {got['cell']} != manifest {want['cell']}")
        elif got["split"] != want["split"]:
            failures.append(f"{cid}: in {got['split']}, manifest says {want['split']}")
    for cid in sorted(observed):
        failures.append(f"{cid}: extra candidate not in manifest (split {observed[cid]['split']})")

    if failures:
        print(f"FAIL: {len(failures)} mismatches against {manifest_path_obj}")
        for line in failures[:50]:
            print(" ", line)
        if len(failures) > 50:
            print(f"  ... and {len(failures) - 50} more")
        return 1
    print(
        f"OK: {row_count} rows across {len(SPLIT_FILES)} splits match manifest "
        f"({len(expected)} candidates, seed {manifest['split_seed']})."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--splits-dir",
        required=True,
        help="Directory with geometry_400.csv, vector_construction_256.csv, evaluation_200.csv",
    )
    ap.add_argument("--manifest", default=None, help="manifest.json path (default: next to this script)")
    args = ap.parse_args()
    return verify_splits(args.splits_dir, args.manifest)


if __name__ == "__main__":
    sys.exit(main())
