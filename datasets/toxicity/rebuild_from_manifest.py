#!/usr/bin/env python3
"""Rebuild the exact production toxicity splits from the original texts + manifest.json.

This is the exact-replication path: it needs only the public
`textdetox/multilingual_toxicity_dataset` (en, ar) texts, not a re-run of the
gpt-4.1-mini annotation. Our annotations for the 856 candidates (toxicity and
gender bins, split membership and row order) are in the shipped
`manifest.json`, keyed by text fingerprints, so no third-party text is
redistributed here.

For every manifest candidate the script finds the source text whose
candidate_id (first 16 hex chars of sha256 of f"{language}|{text}" over the
whitespace-stripped text) matches, checks its text_sha256, and writes
geometry_400.csv, vector_construction_256.csv and evaluation_200.csv in the
production column layout and row order. On the production data the output is
byte-identical to the CSVs used in the paper.

Sources (pick one):
  --from-hub                 download data/{en,ar}-00000-of-00001.parquet from the
                             Hugging Face Hub (needs huggingface_hub)
  --source-dir DIR           local files DIR/{en,ar}.{parquet,csv,jsonl} with a `text` column

Example:
  python datasets/toxicity/rebuild_from_manifest.py --from-hub --output-dir toxicity_splits
  python datasets/toxicity/verify_split.py --splits-dir toxicity_splits
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HUB_REPO = "textdetox/multilingual_toxicity_dataset"
LANGUAGE_BIN = {"en": 0, "ar": 1}
SPLITS = ["geometry_400", "vector_construction_256", "evaluation_200"]
COLUMNS = ["text", "toxicity", "toxicity_bin", "gender", "gender_bin", "language", "language_bin", "candidate_id"]


def candidate_id(language: str, text: str) -> str:
    return hashlib.sha256(f"{language}|{text}".encode("utf-8")).hexdigest()[:16]


def read_source(language: str, source_dir: Path | None, revision: str | None) -> pd.Series:
    """The language's source texts, as given (whitespace is stripped by the caller)."""
    if source_dir is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(HUB_REPO, f"data/{language}-00000-of-00001.parquet",
                               repo_type="dataset", revision=revision)
        return pd.read_parquet(path)["text"].astype(str)
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv),
                           (".jsonl", lambda p: pd.read_json(p, lines=True))):
        path = source_dir / f"{language}{suffix}"
        if path.exists():
            return reader(path)["text"].astype(str)
    raise FileNotFoundError(f"no {language}.parquet/.csv/.jsonl in {source_dir}")


def rebuild(output_dir, manifest_path=None, source_dir=None, revision=None) -> int:
    """Write the three split CSVs; return 0 when every manifest candidate was found."""
    manifest_path = Path(manifest_path) if manifest_path else Path(__file__).resolve().parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest["candidates"]
    if any("row" not in entry for entry in candidates.values()):
        print(f"FAIL: {manifest_path} has no row order (manifest_version >= 2 needed)")
        return 1

    texts: dict[str, str] = {}
    for language in sorted({entry["cell"][0] for entry in candidates.values()}):
        for raw in read_source(language, Path(source_dir) if source_dir else None, revision):
            text = raw.strip()
            cid = candidate_id(language, text)
            if cid in candidates and cid not in texts:
                texts[cid] = text

    missing, mismatched = [], []
    rows = {split: [] for split in SPLITS}
    for cid, entry in candidates.items():
        text = texts.get(cid)
        if text is None:
            missing.append(cid)
            continue
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != entry["text_sha256"]:
            mismatched.append(cid)
            continue
        language, toxicity_bin, gender_bin = entry["cell"]
        rows[entry["split"]].append((entry["row"], {
            "text": text, "toxicity": int(toxicity_bin), "toxicity_bin": int(toxicity_bin),
            "gender": int(gender_bin), "gender_bin": int(gender_bin), "language": language,
            "language_bin": LANGUAGE_BIN[language], "candidate_id": cid,
        }))
    if missing or mismatched:
        print(f"FAIL: {len(missing)} manifest candidates not found in the source texts, "
              f"{len(mismatched)} with a different text fingerprint")
        for cid in (missing + mismatched)[:20]:
            print(" ", cid)
        return 1

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        ordered = [record for _, record in sorted(rows[split], key=lambda item: item[0])]
        if [position for position, _ in sorted(rows[split], key=lambda item: item[0])] != list(range(len(ordered))):
            print(f"FAIL: {split} row positions in the manifest are not 0..{len(ordered) - 1}")
            return 1
        pd.DataFrame(ordered, columns=COLUMNS).to_csv(out / f"{split}.csv", index=False)
        print(f"{split}: {len(ordered)} rows -> {out / f'{split}.csv'}")
    print(f"OK: rebuilt {len(candidates)} candidates from source texts and {manifest_path.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-hub", action="store_true", help=f"download {HUB_REPO} (en, ar) from the Hub")
    source.add_argument("--source-dir", help="directory with {en,ar}.{parquet,csv,jsonl} holding a `text` column")
    ap.add_argument("--revision", default=None, help="Hub revision (branch, tag or commit) to download")
    ap.add_argument("--output-dir", default="toxicity_splits", help="where to write the three split CSVs")
    ap.add_argument("--manifest", default=None, help="manifest.json path (default: next to this script)")
    args = ap.parse_args()
    return rebuild(args.output_dir, args.manifest, None if args.from_hub else args.source_dir, args.revision)


if __name__ == "__main__":
    sys.exit(main())
