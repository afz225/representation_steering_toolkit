#!/usr/bin/env python3
"""Re-run the gpt-4.1-mini toxicity re-annotation with the OpenAI Batch API.

You only need this to regenerate the annotations themselves. To reproduce the
paper's toxicity splits exactly, use datasets/toxicity/rebuild_from_manifest.py:
a fresh gpt-4.1-mini run can label some texts differently, which changes which
texts pass the filters and therefore the splits.

Pipeline (one language at a time; production used en and ar):
  1. export-texts      source texts in Hub row order -> texts_{lang}.csv (text, toxic)
  2. build-batch       texts CSV -> batch-input JSONL shards; request i has
                       custom_id f"{lang}_{i:06d}", the exact production format
  3. submit            upload shards and create batches (needs `openai`, OPENAI_API_KEY)
  4. collect           download finished batch outputs (re-run until all are done)
  5. assemble-labeled  batch outputs + texts CSV -> {lang}_labeled.csv, matched by
                       custom_id (batch output order is not guaranteed)
Then: python datasets/toxicity/prepare_toxicity_splits.py --input-dir <dir with {en,ar}_labeled.csv>

Example:
  python scripts/toxicity/batch_label_toxicity.py export-texts --language en --from-hub --output work/texts_en.csv
  python scripts/toxicity/batch_label_toxicity.py build-batch --input work/texts_en.csv --language en --output-dir work/batch_inputs/en
  python scripts/toxicity/batch_label_toxicity.py submit --inputs work/batch_inputs/en --jobs work/jobs_en.json
  python scripts/toxicity/batch_label_toxicity.py collect --jobs work/jobs_en.json --output-dir work/batch_outputs/en
  python scripts/toxicity/batch_label_toxicity.py assemble-labeled --input work/texts_en.csv --language en \
      --results work/batch_outputs/en --output work/labeled/en_labeled.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HUB_REPO = "textdetox/multilingual_toxicity_dataset"

DEVELOPER_PROMPT = """You are labeling short social media texts.

Task:
For the given text, output binary labels for whether the text explicitly mentions:
1. gender
2. ethnicity
3. religion
4. nationality
5. location

Rules:
- Use 1 only for explicit mention, not implication.
- Base the decision only on the text provided.
- Do not infer author identity or target identity unless explicitly stated.
- Nationality and ethnicity are separate.
- Location means a geographic place name or region explicitly mentioned, mentioning a people or community does not count as the location, it must be an explicit mention of a place.
- Justifications must be extremely brief and cite the trigger word or say no explicit mention.
- Return only schema-compliant JSON.
"""

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "mentions_gender": {"type": "integer", "enum": [0, 1]},
        "mentions_ethnicity": {"type": "integer", "enum": [0, 1]},
        "mentions_religion": {"type": "integer", "enum": [0, 1]},
        "mentions_nationality": {"type": "integer", "enum": [0, 1]},
        "mentions_location": {"type": "integer", "enum": [0, 1]},
        "justification_gender": {"type": "string"},
        "justification_ethnicity": {"type": "string"},
        "justification_religion": {"type": "string"},
        "justification_nationality": {"type": "string"},
        "justification_location": {"type": "string"},
    },
    "required": [
        "mentions_gender",
        "mentions_ethnicity",
        "mentions_religion",
        "mentions_nationality",
        "mentions_location",
        "justification_gender",
        "justification_ethnicity",
        "justification_religion",
        "justification_nationality",
        "justification_location",
    ],
    "additionalProperties": False,
}

LABEL_FIELDS = list(LABEL_SCHEMA["required"])
LABELED_COLUMNS = ["text", "toxic", *LABEL_FIELDS]  # production {language}_labeled.csv layout


def make_request(custom_id: str, text: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-4.1-mini",
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": DEVELOPER_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"Text: {text}"}],
                },
            ],
            "text": {"format": {"type": "json_schema", "name": "identity_location_labels", "schema": LABEL_SCHEMA, "strict": True}},
        },
    }


def custom_id(language: str, index: int) -> str:
    return f"{language}_{index:06d}"


def read_texts(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cmd_export_texts(args: argparse.Namespace) -> None:
    import pandas as pd

    if args.from_hub:
        from huggingface_hub import hf_hub_download

        source = hf_hub_download(HUB_REPO, f"data/{args.language}-00000-of-00001.parquet",
                                 repo_type="dataset", revision=args.revision)
    else:
        source = args.source
    frame = pd.read_parquet(source) if str(source).endswith(".parquet") else pd.read_csv(source)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame[["text", "toxic"]].to_csv(out, index=False)
    print(f"wrote {out} ({len(frame)} texts, Hub row order)")


def cmd_build_batch(args: argparse.Namespace) -> None:
    rows = read_texts(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards = [rows[i:i + args.max_rows] for i in range(0, len(rows), args.max_rows)]
    for shard_index, shard in enumerate(shards):
        path = out / f"{args.language}_shard_{shard_index:03d}.jsonl"
        with path.open("w", encoding="utf-8") as sink:
            for offset, row in enumerate(shard):
                index = shard_index * args.max_rows + offset
                request = make_request(custom_id(args.language, index), row["text"])  # unstripped, as in production
                sink.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"wrote {len(shards)} shard(s), {len(rows)} requests -> {out}")


def _client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("submit/collect need the OpenAI SDK: pip install openai (and set OPENAI_API_KEY)")
    return OpenAI()


def cmd_submit(args: argparse.Namespace) -> None:
    client = _client()
    inputs = Path(args.inputs)
    files = sorted(inputs.glob("*.jsonl")) if inputs.is_dir() else [inputs]
    jobs_path = Path(args.jobs)
    jobs = json.loads(jobs_path.read_text(encoding="utf-8")) if jobs_path.exists() else []
    submitted = {job["input_path"] for job in jobs}
    for path in files:
        if str(path) in submitted:
            print(f"skip {path} (already submitted)")
            continue
        upload = client.files.create(file=path.open("rb"), purpose="batch")
        batch = client.batches.create(input_file_id=upload.id, endpoint="/v1/responses",
                                      completion_window="24h")
        jobs.append({"input_path": str(path), "input_file_id": upload.id, "batch_id": batch.id,
                     "status": batch.status, "output_path": None})
        jobs_path.parent.mkdir(parents=True, exist_ok=True)
        jobs_path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        print(f"submitted {path} -> {batch.id}")


def cmd_collect(args: argparse.Namespace) -> None:
    client = _client()
    jobs_path = Path(args.jobs)
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pending = 0
    for job in jobs:
        if job.get("output_path"):
            continue
        batch = client.batches.retrieve(job["batch_id"])
        job["status"] = batch.status
        if batch.status == "completed" and batch.output_file_id:
            path = out / (Path(job["input_path"]).stem + "_output.jsonl")
            path.write_bytes(client.files.content(batch.output_file_id).read())
            job["output_path"] = str(path)
            if batch.error_file_id:
                errors = out / (Path(job["input_path"]).stem + "_errors.jsonl")
                errors.write_bytes(client.files.content(batch.error_file_id).read())
                job["error_path"] = str(errors)
        elif batch.status in ("failed", "expired", "cancelled"):
            print(f"{job['batch_id']}: {batch.status}; resubmit {job['input_path']}")
        else:
            pending += 1
        print(f"{job['batch_id']}: {job['status']}")
    jobs_path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    print(f"{pending} batch(es) still running" if pending else "all batches finished")


def parse_output_line(record: dict) -> dict | None:
    """The label object from one Batch API output line (None on an error line)."""
    response = record.get("response") or {}
    if record.get("error") or response.get("status_code") != 200:
        return None
    # A message can hold several output_text parts (production has some with an empty
    # first part), so join them before parsing.
    text = "".join(content.get("text", "") for item in response.get("body", {}).get("output", [])
                   if item.get("type") == "message"
                   for content in item.get("content", []) if content.get("type") == "output_text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def cmd_assemble(args: argparse.Namespace) -> None:
    texts = read_texts(args.input)
    results = Path(args.results)
    files = sorted(results.glob("*_output.jsonl")) if results.is_dir() else [results]
    labels: dict[str, dict] = {}
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    parsed = parse_output_line(record)
                    if parsed is not None:
                        labels[record["custom_id"]] = parsed
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    with out.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=LABELED_COLUMNS)
        writer.writeheader()
        for index, row in enumerate(texts):
            label = labels.get(custom_id(args.language, index))
            if label is None:
                missing += 1
                label = {}
            writer.writerow({"text": row["text"], "toxic": row["toxic"],
                             **{field: label.get(field, "") for field in LABEL_FIELDS}})
    print(f"wrote {out}: {len(texts) - missing} labeled, {missing} without a label (left blank)")
    if missing and not args.allow_missing:
        sys.exit(f"{missing} texts have no label; resubmit them or pass --allow-missing")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export-texts", help="write the source texts (text, toxic) in Hub row order")
    export.add_argument("--language", required=True, help="language split, e.g. en or ar")
    source = export.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-hub", action="store_true", help=f"download {HUB_REPO} from the Hub")
    source.add_argument("--source", help="local parquet/CSV of that split (columns text, toxic)")
    export.add_argument("--revision", default=None, help="Hub revision to download")
    export.add_argument("--output", required=True)
    export.set_defaults(func=cmd_export_texts)

    build = sub.add_parser("build-batch", help="write batch-input JSONL shards from a texts CSV")
    build.add_argument("--input", required=True, help="texts CSV from export-texts")
    build.add_argument("--language", required=True, help="language code used in custom_id")
    build.add_argument("--output-dir", required=True)
    build.add_argument("--max-rows", type=int, default=500, help="requests per shard (production: 500)")
    build.set_defaults(func=cmd_build_batch)

    submit = sub.add_parser("submit", help="upload shards and create Batch API jobs")
    submit.add_argument("--inputs", required=True, help="a shard JSONL or a directory of them")
    submit.add_argument("--jobs", required=True, help="jobs JSON to create or extend")
    submit.set_defaults(func=cmd_submit)

    collect = sub.add_parser("collect", help="download outputs of finished batches")
    collect.add_argument("--jobs", required=True)
    collect.add_argument("--output-dir", required=True)
    collect.set_defaults(func=cmd_collect)

    assemble = sub.add_parser("assemble-labeled", help="batch outputs + texts -> {language}_labeled.csv")
    assemble.add_argument("--input", required=True, help="the texts CSV used for build-batch")
    assemble.add_argument("--language", required=True)
    assemble.add_argument("--results", required=True, help="an output JSONL or a directory of *_output.jsonl")
    assemble.add_argument("--output", required=True)
    assemble.add_argument("--allow-missing", action="store_true", help="write rows without a label instead of failing")
    assemble.set_defaults(func=cmd_assemble)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
